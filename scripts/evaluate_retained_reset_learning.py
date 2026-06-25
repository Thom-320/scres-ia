#!/usr/bin/env python3
"""Evaluate the Track A retained-learning win condition.

This is a smoke-scale, auditable evaluator for the primary paper contrast:
retained online learner versus an otherwise identical reset-learning learner. It also
selects the robust static Garrido-aligned policy from the 18-action thesis grid
using training seeds only, then evaluates every condition on held-out common
random streams.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
from typing import Any, Callable

import gymnasium as gym
import numpy as np

from stable_baselines3 import DQN, PPO

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.config import (  # noqa: E402
    THESIS_DOWNSTREAM_Q_RANGES,
    THESIS_ROBUSTNESS_DOWNSTREAM_Q_SOURCE,
    TRACK_A_TRAINING_DOWNSTREAM_Q_SOURCE,
)
from supply_chain.external_env_interface import (  # noqa: E402
    get_episode_terminal_metrics,
    make_discrete18_track_a_env,
    make_track_b_env,
)
from supply_chain.scenario_tape import (  # noqa: E402
    RegimePhase,
    ScenarioTape,
    generate_scenario_tape,
)


DEFAULT_OUTPUT_ROOT = Path("outputs/benchmarks/retained_reset_learning")
DEFAULT_TRAIN_SEEDS = (1101, 1102, 1103)
DEFAULT_EVAL_SEEDS = (2201, 2202, 2203)
# Within-block adaptation uses a seed offset from the eval seed so an arm never
# adapts on the exact realization it is evaluated on (same regime, different draw).
ADAPT_SEED_OFFSET = 100_000


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--reward-mode", default="control_v1")
    parser.add_argument(
        "--algo",
        choices=("dqn", "ppo"),
        default="dqn",
        help="Neural learner used for frozen, retained, and reset conditions.",
    )
    parser.add_argument("--observation-version", default="v5")
    parser.add_argument(
        "--track",
        choices=("a", "b"),
        default="a",
        help="a = thesis [6,3] inventory/shift; b = continuous downstream control.",
    )
    parser.add_argument(
        "--mask-obs-indices",
        default=None,
        help=(
            "Comma-separated obs indices to zero (regime-observability ablation). "
            "Track A v5 direct disruption signals are 17,19,23."
        ),
    )
    parser.add_argument("--risk-level", default="current")
    parser.add_argument(
        "--downstream-q-source",
        choices=tuple(sorted(THESIS_DOWNSTREAM_Q_RANGES)),
        default=TRACK_A_TRAINING_DOWNSTREAM_Q_SOURCE,
        help=(
            "Paper-facing Track A default is "
            f"{TRACK_A_TRAINING_DOWNSTREAM_Q_SOURCE}; use "
            f"{THESIS_ROBUSTNESS_DOWNSTREAM_Q_SOURCE} only as robustness."
        ),
    )
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--step-size-hours", type=float, default=168.0)
    parser.add_argument(
        "--decision-cadence",
        choices=("block", "weekly"),
        default="block",
        help=(
            "block chooses one thesis configuration at reset and holds it for "
            "the whole disruption block; weekly reselects every env step and "
            "is a sensitivity lane."
        ),
    )
    parser.add_argument(
        "--stochastic-pt",
        action="store_true",
        help=(
            "Enable mean-preserving stochastic processing times as a "
            "thesis-anchored learning-extension candidate."
        ),
    )
    parser.add_argument("--pretrain-timesteps", type=int, default=0)
    parser.add_argument("--online-timesteps-per-cycle", type=int, default=0)
    # --- learning_extension_v1 frozen regime (two independent persistence chains) ---
    parser.add_argument(
        "--rho-disruption",
        type=float,
        default=None,
        help=(
            "Persistence of the campaign-phase disruption-intensity chain. Setting "
            "either rho enables the persistent regime; an unset rho defaults to "
            "memoryless (1/3). Omit both for the stationary reference lane."
        ),
    )
    parser.add_argument("--rho-demand", type=float, default=None)
    parser.add_argument(
        "--regime-seed",
        type=int,
        default=909,
        help="Exogenous tape seed; kept separate from decision/eval seeds (CRN).",
    )
    parser.add_argument("--train-seeds", default=",".join(map(str, DEFAULT_TRAIN_SEEDS)))
    parser.add_argument("--eval-seeds", default=",".join(map(str, DEFAULT_EVAL_SEEDS)))
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--buffer-size", type=int, default=10_000)
    parser.add_argument("--learning-starts", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--n-steps",
        type=int,
        default=8,
        help="PPO rollout length; ignored by DQN.",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=2,
        help="PPO optimization epochs per rollout; ignored by DQN.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


def regime_enabled(args: argparse.Namespace) -> bool:
    return args.rho_disruption is not None or args.rho_demand is not None


def build_tape(args: argparse.Namespace, n_blocks: int, *, seed: int) -> ScenarioTape | None:
    """Build an exogenous persistence tape, or None for the stationary lane.

    An unset rho defaults to memoryless (1/3) so a single chain can be studied in
    isolation (the two-independent-process ablation).
    """
    if not regime_enabled(args):
        return None
    return generate_scenario_tape(
        n_blocks,
        rho_disruption=args.rho_disruption if args.rho_disruption is not None else 1 / 3,
        rho_demand=args.rho_demand if args.rho_demand is not None else 1 / 3,
        seed=seed,
    )


def env_kwargs(
    args: argparse.Namespace, *, regime: RegimePhase | None = None
) -> dict[str, Any]:
    return {
        "reward_mode": args.reward_mode,
        "observation_version": args.observation_version,
        "risk_level": regime.disruption_level if regime is not None else args.risk_level,
        "demand_mean_multiplier": regime.demand_multiplier if regime is not None else 1.0,
        "downstream_q_source": args.downstream_q_source,
        "step_size_hours": args.step_size_hours,
        "max_steps": args.max_steps,
        "stochastic_pt": args.stochastic_pt,
    }


class BlockDecisionEnv(gym.Wrapper):
    """Turn weekly Track A control into one held configuration per block."""

    def step(self, action: int):
        held_action = int(action)
        terminated = truncated = False
        reward_total = 0.0
        service_loss_area = 0.0
        step_cost_total = 0.0
        surge_hours = 0.0
        block_steps = 0
        obs = None
        info: dict[str, Any] = {}
        while not (terminated or truncated):
            obs, reward, terminated, truncated, info = self.env.step(held_action)
            reward_total += float(reward)
            demanded = float(info.get("new_demanded", 0.0))
            backorder_qty = float(info.get("new_backorder_qty", 0.0))
            service_loss_area += max(0.0, backorder_qty / max(demanded, 1.0))
            step_cost_total += float(info.get("shift_cost_step", 0.0))
            shifts_active = int(info.get("shifts_active", 1))
            surge_hours += max(0, shifts_active - 1) * float(
                getattr(self.unwrapped, "step_size", 168.0)
            )
            block_steps += 1
        enriched = dict(info)
        enriched["decision_cadence"] = "block"
        enriched["held_action"] = held_action
        enriched["block_steps"] = block_steps
        enriched["block_service_loss_area"] = service_loss_area
        enriched["block_step_cost_total"] = step_cost_total
        enriched["block_surge_hours"] = surge_hours
        if obs is None:
            obs = self.observation_space.sample()
        return obs, reward_total, terminated, truncated, enriched


class ObservationMaskWrapper(gym.ObservationWrapper):
    """Zero specific observation indices (regime-observability ablation).

    Used to HIDE the direct disruption/severity signals (e.g. idx 17/19/23 in the
    Track A v5 obs) so the regime must be INFERRED from consequence dynamics. This
    isolates when retained history L_{k-1} adds value: if masking the regime makes
    retained beat reset, retention matters under partial observability.
    """

    def __init__(self, env, indices: list[int]):
        super().__init__(env)
        self._mask_idx = np.array(sorted(set(int(i) for i in indices)), dtype=int)

    def observation(self, obs):
        obs = np.array(obs, dtype=np.float32)
        if self._mask_idx.size:
            obs[self._mask_idx] = 0.0
        return obs


def parse_mask_indices(args: argparse.Namespace) -> list[int]:
    raw = getattr(args, "mask_obs_indices", None)
    if not raw:
        return []
    return [int(x) for x in str(raw).split(",") if x.strip()]


def build_env(args: argparse.Namespace, *, regime: RegimePhase | None = None):
    if getattr(args, "track", "a") == "b":
        # Track B: continuous downstream control, native adaptive_benchmark_v2 regime,
        # v7 obs. The Track A scenario tape (current/increased/severe) does not apply;
        # regime is ignored here. Only reward + horizon are overridden.
        env = make_track_b_env(reward_mode=args.reward_mode, max_steps=args.max_steps)
    else:
        env = make_discrete18_track_a_env(**env_kwargs(args, regime=regime))
    mask = parse_mask_indices(args)
    if mask:
        env = ObservationMaskWrapper(env, mask)
    if args.decision_cadence == "block":
        return BlockDecisionEnv(env)
    return env


def run_episode(
    *,
    args: argparse.Namespace,
    condition: str,
    seed: int,
    cycle_index: int,
    policy_fn: Callable[[np.ndarray, dict[str, Any]], int],
    regime: RegimePhase | None = None,
) -> dict[str, Any]:
    env = build_env(args, regime=regime)
    obs, info = env.reset(seed=seed)
    terminated = truncated = False
    reward_total = 0.0
    service_loss_area = 0.0
    step_cost_total = 0.0
    surge_hours = 0.0
    steps = 0
    first_action = -1
    last_action = -1
    held_action: int | None = None
    if args.decision_cadence == "block":
        held_action = int(policy_fn(obs, info))
        first_action = held_action
    while not (terminated or truncated):
        if held_action is None:
            action = int(policy_fn(obs, info))
            if first_action < 0:
                first_action = action
        else:
            action = held_action
        last_action = action
        obs, reward, terminated, truncated, info = env.step(action)
        reward_total += float(reward)
        demanded = float(info.get("new_demanded", 0.0))
        backorder_qty = float(info.get("new_backorder_qty", 0.0))
        service_loss_area += float(
            info.get(
                "block_service_loss_area",
                max(0.0, backorder_qty / max(demanded, 1.0)),
            )
        )
        step_cost_total += float(
            info.get("block_step_cost_total", info.get("shift_cost_step", 0.0))
        )
        shifts_active = int(info.get("shifts_active", 1))
        surge_hours += float(
            info.get(
                "block_surge_hours",
                max(0, shifts_active - 1) * float(args.step_size_hours),
            )
        )
        steps += int(info.get("block_steps", 1))

    terminal = get_episode_terminal_metrics(env)
    sim = getattr(env.unwrapped, "sim", None)
    pending_qty = float(getattr(sim, "pending_backorder_qty", 0.0))
    lost_orders = float(getattr(sim, "total_unattended_orders", 0.0))
    return {
        "condition": condition,
        "seed": seed,
        "cycle_index": cycle_index,
        "regime_disruption_level": regime.disruption_level if regime else args.risk_level,
        "regime_demand_multiplier": regime.demand_multiplier if regime else 1.0,
        "steps": steps,
        "decision_cadence": args.decision_cadence,
        "first_action": first_action,
        "last_action": last_action,
        "reward_total": reward_total,
        "order_level_ret_mean": terminal["order_level_ret_mean"],
        "fill_rate_order_level": terminal["fill_rate_order_level"],
        "service_loss_area": service_loss_area,
        "pending_backorder_qty": pending_qty,
        "lost_orders": lost_orders,
        "step_cost_total": step_cost_total,
        "surge_hours": surge_hours,
    }


def static_action_name(action: int) -> str:
    level = int(action) // 3
    shift = int(action) % 3 + 1
    return f"I_level_{level}_S{shift}"


def select_robust_static_policy(
    args: argparse.Namespace,
    *,
    train_seeds: list[int],
    train_tape: ScenarioTape | None = None,
) -> tuple[int, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for action in range(18):
        for cycle_index, seed in enumerate(train_seeds):
            rows.append(
                run_episode(
                    args=args,
                    condition=f"static_train_{static_action_name(action)}",
                    seed=seed,
                    cycle_index=cycle_index,
                    policy_fn=lambda _obs, _info, a=action: a,
                    regime=train_tape[cycle_index] if train_tape else None,
                )
            )
            rows[-1]["candidate_action"] = action
            rows[-1]["candidate_policy"] = static_action_name(action)

    def score(action: int) -> tuple[float, float, float]:
        bucket = [row for row in rows if int(row["candidate_action"]) == action]
        mean_ret = float(np.nanmean([row["order_level_ret_mean"] for row in bucket]))
        mean_loss = float(np.nanmean([row["service_loss_area"] for row in bucket]))
        mean_cost = float(np.nanmean([row["step_cost_total"] for row in bucket]))
        return mean_ret, -mean_loss, -mean_cost

    best_action = max(range(18), key=score)
    return best_action, rows


def heuristic_policy(_obs: np.ndarray, info: dict[str, Any]) -> int:
    pending_qty = float(info.get("pending_backorder_qty", 0.0))
    service = float(info.get("service_continuity_step", 1.0))
    if pending_qty > 0.0 or service < 0.90:
        return 17  # max thesis buffer, S3
    if service < 0.97:
        return 10  # I504, S2
    return 0  # no strategic buffer, S1


def condition_name(args: argparse.Namespace, stem: str) -> str:
    return f"{stem}_{args.algo}"


def build_initial_model(args: argparse.Namespace, model_path: Path) -> None:
    env = build_env(args)
    if args.algo == "dqn":
        model = DQN(
            "MlpPolicy",
            env,
            seed=args.seed,
            learning_rate=args.learning_rate,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
            batch_size=args.batch_size,
            verbose=0,
        )
    else:
        n_steps = max(2, int(args.n_steps))
        batch_size = min(max(2, int(args.batch_size)), n_steps)
        model = PPO(
            "MlpPolicy",
            env,
            seed=args.seed,
            learning_rate=args.learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=max(1, int(args.n_epochs)),
            verbose=0,
        )
    if args.pretrain_timesteps > 0:
        model.learn(total_timesteps=args.pretrain_timesteps, progress_bar=False)
    model.save(model_path)
    env.close()


def load_model(args: argparse.Namespace, model_path: Path):
    model_cls = DQN if args.algo == "dqn" else PPO
    return model_cls.load(model_path, env=build_env(args))


def model_policy(model: Any) -> Callable[[np.ndarray, dict[str, Any]], int]:
    def _policy(obs: np.ndarray, _info: dict[str, Any]) -> int:
        action, _state = model.predict(obs, deterministic=True)
        return int(action)

    return _policy


def online_update(
    args: argparse.Namespace,
    model: Any,
    *,
    seed: int,
    regime: RegimePhase | None = None,
) -> None:
    if args.online_timesteps_per_cycle <= 0:
        return
    env = build_env(args, regime=regime)
    env.reset(seed=seed)
    model.set_env(env)
    model.learn(
        total_timesteps=args.online_timesteps_per_cycle,
        reset_num_timesteps=False,
        progress_bar=False,
    )
    env.close()


def paired_delta(
    rows: list[dict[str, Any]],
    metric: str,
    *,
    retained_condition: str,
    reset_condition: str,
) -> dict[str, float]:
    retained = {
        (int(row["seed"]), int(row["cycle_index"])): float(row[metric])
        for row in rows
        if row["condition"] == retained_condition
    }
    reset = {
        (int(row["seed"]), int(row["cycle_index"])): float(row[metric])
        for row in rows
        if row["condition"] == reset_condition
    }
    deltas = [
        retained[key] - reset[key]
        for key in sorted(retained)
        if key in reset and np.isfinite(retained[key]) and np.isfinite(reset[key])
    ]
    if not deltas:
        return {"n": 0, "mean_delta": float("nan")}
    return {"n": len(deltas), "mean_delta": float(np.mean(deltas))}


def main() -> int:
    args = build_parser().parse_args()
    run_label = args.label or f"retained_reset_learning_{utc_stamp()}"
    run_dir = args.output_root / run_label
    run_dir.mkdir(parents=True, exist_ok=False)
    train_seeds = parse_ints(args.train_seeds)
    eval_seeds = parse_ints(args.eval_seeds)

    # Two exogenous tapes (CRN within each): the held-out eval tape carries the
    # persistence the retained learner must exploit; the train tape (distinct seed)
    # is used only to select the robust static policy. None => stationary lane.
    eval_tape = build_tape(args, len(eval_seeds), seed=args.regime_seed)
    train_tape = build_tape(args, len(train_seeds), seed=args.regime_seed + 1)

    static_action, static_train_rows = select_robust_static_policy(
        args,
        train_seeds=train_seeds,
        train_tape=train_tape,
    )
    frozen_condition = condition_name(args, "frozen")
    retained_condition = condition_name(args, "retained_online")
    reset_condition = condition_name(args, "reset_learning")

    rows: list[dict[str, Any]] = []
    for cycle_index, seed in enumerate(eval_seeds):
        regime = eval_tape[cycle_index] if eval_tape else None
        rows.append(
            run_episode(
                args=args,
                condition="robust_static",
                seed=seed,
                cycle_index=cycle_index,
                policy_fn=lambda _obs, _info, a=static_action: a,
                regime=regime,
            )
        )
        rows.append(
            run_episode(
                args=args,
                condition="threshold_heuristic",
                seed=seed,
                cycle_index=cycle_index,
                policy_fn=heuristic_policy,
                regime=regime,
            )
        )

    with tempfile.TemporaryDirectory() as tmp:
        initial_model_path = Path(tmp) / f"initial_{args.algo}.zip"
        build_initial_model(args, initial_model_path)
        frozen_model = load_model(args, initial_model_path)
        retained_model = load_model(args, initial_model_path)

        for cycle_index, seed in enumerate(eval_seeds):
            regime = eval_tape[cycle_index] if eval_tape else None
            adapt_seed = seed + ADAPT_SEED_OFFSET
            # Frozen: zero-adaptation reference (theta_0 every block).
            rows.append(
                run_episode(
                    args=args,
                    condition=frozen_condition,
                    seed=seed,
                    cycle_index=cycle_index,
                    policy_fn=model_policy(frozen_model),
                    regime=regime,
                )
            )
            # Retained: adapt on this block (carrying theta across blocks), then eval.
            online_update(args, retained_model, seed=adapt_seed, regime=regime)
            rows.append(
                run_episode(
                    args=args,
                    condition=retained_condition,
                    seed=seed,
                    cycle_index=cycle_index,
                    policy_fn=model_policy(retained_model),
                    regime=regime,
                )
            )
            # Reset: restore theta_0, adapt on this block only, then eval.
            reset_model = load_model(args, initial_model_path)
            online_update(args, reset_model, seed=adapt_seed, regime=regime)
            rows.append(
                run_episode(
                    args=args,
                    condition=reset_condition,
                    seed=seed,
                    cycle_index=cycle_index,
                    policy_fn=model_policy(reset_model),
                    regime=regime,
                )
            )

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "label": run_label,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "train_seeds": train_seeds,
        "eval_seeds": eval_seeds,
        "regime": {
            "enabled": regime_enabled(args),
            "rho_disruption": eval_tape.rho_disruption if eval_tape else None,
            "rho_demand": eval_tape.rho_demand if eval_tape else None,
            "regime_seed": args.regime_seed,
            "eval_tape_disruption_levels": (
                [p.disruption_level for p in eval_tape.blocks] if eval_tape else None
            ),
            "eval_tape_demand_multipliers": (
                [p.demand_multiplier for p in eval_tape.blocks] if eval_tape else None
            ),
        },
        "robust_static_action": static_action,
        "robust_static_policy": static_action_name(static_action),
        "primary_estimand": (
            f"order_level_ret_mean {retained_condition} - {reset_condition}"
        ),
        "retained_minus_reset_ret": paired_delta(
            rows,
            "order_level_ret_mean",
            retained_condition=retained_condition,
            reset_condition=reset_condition,
        ),
        "retained_minus_reset_service_loss_area": paired_delta(
            rows,
            "service_loss_area",
            retained_condition=retained_condition,
            reset_condition=reset_condition,
        ),
    }
    write_csv(run_dir / "static_training_grid.csv", static_train_rows)
    write_csv(run_dir / "evaluation_metrics.csv", rows)
    write_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Saved to: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
