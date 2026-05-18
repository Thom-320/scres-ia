#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Callable

import gymnasium as gym
import numpy as np

try:
    from sb3_contrib import RecurrentPPO
except ImportError:  # pragma: no cover - runtime guard for optional dependency.
    RecurrentPPO = None
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.external_env_interface import (  # noqa: E402
    THESIS_INVENTORY_PERIODS,
    get_episode_terminal_metrics,
    make_dkana_thesis_faithful_env,
)
from supply_chain.thesis_design import (  # noqa: E402
    ThesisDesignSpec,
    design_spec_for_cfi,
    parse_cf_range,
)

DEFAULT_OUTPUT_ROOT = Path("outputs/benchmarks/thesis_decision_ppo_smoke")
STATIC_POLICIES = ("static_s1", "static_s2", "static_s3")
INVENTORY_POLICIES = tuple(
    f"inventory_I{period}_S1" for period in THESIS_INVENTORY_PERIODS
)
ACTION_DIM = 18
FACTORED_ACTION_DIM = 4


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train PPO+MLP with Garrido thesis decision variables as the "
            "18D action space and a configurable observation surface."
        )
    )
    parser.add_argument("--label", default=None)
    parser.add_argument("--train-timesteps", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--reward-mode", default="control_v1")
    parser.add_argument("--risk-level", default="increased")
    parser.add_argument("--observation-version", default="v5")
    parser.add_argument(
        "--observation-mode",
        choices=[
            "decision_reward",
            "env_reward",
            "env_state_reward",
            "env_sdm_history_reward",
        ],
        default="env_sdm_history_reward",
    )
    parser.add_argument(
        "--action-space-mode",
        choices=["onehot_18d", "factorized"],
        default="factorized",
    )
    parser.add_argument(
        "--algo",
        choices=["ppo_mlp", "recurrent_ppo"],
        default="ppo_mlp",
    )
    parser.add_argument(
        "--ablation-suite",
        action="store_true",
        help="Run a compact observation/reward/architecture ablation suite.",
    )
    parser.add_argument("--step-size-hours", type=float, default=168.0)
    parser.add_argument("--max-steps", type=int, default=260)
    parser.add_argument("--stochastic-pt", action="store_true")
    parser.add_argument(
        "--garrido-cfis",
        default="31-90",
        help="Comma/range list of Garrido thesis static Cf rows to evaluate.",
    )
    parser.add_argument(
        "--learn-initial-decision",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use the first policy action as the thesis-comparable initial buffer/"
            "shift decision before warmup."
        ),
    )
    parser.add_argument(
        "--include-static-grid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate static I_t x S combinations as a stronger non-Garrido baseline.",
    )
    parser.add_argument(
        "--eval-ai-on-garrido-cfis",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate the trained AI policy under each selected Garrido Cf risk row.",
    )
    parser.add_argument(
        "--train-cfis",
        default=None,
        help=(
            "Optional comma/range list of Garrido Cf rows to sample at each "
            "training reset. Defaults to the fixed --risk-level env."
        ),
    )
    parser.add_argument(
        "--policy-net-arch",
        choices=["small", "medium", "large"],
        default="medium",
        help="MLP policy/value size for PPO.",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument(
        "--vec-normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize observations during training and deterministic evaluation.",
    )
    return parser


def env_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "reward_mode": args.reward_mode,
        "risk_level": args.risk_level,
        "observation_version": args.observation_version,
        "observation_mode": args.observation_mode,
        "action_space_mode": args.action_space_mode,
        "step_size_hours": args.step_size_hours,
        "max_steps": args.max_steps,
        "stochastic_pt": args.stochastic_pt,
        "learn_initial_decision": args.learn_initial_decision,
    }


def policy_net_arch(name: str, *, recurrent: bool = False) -> Any:
    if name == "small":
        return (
            {"pi": [64], "vf": [64]} if recurrent else {"pi": [64, 64], "vf": [64, 64]}
        )
    if name == "medium":
        return (
            {"pi": [128], "vf": [128]}
            if recurrent
            else {"pi": [128, 128], "vf": [128, 128]}
        )
    if name == "large":
        return (
            {"pi": [256], "vf": [256]}
            if recurrent
            else {"pi": [256, 256], "vf": [256, 256]}
        )
    raise ValueError(f"Unknown policy_net_arch={name!r}.")


class GarridoCfTrainingWrapper(gym.Wrapper):
    """Sample a Garrido Cf risk row at each training reset."""

    def __init__(
        self,
        env: gym.Env,
        specs: list[ThesisDesignSpec],
        *,
        seed: int,
    ) -> None:
        super().__init__(env)
        if not specs:
            raise ValueError("At least one Garrido Cf spec is required.")
        self.specs = list(specs)
        self.rng = np.random.default_rng(seed)
        self.current_spec: ThesisDesignSpec | None = None

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        spec = self.specs[int(self.rng.integers(0, len(self.specs)))]
        self.current_spec = spec
        base_env = getattr(self.env, "unwrapped", self.env)
        base_env.enabled_risks = set(spec.enabled_risks)
        base_env.risk_overrides = dict(spec.risk_overrides)
        obs, info = self.env.reset(**kwargs)
        info = dict(info)
        info["training_cfi"] = spec.cfi
        info["training_cfi_family"] = spec.family
        info["training_source_cfi"] = spec.source_cfi
        return obs, info

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.current_spec is not None:
            info = dict(info)
            info["training_cfi"] = self.current_spec.cfi
            info["training_cfi_family"] = self.current_spec.family
            info["training_source_cfi"] = self.current_spec.source_cfi
        return obs, reward, terminated, truncated, info


def make_env(args: argparse.Namespace, seed: int) -> Callable[[], Monitor]:
    kwargs = env_kwargs(args)
    train_specs = (
        [design_spec_for_cfi(cfi) for cfi in parse_cf_range(args.train_cfis)]
        if args.train_cfis
        else []
    )

    def _init() -> Monitor:
        env = make_dkana_thesis_faithful_env(**kwargs)
        if train_specs:
            env = GarridoCfTrainingWrapper(env, train_specs, seed=seed)
        env.reset(seed=seed)
        return Monitor(env)

    return _init


def static_action(shifts: int, *, action_space_mode: str) -> np.ndarray:
    if action_space_mode == "factorized":
        return np.array([0, 0, 0, shifts - 1], dtype=np.int64)
    action = np.zeros(ACTION_DIM, dtype=np.float32)
    action[15 + shifts - 1] = 1.0
    return action


def inventory_action(
    period: int, *, shifts: int = 1, action_space_mode: str
) -> np.ndarray:
    period_index = THESIS_INVENTORY_PERIODS.index(int(period))
    if action_space_mode == "factorized":
        level = period_index + 1
        return np.array([level, level, level, shifts - 1], dtype=np.int64)
    action = np.zeros(ACTION_DIM, dtype=np.float32)
    for node_index in range(3):
        action[node_index * 5 + period_index] = 1.0
    action[15 + shifts - 1] = 1.0
    return action


def thesis_design_action(
    spec: ThesisDesignSpec, *, action_space_mode: str
) -> np.ndarray:
    period = spec.inventory_replenishment_period
    if period is None:
        return static_action(spec.shifts, action_space_mode=action_space_mode)
    return inventory_action(
        int(period),
        shifts=spec.shifts,
        action_space_mode=action_space_mode,
    )


def thesis_design_env_kwargs(
    spec: ThesisDesignSpec, *, action_space_mode: str
) -> dict[str, Any]:
    return {
        "enabled_risks": set(spec.enabled_risks),
        "risk_overrides": dict(spec.risk_overrides),
        "initial_action": thesis_design_action(
            spec,
            action_space_mode=action_space_mode,
        ),
    }


def thesis_risk_env_kwargs(spec: ThesisDesignSpec) -> dict[str, Any]:
    return {
        "enabled_risks": set(spec.enabled_risks),
        "risk_overrides": dict(spec.risk_overrides),
    }


def evaluate_action_policy(
    *,
    args: argparse.Namespace,
    policy_name: str,
    action_fn: Callable[[np.ndarray, dict[str, Any]], np.ndarray],
    seed: int,
    episodes: int | None = None,
    env_kwargs_override: dict[str, Any] | None = None,
    policy_metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    kwargs = env_kwargs(args)
    kwargs.update(env_kwargs_override or {})
    metadata = {"baseline_family": "", "cfi": "", "source_cfi": ""}
    metadata.update(policy_metadata or {})
    episode_count = args.eval_episodes if episodes is None else int(episodes)
    for episode in range(episode_count):
        env = make_dkana_thesis_faithful_env(**kwargs)
        obs, info = env.reset(seed=seed + 10_000 + episode)
        terminated = truncated = False
        reward_total = 0.0
        steps = 0
        shift_counts = {1: 0, 2: 0, 3: 0}
        inventory_target_total_sum = 0.0
        assembly_shift_hours = 0.0
        while not (terminated or truncated):
            action = action_fn(np.asarray(obs, dtype=np.float32), info)
            obs, reward, terminated, truncated, info = env.step(action)
            reward_total += float(reward)
            if info.get("action_phase") == "initial_decision":
                continue
            steps += 1
            decision = info.get("thesis_decision", {})
            shift = int(decision.get("assembly_shifts", 1))
            shift_counts[shift] = shift_counts.get(shift, 0) + 1
            inventory_targets = decision.get("inventory_buffer_targets", {})
            if isinstance(inventory_targets, dict):
                inventory_target_total_sum += float(
                    sum(float(value) for value in inventory_targets.values())
                )
            assembly_shift_hours += float(shift) * float(args.step_size_hours)
        terminal = get_episode_terminal_metrics(env)
        sim = getattr(env.unwrapped, "sim", None)
        total_steps = max(1, steps)
        row = {
            "policy": policy_name,
            "episode": episode,
            "seed": seed,
            "eval_seed": seed + 10_000 + episode,
            "steps": steps,
            "reward_total": reward_total,
            "fill_rate_order_level": terminal["fill_rate_order_level"],
            "backorder_rate_order_level": terminal["backorder_rate_order_level"],
            "order_level_ret_mean": terminal["order_level_ret_mean"],
            "pct_steps_S1": 100.0 * shift_counts.get(1, 0) / total_steps,
            "pct_steps_S2": 100.0 * shift_counts.get(2, 0) / total_steps,
            "pct_steps_S3": 100.0 * shift_counts.get(3, 0) / total_steps,
            "assembly_shift_hours": assembly_shift_hours,
            "inventory_target_total_mean": inventory_target_total_sum / total_steps,
            "pending_backorders_count": float(
                len(getattr(sim, "pending_backorders", [])) if sim is not None else 0.0
            ),
            "pending_backorder_qty": float(
                sum(
                    float(getattr(order, "remaining_qty", 0.0))
                    for order in getattr(sim, "pending_backorders", [])
                )
                if sim is not None
                else 0.0
            ),
            "unattended_orders_total": float(
                getattr(sim, "total_unattended_orders", 0.0) if sim is not None else 0.0
            ),
        }
        row.update(metadata)
        rows.append(row)
        env.close()
    return rows


def evaluate_model_policy(
    *,
    args: argparse.Namespace,
    model: Any,
    policy_name: str,
    seed: int,
    vec_normalize: VecNormalize | None = None,
    episodes: int | None = None,
    env_kwargs_override: dict[str, Any] | None = None,
    policy_metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    recurrent = args.algo == "recurrent_ppo"
    lstm_states: Any = None
    episode_starts: np.ndarray | None = None

    def _predict(obs: np.ndarray, info: dict[str, Any]) -> np.ndarray:
        nonlocal lstm_states, episode_starts
        model_obs = obs
        if vec_normalize is not None:
            model_obs = vec_normalize.normalize_obs(obs.reshape(1, -1))[0]
        if recurrent:
            if episode_starts is None:
                episode_starts = np.ones((1,), dtype=bool)
            action, lstm_states = model.predict(
                model_obs,
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=True,
            )
            episode_starts = np.zeros((1,), dtype=bool)
            return np.asarray(action)
        action, _ = model.predict(model_obs, deterministic=True)
        return np.asarray(action)

    rows = []
    episode_count = args.eval_episodes if episodes is None else int(episodes)
    for episode in range(episode_count):
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)
        rows.extend(
            evaluate_action_policy(
                args=args,
                policy_name=policy_name,
                action_fn=_predict,
                seed=seed + episode * 100_000,
                episodes=1,
                env_kwargs_override=env_kwargs_override,
                policy_metadata=policy_metadata,
            )[:1]
        )
    return rows


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for policy in sorted({row["policy"] for row in rows}):
        bucket = [row for row in rows if row["policy"] == policy]
        out.append(
            {
                "policy": policy,
                "episode_count": len(bucket),
                "reward_total_mean": float(
                    np.mean([row["reward_total"] for row in bucket])
                ),
                "fill_rate_order_level_mean": float(
                    np.mean([row["fill_rate_order_level"] for row in bucket])
                ),
                "order_level_ret_mean": float(
                    np.mean([row["order_level_ret_mean"] for row in bucket])
                ),
                "pct_steps_S1_mean": float(
                    np.mean([row["pct_steps_S1"] for row in bucket])
                ),
                "pct_steps_S2_mean": float(
                    np.mean([row["pct_steps_S2"] for row in bucket])
                ),
                "pct_steps_S3_mean": float(
                    np.mean([row["pct_steps_S3"] for row in bucket])
                ),
                "assembly_shift_hours_mean": float(
                    np.mean([row["assembly_shift_hours"] for row in bucket])
                ),
                "inventory_target_total_mean": float(
                    np.mean([row["inventory_target_total_mean"] for row in bucket])
                ),
                "pending_backorders_count_mean": float(
                    np.mean([row["pending_backorders_count"] for row in bucket])
                ),
                "pending_backorder_qty_mean": float(
                    np.mean([row["pending_backorder_qty"] for row in bucket])
                ),
                "unattended_orders_total_mean": float(
                    np.mean([row["unattended_orders_total"] for row in bucket])
                ),
            }
        )
    return out


def best_garrido_by_family(
    aggregate_rows: list[dict[str, Any]], family: str
) -> dict[str, Any] | None:
    prefix = "garrido_Cf"
    family_rows = [
        row
        for row in aggregate_rows
        if str(row["policy"]).startswith(prefix)
        and str(row["policy"]).endswith(f"_{family}")
    ]
    return (
        max(family_rows, key=lambda row: row["fill_rate_order_level_mean"])
        if family_rows
        else None
    )


def best_static_grid(aggregate_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    grid_rows = [
        row for row in aggregate_rows if str(row["policy"]).startswith("static_grid_")
    ]
    return (
        max(grid_rows, key=lambda row: row["fill_rate_order_level_mean"])
        if grid_rows
        else None
    )


def promotion_status(
    aggregate_rows: list[dict[str, Any]],
    *,
    algo_name: str,
) -> dict[str, Any]:
    ai_row = next((row for row in aggregate_rows if row["policy"] == algo_name), None)
    garrido_rows = [
        row for row in aggregate_rows if str(row["policy"]).startswith("garrido_")
    ]
    best_garrido = (
        max(garrido_rows, key=lambda row: row["fill_rate_order_level_mean"])
        if garrido_rows
        else None
    )
    beats_fill_rate = None
    if ai_row is not None and best_garrido is not None:
        beats_fill_rate = (
            ai_row["fill_rate_order_level_mean"]
            > best_garrido["fill_rate_order_level_mean"]
        )
    best_inventory = best_garrido_by_family(aggregate_rows, "inventory")
    best_capacity = best_garrido_by_family(aggregate_rows, "capacity")
    best_grid = best_static_grid(aggregate_rows)
    return {
        "gate_sequence": [
            "10k_x_3_seeds_implementation_sanity",
            "100k_x_5_seeds_publishability_check",
            "500k_x_5_seeds_only_if_100k_gate_passes",
        ],
        "promotion_rule": (
            "Promote to 500k x 5 only if 100k x 5 beats the best Garrido "
            "static Cf baseline on order-level fill rate, does not hide the win "
            "with worse order-level ReT, and reports S3/inventory usage."
        ),
        "ai_policy": ai_row,
        "best_static_garrido_by_fill_rate": best_garrido,
        "best_static_garrido_inventory_by_fill_rate": best_inventory,
        "best_static_garrido_capacity_by_fill_rate": best_capacity,
        "best_static_grid_by_fill_rate": best_grid,
        "beats_best_garrido_fill_rate": beats_fill_rate,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def paired_garrido_comparisons(
    rows: list[dict[str, Any]], *, algo_name: str
) -> list[dict[str, Any]]:
    comparisons = []
    cfi_values = sorted(
        {
            int(row["cfi"])
            for row in rows
            if str(row.get("cfi", "")).strip()
            and (
                str(row["policy"]).startswith("garrido_")
                or str(row["policy"]).startswith(f"{algo_name}_on_")
            )
        }
    )
    for cfi in cfi_values:
        ai_bucket = [
            row
            for row in rows
            if row.get("cfi") == cfi
            and str(row["policy"]).startswith(f"{algo_name}_on_")
        ]
        static_bucket = [
            row
            for row in rows
            if row.get("cfi") == cfi and str(row["policy"]).startswith("garrido_")
        ]
        if not ai_bucket or not static_bucket:
            continue
        ai_fill = float(np.mean([row["fill_rate_order_level"] for row in ai_bucket]))
        static_fill = float(
            np.mean([row["fill_rate_order_level"] for row in static_bucket])
        )
        ai_ret = float(np.mean([row["order_level_ret_mean"] for row in ai_bucket]))
        static_ret = float(
            np.mean([row["order_level_ret_mean"] for row in static_bucket])
        )
        comparisons.append(
            {
                "cfi": cfi,
                "family": static_bucket[0].get("baseline_family", ""),
                "source_cfi": static_bucket[0].get("source_cfi", ""),
                "ai_fill_rate_order_level_mean": ai_fill,
                "garrido_fill_rate_order_level_mean": static_fill,
                "fill_rate_delta_ai_minus_garrido": ai_fill - static_fill,
                "ai_order_level_ret_mean": ai_ret,
                "garrido_order_level_ret_mean": static_ret,
                "ret_delta_ai_minus_garrido": ai_ret - static_ret,
                "ai_wins_fill_rate": ai_fill > static_fill,
            }
        )
    return comparisons


def train_model(
    args: argparse.Namespace, run_dir: Path
) -> tuple[Any, VecNormalize | None]:
    if args.algo == "recurrent_ppo" and RecurrentPPO is None:
        raise RuntimeError("recurrent_ppo requested but sb3_contrib is not installed.")
    if args.algo == "recurrent_ppo" and args.action_space_mode != "factorized":
        raise ValueError(
            "recurrent_ppo is supported only with action_space_mode=factorized."
        )

    train_env = DummyVecEnv([make_env(args, args.seed)])
    vec_normalize: VecNormalize | None = None
    vec_env: Any = train_env
    if args.vec_normalize:
        vec_normalize = VecNormalize(
            train_env,
            norm_obs=True,
            norm_reward=False,
            clip_obs=10.0,
        )
        vec_env = vec_normalize
    if args.algo == "recurrent_ppo":
        model: Any = RecurrentPPO(
            "MlpLstmPolicy",
            vec_env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            policy_kwargs={
                "net_arch": policy_net_arch(args.policy_net_arch, recurrent=True),
                "lstm_hidden_size": 128,
                "n_lstm_layers": 1,
            },
            seed=args.seed,
            verbose=0,
            device="cpu",
        )
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            policy_kwargs={"net_arch": policy_net_arch(args.policy_net_arch)},
            seed=args.seed,
            verbose=0,
            device="cpu",
        )
    model.learn(total_timesteps=args.train_timesteps)
    model.save(run_dir / f"{args.algo}_thesis_decision")
    if vec_normalize is not None:
        vec_normalize.training = False
        vec_normalize.save(run_dir / "vecnormalize.pkl")
    return model, vec_normalize


def run_single(args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=False)

    model, vec_normalize = train_model(args, run_dir)

    rows: list[dict[str, Any]] = []
    rows.extend(
        evaluate_model_policy(
            args=args,
            model=model,
            policy_name=args.algo,
            seed=args.seed,
            vec_normalize=vec_normalize,
        )
    )
    selected_specs = [
        design_spec_for_cfi(cfi) for cfi in parse_cf_range(args.garrido_cfis)
    ]
    if args.eval_ai_on_garrido_cfis:
        for spec in selected_specs:
            rows.extend(
                evaluate_model_policy(
                    args=args,
                    model=model,
                    policy_name=f"{args.algo}_on_{spec.label}_{spec.family}",
                    seed=args.seed + spec.cfi * 1_000_000,
                    vec_normalize=vec_normalize,
                    episodes=1,
                    env_kwargs_override=thesis_risk_env_kwargs(spec),
                    policy_metadata={
                        "baseline_family": spec.family,
                        "cfi": spec.cfi,
                        "source_cfi": spec.source_cfi,
                    },
                )
            )
    for idx, policy in enumerate(STATIC_POLICIES, start=1):
        rows.extend(
            evaluate_action_policy(
                args=args,
                policy_name=policy,
                action_fn=lambda obs, info, shifts=idx: static_action(
                    shifts, action_space_mode=args.action_space_mode
                ),
                seed=args.seed,
                policy_metadata={
                    "baseline_family": "capacity_reference",
                    "cfi": "",
                    "source_cfi": "",
                },
            )
        )
    for policy_name, period in zip(
        INVENTORY_POLICIES, THESIS_INVENTORY_PERIODS, strict=True
    ):
        rows.extend(
            evaluate_action_policy(
                args=args,
                policy_name=policy_name,
                action_fn=lambda obs, info, p=period: inventory_action(
                    p, action_space_mode=args.action_space_mode
                ),
                seed=args.seed,
                policy_metadata={
                    "baseline_family": "inventory_reference",
                    "cfi": "",
                    "source_cfi": "",
                },
            )
        )
    if args.include_static_grid:
        for period in THESIS_INVENTORY_PERIODS:
            for shifts in (1, 2, 3):
                rows.extend(
                    evaluate_action_policy(
                        args=args,
                        policy_name=f"static_grid_I{period}_S{shifts}",
                        action_fn=lambda obs, info, p=period, s=shifts: inventory_action(
                            p,
                            shifts=s,
                            action_space_mode=args.action_space_mode,
                        ),
                        seed=args.seed,
                        policy_metadata={
                            "baseline_family": "static_inventory_capacity_grid",
                            "cfi": "",
                            "source_cfi": "",
                        },
                    )
                )
    for spec in selected_specs:

        def garrido_action_fn(
            obs: np.ndarray,
            info: dict[str, Any],
            baseline_spec: ThesisDesignSpec = spec,
        ) -> np.ndarray:
            return thesis_design_action(
                baseline_spec,
                action_space_mode=args.action_space_mode,
            )

        rows.extend(
            evaluate_action_policy(
                args=args,
                policy_name=f"garrido_{spec.label}_{spec.family}",
                action_fn=garrido_action_fn,
                seed=args.seed,
                env_kwargs_override=thesis_design_env_kwargs(
                    spec,
                    action_space_mode=args.action_space_mode,
                ),
                policy_metadata={
                    "baseline_family": spec.family,
                    "cfi": spec.cfi,
                    "source_cfi": spec.source_cfi,
                },
            )
        )
    rng = np.random.default_rng(args.seed)

    if args.action_space_mode == "factorized":

        def random_action_fn(obs: np.ndarray, info: dict[str, Any]) -> np.ndarray:
            return np.array(
                [
                    rng.integers(0, 6),
                    rng.integers(0, 6),
                    rng.integers(0, 6),
                    rng.integers(0, 3),
                ],
                dtype=np.int64,
            )

    else:

        def random_action_fn(obs: np.ndarray, info: dict[str, Any]) -> np.ndarray:
            return rng.uniform(0.0, 1.0, ACTION_DIM).astype(np.float32)

    rows.extend(
        evaluate_action_policy(
            args=args,
            policy_name="random",
            action_fn=random_action_fn,
            seed=args.seed,
            policy_metadata={
                "baseline_family": "random",
                "cfi": "",
                "source_cfi": "",
            },
        )
    )

    aggregate_rows = aggregate(rows)
    gate_status = promotion_status(aggregate_rows, algo_name=args.algo)
    paired_comparisons = paired_garrido_comparisons(rows, algo_name=args.algo)
    summary = {
        "created_at": utc_now_iso(),
        "env_kwargs": env_kwargs(args),
        "garrido_cfis": args.garrido_cfis,
        "train_cfis": args.train_cfis,
        "include_static_grid": args.include_static_grid,
        "eval_ai_on_garrido_cfis": args.eval_ai_on_garrido_cfis,
        "train_timesteps": args.train_timesteps,
        "eval_episodes": args.eval_episodes,
        "seed": args.seed,
        "action_contract": "thesis_faithful_dkana_v1",
        "action_space_mode": args.action_space_mode,
        "action_dim": (
            FACTORED_ACTION_DIM
            if args.action_space_mode == "factorized"
            else ACTION_DIM
        ),
        "algo": args.algo,
        "policy_net_arch": args.policy_net_arch,
        "best_static_garrido_by_fill_rate": gate_status[
            "best_static_garrido_by_fill_rate"
        ],
        "best_static_garrido_inventory_by_fill_rate": gate_status[
            "best_static_garrido_inventory_by_fill_rate"
        ],
        "best_static_garrido_capacity_by_fill_rate": gate_status[
            "best_static_garrido_capacity_by_fill_rate"
        ],
        "best_static_grid_by_fill_rate": gate_status["best_static_grid_by_fill_rate"],
        "promotion_gate": gate_status,
        "paired_garrido_win_count": int(
            sum(row["ai_wins_fill_rate"] for row in paired_comparisons)
        ),
        "paired_garrido_comparison_count": len(paired_comparisons),
        "paired_garrido_comparisons": paired_comparisons,
        "aggregate": aggregate_rows,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    write_csv(run_dir / "episode_metrics.csv", rows)
    write_csv(run_dir / "policy_summary.csv", summary["aggregate"])
    write_csv(run_dir / "paired_garrido_comparisons.csv", paired_comparisons)
    if vec_normalize is not None:
        vec_normalize.close()

    print(json.dumps(summary["aggregate"], indent=2))
    print(f"Saved to: {run_dir}")
    return summary


def ablation_configs(args: argparse.Namespace) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    for observation_mode in (
        "decision_reward",
        "env_reward",
        "env_state_reward",
        "env_sdm_history_reward",
    ):
        configs.append(
            {
                "name": f"{args.algo}_{observation_mode}_{args.reward_mode}",
                "observation_mode": observation_mode,
                "reward_mode": args.reward_mode,
                "algo": args.algo,
            }
        )
    for reward_mode in ("control_v1", "ReT_seq_v1"):
        configs.append(
            {
                "name": f"{args.algo}_{args.observation_mode}_{reward_mode}",
                "observation_mode": args.observation_mode,
                "reward_mode": reward_mode,
                "algo": args.algo,
            }
        )
    if RecurrentPPO is not None:
        configs.append(
            {
                "name": f"recurrent_ppo_{args.observation_mode}_{args.reward_mode}",
                "observation_mode": args.observation_mode,
                "reward_mode": args.reward_mode,
                "algo": "recurrent_ppo",
            }
        )

    deduped = []
    seen = set()
    for config in configs:
        key = (config["observation_mode"], config["reward_mode"], config["algo"])
        if key not in seen:
            deduped.append(config)
            seen.add(key)
    return deduped


def main() -> int:
    args = build_parser().parse_args()
    label = args.label or f"{utc_now_iso().replace(':', '').replace('+', 'Z')}"
    run_dir = args.output_root / label
    if args.ablation_suite:
        run_dir.mkdir(parents=True, exist_ok=False)
        summaries = []
        for config in ablation_configs(args):
            child_args = argparse.Namespace(**vars(args))
            child_args.ablation_suite = False
            child_args.observation_mode = config["observation_mode"]
            child_args.reward_mode = config["reward_mode"]
            child_args.algo = config["algo"]
            summaries.append(run_single(child_args, run_dir / config["name"]))
        (run_dir / "ablation_summary.json").write_text(
            json.dumps(summaries, indent=2), encoding="utf-8"
        )
        print(f"Saved ablation suite to: {run_dir}")
        return 0

    run_single(args, run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
