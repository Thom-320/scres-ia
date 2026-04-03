#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import statistics
import sys
from typing import Any

import numpy as np
try:
    from sb3_contrib import RecurrentPPO
except ImportError:  # pragma: no cover - runtime guard for optional dependency.
    RecurrentPPO = None
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.benchmark_control_reward import build_metric_contract_metadata
from scripts.track_b_heuristics import HEURISTIC_POLICY_NAMES, make_heuristic_defaults
from supply_chain.config import OPERATIONS
from supply_chain.env_experimental_shifts import REWARD_MODE_OPTIONS
from supply_chain.external_env_interface import (
    get_episode_terminal_metrics,
    get_track_b_env_spec,
    make_track_b_env,
    spec_to_dict,
)

DEFAULT_OUTPUT_ROOT = Path("outputs/benchmarks")
DEFAULT_SEEDS = (11, 22, 33)
DEFAULT_TRAIN_TIMESTEPS = 100_000
DEFAULT_EVAL_EPISODES = 10
DEFAULT_MAX_STEPS = 260
DEFAULT_RET_SEQ_KAPPA = 0.20
EVAL_EPISODE_SEED_OFFSET = 50_000
DOWNSTREAM_NEAR_MAX_THRESHOLD = 1.90

PRIMARY_METRICS = (
    "reward_total",
    "fill_rate",
    "backorder_rate",
    "order_level_ret_mean",
    "flow_fill_rate",
    "flow_backorder_rate",
    "terminal_rolling_fill_rate_4w",
    "terminal_rolling_backorder_rate_4w",
    "pct_steps_S1",
    "pct_steps_S2",
    "pct_steps_S3",
    "op10_multiplier_step_mean",
    "op12_multiplier_step_mean",
    "op10_multiplier_step_p95",
    "op12_multiplier_step_p95",
    "pct_steps_op10_multiplier_ge_190",
    "pct_steps_op12_multiplier_ge_190",
    "pct_steps_both_downstream_ge_190",
    "assembly_hours_total",
    "assembly_cost_index",
)

HOURS_PER_SHIFT = 8.0

EPISODE_FIELDS = [
    "policy",
    "seed",
    "episode",
    "eval_seed",
    "steps",
    "reward_total",
    "fill_rate",
    "backorder_rate",
    "order_level_ret_mean",
    "flow_fill_rate",
    "flow_backorder_rate",
    "terminal_rolling_fill_rate_4w",
    "terminal_rolling_backorder_rate_4w",
    "pct_steps_S1",
    "pct_steps_S2",
    "pct_steps_S3",
    "op10_multiplier_step_mean",
    "op12_multiplier_step_mean",
    "op10_multiplier_step_p95",
    "op12_multiplier_step_p95",
    "pct_steps_op10_multiplier_ge_190",
    "pct_steps_op12_multiplier_ge_190",
    "pct_steps_both_downstream_ge_190",
    "assembly_hours_total",
    "assembly_cost_index",
]

COMPARISON_FIELDS = [
    "reward_mode",
    "reward_family",
    "action_contract",
    "observation_version",
    "risk_level",
    "learned_policy",
    "baseline_policy",
    "best_static_policy",
    "learned_reward_mean",
    "learned_fill_rate_mean",
    "learned_backorder_rate_mean",
    "learned_order_level_ret_mean",
    "baseline_reward_mean",
    "baseline_fill_rate_mean",
    "baseline_backorder_rate_mean",
    "baseline_order_level_ret_mean",
    "best_static_reward_mean",
    "best_static_fill_rate_mean",
    "best_static_backorder_rate_mean",
    "best_static_order_level_ret_mean",
    "learned_fill_gap_vs_baseline_pp",
    "learned_fill_gap_vs_best_static_pp",
    "learned_reward_gap_vs_best_static",
    "learned_order_level_ret_gap_vs_best_static",
    "learned_beats_s2_neutral_by_fill",
    "learned_matches_best_static_by_fill",
    "promote_to_long_run",
]


@dataclass(frozen=True)
class StaticPolicySpec:
    label: str
    assembly_shifts: int
    downstream_multiplier: float


STATIC_POLICY_SPECS: tuple[StaticPolicySpec, ...] = (
    StaticPolicySpec(label="s1_d1.00", assembly_shifts=1, downstream_multiplier=1.0),
    StaticPolicySpec(label="s1_d1.50", assembly_shifts=1, downstream_multiplier=1.5),
    StaticPolicySpec(label="s1_d2.00", assembly_shifts=1, downstream_multiplier=2.0),
    StaticPolicySpec(label="s2_d1.00", assembly_shifts=2, downstream_multiplier=1.0),
    StaticPolicySpec(label="s2_d1.50", assembly_shifts=2, downstream_multiplier=1.5),
    StaticPolicySpec(label="s2_d2.00", assembly_shifts=2, downstream_multiplier=2.0),
    StaticPolicySpec(label="s3_d1.00", assembly_shifts=3, downstream_multiplier=1.0),
    StaticPolicySpec(label="s3_d1.50", assembly_shifts=3, downstream_multiplier=1.5),
    StaticPolicySpec(label="s3_d2.00", assembly_shifts=3, downstream_multiplier=2.0),
)

STATIC_POLICY_ORDER = tuple(policy.label for policy in STATIC_POLICY_SPECS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a short PPO smoke test on the minimal Track B environment "
            "and compare it against the strongest static policies from the DOE."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write the smoke benchmark bundle.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help="Training seeds. One PPO model is trained per seed.",
    )
    parser.add_argument(
        "--train-timesteps",
        type=int,
        default=DEFAULT_TRAIN_TIMESTEPS,
        help="Total PPO timesteps per seed.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=DEFAULT_EVAL_EPISODES,
        help="Evaluation episodes per policy and seed.",
    )
    parser.add_argument(
        "--reward-mode",
        default="ReT_seq_v1",
        choices=list(REWARD_MODE_OPTIONS),
        help="Track B training reward.",
    )
    parser.add_argument(
        "--ret-seq-kappa",
        type=float,
        default=DEFAULT_RET_SEQ_KAPPA,
        help="ReT_seq_v1 kappa. Ignored by other reward modes.",
    )
    parser.add_argument(
        "--risk-level",
        default="adaptive_benchmark_v2",
        help="Track B risk profile.",
    )
    parser.add_argument(
        "--algo",
        choices=["ppo", "recurrent_ppo"],
        default="ppo",
        help="Learned-policy algorithm for the Track B adaptive lane.",
    )
    parser.add_argument(
        "--step-size-hours",
        type=float,
        default=168.0,
        help="Decision cadence in hours.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help="Episode horizon in decision steps.",
    )
    parser.add_argument(
        "--eval-risk-levels",
        nargs="*",
        default=None,
        help=(
            "Additional risk levels for cross-scenario evaluation of the trained "
            "model. E.g. --eval-risk-levels current increased severe. "
            "The model is always trained on --risk-level; these are eval-only."
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    return parser


def default_output_dir(train_timesteps: int) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_OUTPUT_ROOT / f"track_b_smoke_{train_timesteps}_{timestamp}"


def learned_policy_name(args: argparse.Namespace | None = None) -> str:
    return str(getattr(args, "algo", "ppo")) if args is not None else "ppo"


def model_filename(args: argparse.Namespace | None = None) -> str:
    policy = learned_policy_name(args)
    return "ppo_model.zip" if policy == "ppo" else f"{policy}_model.zip"


def ensure_algo_dependencies(args: argparse.Namespace) -> None:
    if learned_policy_name(args) == "recurrent_ppo" and RecurrentPPO is None:
        raise ImportError(
            "Track B recurrent_ppo requires sb3-contrib. Install requirements.txt."
        )


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def ci95(values: list[float]) -> tuple[float, float]:
    if len(values) < 2:
        value = float(values[0]) if values else float("nan")
        return value, value
    arr = np.asarray(values, dtype=np.float64)
    half = 1.96 * arr.std(ddof=1) / np.sqrt(len(arr))
    mean = arr.mean()
    return float(mean - half), float(mean + half)


def build_env_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "reward_mode": args.reward_mode,
        "ret_seq_kappa": args.ret_seq_kappa,
        "risk_level": args.risk_level,
        "step_size_hours": args.step_size_hours,
        "max_steps": args.max_steps,
    }


def build_static_policy_action(policy: StaticPolicySpec) -> dict[str, float | int]:
    downstream_multiplier = float(policy.downstream_multiplier)
    return {
        "op3_q": float(OPERATIONS[3]["q"]),
        "op3_rop": float(OPERATIONS[3]["rop"]),
        "op9_q_min": float(OPERATIONS[9]["q"][0]),
        "op9_q_max": float(OPERATIONS[9]["q"][1]),
        "op9_rop": float(OPERATIONS[9]["rop"]),
        "op10_q_min": float(OPERATIONS[10]["q"][0]) * downstream_multiplier,
        "op10_q_max": float(OPERATIONS[10]["q"][1]) * downstream_multiplier,
        "op12_q_min": float(OPERATIONS[12]["q"][0]) * downstream_multiplier,
        "op12_q_max": float(OPERATIONS[12]["q"][1]) * downstream_multiplier,
        "assembly_shifts": int(policy.assembly_shifts),
    }


def extract_downstream_multipliers(final_info: dict[str, Any]) -> tuple[float, float]:
    clipped_action = final_info.get("clipped_action")
    if isinstance(clipped_action, (list, tuple)) and len(clipped_action) >= 7:
        return (
            float(1.25 + 0.75 * float(clipped_action[5])),
            float(1.25 + 0.75 * float(clipped_action[6])),
        )

    raw_action = final_info.get("raw_action")
    if isinstance(raw_action, dict):
        op10_base = float(OPERATIONS[10]["q"][0])
        op12_base = float(OPERATIONS[12]["q"][0])
        return (
            float(raw_action.get("op10_q_min", op10_base)) / op10_base,
            float(raw_action.get("op12_q_min", op12_base)) / op12_base,
        )

    return 1.0, 1.0


def make_monitored_training_env(
    args: argparse.Namespace, seed: int
) -> callable[[], Monitor]:
    env_kwargs = build_env_kwargs(args)
    wrapper_cls = getattr(args, "_ablation_wrapper", None)

    def _init() -> Monitor:
        env = make_track_b_env(**env_kwargs)
        if wrapper_cls is not None:
            env = wrapper_cls(env)
        env.reset(seed=seed)
        return Monitor(env)

    return _init


def train_ppo(
    args: argparse.Namespace, seed: int, run_dir: Path
) -> tuple[Any, VecNormalize]:
    ensure_algo_dependencies(args)
    vec_env = DummyVecEnv([make_monitored_training_env(args, seed)])
    vec_norm = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
    )
    algo = learned_policy_name(args)
    if algo == "ppo":
        model: Any = PPO(
            "MlpPolicy",
            vec_norm,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            policy_kwargs={"net_arch": {"pi": [64, 64], "vf": [64, 64]}},
            seed=seed,
            verbose=0,
            device="cpu",
        )
    else:
        model = RecurrentPPO(
            "MlpLstmPolicy",
            vec_norm,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            policy_kwargs={
                "net_arch": {"pi": [64], "vf": [64]},
                "lstm_hidden_size": 128,
                "n_lstm_layers": 1,
                "shared_lstm": False,
                "enable_critic_lstm": True,
            },
            seed=seed,
            verbose=0,
            device="cpu",
        )
    model.learn(total_timesteps=args.train_timesteps)
    model.save(run_dir / model_filename(args))
    vec_norm.save(str(run_dir / "vec_normalize.pkl"))
    return model, vec_norm


def _finalize_episode_row(
    *,
    policy: str,
    seed: int,
    episode: int,
    eval_seed: int,
    steps: int,
    reward_total: float,
    demanded_total: float,
    backorder_qty_total: float,
    shift_counts: dict[int, int],
    op10_multipliers: list[float],
    op12_multipliers: list[float],
    track_b_context: dict[str, Any],
    terminal_metrics: dict[str, float],
) -> dict[str, Any]:
    total_steps = max(1, steps)
    if demanded_total > 0.0:
        flow_backorder_rate = backorder_qty_total / demanded_total
        flow_fill_rate = 1.0 - flow_backorder_rate
    else:
        flow_backorder_rate = 0.0
        flow_fill_rate = 1.0
    op10_arr = np.asarray(op10_multipliers or [1.0], dtype=np.float64)
    op12_arr = np.asarray(op12_multipliers or [1.0], dtype=np.float64)
    return {
        "policy": policy,
        "seed": seed,
        "episode": episode,
        "eval_seed": eval_seed,
        "steps": steps,
        "reward_total": reward_total,
        "fill_rate": float(terminal_metrics["fill_rate_order_level"]),
        "backorder_rate": float(terminal_metrics["backorder_rate_order_level"]),
        "order_level_ret_mean": float(terminal_metrics["order_level_ret_mean"]),
        "flow_fill_rate": flow_fill_rate,
        "flow_backorder_rate": flow_backorder_rate,
        "terminal_rolling_fill_rate_4w": float(track_b_context["rolling_fill_rate_4w"]),
        "terminal_rolling_backorder_rate_4w": float(
            track_b_context["rolling_backorder_rate_4w"]
        ),
        "pct_steps_S1": 100.0 * shift_counts.get(1, 0) / total_steps,
        "pct_steps_S2": 100.0 * shift_counts.get(2, 0) / total_steps,
        "pct_steps_S3": 100.0 * shift_counts.get(3, 0) / total_steps,
        "op10_multiplier_step_mean": float(np.mean(op10_arr)),
        "op12_multiplier_step_mean": float(np.mean(op12_arr)),
        "op10_multiplier_step_p95": float(np.percentile(op10_arr, 95)),
        "op12_multiplier_step_p95": float(np.percentile(op12_arr, 95)),
        "pct_steps_op10_multiplier_ge_190": 100.0
        * float(np.mean(op10_arr >= DOWNSTREAM_NEAR_MAX_THRESHOLD)),
        "pct_steps_op12_multiplier_ge_190": 100.0
        * float(np.mean(op12_arr >= DOWNSTREAM_NEAR_MAX_THRESHOLD)),
        "pct_steps_both_downstream_ge_190": 100.0
        * float(
            np.mean(
                (op10_arr >= DOWNSTREAM_NEAR_MAX_THRESHOLD)
                & (op12_arr >= DOWNSTREAM_NEAR_MAX_THRESHOLD)
            )
        ),
        "assembly_hours_total": sum(
            shift_counts.get(s, 0) * s * HOURS_PER_SHIFT * 7.0
            for s in (1, 2, 3)
        ),
        "assembly_cost_index": sum(
            shift_counts.get(s, 0) * s for s in (1, 2, 3)
        )
        / (3.0 * total_steps),
    }


def evaluate_static_policy(
    policy: StaticPolicySpec, *, args: argparse.Namespace, seed: int
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    env_kwargs = build_env_kwargs(args)
    action_payload = build_static_policy_action(policy)

    for episode_idx in range(args.eval_episodes):
        eval_seed = seed + EVAL_EPISODE_SEED_OFFSET + episode_idx
        env = make_track_b_env(**env_kwargs)
        obs, info = env.reset(seed=eval_seed)
        del obs
        terminated = False
        truncated = False
        reward_total = 0.0
        demanded_total = 0.0
        backorder_qty_total = 0.0
        steps = 0
        shift_counts = {1: 0, 2: 0, 3: 0}
        op10_multipliers: list[float] = []
        op12_multipliers: list[float] = []
        final_info = info

        while not (terminated or truncated):
            _, reward, terminated, truncated, final_info = env.step(action_payload)
            reward_total += float(reward)
            demanded_total += float(final_info.get("new_demanded", 0.0))
            backorder_qty_total += float(final_info.get("new_backorder_qty", 0.0))
            shift_counts[int(final_info.get("shifts_active", 1))] += 1
            op10_mult, op12_mult = extract_downstream_multipliers(final_info)
            op10_multipliers.append(op10_mult)
            op12_multipliers.append(op12_mult)
            steps += 1

        rows.append(
            _finalize_episode_row(
                policy=policy.label,
                seed=seed,
                episode=episode_idx + 1,
                eval_seed=eval_seed,
                steps=steps,
                reward_total=reward_total,
                demanded_total=demanded_total,
                backorder_qty_total=backorder_qty_total,
                shift_counts=shift_counts,
                op10_multipliers=op10_multipliers,
                op12_multipliers=op12_multipliers,
                track_b_context=final_info["state_constraint_context"][
                    "track_b_context"
                ],
                terminal_metrics=get_episode_terminal_metrics(env),
            )
        )
        env.close()
    return rows


def evaluate_trained_policy(
    *, args: argparse.Namespace, seed: int, model: Any, vec_norm: VecNormalize
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    env_kwargs = build_env_kwargs(args)
    vec_norm.training = False
    algo = learned_policy_name(args)
    is_recurrent = algo == "recurrent_ppo"

    for episode_idx in range(args.eval_episodes):
        eval_seed = seed + EVAL_EPISODE_SEED_OFFSET + episode_idx
        env = make_track_b_env(**env_kwargs)
        obs, info = env.reset(seed=eval_seed)
        terminated = False
        truncated = False
        reward_total = 0.0
        demanded_total = 0.0
        backorder_qty_total = 0.0
        steps = 0
        shift_counts = {1: 0, 2: 0, 3: 0}
        op10_multipliers: list[float] = []
        op12_multipliers: list[float] = []
        final_info = info
        lstm_states: Any = None
        episode_start = np.ones((1,), dtype=bool)

        while not (terminated or truncated):
            obs_norm = vec_norm.normalize_obs(
                np.asarray(obs, dtype=np.float32)[None, :]
            )
            if is_recurrent:
                action, lstm_states = model.predict(
                    obs_norm,
                    state=lstm_states,
                    episode_start=episode_start,
                    deterministic=True,
                )
            else:
                action, _ = model.predict(obs_norm, deterministic=True)
            obs, reward, terminated, truncated, final_info = env.step(
                np.asarray(action[0], dtype=np.float32)
            )
            reward_total += float(reward)
            demanded_total += float(final_info.get("new_demanded", 0.0))
            backorder_qty_total += float(final_info.get("new_backorder_qty", 0.0))
            shift_counts[int(final_info.get("shifts_active", 1))] += 1
            op10_mult, op12_mult = extract_downstream_multipliers(final_info)
            op10_multipliers.append(op10_mult)
            op12_multipliers.append(op12_mult)
            steps += 1
            episode_start = np.array([terminated or truncated], dtype=bool)

        rows.append(
            _finalize_episode_row(
                policy=algo,
                seed=seed,
                episode=episode_idx + 1,
                eval_seed=eval_seed,
                steps=steps,
                reward_total=reward_total,
                demanded_total=demanded_total,
                backorder_qty_total=backorder_qty_total,
                shift_counts=shift_counts,
                op10_multipliers=op10_multipliers,
                op12_multipliers=op12_multipliers,
                track_b_context=final_info["state_constraint_context"][
                    "track_b_context"
                ],
                terminal_metrics=get_episode_terminal_metrics(env),
            )
        )
        env.close()
    return rows


def evaluate_heuristic_policy(
    label: str,
    heuristic: Any,
    *,
    args: argparse.Namespace,
    seed: int,
) -> list[dict[str, Any]]:
    """Evaluate a Track B heuristic that takes (obs, info) → 7D action array."""
    rows: list[dict[str, Any]] = []
    env_kwargs = build_env_kwargs(args)
    heuristic.reset()

    for episode_idx in range(args.eval_episodes):
        eval_seed = seed + EVAL_EPISODE_SEED_OFFSET + episode_idx
        env = make_track_b_env(**env_kwargs)
        obs, info = env.reset(seed=eval_seed)
        terminated = False
        truncated = False
        reward_total = 0.0
        demanded_total = 0.0
        backorder_qty_total = 0.0
        steps = 0
        shift_counts = {1: 0, 2: 0, 3: 0}
        op10_multipliers: list[float] = []
        op12_multipliers: list[float] = []
        final_info = info
        heuristic.reset()

        while not (terminated or truncated):
            action = heuristic(obs, final_info)
            obs, reward, terminated, truncated, final_info = env.step(
                np.asarray(action, dtype=np.float32)
            )
            reward_total += float(reward)
            demanded_total += float(final_info.get("new_demanded", 0.0))
            backorder_qty_total += float(final_info.get("new_backorder_qty", 0.0))
            shift_counts[int(final_info.get("shifts_active", 1))] += 1
            op10_mult, op12_mult = extract_downstream_multipliers(final_info)
            op10_multipliers.append(op10_mult)
            op12_multipliers.append(op12_mult)
            steps += 1

        rows.append(
            _finalize_episode_row(
                policy=label,
                seed=seed,
                episode=episode_idx + 1,
                eval_seed=eval_seed,
                steps=steps,
                reward_total=reward_total,
                demanded_total=demanded_total,
                backorder_qty_total=backorder_qty_total,
                shift_counts=shift_counts,
                op10_multipliers=op10_multipliers,
                op12_multipliers=op12_multipliers,
                track_b_context=final_info["state_constraint_context"][
                    "track_b_context"
                ],
                terminal_metrics=get_episode_terminal_metrics(env),
            )
        )
        env.close()
    return rows


def aggregate_seed_metrics(episode_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in episode_rows:
        grouped.setdefault((str(row["policy"]), int(row["seed"])), []).append(row)

    seed_rows: list[dict[str, Any]] = []
    for (policy, seed), rows in sorted(grouped.items()):
        out_row: dict[str, Any] = {
            "policy": policy,
            "seed": seed,
            "episodes": len(rows),
        }
        for metric in PRIMARY_METRICS:
            values = [float(row[metric]) for row in rows]
            out_row[f"{metric}_mean"] = float(statistics.fmean(values))
            out_row[f"{metric}_std"] = (
                float(statistics.stdev(values)) if len(values) > 1 else 0.0
            )
        seed_rows.append(out_row)
    return seed_rows


def aggregate_policy_metrics(
    seed_rows: list[dict[str, Any]], *, learned_policy: str = "ppo"
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in seed_rows:
        grouped.setdefault(str(row["policy"]), []).append(row)

    policy_rows: list[dict[str, Any]] = []
    for policy in (*STATIC_POLICY_ORDER, *HEURISTIC_POLICY_NAMES, learned_policy):
        rows = grouped.get(policy, [])
        if not rows:
            continue
        out_row: dict[str, Any] = {
            "policy": policy,
            "seed_count": len(rows),
        }
        for metric in PRIMARY_METRICS:
            values = [float(row[f"{metric}_mean"]) for row in rows]
            ci_low, ci_high = ci95(values)
            out_row[f"{metric}_mean"] = float(statistics.fmean(values))
            out_row[f"{metric}_std"] = (
                float(statistics.stdev(values)) if len(values) > 1 else 0.0
            )
            out_row[f"{metric}_ci95_low"] = ci_low
            out_row[f"{metric}_ci95_high"] = ci_high
        policy_rows.append(out_row)
    return policy_rows


def build_decision_summary(
    policy_rows: list[dict[str, Any]], *, learned_policy: str = "ppo"
) -> dict[str, Any]:
    def ret_metric(row: dict[str, Any]) -> float:
        return float(
            row.get("order_level_ret_mean_mean", row.get("order_level_ret_mean", 0.0))
        )

    by_policy = {str(row["policy"]): row for row in policy_rows}
    baseline = by_policy["s2_d1.00"]
    best_static = max(
        (by_policy[policy_name] for policy_name in STATIC_POLICY_ORDER),
        key=lambda row: (
            float(row["fill_rate_mean"]),
            ret_metric(row),
            -float(row["backorder_rate_mean"]),
        ),
    )
    learned_row = by_policy[learned_policy]
    fill_gap_vs_baseline_pp = 100.0 * (
        float(learned_row["fill_rate_mean"]) - float(baseline["fill_rate_mean"])
    )
    fill_gap_vs_best_static_pp = 100.0 * (
        float(learned_row["fill_rate_mean"]) - float(best_static["fill_rate_mean"])
    )
    reward_gap_vs_best_static = float(learned_row["reward_total_mean"]) - float(
        best_static["reward_total_mean"]
    )
    ret_gap_vs_best_static = ret_metric(learned_row) - ret_metric(best_static)
    decision = {
        "learned_policy": learned_policy,
        "baseline_policy": "s2_d1.00",
        "best_static_policy": str(best_static["policy"]),
        "learned_fill_gap_vs_s2_neutral_pp": fill_gap_vs_baseline_pp,
        "learned_fill_gap_vs_best_static_pp": fill_gap_vs_best_static_pp,
        "learned_reward_gap_vs_best_static": reward_gap_vs_best_static,
        "learned_order_level_ret_gap_vs_best_static": ret_gap_vs_best_static,
        "learned_beats_s2_neutral_by_fill": fill_gap_vs_baseline_pp > 0.0,
        "learned_matches_best_static_by_fill": fill_gap_vs_best_static_pp >= -0.5,
        "promote_to_long_run": (
            fill_gap_vs_baseline_pp > 0.0 and fill_gap_vs_best_static_pp >= -1.0
        ),
    }
    if learned_policy == "ppo":
        decision.update(
            {
                "ppo_fill_gap_vs_s2_neutral_pp": fill_gap_vs_baseline_pp,
                "ppo_fill_gap_vs_best_static_pp": fill_gap_vs_best_static_pp,
                "ppo_reward_gap_vs_best_static": reward_gap_vs_best_static,
                "ppo_order_level_ret_gap_vs_best_static": ret_gap_vs_best_static,
                "ppo_beats_s2_neutral_by_fill": fill_gap_vs_baseline_pp > 0.0,
                "ppo_matches_best_static_by_fill": fill_gap_vs_best_static_pp
                >= -0.5,
            }
        )
    return decision


def build_reward_contract(reward_mode: str) -> dict[str, Any]:
    reward_family = (
        "operational_penalty"
        if reward_mode in ("control_v1", "control_v1_pbrs")
        else "resilience_index"
    )
    return {
        "reward_mode": reward_mode,
        "reward_family": reward_family,
        "cross_mode_reward_comparison_allowed": False,
        "within_run_reward_comparison_allowed": True,
        "selection_metrics": [
            "fill_rate",
            "backorder_rate",
            "order_level_ret_mean",
            "reward_total_within_same_reward_mode_only",
        ],
    }


def build_comparison_rows(
    policy_rows: list[dict[str, Any]], *, args: argparse.Namespace
) -> list[dict[str, Any]]:
    learned_policy = learned_policy_name(args)
    by_policy = {str(row["policy"]): row for row in policy_rows}
    baseline = by_policy["s2_d1.00"]
    best_static_name = max(
        STATIC_POLICY_ORDER,
        key=lambda policy: (
            float(by_policy[policy]["fill_rate_mean"]),
            float(by_policy[policy]["order_level_ret_mean_mean"]),
            -float(by_policy[policy]["backorder_rate_mean"]),
        ),
    )
    best_static = by_policy[best_static_name]
    learned_row = by_policy[learned_policy]
    reward_contract = build_reward_contract(str(args.reward_mode))
    row = {
        "reward_mode": str(args.reward_mode),
        "reward_family": reward_contract["reward_family"],
        "action_contract": "track_b_v1",
        "observation_version": "v7",
        "risk_level": str(args.risk_level),
        "learned_policy": learned_policy,
        "baseline_policy": "s2_d1.00",
        "best_static_policy": best_static_name,
        "learned_reward_mean": float(learned_row["reward_total_mean"]),
        "learned_fill_rate_mean": float(learned_row["fill_rate_mean"]),
        "learned_backorder_rate_mean": float(learned_row["backorder_rate_mean"]),
        "learned_order_level_ret_mean": float(
            learned_row["order_level_ret_mean_mean"]
        ),
        "baseline_reward_mean": float(baseline["reward_total_mean"]),
        "baseline_fill_rate_mean": float(baseline["fill_rate_mean"]),
        "baseline_backorder_rate_mean": float(baseline["backorder_rate_mean"]),
        "baseline_order_level_ret_mean": float(
            baseline["order_level_ret_mean_mean"]
        ),
        "best_static_reward_mean": float(best_static["reward_total_mean"]),
        "best_static_fill_rate_mean": float(best_static["fill_rate_mean"]),
        "best_static_backorder_rate_mean": float(best_static["backorder_rate_mean"]),
        "best_static_order_level_ret_mean": float(
            best_static["order_level_ret_mean_mean"]
        ),
        "learned_fill_gap_vs_baseline_pp": float(
            100.0
            * (
                float(learned_row["fill_rate_mean"]) - float(baseline["fill_rate_mean"])
            )
        ),
        "learned_fill_gap_vs_best_static_pp": float(
            100.0
            * (
                float(learned_row["fill_rate_mean"])
                - float(best_static["fill_rate_mean"])
            )
        ),
        "learned_reward_gap_vs_best_static": float(
            float(learned_row["reward_total_mean"])
            - float(best_static["reward_total_mean"])
        ),
        "learned_order_level_ret_gap_vs_best_static": float(
            float(learned_row["order_level_ret_mean_mean"])
            - float(best_static["order_level_ret_mean_mean"])
        ),
        "learned_beats_s2_neutral_by_fill": bool(
            float(learned_row["fill_rate_mean"]) > float(baseline["fill_rate_mean"])
        ),
        "learned_matches_best_static_by_fill": bool(
            (
                100.0
                * (
                    float(learned_row["fill_rate_mean"])
                    - float(best_static["fill_rate_mean"])
                )
            )
            >= -0.5
        ),
        "promote_to_long_run": bool(
            (
                100.0
                * (
                    float(learned_row["fill_rate_mean"])
                    - float(baseline["fill_rate_mean"])
                )
            )
            > 0.0
            and (
                100.0
                * (
                    float(learned_row["fill_rate_mean"])
                    - float(best_static["fill_rate_mean"])
                )
            )
            >= -1.0
        ),
    }
    if learned_policy == "ppo":
        row.update(
            {
                "ppo_reward_mean": row["learned_reward_mean"],
                "ppo_fill_rate_mean": row["learned_fill_rate_mean"],
                "ppo_backorder_rate_mean": row["learned_backorder_rate_mean"],
                "ppo_order_level_ret_mean": row["learned_order_level_ret_mean"],
                "ppo_fill_gap_vs_baseline_pp": row["learned_fill_gap_vs_baseline_pp"],
                "ppo_fill_gap_vs_best_static_pp": row[
                    "learned_fill_gap_vs_best_static_pp"
                ],
                "ppo_reward_gap_vs_best_static": row[
                    "learned_reward_gap_vs_best_static"
                ],
                "ppo_order_level_ret_gap_vs_best_static": row[
                    "learned_order_level_ret_gap_vs_best_static"
                ],
                "ppo_beats_s2_neutral_by_fill": row[
                    "learned_beats_s2_neutral_by_fill"
                ],
                "ppo_matches_best_static_by_fill": row[
                    "learned_matches_best_static_by_fill"
                ],
            }
        )
    return [row]


def render_markdown(summary: dict[str, Any]) -> str:
    config = summary["config"]
    decision = summary["decision"]
    lines = [
        "# Track B Smoke Benchmark",
        "",
        "## Config",
        "",
        f"- Train timesteps: {config['train_timesteps']}",
        f"- Seeds: {config['seeds']}",
        f"- Eval episodes: {config['eval_episodes']}",
        f"- Reward mode: {config['reward_mode']}",
        f"- Risk level: {config['risk_level']}",
        "",
        "## Policy Summary",
        "",
        "| Policy | Reward | Fill | Backorder | Order-level ReT | Rolling fill 4w | Shift mix | Asm hrs | Cost idx |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    for row in summary["policy_summary"]:
        shift_mix = (
            f"{float(row['pct_steps_S1_mean']):.1f}/"
            f"{float(row['pct_steps_S2_mean']):.1f}/"
            f"{float(row['pct_steps_S3_mean']):.1f}"
        )
        lines.append(
            "| {policy} | {reward:.2f} | {fill:.3f} | {backorder:.3f} | "
            "{ret:.3f} | {rolling_fill:.3f} | {shift_mix} | "
            "{asm_hrs:.0f} | {cost_idx:.3f} |".format(
                policy=row["policy"],
                reward=float(row["reward_total_mean"]),
                fill=float(row["fill_rate_mean"]),
                backorder=float(row["backorder_rate_mean"]),
                ret=float(row["order_level_ret_mean_mean"]),
                rolling_fill=float(row["terminal_rolling_fill_rate_4w_mean"]),
                shift_mix=shift_mix,
                asm_hrs=float(row.get("assembly_hours_total_mean", 0.0)),
                cost_idx=float(row.get("assembly_cost_index_mean", 0.0)),
            )
        )

    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Learned policy: `{decision['learned_policy']}`",
            f"- Best static policy: `{decision['best_static_policy']}`",
            (
                f"- {decision['learned_policy']} fill gap vs `s2_d1.00`: "
                f"{float(decision['learned_fill_gap_vs_s2_neutral_pp']):+.2f} pp"
            ),
            (
                f"- {decision['learned_policy']} fill gap vs best static: "
                f"{float(decision['learned_fill_gap_vs_best_static_pp']):+.2f} pp"
            ),
            (
                f"- {decision['learned_policy']} reward gap vs best static: "
                f"{float(decision['learned_reward_gap_vs_best_static']):+.2f}"
            ),
            (
                f"- {decision['learned_policy']} order-level ReT gap vs best static: "
                f"{float(decision['learned_order_level_ret_gap_vs_best_static']):+.4f}"
            ),
            f"- Promote to long run: `{decision['promote_to_long_run']}`",
            "",
        ]
    )
    return "\n".join(lines)


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir or default_output_dir(args.train_timesteps)
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    episode_rows: list[dict[str, Any]] = []
    trained_models: list[dict[str, Any]] = []
    learned_policy = learned_policy_name(args)

    for seed in args.seeds:
        run_dir = models_dir / f"seed{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        model, vec_norm = train_ppo(args, int(seed), run_dir)
        trained_models.append(
            {
                "seed": int(seed),
                "algo": learned_policy,
                "train_timesteps": int(args.train_timesteps),
                "model_path": str((run_dir / model_filename(args)).resolve()),
                "vec_normalize_path": str((run_dir / "vec_normalize.pkl").resolve()),
            }
        )

        for policy in STATIC_POLICY_SPECS:
            episode_rows.extend(
                evaluate_static_policy(policy, args=args, seed=int(seed))
            )
        for h_label, h_policy in make_heuristic_defaults().items():
            episode_rows.extend(
                evaluate_heuristic_policy(
                    h_label, h_policy, args=args, seed=int(seed)
                )
            )
        episode_rows.extend(
            evaluate_trained_policy(
                args=args, seed=int(seed), model=model, vec_norm=vec_norm
            )
        )
        vec_norm.close()

    seed_rows = aggregate_seed_metrics(episode_rows)
    policy_rows = aggregate_policy_metrics(seed_rows, learned_policy=learned_policy)
    decision = build_decision_summary(policy_rows, learned_policy=learned_policy)
    comparison_rows = build_comparison_rows(policy_rows, args=args)

    episode_csv = output_dir / "episode_metrics.csv"
    seed_csv = output_dir / "seed_metrics.csv"
    policy_csv = output_dir / "policy_summary.csv"
    comparison_csv = output_dir / "comparison_table.csv"
    summary_json = output_dir / "summary.json"
    summary_md = output_dir / "summary.md"

    save_csv(episode_csv, episode_rows)
    save_csv(seed_csv, seed_rows)
    save_csv(policy_csv, policy_rows)
    save_csv(comparison_csv, comparison_rows)
    reward_contract = build_reward_contract(str(args.reward_mode))

    summary = {
        "config": {
            "seeds": [int(seed) for seed in args.seeds],
            "train_timesteps": int(args.train_timesteps),
            "eval_episodes": int(args.eval_episodes),
            "algo": learned_policy,
            "reward_mode": args.reward_mode,
            "ret_seq_kappa": float(args.ret_seq_kappa),
            "risk_level": args.risk_level,
            "step_size_hours": float(args.step_size_hours),
            "max_steps": int(args.max_steps),
            "observation_version": "v7",
            "action_contract": "track_b_v1",
            "year_basis": "thesis",
            "stochastic_pt": True,
            "learning_rate": float(args.learning_rate),
            "n_steps": int(args.n_steps),
            "batch_size": int(args.batch_size),
            "n_epochs": int(args.n_epochs),
            "gamma": float(args.gamma),
            "gae_lambda": float(args.gae_lambda),
            "clip_range": float(args.clip_range),
        },
        "backbone": {
            "code_ref": "HEAD",
            "benchmark_protocol": "track_b_minimal_v1",
            "env_variant": "track_b_adaptive_control",
            "algo": learned_policy,
            "reward_mode": args.reward_mode,
            "observation_version": "v7",
            "action_contract": "track_b_v1",
            "risk_level": args.risk_level,
            "year_basis": "thesis",
            "stochastic_pt": True,
            "step_size_hours": float(args.step_size_hours),
            "max_steps": int(args.max_steps),
        },
        "env_spec": spec_to_dict(
            get_track_b_env_spec(
                reward_mode=args.reward_mode,
                observation_version="v7",
                step_size_hours=args.step_size_hours,
            )
        ),
        "metric_contract": build_metric_contract_metadata(),
        "reward_contract": reward_contract,
        "benchmark_metadata": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "command": getattr(args, "invocation", None),
        },
        "trained_models": trained_models,
        "policies": [policy.label for policy in STATIC_POLICY_SPECS] + [learned_policy],
        "decision": decision,
        "seed_metrics": seed_rows,
        "policy_summary": policy_rows,
        "comparison_table": comparison_rows,
        "artifacts": {
            "episode_metrics_csv": str(episode_csv.resolve()),
            "seed_metrics_csv": str(seed_csv.resolve()),
            "policy_summary_csv": str(policy_csv.resolve()),
            "comparison_table_csv": str(comparison_csv.resolve()),
            "summary_json": str(summary_json.resolve()),
            "summary_md": str(summary_md.resolve()),
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary_md.write_text(render_markdown(summary), encoding="utf-8")

    # Cross-scenario evaluation (if --eval-risk-levels provided)
    eval_risk_levels = getattr(args, "eval_risk_levels", None) or []
    if eval_risk_levels and trained_models:
        cross_eval_dir = output_dir / "cross_scenario"
        cross_eval_dir.mkdir(parents=True, exist_ok=True)
        for risk_level in eval_risk_levels:
            if risk_level == args.risk_level:
                continue
            cross_rows: list[dict[str, Any]] = []
            cross_args = argparse.Namespace(**vars(args))
            cross_args.risk_level = risk_level
            for model_info in trained_models:
                seed = int(model_info["seed"])
                model_path = Path(model_info["model_path"])
                vec_path = Path(model_info["vec_normalize_path"])
                if not model_path.exists() or not vec_path.exists():
                    continue
                algo_name = model_info["algo"]
                if algo_name == "recurrent_ppo" and RecurrentPPO is not None:
                    loaded_model = RecurrentPPO.load(str(model_path))
                else:
                    loaded_model = PPO.load(str(model_path))
                cross_env = DummyVecEnv(
                    [make_monitored_training_env(cross_args, seed)]
                )
                loaded_vec = VecNormalize.load(str(vec_path), cross_env)
                loaded_vec.training = False
                for policy in STATIC_POLICY_SPECS:
                    cross_rows.extend(
                        evaluate_static_policy(
                            policy, args=cross_args, seed=seed
                        )
                    )
                for h_label, h_policy in make_heuristic_defaults().items():
                    cross_rows.extend(
                        evaluate_heuristic_policy(
                            h_label, h_policy, args=cross_args, seed=seed
                        )
                    )
                cross_rows.extend(
                    evaluate_trained_policy(
                        args=cross_args,
                        seed=seed,
                        model=loaded_model,
                        vec_norm=loaded_vec,
                    )
                )
                loaded_vec.close()
            if cross_rows:
                cross_seed = aggregate_seed_metrics(cross_rows)
                cross_policy = aggregate_policy_metrics(
                    cross_seed, learned_policy=learned_policy
                )
                save_csv(
                    cross_eval_dir / f"episode_metrics_{risk_level}.csv",
                    cross_rows,
                )
                save_csv(
                    cross_eval_dir / f"policy_summary_{risk_level}.csv",
                    cross_policy,
                )
                print(f"  Cross-eval ({risk_level}): {len(cross_rows)} episodes")

    return summary


def main() -> None:
    args = build_parser().parse_args()
    args.invocation = "python scripts/run_track_b_smoke.py " + " ".join(sys.argv[1:])
    summary = run_smoke(args)
    print(f"Wrote Track B smoke bundle to {args.output_dir or 'auto output dir'}")
    learned_policy = summary["decision"]["learned_policy"]
    for row in summary["policy_summary"]:
        print(
            f"{row['policy']}: reward={float(row['reward_total_mean']):.2f}, "
            f"fill={float(row['fill_rate_mean']):.3f}, "
            f"backorder={float(row['backorder_rate_mean']):.3f}, "
            f"ret={float(row['order_level_ret_mean_mean']):.3f}"
        )
    print(
        "Decision: "
        f"best_static={summary['decision']['best_static_policy']}, "
        f"{learned_policy}_vs_s2_fill_pp={float(summary['decision']['learned_fill_gap_vs_s2_neutral_pp']):+.2f}, "
        f"{learned_policy}_vs_best_fill_pp={float(summary['decision']['learned_fill_gap_vs_best_static_pp']):+.2f}, "
        f"promote={summary['decision']['promote_to_long_run']}"
    )


if __name__ == "__main__":
    main()
