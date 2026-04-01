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
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.benchmark_control_reward import build_metric_contract_metadata
from supply_chain.config import OPERATIONS
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

PRIMARY_METRICS = (
    "reward_total",
    "fill_rate",
    "backorder_rate",
    "order_level_ret_mean",
    "ret_thesis_corrected_total",
    "ret_unified_total",
    "ret_unified_fr_mean",
    "ret_unified_rc_mean",
    "ret_unified_ce_mean",
    "ret_unified_gate_mean",
    "flow_fill_rate",
    "flow_backorder_rate",
    "fill_rate_state_terminal",
    "backorder_rate_state_terminal",
    "terminal_rolling_fill_rate_4w",
    "terminal_rolling_backorder_rate_4w",
    "order_count",
    "completed_order_count",
    "completed_order_fraction",
    "order_case_fill_rate_share",
    "order_case_autotomy_share",
    "order_case_recovery_share",
    "order_case_non_recovery_share",
    "order_case_unfulfilled_share",
    "pct_steps_S1",
    "pct_steps_S2",
    "pct_steps_S3",
)

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
    "ret_thesis_corrected_total",
    "ret_unified_total",
    "ret_unified_fr_mean",
    "ret_unified_rc_mean",
    "ret_unified_ce_mean",
    "ret_unified_gate_mean",
    "flow_fill_rate",
    "flow_backorder_rate",
    "fill_rate_state_terminal",
    "backorder_rate_state_terminal",
    "terminal_rolling_fill_rate_4w",
    "terminal_rolling_backorder_rate_4w",
    "order_count",
    "completed_order_count",
    "completed_order_fraction",
    "order_case_fill_rate_share",
    "order_case_autotomy_share",
    "order_case_recovery_share",
    "order_case_non_recovery_share",
    "order_case_unfulfilled_share",
    "pct_steps_S1",
    "pct_steps_S2",
    "pct_steps_S3",
]

POLICY_ORDER = ("s2_d1.00", "s3_d1.00", "s3_d2.00", "ppo")
COMPARISON_FIELDS = [
    "reward_mode",
    "reward_family",
    "action_contract",
    "observation_version",
    "risk_level",
    "learned_policy",
    "baseline_policy",
    "best_static_policy",
    "ppo_reward_mean",
    "ppo_fill_rate_mean",
    "ppo_backorder_rate_mean",
    "ppo_order_level_ret_mean",
    "ppo_ret_thesis_corrected_mean",
    "ppo_ret_unified_mean",
    "baseline_reward_mean",
    "baseline_fill_rate_mean",
    "baseline_backorder_rate_mean",
    "baseline_order_level_ret_mean",
    "baseline_ret_thesis_corrected_mean",
    "baseline_ret_unified_mean",
    "best_static_reward_mean",
    "best_static_fill_rate_mean",
    "best_static_backorder_rate_mean",
    "best_static_order_level_ret_mean",
    "best_static_ret_thesis_corrected_mean",
    "best_static_ret_unified_mean",
    "ppo_fill_gap_vs_baseline_pp",
    "ppo_fill_gap_vs_best_static_pp",
    "ppo_reward_gap_vs_best_static",
    "ppo_order_level_ret_gap_vs_best_static",
    "ppo_ret_thesis_corrected_gap_vs_best_static",
    "ppo_ret_unified_gap_vs_best_static",
    "ppo_beats_s2_neutral_by_fill",
    "ppo_matches_best_static_by_fill",
    "promote_to_long_run",
]


@dataclass(frozen=True)
class StaticPolicySpec:
    label: str
    assembly_shifts: int
    downstream_multiplier: float


STATIC_POLICY_SPECS: tuple[StaticPolicySpec, ...] = (
    StaticPolicySpec(label="s2_d1.00", assembly_shifts=2, downstream_multiplier=1.0),
    StaticPolicySpec(label="s3_d1.00", assembly_shifts=3, downstream_multiplier=1.0),
    StaticPolicySpec(label="s3_d2.00", assembly_shifts=3, downstream_multiplier=2.0),
)


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


def init_audit_accumulator() -> dict[str, float]:
    return {
        "ret_thesis_corrected_total": 0.0,
        "ret_unified_total": 0.0,
        "ret_unified_fr_sum": 0.0,
        "ret_unified_rc_sum": 0.0,
        "ret_unified_ce_sum": 0.0,
        "ret_unified_gate_sum": 0.0,
    }


def update_audit_accumulator(
    accumulator: dict[str, float], info: dict[str, Any]
) -> None:
    accumulator["ret_thesis_corrected_total"] += float(
        info.get("ret_thesis_corrected_step", 0.0)
    )
    accumulator["ret_unified_total"] += float(info.get("ret_unified_step", 0.0))
    accumulator["ret_unified_fr_sum"] += float(info.get("ret_unified_fr", 0.0))
    accumulator["ret_unified_rc_sum"] += float(info.get("ret_unified_rc", 0.0))
    accumulator["ret_unified_ce_sum"] += float(info.get("ret_unified_ce", 0.0))
    accumulator["ret_unified_gate_sum"] += float(info.get("ret_unified_gate", 0.0))


def make_monitored_training_env(
    args: argparse.Namespace, seed: int
) -> callable[[], Monitor]:
    env_kwargs = build_env_kwargs(args)

    def _init() -> Monitor:
        env = make_track_b_env(**env_kwargs)
        env.reset(seed=seed)
        return Monitor(env)

    return _init


def train_ppo(
    args: argparse.Namespace, seed: int, run_dir: Path
) -> tuple[PPO, VecNormalize]:
    vec_env = DummyVecEnv([make_monitored_training_env(args, seed)])
    vec_norm = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
    )
    model = PPO(
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
    model.learn(total_timesteps=args.train_timesteps)
    model.save(run_dir / "ppo_model.zip")
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
    audit_accumulator: dict[str, float],
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
        "ret_thesis_corrected_total": float(
            audit_accumulator["ret_thesis_corrected_total"]
        ),
        "ret_unified_total": float(audit_accumulator["ret_unified_total"]),
        "ret_unified_fr_mean": float(audit_accumulator["ret_unified_fr_sum"])
        / total_steps,
        "ret_unified_rc_mean": float(audit_accumulator["ret_unified_rc_sum"])
        / total_steps,
        "ret_unified_ce_mean": float(audit_accumulator["ret_unified_ce_sum"])
        / total_steps,
        "ret_unified_gate_mean": float(audit_accumulator["ret_unified_gate_sum"])
        / total_steps,
        "flow_fill_rate": flow_fill_rate,
        "flow_backorder_rate": flow_backorder_rate,
        "fill_rate_state_terminal": float(terminal_metrics["fill_rate_state_terminal"]),
        "backorder_rate_state_terminal": float(
            terminal_metrics["backorder_rate_state_terminal"]
        ),
        "terminal_rolling_fill_rate_4w": float(track_b_context["rolling_fill_rate_4w"]),
        "terminal_rolling_backorder_rate_4w": float(
            track_b_context["rolling_backorder_rate_4w"]
        ),
        "order_count": float(terminal_metrics["order_count"]),
        "completed_order_count": float(terminal_metrics["completed_order_count"]),
        "completed_order_fraction": float(terminal_metrics["completed_order_fraction"]),
        "order_case_fill_rate_share": float(
            terminal_metrics["order_case_fill_rate_share"]
        ),
        "order_case_autotomy_share": float(
            terminal_metrics["order_case_autotomy_share"]
        ),
        "order_case_recovery_share": float(
            terminal_metrics["order_case_recovery_share"]
        ),
        "order_case_non_recovery_share": float(
            terminal_metrics["order_case_non_recovery_share"]
        ),
        "order_case_unfulfilled_share": float(
            terminal_metrics["order_case_unfulfilled_share"]
        ),
        "pct_steps_S1": 100.0 * shift_counts.get(1, 0) / total_steps,
        "pct_steps_S2": 100.0 * shift_counts.get(2, 0) / total_steps,
        "pct_steps_S3": 100.0 * shift_counts.get(3, 0) / total_steps,
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
        audit_accumulator = init_audit_accumulator()
        final_info = info

        while not (terminated or truncated):
            _, reward, terminated, truncated, final_info = env.step(action_payload)
            reward_total += float(reward)
            demanded_total += float(final_info.get("new_demanded", 0.0))
            backorder_qty_total += float(final_info.get("new_backorder_qty", 0.0))
            shift_counts[int(final_info.get("shifts_active", 1))] += 1
            update_audit_accumulator(audit_accumulator, final_info)
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
                audit_accumulator=audit_accumulator,
                track_b_context=final_info["state_constraint_context"][
                    "track_b_context"
                ],
                terminal_metrics=get_episode_terminal_metrics(env),
            )
        )
        env.close()
    return rows


def evaluate_trained_policy(
    *, args: argparse.Namespace, seed: int, model: PPO, vec_norm: VecNormalize
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    env_kwargs = build_env_kwargs(args)
    vec_norm.training = False

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
        audit_accumulator = init_audit_accumulator()
        final_info = info

        while not (terminated or truncated):
            obs_norm = vec_norm.normalize_obs(
                np.asarray(obs, dtype=np.float32)[None, :]
            )
            action, _ = model.predict(obs_norm, deterministic=True)
            obs, reward, terminated, truncated, final_info = env.step(
                np.asarray(action[0], dtype=np.float32)
            )
            reward_total += float(reward)
            demanded_total += float(final_info.get("new_demanded", 0.0))
            backorder_qty_total += float(final_info.get("new_backorder_qty", 0.0))
            shift_counts[int(final_info.get("shifts_active", 1))] += 1
            update_audit_accumulator(audit_accumulator, final_info)
            steps += 1

        rows.append(
            _finalize_episode_row(
                policy="ppo",
                seed=seed,
                episode=episode_idx + 1,
                eval_seed=eval_seed,
                steps=steps,
                reward_total=reward_total,
                demanded_total=demanded_total,
                backorder_qty_total=backorder_qty_total,
                shift_counts=shift_counts,
                audit_accumulator=audit_accumulator,
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


def aggregate_policy_metrics(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in seed_rows:
        grouped.setdefault(str(row["policy"]), []).append(row)

    policy_rows: list[dict[str, Any]] = []
    for policy in POLICY_ORDER:
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


def build_decision_summary(policy_rows: list[dict[str, Any]]) -> dict[str, Any]:
    def ret_metric(row: dict[str, Any]) -> float:
        return float(
            row.get("order_level_ret_mean_mean", row.get("order_level_ret_mean", 0.0))
        )

    by_policy = {str(row["policy"]): row for row in policy_rows}
    baseline = by_policy["s2_d1.00"]
    best_static = max(
        (by_policy["s2_d1.00"], by_policy["s3_d1.00"], by_policy["s3_d2.00"]),
        key=lambda row: (
            float(row["fill_rate_mean"]),
            ret_metric(row),
            -float(row["backorder_rate_mean"]),
        ),
    )
    ppo_row = by_policy["ppo"]
    fill_gap_vs_baseline_pp = 100.0 * (
        float(ppo_row["fill_rate_mean"]) - float(baseline["fill_rate_mean"])
    )
    fill_gap_vs_best_static_pp = 100.0 * (
        float(ppo_row["fill_rate_mean"]) - float(best_static["fill_rate_mean"])
    )
    reward_gap_vs_best_static = float(ppo_row["reward_total_mean"]) - float(
        best_static["reward_total_mean"]
    )
    ret_gap_vs_best_static = ret_metric(ppo_row) - ret_metric(best_static)
    ret_corr_gap_vs_best_static = float(ppo_row["ret_thesis_corrected_total_mean"]) - (
        float(best_static["ret_thesis_corrected_total_mean"])
    )
    ret_unified_gap_vs_best_static = float(ppo_row["ret_unified_total_mean"]) - float(
        best_static["ret_unified_total_mean"]
    )
    return {
        "baseline_policy": "s2_d1.00",
        "best_static_policy": str(best_static["policy"]),
        "ppo_fill_gap_vs_s2_neutral_pp": fill_gap_vs_baseline_pp,
        "ppo_fill_gap_vs_best_static_pp": fill_gap_vs_best_static_pp,
        "ppo_reward_gap_vs_best_static": reward_gap_vs_best_static,
        "ppo_order_level_ret_gap_vs_best_static": ret_gap_vs_best_static,
        "ppo_ret_thesis_corrected_gap_vs_best_static": ret_corr_gap_vs_best_static,
        "ppo_ret_unified_gap_vs_best_static": ret_unified_gap_vs_best_static,
        "ppo_beats_s2_neutral_by_fill": fill_gap_vs_baseline_pp > 0.0,
        "ppo_matches_best_static_by_fill": fill_gap_vs_best_static_pp >= -0.5,
        "promote_to_long_run": (
            fill_gap_vs_baseline_pp > 0.0 and fill_gap_vs_best_static_pp >= -1.0
        ),
    }


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
    by_policy = {str(row["policy"]): row for row in policy_rows}
    baseline = by_policy["s2_d1.00"]
    best_static_name = max(
        ("s2_d1.00", "s3_d1.00", "s3_d2.00"),
        key=lambda policy: (
            float(by_policy[policy]["fill_rate_mean"]),
            float(by_policy[policy]["order_level_ret_mean_mean"]),
            -float(by_policy[policy]["backorder_rate_mean"]),
        ),
    )
    best_static = by_policy[best_static_name]
    ppo_row = by_policy["ppo"]
    reward_contract = build_reward_contract(str(args.reward_mode))
    return [
        {
            "reward_mode": str(args.reward_mode),
            "reward_family": reward_contract["reward_family"],
            "action_contract": "track_b_v1",
            "observation_version": "v7",
            "risk_level": str(args.risk_level),
            "learned_policy": "ppo",
            "baseline_policy": "s2_d1.00",
            "best_static_policy": best_static_name,
            "ppo_reward_mean": float(ppo_row["reward_total_mean"]),
            "ppo_fill_rate_mean": float(ppo_row["fill_rate_mean"]),
            "ppo_backorder_rate_mean": float(ppo_row["backorder_rate_mean"]),
            "ppo_order_level_ret_mean": float(ppo_row["order_level_ret_mean_mean"]),
            "ppo_ret_thesis_corrected_mean": float(
                ppo_row["ret_thesis_corrected_total_mean"]
            ),
            "ppo_ret_unified_mean": float(ppo_row["ret_unified_total_mean"]),
            "baseline_reward_mean": float(baseline["reward_total_mean"]),
            "baseline_fill_rate_mean": float(baseline["fill_rate_mean"]),
            "baseline_backorder_rate_mean": float(baseline["backorder_rate_mean"]),
            "baseline_order_level_ret_mean": float(
                baseline["order_level_ret_mean_mean"]
            ),
            "baseline_ret_thesis_corrected_mean": float(
                baseline["ret_thesis_corrected_total_mean"]
            ),
            "baseline_ret_unified_mean": float(baseline["ret_unified_total_mean"]),
            "best_static_reward_mean": float(best_static["reward_total_mean"]),
            "best_static_fill_rate_mean": float(best_static["fill_rate_mean"]),
            "best_static_backorder_rate_mean": float(
                best_static["backorder_rate_mean"]
            ),
            "best_static_order_level_ret_mean": float(
                best_static["order_level_ret_mean_mean"]
            ),
            "best_static_ret_thesis_corrected_mean": float(
                best_static["ret_thesis_corrected_total_mean"]
            ),
            "best_static_ret_unified_mean": float(
                best_static["ret_unified_total_mean"]
            ),
            "ppo_fill_gap_vs_baseline_pp": float(
                100.0
                * (float(ppo_row["fill_rate_mean"]) - float(baseline["fill_rate_mean"]))
            ),
            "ppo_fill_gap_vs_best_static_pp": float(
                100.0
                * (
                    float(ppo_row["fill_rate_mean"])
                    - float(best_static["fill_rate_mean"])
                )
            ),
            "ppo_reward_gap_vs_best_static": float(
                float(ppo_row["reward_total_mean"])
                - float(best_static["reward_total_mean"])
            ),
            "ppo_order_level_ret_gap_vs_best_static": float(
                float(ppo_row["order_level_ret_mean_mean"])
                - float(best_static["order_level_ret_mean_mean"])
            ),
            "ppo_ret_thesis_corrected_gap_vs_best_static": float(
                float(ppo_row["ret_thesis_corrected_total_mean"])
                - float(best_static["ret_thesis_corrected_total_mean"])
            ),
            "ppo_ret_unified_gap_vs_best_static": float(
                float(ppo_row["ret_unified_total_mean"])
                - float(best_static["ret_unified_total_mean"])
            ),
            "ppo_beats_s2_neutral_by_fill": bool(
                float(ppo_row["fill_rate_mean"]) > float(baseline["fill_rate_mean"])
            ),
            "ppo_matches_best_static_by_fill": bool(
                (
                    100.0
                    * (
                        float(ppo_row["fill_rate_mean"])
                        - float(best_static["fill_rate_mean"])
                    )
                )
                >= -0.5
            ),
            "promote_to_long_run": bool(
                (
                    100.0
                    * (
                        float(ppo_row["fill_rate_mean"])
                        - float(baseline["fill_rate_mean"])
                    )
                )
                > 0.0
                and (
                    100.0
                    * (
                        float(ppo_row["fill_rate_mean"])
                        - float(best_static["fill_rate_mean"])
                    )
                )
                >= -1.0
            ),
        }
    ]


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
        "| Policy | Reward | Fill | Backorder | Order-level ReT | ReT corrected | ReT unified | Rolling fill 4w | Shift mix |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summary["policy_summary"]:
        shift_mix = (
            f"{float(row['pct_steps_S1_mean']):.1f}/"
            f"{float(row['pct_steps_S2_mean']):.1f}/"
            f"{float(row['pct_steps_S3_mean']):.1f}"
        )
        lines.append(
            "| {policy} | {reward:.2f} | {fill:.3f} | {backorder:.3f} | "
            "{ret:.3f} | {ret_corr:.2f} | {ret_unified:.2f} | {rolling_fill:.3f} | {shift_mix} |".format(
                policy=row["policy"],
                reward=float(row["reward_total_mean"]),
                fill=float(row["fill_rate_mean"]),
                backorder=float(row["backorder_rate_mean"]),
                ret=float(row["order_level_ret_mean_mean"]),
                ret_corr=float(row["ret_thesis_corrected_total_mean"]),
                ret_unified=float(row["ret_unified_total_mean"]),
                rolling_fill=float(row["terminal_rolling_fill_rate_4w_mean"]),
                shift_mix=shift_mix,
            )
        )

    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Best static policy: `{decision['best_static_policy']}`",
            (
                f"- PPO fill gap vs `s2_d1.00`: "
                f"{float(decision['ppo_fill_gap_vs_s2_neutral_pp']):+.2f} pp"
            ),
            (
                f"- PPO fill gap vs best static: "
                f"{float(decision['ppo_fill_gap_vs_best_static_pp']):+.2f} pp"
            ),
            (
                f"- PPO reward gap vs best static: "
                f"{float(decision['ppo_reward_gap_vs_best_static']):+.2f}"
            ),
            (
                f"- PPO order-level ReT gap vs best static: "
                f"{float(decision['ppo_order_level_ret_gap_vs_best_static']):+.4f}"
            ),
            (
                f"- PPO ReT corrected gap vs best static: "
                f"{float(decision['ppo_ret_thesis_corrected_gap_vs_best_static']):+.2f}"
            ),
            (
                f"- PPO ReT unified gap vs best static: "
                f"{float(decision['ppo_ret_unified_gap_vs_best_static']):+.2f}"
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

    for seed in args.seeds:
        run_dir = models_dir / f"seed{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        model, vec_norm = train_ppo(args, int(seed), run_dir)
        trained_models.append(
            {
                "seed": int(seed),
                "train_timesteps": int(args.train_timesteps),
                "model_path": str((run_dir / "ppo_model.zip").resolve()),
                "vec_normalize_path": str((run_dir / "vec_normalize.pkl").resolve()),
            }
        )

        for policy in STATIC_POLICY_SPECS:
            episode_rows.extend(
                evaluate_static_policy(policy, args=args, seed=int(seed))
            )
        episode_rows.extend(
            evaluate_trained_policy(
                args=args, seed=int(seed), model=model, vec_norm=vec_norm
            )
        )
        vec_norm.close()

    seed_rows = aggregate_seed_metrics(episode_rows)
    policy_rows = aggregate_policy_metrics(seed_rows)
    decision = build_decision_summary(policy_rows)
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
        "policies": [policy.label for policy in STATIC_POLICY_SPECS] + ["ppo"],
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
    return summary


def main() -> None:
    args = build_parser().parse_args()
    args.invocation = "python scripts/run_track_b_smoke.py " + " ".join(sys.argv[1:])
    summary = run_smoke(args)
    print(f"Wrote Track B smoke bundle to {args.output_dir or 'auto output dir'}")
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
        f"ppo_vs_s2_fill_pp={float(summary['decision']['ppo_fill_gap_vs_s2_neutral_pp']):+.2f}, "
        f"ppo_vs_best_fill_pp={float(summary['decision']['ppo_fill_gap_vs_best_static_pp']):+.2f}, "
        f"promote={summary['decision']['promote_to_long_run']}"
    )


if __name__ == "__main__":
    main()
