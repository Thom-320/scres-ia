#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.config import DEFAULT_YEAR_BASIS, YEAR_BASIS_OPTIONS
from supply_chain.external_env_interface import make_shift_control_env

POLICY_ORDER = ("static_s1", "static_s2", "static_s3", "ppo")
STATIC_POLICY_ORDER = ("static_s1", "static_s2", "static_s3")
FIXED_POLICY_ACTIONS: dict[str, np.ndarray] = {
    "static_s1": np.array([0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32),
    "static_s2": np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    "static_s3": np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
}
EVAL_EPISODE_SEED_OFFSET = 80_000
SURVIVOR_REWARD_MARGIN = 1.0
SURVIVOR_FILL_RATE_MARGIN = 0.01
PPO_SERVICE_TOLERANCE = 0.01
PRIMARY_METRICS = (
    "reward_total",
    "service_loss_total",
    "shift_cost_total",
    "mean_disruption_fraction",
    "fill_rate",
    "backorder_rate",
    "ret_thesis_corrected_total",
    "pct_steps_S1",
    "pct_steps_S2",
    "pct_steps_S3",
)
EPISODE_FIELDNAMES = [
    "phase",
    "policy",
    "seed",
    "episode",
    "eval_seed",
    "w_bo",
    "w_cost",
    "w_disr",
    "steps",
    "reward_total",
    "service_loss_total",
    "shift_cost_total",
    "mean_disruption_fraction",
    "fill_rate",
    "backorder_rate",
    "ret_thesis_corrected_total",
    "demanded_total",
    "delivered_total",
    "backorder_qty_total",
    "pct_steps_S1",
    "pct_steps_S2",
    "pct_steps_S3",
]
POLICY_SUMMARY_FIELDNAMES = [
    "phase",
    "policy",
    "w_bo",
    "w_cost",
    "w_disr",
    "seed_count",
]
for _metric in PRIMARY_METRICS:
    POLICY_SUMMARY_FIELDNAMES.extend(
        [
            f"{_metric}_mean",
            f"{_metric}_std",
            f"{_metric}_ci95_low",
            f"{_metric}_ci95_high",
        ]
    )
COMPARISON_FIELDNAMES = [
    "w_bo",
    "w_cost",
    "w_disr",
    "best_static_policy",
    "static_reward_gap_best_minus_s1",
    "ppo_reward_mean",
    "static_s2_reward_mean",
    "best_static_reward_mean",
    "ppo_fill_rate_mean",
    "static_s2_fill_rate_mean",
    "best_static_fill_rate_mean",
    "ppo_backorder_rate_mean",
    "static_s2_backorder_rate_mean",
    "best_static_backorder_rate_mean",
    "ppo_ret_thesis_corrected_total_mean",
    "static_s2_ret_thesis_corrected_total_mean",
    "best_static_ret_thesis_corrected_total_mean",
    "ppo_pct_steps_S1_mean",
    "ppo_pct_steps_S2_mean",
    "ppo_pct_steps_S3_mean",
    "ppo_beats_static_s2",
    "ppo_beats_best_static",
    "collapsed_to_S1",
    "collapsed_to_S2",
    "collapsed_to_S3",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark a control-oriented reward on the MFSC shift-control env."
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    parser.add_argument("--train-timesteps", type=int, default=50_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--step-size-hours", type=float, default=168.0)
    parser.add_argument("--max-steps", type=int, default=260)
    parser.add_argument(
        "--risk-level",
        choices=["current", "increased", "severe"],
        default="increased",
    )
    parser.add_argument(
        "--year-basis",
        choices=YEAR_BASIS_OPTIONS,
        default=DEFAULT_YEAR_BASIS,
    )
    parser.add_argument("--w-bo", type=float, nargs="+", default=[1.0, 2.0, 4.0])
    parser.add_argument("--w-cost", type=float, nargs="+", default=[0.02, 0.06, 0.10])
    parser.add_argument(
        "--w-disr",
        type=float,
        nargs="+",
        default=[0.0],
        help="Control reward disruption weights. Default keeps the first round at 0.0.",
    )
    parser.add_argument(
        "--max-survivors",
        type=int,
        default=4,
        help="Maximum number of weight combinations forwarded to PPO.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/benchmarks/control_reward"),
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    return parser


def ci95(values: list[float]) -> tuple[float, float]:
    if len(values) < 2:
        value = float(values[0]) if values else float("nan")
        return value, value
    arr = np.asarray(values, dtype=np.float64)
    half = 1.96 * arr.std(ddof=1) / np.sqrt(len(arr))
    mean = arr.mean()
    return float(mean - half), float(mean + half)


def static_policy_action(policy: str) -> np.ndarray:
    if policy not in FIXED_POLICY_ACTIONS:
        raise ValueError(f"Unsupported fixed policy {policy!r}.")
    return FIXED_POLICY_ACTIONS[policy].copy()


def make_weight_combos(args: argparse.Namespace) -> list[dict[str, float]]:
    combos: list[dict[str, float]] = []
    for w_bo in args.w_bo:
        for w_cost in args.w_cost:
            for w_disr in args.w_disr:
                combos.append(
                    {
                        "w_bo": float(w_bo),
                        "w_cost": float(w_cost),
                        "w_disr": float(w_disr),
                    }
                )
    return combos


def build_env_kwargs(
    args: argparse.Namespace, weight_combo: dict[str, float]
) -> dict[str, Any]:
    return {
        "reward_mode": "control_v1",
        "step_size_hours": args.step_size_hours,
        "risk_level": args.risk_level,
        "max_steps": args.max_steps,
        "year_basis": args.year_basis,
        **weight_combo,
    }


def make_monitored_training_env(
    args: argparse.Namespace, seed: int, weight_combo: dict[str, float]
) -> callable:
    env_kwargs = build_env_kwargs(args, weight_combo)

    def _init() -> Monitor:
        env = make_shift_control_env(**env_kwargs)
        env.reset(seed=seed)
        return Monitor(env)

    return _init


def train_ppo(
    args: argparse.Namespace, seed: int, weight_combo: dict[str, float]
) -> tuple[PPO, DummyVecEnv]:
    vec_env = DummyVecEnv([make_monitored_training_env(args, seed, weight_combo)])
    model = PPO(
        "MlpPolicy",
        vec_env,
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
    return model, vec_env


def finalize_episode_metrics(
    *,
    phase: str,
    policy: str,
    seed: int,
    episode: int,
    eval_seed: int,
    weight_combo: dict[str, float],
    steps: int,
    reward_total: float,
    service_loss_total: float,
    shift_cost_total: float,
    disruption_fraction_total: float,
    ret_thesis_corrected_total: float,
    demanded_total: float,
    delivered_total: float,
    backorder_qty_total: float,
    shift_counts: dict[int, int],
) -> dict[str, Any]:
    if demanded_total > 0:
        backorder_rate = backorder_qty_total / demanded_total
        fill_rate = 1.0 - backorder_rate
    else:
        backorder_rate = 0.0
        fill_rate = 1.0
    total_steps = max(1, steps)
    return {
        "phase": phase,
        "policy": policy,
        "seed": seed,
        "episode": episode,
        "eval_seed": eval_seed,
        "w_bo": weight_combo["w_bo"],
        "w_cost": weight_combo["w_cost"],
        "w_disr": weight_combo["w_disr"],
        "steps": steps,
        "reward_total": reward_total,
        "service_loss_total": service_loss_total,
        "shift_cost_total": shift_cost_total,
        "mean_disruption_fraction": disruption_fraction_total / total_steps,
        "fill_rate": fill_rate,
        "backorder_rate": backorder_rate,
        "ret_thesis_corrected_total": ret_thesis_corrected_total,
        "demanded_total": demanded_total,
        "delivered_total": delivered_total,
        "backorder_qty_total": backorder_qty_total,
        "pct_steps_S1": 100.0 * shift_counts.get(1, 0) / total_steps,
        "pct_steps_S2": 100.0 * shift_counts.get(2, 0) / total_steps,
        "pct_steps_S3": 100.0 * shift_counts.get(3, 0) / total_steps,
    }


def evaluate_policy(
    phase: str,
    policy: str,
    *,
    args: argparse.Namespace,
    weight_combo: dict[str, float],
    seed: int,
    model: PPO | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    env_kwargs = build_env_kwargs(args, weight_combo)

    for episode_idx in range(args.eval_episodes):
        eval_seed = seed + EVAL_EPISODE_SEED_OFFSET + episode_idx
        env = make_shift_control_env(**env_kwargs)
        obs, _ = env.reset(seed=eval_seed)
        terminated = False
        truncated = False
        reward_total = 0.0
        service_loss_total = 0.0
        shift_cost_total = 0.0
        disruption_fraction_total = 0.0
        ret_thesis_corrected_total = 0.0
        demanded_total = 0.0
        delivered_total = 0.0
        backorder_qty_total = 0.0
        steps = 0
        shift_counts = {1: 0, 2: 0, 3: 0}

        while not (terminated or truncated):
            if policy == "ppo":
                if model is None:
                    raise ValueError("PPO evaluation requires a trained model.")
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = static_policy_action(policy)

            obs, reward, terminated, truncated, info = env.step(action)
            reward_total += float(reward)
            service_loss_total += float(info.get("service_loss_step", 0.0))
            shift_cost_total += float(info.get("shift_cost_step", 0.0))
            disruption_fraction_total += float(
                info.get("disruption_fraction_step", 0.0)
            )
            ret_thesis_corrected_total += float(
                info.get("ret_thesis_corrected_step", 0.0)
            )
            demanded_total += float(info.get("new_demanded", 0.0))
            delivered_total += float(info.get("new_delivered", 0.0))
            backorder_qty_total += float(info.get("new_backorder_qty", 0.0))
            shift_counts[int(info.get("shifts_active", 1))] += 1
            steps += 1

        rows.append(
            finalize_episode_metrics(
                phase=phase,
                policy=policy,
                seed=seed,
                episode=episode_idx + 1,
                eval_seed=eval_seed,
                weight_combo=weight_combo,
                steps=steps,
                reward_total=reward_total,
                service_loss_total=service_loss_total,
                shift_cost_total=shift_cost_total,
                disruption_fraction_total=disruption_fraction_total,
                ret_thesis_corrected_total=ret_thesis_corrected_total,
                demanded_total=demanded_total,
                delivered_total=delivered_total,
                backorder_qty_total=backorder_qty_total,
                shift_counts=shift_counts,
            )
        )
        env.close()

    return rows


def aggregate_seed_metrics(episode_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, float, float, float, int], list[dict[str, Any]]] = {}
    for row in episode_rows:
        key = (
            str(row["phase"]),
            str(row["policy"]),
            float(row["w_bo"]),
            float(row["w_cost"]),
            float(row["w_disr"]),
            int(row["seed"]),
        )
        grouped.setdefault(key, []).append(row)

    seed_rows: list[dict[str, Any]] = []
    for (phase, policy, w_bo, w_cost, w_disr, seed), rows in sorted(grouped.items()):
        seed_row: dict[str, Any] = {
            "phase": phase,
            "policy": policy,
            "w_bo": w_bo,
            "w_cost": w_cost,
            "w_disr": w_disr,
            "seed": seed,
            "episodes": len(rows),
        }
        for metric in PRIMARY_METRICS:
            values = [float(row[metric]) for row in rows]
            seed_row[f"{metric}_mean"] = float(np.mean(values))
            seed_row[f"{metric}_std"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            )
        seed_rows.append(seed_row)
    return seed_rows


def aggregate_policy_metrics(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, float, float, float], list[dict[str, Any]]] = {}
    for row in seed_rows:
        key = (
            str(row["phase"]),
            str(row["policy"]),
            float(row["w_bo"]),
            float(row["w_cost"]),
            float(row["w_disr"]),
        )
        grouped.setdefault(key, []).append(row)

    policy_rows: list[dict[str, Any]] = []
    for (phase, policy, w_bo, w_cost, w_disr), rows in sorted(grouped.items()):
        out_row: dict[str, Any] = {
            "phase": phase,
            "policy": policy,
            "w_bo": w_bo,
            "w_cost": w_cost,
            "w_disr": w_disr,
            "seed_count": len(rows),
        }
        for metric in PRIMARY_METRICS:
            values = [float(row[f"{metric}_mean"]) for row in rows]
            ci_low, ci_high = ci95(values)
            out_row[f"{metric}_mean"] = float(np.mean(values))
            out_row[f"{metric}_std"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            )
            out_row[f"{metric}_ci95_low"] = ci_low
            out_row[f"{metric}_ci95_high"] = ci_high
        policy_rows.append(out_row)
    return policy_rows


def save_csv(
    path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None
) -> None:
    resolved_fieldnames = fieldnames or (list(rows[0].keys()) if rows else None)
    if resolved_fieldnames is None:
        return
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=resolved_fieldnames)
        writer.writeheader()
        if rows:
            writer.writerows(rows)


def policy_lookup(
    policy_rows: list[dict[str, Any]],
    phase: str,
    policy: str,
    weight_combo: dict[str, float],
) -> dict[str, Any] | None:
    for row in policy_rows:
        if (
            row["phase"] == phase
            and row["policy"] == policy
            and float(row["w_bo"]) == float(weight_combo["w_bo"])
            and float(row["w_cost"]) == float(weight_combo["w_cost"])
            and float(row["w_disr"]) == float(weight_combo["w_disr"])
        ):
            return row
    return None


def pick_survivors(
    policy_rows: list[dict[str, Any]], args: argparse.Namespace
) -> list[dict[str, Any]]:
    survivors: list[dict[str, Any]] = []
    combos = {
        (float(row["w_bo"]), float(row["w_cost"]), float(row["w_disr"]))
        for row in policy_rows
        if row["phase"] == "static_screen"
    }
    for w_bo, w_cost, w_disr in sorted(combos):
        weight_combo = {"w_bo": w_bo, "w_cost": w_cost, "w_disr": w_disr}
        s1_row = policy_lookup(policy_rows, "static_screen", "static_s1", weight_combo)
        s2_row = policy_lookup(policy_rows, "static_screen", "static_s2", weight_combo)
        s3_row = policy_lookup(policy_rows, "static_screen", "static_s3", weight_combo)
        if s1_row is None or s2_row is None or s3_row is None:
            continue
        best_static = max(
            (s1_row, s2_row, s3_row), key=lambda row: float(row["reward_total_mean"])
        )
        reward_gap = float(best_static["reward_total_mean"]) - float(
            s1_row["reward_total_mean"]
        )
        fill_gap = float(best_static["fill_rate_mean"]) - float(
            s1_row["fill_rate_mean"]
        )
        if (
            best_static["policy"] in {"static_s2", "static_s3"}
            and reward_gap > SURVIVOR_REWARD_MARGIN
            and fill_gap > SURVIVOR_FILL_RATE_MARGIN
        ):
            survivors.append(
                {
                    **weight_combo,
                    "best_static_policy": best_static["policy"],
                    "static_reward_gap_best_minus_s1": reward_gap,
                    "static_fill_rate_gap_best_minus_s1": fill_gap,
                }
            )
    survivors.sort(
        key=lambda row: float(row["static_reward_gap_best_minus_s1"]), reverse=True
    )
    return survivors[: args.max_survivors]


def compare_policy_to_baseline(
    policy_row: dict[str, Any] | None, baseline_row: dict[str, Any] | None
) -> bool:
    if policy_row is None or baseline_row is None:
        return False
    return (
        float(policy_row["reward_total_mean"])
        > float(baseline_row["reward_total_mean"])
        and float(policy_row["fill_rate_mean"])
        >= float(baseline_row["fill_rate_mean"]) - PPO_SERVICE_TOLERANCE
        and float(policy_row["backorder_rate_mean"])
        <= float(baseline_row["backorder_rate_mean"]) + PPO_SERVICE_TOLERANCE
    )


def build_comparison_rows(
    policy_rows: list[dict[str, Any]], survivors: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    comparison_rows: list[dict[str, Any]] = []
    for survivor in survivors:
        weight_combo = {
            "w_bo": survivor["w_bo"],
            "w_cost": survivor["w_cost"],
            "w_disr": survivor["w_disr"],
        }
        ppo_row = policy_lookup(policy_rows, "ppo_eval", "ppo", weight_combo)
        s2_row = policy_lookup(policy_rows, "static_screen", "static_s2", weight_combo)
        best_row = policy_lookup(
            policy_rows, "static_screen", survivor["best_static_policy"], weight_combo
        )
        comparison_rows.append(
            {
                "w_bo": survivor["w_bo"],
                "w_cost": survivor["w_cost"],
                "w_disr": survivor["w_disr"],
                "best_static_policy": survivor["best_static_policy"],
                "static_reward_gap_best_minus_s1": survivor[
                    "static_reward_gap_best_minus_s1"
                ],
                "ppo_reward_mean": (
                    float(ppo_row["reward_total_mean"]) if ppo_row else None
                ),
                "static_s2_reward_mean": (
                    float(s2_row["reward_total_mean"]) if s2_row else None
                ),
                "best_static_reward_mean": (
                    float(best_row["reward_total_mean"]) if best_row else None
                ),
                "ppo_fill_rate_mean": (
                    float(ppo_row["fill_rate_mean"]) if ppo_row else None
                ),
                "static_s2_fill_rate_mean": (
                    float(s2_row["fill_rate_mean"]) if s2_row else None
                ),
                "best_static_fill_rate_mean": (
                    float(best_row["fill_rate_mean"]) if best_row else None
                ),
                "ppo_backorder_rate_mean": (
                    float(ppo_row["backorder_rate_mean"]) if ppo_row else None
                ),
                "static_s2_backorder_rate_mean": (
                    float(s2_row["backorder_rate_mean"]) if s2_row else None
                ),
                "best_static_backorder_rate_mean": (
                    float(best_row["backorder_rate_mean"]) if best_row else None
                ),
                "ppo_ret_thesis_corrected_total_mean": (
                    float(ppo_row["ret_thesis_corrected_total_mean"])
                    if ppo_row
                    else None
                ),
                "static_s2_ret_thesis_corrected_total_mean": (
                    float(s2_row["ret_thesis_corrected_total_mean"]) if s2_row else None
                ),
                "best_static_ret_thesis_corrected_total_mean": (
                    float(best_row["ret_thesis_corrected_total_mean"])
                    if best_row
                    else None
                ),
                "ppo_pct_steps_S1_mean": (
                    float(ppo_row["pct_steps_S1_mean"]) if ppo_row else None
                ),
                "ppo_pct_steps_S2_mean": (
                    float(ppo_row["pct_steps_S2_mean"]) if ppo_row else None
                ),
                "ppo_pct_steps_S3_mean": (
                    float(ppo_row["pct_steps_S3_mean"]) if ppo_row else None
                ),
                "ppo_beats_static_s2": compare_policy_to_baseline(ppo_row, s2_row),
                "ppo_beats_best_static": compare_policy_to_baseline(ppo_row, best_row),
                "collapsed_to_S1": bool(
                    ppo_row and float(ppo_row["pct_steps_S1_mean"]) > 90.0
                ),
                "collapsed_to_S2": bool(
                    ppo_row and float(ppo_row["pct_steps_S2_mean"]) > 90.0
                ),
                "collapsed_to_S3": bool(
                    ppo_row and float(ppo_row["pct_steps_S3_mean"]) > 90.0
                ),
            }
        )
    return comparison_rows


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    weight_combos = make_weight_combos(args)
    episode_rows: list[dict[str, Any]] = []
    trained_models: list[dict[str, Any]] = []

    for weight_combo in weight_combos:
        for seed in args.seeds:
            for policy in STATIC_POLICY_ORDER:
                episode_rows.extend(
                    evaluate_policy(
                        "static_screen",
                        policy,
                        args=args,
                        weight_combo=weight_combo,
                        seed=seed,
                    )
                )

    seed_rows = aggregate_seed_metrics(episode_rows)
    policy_rows = aggregate_policy_metrics(seed_rows)
    survivors = pick_survivors(policy_rows, args)

    for survivor in survivors:
        weight_combo = {
            "w_bo": survivor["w_bo"],
            "w_cost": survivor["w_cost"],
            "w_disr": survivor["w_disr"],
        }
        for seed in args.seeds:
            model, vec_env = train_ppo(args, seed, weight_combo)
            trained_models.append(
                {
                    "seed": seed,
                    "w_bo": survivor["w_bo"],
                    "w_cost": survivor["w_cost"],
                    "w_disr": survivor["w_disr"],
                    "train_timesteps": args.train_timesteps,
                }
            )
            episode_rows.extend(
                evaluate_policy(
                    "ppo_eval",
                    "ppo",
                    args=args,
                    weight_combo=weight_combo,
                    seed=seed,
                    model=model,
                )
            )
            vec_env.close()

    seed_rows = aggregate_seed_metrics(episode_rows)
    policy_rows = aggregate_policy_metrics(seed_rows)
    comparison_rows = build_comparison_rows(policy_rows, survivors)

    episode_csv = args.output_dir / "episode_metrics.csv"
    policy_csv = args.output_dir / "policy_summary.csv"
    comparison_csv = args.output_dir / "comparison_table.csv"
    summary_json = args.output_dir / "summary.json"

    save_csv(episode_csv, episode_rows, EPISODE_FIELDNAMES)
    save_csv(policy_csv, policy_rows, POLICY_SUMMARY_FIELDNAMES)
    save_csv(comparison_csv, comparison_rows, COMPARISON_FIELDNAMES)

    summary = {
        "config": {
            "seeds": args.seeds,
            "train_timesteps": args.train_timesteps,
            "eval_episodes": args.eval_episodes,
            "step_size_hours": args.step_size_hours,
            "max_steps": args.max_steps,
            "risk_level": args.risk_level,
            "year_basis": args.year_basis,
            "w_bo": args.w_bo,
            "w_cost": args.w_cost,
            "w_disr": args.w_disr,
            "max_survivors": args.max_survivors,
            "learning_rate": args.learning_rate,
            "n_steps": args.n_steps,
            "batch_size": args.batch_size,
            "n_epochs": args.n_epochs,
            "gamma": args.gamma,
            "gae_lambda": args.gae_lambda,
            "clip_range": args.clip_range,
        },
        "phases": ["static_screen", "ppo_eval"],
        "policies": list(POLICY_ORDER),
        "weight_combinations": weight_combos,
        "survivors": survivors,
        "trained_models": trained_models,
        "ppo_skipped": not survivors,
        "artifacts": {
            "episode_metrics_csv": str(episode_csv),
            "policy_summary_csv": str(policy_csv),
            "comparison_table_csv": str(comparison_csv),
            "summary_json": str(summary_json),
        },
        "policy_summary": policy_rows,
        "comparison_table": comparison_rows,
    }
    with summary_json.open("w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, indent=2)
    return summary


def main() -> None:
    args = build_parser().parse_args()
    summary = run_benchmark(args)
    print(f"Wrote control reward benchmark artifacts to {args.output_dir}")
    print(f"Survivors forwarded to PPO: {len(summary['survivors'])}")
    for row in summary["comparison_table"]:
        print(
            f"w_bo={row['w_bo']:.2f}, w_cost={row['w_cost']:.2f}, "
            f"best_static={row['best_static_policy']}, "
            f"ppo_beats_best_static={row['ppo_beats_best_static']}, "
            f"collapsed_to_S1={row['collapsed_to_S1']}"
        )


if __name__ == "__main__":
    main()
