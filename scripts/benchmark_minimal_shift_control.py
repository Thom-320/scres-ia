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

POLICY_ORDER = ("static_s1", "static_s2", "ppo")
FIXED_POLICY_ACTIONS: dict[str, np.ndarray] = {
    "static_s1": np.array([0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32),
    "static_s2": np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
}
EVAL_EPISODE_SEED_OFFSET = 50_000
PRIMARY_METRICS = (
    "reward_total",
    "fill_rate",
    "backorder_rate",
    "cost_total",
    "cost_mean",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a minimal multi-seed benchmark on the MFSC shift-control env."
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[11, 22, 33],
        help="Benchmark seeds. One PPO model is trained per seed.",
    )
    parser.add_argument(
        "--train-timesteps",
        type=int,
        default=10_000,
        help="Total PPO training timesteps per benchmark seed.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes per policy and seed.",
    )
    parser.add_argument(
        "--step-size-hours",
        type=float,
        default=168.0,
        help="Environment step size in hours.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=260,
        help="Maximum steps per episode.",
    )
    parser.add_argument(
        "--risk-level",
        choices=["current", "increased", "severe"],
        default="increased",
        help="Risk parameter level passed to make_shift_control_env().",
    )
    parser.add_argument(
        "--year-basis",
        choices=YEAR_BASIS_OPTIONS,
        default=DEFAULT_YEAR_BASIS,
        help="Annualization basis passed to make_shift_control_env().",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/benchmarks/minimal_shift_control"),
        help="Directory for CSV and JSON artifacts.",
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


def build_env_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "step_size_hours": args.step_size_hours,
        "risk_level": args.risk_level,
        "max_steps": args.max_steps,
        "year_basis": args.year_basis,
    }


def make_monitored_training_env(args: argparse.Namespace, seed: int) -> callable:
    env_kwargs = build_env_kwargs(args)

    def _init() -> Monitor:
        env = make_shift_control_env(**env_kwargs)
        env.reset(seed=seed)
        return Monitor(env)

    return _init


def train_ppo(args: argparse.Namespace, seed: int) -> tuple[PPO, DummyVecEnv]:
    vec_env = DummyVecEnv([make_monitored_training_env(args, seed)])
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
    policy: str,
    seed: int,
    episode: int,
    eval_seed: int,
    steps: int,
    reward_total: float,
    demanded_total: float,
    delivered_total: float,
    backorder_qty_total: float,
    cost_total: float,
) -> dict[str, Any]:
    if demanded_total > 0:
        backorder_rate = backorder_qty_total / demanded_total
        fill_rate = 1.0 - backorder_rate
    else:
        backorder_rate = 0.0
        fill_rate = 1.0
    cost_mean = cost_total / max(1, steps)
    return {
        "policy": policy,
        "seed": seed,
        "episode": episode,
        "eval_seed": eval_seed,
        "steps": steps,
        "reward_total": reward_total,
        "fill_rate": fill_rate,
        "backorder_rate": backorder_rate,
        "cost_total": cost_total,
        "cost_mean": cost_mean,
        "demanded_total": demanded_total,
        "delivered_total": delivered_total,
        "backorder_qty_total": backorder_qty_total,
    }


def evaluate_policy(
    policy: str,
    *,
    args: argparse.Namespace,
    seed: int,
    model: PPO | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    env_kwargs = build_env_kwargs(args)

    for episode_idx in range(args.eval_episodes):
        eval_seed = seed + EVAL_EPISODE_SEED_OFFSET + episode_idx
        env = make_shift_control_env(**env_kwargs)
        obs, _ = env.reset(seed=eval_seed)
        terminated = False
        truncated = False
        reward_total = 0.0
        demanded_total = 0.0
        delivered_total = 0.0
        backorder_qty_total = 0.0
        cost_total = 0.0
        steps = 0

        while not (terminated or truncated):
            if policy == "ppo":
                if model is None:
                    raise ValueError("PPO evaluation requires a trained model.")
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = static_policy_action(policy)

            obs, reward, terminated, truncated, info = env.step(action)
            reward_total += float(reward)
            demanded_total += float(info.get("new_demanded", 0.0))
            delivered_total += float(info.get("new_delivered", 0.0))
            backorder_qty_total += float(info.get("new_backorder_qty", 0.0))
            cost_total += float(info.get("shift_cost_linear", 0.0))
            steps += 1

        rows.append(
            finalize_episode_metrics(
                policy=policy,
                seed=seed,
                episode=episode_idx + 1,
                eval_seed=eval_seed,
                steps=steps,
                reward_total=reward_total,
                demanded_total=demanded_total,
                delivered_total=delivered_total,
                backorder_qty_total=backorder_qty_total,
                cost_total=cost_total,
            )
        )
        env.close()

    return rows


def aggregate_seed_metrics(
    episode_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in episode_rows:
        grouped.setdefault((str(row["policy"]), int(row["seed"])), []).append(row)

    seed_rows: list[dict[str, Any]] = []
    for (policy, seed), rows in sorted(grouped.items()):
        seed_row: dict[str, Any] = {
            "policy": policy,
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


def aggregate_policy_metrics(
    seed_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
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
            out_row[f"{metric}_mean"] = float(np.mean(values))
            out_row[f"{metric}_std"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            )
            out_row[f"{metric}_ci95_low"] = ci_low
            out_row[f"{metric}_ci95_high"] = ci_high
        policy_rows.append(out_row)
    return policy_rows


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    episode_rows: list[dict[str, Any]] = []
    trained_policies: list[dict[str, Any]] = []

    for seed in args.seeds:
        model, vec_env = train_ppo(args, seed)
        trained_policies.append({"seed": seed, "train_timesteps": args.train_timesteps})

        for policy in POLICY_ORDER:
            rows = evaluate_policy(
                policy,
                args=args,
                seed=seed,
                model=model if policy == "ppo" else None,
            )
            episode_rows.extend(rows)
        vec_env.close()

    seed_rows = aggregate_seed_metrics(episode_rows)
    policy_rows = aggregate_policy_metrics(seed_rows)

    episode_csv = args.output_dir / "episode_metrics.csv"
    policy_csv = args.output_dir / "policy_summary.csv"
    summary_json = args.output_dir / "summary.json"

    save_csv(episode_csv, episode_rows)
    save_csv(policy_csv, policy_rows)

    summary = {
        "config": {
            "seeds": args.seeds,
            "train_timesteps": args.train_timesteps,
            "eval_episodes": args.eval_episodes,
            "step_size_hours": args.step_size_hours,
            "max_steps": args.max_steps,
            "risk_level": args.risk_level,
            "year_basis": args.year_basis,
            "learning_rate": args.learning_rate,
            "n_steps": args.n_steps,
            "batch_size": args.batch_size,
            "n_epochs": args.n_epochs,
            "gamma": args.gamma,
            "gae_lambda": args.gae_lambda,
            "clip_range": args.clip_range,
        },
        "policies": list(POLICY_ORDER),
        "trained_models": trained_policies,
        "artifacts": {
            "episode_metrics_csv": str(episode_csv),
            "policy_summary_csv": str(policy_csv),
            "summary_json": str(summary_json),
        },
        "seed_metrics": seed_rows,
        "policy_summary": policy_rows,
    }
    with summary_json.open("w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, indent=2)

    return summary


def main() -> None:
    args = build_parser().parse_args()
    summary = run_benchmark(args)
    print(f"Wrote minimal benchmark artifacts to {args.output_dir}")
    for row in summary["policy_summary"]:
        print(
            f"{row['policy']}: reward={row['reward_total_mean']:.3f}, "
            f"fill_rate={row['fill_rate_mean']:.3f}, "
            f"backorder_rate={row['backorder_rate_mean']:.3f}, "
            f"cost_mean={row['cost_mean_mean']:.3f}"
        )


if __name__ == "__main__":
    main()
