#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a multi-seed PPO benchmark on shift_control + ReT_thesis."
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[11, 22, 33],
        help="Random seeds to evaluate.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=10_000,
        help="Training timesteps per seed.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/benchmarks/ppo_shift_control_ret_thesis"),
        help="Directory where per-seed runs and the benchmark summary are stored.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
        help="Evaluation episodes passed to train_agent.py.",
    )
    parser.add_argument(
        "--random-episodes",
        type=int,
        default=10,
        help="Random baseline episodes passed to train_agent.py.",
    )
    parser.add_argument(
        "--step-size-hours",
        type=float,
        default=168.0,
        help="Step size passed to train_agent.py.",
    )
    parser.add_argument(
        "--risk-level",
        choices=["current", "increased", "severe"],
        default="current",
        help="Risk setting passed to train_agent.py.",
    )
    parser.add_argument(
        "--stochastic-pt",
        action="store_true",
        help="Enable stochastic processing times.",
    )
    return parser


def mean_std(values: list[float]) -> tuple[float, float]:
    mean = float(statistics.fmean(values))
    std = float(statistics.stdev(values)) if len(values) > 1 else 0.0
    return mean, std


def run_seed(args: argparse.Namespace, seed: int) -> dict[str, Any]:
    run_dir = args.output_root / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "train_agent.py",
        "--timesteps",
        str(args.timesteps),
        "--seed",
        str(seed),
        "--env-variant",
        "shift_control",
        "--reward-mode",
        "ReT_thesis",
        "--eval-episodes",
        str(args.eval_episodes),
        "--random-episodes",
        str(args.random_episodes),
        "--step-size-hours",
        str(args.step_size_hours),
        "--risk-level",
        args.risk_level,
        "--output-dir",
        str(run_dir),
    ]
    if args.stochastic_pt:
        command.append("--stochastic-pt")

    subprocess.run(command, check=True)
    log_path = run_dir / "training_log.json"
    with log_path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def build_summary(
    args: argparse.Namespace, runs: list[dict[str, Any]]
) -> dict[str, Any]:
    trained_means = [float(run["trained_eval"]["mean"]) for run in runs]
    random_means = [float(run["random_baseline"]["mean"]) for run in runs]
    improvements = [float(run["improvement_pct"]) for run in runs]
    trained_mean, trained_std = mean_std(trained_means)
    random_mean, random_std = mean_std(random_means)
    improvement_mean, improvement_std = mean_std(improvements)
    return {
        "benchmark": "ppo_shift_control_ret_thesis",
        "config": {
            "seeds": args.seeds,
            "timesteps": args.timesteps,
            "eval_episodes": args.eval_episodes,
            "random_episodes": args.random_episodes,
            "step_size_hours": args.step_size_hours,
            "risk_level": args.risk_level,
            "stochastic_pt": args.stochastic_pt,
        },
        "aggregate": {
            "trained_eval_mean": trained_mean,
            "trained_eval_std": trained_std,
            "random_baseline_mean": random_mean,
            "random_baseline_std": random_std,
            "improvement_pct_mean": improvement_mean,
            "improvement_pct_std": improvement_std,
        },
        "runs": [
            {
                "seed": int(run["config"]["seed"]),
                "trained_eval_mean": float(run["trained_eval"]["mean"]),
                "random_baseline_mean": float(run["random_baseline"]["mean"]),
                "improvement_pct": float(run["improvement_pct"]),
                "output_dir": run["artifacts"]["model"].rsplit("/", 1)[0],
            }
            for run in runs
        ],
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    runs = [run_seed(args, seed) for seed in args.seeds]
    summary = build_summary(args, runs)

    summary_path = args.output_root / "benchmark_summary.json"
    with summary_path.open("w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, indent=2)

    print(f"Wrote benchmark summary to {summary_path}")


if __name__ == "__main__":
    main()
