#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_RUNS = (
    "outputs/benchmarks/control_reward_500k_increased_stopt",
    "outputs/benchmarks/control_reward_500k_severe_stopt",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute paired seed-level inference for control-reward PPO runs."
    )
    parser.add_argument(
        "--run-dirs",
        nargs="+",
        type=Path,
        default=[Path(path) for path in DEFAULT_RUNS],
        help="Benchmark directories containing comparison_table.csv and episode_metrics.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "docs/artifacts/control_reward/control_reward_500k_seed_inference"
        ),
        help="Tracked directory for the inference summary bundle.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=10_000,
        help="Number of bootstrap resamples for the paired mean-difference CI.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260311,
        help="Random seed for bootstrap reproducibility.",
    )
    return parser


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file_obj:
        return list(csv.DictReader(file_obj))


def group_seed_means(
    rows: list[dict[str, str]], *, phase: str, policy: str
) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        if row["phase"] != phase or row["policy"] != policy:
            continue
        grouped.setdefault(row["seed"], []).append(float(row["reward_total"]))
    return {seed: float(np.mean(values)) for seed, values in grouped.items()}


def paired_bootstrap_ci(
    diffs: np.ndarray, *, n_samples: int, rng: np.random.Generator
) -> tuple[float, float]:
    if diffs.size == 0:
        return float("nan"), float("nan")
    if diffs.size == 1:
        value = float(diffs[0])
        return value, value
    samples = rng.choice(diffs, size=(n_samples, diffs.size), replace=True)
    means = samples.mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def exact_sign_flip_pvalue(diffs: np.ndarray) -> float:
    if diffs.size == 0:
        return float("nan")
    observed = abs(float(diffs.mean()))
    flips = np.asarray(list(product([-1.0, 1.0], repeat=diffs.size)), dtype=np.float64)
    flipped_means = (flips * diffs).mean(axis=1)
    return float(np.mean(np.abs(flipped_means) >= observed))


def infer_run(
    run_dir: Path, *, bootstrap_samples: int, rng: np.random.Generator
) -> dict[str, Any]:
    comparison_rows = load_csv_rows(run_dir / "comparison_table.csv")
    if len(comparison_rows) != 1:
        raise ValueError(f"Expected exactly one comparison row in {run_dir}")
    comparison = comparison_rows[0]
    best_static_policy = comparison["best_static_policy"]

    episode_rows = load_csv_rows(run_dir / "episode_metrics.csv")
    ppo_means = group_seed_means(episode_rows, phase="ppo_eval", policy="ppo")
    static_means = group_seed_means(
        episode_rows, phase="static_screen", policy=best_static_policy
    )

    shared_seeds = sorted(set(ppo_means) & set(static_means), key=int)
    ppo_values = np.asarray(
        [ppo_means[seed] for seed in shared_seeds], dtype=np.float64
    )
    static_values = np.asarray(
        [static_means[seed] for seed in shared_seeds], dtype=np.float64
    )
    diffs = ppo_values - static_values

    ci_low, ci_high = paired_bootstrap_ci(diffs, n_samples=bootstrap_samples, rng=rng)
    p_value = exact_sign_flip_pvalue(diffs)

    reward_direction = "better" if diffs.mean() > 0 else "worse"
    return {
        "run_dir": str(run_dir),
        "best_static_policy": best_static_policy,
        "scenario": run_dir.name,
        "seed_count": len(shared_seeds),
        "shared_seeds": shared_seeds,
        "ppo_seed_reward_means": {seed: ppo_means[seed] for seed in shared_seeds},
        "best_static_seed_reward_means": {
            seed: static_means[seed] for seed in shared_seeds
        },
        "paired_reward_differences": {
            seed: float(ppo_means[seed] - static_means[seed]) for seed in shared_seeds
        },
        "mean_difference": float(diffs.mean()),
        "bootstrap_ci95": [ci_low, ci_high],
        "exact_sign_flip_pvalue": p_value,
        "summary": (
            f"PPO is {reward_direction} than {best_static_policy} by "
            f"{float(diffs.mean()):.3f} control-reward points on shared seed means; "
            f"bootstrap CI95 [{ci_low:.3f}, {ci_high:.3f}], exact sign-flip p={p_value:.3f}."
        ),
    }


def render_markdown(results: list[dict[str, Any]]) -> str:
    lines = [
        "# Seed-Level Inference for 500k Control-Reward Runs",
        "",
        "This note uses paired seed-level reward means (`ppo_eval` vs. best fixed static policy) rather than pooled episode rows.",
        "",
    ]
    for result in results:
        lines.extend(
            [
                f"## {result['scenario']}",
                "",
                f"- Best static policy: `{result['best_static_policy']}`",
                f"- Shared seeds: {', '.join(result['shared_seeds'])}",
                f"- Mean reward difference (`PPO - best_static`): {result['mean_difference']:.3f}",
                f"- Bootstrap CI95: [{result['bootstrap_ci95'][0]:.3f}, {result['bootstrap_ci95'][1]:.3f}]",
                f"- Exact sign-flip p-value: {result['exact_sign_flip_pvalue']:.3f}",
                f"- Interpretation: {result['summary']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Use in the paper",
            "",
            "- Treat this as minimal inferential support, not a full significance section.",
            "- Keep the manuscript language at `preliminary`, `competitive`, or `stronger under severe stress`.",
            "- Do not claim formal statistical significance unless you intentionally elevate the inference section later.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = build_parser().parse_args()
    rng = np.random.default_rng(args.seed)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    results = [
        infer_run(run_dir, bootstrap_samples=args.bootstrap_samples, rng=rng)
        for run_dir in args.run_dirs
    ]
    payload = {"bootstrap_samples": args.bootstrap_samples, "results": results}

    json_path = output_dir / "seed_inference.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    markdown_path = output_dir / "seed_inference.md"
    markdown_path.write_text(render_markdown(results), encoding="utf-8")

    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")


if __name__ == "__main__":
    main()
