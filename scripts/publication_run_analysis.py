#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.control_reward_seed_inference import (
    exact_sign_flip_pvalue,
    paired_bootstrap_ci,
)

DEFAULT_RUNS = (
    "outputs/benchmarks/control_reward",
    "outputs/benchmarks/control_reward_sac_fs1",
    "outputs/benchmarks/control_reward_ppo_fs4",
    "outputs/benchmarks/control_reward_recurrent_ppo_fs1",
)

BASELINE_TABLE_FIELDS = [
    "run_label",
    "scenario",
    "algo",
    "frame_stack",
    "observation_version",
    "w_bo",
    "w_cost",
    "w_disr",
    "policy_role",
    "policy_name",
    "reward_total_mean",
    "fill_rate_mean",
    "backorder_rate_mean",
    "ret_thesis_corrected_total_mean",
]

COMPARISON_FIELDS = [
    "run_label",
    "scenario",
    "algo",
    "frame_stack",
    "observation_version",
    "w_bo",
    "w_cost",
    "w_disr",
    "learned_policy",
    "best_static_policy",
    "best_heuristic_policy",
    "mean_diff_vs_best_static",
    "ci95_low_vs_best_static",
    "ci95_high_vs_best_static",
    "sign_flip_pvalue_vs_best_static",
    "mean_diff_vs_best_heuristic",
    "ci95_low_vs_best_heuristic",
    "ci95_high_vs_best_heuristic",
    "sign_flip_pvalue_vs_best_heuristic",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize publication-ready comparisons across control benchmark runs."
    )
    parser.add_argument(
        "--run-dirs",
        nargs="+",
        type=Path,
        default=[Path(path) for path in DEFAULT_RUNS],
        help="Benchmark directories containing summary.json, policy_summary.csv, comparison_table.csv, and episode_metrics.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/artifacts/control_reward/publication_run_analysis"),
        help="Directory for publication analysis outputs.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=10_000,
        help="Bootstrap samples for paired seed-mean confidence intervals.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260312,
        help="Random seed for bootstrap reproducibility.",
    )
    return parser


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file_obj:
        return list(csv.DictReader(file_obj))


def save_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        if rows:
            writer.writerows(rows)


def run_label(config: dict[str, Any]) -> str:
    return (
        f"{config['algo']}_fs{int(config['frame_stack'])}_"
        f"{str(config['observation_version'])}"
    )


def policy_lookup(
    rows: list[dict[str, str]],
    *,
    phase: str,
    policy: str,
    weight_combo: dict[str, float],
) -> dict[str, str] | None:
    for row in rows:
        if (
            row["phase"] == phase
            and row["policy"] == policy
            and float(row["w_bo"]) == float(weight_combo["w_bo"])
            and float(row["w_cost"]) == float(weight_combo["w_cost"])
            and float(row["w_disr"]) == float(weight_combo["w_disr"])
        ):
            return row
    return None


def group_seed_means(
    rows: list[dict[str, str]],
    *,
    phase: str,
    policy: str,
    weight_combo: dict[str, float],
) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        if (
            row["phase"] != phase
            or row["policy"] != policy
            or float(row["w_bo"]) != float(weight_combo["w_bo"])
            or float(row["w_cost"]) != float(weight_combo["w_cost"])
            or float(row["w_disr"]) != float(weight_combo["w_disr"])
        ):
            continue
        grouped.setdefault(row["seed"], []).append(float(row["reward_total"]))
    return {seed: float(np.mean(values)) for seed, values in grouped.items()}


def paired_stats(
    episode_rows: list[dict[str, str]],
    *,
    learned_phase: str,
    learned_policy: str,
    baseline_phase: str,
    baseline_policy: str,
    weight_combo: dict[str, float],
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> dict[str, Any] | None:
    learned_means = group_seed_means(
        episode_rows,
        phase=learned_phase,
        policy=learned_policy,
        weight_combo=weight_combo,
    )
    baseline_means = group_seed_means(
        episode_rows,
        phase=baseline_phase,
        policy=baseline_policy,
        weight_combo=weight_combo,
    )
    shared_seeds = sorted(set(learned_means) & set(baseline_means), key=int)
    if not shared_seeds:
        return None
    diffs = np.asarray(
        [
            float(learned_means[seed]) - float(baseline_means[seed])
            for seed in shared_seeds
        ],
        dtype=np.float64,
    )
    ci_low, ci_high = paired_bootstrap_ci(
        diffs,
        n_samples=bootstrap_samples,
        rng=rng,
    )
    return {
        "shared_seeds": shared_seeds,
        "mean_difference": float(diffs.mean()),
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "sign_flip_pvalue": float(exact_sign_flip_pvalue(diffs)),
    }


def analyze_run(
    run_dir: Path,
    *,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    config = summary["config"]
    scenario = str(config["risk_level"])
    label = run_label(config)
    policy_rows = load_csv_rows(run_dir / "policy_summary.csv")
    comparison_rows = load_csv_rows(run_dir / "comparison_table.csv")
    episode_rows = load_csv_rows(run_dir / "episode_metrics.csv")

    baseline_rows: list[dict[str, Any]] = []
    comparison_summary_rows: list[dict[str, Any]] = []

    learned_phase = f"{config['algo']}_eval"
    learned_policy = str(config["algo"])

    for comparison in comparison_rows:
        weight_combo = {
            "w_bo": float(comparison["w_bo"]),
            "w_cost": float(comparison["w_cost"]),
            "w_disr": float(comparison["w_disr"]),
        }
        policy_specs = [
            ("static_s1", "static_screen", "static_s1"),
            ("static_s2", "static_screen", "static_s2"),
            ("static_s3", "static_screen", "static_s3"),
            ("random", "random_eval", "random"),
            (
                "best_heuristic",
                "heuristic_eval",
                comparison.get("best_heuristic_policy") or "",
            ),
            ("learned", learned_phase, comparison["learned_policy"]),
        ]
        for role, phase, policy in policy_specs:
            if not policy:
                continue
            row = policy_lookup(
                policy_rows,
                phase=phase,
                policy=policy,
                weight_combo=weight_combo,
            )
            if row is None:
                continue
            baseline_rows.append(
                {
                    "run_label": label,
                    "scenario": scenario,
                    "algo": config["algo"],
                    "frame_stack": int(config["frame_stack"]),
                    "observation_version": config["observation_version"],
                    "w_bo": weight_combo["w_bo"],
                    "w_cost": weight_combo["w_cost"],
                    "w_disr": weight_combo["w_disr"],
                    "policy_role": role,
                    "policy_name": policy,
                    "reward_total_mean": float(row["reward_total_mean"]),
                    "fill_rate_mean": float(row["fill_rate_mean"]),
                    "backorder_rate_mean": float(row["backorder_rate_mean"]),
                    "ret_thesis_corrected_total_mean": float(
                        row["ret_thesis_corrected_total_mean"]
                    ),
                }
            )

        best_static_policy = comparison["best_static_policy"]
        best_static_phase = (
            "heuristic_eval"
            if best_static_policy.startswith("heuristic_")
            else "static_screen"
        )
        static_stats = paired_stats(
            episode_rows,
            learned_phase=learned_phase,
            learned_policy=learned_policy,
            baseline_phase=best_static_phase,
            baseline_policy=best_static_policy,
            weight_combo=weight_combo,
            bootstrap_samples=bootstrap_samples,
            rng=rng,
        )
        heuristic_stats = None
        best_heuristic_policy = comparison.get("best_heuristic_policy") or ""
        if best_heuristic_policy:
            heuristic_stats = paired_stats(
                episode_rows,
                learned_phase=learned_phase,
                learned_policy=learned_policy,
                baseline_phase="heuristic_eval",
                baseline_policy=best_heuristic_policy,
                weight_combo=weight_combo,
                bootstrap_samples=bootstrap_samples,
                rng=rng,
            )
        comparison_summary_rows.append(
            {
                "run_label": label,
                "scenario": scenario,
                "algo": config["algo"],
                "frame_stack": int(config["frame_stack"]),
                "observation_version": config["observation_version"],
                "w_bo": weight_combo["w_bo"],
                "w_cost": weight_combo["w_cost"],
                "w_disr": weight_combo["w_disr"],
                "learned_policy": learned_policy,
                "best_static_policy": best_static_policy,
                "best_heuristic_policy": best_heuristic_policy or None,
                "mean_diff_vs_best_static": (
                    static_stats["mean_difference"] if static_stats else None
                ),
                "ci95_low_vs_best_static": (
                    static_stats["ci95_low"] if static_stats else None
                ),
                "ci95_high_vs_best_static": (
                    static_stats["ci95_high"] if static_stats else None
                ),
                "sign_flip_pvalue_vs_best_static": (
                    static_stats["sign_flip_pvalue"] if static_stats else None
                ),
                "mean_diff_vs_best_heuristic": (
                    heuristic_stats["mean_difference"] if heuristic_stats else None
                ),
                "ci95_low_vs_best_heuristic": (
                    heuristic_stats["ci95_low"] if heuristic_stats else None
                ),
                "ci95_high_vs_best_heuristic": (
                    heuristic_stats["ci95_high"] if heuristic_stats else None
                ),
                "sign_flip_pvalue_vs_best_heuristic": (
                    heuristic_stats["sign_flip_pvalue"] if heuristic_stats else None
                ),
            }
        )

    run_payload = {
        "run_dir": str(run_dir),
        "run_label": label,
        "scenario": scenario,
        "config": config,
        "comparison_count": len(comparison_rows),
    }
    return baseline_rows, comparison_summary_rows, run_payload


def render_markdown(
    baseline_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# Publication Run Analysis",
        "",
        "This note summarizes the benchmark runs used for the paper-level comparison package.",
        "",
        "## Baseline Table",
        "",
        "| Run | Scenario | Role | Policy | Reward | Fill rate | Backorder rate | ReT corrected |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in baseline_rows:
        lines.append(
            f"| `{row['run_label']}` | `{row['scenario']}` | `{row['policy_role']}` | "
            f"`{row['policy_name']}` | {row['reward_total_mean']:.3f} | "
            f"{row['fill_rate_mean']:.3f} | {row['backorder_rate_mean']:.3f} | "
            f"{row['ret_thesis_corrected_total_mean']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Seed-Mean Comparisons",
            "",
            "| Run | Scenario | Learned | Best static | Diff vs static | CI95 | Best heuristic | Diff vs heuristic | CI95 |",
            "| --- | --- | --- | --- | ---: | --- | --- | ---: | --- |",
        ]
    )
    for row in comparison_rows:
        static_ci = (
            f"[{row['ci95_low_vs_best_static']:.3f}, {row['ci95_high_vs_best_static']:.3f}]"
            if row["ci95_low_vs_best_static"] is not None
            else "NA"
        )
        heuristic_ci = (
            f"[{row['ci95_low_vs_best_heuristic']:.3f}, {row['ci95_high_vs_best_heuristic']:.3f}]"
            if row["ci95_low_vs_best_heuristic"] is not None
            else "NA"
        )
        heuristic_diff = (
            f"{row['mean_diff_vs_best_heuristic']:.3f}"
            if row["mean_diff_vs_best_heuristic"] is not None
            else "NA"
        )
        lines.append(
            f"| `{row['run_label']}` | `{row['scenario']}` | `{row['learned_policy']}` | "
            f"`{row['best_static_policy']}` | {row['mean_diff_vs_best_static']:.3f} | {static_ci} | "
            f"`{row['best_heuristic_policy'] or 'NA'}` | {heuristic_diff} | {heuristic_ci} |"
        )
    lines.extend(
        [
            "",
            "## Usage",
            "",
            "- Use this package for baseline tables and cautious seed-mean comparison language.",
            "- Treat the intervals as support for direction and magnitude, not as a full significance section.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    baseline_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    run_payloads: list[dict[str, Any]] = []

    for run_dir in args.run_dirs:
        if not (run_dir / "summary.json").exists():
            continue
        run_baselines, run_comparisons, run_payload = analyze_run(
            run_dir,
            bootstrap_samples=args.bootstrap_samples,
            rng=rng,
        )
        baseline_rows.extend(run_baselines)
        comparison_rows.extend(run_comparisons)
        run_payloads.append(run_payload)

    save_csv(
        args.output_dir / "baseline_table.csv",
        baseline_rows,
        BASELINE_TABLE_FIELDS,
    )
    save_csv(
        args.output_dir / "algorithm_comparison.csv",
        comparison_rows,
        COMPARISON_FIELDS,
    )

    payload = {
        "bootstrap_samples": args.bootstrap_samples,
        "run_count": len(run_payloads),
        "runs": run_payloads,
        "baseline_table_rows": baseline_rows,
        "comparison_rows": comparison_rows,
    }
    (args.output_dir / "publication_run_analysis.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "publication_run_analysis.md").write_text(
        render_markdown(baseline_rows, comparison_rows),
        encoding="utf-8",
    )

    print(f"Wrote {args.output_dir / 'baseline_table.csv'}")
    print(f"Wrote {args.output_dir / 'algorithm_comparison.csv'}")
    print(f"Wrote {args.output_dir / 'publication_run_analysis.json'}")
    print(f"Wrote {args.output_dir / 'publication_run_analysis.md'}")


if __name__ == "__main__":
    main()
