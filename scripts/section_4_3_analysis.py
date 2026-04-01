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
    paired_cohens_d,
)

BASELINE_TABLE_FIELDS = [
    "run_label",
    "scenario",
    "algo",
    "frame_stack",
    "observation_version",
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
    "learned_policy",
    "best_static_policy",
    "best_heuristic_policy",
    "mean_diff_vs_best_static",
    "ci95_low_vs_best_static",
    "ci95_high_vs_best_static",
    "sign_flip_pvalue_vs_best_static",
    "cohens_d_vs_best_static",
    "mean_diff_vs_best_heuristic",
    "ci95_low_vs_best_heuristic",
    "ci95_high_vs_best_heuristic",
    "sign_flip_pvalue_vs_best_heuristic",
    "cohens_d_vs_best_heuristic",
]

SCENARIO_PHASES = {
    "increased": {
        "static": "static_screen",
        "heuristic": "heuristic_eval",
        "random": "random_eval",
    },
    "severe": {
        "static": "cross_eval_severe",
        "heuristic": "cross_eval_severe",
        "random": "cross_eval_severe",
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build Section 4.3 algorithm comparison tables."
    )
    parser.add_argument("--run-dirs", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--baseline-run-dir",
        type=Path,
        default=Path("outputs/benchmarks/control_reward"),
        help="Reference benchmark run containing the frozen static/heuristic/random baselines.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/artifacts/control_reward/section4_3_algorithm_comparison"),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260321)
    return parser


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        if rows:
            writer.writerows(rows)


def run_label(config: dict[str, Any]) -> str:
    return f"{config['algo']}_fs{int(config['frame_stack'])}_{config['observation_version']}"


def learned_phase_and_policy(config: dict[str, Any], scenario: str) -> tuple[str, str]:
    algo = str(config["algo"])
    policy = algo
    if scenario == "increased":
        return f"{algo}_eval", policy
    return "cross_eval_severe", policy


def policy_rows_for_phase(
    rows: list[dict[str, str]], phase: str, policy: str
) -> list[dict[str, str]]:
    return [r for r in rows if r["phase"] == phase and r["policy"] == policy]


def mean_row(rows: list[dict[str, str]]) -> dict[str, float] | None:
    if not rows:
        return None
    r = rows[0]
    return {
        "reward_total_mean": float(r["reward_total_mean"]),
        "fill_rate_mean": float(r["fill_rate_mean"]),
        "backorder_rate_mean": float(r["backorder_rate_mean"]),
        "ret_thesis_corrected_total_mean": float(r["ret_thesis_corrected_total_mean"]),
    }


def group_seed_means(
    rows: list[dict[str, str]], phase: str, policy: str
) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        if row["phase"] != phase or row["policy"] != policy:
            continue
        grouped.setdefault(row["seed"], []).append(float(row["reward_total"]))
    return {seed: float(np.mean(vals)) for seed, vals in grouped.items()}


def paired_stats(
    learned_episode_rows: list[dict[str, str]],
    baseline_episode_rows: list[dict[str, str]],
    learned_phase: str,
    learned_policy: str,
    baseline_phase: str,
    baseline_policy: str,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> dict[str, Any] | None:
    learned = group_seed_means(learned_episode_rows, learned_phase, learned_policy)
    baseline = group_seed_means(baseline_episode_rows, baseline_phase, baseline_policy)
    shared = sorted(set(learned) & set(baseline), key=int)
    if not shared:
        return None
    diffs = np.asarray([learned[s] - baseline[s] for s in shared], dtype=np.float64)
    lo, hi = paired_bootstrap_ci(diffs, n_samples=bootstrap_samples, rng=rng)
    return {
        "mean_difference": float(diffs.mean()),
        "ci95_low": lo,
        "ci95_high": hi,
        "sign_flip_pvalue": float(exact_sign_flip_pvalue(diffs)),
        "cohens_d": float(paired_cohens_d(diffs)),
    }


def analyze_run(
    run_dir: Path,
    baseline_policy_rows: list[dict[str, str]],
    baseline_episode_rows: list[dict[str, str]],
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    config = summary["config"]
    label = run_label(config)
    policy_rows = load_csv_rows(run_dir / "policy_summary.csv")
    episode_rows = load_csv_rows(run_dir / "episode_metrics.csv")

    baseline_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []

    for scenario, phases in SCENARIO_PHASES.items():
        learned_phase, learned_policy = learned_phase_and_policy(config, scenario)

        static_candidates = []
        for policy in ("static_s1", "static_s2", "static_s3"):
            row = mean_row(
                policy_rows_for_phase(baseline_policy_rows, phases["static"], policy)
            )
            if row is not None:
                static_candidates.append((policy, row))
                baseline_rows.append(
                    {
                        "run_label": label,
                        "scenario": scenario,
                        "algo": config["algo"],
                        "frame_stack": int(config["frame_stack"]),
                        "observation_version": config["observation_version"],
                        "policy_role": "static",
                        "policy_name": policy,
                        **row,
                    }
                )

        heuristic_candidates = []
        for policy in (
            "heuristic_hysteresis",
            "heuristic_disruption",
            "heuristic_tuned",
            "heuristic_cycle_guard",
        ):
            row = mean_row(
                policy_rows_for_phase(baseline_policy_rows, phases["heuristic"], policy)
            )
            if row is not None:
                heuristic_candidates.append((policy, row))
                baseline_rows.append(
                    {
                        "run_label": label,
                        "scenario": scenario,
                        "algo": config["algo"],
                        "frame_stack": int(config["frame_stack"]),
                        "observation_version": config["observation_version"],
                        "policy_role": "heuristic",
                        "policy_name": policy,
                        **row,
                    }
                )

        random_row = mean_row(
            policy_rows_for_phase(baseline_policy_rows, phases["random"], "random")
        )
        if random_row is not None:
            baseline_rows.append(
                {
                    "run_label": label,
                    "scenario": scenario,
                    "algo": config["algo"],
                    "frame_stack": int(config["frame_stack"]),
                    "observation_version": config["observation_version"],
                    "policy_role": "random",
                    "policy_name": "random",
                    **random_row,
                }
            )

        learned_row = mean_row(
            policy_rows_for_phase(policy_rows, learned_phase, learned_policy)
        )
        if learned_row is not None:
            baseline_rows.append(
                {
                    "run_label": label,
                    "scenario": scenario,
                    "algo": config["algo"],
                    "frame_stack": int(config["frame_stack"]),
                    "observation_version": config["observation_version"],
                    "policy_role": "learned",
                    "policy_name": learned_policy,
                    **learned_row,
                }
            )

        if learned_row is None or not static_candidates:
            continue

        best_static_policy, _ = max(
            static_candidates, key=lambda item: item[1]["reward_total_mean"]
        )
        best_heuristic_policy = None
        if heuristic_candidates:
            best_heuristic_policy, _ = max(
                heuristic_candidates, key=lambda item: item[1]["reward_total_mean"]
            )

        static_stats = paired_stats(
            episode_rows,
            baseline_episode_rows,
            learned_phase,
            learned_policy,
            phases["static"],
            best_static_policy,
            bootstrap_samples,
            rng,
        )
        heuristic_stats = None
        if best_heuristic_policy is not None:
            heuristic_stats = paired_stats(
                episode_rows,
                baseline_episode_rows,
                learned_phase,
                learned_policy,
                phases["heuristic"],
                best_heuristic_policy,
                bootstrap_samples,
                rng,
            )

        comparison_rows.append(
            {
                "run_label": label,
                "scenario": scenario,
                "algo": config["algo"],
                "frame_stack": int(config["frame_stack"]),
                "observation_version": config["observation_version"],
                "learned_policy": learned_policy,
                "best_static_policy": best_static_policy,
                "best_heuristic_policy": best_heuristic_policy,
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
                "cohens_d_vs_best_static": (
                    static_stats["cohens_d"] if static_stats else None
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
                "cohens_d_vs_best_heuristic": (
                    heuristic_stats["cohens_d"] if heuristic_stats else None
                ),
            }
        )

    return (
        baseline_rows,
        comparison_rows,
        {"run_dir": str(run_dir), "config": config, "run_label": label},
    )


def render_markdown(
    baseline_rows: list[dict[str, Any]], comparison_rows: list[dict[str, Any]]
) -> str:
    lines = [
        "# Section 4.3 Algorithm Comparison",
        "",
        "This package summarizes the matched-budget algorithm comparison for the paper-facing benchmark family.",
        "",
        "## Baseline Table",
        "",
        "| Run | Scenario | Role | Policy | Reward | Fill rate | Backorder rate | ReT corrected |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in baseline_rows:
        lines.append(
            f"| `{row['run_label']}` | `{row['scenario']}` | `{row['policy_role']}` | `{row['policy_name']}` | {row['reward_total_mean']:.3f} | {row['fill_rate_mean']:.3f} | {row['backorder_rate_mean']:.3f} | {row['ret_thesis_corrected_total_mean']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Seed-Mean Comparisons",
            "",
            "| Run | Scenario | Learned | Best static | Diff vs static | CI95 | d | Best heuristic | Diff vs heuristic | CI95 | d |",
            "| --- | --- | --- | --- | ---: | --- | ---: | --- | ---: | --- | ---: |",
        ]
    )
    for row in comparison_rows:
        static_ci = (
            f"[{row['ci95_low_vs_best_static']:.3f}, {row['ci95_high_vs_best_static']:.3f}]"
            if row["ci95_low_vs_best_static"] is not None
            else "NA"
        )
        heur_ci = (
            f"[{row['ci95_low_vs_best_heuristic']:.3f}, {row['ci95_high_vs_best_heuristic']:.3f}]"
            if row["ci95_low_vs_best_heuristic"] is not None
            else "NA"
        )
        heur_diff = (
            f"{row['mean_diff_vs_best_heuristic']:.3f}"
            if row["mean_diff_vs_best_heuristic"] is not None
            else "NA"
        )
        static_d = (
            f"{row['cohens_d_vs_best_static']:.2f}"
            if row["cohens_d_vs_best_static"] is not None
            else "NA"
        )
        heur_d = (
            f"{row['cohens_d_vs_best_heuristic']:.2f}"
            if row["cohens_d_vs_best_heuristic"] is not None
            else "NA"
        )
        lines.append(
            f"| `{row['run_label']}` | `{row['scenario']}` | `{row['learned_policy']}` | `{row['best_static_policy']}` | {row['mean_diff_vs_best_static']:.3f} | {static_ci} | {static_d} | `{row['best_heuristic_policy'] or 'NA'}` | {heur_diff} | {heur_ci} | {heur_d} |"
        )
    return "\\n".join(lines) + "\\n"


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    baseline_policy_rows = load_csv_rows(args.baseline_run_dir / "policy_summary.csv")
    baseline_episode_rows = load_csv_rows(args.baseline_run_dir / "episode_metrics.csv")
    baseline_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    for run_dir in args.run_dirs:
        if not (run_dir / "summary.json").exists():
            continue
        b_rows, c_rows, payload = analyze_run(
            run_dir,
            baseline_policy_rows,
            baseline_episode_rows,
            args.bootstrap_samples,
            rng,
        )
        baseline_rows.extend(b_rows)
        comparison_rows.extend(c_rows)
        runs.append(payload)
    save_csv(
        args.output_dir / "baseline_table.csv", baseline_rows, BASELINE_TABLE_FIELDS
    )
    save_csv(
        args.output_dir / "algorithm_comparison.csv", comparison_rows, COMPARISON_FIELDS
    )
    payload = {
        "run_count": len(runs),
        "runs": runs,
        "baseline_table_rows": baseline_rows,
        "comparison_rows": comparison_rows,
    }
    (args.output_dir / "section_4_3_analysis.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    (args.output_dir / "section_4_3_analysis.md").write_text(
        render_markdown(baseline_rows, comparison_rows), encoding="utf-8"
    )
    print(f"Wrote {args.output_dir / 'baseline_table.csv'}")
    print(f"Wrote {args.output_dir / 'algorithm_comparison.csv'}")
    print(f"Wrote {args.output_dir / 'section_4_3_analysis.json'}")
    print(f"Wrote {args.output_dir / 'section_4_3_analysis.md'}")


if __name__ == "__main__":
    main()
