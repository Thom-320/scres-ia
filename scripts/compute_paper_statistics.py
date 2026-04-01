#!/usr/bin/env python3
"""Compute paper-facing paired statistics for the frozen manuscript claims.

This script uses paired seed-level comparisons for Track B because PPO and the
static baselines are evaluated on shared seed sets. It writes:

- `formal_statistics.json`
- `formal_statistics.csv`
- `formal_statistics_table.md`

The main test outputs are:

- exact Wilcoxon signed-rank p-value (two-sided, when available)
- exact sign-flip p-value
- paired Cohen's d
- paired bootstrap CI95 for PPO - comparator

Usage:
    python scripts/compute_paper_statistics.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.control_reward_seed_inference import (
    exact_sign_flip_pvalue,
    paired_bootstrap_ci,
    paired_cohens_d,
)

TRACK_B_SUMMARY = (
    ROOT / "outputs" / "track_b_benchmarks" / "track_b_ret_seq_k020_500k_rerun1" / "summary.json"
)
ABLATION_JSON = ROOT / "outputs" / "track_b_ablation_5d_vs_7d.json"
OUT_DIR = ROOT / "outputs" / "paper_statistics"
TRACKED_MD_PATH = (
    ROOT / "docs" / "manuscript_current" / "submission" / "formal_statistics_table.md"
)
PAIRWISE_COMPARATORS = ("s2_d1.00", "s3_d2.00")
PAIRWISE_METRICS = (
    ("fill_rate", "higher"),
    ("reward_total", "higher"),
    ("order_level_ret_mean", "higher"),
)


def load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def grouped_seed_metrics(summary: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in summary["seed_metrics"]:
        grouped.setdefault(row["policy"], []).append(row)
    return grouped


def seed_metric_map(rows: list[dict[str, Any]], metric: str) -> dict[int, float]:
    field = f"{metric}_mean"
    return {int(row["seed"]): float(row[field]) for row in rows}


def exact_wilcoxon_pvalue(diffs: np.ndarray) -> float | None:
    nonzero = diffs[np.abs(diffs) > 0.0]
    if len(nonzero) == 0:
        return None
    result = wilcoxon(
        nonzero,
        alternative="two-sided",
        zero_method="wilcox",
        method="exact",
    )
    return float(result.pvalue)


def pairwise_row(
    grouped: dict[str, list[dict[str, Any]]],
    *,
    comparator: str,
    metric: str,
    goal: str,
) -> dict[str, Any]:
    ppo_values = seed_metric_map(grouped["ppo"], metric)
    cmp_values = seed_metric_map(grouped[comparator], metric)
    shared_seeds = sorted(set(ppo_values) & set(cmp_values))
    ppo_arr = np.asarray([ppo_values[s] for s in shared_seeds], dtype=np.float64)
    cmp_arr = np.asarray([cmp_values[s] for s in shared_seeds], dtype=np.float64)
    diffs = ppo_arr - cmp_arr
    better_diffs = diffs if goal == "higher" else -diffs
    ci_low, ci_high = paired_bootstrap_ci(diffs, n_samples=10_000, rng=np.random.default_rng(20260401))
    wins = int(np.sum(better_diffs > 0.0))
    losses = int(np.sum(better_diffs < 0.0))
    ties = int(np.sum(np.isclose(better_diffs, 0.0)))

    return {
        "comparison": f"Track B 500k: PPO vs {comparator}",
        "comparator_policy": comparator,
        "metric": metric,
        "goal": goal,
        "seed_count": len(shared_seeds),
        "shared_seeds": ",".join(str(seed) for seed in shared_seeds),
        "ppo_mean": float(np.mean(ppo_arr)),
        "baseline_mean": float(np.mean(cmp_arr)),
        "ppo_minus_baseline": float(np.mean(diffs)),
        "improvement_toward_goal": float(np.mean(better_diffs)),
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "wilcoxon_pvalue": exact_wilcoxon_pvalue(diffs),
        "sign_flip_pvalue": float(exact_sign_flip_pvalue(diffs)),
        "cohens_d": float(paired_cohens_d(diffs)),
        "ppo_wins": wins,
        "baseline_wins": losses,
        "ties": ties,
        "note": (
            "Paired seed-level comparison. With n=5, exact two-sided p-values remain coarse "
            "even when all shared seeds favor PPO."
        ),
    }


def ablation_row(ablation: dict[str, Any]) -> dict[str, Any]:
    d5 = ablation["5d"]
    d7 = ablation["7d"]
    return {
        "comparison": "Ablation: 7D vs 5D (matched, 100k x 3 seeds)",
        "metric": "fill_rate",
        "goal": "higher",
        "condition_7d": float(d7["ppo_f"]),
        "condition_5d": float(d5["ppo_f"]),
        "difference_pp": round((float(d7["ppo_f"]) - float(d5["ppo_f"])) * 100, 1),
        "reward_7d": float(d7["ppo_r"]),
        "reward_5d": float(d5["ppo_r"]),
        "s2_fill": float(d5["s2_f"]),
        "note": "Aggregate-only row. Per-seed paired tests require the underlying seed-level ablation export.",
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def render_markdown(rows: list[dict[str, Any]]) -> str:
    pairwise = [row for row in rows if "ppo_mean" in row]
    ablation = [row for row in rows if "condition_7d" in row]

    lines = [
        "# Formal Statistics",
        "",
        "## Track B Pairwise Seed-Level Comparisons",
        "",
        "| Comparator | Metric | PPO mean | Baseline mean | PPO - baseline | CI95 | Wilcoxon p | Sign-flip p | Cohen d | Wins |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in pairwise:
        ci = f"[{row['ci95_low']:+.5f}, {row['ci95_high']:+.5f}]"
        wilcoxon_p = (
            f"{row['wilcoxon_pvalue']:.4f}" if row["wilcoxon_pvalue"] is not None else "NA"
        )
        lines.append(
            "| "
            f"{row['comparator_policy']} | {row['metric']} | "
            f"{row['ppo_mean']:.5f} | {row['baseline_mean']:.5f} | "
            f"{row['ppo_minus_baseline']:+.5f} | {ci} | "
            f"{wilcoxon_p} | {row['sign_flip_pvalue']:.4f} | "
            f"{row['cohens_d']:+.2f} | {row['ppo_wins']}/{row['seed_count']} |"
        )

    if ablation:
        row = ablation[0]
        lines.extend(
            [
                "",
                "## Matched Ablation",
                "",
                "| Comparison | 7D fill | 5D fill | Difference | 7D reward | 5D reward | Note |",
                "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
                "| "
                f"7D vs 5D | {row['condition_7d']:.5f} | {row['condition_5d']:.5f} | "
                f"+{row['difference_pp']:.1f} pp | {row['reward_7d']:.2f} | {row['reward_5d']:.2f} | "
                f"{row['note']} |",
            ]
        )

    lines.extend(
        [
            "",
            "## Interpretation Note",
            "",
            "The paired seed-level tests are the defensible inferential unit for the frozen Track B bundle because PPO and the static comparators share the same evaluation seeds. With only five shared seeds, the exact two-sided p-values remain coarse even when PPO wins on every seed; this is why the table reports direction, effect size, and wins alongside p-values rather than overclaiming significance.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    summary = load_summary(TRACK_B_SUMMARY)
    grouped = grouped_seed_metrics(summary)
    for comparator in PAIRWISE_COMPARATORS:
        for metric, goal in PAIRWISE_METRICS:
            rows.append(pairwise_row(grouped, comparator=comparator, metric=metric, goal=goal))

    if ABLATION_JSON.exists():
        rows.append(ablation_row(load_summary(ABLATION_JSON)))

    json_path = OUT_DIR / "formal_statistics.json"
    csv_path = OUT_DIR / "formal_statistics.csv"
    md_path = OUT_DIR / "formal_statistics_table.md"
    tracked_md_path = TRACKED_MD_PATH
    tracked_md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    write_csv(csv_path, rows)
    markdown = render_markdown(rows)
    md_path.write_text(markdown, encoding="utf-8")
    tracked_md_path.write_text(markdown, encoding="utf-8")

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {tracked_md_path}")


if __name__ == "__main__":
    main()
