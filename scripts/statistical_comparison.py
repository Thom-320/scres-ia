#!/usr/bin/env python3
"""
Compute formal statistical tests for PPO vs baselines from benchmark outputs.

Reads episode_metrics.csv from a benchmark run and produces:
  - Mann-Whitney U test (non-parametric, unpaired)
  - Welch's t-test (parametric, unpaired)
  - Cohen's d effect size
  - Bootstrap CI95 for the difference
  - Summary table in CSV and markdown

Usage:
    python scripts/statistical_comparison.py --input-dir outputs/benchmarks/final_ret_seq_v1_500k
    python scripts/statistical_comparison.py --input-dir outputs/benchmarks/my_run --metric fill_rate
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d for independent samples (pooled SD)."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    pooled_std = np.sqrt(
        ((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1))
        / (na + nb - 2)
    )
    if pooled_std < 1e-12:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled_std)


def bootstrap_ci(
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap CI for mean(a) - mean(b)."""
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        a_sample = rng.choice(a, size=len(a), replace=True)
        b_sample = rng.choice(b, size=len(b), replace=True)
        diffs[i] = np.mean(a_sample) - np.mean(b_sample)
    alpha = (1 - ci) / 2
    return float(np.percentile(diffs, 100 * alpha)), float(
        np.percentile(diffs, 100 * (1 - alpha))
    )


def compare_two(
    ppo_values: np.ndarray,
    baseline_values: np.ndarray,
    baseline_name: str,
    metric_name: str,
) -> dict:
    """Run all statistical tests comparing PPO vs one baseline."""
    ppo_mean = float(np.mean(ppo_values))
    base_mean = float(np.mean(baseline_values))
    diff = ppo_mean - base_mean

    # Mann-Whitney U (non-parametric)
    if len(ppo_values) >= 3 and len(baseline_values) >= 3:
        u_stat, mw_p = scipy_stats.mannwhitneyu(
            ppo_values, baseline_values, alternative="two-sided"
        )
    else:
        u_stat, mw_p = float("nan"), float("nan")

    # Welch's t-test (parametric, unequal variance)
    if len(ppo_values) >= 2 and len(baseline_values) >= 2:
        t_stat, welch_p = scipy_stats.ttest_ind(
            ppo_values, baseline_values, equal_var=False
        )
    else:
        t_stat, welch_p = float("nan"), float("nan")

    # Cohen's d
    d = cohens_d(ppo_values, baseline_values)

    # Bootstrap CI95
    ci_lo, ci_hi = bootstrap_ci(ppo_values, baseline_values)

    return {
        "baseline": baseline_name,
        "metric": metric_name,
        "ppo_mean": round(ppo_mean, 4),
        "ppo_std": round(float(np.std(ppo_values, ddof=1)), 4),
        "ppo_n": len(ppo_values),
        "baseline_mean": round(base_mean, 4),
        "baseline_std": round(float(np.std(baseline_values, ddof=1)), 4),
        "baseline_n": len(baseline_values),
        "diff": round(diff, 4),
        "mann_whitney_U": round(float(u_stat), 2),
        "mann_whitney_p": round(float(mw_p), 4),
        "welch_t": round(float(t_stat), 4),
        "welch_p": round(float(welch_p), 4),
        "cohens_d": round(d, 4),
        "bootstrap_ci95_lo": round(ci_lo, 4),
        "bootstrap_ci95_hi": round(ci_hi, 4),
        "ppo_wins": diff > 0,
        "significant_005": float(mw_p) < 0.05 if not np.isnan(mw_p) else False,
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_episode_metrics(input_dir: Path) -> list[dict]:
    """Load episode_metrics.csv from a benchmark run."""
    csv_path = input_dir / "episode_metrics.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No episode_metrics.csv in {input_dir}")
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def extract_seed_means(
    rows: list[dict], phase: str, policy: str, metric: str
) -> np.ndarray:
    """Aggregate metric by seed (mean over episodes per seed)."""
    by_seed: dict[int, list[float]] = {}
    for r in rows:
        if r.get("phase") == phase and r.get("policy") == policy:
            seed = int(r["seed"])
            val = float(r.get(metric, 0))
            by_seed.setdefault(seed, []).append(val)
    return np.array([np.mean(v) for v in by_seed.values()])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

BASELINE_PHASES = {
    "static_s1": "static_screen",
    "static_s2": "static_screen",
    "static_s3": "static_screen",
    "garrido_cf_s1": "static_screen",
    "garrido_cf_s2": "static_screen",
    "garrido_cf_s3": "static_screen",
    "heuristic_hysteresis": "heuristic_eval",
    "heuristic_disruption": "heuristic_eval",
    "heuristic_tuned": "heuristic_eval",
    "heuristic_cycle_guard": "heuristic_eval",
    "random": "random_eval",
}

METRICS = ["reward_total", "fill_rate", "backorder_rate"]


def main():
    parser = argparse.ArgumentParser(
        description="Statistical comparison of PPO vs baselines"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Benchmark output directory containing episode_metrics.csv",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=METRICS,
        help="Metrics to compare (column names in episode_metrics.csv)",
    )
    parser.add_argument(
        "--ppo-phase",
        default="ppo_eval",
        help="Phase name for the learned policy",
    )
    parser.add_argument(
        "--ppo-policy",
        default="ppo",
        help="Policy name for the learned agent",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (defaults to input-dir/statistical_tests/)",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.input_dir / "statistical_tests"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_episode_metrics(args.input_dir)

    # Find which baselines actually exist in this run
    available_policies = {r["policy"] for r in rows}
    available_baselines = {
        p: phase
        for p, phase in BASELINE_PHASES.items()
        if p in available_policies
    }

    all_results = []

    for metric in args.metrics:
        ppo_values = extract_seed_means(
            rows, args.ppo_phase, args.ppo_policy, metric
        )
        if len(ppo_values) == 0:
            print(f"WARNING: No PPO data for metric {metric}")
            continue

        for baseline_name, baseline_phase in available_baselines.items():
            base_values = extract_seed_means(
                rows, baseline_phase, baseline_name, metric
            )
            if len(base_values) == 0:
                continue

            result = compare_two(ppo_values, base_values, baseline_name, metric)
            all_results.append(result)

            sig = "*" if result["significant_005"] else ""
            win = "+" if result["ppo_wins"] else "-"
            print(
                f"{metric:20s} vs {baseline_name:25s}: "
                f"diff={result['diff']:+.4f} p={result['mann_whitney_p']:.4f}{sig} "
                f"d={result['cohens_d']:+.4f} "
                f"CI95=[{result['bootstrap_ci95_lo']:+.4f}, {result['bootstrap_ci95_hi']:+.4f}] {win}"
            )

    # Save CSV
    if all_results:
        fieldnames = list(all_results[0].keys())
        csv_path = args.output_dir / "statistical_tests.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)

        # Save markdown summary
        md_path = args.output_dir / "statistical_tests.md"
        with open(md_path, "w") as f:
            f.write("# Statistical Comparison: PPO vs Baselines\n\n")
            f.write(f"Source: `{args.input_dir}`\n\n")

            for metric in args.metrics:
                metric_results = [r for r in all_results if r["metric"] == metric]
                if not metric_results:
                    continue
                f.write(f"## {metric}\n\n")
                f.write(
                    "| Baseline | PPO mean | Base mean | Diff | Mann-Whitney p | Cohen's d | CI95 | Sig |\n"
                )
                f.write(
                    "|----------|----------|-----------|------|----------------|-----------|------|-----|\n"
                )
                for r in sorted(metric_results, key=lambda x: -x["diff"]):
                    sig = "\\*" if r["significant_005"] else ""
                    f.write(
                        f"| {r['baseline']} | {r['ppo_mean']:.3f} | {r['baseline_mean']:.3f} "
                        f"| {r['diff']:+.3f} | {r['mann_whitney_p']:.4f} | {r['cohens_d']:+.2f} "
                        f"| [{r['bootstrap_ci95_lo']:+.3f}, {r['bootstrap_ci95_hi']:+.3f}] | {sig} |\n"
                    )
                f.write("\n")

        # Save JSON
        with open(args.output_dir / "statistical_tests.json", "w") as f:
            json.dump(all_results, f, indent=2)

        print(f"\nSaved to {csv_path}")
        print(f"         {md_path}")
    else:
        print("No results to save.")


if __name__ == "__main__":
    main()
