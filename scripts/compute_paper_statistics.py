#!/usr/bin/env python3
"""Compute formal statistical tests for the paper.

Reads seed-level metrics from Track B 500k and the ablation,
then computes Mann-Whitney U / Wilcoxon rank-sum tests and
rank-biserial effect sizes for the main comparisons.

Usage:
    python scripts/compute_paper_statistics.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# ── Helpers ──────────────────────────────────────────────────


def load_seed_metrics(summary_path: Path) -> dict[str, list[dict]]:
    """Group seed_metrics by policy."""
    with open(summary_path) as f:
        data = json.load(f)
    grouped: dict[str, list[dict]] = {}
    for row in data["seed_metrics"]:
        pol = row["policy"]
        grouped.setdefault(pol, []).append(row)
    return grouped


def rank_biserial(u: float, n1: int, n2: int) -> float:
    """Rank-biserial correlation r = 1 - 2U/(n1*n2)."""
    return 1.0 - (2.0 * u) / (n1 * n2)


def mann_whitney(x: np.ndarray, y: np.ndarray) -> dict:
    """Mann-Whitney U test with rank-biserial effect size."""
    from scipy.stats import mannwhitneyu

    stat, p = mannwhitneyu(x, y, alternative="two-sided")
    r = rank_biserial(stat, len(x), len(y))
    return {"U": float(stat), "p": float(p), "r_rb": float(r), "n1": len(x), "n2": len(y)}


def bootstrap_ci(
    x: np.ndarray, stat_fn=np.mean, n_boot: int = 10_000, alpha: float = 0.05
) -> tuple[float, float]:
    """Bootstrap confidence interval."""
    rng = np.random.default_rng(42)
    boot = np.array([stat_fn(rng.choice(x, size=len(x), replace=True)) for _ in range(n_boot)])
    lo = float(np.percentile(boot, 100 * alpha / 2))
    hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    return lo, hi


# ── Main ─────────────────────────────────────────────────────


def main() -> None:
    results: list[dict] = []

    # ── Track B 500k: PPO vs statics ────────────────────────
    tb_path = ROOT / "outputs" / "track_b_benchmarks" / "track_b_ret_seq_k020_500k_rerun1" / "summary.json"
    if tb_path.exists():
        grouped = load_seed_metrics(tb_path)
        ppo_fill = np.array([r["fill_rate_mean"] for r in grouped["ppo"]])
        ppo_reward = np.array([r["reward_total_mean"] for r in grouped["ppo"]])
        ppo_ret = np.array([r["order_level_ret_mean_mean"] for r in grouped["ppo"]])

        for baseline_name in ["s2_d1.00", "s3_d2.00"]:
            if baseline_name not in grouped:
                continue
            bl_fill = np.array([r["fill_rate_mean"] for r in grouped[baseline_name]])
            bl_reward = np.array([r["reward_total_mean"] for r in grouped[baseline_name]])
            bl_ret = np.array([r["order_level_ret_mean_mean"] for r in grouped[baseline_name]])

            for metric_name, ppo_vals, bl_vals in [
                ("fill_rate", ppo_fill, bl_fill),
                ("reward_total", ppo_reward, bl_reward),
                ("order_level_ret_mean", ppo_ret, bl_ret),
            ]:
                mw = mann_whitney(ppo_vals, bl_vals)
                ppo_ci = bootstrap_ci(ppo_vals)
                bl_ci = bootstrap_ci(bl_vals)
                results.append(
                    {
                        "comparison": f"Track B 500k: PPO vs {baseline_name}",
                        "metric": metric_name,
                        "ppo_mean": float(np.mean(ppo_vals)),
                        "ppo_ci95": list(ppo_ci),
                        "baseline_mean": float(np.mean(bl_vals)),
                        "baseline_ci95": list(bl_ci),
                        "mann_whitney_U": mw["U"],
                        "p_value": mw["p"],
                        "rank_biserial_r": mw["r_rb"],
                        "n_ppo": mw["n1"],
                        "n_baseline": mw["n2"],
                    }
                )
        print(f"Track B 500k: computed {len(results)} comparisons")
    else:
        print(f"WARNING: {tb_path} not found, skipping Track B 500k")

    # ── Ablation 5D vs 7D ───────────────────────────────────
    abl_path = ROOT / "outputs" / "track_b_ablation_5d_vs_7d.json"
    if abl_path.exists():
        with open(abl_path) as f:
            abl = json.load(f)
        # The ablation file has aggregate means, not per-seed.
        # Report it as-is with a note that formal per-seed stats
        # require the underlying seed-level data.
        d5 = abl.get("5d", {})
        d7 = abl.get("7d", {})
        fill_5d = float(d5.get("ppo_f", 0))
        fill_7d = float(d7.get("ppo_f", 0))
        rew_5d = float(d5.get("ppo_r", 0))
        rew_7d = float(d7.get("ppo_r", 0))
        results.append(
            {
                "comparison": "Ablation: 7D vs 5D (matched, 100k x 3 seeds)",
                "metric": "fill_rate",
                "condition_7d": fill_7d,
                "condition_5d": fill_5d,
                "difference_pp": round((fill_7d - fill_5d) * 100, 1),
                "reward_7d": rew_7d,
                "reward_5d": rew_5d,
                "s2_fill": float(d5.get("s2_f", 0)),
                "note": "Aggregate means from 3 seeds; per-seed Wilcoxon requires seed-level export",
            }
        )
        print("Ablation: reported aggregate comparison")
    else:
        print(f"WARNING: {abl_path} not found, skipping ablation")

    # ── Output ──────────────────────────────────────────────
    out_dir = ROOT / "outputs" / "paper_statistics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "formal_statistics.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {out_path}")

    # ── Console summary ─────────────────────────────────────
    print("\n" + "=" * 72)
    print("PAPER STATISTICS SUMMARY")
    print("=" * 72)
    for r in results:
        print(f"\n  {r['comparison']} — {r['metric']}")
        if "ppo_mean" in r:
            print(f"    PPO:      {r['ppo_mean']:.6f}  CI95: [{r['ppo_ci95'][0]:.6f}, {r['ppo_ci95'][1]:.6f}]")
            print(f"    Baseline: {r['baseline_mean']:.6f}  CI95: [{r['baseline_ci95'][0]:.6f}, {r['baseline_ci95'][1]:.6f}]")
            print(f"    Mann-Whitney U={r['mann_whitney_U']:.1f}, p={r['p_value']:.6f}, r_rb={r['rank_biserial_r']:.4f}")
        elif "condition_7d" in r:
            print(f"    7D: {r['condition_7d']:.6f}  |  5D: {r['condition_5d']:.6f}  |  diff: {r.get('difference_pp', 0):.1f} pp")
            print(f"    Note: {r['note']}")
    print()


if __name__ == "__main__":
    main()
