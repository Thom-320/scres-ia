#!/usr/bin/env python3
"""Analyze PPO action distributions from Track B benchmark results.

Reads policy_summary.csv files from reward sweep outputs and produces:
1. Shift distribution comparison (PPO vs statics vs heuristics)
2. Downstream multiplier statistics
3. Cost-efficiency comparison table
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys


def load_policy_summaries(sweep_dir: Path) -> list[dict[str, str]]:
    """Load and merge policy_summary.csv from all run subdirectories."""
    rows: list[dict[str, str]] = []
    for summary_csv in sorted(sweep_dir.rglob("policy_summary.csv")):
        run_name = summary_csv.parent.name
        with open(summary_csv, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                row["run_name"] = run_name
                rows.append(row)
    return rows


def fmt(val: str, decimals: int = 4) -> str:
    try:
        return f"{float(val):.{decimals}f}"
    except (ValueError, TypeError):
        return val


def print_action_analysis(rows: list[dict[str, str]]) -> None:
    # Group by policy across runs (first occurrence wins for statics)
    by_policy: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_policy.setdefault(row["policy"], []).append(row)

    print("=" * 100)
    print("TRACK B ACTION ANALYSIS")
    print("=" * 100)

    # Table 1: Shift distribution + fill rate + cost
    print("\n## Shift Distribution & Cost Efficiency\n")
    header = (
        f"{'Policy':<28} {'Fill':>8} {'BO':>8} {'ReT':>8} "
        f"{'S1%':>6} {'S2%':>6} {'S3%':>6} "
        f"{'Asm hrs':>9} {'Cost idx':>9}"
    )
    print(header)
    print("-" * len(header))

    for policy, policy_rows in sorted(by_policy.items()):
        row = policy_rows[0]  # Use first run's summary
        s1 = float(row.get("pct_steps_S1_mean", 0))
        s2 = float(row.get("pct_steps_S2_mean", 0))
        s3 = float(row.get("pct_steps_S3_mean", 0))
        asm = float(row.get("assembly_hours_total_mean", 0))
        cost_idx = float(row.get("assembly_cost_index_mean", 0))
        print(
            f"{policy:<28} "
            f"{fmt(row.get('fill_rate_mean', ''), 6):>8} "
            f"{fmt(row.get('backorder_rate_mean', ''), 6):>8} "
            f"{fmt(row.get('order_level_ret_mean_mean', ''), 4):>8} "
            f"{s1:>6.1f} {s2:>6.1f} {s3:>6.1f} "
            f"{asm:>9.0f} {cost_idx:>9.3f}"
        )

    # Table 2: Downstream multiplier analysis
    print("\n## Downstream Multiplier Analysis\n")
    header2 = (
        f"{'Policy':<28} "
        f"{'Op10 mean':>10} {'Op10 p95':>10} {'Op10 >=1.9%':>12} "
        f"{'Op12 mean':>10} {'Op12 p95':>10} {'Op12 >=1.9%':>12}"
    )
    print(header2)
    print("-" * len(header2))

    for policy, policy_rows in sorted(by_policy.items()):
        row = policy_rows[0]
        op10_mean = row.get("op10_multiplier_step_mean_mean", "")
        op10_p95 = row.get("op10_multiplier_step_p95_mean", "")
        op10_ge190 = row.get("pct_steps_op10_multiplier_ge_190_mean", "")
        op12_mean = row.get("op12_multiplier_step_mean_mean", "")
        op12_p95 = row.get("op12_multiplier_step_p95_mean", "")
        op12_ge190 = row.get("pct_steps_op12_multiplier_ge_190_mean", "")
        print(
            f"{policy:<28} "
            f"{fmt(op10_mean, 3):>10} {fmt(op10_p95, 3):>10} {fmt(op10_ge190, 1):>12} "
            f"{fmt(op12_mean, 3):>10} {fmt(op12_p95, 3):>10} {fmt(op12_ge190, 1):>12}"
        )

    # Table 3: PPO cost savings vs best static
    print("\n## PPO Cost Savings vs Best Static (s3_d2.00)\n")
    ppo_rows = by_policy.get("ppo", [])
    s3d2_rows = by_policy.get("s3_d2.00", [])
    if ppo_rows and s3d2_rows:
        ppo = ppo_rows[0]
        s3d2 = s3d2_rows[0]
        ppo_asm = float(ppo.get("assembly_hours_total_mean", 0))
        s3d2_asm = float(s3d2.get("assembly_hours_total_mean", 0))
        if s3d2_asm > 0:
            savings_pct = 100.0 * (1.0 - ppo_asm / s3d2_asm)
            print(f"  PPO assembly hours:    {ppo_asm:,.0f}")
            print(f"  s3_d2.00 assembly hours: {s3d2_asm:,.0f}")
            print(f"  Savings:               {savings_pct:.1f}%")
            print(f"  PPO fill rate:         {fmt(ppo.get('fill_rate_mean', ''), 6)}")
            print(f"  s3_d2.00 fill rate:    {fmt(s3d2.get('fill_rate_mean', ''), 6)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Track B action distributions")
    parser.add_argument(
        "sweep_dir",
        type=Path,
        help="Path to reward sweep output directory (contains run subdirs)",
    )
    args = parser.parse_args()

    if not args.sweep_dir.exists():
        print(f"Error: {args.sweep_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    rows = load_policy_summaries(args.sweep_dir)
    if not rows:
        print(f"No policy_summary.csv found under {args.sweep_dir}", file=sys.stderr)
        sys.exit(1)

    print_action_analysis(rows)


if __name__ == "__main__":
    main()
