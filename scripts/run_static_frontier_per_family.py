#!/usr/bin/env python3
"""Build the dense static frontier per risk family (R1-only, R2-only, R3-only).

For each family F in {R1, R2, R3}, run all combinations of:
  - 6 inventory levels (0, 168, 336, 504, 672, 1344) ×
  - 3 shift levels (1, 2, 3) ×
  - 3 seeds (1, 2, 3) = 54 cells per family

All runs use:
  - `enabled_risks=family.risks` (R1_RISKS / R2_RISKS / R3_RISKS)
  - `risk_level=current` (NOT increased — the freeze is current)
  - `risk_occurrence_mode=thesis_window`
  - `demand_on_hand_fulfillment_delay=54.0` (freeze)
  - `ret_recovery_period_mode=disruption`

This is the "fair bar" against which the RL family-lane policy is compared.

Output:
  outputs/benchmarks/family_static_frontier_2026-06-29/
    design_matrix.csv          — (cfi, family, ...) per cell
    summary.csv                — (family, level, shift, seed, mean_ret, ...)
    family_summary.csv         — (family, level, shift) aggregated across seeds
    best_per_family.csv        — (family, best_level, best_shift, mean_ret)
    frontier_audit.json        — machine-readable summary
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.config import (  # noqa: E402
    INVENTORY_BUFFERS,
    RET_RECOVERY_PERIOD_MODE,
    THESIS_FAITHFUL_PROTOCOL as P,
    THESIS_REPLICATION_DOWNSTREAM_Q_SOURCE as DQ,
)
from supply_chain.supply_chain import MFSCSimulation  # noqa: E402
from supply_chain.thesis_design import (  # noqa: E402
    R1_RISKS,
    R2_RISKS,
    R3_RISKS,
)


FAMILIES = {
    "R1": {
        "risks": R1_RISKS,
        "horizon_hours": 161_280.0,  # 20 years (R1 is the well-tested family)
    },
    "R2": {
        "risks": R2_RISKS,
        "horizon_hours": 80_640.0,  # 10 years (per thesis CF11-CF20 design)
    },
    "R3": {
        "risks": R3_RISKS,
        "horizon_hours": 161_280.0,  # 20 years (R3 needs 20y to see 1 black-swan)
    },
}

INVENTORY_LEVELS = [0, 168, 336, 504, 672, 1344]
SHIFT_LEVELS = [1, 2, 3]


def build_sim(
    *,
    family: str,
    period: int,
    shifts: int,
    seed: int,
) -> MFSCSimulation:
    bufs = dict(INVENTORY_BUFFERS[period]) if period else None
    return MFSCSimulation(
        shifts=shifts,
        initial_buffers=bufs,
        seed=seed,
        horizon=FAMILIES[family]["horizon_hours"],
        risks_enabled=True,
        risk_level="current",
        enabled_risks=set(FAMILIES[family]["risks"]),
        risk_occurrence_mode=P["risk_occurrence_mode"],
        year_basis=P["year_basis"],
        warmup_trigger=P["warmup_trigger"],
        r14_defect_mode=P["r14_defect_mode"],
        downstream_q_source=DQ,
        raw_material_flow_mode=P["raw_material_flow_mode"],
        raw_material_order_up_to_multiplier=P[
            "raw_material_order_up_to_multiplier"
        ],
        demand_on_hand_fulfillment_delay=P["demand_on_hand_fulfillment_delay"],
        ret_recovery_period_mode=RET_RECOVERY_PERIOD_MODE,
        backorder_overflow_mode="largest",
        inventory_replenishment_period=(float(period) if period else None),
    )


def run_sim(sim: MFSCSimulation) -> dict[str, Any]:
    sim.run()
    served = [
        o
        for o in sim.orders
        if not bool(getattr(o, "metrics_excluded", False))
        and getattr(o, "OATj", None) is not None
        and not bool(getattr(o, "lost", False))
    ]
    lost = sum(1 for o in sim.orders if bool(getattr(o, "lost", False)))
    ctj = np.asarray(
        [float(o.CTj) for o in served if o.CTj is not None]
    )
    if ctj.size == 0:
        return {
            "mean_ret": 0.0,
            "fill_rate": 0.0,
            "n_orders": 0,
            "lost_orders": float(lost),
            "pending_qty": float(sim.pending_backorder_qty),
            "ct_p99": 0.0,
        }
    ret = sim.compute_order_level_ret()
    return {
        "mean_ret": float(ret["mean_ret"]),
        "fill_rate": float(ret["fill_rate_order_level"]),
        "n_orders": len(sim.orders),
        "lost_orders": float(lost),
        "pending_qty": float(sim.pending_backorder_qty),
        "ct_p99": float(np.percentile(ctj, 99)),
    }


def parse_families(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families", default="R1,R2,R3")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/benchmarks/family_static_frontier_2026-06-29"),
    )
    args = parser.parse_args()

    families = parse_families(args.families)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    rows: list[dict[str, Any]] = []
    for family in families:
        if family not in FAMILIES:
            print(f"Skipping unknown family {family}")
            continue
        for period in INVENTORY_LEVELS:
            for shifts in SHIFT_LEVELS:
                for seed in seeds:
                    sim = build_sim(
                        family=family,
                        period=period,
                        shifts=shifts,
                        seed=seed,
                    )
                    metrics = run_sim(sim)
                    rows.append(
                        {
                            "family": family,
                            "level": period,
                            "shift": shifts,
                            "seed": seed,
                            **metrics,
                        }
                    )

    fieldnames = list(rows[0].keys())
    with (args.output_dir / "summary.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    family_summary: list[dict[str, Any]] = []
    for family in families:
        frows = [r for r in rows if r["family"] == family]
        if not frows:
            continue
        for period in INVENTORY_LEVELS:
            for shifts in SHIFT_LEVELS:
                prows = [
                    r
                    for r in frows
                    if r["level"] == period and r["shift"] == shifts
                ]
                if not prows:
                    continue
                family_summary.append(
                    {
                        "family": family,
                        "level": period,
                        "shift": shifts,
                        "mean_ret_mean": fmean(r["mean_ret"] for r in prows),
                        "mean_ret_sd": pstdev(
                            [r["mean_ret"] for r in prows]
                        )
                        if len(prows) > 1
                        else 0.0,
                        "fill_rate_mean": fmean(
                            r["fill_rate"] for r in prows
                        ),
                        "lost_orders_mean": fmean(
                            r["lost_orders"] for r in prows
                        ),
                        "pending_qty_mean": fmean(
                            r["pending_qty"] for r in prows
                        ),
                        "ct_p99_mean": fmean(r["ct_p99"] for r in prows),
                    }
                )

    family_summary_fields = list(family_summary[0].keys())
    with (args.output_dir / "family_summary.csv").open(
        "w", newline=""
    ) as fh:
        writer = csv.DictWriter(fh, fieldnames=family_summary_fields)
        writer.writeheader()
        writer.writerows(family_summary)

    best_per_family: list[dict[str, Any]] = []
    for family in families:
        frows = [r for r in family_summary if r["family"] == family]
        if not frows:
            continue
        best = max(frows, key=lambda r: r["mean_ret_mean"])
        worst = min(frows, key=lambda r: r["mean_ret_mean"])
        best_per_family.append(
            {
                "family": family,
                "best_level": best["level"],
                "best_shift": best["shift"],
                "best_mean_ret": best["mean_ret_mean"],
                "best_mean_ret_sd": best["mean_ret_sd"],
                "best_fill_rate": best["fill_rate_mean"],
                "best_lost_orders": best["lost_orders_mean"],
                "worst_level": worst["level"],
                "worst_shift": worst["shift"],
                "worst_mean_ret": worst["mean_ret_mean"],
            }
        )

    best_fields = list(best_per_family[0].keys())
    with (args.output_dir / "best_per_family.csv").open(
        "w", newline=""
    ) as fh:
        writer = csv.DictWriter(fh, fieldnames=best_fields)
        writer.writeheader()
        writer.writerows(best_per_family)

    design_matrix = [
        {
            "family": family,
            "level": period,
            "shift": shifts,
            "horizon_hours": FAMILIES[family]["horizon_hours"],
            "enabled_risks": ",".join(FAMILIES[family]["risks"]),
        }
        for family in families
        for period in INVENTORY_LEVELS
        for shifts in SHIFT_LEVELS
    ]
    with (args.output_dir / "design_matrix.csv").open(
        "w", newline=""
    ) as fh:
        writer = csv.DictWriter(
            fh, fieldnames=list(design_matrix[0].keys())
        )
        writer.writeheader()
        writer.writerows(design_matrix)

    elapsed = time.time() - started
    audit = {
        "families": families,
        "seeds": seeds,
        "inventory_levels": INVENTORY_LEVELS,
        "shift_levels": SHIFT_LEVELS,
        "wall_seconds": elapsed,
        "n_cells": len(family_summary),
        "best_per_family": best_per_family,
    }
    (args.output_dir / "frontier_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )

    lines = [
        "# Family Static Frontier (2026-06-29)",
        "",
        "Dense static frontier per risk family, used as the fair bar for the",
        "Garrido-pure RL family-lane (Entregable 3.2).",
        "",
        "## Best per family (highest mean ReT across the 6×3 grid)",
        "",
        "| Family | Best level | Best shift | Mean ReT | ± sd | Fill rate | Lost |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in best_per_family:
        lines.append(
            f"| {row['family']} | {row['best_level']} | {row['best_shift']} "
            f"| {row['best_mean_ret']:.4f} "
            f"| ±{row['best_mean_ret_sd']:.4f} "
            f"| {row['best_fill_rate']:.3f} "
            f"| {row['best_lost_orders']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Worst per family (lowest mean ReT)",
            "",
            "| Family | Worst level | Worst shift | Mean ReT |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in best_per_family:
        lines.append(
            f"| {row['family']} | {row['worst_level']} | {row['worst_shift']} "
            f"| {row['worst_mean_ret']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"Cells per family: {len(INVENTORY_LEVELS) * len(SHIFT_LEVELS)} (6 levels × 3 shifts)",
            f"Seeds per cell: {len(seeds)}",
            f"Total runs: {len(rows)}",
            f"Wall time: {elapsed:.1f}s",
        ]
    )
    (args.output_dir / "frontier_audit.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    print(f"WROTE {args.output_dir} (wall {elapsed:.1f}s)")
    for row in best_per_family:
        print(
            f"  {row['family']}: best=L{row['best_level']}_S{row['best_shift']} "
            f"mean_ret={row['best_mean_ret']:.4f} lost={row['best_lost_orders']:.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
