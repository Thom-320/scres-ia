#!/usr/bin/env python3
"""Audit endogenous DES CTj/RPj/DPj tails against Garrido raw workbooks.

This is the G6 tail gate: formula replay may pass while the endogenous DES
still diverges in the trajectory distribution. The script compares pooled
order-level quantiles for:

* R1 family: CF1-CF10, risks R11-R14.
* R2 family: CF11-CF20, risks R21-R24.

Excel headers are read through the canonical Garrido extractor, so Raw_data2's
11-column risk block is handled dynamically.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.config import (  # noqa: E402
    THESIS_FAITHFUL_PROTOCOL as P,
    THESIS_REPLICATION_DOWNSTREAM_Q_SOURCE as DQ,
)
from supply_chain.garrido_replication import load_raw_garrido_targets  # noqa: E402
from supply_chain.supply_chain import MFSCSimulation, SIMULATION_HORIZON  # noqa: E402
from supply_chain.thesis_design import R1_RISKS, R2_RISKS  # noqa: E402


FAMILIES = {
    "R1": {
        "cf_range": range(1, 11),
        "enabled_risks": tuple(R1_RISKS),
    },
    "R2": {
        "cf_range": range(11, 21),
        "enabled_risks": tuple(R2_RISKS),
    },
}
METRICS = ("CTj", "RPj", "DPj", "APj")
QUANTILES = (50, 90, 95, 99)


def parse_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_combos(value: str) -> list[tuple[str, str]]:
    combos: list[tuple[str, str]] = []
    for part in value.split(","):
        token = part.strip()
        if not token:
            continue
        rp_mode, overflow_mode = token.split("/", maxsplit=1)
        combos.append((rp_mode.strip(), overflow_mode.strip()))
    return combos


def quantiles(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0.0}
    out: dict[str, float] = {"n": float(arr.size), "mean": float(arr.mean())}
    for q in QUANTILES:
        out[f"p{q}"] = float(np.percentile(arr, q))
    out["max"] = float(arr.max())
    return out


def excel_family_quantiles() -> dict[str, dict[str, Any]]:
    targets = load_raw_garrido_targets()
    result: dict[str, dict[str, Any]] = {}
    for family, spec in FAMILIES.items():
        pools = {metric: [] for metric in METRICS}
        n_orders = 0
        risk_columns: set[str] = set()
        for cfi in spec["cf_range"]:
            target = targets[cfi]
            risk_columns.update(target.risk_columns)
            for order in target.orders:
                n_orders += 1
                pools["CTj"].append(float(order.ctj))
                pools["RPj"].append(float(order.rpj))
                pools["DPj"].append(float(order.dpj))
                pools["APj"].append(float(order.apj))
        result[family] = {
            "source": "excel_raw_workbooks",
            "n_orders": n_orders,
            "risk_columns": sorted(risk_columns),
            "quantiles": {
                metric: quantiles(values) for metric, values in pools.items()
            },
        }
    return result


def make_sim(
    *,
    enabled_risks: tuple[str, ...],
    seed: int,
    horizon: float,
    rp_mode: str,
    overflow_mode: str,
    demand_on_hand_fulfillment_delay: float,
) -> MFSCSimulation:
    return MFSCSimulation(
        shifts=1,
        seed=seed,
        horizon=horizon,
        risks_enabled=True,
        risk_level="current",
        enabled_risks=set(enabled_risks),
        risk_occurrence_mode="thesis_window",
        risk_attribution_source="des_events",
        ret_recovery_period_mode=rp_mode,
        backorder_overflow_mode=overflow_mode,
        year_basis=P["year_basis"],
        warmup_trigger=P["warmup_trigger"],
        r14_defect_mode=P["r14_defect_mode"],
        downstream_q_source=DQ,
        raw_material_flow_mode=P["raw_material_flow_mode"],
        raw_material_order_up_to_multiplier=P[
            "raw_material_order_up_to_multiplier"
        ],
        demand_on_hand_fulfillment_delay=demand_on_hand_fulfillment_delay,
    )


def des_family_quantiles(
    *,
    seeds: list[int],
    horizon: float,
    combos: list[tuple[str, str]],
    demand_on_hand_fulfillment_delay: float,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for rp_mode, overflow_mode in combos:
        combo_key = f"{rp_mode}/{overflow_mode}"
        result[combo_key] = {}
        for family, spec in FAMILIES.items():
            pools = {metric: [] for metric in METRICS}
            lost = 0
            served = 0
            event_counts: list[int] = []
            for seed in seeds:
                sim = make_sim(
                    enabled_risks=spec["enabled_risks"],
                    seed=seed,
                    horizon=horizon,
                    rp_mode=rp_mode,
                    overflow_mode=overflow_mode,
                    demand_on_hand_fulfillment_delay=(
                        demand_on_hand_fulfillment_delay
                    ),
                )
                sim.run()
                event_counts.append(len(sim.risk_events))
                for order in sim.orders:
                    if getattr(order, "metrics_excluded", False):
                        continue
                    if getattr(order, "lost", False):
                        lost += 1
                        continue
                    if order.CTj is None or order.OATj is None:
                        continue
                    served += 1
                    pools["CTj"].append(float(order.CTj))
                    pools["RPj"].append(float(getattr(order, "RPj", 0.0) or 0.0))
                    pools["DPj"].append(float(getattr(order, "DPj", 0.0) or 0.0))
                    pools["APj"].append(float(getattr(order, "APj", 0.0) or 0.0))
            result[combo_key][family] = {
                "source": "des_events",
                "rp_mode": rp_mode,
                "overflow_mode": overflow_mode,
                "served_orders": served,
                "lost_orders": lost,
                "risk_events_mean": statistics.fmean(event_counts),
                "quantiles": {
                    metric: quantiles(values) for metric, values in pools.items()
                },
            }
    return result


def flat_rows(
    excel: dict[str, dict[str, Any]],
    des: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family, family_data in excel.items():
        for metric, q in family_data["quantiles"].items():
            for key, value in q.items():
                rows.append(
                    {
                        "source": "excel",
                        "combo": "excel",
                        "family": family,
                        "metric": metric,
                        "stat": key,
                        "value": value,
                        "excel_value": value,
                        "ratio_to_excel": 1.0,
                    }
                )
    for combo, combo_data in des.items():
        for family, family_data in combo_data.items():
            for metric, q in family_data["quantiles"].items():
                for key, value in q.items():
                    excel_value = excel[family]["quantiles"][metric].get(key)
                    ratio = (
                        float(value) / float(excel_value)
                        if excel_value not in (None, 0.0)
                        and math.isfinite(float(excel_value))
                        else float("nan")
                    )
                    rows.append(
                        {
                            "source": "des_events",
                            "combo": combo,
                            "family": family,
                            "metric": metric,
                            "stat": key,
                            "value": value,
                            "excel_value": excel_value,
                            "ratio_to_excel": ratio,
                        }
                    )
    return rows


def write_report(
    output_dir: Path,
    rows: list[dict[str, Any]],
    des: dict[str, dict[str, Any]],
) -> None:
    lines = [
        "# Garrido Tail Fidelity Audit",
        "",
        "This audit compares pooled order-level CTj/RPj/DPj/APj quantiles "
        "from the raw Garrido workbooks against the endogenous DES lane.",
        "",
        "## Key p99 Ratios",
        "",
        "| Combo | Family | CTj p99 ratio | RPj p99 ratio | Lost orders |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for combo, combo_data in des.items():
        for family, family_data in combo_data.items():
            ct = next(
                row
                for row in rows
                if row["combo"] == combo
                and row["family"] == family
                and row["metric"] == "CTj"
                and row["stat"] == "p99"
            )
            rp = next(
                row
                for row in rows
                if row["combo"] == combo
                and row["family"] == family
                and row["metric"] == "RPj"
                and row["stat"] == "p99"
            )
            lines.append(
                f"| `{combo}` | {family} | {ct['ratio_to_excel']:.2f} | "
                f"{rp['ratio_to_excel']:.2f} | {family_data['lost_orders']} |"
            )
    lines.extend(
        [
            "",
            "Interpretation: values above 1.0 are heavier than Garrido. "
            "If CTj remains above 1.0 after RPj is corrected, the blocker is "
            "queue/backlog dynamics rather than the ReT formula alone.",
            "",
        ]
    )
    (output_dir / "audit_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--horizon-hours", type=float, default=float(SIMULATION_HORIZON))
    parser.add_argument(
        "--combos",
        default="disruption/largest,disruption/oldest,elapsed/largest",
        help="Comma-separated rp_mode/backorder_overflow_mode combinations.",
    )
    parser.add_argument(
        "--demand-on-hand-fulfillment-delay",
        type=float,
        default=float(P["demand_on_hand_fulfillment_delay"]),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/audits/garrido_tail_fidelity_2026-06-26"),
    )
    args = parser.parse_args()

    seeds = parse_ints(args.seeds)
    combos = parse_combos(args.combos)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    excel = excel_family_quantiles()
    des = des_family_quantiles(
        seeds=seeds,
        horizon=float(args.horizon_hours),
        combos=combos,
        demand_on_hand_fulfillment_delay=float(
            args.demand_on_hand_fulfillment_delay
        ),
    )
    rows = flat_rows(excel, des)

    with (args.output_dir / "tail_quantiles.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "tail_quantiles.json").write_text(
        json.dumps(
            {
                "seeds": seeds,
                "horizon_hours": float(args.horizon_hours),
                "demand_on_hand_fulfillment_delay": float(
                    args.demand_on_hand_fulfillment_delay
                ),
                "excel": excel,
                "des": des,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_report(args.output_dir, rows, des)

    print(f"WROTE {args.output_dir}")
    for combo, combo_data in des.items():
        for family, family_data in combo_data.items():
            ct_p99 = next(
                row
                for row in rows
                if row["combo"] == combo
                and row["family"] == family
                and row["metric"] == "CTj"
                and row["stat"] == "p99"
            )
            rp_p99 = next(
                row
                for row in rows
                if row["combo"] == combo
                and row["family"] == family
                and row["metric"] == "RPj"
                and row["stat"] == "p99"
            )
            print(
                f"{combo:18} {family}: CTj p99 ratio={ct_p99['ratio_to_excel']:.2f} "
                f"RPj p99 ratio={rp_p99['ratio_to_excel']:.2f} "
                f"lost={family_data['lost_orders']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
