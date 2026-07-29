#!/usr/bin/env python3
"""Audit endogenous R11-R24 calibration against Garrido Excel and thesis rates.

The previous gates proved that the Excel formula is reproduced and that the
remaining endogenous DES divergence is in the trajectory tail. This script
separates three possible causes:

1. Risk process calibration: event counts/durations vs thesis occurrence rates.
2. Workbook-visible risk attribution: raw Excel risk columns vs DES order flags.
3. Backlog catch-up: how long the pending-order queue remains non-empty after
   R11/R13 disruption recovery, by shift level.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.config import (  # noqa: E402
    DEMAND,
    HOURS_PER_DAY,
    HOURS_PER_WEEK,
    HOURS_PER_YEAR_THESIS,
    RISKS_CURRENT,
    THESIS_FAITHFUL_PROTOCOL as P,
    THESIS_REPLICATION_DOWNSTREAM_Q_SOURCE as DQ,
)
from supply_chain.garrido_replication import (  # noqa: E402
    GarridoCFTarget,
    load_raw_garrido_targets,
)
from supply_chain.supply_chain import MFSCSimulation  # noqa: E402
from supply_chain.thesis_design import (  # noqa: E402
    R1_RISKS,
    R2_RISKS,
    design_spec_for_cfi,
    parse_cf_range,
)


RISK_COLUMN_OP_MAP: dict[str, tuple[str, tuple[int, ...]]] = {
    "R11_1": ("R11", (5,)),
    "R11_2": ("R11", (6,)),
    "R12": ("R12", (1,)),
    "R13": ("R13", (2,)),
    "R14": ("R14", (7,)),
    "R21_1": ("R21", (3,)),
    "R21_2": ("R21", (5,)),
    "R21_3": ("R21", (6,)),
    "R21_4": ("R21", (7,)),
    "R21_5": ("R21", (9,)),
    "R22_1": ("R22", (4,)),
    "R22_2": ("R22", (8,)),
    "R22_3": ("R22", (10,)),
    "R22_4": ("R22", (12,)),
    "R23": ("R23", (11,)),
    "R24": ("R24", (13,)),
}
FAMILIES = {
    "R1": tuple(R1_RISKS),
    "R2": tuple(R2_RISKS),
}
QUANTILES = (50, 90, 95, 99)


def parse_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_shifts(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def quantiles(values: Iterable[float]) -> dict[str, float]:
    arr = np.asarray([v for v in values if math.isfinite(float(v))], dtype=float)
    if arr.size == 0:
        return {"n": 0.0}
    out = {"n": float(arr.size), "mean": float(arr.mean())}
    for q in QUANTILES:
        out[f"p{q}"] = float(np.percentile(arr, q))
    out["max"] = float(arr.max())
    return out


def _risk_table(cfi: int, risk_id: str) -> dict[str, Any]:
    spec = design_spec_for_cfi(cfi)
    level = spec.risk_overrides.get(risk_id, "current")
    if level == "current":
        return dict(RISKS_CURRENT[risk_id]["occurrence"])
    from supply_chain.config import RISKS_INCREASED  # local import keeps module tidy

    table = dict(RISKS_CURRENT[risk_id]["occurrence"])
    table.update(RISKS_INCREASED.get(risk_id, {}))
    return table


def expected_rate_for_cfi(cfi: int, risk_id: str) -> dict[str, float]:
    """Return thesis-window expected annual group and unit rates for a risk."""
    occ = _risk_table(cfi, risk_id)
    dist = str(occ.get("dist", ""))
    if dist == "uniform":
        b = float(occ["b"])
        events = HOURS_PER_YEAR_THESIS / b
        if risk_id == "R24":
            from supply_chain.config import RISKS_INCREASED

            spec = design_spec_for_cfi(cfi)
            level = spec.risk_overrides.get("R24", "current")
            lo = RISKS_CURRENT["R24"]["surge"]["lo"]
            hi = RISKS_CURRENT["R24"]["surge"]["hi"]
            if level != "current":
                lo = RISKS_INCREASED["R24"].get("surge_lo", lo)
                hi = RISKS_INCREASED["R24"].get("surge_hi", hi)
            units = events * ((float(lo) + float(hi)) / 2.0)
        else:
            units = events
        return {"expected_event_groups_per_year": events, "expected_units_per_year": units}
    if dist == "binomial":
        n = float(occ["n"])
        p = float(occ["p"])
        if risk_id == "R12":
            cycles = HOURS_PER_YEAR_THESIS / 4032.0
        elif risk_id == "R13":
            cycles = HOURS_PER_YEAR_THESIS / HOURS_PER_WEEK
        elif risk_id == "R14":
            cycles = (HOURS_PER_YEAR_THESIS / HOURS_PER_DAY) * (
                DEMAND["operating_days_per_week"] / 7.0
            )
        else:
            cycles = HOURS_PER_YEAR_THESIS / HOURS_PER_WEEK
        event_groups = cycles * (1.0 - ((1.0 - p) ** n))
        units = cycles * n * p
        return {
            "expected_event_groups_per_year": event_groups,
            "expected_units_per_year": units,
        }
    return {"expected_event_groups_per_year": float("nan"), "expected_units_per_year": float("nan")}


def build_sim(
    *,
    cfi: int,
    seed: int,
    shifts: int | None = None,
    enabled_risks: tuple[str, ...] | None = None,
) -> MFSCSimulation:
    spec = design_spec_for_cfi(cfi)
    return MFSCSimulation(
        shifts=int(shifts if shifts is not None else spec.shifts),
        seed=int(seed),
        horizon=float(spec.horizon_hours),
        risks_enabled=True,
        enabled_risks=set(enabled_risks or spec.enabled_risks),
        risk_overrides=spec.risk_overrides,
        risk_level="current",
        risk_occurrence_mode="thesis_window",
        risk_attribution_source="des_events",
        year_basis=P["year_basis"],
        warmup_trigger=P["warmup_trigger"],
        r14_defect_mode=P["r14_defect_mode"],
        downstream_q_source=DQ,
        raw_material_flow_mode=P["raw_material_flow_mode"],
        raw_material_order_up_to_multiplier=P[
            "raw_material_order_up_to_multiplier"
        ],
        demand_on_hand_fulfillment_delay=float(
            P["demand_on_hand_fulfillment_delay"]
        ),
    )


def excel_column_rows(targets: dict[int, GarridoCFTarget], cfi_values: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cfi in cfi_values:
        target = targets[cfi]
        for col in target.risk_columns:
            values = [float(order.risk_values.get(col, 0.0) or 0.0) for order in target.orders]
            positives = [value for value in values if value > 0.0]
            rows.append(
                {
                    "source": "excel",
                    "cfi": cfi,
                    "risk_column": col,
                    "n_orders": len(values),
                    "positive_count": len(positives),
                    "positive_share": len(positives) / max(1, len(values)),
                    "positive_sum": sum(positives),
                    "positive_mean": statistics.fmean(positives) if positives else 0.0,
                    **{f"positive_{k}": v for k, v in quantiles(positives).items()},
                }
            )
    return rows


def des_order_visible_risk_values(order: Any, risk_columns: tuple[str, ...]) -> dict[str, float]:
    values = {col: 0.0 for col in risk_columns}
    refs = list(getattr(order, "ret_risk_event_refs", []) or [])
    for ref in refs:
        risk_id = str(ref.get("risk_id", ""))
        affected_ops = tuple(int(op) for op in (ref.get("affected_ops", []) or []))
        duration = float(ref.get("duration", 0.0) or 0.0)
        magnitude = float(ref.get("magnitude", 0.0) or 0.0)
        contribution = max(duration, magnitude, 1.0)
        for col in risk_columns:
            mapped = RISK_COLUMN_OP_MAP.get(col)
            if mapped is None:
                continue
            mapped_risk, mapped_ops = mapped
            if mapped_risk != risk_id:
                continue
            if not affected_ops or set(affected_ops).intersection(mapped_ops):
                values[col] += contribution

    # Fallback for indicators that do not have detailed refs.
    indicators = getattr(order, "ret_risk_indicators", {}) or {}
    for col in risk_columns:
        if values[col] > 0.0:
            continue
        mapped = RISK_COLUMN_OP_MAP.get(col)
        if mapped is None:
            continue
        risk_id, _ops = mapped
        if col in indicators:
            values[col] = float(indicators[col])
        elif risk_id in indicators:
            values[col] = float(indicators[risk_id])
    return values


def run_des_cf(
    *,
    targets: dict[int, GarridoCFTarget],
    cfi_values: list[int],
    seeds: list[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    event_rows: list[dict[str, Any]] = []
    column_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    for cfi in cfi_values:
        target = targets[cfi]
        years = float(design_spec_for_cfi(cfi).horizon_hours) / HOURS_PER_YEAR_THESIS
        column_acc: dict[str, list[float]] = defaultdict(list)
        for seed in seeds:
            sim = build_sim(cfi=cfi, seed=seed)
            sim.run()
            event_by_risk: dict[str, list[Any]] = defaultdict(list)
            for event in sim.risk_events:
                event_by_risk[str(event.risk_id)].append(event)
            for risk_id in sorted(design_spec_for_cfi(cfi).enabled_risks):
                events = event_by_risk.get(risk_id, [])
                durations = [float(event.duration) for event in events]
                magnitudes = [float(getattr(event, "magnitude", 0.0) or 0.0) for event in events]
                expected = expected_rate_for_cfi(cfi, risk_id)
                event_rows.append(
                    {
                        "cfi": cfi,
                        "seed": seed,
                        "risk_id": risk_id,
                        "event_count": len(events),
                        "event_count_per_year": len(events) / years,
                        "expected_event_groups_per_year": expected[
                            "expected_event_groups_per_year"
                        ],
                        "event_count_ratio": (
                            (len(events) / years)
                            / expected["expected_event_groups_per_year"]
                            if expected["expected_event_groups_per_year"]
                            else float("nan")
                        ),
                        "magnitude_sum_per_year": sum(magnitudes) / years,
                        "expected_units_per_year": expected["expected_units_per_year"],
                        "duration_mean": statistics.fmean(durations) if durations else 0.0,
                        "duration_p95": float(np.percentile(durations, 95)) if durations else 0.0,
                        "duration_max": max(durations) if durations else 0.0,
                    }
                )

            nonexcluded = [
                order
                for order in sim.orders
                if not bool(getattr(order, "metrics_excluded", False))
            ]
            for order in nonexcluded:
                values = des_order_visible_risk_values(order, target.risk_columns)
                for col, value in values.items():
                    column_acc[col].append(float(value))
            served_ct = [
                float(order.CTj)
                for order in nonexcluded
                if getattr(order, "CTj", None) is not None
            ]
            trajectory_rows.append(
                {
                    "cfi": cfi,
                    "seed": seed,
                    "n_orders": len(nonexcluded),
                    "served_orders": len(served_ct),
                    "lost_orders": sum(1 for order in nonexcluded if bool(getattr(order, "lost", False))),
                    "pending_backorders_terminal": len(sim.pending_backorders),
                    "pending_backorder_qty_terminal": float(sim.pending_backorder_qty),
                    "ct_p50": quantiles(served_ct).get("p50", 0.0),
                    "ct_p95": quantiles(served_ct).get("p95", 0.0),
                    "ct_p99": quantiles(served_ct).get("p99", 0.0),
                    "ct_max": quantiles(served_ct).get("max", 0.0),
                }
            )
        for col in target.risk_columns:
            values = column_acc[col]
            positives = [value for value in values if value > 0.0]
            column_rows.append(
                {
                    "source": "des",
                    "cfi": cfi,
                    "risk_column": col,
                    "n_orders": len(values),
                    "positive_count": len(positives),
                    "positive_share": len(positives) / max(1, len(values)),
                    "positive_sum": sum(positives),
                    "positive_mean": statistics.fmean(positives) if positives else 0.0,
                    **{f"positive_{k}": v for k, v in quantiles(positives).items()},
                }
            )
    return event_rows, column_rows, trajectory_rows


def backlog_intervals(sim: MFSCSimulation) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []
    horizon = float(sim.horizon)
    baseline_delay = float(getattr(sim, "demand_on_hand_fulfillment_delay", 0.0) or 0.0)
    for order in sim.orders:
        if bool(getattr(order, "metrics_excluded", False)):
            continue
        optj = float(order.OPTj)
        if bool(getattr(order, "lost", False)) or getattr(order, "OATj", None) is None:
            start = optj + baseline_delay
            end = horizon
        else:
            end = float(order.OATj)
            if (end - optj) <= baseline_delay + 1e-9:
                continue
            start = optj + baseline_delay
        if end > start:
            intervals.append((start, min(end, horizon)))
    intervals.sort()
    merged: list[tuple[float, float]] = []
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def catchup_after_events(sim: MFSCSimulation, risk_ids: set[str]) -> list[float]:
    intervals = backlog_intervals(sim)
    waits: list[float] = []
    for event in sim.risk_events:
        if str(event.risk_id) not in risk_ids:
            continue
        t = float(event.end_time)
        wait = 0.0
        for start, end in intervals:
            if start <= t <= end:
                wait = max(0.0, end - t)
                break
            if t < start:
                break
        waits.append(wait)
    return waits


def catchup_rows(*, seeds: list[int], shifts: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    # Representative current-risk CFs for each family.
    family_specs = {"R1": (1, tuple(R1_RISKS), {"R11", "R13"}), "R2": (11, tuple(R2_RISKS), {"R21", "R22", "R23"})}
    for family, (cfi, enabled, catchup_risks) in family_specs.items():
        for shift_count in shifts:
            for seed in seeds:
                sim = build_sim(cfi=cfi, seed=seed, shifts=shift_count, enabled_risks=enabled)
                sim.run()
                intervals = backlog_intervals(sim)
                waits = catchup_after_events(sim, catchup_risks)
                interval_lengths = [end - start for start, end in intervals]
                event_counts = Counter(str(event.risk_id) for event in sim.risk_events)
                rows.append(
                    {
                        "family": family,
                        "cfi": cfi,
                        "shifts": shift_count,
                        "seed": seed,
                        "n_backlog_intervals": len(intervals),
                        "backlog_positive_hours": sum(interval_lengths),
                        "backlog_positive_share": sum(interval_lengths) / max(float(sim.horizon), 1.0),
                        "max_backlog_episode_hours": max(interval_lengths) if interval_lengths else 0.0,
                        "catchup_event_count": len(waits),
                        "catchup_zero_share": sum(1 for wait in waits if wait <= 0.0) / max(1, len(waits)),
                        "catchup_p50": quantiles(waits).get("p50", 0.0),
                        "catchup_p95": quantiles(waits).get("p95", 0.0),
                        "catchup_p99": quantiles(waits).get("p99", 0.0),
                        "catchup_max": quantiles(waits).get("max", 0.0),
                        "lost_orders": sum(1 for order in sim.orders if bool(getattr(order, "lost", False))),
                        "served_orders": sum(1 for order in sim.orders if getattr(order, "OATj", None) is not None),
                        "terminal_pending_backorders": len(sim.pending_backorders),
                        "event_counts": dict(event_counts),
                    }
                )
    return rows


def aggregate_event_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["cfi"]), str(row["risk_id"]))].append(row)
    out: list[dict[str, Any]] = []
    for (cfi, risk_id), group in sorted(grouped.items()):
        out.append(
            {
                "cfi": cfi,
                "risk_id": risk_id,
                "n": len(group),
                "event_count_per_year_mean": statistics.fmean(row["event_count_per_year"] for row in group),
                "expected_event_groups_per_year": group[0]["expected_event_groups_per_year"],
                "event_count_ratio_mean": statistics.fmean(row["event_count_ratio"] for row in group),
                "magnitude_sum_per_year_mean": statistics.fmean(row["magnitude_sum_per_year"] for row in group),
                "expected_units_per_year": group[0]["expected_units_per_year"],
                "duration_mean": statistics.fmean(row["duration_mean"] for row in group),
                "duration_p95_mean": statistics.fmean(row["duration_p95"] for row in group),
                "duration_max": max(row["duration_max"] for row in group),
            }
        )
    return out


def compare_visible_columns(
    excel_rows: list[dict[str, Any]],
    des_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    excel = {
        (int(row["cfi"]), str(row["risk_column"])): row
        for row in excel_rows
    }
    des = {
        (int(row["cfi"]), str(row["risk_column"])): row
        for row in des_rows
    }
    rows: list[dict[str, Any]] = []
    for key, excel_row in sorted(excel.items()):
        des_row = des.get(key)
        if des_row is None:
            continue
        excel_share = float(excel_row["positive_share"])
        des_share = float(des_row["positive_share"])
        rows.append(
            {
                "cfi": key[0],
                "family": "R1" if key[0] <= 10 else "R2",
                "risk_column": key[1],
                "excel_positive_share": excel_share,
                "des_positive_share": des_share,
                "positive_share_ratio": (
                    des_share / excel_share if excel_share > 0.0 else float("nan")
                ),
                "excel_positive_count": int(excel_row["positive_count"]),
                "des_positive_count": int(des_row["positive_count"]),
                "excel_positive_sum": float(excel_row["positive_sum"]),
                "des_positive_sum": float(des_row["positive_sum"]),
            }
        )
    return rows


def aggregate_visible_comparison(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["family"]), str(row["risk_column"]))].append(row)
    out: list[dict[str, Any]] = []
    for (family, risk_column), group in sorted(grouped.items()):
        excel_count = sum(int(row["excel_positive_count"]) for row in group)
        des_count = sum(int(row["des_positive_count"]) for row in group)
        # Reconstruct denominators from count/share where possible.
        excel_n = sum(
            int(round(row["excel_positive_count"] / row["excel_positive_share"]))
            for row in group
            if float(row["excel_positive_share"]) > 0.0
        )
        des_n = sum(
            int(round(row["des_positive_count"] / row["des_positive_share"]))
            for row in group
            if float(row["des_positive_share"]) > 0.0
        )
        excel_share = excel_count / max(1, excel_n)
        des_share = des_count / max(1, des_n)
        out.append(
            {
                "family": family,
                "risk_column": risk_column,
                "excel_positive_share": excel_share,
                "des_positive_share": des_share,
                "positive_share_ratio": (
                    des_share / excel_share if excel_share > 0.0 else float("nan")
                ),
                "excel_positive_count": excel_count,
                "des_positive_count": des_count,
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    output_dir: Path,
    event_summary: list[dict[str, Any]],
    visible_summary: list[dict[str, Any]],
    catchup: list[dict[str, Any]],
) -> None:
    lines = [
        "# Garrido R11-R24 Calibration Audit",
        "",
        "This audit separates risk process calibration, workbook-visible risk "
        "attribution, and backlog catch-up after disruptions.",
        "",
        "## Event Frequency Snapshot",
        "",
        "| CF | Risk | DES events/year | Expected groups/year | Ratio | Duration mean | Duration max |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in event_summary[:60]:
        lines.append(
            f"| {row['cfi']} | {row['risk_id']} | "
            f"{row['event_count_per_year_mean']:.2f} | "
            f"{row['expected_event_groups_per_year']:.2f} | "
            f"{row['event_count_ratio_mean']:.2f} | "
            f"{row['duration_mean']:.1f} | {row['duration_max']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Workbook-Visible Risk Column Snapshot",
            "",
            "| Family | Column | Excel positive share | DES positive share | Ratio |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in visible_summary:
        lines.append(
            f"| {row['family']} | {row['risk_column']} | "
            f"{row['excel_positive_share']:.3f} | "
            f"{row['des_positive_share']:.3f} | "
            f"{row['positive_share_ratio']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Backlog Catch-Up Snapshot",
            "",
            "| Family | Shifts | Seed | Backlog positive share | Max backlog episode h | Catch-up p95 h | Lost |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in catchup:
        lines.append(
            f"| {row['family']} | {row['shifts']} | {row['seed']} | "
            f"{row['backlog_positive_share']:.3f} | "
            f"{row['max_backlog_episode_hours']:.0f} | "
            f"{row['catchup_p95']:.0f} | {row['lost_orders']} |"
        )
    lines.extend(
        [
            "",
            "Interpretation guide:",
            "",
            "- Event ratios near 1.0 mean the occurrence process is calibrated.",
            "- Large catch-up times with event ratios near 1.0 point to queue "
            "recovery/capacity dynamics rather than risk-frequency mismatch.",
            "- Compare S1/S2/S3 rows to see whether the gap is specific to the "
            "tight S1 utilization regime.",
            "",
        ]
    )
    (output_dir / "audit_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cf-range", default="1-20")
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--catchup-seeds", default="1,2,3")
    parser.add_argument("--catchup-shifts", default="1,2,3")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/audits/garrido_risk_calibration_2026-06-26"),
    )
    args = parser.parse_args()

    cfi_values = parse_cf_range(args.cf_range)
    seeds = parse_ints(args.seeds)
    catchup_seed_values = parse_ints(args.catchup_seeds)
    shift_values = parse_shifts(args.catchup_shifts)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    targets = load_raw_garrido_targets()
    excel_columns = excel_column_rows(targets, cfi_values)
    event_rows, des_columns, trajectory_rows = run_des_cf(
        targets=targets,
        cfi_values=cfi_values,
        seeds=seeds,
    )
    event_summary = aggregate_event_rows(event_rows)
    visible_comparison = compare_visible_columns(excel_columns, des_columns)
    visible_summary = aggregate_visible_comparison(visible_comparison)
    catchup = catchup_rows(seeds=catchup_seed_values, shifts=shift_values)

    write_csv(args.output_dir / "excel_visible_risk_columns.csv", excel_columns)
    write_csv(args.output_dir / "des_visible_risk_columns.csv", des_columns)
    write_csv(
        args.output_dir / "visible_risk_column_comparison.csv",
        visible_comparison,
    )
    write_csv(
        args.output_dir / "visible_risk_column_summary.csv",
        visible_summary,
    )
    write_csv(args.output_dir / "des_event_rows.csv", event_rows)
    write_csv(args.output_dir / "des_event_summary.csv", event_summary)
    write_csv(args.output_dir / "des_trajectory_rows.csv", trajectory_rows)
    write_csv(args.output_dir / "backlog_catchup.csv", catchup)
    (args.output_dir / "audit.json").write_text(
        json.dumps(
            {
                "cf_range": cfi_values,
                "seeds": seeds,
                "catchup_seeds": catchup_seed_values,
                "catchup_shifts": shift_values,
                "event_summary": event_summary,
                "visible_risk_column_summary": visible_summary,
                "catchup": catchup,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_report(args.output_dir, event_summary, visible_summary, catchup)

    print(f"WROTE {args.output_dir}")
    print("Event ratios (first 12):")
    for row in event_summary[:12]:
        print(
            f"  CF{row['cfi']:02d} {row['risk_id']}: "
            f"{row['event_count_ratio_mean']:.2f}x expected, "
            f"duration_mean={row['duration_mean']:.1f}h"
        )
    print("Catch-up rows:")
    for row in catchup:
        print(
            f"  {row['family']} S{row['shifts']} seed={row['seed']}: "
            f"backlog_share={row['backlog_positive_share']:.3f}, "
            f"max_episode={row['max_backlog_episode_hours']:.0f}h, "
            f"catchup_p95={row['catchup_p95']:.0f}h, lost={row['lost_orders']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
