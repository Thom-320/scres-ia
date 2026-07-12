#!/usr/bin/env python3
"""Audit R22/R23/R24 event-to-order attribution and downstream backlog discharge.

The R11-R24 calibration gate showed that R2 event counts are close to thesis
rates, but workbook-visible order shares for R22/R23/R24 are too low in the
endogenous DES. This script tests the mechanism: current attribution marks
orders overlapping the event duration, while downstream service impact can last
until the backlog clears.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.audit_garrido_risk_calibration import (  # noqa: E402
    RISK_COLUMN_OP_MAP,
    backlog_intervals,
    build_sim,
    des_order_visible_risk_values,
    quantiles,
)
from supply_chain.garrido_replication import load_raw_garrido_targets  # noqa: E402
from supply_chain.thesis_design import parse_cf_range  # noqa: E402


R2_COLUMNS = (
    "R21_1",
    "R21_2",
    "R21_3",
    "R21_4",
    "R21_5",
    "R22_1",
    "R22_2",
    "R22_3",
    "R22_4",
    "R23",
    "R24",
)
RISK_IDS = ("R21", "R22", "R23", "R24")


def parse_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _order_interval(order: Any, horizon: float) -> tuple[float, float] | None:
    if bool(getattr(order, "metrics_excluded", False)):
        return None
    start = float(order.OPTj)
    if getattr(order, "OATj", None) is None:
        end = horizon
    else:
        end = float(order.OATj)
    if end <= start:
        return None
    return start, end


def _overlaps(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    return max(a_start, b_start) < min(a_end, b_end)


def _backlog_clear_time_after(
    intervals: list[tuple[float, float]],
    event_end: float,
) -> float:
    for start, end in intervals:
        if start <= event_end <= end:
            return end
        if event_end < start:
            return event_end
    return event_end


def excel_r2_column_summary(cfi_values: list[int]) -> list[dict[str, Any]]:
    targets = load_raw_garrido_targets()
    rows: list[dict[str, Any]] = []
    for cfi in cfi_values:
        target = targets[cfi]
        for col in target.risk_columns:
            if col not in R2_COLUMNS:
                continue
            values = [
                float(order.risk_values.get(col, 0.0) or 0.0)
                for order in target.orders
            ]
            positives = [value for value in values if value > 0.0]
            rows.append(
                {
                    "source": "excel",
                    "cfi": cfi,
                    "risk_column": col,
                    "positive_share": len(positives) / max(1, len(values)),
                    "positive_count": len(positives),
                    "n_orders": len(values),
                    "positive_sum": sum(positives),
                    "positive_mean": float(np.mean(positives)) if positives else 0.0,
                }
            )
    return rows


def event_to_order_rows(cfi_values: list[int], seeds: list[int]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    targets = load_raw_garrido_targets()
    event_rows: list[dict[str, Any]] = []
    des_column_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    for cfi in cfi_values:
        target = targets[cfi]
        risk_columns = tuple(col for col in target.risk_columns if col in R2_COLUMNS)
        for seed in seeds:
            sim = build_sim(cfi=cfi, seed=seed)
            sim.run()
            horizon = float(sim.horizon)
            orders = [
                order
                for order in sim.orders
                if not bool(getattr(order, "metrics_excluded", False))
            ]
            intervals = backlog_intervals(sim)
            current_column_values: dict[str, list[float]] = {col: [] for col in risk_columns}
            tail_column_values: dict[str, list[float]] = {col: [] for col in risk_columns}

            order_intervals = {
                id(order): _order_interval(order, horizon)
                for order in orders
            }
            current_flagged_by_order: dict[int, set[str]] = {}
            for order in orders:
                values = des_order_visible_risk_values(order, risk_columns)
                flags = {col for col, value in values.items() if value > 0.0}
                current_flagged_by_order[id(order)] = flags
                for col, value in values.items():
                    current_column_values[col].append(float(value))

            for event in sim.risk_events:
                risk_id = str(event.risk_id)
                if risk_id not in RISK_IDS:
                    continue
                affected_ops = tuple(int(op) for op in getattr(event, "affected_ops", []) or [])
                mapped_columns = [
                    col
                    for col in risk_columns
                    if RISK_COLUMN_OP_MAP.get(col, ("", ()))[0] == risk_id
                    and (
                        not affected_ops
                        or set(RISK_COLUMN_OP_MAP.get(col, ("", ()))[1]).intersection(affected_ops)
                    )
                ]
                event_start = float(event.start_time)
                event_end = float(event.end_time)
                clear_time = _backlog_clear_time_after(intervals, event_end)
                overlap_orders: list[Any] = []
                tail_orders: list[Any] = []
                current_flagged_orders: list[Any] = []
                for order in orders:
                    interval = order_intervals[id(order)]
                    if interval is None:
                        continue
                    optj, oatj = interval
                    if _overlaps(optj, oatj, event_start, event_end):
                        overlap_orders.append(order)
                    if clear_time > event_end and _overlaps(optj, oatj, event_start, clear_time):
                        tail_orders.append(order)
                    elif clear_time <= event_end and _overlaps(optj, oatj, event_start, event_end):
                        tail_orders.append(order)
                    if current_flagged_by_order[id(order)].intersection(mapped_columns):
                        current_flagged_orders.append(order)

                for order in tail_orders:
                    for col in mapped_columns:
                        # Tail-propagated attribution uses 1 as a visibility flag;
                        # this is only an audit, not a proposed formula change.
                        tail_column_values[col].append(1.0)

                event_rows.append(
                    {
                        "cfi": cfi,
                        "seed": seed,
                        "risk_id": risk_id,
                        "risk_columns": "|".join(mapped_columns),
                        "affected_ops": "|".join(str(op) for op in affected_ops),
                        "start_time": event_start,
                        "end_time": event_end,
                        "duration": float(event.duration),
                        "magnitude": float(getattr(event, "magnitude", 0.0) or 0.0),
                        "clear_time_after_event": clear_time,
                        "backlog_tail_hours": max(0.0, clear_time - event_end),
                        "orders_time_overlap": len({id(order) for order in overlap_orders}),
                        "orders_current_flagged": len({id(order) for order in current_flagged_orders}),
                        "orders_tail_window": len({id(order) for order in tail_orders}),
                    }
                )

            for col, values in current_column_values.items():
                positives = [value for value in values if value > 0.0]
                des_column_rows.append(
                    {
                        "mode": "current_attribution",
                        "cfi": cfi,
                        "seed": seed,
                        "risk_column": col,
                        "positive_share": len(positives) / max(1, len(values)),
                        "positive_count": len(positives),
                        "n_orders": len(values),
                        "positive_sum": sum(positives),
                        "positive_mean": float(np.mean(positives)) if positives else 0.0,
                    }
                )
            for col in risk_columns:
                # Use unique order ids for tail visibility by reconstructing from event rows
                # is overkill for the audit; count positives at event-hit granularity to
                # show the direction of the gap.
                values = tail_column_values[col]
                des_column_rows.append(
                    {
                        "mode": "tail_window_event_hits",
                        "cfi": cfi,
                        "seed": seed,
                        "risk_column": col,
                        "positive_share": len(values) / max(1, len(orders)),
                        "positive_count": len(values),
                        "n_orders": len(orders),
                        "positive_sum": sum(values),
                        "positive_mean": float(np.mean(values)) if values else 0.0,
                    }
                )

            terminal_detail = sim._inventory_detail()
            run_rows.append(
                {
                    "cfi": cfi,
                    "seed": seed,
                    "n_orders": len(orders),
                    "lost_orders": sum(1 for order in orders if bool(getattr(order, "lost", False))),
                    "pending_backorders_terminal": len(sim.pending_backorders),
                    "pending_backorder_qty_terminal": float(sim.pending_backorder_qty),
                    "rations_sb_dispatch_terminal": terminal_detail["rations_sb_dispatch"],
                    "rations_cssu_terminal": terminal_detail["rations_cssu"],
                    "rations_theatre_terminal": terminal_detail["rations_theatre"],
                    "delivery_events": len(sim.delivery_events),
                    "total_delivered": float(sim.total_delivered),
                    "backlog_interval_count": len(intervals),
                    "backlog_positive_hours": sum(end - start for start, end in intervals),
                    "backlog_max_interval": max((end - start for start, end in intervals), default=0.0),
                }
            )
    return event_rows, des_column_rows, run_rows


def aggregate_columns(excel_rows: list[dict[str, Any]], des_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    excel_group: dict[tuple[int, str], dict[str, float]] = {}
    for row in excel_rows:
        key = (int(row["cfi"]), str(row["risk_column"]))
        excel_group[key] = {
            "positive_count": float(row["positive_count"]),
            "n_orders": float(row["n_orders"]),
            "positive_share": float(row["positive_share"]),
        }
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in des_rows:
        grouped[(str(row["mode"]), int(row["cfi"]), str(row["risk_column"]))].append(row)

    rows: list[dict[str, Any]] = []
    for (mode, cfi, col), group in sorted(grouped.items()):
        excel = excel_group.get((cfi, col), {})
        des_count = sum(float(row["positive_count"]) for row in group)
        des_n = sum(float(row["n_orders"]) for row in group)
        des_share = des_count / max(1.0, des_n)
        excel_share = float(excel.get("positive_share", 0.0))
        rows.append(
            {
                "mode": mode,
                "cfi": cfi,
                "risk_column": col,
                "excel_positive_share": excel_share,
                "des_positive_share": des_share,
                "share_ratio": des_share / excel_share if excel_share > 0.0 else float("nan"),
                "excel_positive_count": int(excel.get("positive_count", 0.0)),
                "des_positive_count": int(des_count),
            }
        )
    return rows


def aggregate_event_rows(event_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in event_rows:
        grouped[(int(row["cfi"]), str(row["risk_id"]))].append(row)
    rows: list[dict[str, Any]] = []
    for (cfi, risk_id), group in sorted(grouped.items()):
        rows.append(
            {
                "cfi": cfi,
                "risk_id": risk_id,
                "event_count": len(group),
                "duration_mean": float(np.mean([float(row["duration"]) for row in group])),
                "tail_hours_p50": quantiles([float(row["backlog_tail_hours"]) for row in group]).get("p50", 0.0),
                "tail_hours_p95": quantiles([float(row["backlog_tail_hours"]) for row in group]).get("p95", 0.0),
                "orders_overlap_mean": float(np.mean([float(row["orders_time_overlap"]) for row in group])),
                "orders_current_flagged_mean": float(np.mean([float(row["orders_current_flagged"]) for row in group])),
                "orders_tail_window_mean": float(np.mean([float(row["orders_tail_window"]) for row in group])),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    output_dir: Path,
    event_summary: list[dict[str, Any]],
    column_summary: list[dict[str, Any]],
    run_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Garrido R2 Downstream Attribution Audit",
        "",
        "This audit tests whether R22/R23/R24 are under-attributed because the DES "
        "marks only direct event overlap while the operational backlog tail persists "
        "after the event ends.",
        "",
        "## Event Tail Summary",
        "",
        "| CF | Risk | Events | Tail p50 h | Tail p95 h | Orders overlap | Current flagged | Tail-window orders |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in event_summary:
        lines.append(
            f"| {row['cfi']} | {row['risk_id']} | {row['event_count']} | "
            f"{row['tail_hours_p50']:.0f} | {row['tail_hours_p95']:.0f} | "
            f"{row['orders_overlap_mean']:.1f} | "
            f"{row['orders_current_flagged_mean']:.1f} | "
            f"{row['orders_tail_window_mean']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Column Share Summary",
            "",
            "| Mode | CF | Column | Excel share | DES/audit share | Ratio |",
            "| --- | ---: | --- | ---: | ---: | ---: |",
        ]
    )
    for row in column_summary:
        if row["risk_column"] not in {"R22_1", "R22_2", "R22_3", "R22_4", "R23", "R24"}:
            continue
        lines.append(
            f"| {row['mode']} | {row['cfi']} | {row['risk_column']} | "
            f"{row['excel_positive_share']:.3f} | "
            f"{row['des_positive_share']:.3f} | {row['share_ratio']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Run-Terminal Downstream State",
            "",
            "| CF | Seed | Lost | Pending qty | SB dispatch | CSSU | Theatre | Backlog max h |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in run_rows:
        lines.append(
            f"| {row['cfi']} | {row['seed']} | {row['lost_orders']} | "
            f"{row['pending_backorder_qty_terminal']:.0f} | "
            f"{row['rations_sb_dispatch_terminal']:.0f} | "
            f"{row['rations_cssu_terminal']:.0f} | "
            f"{row['rations_theatre_terminal']:.0f} | "
            f"{row['backlog_max_interval']:.0f} |"
        )
    lines.extend(
        [
            "",
            "Interpretation guide:",
            "",
            "- If tail-window orders greatly exceed current-flagged orders, the DES "
            "is likely under-attributing downstream risk impact to later orders.",
            "- If terminal CSSU/theatre inventories are low with persistent pending "
            "backorders, the blocker is downstream discharge/catch-up rather than "
            "assembly shifts.",
            "",
        ]
    )
    (output_dir / "audit_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cf-range", default="11-20")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/audits/garrido_r2_downstream_2026-06-26"),
    )
    args = parser.parse_args()

    cfi_values = parse_cf_range(args.cf_range)
    seeds = parse_ints(args.seeds)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    excel_rows = excel_r2_column_summary(cfi_values)
    event_rows, des_column_rows, run_rows = event_to_order_rows(cfi_values, seeds)
    column_summary = aggregate_columns(excel_rows, des_column_rows)
    event_summary = aggregate_event_rows(event_rows)

    write_csv(args.output_dir / "excel_r2_columns.csv", excel_rows)
    write_csv(args.output_dir / "event_to_order_rows.csv", event_rows)
    write_csv(args.output_dir / "event_to_order_summary.csv", event_summary)
    write_csv(args.output_dir / "des_r2_column_modes.csv", des_column_rows)
    write_csv(args.output_dir / "r2_column_mode_summary.csv", column_summary)
    write_csv(args.output_dir / "run_terminal_downstream.csv", run_rows)
    (args.output_dir / "audit.json").write_text(
        json.dumps(
            {
                "cf_range": cfi_values,
                "seeds": seeds,
                "event_summary": event_summary,
                "column_summary": column_summary,
                "run_terminal_downstream": run_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_report(args.output_dir, event_summary, column_summary, run_rows)

    print(f"WROTE {args.output_dir}")
    for row in event_summary[:20]:
        print(
            f"CF{row['cfi']:02d} {row['risk_id']}: "
            f"tail_p95={row['tail_hours_p95']:.0f}h "
            f"current={row['orders_current_flagged_mean']:.1f} "
            f"tail_orders={row['orders_tail_window_mean']:.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
