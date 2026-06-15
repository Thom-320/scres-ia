#!/usr/bin/env python3
"""Report thesis backbone operation constants from the current code.

This covers the non-RL DES backbone used before risk and decision experiments:

- Op1-Op13 processing times, quantities, reorder points, risks, and units.
- Table 6.4 regular demand process.
- Thesis time constants used by the deterministic and stochastic gates.

The expected values are kept in this script as extracted thesis targets, then
compared against ``supply_chain.config`` so the report can catch drift.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.config import (
    ASSEMBLY_RATE,
    DAYS_PER_WEEK,
    DEMAND,
    HOURS_PER_MONTH,
    HOURS_PER_WEEK,
    HOURS_PER_YEAR_THESIS,
    LEAD_TIME_PROMISE,
    NUM_RAW_MATERIALS,
    NUM_SUPPLIERS,
    OPERATIONS,
    RATIONS_PER_BATCH,
    THESIS_DOWNSTREAM_Q_RANGES,
    WARMUP,
)

DEFAULT_OUTPUT_ROOT = Path("outputs/benchmarks/thesis_operations_table")

EXPECTED_OPERATIONS: dict[int, dict[str, Any]] = {
    1: {"pt": 672, "q": 12, "rop": 4032, "risks": ["R12"], "num_units": 1},
    2: {"pt": 24, "q": 190_000, "rop": 672, "risks": ["R13"], "num_units": 12},
    3: {"pt": 24, "q": 15_500, "rop": 168, "risks": ["R21"], "num_units": 1},
    4: {"pt": 24, "q": 15_500, "rop": 168, "risks": ["R22"], "num_units": 1},
    5: {
        "pt": 1 / 320.5,
        "q": 1,
        "rop": 1 / 320.5,
        "risks": ["R11", "R21", "R3"],
        "num_units": 1,
    },
    6: {
        "pt": 1 / 320.5,
        "q": 1,
        "rop": 1 / 320.5,
        "risks": ["R11", "R21", "R3"],
        "num_units": 1,
    },
    7: {
        "pt": 1 / 320.5,
        "q": 5_000,
        "rop": 48,
        "risks": ["R14", "R21", "R3"],
        "num_units": 1,
    },
    8: {"pt": 24, "q": 5_000, "rop": 48, "risks": ["R22"], "num_units": 1},
    9: {
        "pt": 24,
        "q": [2_400, 2_600],
        "rop": 24,
        "risks": ["R21", "R3"],
        "num_units": 1,
    },
    10: {"pt": 24, "q": [2_400, 2_600], "rop": 24, "risks": ["R22"], "num_units": 1},
    11: {"pt": 0, "q": [2_400, 2_600], "rop": 24, "risks": ["R23"], "num_units": 2},
    12: {"pt": 24, "q": [2_400, 2_600], "rop": 24, "risks": ["R22"], "num_units": 1},
    13: {"pt": 0, "q": 0, "rop": 0, "risks": ["R24"], "num_units": 1},
}

EXPECTED_DEMAND = {
    "distribution": "uniform_discrete",
    "a": 2_400,
    "b": 2_600,
    "frequency_hrs": 24,
    "operating_days_per_week": 6,
}

EXPECTED_TIME_CONSTANTS = {
    "assembly_rate": 320.5,
    "days_per_week": 6,
    "hours_per_week": 168,
    "hours_per_month": 672,
    "hours_per_year_thesis": 8_064,
    "num_raw_materials": 12,
    "num_suppliers": 12,
    "rations_per_batch": 5_000,
    "lead_time_promise": 48,
    "warmup_trigger_op": 9,
    "warmup_trigger_quantity": 5_000,
    "warmup_estimated_deterministic_hrs": 838.8,
}


def normalize_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    return value


def values_match(actual: Any, expected: Any) -> bool:
    actual = normalize_value(actual)
    expected = normalize_value(expected)
    if isinstance(actual, float) or isinstance(expected, float):
        return math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12)
    return actual == expected


def format_value(value: Any) -> str:
    value = normalize_value(value)
    if isinstance(value, float):
        return f"{value:.12g}"
    return json.dumps(value, sort_keys=True)


def operation_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for op_id, expected in EXPECTED_OPERATIONS.items():
        actual = OPERATIONS[op_id]
        for field, expected_value in expected.items():
            actual_value = normalize_value(actual[field])
            rows.append(
                {
                    "section": "operations",
                    "item": f"Op{op_id}",
                    "field": field,
                    "expected": format_value(expected_value),
                    "actual": format_value(actual_value),
                    "status": (
                        "MATCH"
                        if values_match(actual_value, expected_value)
                        else "MISMATCH"
                    ),
                }
            )
    return rows


def demand_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for field, expected_value in EXPECTED_DEMAND.items():
        actual_value = DEMAND[field]
        rows.append(
            {
                "section": "demand",
                "item": "Table 6.4",
                "field": field,
                "expected": format_value(expected_value),
                "actual": format_value(actual_value),
                "status": (
                    "MATCH"
                    if values_match(actual_value, expected_value)
                    else "MISMATCH"
                ),
            }
        )
    for op_name, expected_range in {
        "op9": [2_400, 2_600],
        "op10": [2_400, 2_600],
        "op12": [2_400, 2_600],
    }.items():
        actual_range = normalize_value(
            THESIS_DOWNSTREAM_Q_RANGES["figure_6_2"][op_name]
        )
        rows.append(
            {
                "section": "downstream_q",
                "item": "Figure 6.2",
                "field": op_name,
                "expected": format_value(expected_range),
                "actual": format_value(actual_range),
                "status": (
                    "MATCH"
                    if values_match(actual_range, expected_range)
                    else "MISMATCH"
                ),
            }
        )
    return rows


def time_constant_rows() -> list[dict[str, str]]:
    actuals = {
        "assembly_rate": ASSEMBLY_RATE,
        "days_per_week": DAYS_PER_WEEK,
        "hours_per_week": HOURS_PER_WEEK,
        "hours_per_month": HOURS_PER_MONTH,
        "hours_per_year_thesis": HOURS_PER_YEAR_THESIS,
        "num_raw_materials": NUM_RAW_MATERIALS,
        "num_suppliers": NUM_SUPPLIERS,
        "rations_per_batch": RATIONS_PER_BATCH,
        "lead_time_promise": LEAD_TIME_PROMISE,
        "warmup_trigger_op": WARMUP["trigger_op"],
        "warmup_trigger_quantity": WARMUP["trigger_quantity"],
        "warmup_estimated_deterministic_hrs": WARMUP["estimated_deterministic_hrs"],
    }
    rows: list[dict[str, str]] = []
    for field, expected_value in EXPECTED_TIME_CONSTANTS.items():
        actual_value = actuals[field]
        rows.append(
            {
                "section": "time_constants",
                "item": "thesis_backbone",
                "field": field,
                "expected": format_value(expected_value),
                "actual": format_value(actual_value),
                "status": (
                    "MATCH"
                    if values_match(actual_value, expected_value)
                    else "MISMATCH"
                ),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, str]]) -> None:
    status = "PASS" if all(row["status"] == "MATCH" for row in rows) else "FAIL"
    lines = [
        "# Thesis Operations Backbone Report",
        "",
        f"Created UTC: `{datetime.now(timezone.utc).isoformat()}`",
        f"Overall status: `{status}`",
        "",
        "## Summary",
        "",
        "| section | rows | mismatches |",
        "|---|---:|---:|",
    ]
    sections = sorted({row["section"] for row in rows})
    for section in sections:
        section_rows = [row for row in rows if row["section"] == section]
        mismatches = sum(row["status"] != "MATCH" for row in section_rows)
        lines.append(f"| {section} | {len(section_rows)} | {mismatches} |")

    lines += [
        "",
        "## Rows",
        "",
        "| section | item | field | expected | actual | status |",
        "|---|---|---|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['section']} | {row['item']} | {row['field']} | "
            f"{row['expected']} | {row['actual']} | {row['status']} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="current")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    out_dir = args.output_root / args.label
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = operation_rows() + demand_rows() + time_constant_rows()
    status = "PASS" if all(row["status"] == "MATCH" for row in rows) else "FAIL"

    write_csv(out_dir / "thesis_operations_table.csv", rows)
    (out_dir / "thesis_operations_table.json").write_text(
        json.dumps({"status": status, "rows": rows}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_markdown(out_dir / "THESIS_OPERATIONS_BACKBONE.md", rows)
    print(out_dir / "THESIS_OPERATIONS_BACKBONE.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
