#!/usr/bin/env python3
"""Report thesis decision-table constants from the current code.

This covers the static decision variables used by Garrido-Rios (2017):

- Table 6.16 inventory buffers at Op3/Op5/Op9.
- Table 6.20 capacity settings by number of shifts.

The script compares current code constants against the extracted thesis values
and writes machine-readable and markdown artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.config import CAPACITY_BY_SHIFTS, INVENTORY_BUFFERS, NUM_RAW_MATERIALS

DEFAULT_OUTPUT_ROOT = Path("outputs/benchmarks/thesis_decision_tables")

EXPECTED_INVENTORY_BUFFERS = {
    168: {"op3_rm": 15_360, "op5_rm": 15_360, "op9_rations": 15_750},
    336: {"op3_rm": 30_720, "op5_rm": 30_720, "op9_rations": 31_500},
    504: {"op3_rm": 46_080, "op5_rm": 46_080, "op9_rations": 47_250},
    672: {"op3_rm": 61_440, "op5_rm": 61_440, "op9_rations": 63_000},
    1344: {"op3_rm": 122_880, "op5_rm": 122_880, "op9_rations": 126_000},
}

EXPECTED_CAPACITY_BY_SHIFTS = {
    1: {
        "op3_q": 15_500,
        "op4_q": 15_500,
        "op7_q": 5_000,
        "op7_rop": 48,
        "op8_q": 5_000,
        "op8_rop": 48,
        "theoretical_capacity_hrs": 8,
        "theoretical_capacity_rations": 2_564,
    },
    2: {
        "op3_q": 31_000,
        "op4_q": 31_000,
        "op7_q": 5_000,
        "op7_rop": 24,
        "op8_q": 5_000,
        "op8_rop": 24,
        "theoretical_capacity_hrs": 16,
        "theoretical_capacity_rations": 5_128,
    },
    3: {
        "op3_q": 47_000,
        "op4_q": 47_000,
        "op7_q": 7_000,
        "op7_rop": 24,
        "op8_q": 7_000,
        "op8_rop": 24,
        "theoretical_capacity_hrs": 24,
        "theoretical_capacity_rations": 7_692,
    },
}


def inventory_rows() -> list[dict[str, Any]]:
    rows = []
    for period, expected in EXPECTED_INVENTORY_BUFFERS.items():
        actual = INVENTORY_BUFFERS[period]
        for key, expected_value in expected.items():
            actual_value = float(actual[key])
            rows.append(
                {
                    "table": "6.16",
                    "period_hours": period,
                    "field": key,
                    "expected": float(expected_value),
                    "actual": actual_value,
                    "status": (
                        "MATCH" if actual_value == float(expected_value) else "MISMATCH"
                    ),
                }
            )
    return rows


def capacity_rows() -> list[dict[str, Any]]:
    rows = []
    for shifts, expected in EXPECTED_CAPACITY_BY_SHIFTS.items():
        actual = CAPACITY_BY_SHIFTS[shifts]
        for key, expected_value in expected.items():
            actual_value = float(actual[key])
            rows.append(
                {
                    "table": "6.20",
                    "shifts": shifts,
                    "field": key,
                    "expected": float(expected_value),
                    "actual": actual_value,
                    "status": (
                        "MATCH" if actual_value == float(expected_value) else "MISMATCH"
                    ),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "table",
        "period_hours",
        "shifts",
        "field",
        "expected",
        "actual",
        "status",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(
    path: Path, inventory: list[dict[str, Any]], capacity: list[dict[str, Any]]
) -> None:
    all_rows = inventory + capacity
    status = "PASS" if all(row["status"] == "MATCH" for row in all_rows) else "FAIL"
    lines = [
        "# Thesis Decision Tables Report",
        "",
        f"Created UTC: `{datetime.now(timezone.utc).isoformat()}`",
        f"Overall status: `{status}`",
        f"NUM_RAW_MATERIALS: `{NUM_RAW_MATERIALS}`",
        "",
        "## Table 6.16 Inventory Buffers",
        "",
        "| period_hours | field | expected | actual | status |",
        "|---:|---|---:|---:|---|",
    ]
    for row in inventory:
        lines.append(
            f"| {row['period_hours']} | {row['field']} | "
            f"{row['expected']:.0f} | {row['actual']:.0f} | {row['status']} |"
        )

    lines += [
        "",
        "## Table 6.20 Capacity By Shifts",
        "",
        "| shifts | field | expected | actual | status |",
        "|---:|---|---:|---:|---|",
    ]
    for row in capacity:
        lines.append(
            f"| {row['shifts']} | {row['field']} | "
            f"{row['expected']:.0f} | {row['actual']:.0f} | {row['status']} |"
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
    inv = inventory_rows()
    cap = capacity_rows()
    all_rows = inv + cap
    write_csv(out_dir / "thesis_decision_tables.csv", all_rows)
    (out_dir / "thesis_decision_tables.json").write_text(
        json.dumps(
            {
                "status": (
                    "PASS"
                    if all(row["status"] == "MATCH" for row in all_rows)
                    else "FAIL"
                ),
                "num_raw_materials": NUM_RAW_MATERIALS,
                "inventory": inv,
                "capacity": cap,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    write_markdown(out_dir / "THESIS_DECISION_TABLES.md", inv, cap)
    print(out_dir / "THESIS_DECISION_TABLES.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
