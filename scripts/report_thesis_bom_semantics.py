#!/usr/bin/env python3
"""Report Table 6.1 BOM and raw-material flow semantics.

This gate makes explicit the assumption behind the structural inventory fix:
one ration is a kit composed of one unit from each of rm1..rm12. The legacy
validated lane keeps historical aggregate consumption, while the repaired
kit-equivalent lane consumes 12 aggregate raw-material units per ration and
scales raw-material inventory targets accordingly.
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

from supply_chain.config import NUM_RAW_MATERIALS, RAW_MATERIAL_COMPONENTS
from supply_chain.supply_chain import MFSCSimulation

DEFAULT_OUTPUT_ROOT = Path("outputs/benchmarks/thesis_bom_semantics")

EXPECTED_COMPONENTS = [
    {"id": "rm1", "item": "meat pastry", "quantity": "1 portion", "weight_gr": 150},
    {
        "id": "rm2",
        "item": "chocolate with cheese",
        "quantity": "1 bar",
        "weight_gr": 25,
    },
    {"id": "rm3", "item": "wheat bread", "quantity": "1 piece", "weight_gr": 100},
    {
        "id": "rm4",
        "item": "chickpeas soup",
        "quantity": "1 portion",
        "weight_gr": 180,
    },
    {
        "id": "rm5",
        "item": "hydrating drink",
        "quantity": "1 sachet",
        "weight_gr": 36,
    },
    {"id": "rm6", "item": "corn bread", "quantity": "1 piece", "weight_gr": 100},
    {"id": "rm7", "item": "meat goulash", "quantity": "1 portion", "weight_gr": 180},
    {"id": "rm8", "item": "fruit bread", "quantity": "1 piece", "weight_gr": 100},
    {"id": "rm9", "item": "sugar cane", "quantity": "1 bar", "weight_gr": 125},
    {
        "id": "rm10",
        "item": "condensed milk",
        "quantity": "1 can",
        "weight_gr": 100,
    },
    {
        "id": "rm11",
        "item": "peanuts with sesame",
        "quantity": "1 bag",
        "weight_gr": 100,
    },
    {"id": "rm12", "item": "mixed fruit", "quantity": "1 bag", "weight_gr": 50},
]


def component_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for expected, actual in zip(
        EXPECTED_COMPONENTS, RAW_MATERIAL_COMPONENTS, strict=True
    ):
        rows.append(
            {
                "section": "table_6_1_component",
                "id": expected["id"],
                "expected_item": expected["item"],
                "actual_item": actual["item"],
                "expected_quantity": expected["quantity"],
                "actual_quantity": actual["quantity"],
                "expected_weight_gr": expected["weight_gr"],
                "actual_weight_gr": actual["weight_gr"],
                "status": "MATCH" if actual == expected else "MISMATCH",
            }
        )
    if len(RAW_MATERIAL_COMPONENTS) != len(EXPECTED_COMPONENTS):
        rows.append(
            {
                "section": "table_6_1_component",
                "id": "component_count",
                "expected_item": str(len(EXPECTED_COMPONENTS)),
                "actual_item": str(len(RAW_MATERIAL_COMPONENTS)),
                "expected_quantity": "",
                "actual_quantity": "",
                "expected_weight_gr": "",
                "actual_weight_gr": "",
                "status": "MISMATCH",
            }
        )
    return rows


def semantics_rows() -> list[dict[str, Any]]:
    legacy = MFSCSimulation(raw_material_flow_mode="legacy_validated")
    kit = MFSCSimulation(raw_material_flow_mode="kit_equivalent_order_up_to")
    checks = [
        {
            "id": "num_raw_materials",
            "expected": NUM_RAW_MATERIALS,
            "actual": len(RAW_MATERIAL_COMPONENTS),
        },
        {
            "id": "legacy_units_per_ration",
            "expected": 1.0,
            "actual": legacy._raw_units_per_ration,
        },
        {
            "id": "kit_units_per_ration",
            "expected": float(NUM_RAW_MATERIALS),
            "actual": kit._raw_units_per_ration,
        },
        {
            "id": "kit_canonical_mode",
            "expected": "bom_total_units_order_up_to",
            "actual": kit.raw_material_flow_mode,
        },
    ]
    return [
        {
            "section": "flow_semantics",
            "id": check["id"],
            "expected_item": str(check["expected"]),
            "actual_item": str(check["actual"]),
            "expected_quantity": "",
            "actual_quantity": "",
            "expected_weight_gr": "",
            "actual_weight_gr": "",
            "status": "MATCH" if check["actual"] == check["expected"] else "MISMATCH",
        }
        for check in checks
    ]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    status = "PASS" if all(row["status"] == "MATCH" for row in rows) else "FAIL"
    lines = [
        "# Thesis BOM Semantics Report",
        "",
        f"Created UTC: `{datetime.now(timezone.utc).isoformat()}`",
        f"Overall status: `{status}`",
        "",
        "## Table 6.1 Components",
        "",
        "| rm | expected item | actual item | quantity | weight_gr | status |",
        "|---|---|---|---|---:|---|",
    ]
    for row in rows:
        if row["section"] != "table_6_1_component":
            continue
        lines.append(
            f"| {row['id']} | {row['expected_item']} | {row['actual_item']} | "
            f"{row['actual_quantity']} | {row['actual_weight_gr']} | {row['status']} |"
        )

    lines += [
        "",
        "## Flow Semantics",
        "",
        "| check | expected | actual | status |",
        "|---|---:|---:|---|",
    ]
    for row in rows:
        if row["section"] != "flow_semantics":
            continue
        lines.append(
            f"| {row['id']} | {row['expected_item']} | "
            f"{row['actual_item']} | {row['status']} |"
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
    rows = component_rows() + semantics_rows()
    status = "PASS" if all(row["status"] == "MATCH" for row in rows) else "FAIL"

    write_csv(out_dir / "thesis_bom_semantics.csv", rows)
    (out_dir / "thesis_bom_semantics.json").write_text(
        json.dumps({"status": status, "rows": rows}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_markdown(out_dir / "THESIS_BOM_SEMANTICS.md", rows)
    print(out_dir / "THESIS_BOM_SEMANTICS.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
