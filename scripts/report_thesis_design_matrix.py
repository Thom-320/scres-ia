#!/usr/bin/env python3
"""Report the Garrido-Rios Cf1-Cf90 design matrix from current code.

This checks the scenario ladder encoded by ``supply_chain.thesis_design``:

- Cf1-Cf30: risk-only factorial rows.
- Cf31-Cf60: inventory moderation rows mapped back to Cf1-Cf30.
- Cf61-Cf90: capacity moderation rows mapped back to Cf1-Cf30.
- R1/R2 rows run for 10 thesis years; R3 rows run for 20 thesis years.

The expected design values are duplicated here so this script acts as an
independent drift check rather than a simple dump of ``thesis_design.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.config import HOURS_PER_YEAR_THESIS, INVENTORY_BUFFERS
from supply_chain.thesis_design import design_matrix

DEFAULT_OUTPUT_ROOT = Path("outputs/benchmarks/thesis_design_matrix")

EXPECTED_R1_RISKS = ("R11", "R12", "R13", "R14")
EXPECTED_R2_RISKS = ("R21", "R22", "R23", "R24")
EXPECTED_R3_RISKS = ("R3",)

EXPECTED_RISK_PATTERNS: dict[int, tuple[bool, ...]] = {
    1: (False, False, True, True),
    2: (False, True, False, False),
    3: (True, False, True, True),
    4: (True, True, True, False),
    5: (False, False, True, False),
    6: (True, True, False, True),
    7: (True, False, False, True),
    8: (True, False, False, False),
    9: (False, True, True, True),
    10: (False, True, False, True),
    11: (True, False, True, True),
    12: (True, False, False, False),
    13: (True, True, False, True),
    14: (True, True, True, False),
    15: (False, False, True, True),
    16: (False, True, False, False),
    17: (False, True, True, False),
    18: (False, False, True, False),
    19: (False, True, False, True),
    20: (True, True, True, True),
    21: (False,),
    22: (True,),
    23: (True,),
    24: (True,),
    25: (True,),
    26: (False,),
    27: (False,),
    28: (False,),
    29: (True,),
    30: (False,),
}

EXPECTED_INVENTORY_PERIOD_BY_CFI = {
    31: 504,
    32: 336,
    33: 168,
    34: 1344,
    35: 336,
    36: 1344,
    37: 672,
    38: 672,
    39: 168,
    40: 504,
    41: 1344,
    42: 336,
    43: 504,
    44: 168,
    45: 504,
    46: 1344,
    47: 168,
    48: 336,
    49: 672,
    50: 672,
    51: 672,
    52: 1344,
    53: 672,
    54: 1344,
    55: 504,
    56: 504,
    57: 336,
    58: 336,
    59: 168,
    60: 168,
}

EXPECTED_SHIFTS_BY_CFI = {
    61: 2,
    62: 1,
    63: 3,
    64: 3,
    65: 1,
    66: 2,
    67: 1,
    68: 2,
    69: 3,
    70: 3,
    71: 1,
    72: 3,
    73: 2,
    74: 3,
    75: 2,
    76: 3,
    77: 2,
    78: 1,
    79: 2,
    80: 1,
    81: 1,
    82: 3,
    83: 2,
    84: 3,
    85: 2,
    86: 3,
    87: 2,
    88: 1,
    89: 2,
    90: 1,
}


def expected_source_cfi(cfi: int) -> int:
    if 1 <= cfi <= 30:
        return cfi
    if 31 <= cfi <= 60:
        return cfi - 30
    if 61 <= cfi <= 90:
        return cfi - 60
    raise ValueError(f"Cf{cfi} is outside 1..90")


def expected_family(cfi: int) -> str:
    if 1 <= cfi <= 10:
        return "risk_r1"
    if 11 <= cfi <= 20:
        return "risk_r2"
    if 21 <= cfi <= 30:
        return "risk_r3"
    if 31 <= cfi <= 60:
        return "inventory"
    if 61 <= cfi <= 90:
        return "capacity"
    raise ValueError(f"Cf{cfi} is outside 1..90")


def expected_risks(source_cfi: int) -> tuple[str, ...]:
    if 1 <= source_cfi <= 10:
        return EXPECTED_R1_RISKS
    if 11 <= source_cfi <= 20:
        return EXPECTED_R2_RISKS
    if 21 <= source_cfi <= 30:
        return EXPECTED_R3_RISKS
    raise ValueError(f"Cf{source_cfi} has no risk group")


def expected_overrides(source_cfi: int) -> dict[str, str]:
    risks = expected_risks(source_cfi)
    pattern = EXPECTED_RISK_PATTERNS[source_cfi]
    return {
        risk_id: "increased" if is_increased else "current"
        for risk_id, is_increased in zip(risks, pattern, strict=True)
    }


def expected_horizon_hours(source_cfi: int) -> float:
    years = 20 if 21 <= source_cfi <= 30 else 10
    return float(years * HOURS_PER_YEAR_THESIS)


def expected_initial_buffers(cfi: int) -> dict[str, float] | None:
    period = EXPECTED_INVENTORY_PERIOD_BY_CFI.get(cfi)
    if period is None:
        return None
    return {key: float(value) for key, value in INVENTORY_BUFFERS[period].items()}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


def row_for_spec(spec: Any) -> dict[str, Any]:
    cfi = int(spec.cfi)
    source_cfi = expected_source_cfi(cfi)
    expected_period = EXPECTED_INVENTORY_PERIOD_BY_CFI.get(cfi)
    expected_buffers = expected_initial_buffers(cfi)
    expected_shift = EXPECTED_SHIFTS_BY_CFI.get(cfi, 1)

    checks = {
        "family": spec.family == expected_family(cfi),
        "source_cfi": spec.source_cfi == source_cfi,
        "enabled_risks": tuple(spec.enabled_risks) == expected_risks(source_cfi),
        "risk_overrides": dict(spec.risk_overrides) == expected_overrides(source_cfi),
        "shifts": spec.shifts == expected_shift,
        "inventory_replenishment_period": spec.inventory_replenishment_period
        == (float(expected_period) if expected_period is not None else None),
        "initial_buffers": spec.initial_buffers == expected_buffers,
        "horizon_hours": spec.horizon_hours == expected_horizon_hours(source_cfi),
    }
    return {
        "Cfi": cfi,
        "status": "MATCH" if all(checks.values()) else "MISMATCH",
        "failed_fields": ",".join(
            field for field, matched in checks.items() if not matched
        ),
        "family": spec.family,
        "expected_family": expected_family(cfi),
        "source_cfi": spec.source_cfi,
        "expected_source_cfi": source_cfi,
        "enabled_risks": ",".join(spec.enabled_risks),
        "expected_enabled_risks": ",".join(expected_risks(source_cfi)),
        "risk_overrides": canonical_json(spec.risk_overrides),
        "expected_risk_overrides": canonical_json(expected_overrides(source_cfi)),
        "shifts": spec.shifts,
        "expected_shifts": expected_shift,
        "inventory_replenishment_period": spec.inventory_replenishment_period,
        "expected_inventory_replenishment_period": expected_period,
        "initial_buffers": canonical_json(spec.initial_buffers or {}),
        "expected_initial_buffers": canonical_json(expected_buffers or {}),
        "horizon_hours": spec.horizon_hours,
        "expected_horizon_hours": expected_horizon_hours(source_cfi),
    }


def summary_for(rows: list[dict[str, Any]]) -> dict[str, Any]:
    status = "PASS" if all(row["status"] == "MATCH" for row in rows) else "FAIL"
    families = Counter(row["family"] for row in rows)
    horizon_years = Counter(
        str(int(float(row["horizon_hours"]) / HOURS_PER_YEAR_THESIS)) for row in rows
    )
    inventory_periods = Counter(
        str(int(row["inventory_replenishment_period"]))
        for row in rows
        if row["inventory_replenishment_period"] is not None
    )
    shifts = Counter(str(int(row["shifts"])) for row in rows)
    return {
        "status": status,
        "row_count": len(rows),
        "mismatch_count": sum(row["status"] != "MATCH" for row in rows),
        "family_counts": dict(sorted(families.items())),
        "horizon_year_counts": dict(sorted(horizon_years.items())),
        "inventory_period_counts": dict(sorted(inventory_periods.items())),
        "shift_counts": dict(sorted(shifts.items())),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(
    path: Path, rows: list[dict[str, Any]], summary: dict[str, Any]
) -> None:
    lines = [
        "# Thesis Design Matrix Report",
        "",
        f"Created UTC: `{datetime.now(timezone.utc).isoformat()}`",
        f"Overall status: `{summary['status']}`",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| row_count | {summary['row_count']} |",
        f"| mismatch_count | {summary['mismatch_count']} |",
        f"| family_counts | `{canonical_json(summary['family_counts'])}` |",
        f"| horizon_year_counts | `{canonical_json(summary['horizon_year_counts'])}` |",
        f"| inventory_period_counts | `{canonical_json(summary['inventory_period_counts'])}` |",
        f"| shift_counts | `{canonical_json(summary['shift_counts'])}` |",
        "",
        "## Design Rows",
        "",
        "| Cfi | status | family | source_cfi | risks | risk_overrides | shifts | I period | horizon_hours | failed_fields |",
        "|---:|---|---|---:|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['Cfi']} | {row['status']} | {row['family']} | "
            f"{row['source_cfi']} | {row['enabled_risks']} | "
            f"`{row['risk_overrides']}` | {row['shifts']} | "
            f"{row['inventory_replenishment_period'] or ''} | "
            f"{float(row['horizon_hours']):.0f} | {row['failed_fields']} |"
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

    rows = [row_for_spec(spec) for spec in design_matrix()]
    summary = summary_for(rows)
    write_csv(out_dir / "thesis_design_matrix.csv", rows)
    (out_dir / "thesis_design_matrix.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_markdown(out_dir / "THESIS_DESIGN_MATRIX.md", rows, summary)
    print(out_dir / "THESIS_DESIGN_MATRIX.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
