#!/usr/bin/env python3
"""Report thesis SDM schema and ReT Eq. 5.5 cases from current code.

This is a narrow Garrido-Rios fidelity gate for the reporting layer:

- Table 6.25 Simulation Data Matrix (SDM) columns.
- Figure 5.6 / Eq. 5.1-5.5 ReT weights and case selection.

It compares the current implementation against independently listed thesis
targets and writes CSV/JSON/Markdown artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.config import (
    OUTPUT_COLUMNS,
    RET_RE_MAX,
    RET_RE_MIN,
    RET_RE_RECOVERY,
    THESIS_FAITHFUL_PROTOCOL,
)
from supply_chain.ret_thesis import compute_ret_per_order

DEFAULT_OUTPUT_ROOT = Path("outputs/benchmarks/thesis_ret_schema")

EXPECTED_OUTPUT_COLUMNS = [
    "Cfi",
    "j",
    "OPTj",
    "OATj",
    "CTj",
    "LTj",
    "Bt",
    "Ut",
    "APj",
    "RPj",
    "DPj",
    "Rcr_Op",
]

EXPECTED_RET_WEIGHTS = {
    "max": 1.0,
    "mean": 0.5,
    "min": 0.0,
}


@dataclass(frozen=True)
class SyntheticOrder:
    j: int
    OPTj: float
    OATj: float | None
    CTj: float | None
    LTj: float
    APj: float = 0.0
    RPj: float = 0.0
    DPj: float = 0.0


RET_CASES = [
    {
        "case": "fill_rate",
        "order": SyntheticOrder(j=1, OPTj=0.0, OATj=40.0, CTj=40.0, LTj=48.0),
        "fill_rate": 0.75,
        "expected_ret": 0.75,
        "expected_case": "fill_rate",
    },
    {
        "case": "autotomy",
        "order": SyntheticOrder(j=2, OPTj=0.0, OATj=40.0, CTj=40.0, LTj=48.0, APj=24.0),
        "fill_rate": 0.75,
        "expected_ret": 0.5,
        "expected_case": "autotomy",
    },
    {
        "case": "recovery",
        "order": SyntheticOrder(j=3, OPTj=0.0, OATj=72.0, CTj=72.0, LTj=48.0, RPj=24.0),
        "fill_rate": 0.75,
        "expected_ret": 0.5 / 24.0,
        "expected_case": "recovery",
    },
    {
        "case": "non_recovery",
        "order": SyntheticOrder(j=4, OPTj=0.0, OATj=80.0, CTj=80.0, LTj=48.0, DPj=80.0),
        "fill_rate": 0.75,
        "expected_ret": 0.0,
        "expected_case": "non_recovery",
    },
    {
        "case": "unfulfilled",
        "order": SyntheticOrder(j=5, OPTj=0.0, OATj=None, CTj=None, LTj=48.0),
        "fill_rate": 0.75,
        "expected_ret": 0.0,
        "expected_case": "unfulfilled",
    },
]


def status_for(actual: Any, expected: Any) -> str:
    if isinstance(actual, float) or isinstance(expected, float):
        return (
            "MATCH"
            if math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12)
            else "MISMATCH"
        )
    return "MATCH" if actual == expected else "MISMATCH"


def schema_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    max_len = max(len(OUTPUT_COLUMNS), len(EXPECTED_OUTPUT_COLUMNS))
    for index in range(max_len):
        actual = OUTPUT_COLUMNS[index] if index < len(OUTPUT_COLUMNS) else ""
        expected = (
            EXPECTED_OUTPUT_COLUMNS[index]
            if index < len(EXPECTED_OUTPUT_COLUMNS)
            else ""
        )
        rows.append(
            {
                "section": "sdm_schema",
                "item": f"column_{index + 1}",
                "expected": expected,
                "actual": actual,
                "status": status_for(actual, expected),
            }
        )
    return rows


def weight_rows() -> list[dict[str, Any]]:
    actuals = {
        "max": RET_RE_MAX,
        "mean": RET_RE_RECOVERY,
        "min": RET_RE_MIN,
        "protocol_max": THESIS_FAITHFUL_PROTOCOL["ret_weights"]["max"],
        "protocol_mean": THESIS_FAITHFUL_PROTOCOL["ret_weights"]["mean"],
        "protocol_min": THESIS_FAITHFUL_PROTOCOL["ret_weights"]["min"],
    }
    expected = {
        **EXPECTED_RET_WEIGHTS,
        "protocol_max": EXPECTED_RET_WEIGHTS["max"],
        "protocol_mean": EXPECTED_RET_WEIGHTS["mean"],
        "protocol_min": EXPECTED_RET_WEIGHTS["min"],
    }
    return [
        {
            "section": "ret_weights",
            "item": key,
            "expected": expected[key],
            "actual": actuals[key],
            "status": status_for(actuals[key], expected[key]),
        }
        for key in actuals
    ]


def ret_case_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in RET_CASES:
        actual_ret, actual_case = compute_ret_per_order(
            case["order"],
            fill_rate=case["fill_rate"],
            ret_weights=THESIS_FAITHFUL_PROTOCOL["ret_weights"],
        )
        status = (
            "MATCH"
            if actual_case == case["expected_case"]
            and math.isclose(
                actual_ret,
                float(case["expected_ret"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            else "MISMATCH"
        )
        rows.append(
            {
                "section": "ret_cases",
                "item": case["case"],
                "expected": f"{case['expected_case']}:{case['expected_ret']:.12g}",
                "actual": f"{actual_case}:{actual_ret:.12g}",
                "status": status,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    status = "PASS" if all(row["status"] == "MATCH" for row in rows) else "FAIL"
    lines = [
        "# Thesis ReT and SDM Schema Report",
        "",
        f"Created UTC: `{datetime.now(timezone.utc).isoformat()}`",
        f"Overall status: `{status}`",
        "",
        "## Summary",
        "",
        "| section | rows | mismatches |",
        "|---|---:|---:|",
    ]
    for section in sorted({row["section"] for row in rows}):
        section_rows = [row for row in rows if row["section"] == section]
        mismatches = sum(row["status"] != "MATCH" for row in section_rows)
        lines.append(f"| {section} | {len(section_rows)} | {mismatches} |")

    lines += [
        "",
        "## Rows",
        "",
        "| section | item | expected | actual | status |",
        "|---|---|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['section']} | {row['item']} | {row['expected']} | "
            f"{row['actual']} | {row['status']} |"
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
    rows = schema_rows() + weight_rows() + ret_case_rows()
    status = "PASS" if all(row["status"] == "MATCH" for row in rows) else "FAIL"

    write_csv(out_dir / "thesis_ret_schema.csv", rows)
    (out_dir / "thesis_ret_schema.json").write_text(
        json.dumps({"status": status, "rows": rows}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_markdown(out_dir / "THESIS_RET_SCHEMA.md", rows)
    print(out_dir / "THESIS_RET_SCHEMA.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
