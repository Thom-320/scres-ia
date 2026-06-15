#!/usr/bin/env python3
"""Report Table 6.11 risk-frequency fidelity.

Table 6.11 reports expected current-risk frequencies per year and per 20-year
simulation run. This audit compares those thesis targets against the current
DES process semantics. It is intentionally allowed to report FAIL: a failure is
evidence of a remaining thesis-fidelity gap, not a test harness error.
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

from supply_chain.config import (  # noqa: E402
    DAYS_PER_WEEK,
    HOURS_PER_DAY,
    HOURS_PER_WEEK,
    HOURS_PER_YEAR_THESIS,
    OPERATIONS,
    RISKS_CURRENT,
    SIMULATION_HORIZON,
)

DEFAULT_OUTPUT_ROOT = Path("outputs/benchmarks/thesis_risk_frequency")
RUN_YEARS = SIMULATION_HORIZON / HOURS_PER_YEAR_THESIS

EXPECTED_TABLE_6_11 = {
    "R11": {"unit": "Breakdowns", "events_per_year": 48.0, "events_per_run": 960.0},
    "R12": {
        "unit": "Delayed contracts",
        "events_per_year": 2.0 + 1.0 / 6.0,
        "events_per_run": 44.0,
    },
    "R13": {
        "unit": "Delayed deliveries",
        "events_per_year": 58.0,
        "events_per_run": 1152.0,
    },
    "R14": {
        "unit": "Defective products",
        "events_per_year": 22153.0,
        "events_per_run": 443059.0,
    },
    "R21": {
        "unit": "Natural disasters",
        "events_per_year": 0.5,
        "events_per_run": 10.0,
    },
    "R22": {"unit": "Attacks", "events_per_year": 2.0, "events_per_run": 40.0},
    "R23": {"unit": "Attacks", "events_per_year": 1.0, "events_per_run": 20.0},
    "R24": {
        "unit": "Contingent orders",
        "events_per_year": 12.0,
        "events_per_run": 240.0,
    },
    "R3": {
        "unit": "Destructive attacks",
        "events_per_year": 1.0 / 20.0,
        "events_per_run": 1.0,
    },
}


def annual_uniform_renewal_frequency(risk_id: str) -> float:
    """Expected events/year for current renewal-style uniform inter-arrivals."""
    occurrence = RISKS_CURRENT[risk_id]["occurrence"]
    mean_interval = (float(occurrence["a"]) + float(occurrence["b"])) / 2.0
    return HOURS_PER_YEAR_THESIS / mean_interval


def implementation_estimate(risk_id: str) -> float:
    """Expected current implementation events/year under current code semantics."""
    occurrence = RISKS_CURRENT[risk_id]["occurrence"]
    if occurrence["dist"] == "uniform":
        return annual_uniform_renewal_frequency(risk_id)

    n = float(occurrence["n"])
    p = float(occurrence["p"])
    if risk_id == "R12":
        cycles_per_year = HOURS_PER_YEAR_THESIS / float(OPERATIONS[1]["rop"])
        return cycles_per_year * n * p
    if risk_id == "R13":
        cycles_per_year = HOURS_PER_YEAR_THESIS / float(OPERATIONS[2]["rop"])
        return cycles_per_year * n * p
    if risk_id == "R14":
        operating_days_per_year = HOURS_PER_YEAR_THESIS / HOURS_PER_WEEK * DAYS_PER_WEEK
        return operating_days_per_year * n * p

    raise ValueError(f"Unsupported risk for Table 6.11 estimate: {risk_id}")


def thesis_table_frequency_from_distribution(risk_id: str) -> float:
    """Frequency implied by Table 6.11's period-window interpretation."""
    occurrence = RISKS_CURRENT[risk_id]["occurrence"]
    if occurrence["dist"] == "uniform":
        return HOURS_PER_YEAR_THESIS / float(occurrence["b"])
    return EXPECTED_TABLE_6_11[risk_id]["events_per_year"]


def matches_table(actual_run: float, expected_run: float) -> bool:
    tolerance = max(0.5, abs(expected_run) * 0.02)
    return math.isclose(actual_run, expected_run, rel_tol=0.0, abs_tol=tolerance)


def rows() -> list[dict[str, Any]]:
    output = []
    for risk_id, expected in EXPECTED_TABLE_6_11.items():
        implementation_per_year = implementation_estimate(risk_id)
        implementation_per_run = implementation_per_year * RUN_YEARS
        table_formula_per_year = thesis_table_frequency_from_distribution(risk_id)
        table_formula_per_run = table_formula_per_year * RUN_YEARS
        status = (
            "MATCH"
            if matches_table(implementation_per_run, expected["events_per_run"])
            else "MISMATCH"
        )
        output.append(
            {
                "risk_id": risk_id,
                "unit": expected["unit"],
                "table_events_per_year": float(expected["events_per_year"]),
                "table_events_per_run": float(expected["events_per_run"]),
                "table_formula_events_per_year": float(table_formula_per_year),
                "table_formula_events_per_run": float(table_formula_per_run),
                "implementation_events_per_year": float(implementation_per_year),
                "implementation_events_per_run": float(implementation_per_run),
                "status": status,
            }
        )
    return output


def write_csv(path: Path, risk_rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(risk_rows[0].keys()))
        writer.writeheader()
        writer.writerows(risk_rows)


def write_markdown(path: Path, risk_rows: list[dict[str, Any]]) -> None:
    status = "PASS" if all(row["status"] == "MATCH" for row in risk_rows) else "FAIL"
    lines = [
        "# Thesis Risk Frequency Report",
        "",
        f"Created UTC: `{datetime.now(timezone.utc).isoformat()}`",
        f"Overall status: `{status}`",
        "",
        "Table 6.11 is interpreted as a frequency gate. A `FAIL` here means the",
        "current DES process semantics do not reproduce the table frequency, even",
        "if the raw Table 6.12 distribution constants are present.",
        "",
        "| risk | unit | table/year | impl/year | table/run | impl/run | status |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in risk_rows:
        lines.append(
            f"| {row['risk_id']} | {row['unit']} | "
            f"{row['table_events_per_year']:.4f} | "
            f"{row['implementation_events_per_year']:.4f} | "
            f"{row['table_events_per_run']:.2f} | "
            f"{row['implementation_events_per_run']:.2f} | {row['status']} |"
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
    risk_rows = rows()
    status = "PASS" if all(row["status"] == "MATCH" for row in risk_rows) else "FAIL"

    write_csv(out_dir / "thesis_risk_frequency.csv", risk_rows)
    (out_dir / "thesis_risk_frequency.json").write_text(
        json.dumps({"status": status, "rows": risk_rows}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_markdown(out_dir / "THESIS_RISK_FREQUENCY.md", risk_rows)
    print(out_dir / "THESIS_RISK_FREQUENCY.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
