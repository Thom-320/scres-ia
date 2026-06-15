#!/usr/bin/env python3
"""Report Table 6.12 risk-distribution constants from the current code."""

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

from supply_chain.config import RISKS_CURRENT, RISKS_INCREASED

DEFAULT_OUTPUT_ROOT = Path("outputs/benchmarks/thesis_risk_tables")

EXPECTED_CURRENT = {
    "R11": {"occurrence": "U(1,168)", "recovery": "Exp(mean=2)", "ops": [5, 6]},
    "R12": {"occurrence": "B(n=12,p=1/11)", "recovery": "", "ops": [1]},
    "R13": {"occurrence": "B(n=12,p=1/10)", "recovery": "", "ops": [2]},
    "R14": {"occurrence": "B(n=2564,p=3/100)", "recovery": "", "ops": [7]},
    "R21": {
        "occurrence": "U(1,16128)",
        "recovery": "Exp(mean=120)",
        "ops": [3, 5, 6, 7, 9],
    },
    "R22": {
        "occurrence": "U(1,4032)",
        "recovery": "Exp(mean=24)",
        "ops": [4, 8, 10, 12],
    },
    "R23": {"occurrence": "U(1,8064)", "recovery": "Exp(mean=120)", "ops": [11]},
    "R24": {"occurrence": "U(1,672)", "recovery": "surge=U(2400,2600)", "ops": [13]},
    "R3": {"occurrence": "U(1,161280)", "recovery": "fixed=672", "ops": [5, 6, 7, 9]},
}

EXPECTED_INCREASED = {
    "R11": "U(1,42)",
    "R12": "B(n=12,p=4/11)",
    "R13": "B(n=12,p=4/10)",
    "R14": "B(n=2564,p=8/100)",
    "R21": "U(1,4032)",
    "R22": "U(1,1344)",
    "R23": "U(1,1344)",
    "R24": "U(1,336)",
    "R3": "U(1,80640)",
}


def format_occurrence(spec: dict[str, Any]) -> str:
    dist = spec["dist"]
    if dist == "uniform":
        return f"U({int(spec['a'])},{int(spec['b'])})"
    if dist == "binomial":
        return f"B(n={int(spec['n'])},p={format_probability(float(spec['p']))})"
    raise ValueError(f"Unsupported distribution {dist!r}")


def format_probability(value: float) -> str:
    known = {
        1 / 11: "1/11",
        4 / 11: "4/11",
        8 / 11: "8/11",
        1 / 10: "1/10",
        4 / 10: "4/10",
        8 / 10: "8/10",
        3 / 100: "3/100",
        8 / 100: "8/100",
        12 / 100: "12/100",
    }
    for expected, label in known.items():
        if math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-12):
            return label
    return f"{value:.12g}"


def format_recovery_or_surge(risk_id: str, spec: dict[str, Any]) -> str:
    if "recovery" in spec:
        recovery = spec["recovery"]
        if recovery["dist"] == "exponential":
            return f"Exp(mean={int(recovery['mean'])})"
        if recovery["dist"] == "fixed":
            return f"fixed={int(recovery['duration'])}"
        raise ValueError(f"Unsupported recovery distribution {recovery['dist']!r}")
    if risk_id == "R24":
        surge = spec["surge"]
        return f"surge=U({int(surge['lo'])},{int(surge['hi'])})"
    return ""


def current_rows() -> list[dict[str, Any]]:
    rows = []
    for risk_id, expected in EXPECTED_CURRENT.items():
        actual = RISKS_CURRENT[risk_id]
        actual_occurrence = format_occurrence(actual["occurrence"])
        actual_recovery = format_recovery_or_surge(risk_id, actual)
        actual_ops = list(actual["affected_ops"])
        rows.append(
            {
                "level": "current",
                "risk_id": risk_id,
                "expected_occurrence": expected["occurrence"],
                "actual_occurrence": actual_occurrence,
                "expected_recovery": expected["recovery"],
                "actual_recovery": actual_recovery,
                "expected_ops": json.dumps(expected["ops"]),
                "actual_ops": json.dumps(actual_ops),
                "status": (
                    "MATCH"
                    if actual_occurrence == expected["occurrence"]
                    and actual_recovery == expected["recovery"]
                    and actual_ops == expected["ops"]
                    else "MISMATCH"
                ),
            }
        )
    return rows


def increased_rows() -> list[dict[str, Any]]:
    rows = []
    for risk_id, expected_occurrence in EXPECTED_INCREASED.items():
        actual_occurrence = format_occurrence(RISKS_INCREASED[risk_id])
        rows.append(
            {
                "level": "increased",
                "risk_id": risk_id,
                "expected_occurrence": expected_occurrence,
                "actual_occurrence": actual_occurrence,
                "expected_recovery": "",
                "actual_recovery": "",
                "expected_ops": "",
                "actual_ops": "",
                "status": (
                    "MATCH" if actual_occurrence == expected_occurrence else "MISMATCH"
                ),
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
        "# Thesis Risk Tables Report",
        "",
        f"Created UTC: `{datetime.now(timezone.utc).isoformat()}`",
        f"Overall status: `{status}`",
        "",
        "## Table 6.12 Current Risk Level",
        "",
        "| risk | expected occurrence | actual occurrence | expected recovery/surge | actual recovery/surge | ops | status |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        if row["level"] != "current":
            continue
        lines.append(
            f"| {row['risk_id']} | {row['expected_occurrence']} | "
            f"{row['actual_occurrence']} | {row['expected_recovery']} | "
            f"{row['actual_recovery']} | {row['actual_ops']} | {row['status']} |"
        )
    lines += [
        "",
        "## Table 6.12 Increased Risk Level",
        "",
        "| risk | expected occurrence | actual occurrence | status |",
        "|---|---|---|---|",
    ]
    for row in rows:
        if row["level"] != "increased":
            continue
        lines.append(
            f"| {row['risk_id']} | {row['expected_occurrence']} | "
            f"{row['actual_occurrence']} | {row['status']} |"
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
    rows = current_rows() + increased_rows()
    payload = {
        "status": "PASS" if all(row["status"] == "MATCH" for row in rows) else "FAIL",
        "rows": rows,
    }
    write_csv(out_dir / "thesis_risk_tables.csv", rows)
    (out_dir / "thesis_risk_tables.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_markdown(out_dir / "THESIS_RISK_TABLES.md", rows)
    print(out_dir / "THESIS_RISK_TABLES.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
