#!/usr/bin/env python3
"""Collect dynamic-policy wins and near-misses from comparison runner outputs.

The comparison runner emits one ``policy_summary.csv`` per experiment.  This
script compares ``ppo_dynamic`` against the best static policy in the same
run/regime for several lenses and writes a compact registry.  It is deliberately
lightweight so we can run many 1-2 seed exploratory cells without losing the
interesting cases.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any


METRICS: dict[str, dict[str, Any]] = {
    "cd_sigmoid_mean": {
        "column": "cd_sigmoid_mean_mean",
        "direction": "max",
        "near": 0.01,
    },
    "excel_ret": {
        "column": "mean_ret_excel_formula_mean",
        "direction": "max",
        "near": 0.0001,
    },
    "flow_fill": {
        "column": "flow_fill_rate_mean",
        "direction": "max",
        "near": 0.01,
    },
    "service_loss_cvar95": {
        "column": "service_loss_cvar95_mean",
        "direction": "min",
        "near": 0.01,
    },
}

RESOURCE_COLUMNS = (
    "resource_composite_total_mean",
    "extra_shift_hours_total_mean",
    "strategic_buffer_target_units_mean_mean",
)


def _float(value: str | None) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def _isfinite(value: float) -> bool:
    return math.isfinite(float(value))


def _advantage(dynamic: float, static: float, direction: str) -> float:
    if direction == "min":
        return static - dynamic
    return dynamic - static


def _run_name(path: Path) -> str:
    try:
        return path.parent.name
    except Exception:
        return str(path)


def collect(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for policy_summary in sorted(root.glob("**/policy_summary.csv")):
        with policy_summary.open(newline="") as handle:
            records = list(csv.DictReader(handle))
        if not records:
            continue
        regimes = sorted({record.get("regime", "") for record in records})
        for regime in regimes:
            regime_records = [r for r in records if r.get("regime") == regime]
            dynamic = next(
                (r for r in regime_records if r.get("policy") == "ppo_dynamic"),
                None,
            )
            if dynamic is None:
                continue
            static_records = [
                r
                for r in regime_records
                if r.get("policy") != "ppo_dynamic"
                and not str(r.get("policy", "")).startswith("heuristic_")
            ]
            if not static_records:
                continue
            for metric_name, spec in METRICS.items():
                column = str(spec["column"])
                direction = str(spec["direction"])
                candidates = [
                    (r, _float(r.get(column)))
                    for r in static_records
                    if _isfinite(_float(r.get(column)))
                ]
                dynamic_value = _float(dynamic.get(column))
                if not candidates or not _isfinite(dynamic_value):
                    continue
                if direction == "min":
                    best_static, best_value = min(candidates, key=lambda item: item[1])
                else:
                    best_static, best_value = max(candidates, key=lambda item: item[1])
                advantage = _advantage(dynamic_value, best_value, direction)
                near = float(spec["near"])
                status = (
                    "win"
                    if advantage > 0.0
                    else "near_miss"
                    if advantage >= -near
                    else "loss"
                )
                if status == "loss":
                    continue
                out: dict[str, Any] = {
                    "status": status,
                    "metric": metric_name,
                    "direction": direction,
                    "run": _run_name(policy_summary),
                    "regime": regime,
                    "dynamic_policy": "ppo_dynamic",
                    "best_static_policy": best_static.get("policy", ""),
                    "dynamic_value": dynamic_value,
                    "best_static_value": best_value,
                    "advantage_positive_good": advantage,
                    "near_threshold": near,
                    "policy_summary": str(policy_summary),
                }
                for resource_col in RESOURCE_COLUMNS:
                    dynamic_resource = _float(dynamic.get(resource_col))
                    static_resource = _float(best_static.get(resource_col))
                    if _isfinite(dynamic_resource) and _isfinite(static_resource):
                        out[f"dynamic_{resource_col}"] = dynamic_resource
                        out[f"best_static_{resource_col}"] = static_resource
                        out[f"delta_{resource_col}"] = (
                            dynamic_resource - static_resource
                        )
                rows.append(out)
    rows.sort(
        key=lambda row: (
            0 if row["status"] == "win" else 1,
            row["metric"],
            -float(row["advantage_positive_good"]),
        )
    )
    return rows


def write_outputs(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    csv_path = output_dir / f"promising_dynamic_cases_{stamp}.csv"
    json_path = output_dir / f"promising_dynamic_cases_{stamp}.json"
    md_path = output_dir / f"promising_dynamic_cases_{stamp}.md"
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2))

    lines = [
        "# Promising Dynamic Cases",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "Status is computed against the best static policy in the same run/regime.",
        "Positive advantage is always good; `near_miss` means within the metric-specific tolerance.",
        "",
        "| status | metric | run | regime | best static | advantage | dynamic | static |",
        "|---|---|---|---|---|---:|---:|---:|",
    ]
    for row in rows[:80]:
        lines.append(
            "| {status} | {metric} | {run} | {regime} | {best_static_policy} | "
            "{advantage_positive_good:.6g} | {dynamic_value:.6g} | "
            "{best_static_value:.6g} |".format(**row)
        )
    md_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        action="append",
        default=[
            Path("outputs/benchmarks/garrido_dynamic_cd"),
            Path("outputs/calibration"),
        ],
        help="Root(s) to scan for policy_summary.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/diagnostics/promising_dynamic_cases"),
    )
    args = parser.parse_args()
    rows: list[dict[str, Any]] = []
    for root in args.root:
        rows.extend(collect(root))
    write_outputs(rows, args.output_dir)
    print(f"Collected {len(rows)} wins/near-misses.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
