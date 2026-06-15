#!/usr/bin/env python3
"""Report deterministic Table 6.10 reproduction.

This is a narrow Garrido-fidelity artifact: it runs Cf0 deterministic production
and compares the year-by-year output against the thesis validation table.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.config import (
    RAW_MATERIAL_FLOW_MODE_OPTIONS,
    SIMULATION_HORIZON,
    THESIS_FAITHFUL_PROTOCOL,
    VALIDATION_TABLE_6_10,
)
from supply_chain.supply_chain import MFSCSimulation

DEFAULT_OUTPUT_ROOT = Path("outputs/benchmarks/table_6_10_reproduction")


def run_cf0_simulation(args: argparse.Namespace) -> MFSCSimulation:
    return MFSCSimulation(
        shifts=1,
        risks_enabled=False,
        seed=args.seed,
        horizon=SIMULATION_HORIZON,
        year_basis="thesis",
        deterministic_baseline=True,
        warmup_trigger="op9_arrival",
        downstream_q_source=THESIS_FAITHFUL_PROTOCOL["downstream_q_source"],
        r14_defect_mode=THESIS_FAITHFUL_PROTOCOL["r14_defect_mode"],
        raw_material_flow_mode=args.raw_material_flow_mode,
        raw_material_order_up_to_multiplier=args.raw_material_order_up_to_multiplier,
    ).run()


def comparison_rows(produced_by_year: dict[int, float]) -> list[dict[str, float]]:
    rows = []
    for index, year in enumerate(VALIDATION_TABLE_6_10["years"], start=1):
        produced = float(produced_by_year[index])
        observed = float(VALIDATION_TABLE_6_10["Pt_observed"][index - 1])
        ecs = float(VALIDATION_TABLE_6_10["ECS_simulated"][index - 1])
        rows.append(
            {
                "year": float(year),
                "pt_observed": observed,
                "thesis_ecs_simulated": ecs,
                "python_produced": produced,
                "python_minus_ecs": produced - ecs,
                "python_minus_observed": produced - observed,
                "python_over_ecs": produced / ecs,
                "python_over_observed": produced / observed,
            }
        )
    return rows


def rmse(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values) / len(values))


def summarize(
    rows: list[dict[str, float]], throughput: dict[str, float]
) -> dict[str, float]:
    ecs_deltas = [row["python_minus_ecs"] for row in rows]
    observed_deltas = [row["python_minus_observed"] for row in rows]
    ecs_values = [row["thesis_ecs_simulated"] for row in rows]
    observed_values = [row["pt_observed"] for row in rows]
    produced_values = [row["python_produced"] for row in rows]
    return {
        "avg_python_production": float(np.mean(produced_values)),
        "avg_thesis_ecs_simulated": float(np.mean(ecs_values)),
        "avg_pt_observed": float(np.mean(observed_values)),
        "rmse_vs_thesis_ecs": rmse(ecs_deltas),
        "rmse_vs_pt_observed": rmse(observed_deltas),
        "thesis_reported_rmse": float(VALIDATION_TABLE_6_10["RMSE"]),
        "avg_relative_delta_vs_ecs": float(np.mean(ecs_deltas) / np.mean(ecs_values)),
        "avg_relative_delta_vs_observed": float(
            np.mean(observed_deltas) / np.mean(observed_values)
        ),
        "avg_annual_delivery": float(throughput["avg_annual_delivery"]),
        "hours_per_year": float(throughput["hours_per_year"]),
        "warmup_start_time": float(throughput["start_time"]),
    }


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(
    path: Path,
    *,
    args: argparse.Namespace,
    rows: list[dict[str, float]],
    summary: dict[str, float],
) -> None:
    lines = [
        "# Table 6.10 Reproduction",
        "",
        f"Created UTC: `{datetime.now(timezone.utc).isoformat()}`",
        f"Raw-material flow mode: `{args.raw_material_flow_mode}`",
        f"Order-up-to multiplier: `{args.raw_material_order_up_to_multiplier}`",
        f"Seed: `{args.seed}`",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for key, value in summary.items():
        lines.append(f"| {key} | {value:.4f} |")

    lines += [
        "",
        "## Year-by-Year Comparison",
        "",
        "| year | observed Pt | thesis ECS | Python produced | Python-ECS | Python/Pt | Python/ECS |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {int(row['year'])} | {row['pt_observed']:.0f} | "
            f"{row['thesis_ecs_simulated']:.0f} | {row['python_produced']:.0f} | "
            f"{row['python_minus_ecs']:.0f} | {row['python_over_observed']:.4f} | "
            f"{row['python_over_ecs']:.4f} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="kit_equivalent_order_up_to")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--raw-material-flow-mode",
        choices=RAW_MATERIAL_FLOW_MODE_OPTIONS,
        default="kit_equivalent_order_up_to",
    )
    parser.add_argument(
        "--raw-material-order-up-to-multiplier", type=float, default=2.0
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    out_dir = args.output_root / args.label
    out_dir.mkdir(parents=True, exist_ok=True)

    sim = run_cf0_simulation(args)
    throughput = sim.get_annual_throughput(start_time=sim.warmup_time, num_years=8)
    rows = comparison_rows(throughput["produced_by_year"])
    summary = summarize(rows, throughput)

    write_csv(out_dir / "table_6_10_comparison.csv", rows)
    payload_text = json.dumps(
        {"summary": summary, "rows": rows, "throughput": throughput},
        indent=2,
        sort_keys=True,
    )
    (out_dir / "table_6_10_comparison.json").write_text(payload_text, encoding="utf-8")
    write_markdown(
        out_dir / "TABLE_6_10_REPRODUCTION.md",
        args=args,
        rows=rows,
        summary=summary,
    )
    print(out_dir / "TABLE_6_10_REPRODUCTION.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
