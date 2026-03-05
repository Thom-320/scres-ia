#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from supply_chain.config import (
    DEFAULT_YEAR_BASIS,
    HOURS_PER_YEAR_GREGORIAN,
    HOURS_PER_YEAR_THESIS,
    VALIDATION_TABLE_6_10,
    WARMUP,
    YEAR_BASIS_OPTIONS,
)
from supply_chain.supply_chain import MFSCSimulation


def _run_basis_validation(
    year_basis: str, seed: int
) -> tuple[pd.DataFrame, float, float]:
    hours_per_year = (
        HOURS_PER_YEAR_THESIS if year_basis == "thesis" else HOURS_PER_YEAR_GREGORIAN
    )
    horizon_8_years = int(np.ceil(WARMUP["estimated_deterministic_hrs"])) + (
        8 * hours_per_year
    )
    sim = MFSCSimulation(
        shifts=1,
        risks_enabled=False,
        seed=seed,
        horizon=horizon_8_years,
        year_basis=year_basis,
        deterministic_baseline=True,
    ).run()
    throughput = sim.get_annual_throughput(start_time=sim.warmup_time, num_years=8)
    sim_yearly = throughput["produced_by_year"]

    years = VALIDATION_TABLE_6_10["years"]
    ecs_thesis = VALIDATION_TABLE_6_10["ECS_simulated"]

    rows = []
    sq_errors = []
    for year, ecs in zip(years, ecs_thesis):
        sim_val = sim_yearly.get(year, 0)
        error = sim_val - ecs
        rel_error = error / ecs if ecs > 0 else 0
        sq_errors.append(error**2)
        rows.append(
            {
                "Year": year,
                "Thesis_ECS": ecs,
                "Our_Model": sim_val,
                "Abs_Error": error,
                "Rel_Error": rel_error,
                "year_basis": year_basis,
            }
        )

    rmse = float(np.sqrt(np.mean(sq_errors)))
    avg_delivery = float(throughput["avg_annual_delivery"])
    return pd.DataFrame(rows), rmse, avg_delivery


def generate_validation_report(
    output_dir: Path, seed: int, official_basis: str
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    official_df, official_rmse, official_avg = _run_basis_validation(
        official_basis, seed
    )
    dual_basis = [basis for basis in YEAR_BASIS_OPTIONS if basis != official_basis][0]
    dual_df, dual_rmse, dual_avg = _run_basis_validation(dual_basis, seed)

    combined_df = pd.concat([official_df, dual_df], ignore_index=True)
    csv_path = output_dir / "validation_table_dual_basis.csv"
    combined_df.to_csv(csv_path, index=False)

    thesis_avg = float(np.mean(VALIDATION_TABLE_6_10["ECS_simulated"]))
    official_diff = (official_avg - thesis_avg) / thesis_avg
    dual_diff = (dual_avg - thesis_avg) / thesis_avg

    print("=" * 78)
    print("  VALIDATION REPORT (Cf0): post-warm-up deterministic comparison")
    print("=" * 78)
    print(f"Official comparison basis: {official_basis}")
    print("Method: deterministic dispatch/demand, 8 full years after warm-up")
    print(f"Official RMSE: {official_rmse:,.0f} (Thesis baseline RMSE: 87,918)")
    print(
        f"Official avg annual delivery: {official_avg:,.0f} ({official_diff:+.2%} vs thesis avg)"
    )
    print("-" * 78)
    print(f"Secondary basis diagnostic: {dual_basis}")
    print(f"Secondary RMSE: {dual_rmse:,.0f}")
    print(
        f"Secondary avg annual delivery: {dual_avg:,.0f} ({dual_diff:+.2%} vs thesis avg)"
    )
    print("-" * 78)
    print(f"CSV saved to: {csv_path}")
    print("=" * 78)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate MFSC validation report.")
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for deterministic run."
    )
    parser.add_argument(
        "--official-basis",
        choices=YEAR_BASIS_OPTIONS,
        default=DEFAULT_YEAR_BASIS,
        help="Official basis used for thesis comparison.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/validation"),
        help="Directory for generated CSV artifacts.",
    )
    return parser


if __name__ == "__main__":
    cli_args = build_parser().parse_args()
    generate_validation_report(
        output_dir=cli_args.output_dir,
        seed=cli_args.seed,
        official_basis=cli_args.official_basis,
    )
