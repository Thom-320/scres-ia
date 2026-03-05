#!/usr/bin/env python3
"""
run_static.py — Run MFSC simulation and validate.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from supply_chain.config import (
    DEFAULT_YEAR_BASIS,
    SIMULATION_HORIZON,
    VALIDATION_TABLE_6_10,
    YEAR_BASIS_OPTIONS,
)
from supply_chain.supply_chain import MFSCSimulation


def run_deterministic(year_basis: str) -> MFSCSimulation:
    print("=" * 60)
    print("  PHASE 1: Deterministic Baseline (Cf0)")
    print("=" * 60)

    t0 = time.time()
    sim = MFSCSimulation(
        shifts=1,
        risks_enabled=False,
        seed=42,
        horizon=SIMULATION_HORIZON,
        year_basis=year_basis,
        deterministic_baseline=True,
    ).run()
    elapsed = time.time() - t0

    print(f"  Completed in {elapsed:.2f}s")
    sim.summary()

    throughput = sim.get_annual_throughput(start_time=sim.warmup_time)
    thesis_avg = np.mean(VALIDATION_TABLE_6_10["ECS_simulated"])
    our_avg = throughput["avg_annual_delivery"]
    diff = (our_avg - thesis_avg) / thesis_avg

    print(
        f"\n  Validation ({year_basis}, post-warm-up): Our avg = {our_avg:,.0f} vs Thesis ECS ="
        f" {thesis_avg:,.0f} ({diff:+.1%})"
    )
    if abs(diff) < 0.15:
        print("  ✅ PHASE 1 PASSED")
    else:
        print("  ❌ PHASE 1 FAILED — diff exceeds ±15%")

    return sim


def run_stochastic(seed: int, year_basis: str) -> MFSCSimulation:
    print("\n" + "=" * 60)
    print("  PHASE 2: Stochastic (Current Risk Levels)")
    print("=" * 60)

    t0 = time.time()
    sim = MFSCSimulation(
        shifts=1,
        risks_enabled=True,
        risk_level="current",
        seed=seed,
        horizon=SIMULATION_HORIZON,
        year_basis=year_basis,
    ).run()
    elapsed = time.time() - t0

    print(f"  Completed in {elapsed:.2f}s")
    sim.summary()

    throughput = sim.get_annual_throughput()
    print("\n  Year-by-year production:")
    for year, qty in sorted(throughput["produced_by_year"].items()):
        if year <= 20:
            print(f"    Year {year:>2}: {qty:>10,} rations")
    return sim


def run_comparison(seed: int, year_basis: str) -> None:
    det = run_deterministic(year_basis=year_basis)
    sto = run_stochastic(seed=seed, year_basis=year_basis)

    det_t = det.get_annual_throughput()
    sto_t = sto.get_annual_throughput()

    print("\n" + "=" * 60)
    print("  COMPARISON: Deterministic vs Stochastic")
    print("=" * 60)

    metrics = [
        (
            "Avg annual production",
            det_t["avg_annual_production"],
            sto_t["avg_annual_production"],
        ),
        (
            "Avg annual delivery",
            det_t["avg_annual_delivery"],
            sto_t["avg_annual_delivery"],
        ),
        ("Fill rate", det._fill_rate(), sto._fill_rate()),
        ("Total backorders", det.total_backorders, sto.total_backorders),
        ("Risk events", 0, len(sto.risk_events)),
    ]

    print(f"  {'Metric':<25} {'Deterministic':>15} {'Stochastic':>15} {'Impact':>10}")
    print(f"  {'-' * 67}")
    for name, d_val, s_val in metrics:
        if isinstance(d_val, float) and d_val < 1:
            print(f"  {name:<25} {d_val:>15.1%} {s_val:>15.1%} {s_val - d_val:>+10.1%}")
        else:
            diff = ((s_val - d_val) / d_val * 100) if d_val > 0 else 0
            print(f"  {name:<25} {d_val:>15,.0f} {s_val:>15,.0f} {diff:>+9.1f}%")

    print(f"{'=' * 60}\n")

    thesis_avg = np.mean(VALIDATION_TABLE_6_10["ECS_simulated"])
    print(f"  Thesis ECS average (S=1, with risks): {thesis_avg:,.0f}")
    print(
        f"  Our stochastic average:               {sto_t['avg_annual_delivery']:,.0f}"
    )
    diff_thesis = (sto_t["avg_annual_delivery"] - thesis_avg) / thesis_avg
    print(f"  Difference: {diff_thesis:+.1%}")

    if abs(diff_thesis) < 0.20:
        print("\n  ✅ PHASE 2 PLAUSIBLE — within ±20% of thesis ECS")
    else:
        print("\n  ⚠️  PHASE 2 NEEDS REVIEW — large deviation from thesis")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run MFSC deterministic/stochastic baselines."
    )
    parser.add_argument(
        "--det-only", action="store_true", help="Run deterministic baseline only."
    )
    parser.add_argument(
        "--sto-only", action="store_true", help="Run stochastic baseline only."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for stochastic run."
    )
    parser.add_argument(
        "--year-basis",
        choices=YEAR_BASIS_OPTIONS,
        default=DEFAULT_YEAR_BASIS,
        help="Annualization basis for throughput metrics.",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.det_only:
        run_deterministic(year_basis=args.year_basis)
    elif args.sto_only:
        run_stochastic(seed=args.seed, year_basis=args.year_basis)
    else:
        run_comparison(seed=args.seed, year_basis=args.year_basis)
