#!/usr/bin/env python3
"""Compute the EXACT thesis order-level ReT (Garrido-Rios 2017, Eq. 5.1-5.5).

Runs the faithful DES (thesis_window risk scheduling, thesis year basis, op9_arrival
warm-up, thesis_strict_op6 defects, repaired raw-material mode, figure_6_2 downstream,
ReT weights {max:1.0, mean:0.5, min:0.0}) over the full 20-year horizon for every thesis
static decision: the 6 strategic-inventory levels (Table 6.16) x 3 shift levels
(Table 6.20). For each it reports the exact mean ReT and the per-case breakdown
(autotomy / recovery / non_recovery / fill_rate / unfulfilled), averaged over seeds.

This is the faithful EVALUATION metric (not a training-reward approximation).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.supply_chain import MFSCSimulation, SIMULATION_HORIZON  # noqa: E402
from supply_chain.config import (  # noqa: E402
    INVENTORY_BUFFERS,
    THESIS_FAITHFUL_PROTOCOL as P,
    THESIS_REPLICATION_DOWNSTREAM_Q_SOURCE as DQ,
)

INVENTORY_LEVELS = [0, 168, 336, 504, 672, 1344]   # I0 .. I1344 (Table 6.16)
CASES = ["autotomy", "recovery", "non_recovery", "fill_rate", "unfulfilled"]


def faithful_sim(shifts: int, period: int, seed: int, risk_level: str) -> MFSCSimulation:
    bufs = dict(INVENTORY_BUFFERS[period]) if period else None
    return MFSCSimulation(
        shifts=shifts, seed=seed, horizon=SIMULATION_HORIZON,
        risks_enabled=True, risk_level=risk_level,
        risk_occurrence_mode="thesis_window",
        year_basis=P["year_basis"], warmup_trigger=P["warmup_trigger"],
        r14_defect_mode=P["r14_defect_mode"], downstream_q_source=DQ,
        raw_material_flow_mode=P["raw_material_flow_mode"],
        raw_material_order_up_to_multiplier=P["raw_material_order_up_to_multiplier"],
        initial_buffers=bufs,
        inventory_replenishment_period=(float(period) if period else None),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default="1,2,3,4,5")
    ap.add_argument("--risk-levels", default="current")
    ap.add_argument("--output", type=Path,
                    default=Path("outputs/benchmarks/exact_thesis_ret/exact_ret.json"))
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    risks = [r.strip() for r in args.risk_levels.split(",") if r.strip()]

    results: dict = {"protocol": "thesis_faithful (thesis_window, figure_6_2, weights 1/0.5/0)",
                     "horizon_hours": SIMULATION_HORIZON, "seeds": seeds, "cells": []}
    for risk in risks:
        print(f"\n=== ReT EXACTA (Eq. 5.5) — riesgo={risk} — horizonte 20 años, {len(seeds)} semillas ===")
        print(f"{'config':14} {'ReT':>8} {'fill':>7}  {'casos (aut/rec/nonrec/fill/unfull)':>34}")
        for period in INVENTORY_LEVELS:
            for S in (1, 2, 3):
                rets, fills = [], []
                cc = {k: 0 for k in CASES}
                for seed in seeds:
                    sim = faithful_sim(S, period, seed, risk)
                    sim.run()
                    r = sim.compute_order_level_ret()
                    rets.append(r["mean_ret"]); fills.append(r["fill_rate_order_level"])
                    for k in CASES:
                        cc[k] += r["case_counts"][k]
                n = len(seeds)
                name = f"I{period}_S{S}"
                cell = {
                    "config": name, "risk_level": risk, "inventory_period": period, "shift": S,
                    "ret_mean": float(np.mean(rets)), "ret_sd": float(np.std(rets)),
                    "fill_rate": float(np.mean(fills)),
                    "case_counts_per_seed": {k: cc[k] / n for k in CASES},
                }
                results["cells"].append(cell)
                cb = "/".join(str(int(cc[k] / n)) for k in CASES)
                print(f"{name:14} {cell['ret_mean']:8.4f} {cell['fill_rate']:7.4f}  {cb:>34}")
        best = max((c for c in results["cells"] if c["risk_level"] == risk), key=lambda c: c["ret_mean"])
        print(f"  -> mejor config (riesgo={risk}): {best['config']} con ReT={best['ret_mean']:.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
