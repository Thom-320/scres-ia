#!/usr/bin/env python3
"""DES-side CTj/RPj/DPj/APj quantiles by risk family (endogenous lane).

Runs the thesis-faithful DES with risk_attribution_source='des_events' (the
diverging lane) for the R1 family (R11-R14) and the R2 family (R21-R24),
endogenous risk generation, delay=54h, and pools per-order CTj/RPj/DPj/APj to
compare against the Garrido Excel targets (see excel_quantiles_by_family.py).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.supply_chain import MFSCSimulation, SIMULATION_HORIZON  # noqa: E402
from supply_chain.config import (  # noqa: E402
    THESIS_FAITHFUL_PROTOCOL as P,
    THESIS_REPLICATION_DOWNSTREAM_Q_SOURCE as DQ,
)
from supply_chain.thesis_design import R1_RISKS, R2_RISKS  # noqa: E402

FAMILIES = {"R1": set(R1_RISKS), "R2": set(R2_RISKS)}


def make_sim(enabled_risks, seed, rp_mode, overflow_mode):
    return MFSCSimulation(
        shifts=1, seed=seed, horizon=SIMULATION_HORIZON,
        risks_enabled=True, risk_level="current",
        enabled_risks=enabled_risks,
        risk_occurrence_mode="thesis_window",
        risk_attribution_source="des_events",
        ret_recovery_period_mode=rp_mode,
        backorder_overflow_mode=overflow_mode,
        year_basis=P["year_basis"], warmup_trigger=P["warmup_trigger"],
        r14_defect_mode=P["r14_defect_mode"], downstream_q_source=DQ,
        raw_material_flow_mode=P["raw_material_flow_mode"],
        raw_material_order_up_to_multiplier=P["raw_material_order_up_to_multiplier"],
        demand_on_hand_fulfillment_delay=P["demand_on_hand_fulfillment_delay"],
    )


def quantiles(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0}
    out = {"n": int(arr.size), "mean": float(arr.mean())}
    for q in (50, 90, 95, 99):
        out[f"p{q}"] = float(np.percentile(arr, q))
    out["max"] = float(arr.max())
    return out


def run_family(name, enabled, seeds, rp_mode, overflow_mode):
    pools = {c: [] for c in ("CTj", "RPj", "DPj", "APj")}
    n_lost = 0
    n_served = 0
    for seed in seeds:
        sim = make_sim(enabled, seed, rp_mode, overflow_mode)
        sim.run()
        for o in sim.orders:
            if getattr(o, "metrics_excluded", False):
                continue
            if getattr(o, "lost", False):
                n_lost += 1
                continue
            if o.CTj is None or o.OATj is None:
                continue
            n_served += 1
            pools["CTj"].append(float(o.CTj))
            pools["APj"].append(float(getattr(o, "APj", 0.0) or 0.0))
            pools["RPj"].append(float(getattr(o, "RPj", 0.0) or 0.0))
            pools["DPj"].append(float(getattr(o, "DPj", 0.0) or 0.0))
    q = {c: quantiles(pools[c]) for c in pools}
    tag = f"{name} rp={rp_mode} overflow={overflow_mode}"
    print(f"\n=== DES {tag} (des_events, delay=54) served={n_served} lost={n_lost} ===")
    for c in ("CTj", "RPj", "DPj", "APj"):
        qq = q[c]
        if qq.get("n"):
            print(f"  {c:4} n={qq['n']:6d} mean={qq['mean']:9.1f} "
                  f"p50={qq['p50']:9.1f} p90={qq['p90']:9.1f} "
                  f"p95={qq['p95']:9.1f} p99={qq['p99']:9.1f} max={qq['max']:10.1f}")
    return {"served": n_served, "lost": n_lost, "quantiles": q}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--modes", default="disruption/oldest,disruption/largest,elapsed/largest",
                    help="comma-separated rp_mode/overflow_mode combos to A/B test")
    ap.add_argument("--output", type=Path,
                    default=Path("sandbox/results/des_quantiles_by_family.json"))
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    combos = [c.strip() for c in args.modes.split(",") if c.strip()]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    result = {}
    for combo in combos:
        rp_mode, overflow_mode = combo.split("/")
        result[combo] = {}
        for fam, enabled in FAMILIES.items():
            result[combo][fam] = run_family(fam, enabled, seeds, rp_mode, overflow_mode)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nSaved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
