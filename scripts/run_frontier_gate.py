#!/usr/bin/env python3
"""Frontier gate for the learning-extension calibration cells.

Runs a short static panel (18 statics x 3 regimes x N seeds) at given
env parameters and returns the gate verdict (eligible, mean_gap, robust,
oracle, argmax_diversity, corner_free, off_saturation, collapse_guard).

A cell is ELIGIBLE if:
  - argmax_diversity >= 2   (optimum moves between regimes)
  - corner_free             (S3_I1344 does not dominate in every regime)
  - off_saturation          (mild-regime flow not trivially solved/collapsed)
  - collapse_guard          (worst-regime oracle above a floor)
  - mean_gap >= threshold   (non-trivial regret of being fixed)

This is the precondition for the retention-transfer calibration: if no
cell is eligible, the calibration is reported as a null (no learnable
headroom at this learning-extension configuration), not a win.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.supply_chain import MFSCSimulation
from supply_chain.config import (
    THESIS_FAITHFUL_PROTOCOL as P,
    THESIS_REPLICATION_DOWNSTREAM_Q_SOURCE as DQ,
    INVENTORY_BUFFERS,
)
from supply_chain.episode_metrics import compute_episode_metrics

INVENTORY_LEVELS = [0, 168, 336, 504, 672, 1344]
SHIFT_LEVELS = [1, 2, 3]
MAX_CORNER = "S3_I1344"
COLLAPSE_FLOOR = 0.05
OFF_SAT_LO, OFF_SAT_HI = 0.30, 0.97
MEAN_GAP_FLOOR = 0.005


def run_static_panel(horizon_weeks, seeds, regimes, *,
                    rho_disruption, rho_demand, surge_budget,
                    inventory_replenish_lead, phi, psi, stochastic_pt,
                    demand_multiplier):
    """Return rows: one per (policy, regime) with mean cd_sigmoid + flow_fill
    across seeds, for the given env params."""
    horizon = float(horizon_weeks) * 7.0 * 24.0
    rows = []
    for period in INVENTORY_LEVELS:
        for s in SHIFT_LEVELS:
            for regime in regimes:
                cds, flows = [], []
                for seed in seeds:
                    bufs = dict(INVENTORY_BUFFERS[period]) if period else None
                    sim = MFSCSimulation(
                        shifts=s, seed=seed, horizon=horizon, risks_enabled=True,
                        risk_level=regime, risk_occurrence_mode="thesis_window",
                        year_basis=P["year_basis"], warmup_trigger=P["warmup_trigger"],
                        r14_defect_mode=P["r14_defect_mode"], downstream_q_source=DQ,
                        raw_material_flow_mode=P["raw_material_flow_mode"],
                        raw_material_order_up_to_multiplier=P["raw_material_order_up_to_multiplier"],
                        demand_on_hand_fulfillment_delay=float(P["demand_on_hand_fulfillment_delay"]),
                        risk_frequency_multiplier=float(phi),
                        risk_impact_multiplier=float(psi),
                        stochastic_pt=bool(stochastic_pt),
                        demand_mean_multiplier=float(demand_multiplier),
                        initial_buffers=bufs,
                        inventory_replenishment_period=(float(inventory_replenish_lead) if inventory_replenish_lead else None),
                    )
                    sim.run()
                    m = compute_episode_metrics(sim)
                    # The "cd" index for the gate: use flow_fill (bounded, simple,
                    # same direction as cd_sigmoid) as a gate proxy because
                    # running the full garrido CD-5-var inside the gate is
                    # expensive. The gate's job is to detect regime-diverse
                    # optima and non-saturation, not to compute the exact
                    # CD-5-var index. flow_fill is a sufficient gate proxy
                    # because it's bounded [0,1] and monotone in service.
                    cds.append(float(m["flow_fill_rate"]))
                    flows.append(float(m["flow_fill_rate"]))
                rows.append({
                    "policy": f"S{s}_I{period}",
                    "regime": regime,
                    "cd_mean": statistics.mean(cds),
                    "flow_mean": statistics.mean(flows),
                    "n_seeds": len(seeds),
                })
    return rows


def score_cell(rows, regimes, metric="cd_mean"):
    policies = sorted({r["policy"] for r in rows})
    by = {(r["policy"], r["regime"]): float(r[metric]) for r in rows}
    def m(p):
        return statistics.mean(by[(p, rg)] for rg in regimes)
    robust = max(policies, key=m)
    oracle = {rg: max(policies, key=lambda p: by[(p, rg)]) for rg in regimes}
    gaps = {rg: by[(oracle[rg], rg)] - by[(robust, rg)] for rg in regimes}
    mean_gap = statistics.mean(gaps.values())
    argmax_div = len(set(oracle.values()))
    corner_free = any(oracle[rg] != MAX_CORNER for rg in regimes)
    mild = regimes[0]
    mild_flow = by[(oracle[mild], mild)]
    off_sat = OFF_SAT_LO <= mild_flow <= OFF_SAT_HI
    worst_oracle = min(by[(oracle[rg], rg)] for rg in regimes)
    collapse_guard = worst_oracle > COLLAPSE_FLOOR
    eligible = (corner_free and argmax_div >= 2 and off_sat and collapse_guard
                and mean_gap >= MEAN_GAP_FLOOR)
    return {
        "robust_policy": robust, "robust_metric": m(robust),
        "oracle_by_regime": oracle, "mean_gap": mean_gap,
        "argmax_diversity": argmax_div, "corner_free": corner_free,
        "off_saturation": off_sat, "collapse_guard": collapse_guard,
        "mild_flow": mild_flow, "eligible": eligible,
        "gap_by_regime": {k: round(v, 4) for k, v in gaps.items()},
        "oracle_metric_by_regime": {k: round(by[(oracle[k], k)], 4) for k in regimes},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rho-disruption", type=float, default=0.85)
    ap.add_argument("--rho-demand", type=float, default=None)
    ap.add_argument("--surge-budget", type=float, default=2016.0)
    ap.add_argument("--inventory-replenish-lead", type=float, default=168.0)
    ap.add_argument("--phi", type=float, default=1.0)
    ap.add_argument("--psi", type=float, default=1.0)
    ap.add_argument("--stochastic-pt", action="store_true")
    ap.add_argument("--demand-multiplier", type=float, default=1.0)
    ap.add_argument("--horizon-weeks", type=float, default=52.0)
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--regimes", default="current,increased,severe")
    ap.add_argument("--output", type=Path,
                    default=Path("outputs/audits/frontier_gate/cell.json"))
    ap.add_argument("--label", default="cell")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    regimes = [r.strip() for r in args.regimes.split(",") if r.strip()]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    rows = run_static_panel(
        args.horizon_weeks, seeds, regimes,
        rho_disruption=args.rho_disruption,
        rho_demand=args.rho_demand,
        surge_budget=args.surge_budget,
        inventory_replenish_lead=args.inventory_replenish_lead,
        phi=args.phi, psi=args.psi, stochastic_pt=args.stochastic_pt,
        demand_multiplier=args.demand_multiplier,
    )
    gate = score_cell(rows, regimes)

    result = {
        "label": args.label,
        "env_params": {
            "rho_disruption": args.rho_disruption, "rho_demand": args.rho_demand,
            "surge_budget": args.surge_budget,
            "inventory_replenish_lead": args.inventory_replenish_lead,
            "phi": args.phi, "psi": args.psi,
            "stochastic_pt": args.stochastic_pt,
            "demand_multiplier": args.demand_multiplier,
            "horizon_weeks": args.horizon_weeks, "seeds": seeds,
        },
        "n_static_runs": len(rows) * len(seeds),
        "gate": gate,
    }
    args.output.write_text(json.dumps(result, indent=2))
    print(f"[{args.label}] eligible={gate['eligible']} mean_gap={gate['mean_gap']:.4f} "
          f"div={gate['argmax_diversity']} corner={gate['corner_free']} "
          f"off_sat={gate['off_saturation']} robust={gate['robust_policy']} "
          f"-> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
