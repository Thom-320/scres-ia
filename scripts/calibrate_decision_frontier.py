#!/usr/bin/env python3
"""Decision-frontier calibration (STATIC POLICIES ONLY) — Contract v2 §6 / review protocol.

For each candidate env cell (phi, psi[, surge_inertia]), run the full 18 Track-A static policies
across regimes {current, increased, severe} and score whether a real DECISION FRONTIER exists,
WITHOUT ever looking at any RL/retained-reset outcome (anti outcome-shopping).

Frontier metric (on the bounded, level-robust `ret_continuous`, with `ret_excel` reported alongside):
  - robust_static  = the single config maximizing MEAN ret_continuous across regimes (best fixed policy).
  - oracle_static  = per-regime best ret_continuous (a clairvoyant regime-switcher).
  - mean_gap       = mean_regime(oracle - robust_in_that_regime) = the regret of being fixed = the
                     headroom a regime-aware dynamic policy could capture. THIS is frontier depth.
  - argmax_diversity = # distinct oracle argmax configs across regimes (>=2 => optimum moves).
  - corner_free    = oracle argmax is NOT S3_I1344 in at least one regime.
  - off_saturation = mild-regime oracle flow_fill in [0.30, 0.97] (not trivially solved, not collapsed).
  - collapse_guard = worst-regime oracle ret_continuous above a floor.

A cell is ELIGIBLE if corner_free AND argmax_diversity>=2 AND off_saturation AND collapse_guard.
Winner = eligible cell with the largest mean_gap (deepest frontier); ties broken by parsimony
(smallest phi+psi). All cells are reported regardless.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.supply_chain import MFSCSimulation, SIMULATION_HORIZON
from supply_chain.config import THESIS_FAITHFUL_PROTOCOL as P, INVENTORY_BUFFERS
from supply_chain.episode_metrics import compute_episode_metrics, merge_resource_metrics

INVENTORY_LEVELS = [0, 168, 336, 504, 672, 1344]
SHIFT_LEVELS = [1, 2, 3]
MAX_CORNER = "S3_I1344"
COLLAPSE_FLOOR = 0.05
OFF_SAT_LO, OFF_SAT_HI = 0.30, 0.97


def run_config(shifts, period, seed, regime, horizon, phi, psi, stochastic_pt, surge_inertia,
               surge_budget):
    bufs = dict(INVENTORY_BUFFERS[period]) if period else None
    kw = dict(
        shifts=shifts, seed=seed, horizon=horizon, risks_enabled=True, risk_level=regime,
        risk_occurrence_mode="thesis_window", year_basis=P["year_basis"],
        warmup_trigger=P["warmup_trigger"], r14_defect_mode=P["r14_defect_mode"],
        downstream_q_source="figure_6_2", raw_material_flow_mode=P["raw_material_flow_mode"],
        raw_material_order_up_to_multiplier=P["raw_material_order_up_to_multiplier"],
        initial_buffers=bufs, inventory_replenishment_period=(float(period) if period else None),
        risk_frequency_multiplier=phi, risk_impact_multiplier=psi, stochastic_pt=stochastic_pt,
    )
    if surge_inertia:
        kw["surge_inertia"] = True
        kw["surge_budget_hours"] = surge_budget
    sim = MFSCSimulation(**kw)
    sim.run()
    panel = compute_episode_metrics(sim)
    buffer_units = float(sum(INVENTORY_BUFFERS[period].values())) if period else 0.0
    return merge_resource_metrics(
        panel, shift_hours=float(shifts) * float(horizon),
        extra_shift_hours=float(shifts - 1) * float(horizon), strategic_buffer_units=buffer_units,
    )


def panel_cell(seeds, regimes, horizon, phi, psi, stochastic_pt, surge_inertia, surge_budget):
    """Return rows: one per (policy, regime) with mean metrics over seeds."""
    rows = []
    for period_i in INVENTORY_LEVELS:
        for s in SHIFT_LEVELS:
            for regime in regimes:
                accs = {}
                for seed in seeds:
                    m = run_config(s, period_i, seed, regime, horizon, phi, psi, stochastic_pt,
                                   surge_inertia, surge_budget)
                    for k, v in m.items():
                        accs.setdefault(k, []).append(float(v))
                row = {"policy": f"S{s}_I{period_i}", "regime": regime}
                for k, vs in accs.items():
                    row[k] = statistics.mean(vs)
                rows.append(row)
    return rows


def score_cell(rows, regimes, metric="ret_continuous"):
    policies = sorted({r["policy"] for r in rows})
    by = {(r["policy"], r["regime"]): r for r in rows}
    # robust = best single fixed policy by mean metric across regimes
    def mean_metric(pol):
        return statistics.mean(by[(pol, rg)][metric] for rg in regimes)
    robust = max(policies, key=mean_metric)
    oracle = {rg: max(policies, key=lambda p: by[(p, rg)][metric]) for rg in regimes}
    gaps = {rg: by[(oracle[rg], rg)][metric] - by[(robust, rg)][metric] for rg in regimes}
    mean_gap = statistics.mean(gaps.values())
    argmax_div = len(set(oracle.values()))
    corner_free = any(oracle[rg] != MAX_CORNER for rg in regimes)
    mild = regimes[0]
    mild_flow = by[(oracle[mild], mild)]["flow_fill_rate"]
    off_sat = OFF_SAT_LO <= mild_flow <= OFF_SAT_HI
    worst_oracle = min(by[(oracle[rg], rg)][metric] for rg in regimes)
    collapse_guard = worst_oracle > COLLAPSE_FLOOR
    eligible = corner_free and argmax_div >= 2 and off_sat and collapse_guard
    return {
        "robust_static": robust, "oracle_by_regime": oracle,
        "gap_by_regime": {k: round(v, 5) for k, v in gaps.items()},
        "mean_gap": round(mean_gap, 5), "argmax_diversity": argmax_div,
        "corner_free": corner_free, "mild_flow": round(mild_flow, 4),
        "off_saturation": off_sat, "worst_oracle_ret_cont": round(worst_oracle, 5),
        "collapse_guard": collapse_guard, "eligible": eligible,
        # outcome-bar view (reported, not used for selection)
        "oracle_ret_excel_by_regime": {
            rg: round(max(by[(p, rg)]["ret_excel"] for p in policies), 5) for rg in regimes},
    }


def parse_cells(spec):
    cells = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        phi, psi = tok.split(":")
        cells.append((float(phi), float(psi)))
    return cells


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="1:1,1:1.5,2:1.5",
                    help="comma list of phi:psi cells, e.g. 1:1,1:1.5,2:1.5")
    ap.add_argument("--regimes", default="current,increased,severe")
    ap.add_argument("--seeds", default="1,2,3,4,5")
    ap.add_argument("--horizon", type=int, default=SIMULATION_HORIZON)
    ap.add_argument("--stochastic-pt", action="store_true")
    ap.add_argument("--surge-inertia", action="store_true")
    ap.add_argument("--surge-budget-hours", type=float, default=2016.0)
    ap.add_argument("--output", default="outputs/experiments/frontier_calibration_2026-06-26")
    args = ap.parse_args()
    cells = parse_cells(args.cells)
    regimes = [r.strip() for r in args.regimes.split(",") if r.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    results = []
    for i, (phi, psi) in enumerate(cells, 1):
        rows = panel_cell(seeds, regimes, args.horizon, phi, psi, args.stochastic_pt,
                          args.surge_inertia, args.surge_budget_hours)
        sc = score_cell(rows, regimes)
        sc.update({"phi": phi, "psi": psi, "stochastic_pt": args.stochastic_pt,
                   "surge_inertia": args.surge_inertia})
        results.append(sc)
        tag = f"phi{phi}_psi{psi}" + ("_inertia" if args.surge_inertia else "")
        with (out / f"panel_{tag}.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"  [{i}/{len(cells)}] phi={phi} psi={psi} inertia={args.surge_inertia}  "
              f"mean_gap={sc['mean_gap']:.4f} div={sc['argmax_diversity']} "
              f"corner_free={sc['corner_free']} off_sat={sc['off_saturation']} "
              f"mild_flow={sc['mild_flow']:.3f} robust={sc['robust_static']} "
              f"oracle={list(sc['oracle_by_regime'].values())} ELIGIBLE={sc['eligible']}",
              flush=True)

    eligible = [r for r in results if r["eligible"]]
    pool = eligible or results
    winner = max(pool, key=lambda r: (r["mean_gap"], -(r["phi"] + r["psi"])))
    summary = {"cells": args.cells, "regimes": regimes, "seeds": seeds, "horizon": args.horizon,
               "surge_inertia": args.surge_inertia, "results": results,
               "winner": winner, "n_eligible": len(eligible)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== FRONTIER WINNER (deepest gap among eligible) ===")
    print(f"  phi={winner['phi']} psi={winner['psi']} inertia={winner['surge_inertia']}  "
          f"mean_gap={winner['mean_gap']:.4f}  diversity={winner['argmax_diversity']}  "
          f"robust={winner['robust_static']} oracle={list(winner['oracle_by_regime'].values())}")
    print(f"  eligible cells: {len(eligible)}/{len(results)}  (eligible = corner_free & div>=2 & "
          f"off_sat & not-collapsed)")
    print(f"\nWROTE {out}/summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
