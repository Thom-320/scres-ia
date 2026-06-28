#!/usr/bin/env python3
"""Track-A CD same-bar GATE audit (statics only, anti outcome-shopping).

Sweep env cells of increasing departure from the thesis and score each by whether
a real DECISION FRONTIER exists under the CD-family metric ``ret_continuous``
(level-robust, CD-adjacent), with ``ret_excel`` reported alongside for
continuity. Per cell we also capture the four gain lenses so the eventual RL win
can be declared in any lens:

  - regret vs hindsight oracle: ``mean_gap`` (headroom a dynamic policy captures)
  - tail / CVaR95 of per-seed service_loss (resilience = tail behaviour)
  - TTR / service_loss_auc (resilience triangle, thesis-native)
  - cross-regime argmax diversity (optimum moves across regimes)

A cell is ELIGIBLE iff corner_free AND argmax_diversity>=2 AND off_saturation AND
collapse_guard AND mean_gap above threshold. Cells are ranked by mean_gap; ties
broken by parsimony (fewest thesis departures). No RL is run here.

Cells (progressive departure):
  v0  faithful
  v1  + stochastic_pt=True
  v2  + demand_multiplier in {1.1, 1.2}
  v4  risk stress phi:psi in {1.5:1.25, 2.0:1.5}   (v3 = continuous buffer is a
      training-time action-space variant; the static frontier is unchanged, so
      it is not a separate gate cell.)
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
    INVENTORY_BUFFERS,
)
from supply_chain.episode_metrics import compute_episode_metrics, merge_resource_metrics

INVENTORY_LEVELS = [0, 168, 336, 504, 672, 1344]
SHIFT_LEVELS = [1, 2, 3]
MAX_CORNER = "S3_I1344"
COLLAPSE_FLOOR = 0.05
OFF_SAT_LO, OFF_SAT_HI = 0.30, 0.97
MEAN_GAP_FLOOR = 0.005  # minimum regret-vs-oracle to be "eligible"


def run_config(shifts, period, seed, regime, horizon, *, phi, psi, stochastic_pt,
               demand_multiplier):
    bufs = dict(INVENTORY_BUFFERS[period]) if period else None
    sim = MFSCSimulation(
        shifts=shifts, seed=seed, horizon=horizon, risks_enabled=True, risk_level=regime,
        risk_occurrence_mode="thesis_window", year_basis=P["year_basis"],
        warmup_trigger=P["warmup_trigger"], r14_defect_mode=P["r14_defect_mode"],
        downstream_q_source="figure_6_2", raw_material_flow_mode=P["raw_material_flow_mode"],
        raw_material_order_up_to_multiplier=P["raw_material_order_up_to_multiplier"],
        demand_on_hand_fulfillment_delay=float(P["demand_on_hand_fulfillment_delay"]),
        initial_buffers=bufs, inventory_replenishment_period=(float(period) if period else None),
        risk_frequency_multiplier=phi, risk_impact_multiplier=psi,
        stochastic_pt=stochastic_pt, demand_mean_multiplier=demand_multiplier,
    )
    sim.run()
    panel = compute_episode_metrics(sim)
    buffer_units = float(sum(INVENTORY_BUFFERS[period].values())) if period else 0.0
    return merge_resource_metrics(
        panel, shift_hours=float(shifts) * float(horizon),
        extra_shift_hours=float(shifts - 1) * float(horizon),
        strategic_buffer_units=buffer_units,
    )


def _cvar(values, alpha=0.05):
    """CVaR_alpha of a loss list (mean of the worst-alpha tail)."""
    if not values:
        return float("nan")
    vs = sorted(float(v) for v in values)
    k = max(1, int(round(alpha * len(vs))))
    return statistics.mean(vs[-k:])


def panel_cell(seeds, regimes, horizon, *, phi, psi, stochastic_pt, demand_multiplier):
    """Rows: one per (policy, regime) with mean metrics over seeds + CVaR95."""
    rows = []
    for period_i in INVENTORY_LEVELS:
        for s in SHIFT_LEVELS:
            for regime in regimes:
                accs: dict = {}
                per_seed_service_loss = []
                for seed in seeds:
                    m = run_config(s, period_i, seed, regime, horizon, phi=phi, psi=psi,
                                   stochastic_pt=stochastic_pt, demand_multiplier=demand_multiplier)
                    per_seed_service_loss.append(float(m.get("service_loss_auc_per_order", 0.0)))
                    for k, v in m.items():
                        try:
                            accs.setdefault(k, []).append(float(v))
                        except (TypeError, ValueError):
                            pass
                row = {"policy": f"S{s}_I{period_i}", "regime": regime}
                for k, vs in accs.items():
                    row[k] = statistics.mean(vs)
                row["service_loss_cvar95_across_seeds"] = _cvar(per_seed_service_loss, 0.05)
                rows.append(row)
    return rows


def score_cell(rows, regimes, metric="ret_continuous"):
    policies = sorted({r["policy"] for r in rows})
    by = {(r["policy"], r["regime"]): r for r in rows}

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
    eligible = (corner_free and argmax_div >= 2 and off_sat and collapse_guard
                and mean_gap >= MEAN_GAP_FLOOR)
    # tail / TTR lenses for the oracle configs (where RL would need to compete)
    ttr_oracle = {rg: by[(oracle[rg], rg)]["ttr_mean"] for rg in regimes}
    cvar_oracle = {rg: by[(oracle[rg], rg)]["service_loss_cvar95_across_seeds"] for rg in regimes}
    # static spread on the metric (how much policy choice matters in worst regime)
    worst_regime = min(regimes, key=lambda r: min(by[(p, r)][metric] for p in policies))
    spread_worst = (max(by[(p, worst_regime)][metric] for p in policies)
                    - min(by[(p, worst_regime)][metric] for p in policies))
    return {
        "robust_static": robust, "oracle_by_regime": oracle,
        "gap_by_regime": {k: round(v, 5) for k, v in gaps.items()},
        "mean_gap": round(mean_gap, 5), "argmax_diversity": argmax_div,
        "corner_free": corner_free, "mild_flow": round(mild_flow, 4),
        "off_saturation": off_sat, "worst_oracle_ret_cont": round(worst_oracle, 5),
        "collapse_guard": collapse_guard, "eligible": eligible,
        "ttr_oracle_by_regime": {k: round(v, 3) for k, v in ttr_oracle.items()},
        "cvar95_oracle_by_regime": {k: round(v, 4) for k, v in cvar_oracle.items()},
        "spread_worst_regime": round(spread_worst, 5),
        "worst_regime": worst_regime,
        "oracle_ret_excel_by_regime": {
            rg: round(max(by[(p, rg)]["ret_excel"] for p in policies), 5) for rg in regimes},
    }


def build_cells():
    """Progressive-departure cell list: (label, phi, psi, spt, dm, departures)."""
    cells = []
    cells.append(("v0_faithful", 1.0, 1.0, False, 1.0, 0))
    cells.append(("v1_spt", 1.0, 1.0, True, 1.0, 1))
    cells.append(("v2_dm1.1", 1.0, 1.0, False, 1.1, 1))
    cells.append(("v2_dm1.2", 1.0, 1.0, False, 1.2, 1))
    cells.append(("v4_phi1.5_psi1.25", 1.5, 1.25, False, 1.0, 2))
    cells.append(("v4_phi2_psi1.5", 2.0, 1.5, False, 1.0, 2))
    cells.append(("v4_spt_phi1.5_psi1.25", 1.5, 1.25, True, 1.0, 3))
    cells.append(("v4_spt_phi2_psi1.5", 2.0, 1.5, True, 1.0, 3))
    return cells


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--regimes", default="current,increased,severe")
    ap.add_argument("--horizon-weeks", type=float, default=52.0)
    ap.add_argument("--output", default="outputs/audits/track_a_cd_gate_audit")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    regimes = [r.strip() for r in args.regimes.split(",") if r.strip()]
    horizon = float(args.horizon_weeks) * 7.0 * 24.0
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    cells = build_cells()
    results = []
    for i, (label, phi, psi, spt, dm, dep) in enumerate(cells, 1):
        rows = panel_cell(seeds, regimes, horizon, phi=phi, psi=psi,
                          stochastic_pt=spt, demand_multiplier=dm)
        sc = score_cell(rows, regimes)
        sc.update({"cell": label, "phi": phi, "psi": psi, "stochastic_pt": spt,
                   "demand_multiplier": dm, "departures": dep})
        results.append(sc)
        with (out / f"panel_{label}.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[{i}/{len(cells)}] {label:24} mean_gap={sc['mean_gap']:.4f} "
              f"div={sc['argmax_diversity']} corner_free={sc['corner_free']} "
              f"off_sat={sc['off_saturation']} spread_w={sc['spread_worst_regime']:.4f} "
              f"robust={sc['robust_static']} oracle={list(sc['oracle_by_regime'].values())} "
              f"ELIGIBLE={sc['eligible']}", flush=True)

    eligible = [r for r in results if r["eligible"]]
    eligible.sort(key=lambda r: (-r["mean_gap"], r["departures"]))
    flat = []
    for r in sorted(results, key=lambda r: (-r["mean_gap"], r["departures"])):
        flat.append({
            "cell": r["cell"], "departures": r["departures"],
            "phi": r["phi"], "psi": r["psi"], "stochastic_pt": r["stochastic_pt"],
            "demand_multiplier": r["demand_multiplier"],
            "mean_gap": r["mean_gap"], "argmax_diversity": r["argmax_diversity"],
            "corner_free": r["corner_free"], "off_saturation": r["off_saturation"],
            "collapse_guard": r["collapse_guard"], "eligible": r["eligible"],
            "spread_worst_regime": r["spread_worst_regime"],
            "robust_static": r["robust_static"],
            "oracle_by_regime": r["oracle_by_regime"],
            "worst_regime": r["worst_regime"],
            "cvar95_oracle_by_regime": r["cvar95_oracle_by_regime"],
            "ttr_oracle_by_regime": r["ttr_oracle_by_regime"],
        })
    with (out / "gate_ranking.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(flat[0].keys()))
        w.writeheader()
        w.writerows(flat)
    (out / "gate_ranking.json").write_text(json.dumps(flat, indent=2))

    print(f"\n=== ELIGIBLE cells ({len(eligible)}/{len(results)}) ===")
    for r in eligible:
        print(f"  {r['cell']:24} mean_gap={r['mean_gap']:.4f} "
              f"div={r['argmax_diversity']} departures={r['departures']}")
    if eligible:
        top = eligible[0]
        print(f"\nTOP CELL: {top['cell']} (mean_gap={top['mean_gap']}, "
              f"departures={top['departures']})")
    else:
        print("\nNO ELIGIBLE CELL -> Track A exhausted under CD-family gate; pivot to Track B.")
    print(f"\nWROTE {out}/gate_ranking.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
