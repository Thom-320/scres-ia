#!/usr/bin/env python3
"""Calibrate the fine-tuned HEADROOM env (Contract v2 §6) — find the (phi, psi) band.

Runs the full 18-config static panel through the faithful env under a grid of risk-frequency
(phi) x risk-impact (psi) multipliers, and reports, per cell:
  - ret_excel SPREAD across the 18 statics (max-min): the depth of the decision frontier.
  - argmax static on ret_excel: is the optimum off the trivial max-buffer/max-shift corner?
  - best-static flow_fill_rate: is the system off-saturation (target ~0.5-0.85, not ~1.0 or ~0)?
  - best-static lost_rate.

Frontier band criterion (Contract v2): pick the most PARSIMONIOUS cell where the spread is
materially deeper than the faithful baseline (~0.001) AND the best static is off-saturation AND
the argmax is not a free max-corner win. The chosen cell is FROZEN before any confirmatory run.

Default regime = severe (most headroom). Pass --regimes to also confirm regime-dependence.
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
MAX_CORNER = "S3_I1344"  # the trivial "throw everything at it" config


def run_config(shifts, period, seed, regime, horizon, phi, psi, stochastic_pt):
    bufs = dict(INVENTORY_BUFFERS[period]) if period else None
    sim = MFSCSimulation(
        shifts=shifts, seed=seed, horizon=horizon, risks_enabled=True, risk_level=regime,
        risk_occurrence_mode="thesis_window", year_basis=P["year_basis"],
        warmup_trigger=P["warmup_trigger"], r14_defect_mode=P["r14_defect_mode"],
        downstream_q_source="figure_6_2", raw_material_flow_mode=P["raw_material_flow_mode"],
        raw_material_order_up_to_multiplier=P["raw_material_order_up_to_multiplier"],
        initial_buffers=bufs, inventory_replenishment_period=(float(period) if period else None),
        risk_frequency_multiplier=phi, risk_impact_multiplier=psi, stochastic_pt=stochastic_pt,
    )
    sim.run()
    panel = compute_episode_metrics(sim)
    buffer_units = float(sum(INVENTORY_BUFFERS[period].values())) if period else 0.0
    return merge_resource_metrics(
        panel, shift_hours=float(shifts) * float(horizon),
        extra_shift_hours=float(shifts - 1) * float(horizon), strategic_buffer_units=buffer_units,
    )


def panel_for_cell(seeds, regime, horizon, phi, psi, stochastic_pt):
    rows = []
    for period_i in INVENTORY_LEVELS:
        for s in SHIFT_LEVELS:
            accs = {}
            for seed in seeds:
                m = run_config(s, period_i, seed, regime, horizon, phi, psi, stochastic_pt)
                for k, v in m.items():
                    accs.setdefault(k, []).append(float(v))
            row = {"policy": f"S{s}_I{period_i}", "shifts": s, "inventory": period_i}
            for k, vs in accs.items():
                row[k] = statistics.mean(vs)
            rows.append(row)
    return rows


def assess(rows):
    rets = [r["ret_excel"] for r in rows]
    best = max(rows, key=lambda r: r["ret_excel"])
    spread = max(rets) - min(rets)
    best_flow = best["flow_fill_rate"]
    off_sat = 0.45 <= best_flow <= 0.88
    corner_free = best["policy"] != MAX_CORNER
    return {
        "ret_spread": round(spread, 5), "ret_best": round(best["ret_excel"], 5),
        "argmax": best["policy"], "best_flow_fill": round(best_flow, 4),
        "best_lost_rate": round(best["lost_rate"], 4),
        "off_saturation": off_sat, "corner_free": corner_free,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--regime", default="severe")
    ap.add_argument("--phi", default="1,2,3")
    ap.add_argument("--psi", default="1,1.5,2")
    ap.add_argument("--stochastic-pt", action="store_true")
    ap.add_argument("--horizon", type=int, default=SIMULATION_HORIZON)
    ap.add_argument("--faithful-spread", type=float, default=0.0014,
                    help="ret_excel spread of the faithful (phi=psi=1) severe panel for reference")
    ap.add_argument("--output", default="outputs/experiments/headroom_calibration_2026-06-26")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    phis = [float(x) for x in args.phi.split(",") if x.strip()]
    psis = [float(x) for x in args.psi.split(",") if x.strip()]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    results = []
    total = len(phis) * len(psis)
    done = 0
    for phi in phis:
        for psi in psis:
            rows = panel_for_cell(seeds, args.regime, args.horizon, phi, psi, args.stochastic_pt)
            a = assess(rows)
            a.update({"phi": phi, "psi": psi, "stochastic_pt": args.stochastic_pt})
            results.append(a)
            # persist the per-config panel for the chosen-cell analysis later
            with (out / f"panel_phi{phi}_psi{psi}.csv").open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            done += 1
            depth = a["ret_spread"] / max(args.faithful_spread, 1e-9)
            print(f"  [{done}/{total}] phi={phi} psi={psi}  spread={a['ret_spread']:.4f} "
                  f"({depth:.1f}x faithful)  argmax={a['argmax']:8} flow={a['best_flow_fill']:.3f} "
                  f"lost={a['best_lost_rate']:.3f}  off_sat={a['off_saturation']} "
                  f"corner_free={a['corner_free']}", flush=True)

    # recommend: deepest spread among cells that are off-saturation AND corner-free
    eligible = [r for r in results if r["off_saturation"] and r["corner_free"]]
    pool = eligible or results
    rec = max(pool, key=lambda r: r["ret_spread"])
    summary = {"regime": args.regime, "seeds": seeds, "horizon": args.horizon,
               "faithful_spread": args.faithful_spread, "grid": results,
               "recommended_cell": rec, "n_eligible": len(eligible)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== RECOMMENDED HEADROOM CELL (deepest frontier, off-saturation, corner-free) ===")
    print(f"  phi={rec['phi']} psi={rec['psi']} stochastic_pt={rec['stochastic_pt']}  "
          f"spread={rec['ret_spread']:.4f} ({rec['ret_spread']/max(args.faithful_spread,1e-9):.1f}x)  "
          f"argmax={rec['argmax']} flow={rec['best_flow_fill']:.3f}")
    print(f"  eligible cells (off-sat & corner-free): {len(eligible)}/{len(results)}")
    print(f"\nWROTE {out}/summary.json + per-cell panels")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
