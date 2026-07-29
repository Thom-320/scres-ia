#!/usr/bin/env python3
"""Full 18-config static baseline panel on the FROZEN faithful env, full metrics.

Runs every Garrido [6 inventory x 3 shift] static config through the thesis-faithful DES
(THESIS_FAITHFUL_PROTOCOL defaults: delay=54, R14=72, thesis_window, kit m2.0, figure_6_2) on the
unified episode-metrics panel, across regimes and seeds with common seeds (paired CRN). This is
the master "what to beat" table — Garrido's own policies scored in our DES.

Note: under the faithful env every order is late (delay 54 > LT 48), so strict on-time fill is
~0; the discriminating bars are ret_excel, flow_fill_rate, lost_rate, ttr, service_loss_auc.
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

# Bars where higher is better (resilience/throughput) vs lower is better (cost/loss).
HIGHER_BETTER = ["ret_excel", "ret_continuous", "flow_fill_rate", "delivered_rations"]
LOWER_BETTER = ["lost_rate", "ttr_mean", "service_loss_auc_per_order", "surge_hours",
                "strategic_buffer_units"]


def run_config(shifts: int, period: int, seed: int, risk: str) -> dict:
    bufs = dict(INVENTORY_BUFFERS[period]) if period else None
    sim = MFSCSimulation(
        shifts=shifts, seed=seed, horizon=SIMULATION_HORIZON, risks_enabled=True,
        risk_level=risk, risk_occurrence_mode="thesis_window", year_basis=P["year_basis"],
        warmup_trigger=P["warmup_trigger"], r14_defect_mode=P["r14_defect_mode"],
        downstream_q_source="figure_6_2", raw_material_flow_mode=P["raw_material_flow_mode"],
        raw_material_order_up_to_multiplier=P["raw_material_order_up_to_multiplier"],
        initial_buffers=bufs,
        inventory_replenishment_period=(float(period) if period else None),
    )
    sim.run()
    panel = compute_episode_metrics(sim)
    shift_hours = float(shifts) * float(SIMULATION_HORIZON)
    extra_shift_hours = float(shifts - 1) * float(SIMULATION_HORIZON)
    buffer_units = float(sum(INVENTORY_BUFFERS[period].values())) if period else 0.0
    return merge_resource_metrics(
        panel, shift_hours=shift_hours, extra_shift_hours=extra_shift_hours,
        strategic_buffer_units=buffer_units,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,2,3,4,5")
    ap.add_argument("--regimes", default="current,increased,severe")
    ap.add_argument("--output", default="outputs/experiments/garrido_static_full_panel_2026-06-26")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    regimes = [r.strip() for r in args.regimes.split(",") if r.strip()]
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)

    configs = [(s, i) for i in INVENTORY_LEVELS for s in SHIFT_LEVELS]
    rows = []
    metric_keys = None
    total = len(configs) * len(regimes)
    done = 0
    for period_i in INVENTORY_LEVELS:
        for s in SHIFT_LEVELS:
            label = f"S{s}_I{period_i}"
            for regime in regimes:
                accs: dict[str, list[float]] = {}
                for seed in seeds:
                    m = run_config(s, period_i, seed, regime)
                    for k, v in m.items():
                        accs.setdefault(k, []).append(float(v))
                row = {"policy": label, "shifts": s, "inventory": period_i, "regime": regime}
                for k, vs in accs.items():
                    row[k] = statistics.mean(vs)
                rows.append(row)
                metric_keys = list(accs.keys())
                done += 1
                print(f"  [{done}/{total}] {label:10} {regime:10} "
                      f"ReT={row['ret_excel']:.4f} flow_fill={row['flow_fill_rate']:.3f} "
                      f"lost={row['lost_rate']:.3f} ttr={row['ttr_mean']:.0f} "
                      f"surge_h={row['surge_hours']/1000:.0f}k buf={row['strategic_buffer_units']/1000:.0f}k",
                      flush=True)

    with (out / "panel.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # "what to beat": best static per key metric per regime + efficient frontier
    best = {}
    for regime in regimes:
        rr = [r for r in rows if r["regime"] == regime]
        reg_best = {}
        for k in HIGHER_BETTER:
            top = max(rr, key=lambda r: r[k]); reg_best[k] = {"policy": top["policy"], "value": round(top[k], 4)}
        for k in LOWER_BETTER:
            top = min(rr, key=lambda r: r[k]); reg_best[k] = {"policy": top["policy"], "value": round(top[k], 4)}
        # efficient frontier: best ret_excel among configs within 2% of the best flow_fill, min surge+buffer
        best_flow = max(r["flow_fill_rate"] for r in rr)
        near = [r for r in rr if r["flow_fill_rate"] >= best_flow - 0.02]
        eff = min(near, key=lambda r: (r["surge_hours"] + r["strategic_buffer_units"]))
        reg_best["efficient_frontier"] = {
            "policy": eff["policy"], "ret_excel": round(eff["ret_excel"], 4),
            "flow_fill_rate": round(eff["flow_fill_rate"], 3),
            "surge_hours": eff["surge_hours"], "strategic_buffer_units": eff["strategic_buffer_units"],
        }
        best[regime] = reg_best

    summary = {"seeds": seeds, "regimes": regimes, "n_configs": len(configs),
               "metric_keys": metric_keys, "what_to_beat": best}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== WHAT TO BEAT (efficient frontier per regime) ===")
    for regime in regimes:
        e = best[regime]["efficient_frontier"]
        print(f"  {regime:10}: {e['policy']:10} ReT={e['ret_excel']:.4f} flow_fill={e['flow_fill_rate']:.3f} "
              f"surge_h={e['surge_hours']/1000:.0f}k buf={e['strategic_buffer_units']/1000:.0f}k")
    print(f"\nWROTE {out}/panel.csv + summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
