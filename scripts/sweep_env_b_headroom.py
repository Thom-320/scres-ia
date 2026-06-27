#!/usr/bin/env python3
"""Env B stress-headroom sweep.

Goal: choose the stressed environment where a dynamic (RL) policy has the most
room to beat the best STATIC policy. Headroom signals (computed per stress
config across a policy set that spans the thesis action grid):

  - spread_across_statics  : std of the service score across the static set.
                             Large spread => the action choice matters a lot,
                             so an adaptive policy has more to gain.
  - within_policy_cv       : mean across policies of the seed-CV of the service
                             score. High CV => stochasticity that anticipation
                             can exploit (stochastic_pt, variable demand).
  - headroom               : spread + 0.5*mean_within_cv  (composite).

Service score (delay=54 makes order-level fill_rate 0 by construction, so we
use a clean service/resource axis): ``flow_fill - lost_rate - backlog_pressure``.

The sweep runs the thesis 6x3 static grid over a coarse stress grid and writes
a CSV ranked by headroom. Env B = the top-ranked config that keeps metrics
non-degenerate and leaves the best static below saturation.
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
from supply_chain.config import (
    THESIS_FAITHFUL_PROTOCOL as P,
    THESIS_REPLICATION_DOWNSTREAM_Q_SOURCE as DQ,
    INVENTORY_BUFFERS,
)
from supply_chain.episode_metrics import compute_episode_metrics

INVENTORY_PERIODS = (0, 168, 336, 504, 672, 1344)
POLICIES = [
    (f"S{shifts}_I{period}", shifts, period)
    for shifts in (1, 2, 3)
    for period in INVENTORY_PERIODS
]


def run_one(
    shifts,
    period,
    seed,
    *,
    horizon,
    risk_level,
    phi,
    psi,
    stochastic_pt,
    demand_multiplier,
):
    bufs = dict(INVENTORY_BUFFERS[period]) if period else None
    sim = MFSCSimulation(
        shifts=shifts, seed=seed, horizon=horizon, risks_enabled=True,
        risk_level=risk_level, risk_occurrence_mode="thesis_window",
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
        inventory_replenishment_period=(float(period) if period else None),
    )
    sim.run()
    panel = compute_episode_metrics(sim)
    lost = float(panel["n_lost"])
    bo_qty = float(getattr(sim, "pending_backorder_qty", 0.0) or 0.0)
    delivered = float(panel["delivered_rations"])
    demanded = max(1.0, float(panel["demanded_rations"]))
    backlog_pressure = min(1.0, bo_qty / demanded)
    service = float(panel["flow_fill_rate"]) - float(panel["lost_rate"]) - backlog_pressure
    return {
        "service": service,
        "lost_orders": lost,
        "backorder_qty": bo_qty,
        "ret_excel": float(panel["ret_excel"]),
        "flow_fill_rate": float(panel["flow_fill_rate"]),
        "service_loss_auc_per_order": float(panel["service_loss_auc_per_order"]),
        "ttr_mean": float(panel["ttr_mean"]),
        "delivered": delivered,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--risk-levels", default="increased,severe")
    ap.add_argument("--phis", default="1.0,1.5,2.0")
    ap.add_argument("--psis", default="1.0,1.25,1.5")
    ap.add_argument("--stochastic", default="False,True")
    ap.add_argument("--demand-multipliers", default="1.0,1.1")
    ap.add_argument("--horizon-weeks", type=float, default=52.0)
    ap.add_argument("--output", default="outputs/experiments/env_b_headroom_sweep")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    risk_levels = [r.strip() for r in args.risk_levels.split(",") if r.strip()]
    phis = [float(x) for x in args.phis.split(",") if x.strip()]
    psis = [float(x) for x in args.psis.split(",") if x.strip()]
    stoch = [s.strip() == "True" for s in args.stochastic.split(",")]
    demand_multipliers = [
        float(x) for x in args.demand_multipliers.split(",") if x.strip()
    ]
    horizon = float(args.horizon_weeks) * 7.0 * 24.0
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for risk in risk_levels:
        for phi in phis:
            for psi in psis:
                for spt in stoch:
                    for demand_multiplier in demand_multipliers:
                        per_policy_service = []      # mean service per policy (for spread)
                        per_policy_cv = []           # seed-CV per policy
                        best_static = None
                        worst_static = None
                        for label, S, period in POLICIES:
                            ss = [
                                run_one(
                                    S,
                                    period,
                                    sd,
                                    horizon=horizon,
                                    risk_level=risk,
                                    phi=phi,
                                    psi=psi,
                                    stochastic_pt=spt,
                                    demand_multiplier=demand_multiplier,
                                )
                                for sd in seeds
                            ]
                            mean_svc = statistics.mean([x["service"] for x in ss])
                            sd_svc = (
                                statistics.pstdev([x["service"] for x in ss])
                                if len(ss) > 1
                                else 0.0
                            )
                            cv = (
                                sd_svc / abs(mean_svc)
                                if abs(mean_svc) > 1e-9
                                else 0.0
                            )
                            per_policy_service.append(mean_svc)
                            per_policy_cv.append(cv)
                            mean_lost = statistics.mean([x["lost_orders"] for x in ss])
                            mean_ret = statistics.mean([x["ret_excel"] for x in ss])
                            mean_flow = statistics.mean([x["flow_fill_rate"] for x in ss])
                            mean_auc = statistics.mean(
                                [x["service_loss_auc_per_order"] for x in ss]
                            )
                            candidate = {
                                "policy": label,
                                "service": mean_svc,
                                "lost": mean_lost,
                                "ret_excel": mean_ret,
                                "flow_fill": mean_flow,
                                "service_loss_auc_per_order": mean_auc,
                            }
                            if best_static is None or mean_svc > best_static["service"]:
                                best_static = candidate
                            if worst_static is None or mean_svc < worst_static["service"]:
                                worst_static = candidate
                        spread = statistics.pstdev(per_policy_service)
                        mean_cv = statistics.mean(per_policy_cv)
                        oracle_gap = best_static["service"] - worst_static["service"]
                        saturation_penalty = max(0.0, best_static["flow_fill"] - 0.95)
                        collapse_penalty = max(0.0, 0.35 - best_static["flow_fill"])
                        headroom = (
                            spread
                            + 0.5 * mean_cv
                            + 0.25 * oracle_gap
                            - saturation_penalty
                            - collapse_penalty
                        )
                        rows.append({
                            "risk_level": risk,
                            "phi": phi,
                            "psi": psi,
                            "stochastic_pt": spt,
                            "demand_multiplier": demand_multiplier,
                            "horizon_weeks": float(args.horizon_weeks),
                            "spread": spread,
                            "within_cv": mean_cv,
                            "oracle_gap": oracle_gap,
                            "saturation_penalty": saturation_penalty,
                            "collapse_penalty": collapse_penalty,
                            "headroom": headroom,
                            "best_static": best_static["policy"],
                            "best_static_service": best_static["service"],
                            "best_static_flow_fill": best_static["flow_fill"],
                            "best_static_lost": best_static["lost"],
                            "best_static_ret": best_static["ret_excel"],
                            "best_static_service_loss_auc_per_order": best_static[
                                "service_loss_auc_per_order"
                            ],
                            "worst_static": worst_static["policy"],
                            "worst_static_service": worst_static["service"],
                            "mean_static_service": statistics.mean(per_policy_service),
                        })
                        print(
                            f"{risk:9} phi={phi:g} psi={psi:g} "
                            f"dm={demand_multiplier:g} spt={spt!s:5} "
                            f"spread={spread:.3f} cv={mean_cv:.3f} "
                            f"gap={oracle_gap:.3f} HEADROOM={headroom:.3f} "
                            f"best={best_static['policy']}"
                            f"(flow={best_static['flow_fill']:.3f},"
                            f"lost={best_static['lost']:.0f})",
                            flush=True,
                        )

    rows.sort(key=lambda r: r["headroom"], reverse=True)
    with (out / "headroom.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (out / "headroom.json").write_text(json.dumps(rows, indent=2))
    print(f"\nWROTE {out}/headroom.csv  (top config)")
    top = rows[0]
    print(f"TOP: risk={top['risk_level']} phi={top['phi']} psi={top['psi']} "
          f"spt={top['stochastic_pt']} headroom={top['headroom']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
