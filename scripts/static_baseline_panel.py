#!/usr/bin/env python3
"""Static-policy baseline PANEL — the 'what to beat' table for the dominance + learning study.

The thesis Excel ReT is low (~0.01-0.22) and, critically, NON-monotone in buffer (a risk-active
gate + AP/LT artifact: buffered configs register an autotomy period on ~all orders, so they lose
the high-scoring undisrupted orders). So we do NOT optimize raw ReT alone. This panel reports a
multi-metric view for each static policy across regimes, with common seeds (paired CRN):

  resilience (clean):  fill_rate (mean), backorder_qty (service-loss proxy), lost_orders,
                       ret_excel (Garrido continuity, reported NOT optimized)
  resources:           shift_hours, mean_strategic_inventory
  throughput:          delivered_rations

Static set covers the thesis original (S1,I0) and the candidate efficient/strong policies.
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
from supply_chain.ret_thesis import compute_order_level_ret_excel_formula

# (label, shifts, inventory_period)  inventory_period 0 == no strategic buffer
POLICIES = [
    ("original_S1_I0", 1, 0),     # thesis Cf0 baseline
    ("S2_I0", 2, 0),
    ("S3_I0", 3, 0),
    ("I168_S1", 1, 168),
    ("I168_S2", 2, 168),
    ("I1344_S3", 3, 1344),        # max thesis lever
]


def run_policy(
    shifts: int,
    period: int,
    seed: int,
    risk: str,
    *,
    demand_on_hand_fulfillment_delay: float = P["demand_on_hand_fulfillment_delay"],
):
    bufs = dict(INVENTORY_BUFFERS[period]) if period else None
    sim = MFSCSimulation(
        shifts=shifts, seed=seed, horizon=SIMULATION_HORIZON, risks_enabled=True,
        risk_level=risk, risk_occurrence_mode="thesis_window", year_basis=P["year_basis"],
        warmup_trigger=P["warmup_trigger"], r14_defect_mode=P["r14_defect_mode"],
        downstream_q_source=DQ, raw_material_flow_mode=P["raw_material_flow_mode"],
        raw_material_order_up_to_multiplier=P["raw_material_order_up_to_multiplier"],
        demand_on_hand_fulfillment_delay=float(demand_on_hand_fulfillment_delay),
        initial_buffers=bufs,
        inventory_replenishment_period=(float(period) if period else None),
    )
    sim.run()
    orders = sim.orders
    ret = compute_order_level_ret_excel_formula(orders, current_time=float(sim.env.now))
    n = len(orders) or 1
    served = [o for o in orders if getattr(o, "OATj", None) is not None]
    lost = sum(1 for o in orders if bool(getattr(o, "lost", False)))
    # fill rate: fraction served within lead time
    on_time = sum(
        1 for o in served
        if getattr(o, "CTj", None) is not None
        and float(o.CTj) <= float(getattr(o, "LTj", 0.0) or 0.0)
    )
    fill = on_time / n
    backorder_qty = float(getattr(sim, "pending_backorder_qty", 0.0) or 0.0)
    delivered = sum(float(getattr(o, "quantity", 0.0) or 0.0) for o in served)
    # resources
    shift_hours = float(shifts) * float(SIMULATION_HORIZON)
    mean_inv = float(sum(INVENTORY_BUFFERS[period].values())) if period else 0.0
    return {
        "ret_excel": ret["mean_ret_excel"],
        "fill_rate": fill,
        "lost_orders": lost,
        "backorder_qty": backorder_qty,
        "delivered_rations": delivered,
        "shift_hours": shift_hours,
        "strategic_inventory": mean_inv,
        "n_orders": n,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,2,3,4,5")
    ap.add_argument("--regimes", default="current,increased,severe")
    ap.add_argument("--output", default="outputs/experiments/static_baseline_panel_2026-06-26")
    ap.add_argument(
        "--demand-on-hand-fulfillment-delay",
        type=float,
        default=P["demand_on_hand_fulfillment_delay"],
    )
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    regimes = [r.strip() for r in args.regimes.split(",") if r.strip()]
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)

    rows = []
    metrics = ["ret_excel", "fill_rate", "lost_orders", "backorder_qty",
               "delivered_rations", "shift_hours", "strategic_inventory"]
    for label, S, period in POLICIES:
        for regime in regimes:
            acc = {m: [] for m in metrics}
            for seed in seeds:
                r = run_policy(
                    S,
                    period,
                    seed,
                    regime,
                    demand_on_hand_fulfillment_delay=(
                        args.demand_on_hand_fulfillment_delay
                    ),
                )
                for m in metrics:
                    acc[m].append(r[m])
            row = {"policy": label, "shifts": S, "inventory": period, "regime": regime}
            for m in metrics:
                row[m] = statistics.mean(acc[m])
            rows.append(row)
            print(f"{label:16} {regime:10} ReT={row['ret_excel']:.4f} fill={row['fill_rate']:.3f} "
                  f"lost={row['lost_orders']:.0f} shift_h={row['shift_hours']/1000:.0f}k inv={row['strategic_inventory']/1000:.0f}k",
                  flush=True)

    with (out / "panel.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    (out / "panel.json").write_text(json.dumps({"seeds": seeds, "regimes": regimes, "rows": rows}, indent=2))
    print(f"\nWROTE {out}/panel.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
