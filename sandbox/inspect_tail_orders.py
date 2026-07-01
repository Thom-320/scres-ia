#!/usr/bin/env python3
"""Inspect why endogenous R1/R2 DES orders develop extreme CTj.

For the longest-CTj served orders, list the risk events overlapping their
[OPTj, OATj] window, the number of compounding events, and the longest single
event. Also reports the global risk-event duration distribution to test whether
the tail comes from giant events or from backlog compounding.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from supply_chain.supply_chain import MFSCSimulation, SIMULATION_HORIZON  # noqa: E402
from supply_chain.config import THESIS_FAITHFUL_PROTOCOL as P  # noqa: E402
from supply_chain.config import THESIS_REPLICATION_DOWNSTREAM_Q_SOURCE as DQ  # noqa: E402
from supply_chain.thesis_design import R1_RISKS, R2_RISKS  # noqa: E402


def make_sim(enabled, seed):
    return MFSCSimulation(
        shifts=1, seed=seed, horizon=SIMULATION_HORIZON,
        risks_enabled=True, risk_level="current", enabled_risks=enabled,
        risk_occurrence_mode="thesis_window", risk_attribution_source="des_events",
        year_basis=P["year_basis"], warmup_trigger=P["warmup_trigger"],
        r14_defect_mode=P["r14_defect_mode"], downstream_q_source=DQ,
        raw_material_flow_mode=P["raw_material_flow_mode"],
        raw_material_order_up_to_multiplier=P["raw_material_order_up_to_multiplier"],
        demand_on_hand_fulfillment_delay=P["demand_on_hand_fulfillment_delay"],
    )


def inspect(name, enabled):
    print(f"\n=========== {name} ===========")
    sim = make_sim(enabled, 1)
    sim.run()
    evs = sim.risk_events
    durs = np.array([float(e.duration) for e in evs if e.duration > 0])
    print(f"risk_events total={len(evs)} with-duration={durs.size}")
    if durs.size:
        for q in (50, 90, 95, 99):
            print(f"   event duration p{q} = {np.percentile(durs,q):.1f}h")
        print(f"   event duration MAX = {durs.max():.1f}h")
        # longest 3 events
        longest = sorted(evs, key=lambda e: e.duration, reverse=True)[:3]
        for e in longest:
            print(f"   longest event: {e.risk_id} dur={e.duration:.0f}h "
                  f"[{e.start_time:.0f},{e.end_time:.0f}] ops={e.affected_ops} mag={e.magnitude:.1f}")

    served = [o for o in sim.orders if o.CTj is not None and o.OATj is not None
              and not getattr(o, "lost", False) and not getattr(o, "metrics_excluded", False)]
    served.sort(key=lambda o: o.CTj, reverse=True)
    print(f"\nserved={len(served)} lost/Ut tracking via total_unattended={sim.total_unattended_orders}")
    print("--- top 5 CTj orders: overlapping risk events ---")
    for o in served[:5]:
        ov = [e for e in evs if e.duration > 0
              and max(e.start_time, o.OPTj) < min(e.end_time, o.OATj)]
        ov_dur = sum(min(e.end_time, o.OATj) - max(e.start_time, o.OPTj) for e in ov)
        biggest = max((e.duration for e in ov), default=0.0)
        ids = {}
        for e in ov:
            ids[e.risk_id] = ids.get(e.risk_id, 0) + 1
        print(f"  j={o.j} CTj={o.CTj:.0f}h OPTj={o.OPTj:.0f} OATj={o.OATj:.0f} "
              f"Q={o.quantity:.0f} | {len(ov)} events overlap={ov_dur:.0f}h "
              f"biggest={biggest:.0f}h ids={ids}")


inspect("R1", set(R1_RISKS))
inspect("R2", set(R2_RISKS))
