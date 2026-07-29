#!/usr/bin/env python3
"""Custodied audit of two ReT endpoint defects. Produces one signed artifact.

Both were found in development and written up in prose first; adversarial review
correctly objected that prose is not custody. This script regenerates every number
from scratch, with per-order rows, and signs the result.

**A** — step-cadence invariance. ReT must not depend on how often `step()` is called.
It does: `_op_down_since[op]` is reset to `env.now` at every step boundary
(supply_chain.py:1856), and the legacy ongoing-disruption attribution then measures a
still-open disruption from that reset value (supply_chain.py:5743), so `RPj` shrinks as
cadence rises and ReT rises with it.

**B** — the fulfilment-delay cliff. `GARRIDO_FULFILLMENT_DELAY_HOURS = 54.0` is the
delay applied when demand is met from on-hand stock, and it is documented as
"Calibrated minimum CTj: no instant orders; just beyond LT=48". Being *just beyond* the
48 h promise is not a detail: it makes `CTj <= LTj` unsatisfiable, so the autotomy
branch of ReT is unreachable and every scored order takes `0.5/RPj`.

The two compose, and that is the point of auditing them together: because autotomy is
dead, essentially all of ReT's signal in this operating regime flows through `RPj` --
which is exactly the quantity defect A makes cadence-dependent.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.config import (  # noqa: E402
    GARRIDO_FULFILLMENT_DELAY_HOURS,
    HOURS_PER_WEEK,
    LEAD_TIME_PROMISE,
)
from supply_chain.config import THESIS_FAITHFUL_PROTOCOL as P  # noqa: E402
from supply_chain.episode_metrics import compute_episode_metrics  # noqa: E402
from supply_chain.expanded_contract_controllers import level_targets  # noqa: E402
from supply_chain.supply_chain import MFSCSimulation  # noqa: E402

R1R = ("R11", "R12", "R13", "R14")
CADENCES = (None, 672.0, 168.0, 24.0, 1.0)  # None = one step over the whole horizon
DELAYS = (54.0, 48.0, 47.0, 36.0, 24.0, 6.0)


def run(*, horizon: float, step_hours: float | None, delay: float,
        risks: bool = True, seed: int = 1_620_001) -> dict:
    sim = MFSCSimulation(
        shifts=1, initial_buffers=level_targets(168),
        inventory_replenishment_period=168.0, seed=seed, horizon=horizon,
        risks_enabled=risks, risk_level="current",
        enabled_risks=set(R1R) if risks else set(),
        risk_overrides={r: "increased" for r in R1R} if risks else {},
        demand_on_hand_fulfillment_delay=delay,
        strict_exogenous_crn=True, year_basis=P["year_basis"],
        warmup_trigger=P["warmup_trigger"], r14_defect_mode=P["r14_defect_mode"])
    step = horizon if step_hours is None else step_hours
    elapsed = 0.0
    while elapsed < horizon:
        s = min(step, horizon - elapsed)
        sim.step(action=None, step_hours=s)
        elapsed += s

    m = compute_episode_metrics(sim)
    scored = [o for o in sim.orders
              if not bool(getattr(o, "metrics_excluded", False))
              and float(getattr(o, "OPTj", 0.0)) >= float(sim.warmup_time)]
    served = [o for o in scored if getattr(o, "OATj", None) is not None]
    rows = [{"j": int(getattr(o, "j", -1)), "OPTj": float(o.OPTj),
             "OATj": float(o.OATj), "CTj": float(o.CTj), "LTj": float(o.LTj or 0.0),
             "APj": float(o.APj or 0.0), "RPj": float(o.RPj or 0.0)}
            for o in served]
    ct = np.array([r["CTj"] for r in rows]) if rows else np.array([0.0])
    return {
        "step_hours": step_hours, "delay": delay,
        "ret_excel": float(m["ret_excel"]),
        "ret_excel_full_ledger": float(m["ret_excel_full_ledger"]),
        "flow_fill_rate": float(m["flow_fill_rate"]),
        "delivered_rations": float(m["delivered_rations"]),
        "lost_orders": float(m["lost_orders"]),
        "fill_rate_on_time": float(m["fill_rate_on_time"]),
        "excel_case_pct_autotomy": float(m["excel_case_pct_autotomy"]),
        "excel_case_pct_recovery": float(m["excel_case_pct_recovery"]),
        "n_scored": len(scored), "n_served": len(served),
        "ctj_min": float(ct.min()), "ctj_median": float(np.median(ct)),
        "n_ctj_le_lt": int((ct <= LEAD_TIME_PROMISE).sum()),
        "n_apj_positive": int(sum(1 for r in rows if r["APj"] > 0)),
        "rows_sha256": sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
        "rows": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--horizon-weeks", type=int, default=52)
    ap.add_argument("--output", type=Path,
                    default=Path("results/metric_audit/ret_defects_v1/result.json"))
    args = ap.parse_args()
    horizon = float(args.horizon_weeks * HOURS_PER_WEEK)
    started = time.perf_counter()

    # A: cadence sweep at the shipped delay.
    cadence = [run(horizon=horizon, step_hours=c,
                   delay=GARRIDO_FULFILLMENT_DELAY_HOURS) for c in CADENCES]
    base = cadence[0]
    for r in cadence:
        r["ret_excel_ratio_vs_one_step"] = r["ret_excel"] / base["ret_excel"]
        # Physical endpoints must be identical; RPj is what moves.
        r["physics_matches_one_step"] = bool(
            abs(r["flow_fill_rate"] - base["flow_fill_rate"]) < 1e-12
            and abs(r["delivered_rations"] - base["delivered_rations"]) < 1e-9)
        pairs = list(zip(base["rows"], r["rows"]))
        r["n_rpj_differs_from_one_step"] = sum(
            1 for a, b in pairs if abs(a["RPj"] - b["RPj"]) > 1e-9)
        r["n_apj_differs_from_one_step"] = sum(
            1 for a, b in pairs if abs(a["APj"] - b["APj"]) > 1e-9)
        r["n_ctj_differs_from_one_step"] = sum(
            1 for a, b in pairs if abs(a["CTj"] - b["CTj"]) > 1e-9)

    # B: the fulfilment-delay cliff, at a single fixed cadence.
    delay = [run(horizon=horizon, step_hours=None, delay=d) for d in DELAYS]
    at54 = next(r for r in delay if r["delay"] == 54.0)
    at48 = next(r for r in delay if r["delay"] == 48.0)

    payload = {
        "schema_version": "ret_metric_defect_audit_v1",
        "claim_status": "DEVELOPMENT_FINDING_METRIC_DEFECT",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lead_time_promise": LEAD_TIME_PROMISE,
        "shipped_fulfillment_delay": GARRIDO_FULFILLMENT_DELAY_HOURS,
        "defect_a_step_cadence": {
            "mechanism": ("step() resets _op_down_since[op] to env.now at every "
                          "boundary (supply_chain.py:1856); the legacy "
                          "ongoing-disruption attribution then measures overlap "
                          "from that reset value (supply_chain.py:5743), so RPj "
                          "shrinks as cadence rises and ReT = 0.5/RPj rises"),
            "sweep": [{k: v for k, v in r.items() if k != "rows"} for r in cadence],
            "ret_excel_spread": max(r["ret_excel"] for r in cadence)
            / min(r["ret_excel"] for r in cadence),
            "physics_invariant_across_all_cadences": all(
                r["physics_matches_one_step"] for r in cadence),
        },
        "defect_b_fulfillment_delay_cliff": {
            "mechanism": ("demand_on_hand_fulfillment_delay defaults to 54 h, six "
                          "hours beyond the 48 h lead-time promise, so CTj <= LTj "
                          "is unsatisfiable and the autotomy branch of ReT is "
                          "unreachable; every scored order takes 0.5/RPj"),
            "sweep": [{k: v for k, v in r.items() if k != "rows"} for r in delay],
            "ret_excel_at_54": at54["ret_excel"],
            "ret_excel_at_48": at48["ret_excel"],
            "ret_excel_ratio_48_over_54": at48["ret_excel"] / at54["ret_excel"],
            "saturates_below_lead_time": len({
                round(r["ret_excel"], 9) for r in delay if r["delay"] < 48.0}) == 1,
        },
        "defects_compose": (
            "because autotomy is unreachable at the shipped delay, essentially all "
            "of ReT's signal flows through RPj -- the quantity defect A makes "
            "cadence-dependent"),
        "elapsed_seconds": time.perf_counter() - started,
    }
    body = json.dumps(payload, indent=1, sort_keys=True)
    payload["self_sha256"] = sha256(body.encode()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")

    rows_path = args.output.with_name("per_order_rows.json")
    rows_path.write_text(json.dumps(
        {"cadence": {str(r["step_hours"]): r["rows"] for r in cadence},
         "delay": {str(r["delay"]): r["rows"] for r in delay}},
        indent=1, sort_keys=True) + "\n")
    print(f"-> {args.output}\n-> {rows_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
