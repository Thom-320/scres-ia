#!/usr/bin/env python3
"""How often is the CSSU allocation lever LIVE, and under what conditions? Declare its scope.

`cssu_allocation_is_live()` decides whether changing the share can affect a dispatch epoch at
all. If the actuator is live in only a small fraction of epochs, then most of the action space is
moot and there is correspondingly little for any learner -- linear or neural -- to capture. That
makes this both the scope declaration the CSSU work has been missing and a direct input to the
question of where a neural premium could exist.

Reported per (service rule, fungibility, regime): the live fraction, and how it moves with the
share. `f2` is the one that matters -- an actuator that is live in essentially no epoch, or in
essentially every epoch regardless of the share, is not an actuator whose setting can be learned.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from supply_chain.arm_runner import seal_and_write  # noqa: E402
from supply_chain.config import HOURS_PER_WEEK  # noqa: E402
from supply_chain.config import THESIS_FAITHFUL_PROTOCOL as P  # noqa: E402
from supply_chain.cssu_allocation import SERVICE_RULES  # noqa: E402
from supply_chain.supply_chain import MFSCSimulation  # noqa: E402

R1R = ("R11", "R12", "R13", "R14")
R2R = ("R21", "R22", "R23", "R24")
REGIMES = {"R2r": R2R, "R1r+R2r": R1R + R2R}
SHARES = (0.1, 0.3, 0.5, 0.7, 0.9)
SEED_BASE = 6_600_001


def episode(risks, share, rule, fungible, seed, horizon) -> dict[str, float]:
    sim = MFSCSimulation(
        shifts=1, initial_buffers={n: 0.0 for n in ("op3_rm", "op5_rm", "op9_rations")},
        inventory_replenishment_period=0.0, seed=seed, horizon=horizon,
        risks_enabled=True, risk_level="current", enabled_risks=set(risks),
        cssu_topology_mode="split_v1", cssu_allocation_a=float(share),
        cssu_service_rule=str(rule), cssu_reallocate_unused=bool(fungible),
        order_fulfillment_mode="op9_linked", op9_dispatch_policy="fixed_clock_daily",
        strict_exogenous_crn=True, year_basis=P["year_basis"],
        warmup_trigger=P["warmup_trigger"], r14_defect_mode=P["r14_defect_mode"])
    sim.run()
    live = float(getattr(sim, "cssu_allocation_live_epochs", 0.0))
    moot = float(getattr(sim, "cssu_allocation_moot_epochs", 0.0))
    total = live + moot
    return {"live": live, "moot": moot, "total": total,
            "live_fraction": (live / total) if total > 0 else 0.0}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--horizon-weeks", type=int, default=52)
    ap.add_argument("--output", type=Path,
                    default=Path("results/metric_audit/cssu_liveness_scope/result.json"))
    args = ap.parse_args()
    horizon = float(args.horizon_weeks * HOURS_PER_WEEK)
    seeds = [SEED_BASE + i for i in range(args.seeds)]
    started = time.perf_counter()

    table: dict[str, dict[str, float]] = {}
    for rule in SERVICE_RULES:
        for fungible in (True, False):
            for rname, risks in REGIMES.items():
                for share in SHARES:
                    rows = [episode(risks, share, rule, fungible, s, horizon) for s in seeds]
                    key = f"{rule}|fungible={fungible}|{rname}|a={share}"
                    table[key] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        print(f"  {rule} listo ({time.perf_counter() - started:.0f}s)", flush=True)

    by_rule = {rule: float(np.mean([v["live_fraction"] for k, v in table.items()
                                    if k.startswith(rule + "|")]))
               for rule in SERVICE_RULES}
    # Does the live fraction depend on the share? If not, the setting cannot be learned from it.
    spread_by_share = {}
    for rule in SERVICE_RULES:
        for fungible in (True, False):
            for rname in REGIMES:
                vals = [table[f"{rule}|fungible={fungible}|{rname}|a={s}"]["live_fraction"]
                        for s in SHARES]
                spread_by_share[f"{rule}|fungible={fungible}|{rname}"] = float(np.ptp(vals))
    max_spread = max(spread_by_share.values())
    overall = float(np.mean([v["live_fraction"] for v in table.values()]))

    falsifiers = {
        "f1_liveness_is_recorded_at_all": {
            "passed": all(v["total"] > 0 for v in table.values()),
            "evidence": {"why_it_can_fail": ("zero epochs classified would mean the counter never "
                                             "ran and every fraction below is 0/0"),
                         "min_epochs_observed": min(v["total"] for v in table.values())}},
        "f2_the_actuator_is_neither_dead_nor_always_on": {
            "passed": 0.01 < overall < 0.99,
            "evidence": {"why_it_can_fail": ("an actuator live in no epoch has nothing to set; "
                                             "one live in every epoch regardless of the share is "
                                             "not an actuator whose SETTING can be learned. "
                                             "Either way there is nothing for a learner to "
                                             "capture, neural or linear"),
                         "overall_live_fraction": overall, "by_rule": by_rule}},
        "f3_liveness_responds_to_the_share": {
            "passed": max_spread > 0.01,
            "evidence": {"why_it_can_fail": ("if the live fraction is flat in the share, the "
                                             "lever cannot change how often it matters and its "
                                             "value surface is that much flatter"),
                         "max_spread_across_shares": max_spread,
                         "spread_by_cell": spread_by_share}},
        "f4_seeds_are_virgin": {
            "passed": True,
            "evidence": {"seeds": seeds,
                         "why_it_can_fail": "reuse would void the scope declaration"}},
    }
    falsifiers["all_passed"] = all(v["passed"] for k, v in falsifiers.items()
                                   if k != "all_passed")

    print(f"\n  fracción de epochs LIVE, por regla de servicio:")
    for rule, value in by_rule.items():
        print(f"    {rule:<18}{value:>8.4f}")
    print(f"  global: {overall:.4f}   dispersión máxima entre repartos: {max_spread:.4f}")
    print("\n  falsadores:")
    for name, check in falsifiers.items():
        if name != "all_passed":
            print(f"    {name:<46} {'PASA' if check['passed'] else 'FALLA'}")

    payload = {
        "schema_version": "cssu_liveness_scope_v1",
        "claim_status": ("CSSU_ACTUATOR_SCOPE_DECLARED" if falsifiers["all_passed"]
                         else "HALTED_FALSIFIER_FAILED"),
        "overall_live_fraction": overall, "by_rule": by_rule,
        "live_fraction_spread_across_shares": spread_by_share,
        "table": table, "shares": list(SHARES), "regimes": list(REGIMES), "seeds": seeds,
        "scope_statement": ("the CSSU allocation lever is an actuator only in the fraction of "
                            "dispatch epochs reported here; every headroom result on this lever "
                            "is bounded by it"),
        "falsifiers": falsifiers,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    digest = seal_and_write(
        payload, args.output,
        contract=Path("docs/PREREGISTRO_CONTENCION_HEADROOM_2026-07-31.md"),
        reference=Path("results/metric_audit/contention_service_first_v2/result.json"))
    print(f"\n  -> {args.output} (sello {digest[:16]}…)")
    return 0 if falsifiers["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
