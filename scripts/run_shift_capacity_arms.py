#!/usr/bin/env python3
"""Executes `docs/PREREGISTRO_TURNO_Y_CAPACIDAD_2026-07-31.md`: A / S / C / SC.

Both variable terms are implemented as MECHANISMS, never as draws: handover is sequential
inside the 8 h shift and a day's freight carries finite rations. Drawing `delta ~ U(0,8)`
would make the contract's own shape falsifier tautological, which is the failure mode this
project spent a day removing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.arm_runner import (  # noqa: E402
    aggregate, build_reference, episode_moments, run_falsifiers, scored_orders,
    seal_and_write, verdict,
)
from supply_chain.config import HOURS_PER_WEEK, LEAD_TIME_PROMISE  # noqa: E402
from supply_chain.config import THESIS_FAITHFUL_PROTOCOL as P  # noqa: E402
from supply_chain.supply_chain import MFSCSimulation  # noqa: E402

FAMILIES = {"R1r": ("R11", "R12", "R13", "R14"), "R2r": ("R21", "R22", "R23", "R24")}
ROOTS = tuple(2_800_001 + i for i in range(12))
ARMS = {
    "A_status_quo": {},
    "S_shift": {"fulfillment_shift_mode": "shift_window"},
    "C_capacity": {"fulfillment_capacity_mode": "daily_freight"},
    "SC_both": {"fulfillment_shift_mode": "shift_window",
                "fulfillment_capacity_mode": "daily_freight"},
}
DELTA_TARGET = {25: 2.0, 50: 4.0, 75: 6.0}   # U(0,8), contract 3.1
DELTA_TOL = 0.25


def run(family: str, seed: int, horizon: float, **kw):
    risks = FAMILIES[family]
    sim = MFSCSimulation(
        shifts=1, initial_buffers={n: 0.0 for n in ("op3_rm", "op5_rm", "op9_rations")},
        inventory_replenishment_period=0.0, seed=seed, horizon=horizon,
        risks_enabled=True, risk_level="current", enabled_risks=set(risks),
        risk_overrides={r: "increased" for r in risks}, strict_exogenous_crn=True,
        year_basis=P["year_basis"], warmup_trigger=P["warmup_trigger"],
        r14_defect_mode=P["r14_defect_mode"], **kw)
    sim.step(action=None, step_hours=horizon)
    return sim


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    # --contract is REQUIRED: a default is how three artifacts got sealed against
    # the wrong document. Previous default was Path("docs/PREREGISTRO_TURNO_Y_CAPACIDAD_2026-07-31.md")
    ap.add_argument("--contract", type=Path,
                    required=True)
    ap.add_argument("--reference", type=Path,
                    default=Path("results/metric_audit/fidelity_reference_v3/result.json"))
    ap.add_argument("--roots", nargs="+", type=int, default=list(ROOTS))
    ap.add_argument("--horizon-weeks", type=int, default=52)
    ap.add_argument("--output", type=Path,
                    default=Path("results/metric_audit/shift_capacity_arms_v1/result.json"))
    args = ap.parse_args()
    horizon = float(args.horizon_weeks * HOURS_PER_WEEK)
    ref_blob = json.loads(args.reference.read_text())
    t0 = time.perf_counter()

    per_arm, ct_all, risk_tape = {}, {}, {}
    for arm, kw in ARMS.items():
        per_arm[arm], ct_all[arm] = {}, []
        for family in FAMILIES:
            rows = []
            for seed in args.roots:
                sim = run(family, seed, horizon, **kw)
                rows.append(episode_moments(sim))
                ct_all[arm] += [float(o.CTj) for o in scored_orders(sim)
                                if o.CTj is not None]
                if family == "R1r" and seed == args.roots[0]:
                    risk_tape[arm] = [(e.risk_id, e.start_time, e.end_time, e.duration)
                                      for e in sim.risk_events]
            per_arm[arm][family] = rows
        print(f"  {arm} ({time.perf_counter() - t0:.0f}s)", flush=True)

    def decomp(arm):
        a = np.array(ct_all[arm])
        k = np.floor((a - 48.0) / 24.0)
        d = (a - 48.0) - 24.0 * k
        u = np.unique(a.round(4))
        return {"n": int(a.size), "min": float(a.min()), "distinct": int(u.size),
                "p25": float(np.percentile(a, 25)), "p50": float(np.percentile(a, 50)),
                "delta_q": {q: float(np.percentile(d, q)) for q in (25, 50, 75)},
                "k_gt0_share": float((k > 0).mean()),
                "band_60_72": float(((a >= 60) & (a < 72)).mean()),
                "band_72_84": float(((a >= 72) & (a < 84)).mean()),
                "band_84_96": float(((a >= 84) & (a < 96)).mean())}

    st = {a: decomp(a) for a in ARMS}
    shift_arms, cap_arms = ("S_shift", "SC_both"), ("C_capacity", "SC_both")

    checks = {
        "f3_1_delta_is_uniform_0_8": lambda: (
            all(abs(st[a]["delta_q"][q] - DELTA_TARGET[q]) <= DELTA_TOL
                for a in shift_arms for q in DELTA_TARGET),
            {"target": DELTA_TARGET, "tolerance": DELTA_TOL,
             "measured": {a: st[a]["delta_q"] for a in shift_arms}}),
        "f3_2_bands_with_empty_gaps": lambda: (
            all(st[a]["band_60_72"] < 0.01 and st[a]["band_84_96"] < 0.01
                and st[a]["band_72_84"] > 0.10 for a in cap_arms),
            {"rule": "gaps < 1% each, [72,84) > 10%",
             "measured": {a: {k: st[a][k] for k in
                              ("band_60_72", "band_72_84", "band_84_96")}
                          for a in cap_arms}}),
        "f3_3_SC_reconstructs_quantiles": lambda: (
            abs(st["SC_both"]["p25"] - 75.0) <= 7.5
            and abs(st["SC_both"]["p50"] - 101.45) <= 10.145,
            {"target": {"p25": 75.0, "p50": 101.45}, "band": "+-10%",
             "measured": {k: st["SC_both"][k] for k in ("p25", "p50")}}),
        "f2_support": lambda: (
            all(st[a]["min"] >= 48.0 - 1e-9 for a in ARMS),
            {"min": {a: st[a]["min"] for a in ARMS},
             "lead_time": float(LEAD_TIME_PROMISE)}),
        "f5_shift_does_not_touch_upstream": lambda: (
            risk_tape["A_status_quo"] == risk_tape["S_shift"],
            {"note": "risk_events must be bitwise identical between A and S"}),
        "f6_ctj_not_a_point_mass": lambda: (
            st["SC_both"]["distinct"] > 500,
            {"required": 500, "measured": st["SC_both"]["distinct"]}),
    }
    fals = run_falsifiers(checks)
    reference = {f: build_reference(ref_blob, f) for f in FAMILIES}
    results = {f: {a: aggregate(per_arm[a][f], reference[f]) for a in ARMS}
               for f in FAMILIES}
    verdicts = {f: verdict(results[f]) for f in FAMILIES}
    fals["f8_epsilon_stable"] = {
        "passed": all(verdicts[f]["set_is_epsilon_stable"] for f in FAMILIES),
        "evidence": {f: verdicts[f]["epsilon_stability"] for f in FAMILIES}}
    fals["all_passed"] = all(v["passed"] for k, v in fals.items()
                             if k != "all_passed" and isinstance(v, dict))

    print(f"\n  {'brazo':<15}{'min':>8}{'dist':>7}{'p25':>8}{'p50':>9}"
          f"{'δp25':>7}{'δp50':>7}{'δp75':>7}")
    print(f"  {'Garrido':<15}{48.01:>8.2f}{'—':>7}{75.00:>8.2f}{101.45:>9.2f}"
          f"{2.00:>7.2f}{4.02:>7.2f}{6.00:>7.2f}")
    for a in ARMS:
        s = st[a]
        print(f"  {a:<15}{s['min']:>8.2f}{s['distinct']:>7}{s['p25']:>8.2f}"
              f"{s['p50']:>9.2f}" + "".join(f"{s['delta_q'][q]:>7.2f}" for q in (25, 50, 75)))
    print("\n  falsadores:")
    for k, v in fals.items():
        if k != "all_passed":
            print(f"    {k:<34} {'PASA' if v['passed'] else 'FALLA'}")

    payload = {
        "schema_version": "shift_capacity_arms_v1",
        "claim_status": ("DEVELOPMENT_PREREGISTERED_FACTORIAL" if fals["all_passed"]
                         else "HALTED_FALSIFIER_FAILED"),
        "arms": ARMS, "roots": list(args.roots), "ctj_decomposition": st,
        "falsifiers": fals,
        "results": results if fals["all_passed"] else None,
        "verdicts": verdicts if fals["all_passed"] else None,
        "results_withheld_note": (None if fals["all_passed"] else
                                  "moments computed but NOT reported: a falsifier failed"),
        "per_episode": per_arm, "elapsed_seconds": time.perf_counter() - t0,
    }
    d = seal_and_write(payload, args.output, contract=args.contract,
                       reference=args.reference, stamp_extra={"arms": sorted(ARMS)})
    print(f"\n  {'TODOS PASAN' if fals['all_passed'] else 'FALLA AL MENOS UNO'}"
          f"  ->  {args.output} (sello {d[:16]}…)")
    return 0 if fals["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
