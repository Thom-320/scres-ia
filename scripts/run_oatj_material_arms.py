#!/usr/bin/env python3
"""Executes `docs/PREREGISTRO_OATJ_MATERIAL_2026-07-31.md`: A / W / L / LW.

Only section 3.1 is genuinely under test -- that LW makes `delta` emerge. The two other
predictions are declared REPLICATIONS of already-measured effects and are barred from
counting as confirmation; they gate the instrument instead.
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
ROOTS = tuple(2_900_001 + i for i in range(12))
ARMS = {
    "A_legacy_const": {"order_fulfillment_mode": "legacy_theatre_stock",
                       "fulfillment_transit_mode": "constant"},
    "W_legacy_waves": {"order_fulfillment_mode": "legacy_theatre_stock",
                       "fulfillment_transit_mode": "freight_waves"},
    "L_linked_const": {"order_fulfillment_mode": "op9_linked",
                       "fulfillment_transit_mode": "constant"},
    "LW_linked_waves": {"order_fulfillment_mode": "op9_linked",
                        "fulfillment_transit_mode": "freight_waves"},
}
GARRIDO = {"p_delayed": 0.835, "ctj_p50": 101.45, "delta_p50": 4.02}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--contract", type=Path,
                    default=Path("docs/PREREGISTRO_OATJ_MATERIAL_2026-07-31.md"))
    ap.add_argument("--reference", type=Path,
                    default=Path("results/metric_audit/fidelity_reference_v3/result.json"))
    ap.add_argument("--roots", nargs="+", type=int, default=list(ROOTS))
    ap.add_argument("--horizon-weeks", type=int, default=52)
    ap.add_argument("--output", type=Path,
                    default=Path("results/metric_audit/oatj_material_arms_v1/result.json"))
    args = ap.parse_args()
    horizon = float(args.horizon_weeks * HOURS_PER_WEEK)
    ref_blob = json.loads(args.reference.read_text())
    t0 = time.perf_counter()

    per_arm, stats = {}, {}
    for arm, kw in ARMS.items():
        per_arm[arm] = {}
        ct_pooled, per_run_distinct, mass_ok = [], [], True
        for family in FAMILIES:
            rows = []
            for seed in args.roots:
                sim = MFSCSimulation(
                    shifts=1,
                    initial_buffers={n: 0.0 for n in ("op3_rm", "op5_rm", "op9_rations")},
                    inventory_replenishment_period=0.0, seed=seed, horizon=horizon,
                    risks_enabled=True, risk_level="current",
                    enabled_risks=set(FAMILIES[family]),
                    risk_overrides={r: "increased" for r in FAMILIES[family]},
                    strict_exogenous_crn=True, year_basis=P["year_basis"],
                    warmup_trigger=P["warmup_trigger"],
                    r14_defect_mode=P["r14_defect_mode"], **kw)
                sim.step(action=None, step_hours=horizon)
                rows.append(episode_moments(sim))
                run_ct = [float(o.CTj) for o in scored_orders(sim) if o.CTj is not None]
                ct_pooled += run_ct
                per_run_distinct.append(len(np.unique(np.round(run_ct, 4))))
                if float(sim.total_delivered) > float(sim.total_produced) + 1e-6:
                    mass_ok = False
            per_arm[arm][family] = rows
        a = np.array(ct_pooled)
        k = np.floor((a - 48.0) / 24.0)
        d = (a - 48.0) - 24.0 * k
        stats[arm] = {
            "n": int(a.size), "min": float(a.min()), "p50": float(np.percentile(a, 50)),
            "distinct_per_run_min": int(min(per_run_distinct)),
            "p_delayed": float((k > 0).mean()),
            "delta_p50": float(np.percentile(d, 50)),
            "delta_gt0_share": float((d > 1e-9).mean()),
            "n_below_lt": int((a < float(LEAD_TIME_PROMISE) - 1e-9).sum()),
            "mass_balance_ok": bool(mass_ok)}
        print(f"  {arm} ({time.perf_counter() - t0:.0f}s)", flush=True)

    S = stats
    checks = {
        "R1_replication_linked_raises_conversion": lambda: (
            abs(S["L_linked_const"]["p_delayed"] - 0.493) <= 0.02
            and abs(S["A_legacy_const"]["p_delayed"] - 0.335) <= 0.02,
            {"declared": "REPLICATION -- cannot count as confirmation",
             "expected": {"A": 0.335, "L": 0.493}, "tolerance": 0.02,
             "measured": {a: S[a]["p_delayed"] for a in ("A_legacy_const", "L_linked_const")}}),
        "R2_replication_waves_drop_floor": lambda: (
            all(abs(S[a]["min"] - 48.0) <= 0.1 for a in ("W_legacy_waves", "LW_linked_waves"))
            and abs(S["A_legacy_const"]["min"] - 54.0) <= 0.1,
            {"declared": "REPLICATION -- cannot count as confirmation",
             "measured": {a: S[a]["min"] for a in ARMS}}),
        "f3_no_order_below_lead_time": lambda: (
            all(S[a]["n_below_lt"] == 0 for a in ARMS),
            {"below_lt": {a: S[a]["n_below_lt"] for a in ARMS}}),
        "f4_mass_balance_in_linked_arms": lambda: (
            all(S[a]["mass_balance_ok"] for a in ("L_linked_const", "LW_linked_waves")),
            {"ok": {a: S[a]["mass_balance_ok"] for a in ARMS}}),
        "P31_GENUINE_delta_emerges_in_LW": lambda: (
            S["LW_linked_waves"]["distinct_per_run_min"] > 500
            and 1.0 <= S["LW_linked_waves"]["delta_p50"] <= 7.0,
            {"declared": "the ONLY genuine prediction of this contract",
             "criterion": "distinct per run > 500 AND delta p50 in [1,7]",
             "measured": {a: {"distinct_per_run_min": S[a]["distinct_per_run_min"],
                              "delta_p50": S[a]["delta_p50"],
                              "delta_gt0_share": S[a]["delta_gt0_share"]} for a in ARMS}}),
    }
    fals = run_falsifiers(checks)
    reference = {f: build_reference(ref_blob, f) for f in FAMILIES}
    results = {f: {a: aggregate(per_arm[a][f], reference[f]) for a in ARMS}
               for f in FAMILIES}
    verdicts = {f: verdict(results[f]) for f in FAMILIES}
    fals["f5_epsilon_stable"] = {
        "passed": all(verdicts[f]["set_is_epsilon_stable"] for f in FAMILIES),
        "evidence": {f: verdicts[f]["epsilon_stability"] for f in FAMILIES}}
    fals["all_passed"] = all(v["passed"] for k, v in fals.items()
                             if k != "all_passed" and isinstance(v, dict))

    print(f"\n  {'brazo':<18}{'min':>7}{'p50':>8}{'dist/run':>10}"
          f"{'demoradas':>11}{'δ p50':>7}{'δ>0':>8}")
    print(f"  {'Garrido':<18}{48.01:>7.2f}{101.45:>8.2f}{'—':>10}{83.5:>10.1f}%{4.02:>7.2f}{98.5:>7.1f}%")
    for a in ARMS:
        s = S[a]
        print(f"  {a:<18}{s['min']:>7.2f}{s['p50']:>8.2f}{s['distinct_per_run_min']:>10}"
              f"{100*s['p_delayed']:>10.1f}%{s['delta_p50']:>7.2f}{100*s['delta_gt0_share']:>7.1f}%")
    print("\n  falsadores:")
    for k, v in fals.items():
        if k != "all_passed":
            print(f"    {k:<42} {'PASA' if v['passed'] else 'FALLA'}")

    payload = {
        "schema_version": "oatj_material_arms_v1",
        "claim_status": ("DEVELOPMENT_PREREGISTERED_FACTORIAL" if fals["all_passed"]
                         else "HALTED_FALSIFIER_FAILED"),
        "arms": ARMS, "roots": list(args.roots), "garrido_targets": GARRIDO,
        "ctj_stats": stats, "falsifiers": fals,
        "results": results if fals["all_passed"] else None,
        "verdicts": verdicts if fals["all_passed"] else None,
        "results_withheld_note": (None if fals["all_passed"] else
                                  "moments computed but NOT reported: a falsifier failed"),
        "per_episode": per_arm, "elapsed_seconds": time.perf_counter() - t0,
    }
    dg = seal_and_write(payload, args.output, contract=args.contract,
                        reference=args.reference, stamp_extra={"arms": sorted(ARMS)})
    print(f"\n  {'TODOS PASAN' if fals['all_passed'] else 'FALLA AL MENOS UNO'}"
          f"  ->  {args.output} (sello {dg[:16]}…)")
    return 0 if fals["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
