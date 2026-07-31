#!/usr/bin/env python3
"""Executes `docs/PREREGISTRO_ENLACE_X_ATRIBUCION_2026-07-31.md`: A / C / L / LC.

Both axes are existing options. d_k governs adoption; an SE-matched d_k is reported as a
DIAGNOSTIC to attribute a d_k change to numerator or denominator, and can adopt nothing.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.arm_runner import (  # noqa: E402
    aggregate, build_reference, episode_moments, run_falsifiers, scored_orders,
    seal_and_write,
)
from supply_chain.config import HOURS_PER_WEEK, LEAD_TIME_PROMISE  # noqa: E402
from supply_chain.config import THESIS_FAITHFUL_PROTOCOL as P  # noqa: E402
from supply_chain.fidelity_moments import EPSILON, epsilon_stability  # noqa: E402
from supply_chain.fidelity_moments import non_dominated  # noqa: E402
from supply_chain.supply_chain import MFSCSimulation  # noqa: E402

FAMILIES = {"R1r": ("R11", "R12", "R13", "R14"), "R2r": ("R21", "R22", "R23", "R24")}
ROOTS = tuple(3_200_001 + i for i in range(12))
ARMS = {
    "A_legacy_des": {"order_fulfillment_mode": "legacy_theatre_stock",
                     "risk_attribution_source": "des_events"},
    "C_legacy_causal": {"order_fulfillment_mode": "legacy_theatre_stock",
                        "risk_attribution_source": "causal_exposure"},
    "L_linked_des": {"order_fulfillment_mode": "op9_linked",
                     "risk_attribution_source": "des_events"},
    "LC_linked_causal": {"order_fulfillment_mode": "op9_linked",
                         "risk_attribution_source": "causal_exposure"},
}
SCORED = ("autotomy_share", "ret_mean", "ret_above_one_share", "rpj_mean", "rpj_p95")
PROTECTED, BASE = "ret_mean", "A_legacy_des"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--contract", type=Path,
                    default=Path("docs/PREREGISTRO_ENLACE_X_ATRIBUCION_2026-07-31.md"))
    ap.add_argument("--reference", type=Path,
                    default=Path("results/metric_audit/fidelity_reference_v3/result.json"))
    ap.add_argument("--roots", nargs="+", type=int, default=list(ROOTS))
    ap.add_argument("--horizon-weeks", type=int, default=52)
    ap.add_argument("--output", type=Path,
                    default=Path("results/metric_audit/link_x_attribution_v1/result.json"))
    args = ap.parse_args()
    horizon = float(args.horizon_weeks * HOURS_PER_WEEK)
    ref_blob = json.loads(args.reference.read_text())
    t0 = time.perf_counter()

    per_arm, aux = {}, {}
    for arm, kw in ARMS.items():
        per_arm[arm] = {}
        below_lt, rpj_no_block, ct = 0, 0, []
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
                for o in scored_orders(sim):
                    if o.CTj is None:
                        continue
                    ct.append(float(o.CTj))
                    if float(o.CTj) < float(LEAD_TIME_PROMISE) - 1e-9:
                        below_lt += 1
                    # CORRECTED 2026-07-31. The first version accepted only
                    # `causal_block_intervals` or `causal_r24_event_ids` as a physical
                    # basis, and reported 492 "leaks". Measured, those orders carry a
                    # REAL R14 event ref inside their own window (e.g. start 1032.0,
                    # duration 72.0): quantity risks attribute through
                    # `_consume_ret_quantity_risk_for_order`, not through a block
                    # interval. The leak was in the falsifier, not the code -- and the
                    # "fix" I tried for it degraded ret_mean 0.38 -> 1.79 while leaving
                    # the count unchanged. An in-window event ref counts.
                    if (kw["risk_attribution_source"] == "causal_exposure"
                            and float(getattr(o, "RPj", 0.0) or 0.0) > 0.0):
                        opt, oat = float(o.OPTj), float(o.OATj)
                        has_ref = any(opt <= float(r.get("start_time", -1)) <= oat
                                      for r in (getattr(o, "ret_risk_event_refs", None)
                                                or []))
                        if not (getattr(o, "causal_block_intervals", None)
                                or getattr(o, "causal_r24_event_ids", None)
                                or has_ref):
                            rpj_no_block += 1
            per_arm[arm][family] = rows
        aux[arm] = {"below_lt": below_lt, "rpj_without_physical_block": rpj_no_block,
                    "ctj_min": float(np.min(ct))}
        print(f"  {arm} ({time.perf_counter() - t0:.0f}s)", flush=True)

    reference = {f: build_reference(ref_blob, f) for f in FAMILIES}
    cells = {f: {a: aggregate(per_arm[a][f], reference[f]) for a in ARMS}
             for f in FAMILIES}

    # SE-matched diagnostic: same numerator, arm A's standard error. Adopts nothing.
    diag = {}
    for f in FAMILIES:
        diag[f] = {}
        for a in ARMS:
            se_base = cells[f][BASE]["moment_se"]
            row = {}
            for m in SCORED:
                R = reference[f][m]
                comb = math.sqrt(R.spread ** 2 / R.n_sheets + se_base[m] ** 2)
                row[m] = (abs(cells[f][a]["moments"][m] - R.mean) / comb
                          if comb > 0 else math.nan)
            diag[f][a] = row

    dk = lambda f, a, m: cells[f][a]["discrepancies"][m]  # noqa: E731
    verdicts = {}
    for f in FAMILIES:
        so = {a: {m: dk(f, a, m) for m in SCORED} for a in ARMS}
        stab = epsilon_stability(so)
        verdicts[f] = {"non_dominated_set": non_dominated(so, EPSILON),
                       "epsilon_stability": stab,
                       "set_is_epsilon_stable": bool(stab["stable"])}

    lvl = lambda f, a, m: cells[f][a]["moments"][m]  # noqa: E731
    checks = {
        "f2_no_order_below_lead_time": lambda: (
            all(aux[a]["below_lt"] == 0 for a in ARMS),
            {"below_lt": {a: aux[a]["below_lt"] for a in ARMS}}),
        "f3_causal_rpj_needs_physical_block": lambda: (
            all(aux[a]["rpj_without_physical_block"] == 0
                for a in ("C_legacy_causal", "LC_linked_causal")),
            {"violations": {a: aux[a]["rpj_without_physical_block"] for a in ARMS}}),
        "f4_L_and_LC_differ_on_rpj_p95": lambda: (
            abs(lvl("R1r", "L_linked_des", "rpj_p95")
                - lvl("R1r", "LC_linked_causal", "rpj_p95")) > 1e-6,
            {"why": "the same axis was silently ignored under op9_linked twice before",
             "L": lvl("R1r", "L_linked_des", "rpj_p95"),
             "LC": lvl("R1r", "LC_linked_causal", "rpj_p95")}),
        "f5_epsilon_stable": lambda: (
            all(verdicts[f]["set_is_epsilon_stable"] for f in FAMILIES),
            {f: verdicts[f]["epsilon_stability"] for f in FAMILIES}),
    }
    fals = run_falsifiers(checks)

    qual = {}
    for a in ARMS:
        if a == BASE:
            continue
        worse = {f"{f}.{m}": float(dk(f, a, m) - dk(f, BASE, m))
                 for f in FAMILIES for m in SCORED
                 if dk(f, a, m) - dk(f, BASE, m) > EPSILON}
        qual[a] = {"protected_ok": bool(all(
            dk(f, a, PROTECTED) - dk(f, BASE, PROTECTED) <= EPSILON for f in FAMILIES)),
            "moments_worse_beyond_epsilon": worse}
        qual[a]["qualifies"] = bool(qual[a]["protected_ok"] and not worse
                                    and fals["all_passed"])
    adoptable = [a for a in qual if qual[a]["qualifies"]]

    print(f"\n  === R1r: nivel y d_k (d_k gobierna; SE-apareada es DIAGNÓSTICO) ===")
    print(f"  {'momento':<22}" + "".join(f"{a.split('_')[0]:>22}" for a in ARMS))
    for m in SCORED:
        print(f"  {m:<22}" + "".join(
            f"{lvl('R1r', a, m):>9.2f}/{dk('R1r', a, m):>5.2f}/{diag['R1r'][a][m]:>5.2f}"
            for a in ARMS))
    print(f"  {'(nivel/d_k/d_k-SEap)':<22}")
    print("\n  falsadores:")
    for k, v in fals.items():
        if k != "all_passed":
            print(f"    {k:<38} {'PASA' if v['passed'] else 'FALLA'}")
    for f in FAMILIES:
        print(f"  conjunto no dominado {f}: {verdicts[f]['non_dominated_set']}")
    print(f"  brazos adoptables: {adoptable or 'ninguno'}")

    payload = {
        "schema_version": "link_x_attribution_v1",
        "claim_status": ("DEVELOPMENT_PREREGISTERED_CROSSING" if fals["all_passed"]
                         else "DEVELOPMENT_PREREGISTERED_CROSSING_FALSIFIER_FAILED"),
        "arms": ARMS, "roots": list(args.roots), "scored_moments": list(SCORED),
        "dk_governs": True,
        "se_matched_dk_is_diagnostic_only": (
            "same numerator against arm A's SE; attributes a d_k change to numerator or "
            "denominator; adopts nothing"),
        "results": cells, "se_matched_dk_diagnostic": diag,
        "verdicts": verdicts, "falsifiers": fals, "aux": aux,
        "acceptance": {"per_arm": qual, "adoptable": adoptable},
        "per_episode": per_arm, "elapsed_seconds": time.perf_counter() - t0,
    }
    dg = seal_and_write(payload, args.output, contract=args.contract,
                        reference=args.reference, stamp_extra={"arms": sorted(ARMS)})
    print(f"\n  -> {args.output} (sello {dg[:16]}…)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
