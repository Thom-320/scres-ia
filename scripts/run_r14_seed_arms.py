#!/usr/bin/env python3
"""Executes `docs/PREREGISTRO_SIEMBRA_R0_R14_2026-07-31.md`: A / N / E.

The strong prediction is a SHAPE -- that arm E reproduces the saturation of RPj against
CTj -- not a moment. Every other axis stays at its default, because this session measured
repeatedly that these axes do not compose.
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
ROOTS = tuple(3_400_001 + i for i in range(12))
ARMS = {"A_pending_min": "pending_min", "N_no_seed": "none", "E_event_time": "event_time"}
SCORED = ("autotomy_share", "ret_mean", "ret_above_one_share", "rpj_mean", "rpj_p95")
PROTECTED, BASE = "ret_mean", "A_pending_min"
GARRIDO_RATIO_P95 = 0.20   # his RPj/CTj at p95
SAT_CRITERION = 0.60       # contract 4.1: median RPj/CTj below this for CTj > 500


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--contract", type=Path,
                    default=Path("docs/PREREGISTRO_SIEMBRA_R0_R14_2026-07-31.md"))
    ap.add_argument("--reference", type=Path,
                    default=Path("results/metric_audit/fidelity_reference_v3/result.json"))
    ap.add_argument("--roots", nargs="+", type=int, default=list(ROOTS))
    ap.add_argument("--horizon-weeks", type=int, default=52)
    ap.add_argument("--output", type=Path,
                    default=Path("results/metric_audit/r14_seed_arms_v1/result.json"))
    args = ap.parse_args()
    horizon = float(args.horizon_weeks * HOURS_PER_WEEK)
    ref_blob = json.loads(args.reference.read_text())
    t0 = time.perf_counter()

    per_arm, shape = {}, {}
    for arm, mode in ARMS.items():
        per_arm[arm] = {}
        ratios_long, ratios_all, viol_rpj_gt_ct, viol_seed, below = [], [], 0, 0, 0
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
                    r14_r0_seed_mode=mode, strict_exogenous_crn=True,
                    year_basis=P["year_basis"], warmup_trigger=P["warmup_trigger"],
                    r14_defect_mode=P["r14_defect_mode"])
                sim.step(action=None, step_hours=horizon)
                rows.append(episode_moments(sim))
                r14_times = {round(float(e.start_time), 6) for e in sim.risk_events
                             if str(getattr(e, "risk_id", "")) == "R14"}
                for o in scored_orders(sim):
                    if o.CTj is None:
                        continue
                    ct = float(o.CTj)
                    rp = float(getattr(o, "RPj", 0.0) or 0.0)
                    if ct < float(LEAD_TIME_PROMISE) - 1e-9:
                        below += 1
                    if rp > ct + 1e-6:
                        viol_rpj_gt_ct += 1
                    if rp > 0.0:
                        ratios_all.append(rp / ct)
                        if ct > 500.0:
                            ratios_long.append(rp / ct)
                    if mode == "event_time" and rp > 0.0:
                        onset = float(o.OATj) - rp
                        only_r14 = {str(k).split("_")[0]
                                    for k in (o.ret_risk_indicators or {})} == {"R14"}
                        if only_r14 and round(onset, 6) not in r14_times:
                            viol_seed += 1
            per_arm[arm][family] = rows
        shape[arm] = {
            "ratio_p50_all": float(np.median(ratios_all)) if ratios_all else float("nan"),
            "ratio_p95_all": float(np.percentile(ratios_all, 95)) if ratios_all else float("nan"),
            "n_long": len(ratios_long),
            "ratio_p50_ctj_gt_500": (float(np.median(ratios_long)) if ratios_long
                                     else float("nan")),
            "rpj_gt_ctj": viol_rpj_gt_ct, "below_lt": below,
            "r14_onsets_not_a_real_event": viol_seed}
        print(f"  {arm} ({time.perf_counter() - t0:.0f}s)", flush=True)

    reference = {f: build_reference(ref_blob, f) for f in FAMILIES}
    cells = {f: {a: aggregate(per_arm[a][f], reference[f]) for a in ARMS} for f in FAMILIES}
    diag = {}
    for f in FAMILIES:
        diag[f] = {}
        for a in ARMS:
            se_b = cells[f][BASE]["moment_se"]
            row = {}
            for m in SCORED:
                R = reference[f][m]
                c = math.sqrt(R.spread ** 2 / R.n_sheets + se_b[m] ** 2)
                row[m] = (abs(cells[f][a]["moments"][m] - R.mean) / c if c > 0 else math.nan)
            diag[f][a] = row
    dk = lambda f, a, m: cells[f][a]["discrepancies"][m]  # noqa: E731
    verdicts = {}
    for f in FAMILIES:
        so = {a: {m: dk(f, a, m) for m in SCORED} for a in ARMS}
        st = epsilon_stability(so, (0.25, 0.5, 1.0, 2.0))
        verdicts[f] = {"non_dominated_set": non_dominated(so, EPSILON),
                       "epsilon_stability": st, "set_is_epsilon_stable": bool(st["stable"])}

    checks = {
        "P41_GENUINE_E_saturates": lambda: (
            shape["E_event_time"]["ratio_p50_ctj_gt_500"] < SAT_CRITERION,
            {"declared": "the contract's strong SHAPE prediction",
             "criterion": f"median RPj/CTj < {SAT_CRITERION} for CTj > 500",
             "garrido_ratio_p95": GARRIDO_RATIO_P95,
             "measured": {a: shape[a]["ratio_p50_ctj_gt_500"] for a in ARMS},
             "n_long": {a: shape[a]["n_long"] for a in ARMS}}),
        "f2_support_and_rpj_le_ctj": lambda: (
            all(shape[a]["below_lt"] == 0 and shape[a]["rpj_gt_ctj"] == 0 for a in ARMS),
            {a: {k: shape[a][k] for k in ("below_lt", "rpj_gt_ctj")} for a in ARMS}),
        "f3_E_onsets_are_real_events": lambda: (
            shape["E_event_time"]["r14_onsets_not_a_real_event"] == 0,
            {"violations": shape["E_event_time"]["r14_onsets_not_a_real_event"],
             "scope": "orders whose ONLY indicator is R14"}),
        "f4_three_arms_differ": lambda: (
            len({round(cells["R1r"][a]["moments"]["rpj_mean"], 6) for a in ARMS}) == 3,
            {"rpj_mean": {a: cells["R1r"][a]["moments"]["rpj_mean"] for a in ARMS},
             "why": "three axes were silently ignored earlier this session"}),
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
        qual[a] = {"shape_ok": bool(shape[a]["ratio_p50_ctj_gt_500"] < SAT_CRITERION),
                   "protected_ok": bool(all(
                       dk(f, a, PROTECTED) - dk(f, BASE, PROTECTED) <= EPSILON
                       for f in FAMILIES)),
                   "moments_worse_beyond_epsilon": worse}
        qual[a]["qualifies"] = bool(qual[a]["shape_ok"] and qual[a]["protected_ok"]
                                    and not worse and fals["all_passed"])
    adoptable = [a for a in qual if qual[a]["qualifies"]]

    print(f"\n  === firma de forma: RPj/CTj (Garrido p95 ≈ {GARRIDO_RATIO_P95}) ===")
    print(f"  {'brazo':<16}{'p50 todas':>11}{'p95 todas':>11}{'p50 CTj>500':>13}{'n largas':>10}")
    for a in ARMS:
        s = shape[a]
        print(f"  {a:<16}{s['ratio_p50_all']:>11.3f}{s['ratio_p95_all']:>11.3f}"
              f"{s['ratio_p50_ctj_gt_500']:>13.3f}{s['n_long']:>10}")
    print(f"\n  === R1r: nivel / d_k / d_k-SE apareada ===")
    print(f"  {'momento':<22}" + "".join(f"{a.split('_')[0]:>23}" for a in ARMS))
    for m in SCORED:
        print(f"  {m:<22}" + "".join(
            f"{cells['R1r'][a]['moments'][m]:>9.2f}/{dk('R1r', a, m):>6.2f}/{diag['R1r'][a][m]:>5.2f}"
            for a in ARMS))
    print("\n  falsadores:")
    for k, v in fals.items():
        if k != "all_passed":
            print(f"    {k:<32} {'PASA' if v['passed'] else 'FALLA'}")
    for f in FAMILIES:
        print(f"  conjunto no dominado {f}: {verdicts[f]['non_dominated_set']}")
    print(f"  brazos adoptables: {adoptable or 'ninguno'}")

    payload = {
        "schema_version": "r14_seed_arms_v1",
        "claim_status": ("DEVELOPMENT_PREREGISTERED_R14_SEED_TEST" if fals["all_passed"]
                         else "DEVELOPMENT_PREREGISTERED_R14_SEED_TEST_FALSIFIER_FAILED"),
        "arms": ARMS, "roots": list(args.roots), "scored_moments": list(SCORED),
        "shape": shape, "results": cells, "se_matched_dk_diagnostic": diag,
        "verdicts": verdicts, "falsifiers": fals,
        "acceptance": {"per_arm": qual, "adoptable": adoptable},
        "per_episode": per_arm, "elapsed_seconds": time.perf_counter() - t0,
    }
    dg = seal_and_write(payload, args.output, contract=args.contract,
                        reference=args.reference, stamp_extra={"arms": sorted(ARMS)})
    print(f"\n  -> {args.output} (sello {dg[:16]}…)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
