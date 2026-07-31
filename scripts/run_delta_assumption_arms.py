#!/usr/bin/env python3
"""Executes `docs/PREREGISTRO_DELTA_SUPUESTO_2026-07-31.md`: A / D / L / LD.

delta is a DECLARED ASSUMPTION, not a mechanism. Scoring an arm on delta or on any
delta-derived statistic is forbidden: it reproduces U(0,8) by construction. Arms are scored
on the other FIVE moments only. The uniformity check is recorded as a CONSTRUCTION_CHECK,
explicitly not a falsifier.
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
    seal_and_write,
)
from supply_chain.config import HOURS_PER_SHIFT, HOURS_PER_WEEK  # noqa: E402
from supply_chain.config import LEAD_TIME_PROMISE  # noqa: E402
from supply_chain.config import THESIS_FAITHFUL_PROTOCOL as P  # noqa: E402
from supply_chain.fidelity_moments import EPSILON, epsilon_stability  # noqa: E402
from supply_chain.fidelity_moments import non_dominated  # noqa: E402
from supply_chain.supply_chain import MFSCSimulation  # noqa: E402

FAMILIES = {"R1r": ("R11", "R12", "R13", "R14"), "R2r": ("R21", "R22", "R23", "R24")}
ROOTS = tuple(3_000_001 + i for i in range(12))
ARMS = {
    "A_legacy_nodelta": {"order_fulfillment_mode": "legacy_theatre_stock",
                         "fulfillment_delta_mode": "off"},
    "D_legacy_delta": {"order_fulfillment_mode": "legacy_theatre_stock",
                       "fulfillment_delta_mode": "shift_uniform"},
    "L_linked_nodelta": {"order_fulfillment_mode": "op9_linked",
                         "fulfillment_delta_mode": "off"},
    "LD_linked_delta": {"order_fulfillment_mode": "op9_linked",
                        "fulfillment_delta_mode": "shift_uniform"},
}
# scored_orders_per_year excluded by the amendment; delta-derived moments never enter.
SCORED = ("autotomy_share", "ret_mean", "ret_above_one_share", "rpj_mean", "rpj_p95")
PROTECTED = "ret_mean"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--contract", type=Path,
                    default=Path("docs/PREREGISTRO_DELTA_SUPUESTO_2026-07-31.md"))
    ap.add_argument("--reference", type=Path,
                    default=Path("results/metric_audit/fidelity_reference_v3/result.json"))
    ap.add_argument("--roots", nargs="+", type=int, default=list(ROOTS))
    ap.add_argument("--horizon-weeks", type=int, default=52)
    ap.add_argument("--output", type=Path,
                    default=Path("results/metric_audit/delta_assumption_arms_v1/result.json"))
    args = ap.parse_args()
    horizon = float(args.horizon_weeks * HOURS_PER_WEEK)
    ref_blob = json.loads(args.reference.read_text())
    t0 = time.perf_counter()

    per_arm, stats, rng_state = {}, {}, {}
    for arm, kw in ARMS.items():
        per_arm[arm] = {}
        ct = []
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
                ct += [float(o.CTj) for o in scored_orders(sim) if o.CTj is not None]
                if family == "R1r" and seed == args.roots[0]:
                    rng_state[arm] = {
                        "fulfillment": str(sim.fulfillment_rng.bit_generator.state),
                        "risk": str(sim.risk_rng.bit_generator.state),
                        "demand": str(sim.demand_rng.bit_generator.state)}
            per_arm[arm][family] = rows
        a = np.array(ct)
        k = np.floor((a - 48.0) / 24.0)
        d = (a - 48.0) - 24.0 * k
        stats[arm] = {"n": int(a.size), "min": float(a.min()),
                      "p50": float(np.percentile(a, 50)),
                      "delta_q": {q: float(np.percentile(d, q)) for q in (25, 50, 75)},
                      "distinct_pooled": int(np.unique(a.round(4)).size),
                      "p_delayed": float((k > 0).mean()),
                      "n_below_lt": int((a < float(LEAD_TIME_PROMISE) - 1e-9).sum())}
        print(f"  {arm} ({time.perf_counter() - t0:.0f}s)", flush=True)

    S = stats
    delta_arms = ("D_legacy_delta", "LD_linked_delta")
    construction_check = {
        "label": "CONSTRUCTION_CHECK -- NOT a falsifier, records only",
        "why_not_a_falsifier": "delta is DRAWN as U(0,8); reproducing it proves nothing",
        "target_U08": {25: 2.0, 50: 4.0, 75: 6.0},
        "measured": {a: S[a]["delta_q"] for a in delta_arms},
    }
    checks = {
        "f3_no_order_below_lead_time": lambda: (
            all(S[a]["n_below_lt"] == 0 for a in ARMS),
            {"below_lt": {a: S[a]["n_below_lt"] for a in ARMS}}),
        "f4_delta_off_never_touches_its_stream": lambda: (
            rng_state["A_legacy_nodelta"]["fulfillment"]
            == rng_state["L_linked_nodelta"]["fulfillment"],
            {"note": "with delta off no draw may be taken, so the stream is pristine"}),
        "f5_other_streams_unperturbed_by_delta": lambda: (
            rng_state["A_legacy_nodelta"]["risk"] == rng_state["D_legacy_delta"]["risk"]
            and rng_state["A_legacy_nodelta"]["demand"]
            == rng_state["D_legacy_delta"]["demand"],
            {"note": "the delta draw must be isolated from risk and demand"}),
    }
    fals = run_falsifiers(checks)

    reference = {f: build_reference(ref_blob, f) for f in FAMILIES}
    results, verdicts = {}, {}
    for f in FAMILIES:
        cells = {a: aggregate(per_arm[a][f], reference[f]) for a in ARMS}
        scored_only = {a: {m: c["discrepancies"][m] for m in SCORED}
                       for a, c in cells.items()}
        stab = epsilon_stability(scored_only)
        results[f] = cells
        verdicts[f] = {"non_dominated_set": non_dominated(scored_only, EPSILON),
                       "epsilon_stability": stab,
                       "set_is_epsilon_stable": bool(stab["stable"]),
                       "scored_moments": list(SCORED)}
    fals["f6_epsilon_stable"] = {
        "passed": all(verdicts[f]["set_is_epsilon_stable"] for f in FAMILIES),
        "evidence": {f: verdicts[f]["epsilon_stability"] for f in FAMILIES}}
    fals["all_passed"] = all(v["passed"] for k, v in fals.items()
                             if k != "all_passed" and isinstance(v, dict))

    dk = lambda f, a, m: results[f][a]["discrepancies"][m]  # noqa: E731
    qual = {}
    for arm in ARMS:
        if arm == "A_legacy_nodelta":
            continue
        worse = {f"{f}.{m}": float(dk(f, arm, m) - dk(f, "A_legacy_nodelta", m))
                 for f in FAMILIES for m in SCORED
                 if dk(f, arm, m) - dk(f, "A_legacy_nodelta", m) > EPSILON}
        qual[arm] = {
            "protected_ok": bool(all(
                dk(f, arm, PROTECTED) - dk(f, "A_legacy_nodelta", PROTECTED) <= EPSILON
                for f in FAMILIES)),
            "moments_worse_beyond_epsilon": worse}
        qual[arm]["qualifies"] = bool(qual[arm]["protected_ok"] and not worse
                                      and fals["all_passed"])
    adoptable = [a for a in qual if qual[a]["qualifies"]]

    print(f"\n  {'brazo':<19}{'min':>7}{'p50':>8}{'demoradas':>11}"
          + "".join(f"{'δp'+str(q):>7}" for q in (25, 50, 75)))
    print(f"  {'Garrido':<19}{48.01:>7.2f}{101.45:>8.2f}{83.5:>10.1f}%"
          f"{2.00:>7.2f}{4.02:>7.2f}{6.00:>7.2f}")
    for a in ARMS:
        s = S[a]
        print(f"  {a:<19}{s['min']:>7.2f}{s['p50']:>8.2f}{100*s['p_delayed']:>10.1f}%"
              + "".join(f"{s['delta_q'][q]:>7.2f}" for q in (25, 50, 75)))
    print(f"\n  d_k por momento puntuado (R1r):")
    print(f"    {'momento':<24}" + "".join(f"{a.split('_')[0]:>9}" for a in ARMS))
    for m in SCORED:
        print(f"    {m:<24}" + "".join(f"{dk('R1r', a, m):>9.2f}" for a in ARMS))
    print("\n  falsadores:")
    for k, v in fals.items():
        if k != "all_passed":
            print(f"    {k:<40} {'PASA' if v['passed'] else 'FALLA'}")
    print(f"\n  conjunto no dominado R1r: {verdicts['R1r']['non_dominated_set']}")
    print(f"  brazos adoptables: {adoptable or 'ninguno'}")

    payload = {
        "schema_version": "delta_assumption_arms_v1",
        "claim_status": ("DEVELOPMENT_PREREGISTERED_ASSUMPTION_TEST" if fals["all_passed"]
                         else "HALTED_FALSIFIER_FAILED"),
        "arms": ARMS, "roots": list(args.roots),
        "scored_moments": list(SCORED),
        "forbidden_as_criterion": ["delta and any delta-derived statistic",
                                   "scored_orders_per_year (amendment section 2)"],
        "construction_check": construction_check,
        "falsifiers": fals, "ctj_stats": stats,
        "results": results, "verdicts": verdicts,
        "acceptance": {"per_arm": qual, "adoptable": adoptable},
        "per_episode": per_arm, "elapsed_seconds": time.perf_counter() - t0,
    }
    dg = seal_and_write(payload, args.output, contract=args.contract,
                        reference=args.reference, stamp_extra={"arms": sorted(ARMS)})
    print(f"\n  -> {args.output} (sello {dg[:16]}…)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
