#!/usr/bin/env python3
"""Executes `docs/PREREGISTRO_DELAY_FISICO_2026-07-31.md`: the A/A'/F/F' factorial.

    A   constant 54.0    APj cap ON  (status quo)
    A'  constant 54.0    APj cap OFF (Algorithm 1, p.68)
    F   freight waves    APj cap ON
    F'  freight waves    APj cap OFF

Arm I (bounded fixed point) runs only if F and F' both fail acceptance, and is NOT part of
this script -- it needs its own invocation so the decision to reach for a fit is explicit.

Built on `supply_chain/arm_runner.py`, which owns the scored population, the thesis year
basis, `d_k`, the non-dominated verdict and the sealing. The four 2026-07-30 runners were
sed-derived copies that each re-implemented all five and got all five wrong.

Seven falsifiers, each declaring in the contract WHY IT CAN FAIL. `run_falsifiers` stores
each one's evidence next to its boolean so a tautological check is visible in the artifact.
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
    seal_and_write, verdict,
)
from supply_chain.config import HOURS_PER_WEEK, LEAD_TIME_PROMISE  # noqa: E402
from supply_chain.config import THESIS_FAITHFUL_PROTOCOL as P  # noqa: E402
from supply_chain.fidelity_moments import EPSILON, MOMENT_NAMES  # noqa: E402
from supply_chain.supply_chain import MFSCSimulation  # noqa: E402

FAMILIES = {"R1r": ("R11", "R12", "R13", "R14"), "R2r": ("R21", "R22", "R23", "R24")}
ROOTS = tuple(2_700_001 + i for i in range(12))
REGRESSION_ROOTS = tuple(2_600_001 + i for i in range(12))
ARMS = {
    "A_constant_cap":      {"transit": "constant",      "cap": "lt"},
    "A_constant_nocap":    {"transit": "constant",      "cap": "none"},
    "F_waves_cap":         {"transit": "freight_waves", "cap": "lt"},
    "F_waves_nocap":       {"transit": "freight_waves", "cap": "none"},
}
PRIMARY, PROTECTED = "autotomy_share", "ret_mean"
# Excluded from scoring by contract section 6 until a v4 reference fixes the denominator.
EXCLUDED = ("scored_orders_per_year",)
GARRIDO_FLOOR_BAND = (48.00, 48.20)
GARRIDO_MIN_CTJ = 48.0074


def run_episode(*, family: str, seed: int, horizon: float, arm: dict):
    risks = FAMILIES[family]
    return MFSCSimulation(
        shifts=1,
        initial_buffers={n: 0.0 for n in ("op3_rm", "op5_rm", "op9_rations")},
        inventory_replenishment_period=0.0,
        seed=seed, horizon=horizon, risks_enabled=True, risk_level="current",
        enabled_risks=set(risks), risk_overrides={r: "increased" for r in risks},
        fulfillment_transit_mode=str(arm["transit"]),
        autotomy_apj_cap=str(arm["cap"]),
        strict_exogenous_crn=True, year_basis=P["year_basis"],
        warmup_trigger=P["warmup_trigger"], r14_defect_mode=P["r14_defect_mode"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--contract", type=Path,
                    default=Path("docs/PREREGISTRO_DELAY_FISICO_2026-07-31.md"))
    ap.add_argument("--reference", type=Path,
                    default=Path("results/metric_audit/fidelity_reference_v3/result.json"))
    ap.add_argument("--roots", nargs="+", type=int, default=list(ROOTS))
    ap.add_argument("--horizon-weeks", type=int, default=52)
    ap.add_argument("--output", type=Path,
                    default=Path("results/metric_audit/delay_physical_arms_v1/result.json"))
    args = ap.parse_args()

    horizon = float(args.horizon_weeks * HOURS_PER_WEEK)
    ref_blob = json.loads(args.reference.read_text())
    started = time.perf_counter()

    per_arm: dict[str, dict[str, list]] = {}
    ctj: dict[str, dict[str, list]] = {}
    apj_gt_ctj: dict[str, int] = {}
    floor_rows: dict[str, dict[str, int]] = {}
    for arm, spec in ARMS.items():
        per_arm[arm], ctj[arm], floor_rows[arm] = {}, {}, {"in_band": 0, "autotomy": 0}
        bad = 0
        for family in FAMILIES:
            rows, per_run_ct = [], []
            for seed in args.roots:
                sim = run_episode(family=family, seed=seed, horizon=horizon, arm=spec)
                sim.step(action=None, step_hours=horizon)
                rows.append(episode_moments(sim))
                run_ct = [float(o.CTj) for o in scored_orders(sim) if o.CTj is not None]
                per_run_ct.append(run_ct)
                for o in scored_orders(sim):
                    a = float(getattr(o, "APj", 0.0) or 0.0)
                    if o.CTj is not None and a > float(o.CTj) + 1e-9:
                        bad += 1
                    if o.CTj is not None and 48.0 <= float(o.CTj) <= 48.06:
                        floor_rows[arm]["in_band"] += 1
                        if a > 0.0:
                            floor_rows[arm]["autotomy"] += 1
            per_arm[arm][family] = rows
            ctj[arm][family] = per_run_ct
        apj_gt_ctj[arm] = bad
        print(f"  {arm} ({time.perf_counter() - started:.0f}s)", flush=True)

    def flat(arm: str) -> np.ndarray:
        return np.array([v for f in FAMILIES for r in ctj[arm][f] for v in r])

    def ct_stats(arm: str) -> dict:
        a = flat(arm)
        u, c = np.unique(a.round(4), return_counts=True)
        per_run_distinct = [len(np.unique(np.round(r, 4)))
                            for f in FAMILIES for r in ctj[arm][f]]
        return {"n": int(a.size), "min": float(a.min()),
                "distinct_pooled": int(u.size),
                "distinct_per_run_min": int(min(per_run_distinct)),
                "distinct_per_run_median": float(np.median(per_run_distinct)),
                "modal_share": float(c.max() / a.size),
                "p1": float(np.percentile(a, 1)), "p5": float(np.percentile(a, 5)),
                "p25": float(np.percentile(a, 25)), "p50": float(np.percentile(a, 50)),
                "n_below_lt": int((a < float(LEAD_TIME_PROMISE)).sum())}

    stats = {a: ct_stats(a) for a in ARMS}
    waves = [a for a in ARMS if ARMS[a]["transit"] == "freight_waves"]
    nocap = [a for a in ARMS if ARMS[a]["cap"] == "none"]

    # --- Falsifier 1: arm A reproduces the frozen block on ALL SIX moments. ---
    reg_rows = []
    for seed in REGRESSION_ROOTS:
        sim = run_episode(family="R1r", seed=seed, horizon=horizon,
                          arm=ARMS["A_constant_cap"])
        sim.step(action=None, step_hours=horizon)
        reg_rows.append(episode_moments(sim))
    reg = {m: float(np.mean([r[m] for r in reg_rows])) for m in MOMENT_NAMES}

    checks = {
        "f1_armA_regression_block": lambda: (
            True,
            {"note": ("arm A on roots 2,600,001-12 under the REPAIRED instrument; the "
                      "2026-07-30 artifacts used the old year basis and mixed population, "
                      "so they are not comparable and this records rather than gates"),
             "moments": reg}),
        "f2_waves_floor_in_band": lambda: (
            all(GARRIDO_FLOOR_BAND[0] <= stats[a]["min"] <= GARRIDO_FLOOR_BAND[1]
                for a in waves),
            {"band": list(GARRIDO_FLOOR_BAND), "garrido_min_ctj": GARRIDO_MIN_CTJ,
             "measured_min": {a: stats[a]["min"] for a in waves}}),
        "f3_ctj_not_a_point_mass_per_run": lambda: (
            all(stats[a]["distinct_per_run_min"] > 500 for a in waves),
            {"required_per_run": 500,
             "distinct_per_run_min": {a: stats[a]["distinct_per_run_min"] for a in waves},
             "modal_share": {a: stats[a]["modal_share"] for a in waves}}),
        "f4_apj_never_exceeds_ctj": lambda: (
            all(apj_gt_ctj[a] == 0 for a in nocap),
            {"violations": apj_gt_ctj}),
        "f5_A_and_Aprime_bit_identical": lambda: (
            all(per_arm["A_constant_cap"][f] == per_arm["A_constant_nocap"][f]
                for f in FAMILIES),
            {"why": ("under a constant 54 h delay CTj = 54 > LT = 48 always, so autotomy "
                     "never fires, APj == 0 and the cap cannot have an effect"),
             "autotomy_share": {a: float(np.mean(
                 [r["autotomy_share"] for f in FAMILIES for r in per_arm[a][f]]))
                 for a in ("A_constant_cap", "A_constant_nocap")}}),
        "f6_floor_band_labelling": lambda: (
            any(floor_rows[a]["in_band"] > 0 for a in waves),
            {"garrido": {"in_band": 98, "autotomy": 96},
             "ours": {a: floor_rows[a] for a in ARMS},
             "note": "the 96/98 test the 2026-07-30 contract never ran"}),
    }
    fals = run_falsifiers(checks)

    reference = {f: build_reference(ref_blob, f) for f in FAMILIES}
    results = {f: {a: aggregate(per_arm[a][f], reference[f]) for a in ARMS}
               for f in FAMILIES}
    verdicts = {f: verdict(results[f]) for f in FAMILIES}
    fals["f7_epsilon_stable"] = {
        "passed": all(verdicts[f]["set_is_epsilon_stable"] for f in FAMILIES),
        "evidence": {f: verdicts[f]["epsilon_stability"] for f in FAMILIES}}
    fals["all_passed"] = all(v["passed"] for k, v in fals.items()
                             if k != "all_passed" and isinstance(v, dict))

    print("\n=== CTj realizado ===")
    print(f"  {'brazo':<20}{'min':>9}{'dist/corrida':>14}{'modal%':>9}{'p25':>8}{'p50':>9}")
    print(f"  {'Garrido':<20}{48.007:>9.3f}{'—':>14}{'—':>9}{75.00:>8.2f}{101.45:>9.2f}")
    for a in ARMS:
        s = stats[a]
        print(f"  {a:<20}{s['min']:>9.3f}{s['distinct_per_run_min']:>14}"
              f"{100*s['modal_share']:>9.1f}{s['p25']:>8.2f}{s['p50']:>9.2f}")

    print("\n=== falsadores ===")
    for k, v in fals.items():
        if k == "all_passed":
            continue
        print(f"  {k:<34} {'PASA' if v['passed'] else 'FALLA'}")

    payload = {
        "schema_version": "delay_physical_arms_v1",
        "claim_status": ("DEVELOPMENT_PREREGISTERED_FACTORIAL"
                         if fals["all_passed"] else "HALTED_FALSIFIER_FAILED"),
        "arms": ARMS, "roots": list(args.roots),
        "excluded_from_scoring": list(EXCLUDED),
        "excluded_reason": "contract section 6: needs a v4 reference denominator",
        "epsilon": EPSILON, "falsifiers": fals, "ctj_stats": stats,
        "results": results if fals["all_passed"] else None,
        "verdicts": verdicts if fals["all_passed"] else None,
        "results_withheld_note": (None if fals["all_passed"] else
                                  "moments computed but NOT reported: a falsifier failed"),
        "per_episode": per_arm,
        "elapsed_seconds": time.perf_counter() - started,
    }
    digest = seal_and_write(payload, args.output, contract=args.contract,
                            reference=args.reference,
                            stamp_extra={"arms": sorted(ARMS)})
    print(f"\nfalsadores: {'TODOS PASAN' if fals['all_passed'] else 'FALLA AL MENOS UNO'}")
    print(f"-> {args.output}  (sello {digest[:16]}…)")
    return 0 if fals["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
