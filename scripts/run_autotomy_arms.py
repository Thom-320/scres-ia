#!/usr/bin/env python3
"""Executes `docs/PREREGISTRO_AUTOTOMIA_2026-07-30.md`: arms A, B and C.

    A  floor 54.0      predicate CTj <= LTj          (status quo)
    B  floor 48.0074   predicate CTj <= LTj          (isolates the floor)
    C  floor 48.0074   predicate CTj - LTj <= tol    (tol in {0.01, 0.05, 0.10})

`LEAD_TIME_PROMISE = 48` is untouched in every arm. The floor is `min(CTj)` from
Garrido's own nine R1r sheets; the tolerance covers his observed autotomy band
[0.0074, 0.048]. Both are declared fits of his PUBLISHED classification, per the
contract's section 7, and the tolerance sweep is reported whole with no selection.

Four falsifiers gate the report, checked before any moment is printed:

1. arm A reproduces the frozen block on roots 2,400,001-12;
2. arm B leaves autotomy_share at exactly 0.000 -- if it fires, the run stops;
3. in B and C the unblocked floor is 48.0074 and no order has CTj < LT;
4. in C no order with APj > 0 also carries RPj > 0 (his data: 96/96).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.config import HOURS_PER_WEEK, LEAD_TIME_PROMISE  # noqa: E402
from supply_chain.config import THESIS_FAITHFUL_PROTOCOL as P  # noqa: E402
from supply_chain.episode_metrics import (  # noqa: E402
    compute_order_level_ret_excel_request_snapshot_ledger as ledger,
)
from supply_chain.fidelity_moments import (  # noqa: E402
    EPSILON, MOMENT_NAMES, moments_from_rows,
)
from supply_chain.provenance import calibration_stamp  # noqa: E402
from supply_chain.supply_chain import MFSCSimulation  # noqa: E402

FAMILIES = {"R1r": ("R11", "R12", "R13", "R14"), "R2r": ("R21", "R22", "R23", "R24")}
ROOTS = tuple(2_500_001 + i for i in range(12))
REGRESSION_ROOTS = tuple(2_400_001 + i for i in range(12))
REGRESSION_EXPECT = {"rpj_p95": 2440.6, "ret_mean": 0.007}

GARRIDO_FLOOR = 48.0074          # min(CTj) across his nine R1r sheets
TOLERANCES = (0.01, 0.05, 0.10)  # 0.05 is the declared primary
PRIMARY_TOL = 0.05
PRIMARY = "autotomy_share"
PROTECTED = "ret_mean"

ARMS: dict[str, dict] = {
    "A_status_quo": {"floor": 54.0, "predicate": "le", "tol": 0.0},
    "B_floor_only": {"floor": GARRIDO_FLOOR, "predicate": "le", "tol": 0.0},
}
for _t in TOLERANCES:
    ARMS[f"C_band_tol{_t:g}"] = {"floor": GARRIDO_FLOOR, "predicate": "band", "tol": _t}
PRIMARY_ARM = f"C_band_tol{PRIMARY_TOL:g}"


def run_episode(*, family: str, seed: int, horizon: float, arm: dict):
    risks = FAMILIES[family]
    return MFSCSimulation(
        shifts=1,
        initial_buffers={n: 0.0 for n in ("op3_rm", "op5_rm", "op9_rations")},
        inventory_replenishment_period=0.0,
        seed=seed, horizon=horizon, risks_enabled=True, risk_level="current",
        enabled_risks=set(risks), risk_overrides={r: "increased" for r in risks},
        demand_on_hand_fulfillment_delay=float(arm["floor"]),
        autotomy_predicate=str(arm["predicate"]),
        autotomy_tolerance_hours=float(arm["tol"]),
        strict_exogenous_crn=True, year_basis=P["year_basis"],
        warmup_trigger=P["warmup_trigger"], r14_defect_mode=P["r14_defect_mode"])


def scored(sim) -> list:
    return [o for o in sim.orders
            if not bool(getattr(o, "metrics_excluded", False))
            and float(getattr(o, "OPTj", 0.0)) >= float(sim.warmup_time)]


def episode_moments(sim, horizon_years: float):
    orders = scored(sim)
    ret = [float(v) for v in ledger(orders, current_time=float(sim.env.now))["ret_values"]]
    g = lambda a: [float(getattr(o, a, 0.0) or 0.0) for o in orders]  # noqa: E731
    return moments_from_rows(apj=g("APj"), rpj=g("RPj"), ret=ret,
                             horizon_years=horizon_years), orders


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--contract", type=Path,
                    default=Path("docs/PREREGISTRO_AUTOTOMIA_2026-07-30.md"))
    ap.add_argument("--reference", type=Path,
                    default=Path("results/metric_audit/fidelity_reference_v3/result.json"))
    ap.add_argument("--roots", nargs="+", type=int, default=list(ROOTS))
    ap.add_argument("--horizon-weeks", type=int, default=52)
    ap.add_argument("--output", type=Path,
                    default=Path("results/metric_audit/autotomy_arms_v1/result.json"))
    args = ap.parse_args()

    horizon = float(args.horizon_weeks * HOURS_PER_WEEK)
    horizon_years = float(args.horizon_weeks) / 52.0
    ref_blob = json.loads(args.reference.read_text())
    reference = ref_blob["reference_by_family"]
    started = time.perf_counter()

    per_arm: dict[str, dict[str, list]] = {}
    floors: dict[str, float] = {}
    below_lt: dict[str, int] = {}
    apj_and_rpj: dict[str, int] = {}
    for arm, spec in ARMS.items():
        per_arm[arm] = {}
        fl, bl, ar = float("inf"), 0, 0
        for family in FAMILIES:
            rows = []
            for seed in args.roots:
                sim = run_episode(family=family, seed=seed, horizon=horizon, arm=spec)
                sim.step(action=None, step_hours=horizon)
                moments, orders = episode_moments(sim, horizon_years)
                rows.append(moments)
                for o in orders:
                    if o.CTj is None:
                        continue
                    ct = float(o.CTj)
                    fl = min(fl, ct)
                    if ct < float(LEAD_TIME_PROMISE):
                        bl += 1
                    if (float(getattr(o, "APj", 0.0) or 0.0) > 0.0
                            and float(getattr(o, "RPj", 0.0) or 0.0) > 0.0):
                        ar += 1
            per_arm[arm][family] = rows
        floors[arm], below_lt[arm], apj_and_rpj[arm] = fl, bl, ar
        print(f"  {arm} ({time.perf_counter() - started:.0f}s)", flush=True)

    # ---- FALSIFIER 1 ----
    reg = []
    for seed in REGRESSION_ROOTS:
        sim = run_episode(family="R1r", seed=seed, horizon=horizon,
                          arm=ARMS["A_status_quo"])
        sim.step(action=None, step_hours=horizon)
        reg.append(episode_moments(sim, horizon_years)[0])
    reg_got = {k: float(np.mean([r[k] for r in reg])) for k in REGRESSION_EXPECT}
    f1 = all(abs(reg_got[k] - v) <= max(0.05 * abs(v), 1e-4)
             for k, v in REGRESSION_EXPECT.items())

    # ---- FALSIFIER 2: arm B must NOT fire autotomy. ----
    b_auto = max(float(np.mean([r["autotomy_share"] for r in per_arm["B_floor_only"][f]]))
                 for f in FAMILIES)
    f2 = b_auto == 0.0

    # ---- FALSIFIER 3: floor and no order below LT, in B and C. ----
    f3 = all(abs(floors[a] - GARRIDO_FLOOR) < 1e-3 and below_lt[a] == 0
             for a in ARMS if a != "A_status_quo")

    # ---- FALSIFIER 4: in C, APj>0 never coincides with RPj>0. ----
    f4 = all(apj_and_rpj[a] == 0 for a in ARMS if a.startswith("C_"))

    summary = {
        "falsifier_1_armA_reproduces_frozen": f1,
        "falsifier_1_expected": REGRESSION_EXPECT, "falsifier_1_got": reg_got,
        "falsifier_2_armB_autotomy_zero": f2, "falsifier_2_armB_max_share": b_auto,
        "falsifier_3_floor_and_no_order_below_lt": f3,
        "falsifier_3_floors": floors, "falsifier_3_below_lt": below_lt,
        "falsifier_4_apj_never_with_rpj": f4,
        "falsifier_4_violations": apj_and_rpj,
        "falsifiers_pass": bool(f1 and f2 and f3 and f4),
    }
    if not summary["falsifiers_pass"]:
        print("\nFALSADOR FALLIDO — no se reportan momentos.")
        print(json.dumps(summary, indent=1, default=str))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(
            {"claim_status": "HALTED_FALSIFIER_FAILED", "summary": summary},
            indent=1, sort_keys=True, default=str) + "\n")
        return 1

    results: dict = {}
    for family in FAMILIES:
        cells = {}
        for arm in ARMS:
            rows = per_arm[arm][family]
            mean = {m: float(np.mean([r[m] for r in rows])) for m in MOMENT_NAMES}
            se = {m: float(np.std([r[m] for r in rows], ddof=1) / math.sqrt(len(rows)))
                  for m in MOMENT_NAMES}
            dk = {}
            for m in MOMENT_NAMES:
                R = reference[family][m]
                comb = math.sqrt(R["spread"] ** 2 / R["n_sheets"] + se[m] ** 2)
                dk[m] = abs(mean[m] - R["mean"]) / comb if comb > 0 else math.nan
            cells[arm] = {"moments": mean, "moment_se": se, "discrepancies": dk,
                          "sum_dk": float(sum(dk.values()))}
        results[family] = cells

    dk = lambda f, a, m: results[f][a]["discrepancies"][m]  # noqa: E731
    worse = {f"{f}.{m}": float(dk(f, PRIMARY_ARM, m) - dk(f, "A_status_quo", m))
             for f in FAMILIES for m in MOMENT_NAMES
             if dk(f, PRIMARY_ARM, m) - dk(f, "A_status_quo", m) > EPSILON}
    acceptance = {
        "primary_arm": PRIMARY_ARM,
        "primary_improves_both_families": bool(all(
            dk(f, PRIMARY_ARM, PRIMARY) < dk(f, "A_status_quo", PRIMARY)
            for f in FAMILIES)),
        "primary_delta_dk": {f: float(dk(f, PRIMARY_ARM, PRIMARY)
                                      - dk(f, "A_status_quo", PRIMARY)) for f in FAMILIES},
        "protected_not_degraded_beyond_epsilon": bool(all(
            dk(f, PRIMARY_ARM, PROTECTED) - dk(f, "A_status_quo", PROTECTED) <= EPSILON
            for f in FAMILIES)),
        "protected_delta_dk": {f: float(dk(f, PRIMARY_ARM, PROTECTED)
                                        - dk(f, "A_status_quo", PROTECTED)) for f in FAMILIES},
        "moments_worse_beyond_epsilon": worse,
        "epsilon": EPSILON,
    }
    acceptance["adopt_C"] = bool(
        acceptance["primary_improves_both_families"]
        and acceptance["protected_not_degraded_beyond_epsilon"] and not worse)

    for family in FAMILIES:
        print(f"\n=== {family} ===")
        head = "".join(f"{a.replace('_status_quo','').replace('_floor_only','').replace('C_band_tol','C tol'):>13}"
                       for a in ARMS)
        print(f"  {'momento':<24}{head}   {'referencia':>11}")
        for m in MOMENT_NAMES:
            vals = "".join(f"{results[family][a]['discrepancies'][m]:>13.1f}" for a in ARMS)
            print(f"  {m + ' (d_k)':<24}{vals}   {reference[family][m]['mean']:>11.3f}")
        print(f"  {'autotomy_share cruda':<24}"
              + "".join(f"{results[family][a]['moments']['autotomy_share']:>13.5f}"
                        for a in ARMS))
        print(f"  {'SUMA d_k':<24}"
              + "".join(f"{results[family][a]['sum_dk']:>13.1f}" for a in ARMS))
    print(f"\nfalsadores: PASAN")
    print(f"veredicto del contrato -> adopt_C = {acceptance['adopt_C']}")

    payload = {
        "schema_version": "autotomy_arms_v1",
        "calibration_provenance": calibration_stamp(
            note="arms differ in the CTj floor and the autotomy predicate; LT untouched"),
        "claim_status": "DEVELOPMENT_PREREGISTERED_THREE_ARM_AUTOTOMY_TEST",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "contract_path": str(args.contract),
        "contract_sha256": sha256(args.contract.read_bytes()).hexdigest(),
        "reference_path": str(args.reference),
        "reference_sha256": ref_blob.get("self_sha256"),
        "lead_time_untouched": float(LEAD_TIME_PROMISE),
        "garrido_floor_source": "min(CTj) over his nine R1r sheets = 48.0074",
        "tolerances_swept": list(TOLERANCES), "primary_tolerance": PRIMARY_TOL,
        "declared_fit_disclosure": ("floor and tolerance come from his data, not the "
                                    "thesis; both reproduce his published classification "
                                    "-- contract section 7"),
        "arms": ARMS, "roots": list(args.roots),
        "falsifiers": summary, "acceptance": acceptance,
        "selection_rule": "the contract's rule, applied verbatim; no arm chosen here",
        "results": results,
        "per_episode": {a: {f: per_arm[a][f] for f in FAMILIES} for a in ARMS},
        "elapsed_seconds": time.perf_counter() - started,
    }
    body = json.dumps(payload, indent=1, sort_keys=True, default=str)
    payload["self_sha256"] = sha256(body.encode()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1, sort_keys=True, default=str) + "\n")
    print(f"\n-> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
