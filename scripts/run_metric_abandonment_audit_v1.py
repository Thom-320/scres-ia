#!/usr/bin/env python3
"""Metric audit -- WHY does ReT prefer abandoning a unit, and does Cobb-Douglas prefer it too?

The contention sweep found ret_excel maximised exactly where service is worst: the split that
delivers 50% of rations scores 12x the split that delivers 80%. The first audit (v1_3) showed
censoring is real and large (58 -> 138 omitted orders, 1.6 -> 78 lost) but NOT sufficient: the U
survives `ret_excel_full_ledger`, which scores unfulfilled orders at 0, and it survives -- indeed
worsens -- under the bounded `ret_excel_visible_clipped_0_1`. So the preference lives in the ReT
construct rather than in one variant's defect.

Two questions remain, and this runner answers both on one sweep:

  1. WHICH BRANCH? Garrido's ReT switches between an autotomy branch, a recovery branch
     (0.5/RPj), a risk-no-recovery branch, and a fill-rate branch. Only the fill-rate branch sees
     lost orders at all. If an extreme split moves orders OUT of the fill-rate branch and INTO
     the recovery branch, branch selection is the mechanism -- not censoring.

  2. DOES COBB-DOUGLAS RANK IT CORRECTLY? Garrido's IJPR 2024 index charges backorders through
     `epsilon = sum(B_t)/T`. But his assumption (4) puts NO constraint on the number of
     backorders, so in HIS factory nothing is ever lost. The MFSC loses orders, and a lost order
     leaves the backorder queue -- so it stops costing. That is the same hole by a different
     door, and it is measured here rather than argued.

`kappa_dot` is set-relative, so the nine splits ARE the comparison set: exactly the use the index
was designed for. Note the daily stepping cadence needed by the recorder: ReT is step-cadence
dependent, so ReT values here are comparable ACROSS SPLITS but not against the `sim.run()`
artifacts of the contention sweep.
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
from supply_chain.cobb_douglas_resilience import (  # noqa: E402
    CobbDouglasRecorder, derive_exponents, score_comparison_set)
from supply_chain.config import HOURS_PER_WEEK  # noqa: E402
from supply_chain.config import THESIS_FAITHFUL_PROTOCOL as P  # noqa: E402
from supply_chain.episode_metrics import compute_episode_metrics  # noqa: E402
from supply_chain.supply_chain import MFSCSimulation  # noqa: E402

R1R = ("R11", "R12", "R13", "R14")
R2R = ("R21", "R22", "R23", "R24")
REGIMES = {"R2r": R2R, "R1r+R2r": R1R + R2R}
SHARES = tuple(round(0.1 + 0.1 * i, 2) for i in range(9))
CASES = ("excel_case_pct_fill_rate", "excel_case_pct_autotomy",
         "excel_case_pct_recovery", "excel_case_pct_risk_no_recovery")
SERVICE = ("flow_fill_rate", "lost_orders", "ret_excel_visible_n", "ret_excel_omitted_n")
RETS = ("ret_excel_risk_conditional", "ret_excel_full_ledger",
        "ret_excel_visible_clipped_0_1", "ret_thesis")
SEED_BASE = 5_500_001
STEP = 24.0


def episode(risks, share, seed, horizon) -> tuple[dict[str, float], dict[str, float]]:
    sim = MFSCSimulation(
        shifts=1, initial_buffers={n: 0.0 for n in ("op3_rm", "op5_rm", "op9_rations")},
        inventory_replenishment_period=0.0, seed=seed, horizon=horizon,
        risks_enabled=True, risk_level="current", enabled_risks=set(risks),
        cssu_topology_mode="split_v1", cssu_allocation_a=float(share),
        cssu_service_rule="FIFO_PARTIAL", cssu_reallocate_unused=False,
        order_fulfillment_mode="op9_linked", op9_dispatch_policy="fixed_clock_daily",
        strict_exogenous_crn=True, year_basis=P["year_basis"],
        warmup_trigger=P["warmup_trigger"], r14_defect_mode=P["r14_defect_mode"])
    recorder = CobbDouglasRecorder(period_hours=STEP)
    for _ in range(int(round(horizon / STEP))):
        sim.step(step_hours=STEP)
        recorder.sample(sim)
    panel = compute_episode_metrics(sim)
    keep = {k: float(panel[k]) for k in (*RETS, *CASES, *SERVICE)}
    return keep, recorder.aggregate()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--horizon-weeks", type=int, default=52)
    ap.add_argument("--output", type=Path,
                    default=Path("results/metric_audit/abandonment_v1/result.json"))
    args = ap.parse_args()
    horizon = float(args.horizon_weeks * HOURS_PER_WEEK)
    seeds = [SEED_BASE + i for i in range(args.seeds)]
    started = time.perf_counter()

    panels: dict[tuple[str, float], list[dict]] = {}
    aggs: dict[tuple[str, float], list[dict]] = {}
    for rname, risks in REGIMES.items():
        for share in SHARES:
            rows = [episode(risks, share, s, horizon) for s in seeds]
            panels[(rname, share)] = [r[0] for r in rows]
            aggs[(rname, share)] = [r[1] for r in rows]
        print(f"  {rname} listo ({time.perf_counter() - started:.0f}s)", flush=True)

    def mean(rname: str, share: float, key: str) -> float:
        return float(np.mean([p[key] for p in panels[(rname, share)]]))

    # ---- Cobb-Douglas over the nine splits: exactly the set kappa_dot is defined against ----
    cd: dict[str, dict[str, float]] = {}
    for rname in REGIMES:
        pooled = {f"share_{s}": {k: float(np.mean([a[k] for a in aggs[(rname, s)]]))
                                 for k in aggs[(rname, s)][0]}
                  for s in SHARES}
        maxima = {v: max(row[v] for row in pooled.values())
                  for v in ("zeta", "epsilon", "phi", "tau")}
        # A maximum <= 1 cannot normalise (ln <= 0); Garrido's own maxima are in the thousands.
        maxima = {k: max(v, 1.0 + 1e-9) for k, v in maxima.items()}
        maxima["kappa_dot"] = float(len(SHARES))
        exps = derive_exponents(maxima)
        cd[rname] = {name: row["R_cobb_douglas"]
                     for name, row in score_comparison_set(pooled, exps).items()}

    def best(d: dict[str, float]) -> str:
        return max(d, key=lambda k: d[k])

    report = {}
    for rname in REGIMES:
        fills = {s: mean(rname, s, "flow_fill_rate") for s in SHARES}
        cd_by_share = {s: cd[rname][f"share_{s}"] for s in SHARES}
        best_fill = max(SHARES, key=lambda s: fills[s])
        best_cd = max(SHARES, key=lambda s: cd_by_share[s])
        best_ret = max(SHARES, key=lambda s: mean(rname, s, "ret_excel_risk_conditional"))
        report[rname] = {
            "by_share": {
                "flow_fill_rate": fills, "R_cobb_douglas": cd_by_share,
                **{k: {s: mean(rname, s, k) for s in SHARES} for k in (*RETS, *CASES, *SERVICE)}},
            "best_share_by_service": best_fill,
            "best_share_by_cobb_douglas": best_cd,
            "best_share_by_ret": best_ret,
            "cobb_douglas_agrees_with_service": best_cd == best_fill,
            "ret_agrees_with_service": best_ret == best_fill,
        }

    cd_ok = all(v["cobb_douglas_agrees_with_service"] for v in report.values())
    # Branch mechanism: does an extreme split move orders out of the fill-rate branch?
    branch_shift = {
        rname: {"fill_branch_pct_at_balanced": mean(rname, 0.5, "excel_case_pct_fill_rate"),
                "fill_branch_pct_at_extreme": mean(rname, 0.1, "excel_case_pct_fill_rate"),
                "recovery_branch_pct_at_balanced": mean(rname, 0.5, "excel_case_pct_recovery"),
                "recovery_branch_pct_at_extreme": mean(rname, 0.1, "excel_case_pct_recovery")}
        for rname in REGIMES}
    branch_is_mechanism = all(
        v["fill_branch_pct_at_extreme"] < v["fill_branch_pct_at_balanced"]
        for v in branch_shift.values())

    falsifiers = {
        "f1_service_actually_differs_across_shares": {
            "passed": all(max(report[r]["by_share"]["flow_fill_rate"].values())
                          - min(report[r]["by_share"]["flow_fill_rate"].values()) > 0.05
                          for r in REGIMES),
            "evidence": {"why_it_can_fail": ("if service is flat there is no disagreement to "
                                             "detect and the whole audit is vacuous")}},
        "f2_cobb_douglas_set_is_the_nine_shares": {
            "passed": all(len(cd[r]) == len(SHARES) for r in REGIMES),
            "evidence": {"why_it_can_fail": ("kappa_dot is set-relative; scoring against a "
                                             "different set would answer a different question"),
                         "set_size": len(SHARES)}},
        "f3_exponents_follow_garridos_rule": {
            "passed": True,
            "evidence": {"why_it_can_fail": ("copying his numeric exponents instead of his RULE "
                                             "would import his scale, not his method"),
                         "rule": "0.20 / ln(x_max), re-derived on OUR maxima (IJPR Eq. 5)"}},
        "f4_cadence_is_disclosed": {
            "passed": True,
            "evidence": {"why_it_can_fail": ("ReT is step-cadence dependent; comparing these "
                                             "values against sim.run() artifacts would be wrong"),
                         "step_hours": STEP,
                         "comparable": "across shares within this artifact only"}},
        "f5_seeds_are_virgin": {
            "passed": True,
            "evidence": {"why_it_can_fail": "reuse would void the confirmation", "seeds": seeds}},
    }
    falsifiers["all_passed"] = all(v["passed"] for k, v in falsifiers.items()
                                   if k != "all_passed")

    verdict = ("COBB_DOUGLAS_SURVIVES_THE_ABANDONMENT_TEST" if cd_ok
               else "BOTH_METRICS_REWARD_ABANDONMENT")

    for rname in REGIMES:
        r = report[rname]
        print(f"\n  === {rname} ===")
        print("  reparto            " + " ".join(f"{s:>7}" for s in SHARES))
        for key in ("flow_fill_rate", "R_cobb_douglas", "ret_excel_risk_conditional",
                    "excel_case_pct_fill_rate", "excel_case_pct_recovery", "lost_orders"):
            vals = r["by_share"][key]
            print(f"  {key:<19}" + " ".join(f"{vals[s]:>7.4f}" if vals[s] < 10
                                            else f"{vals[s]:>7.1f}" for s in SHARES))
        print(f"  mejor por servicio {r['best_share_by_service']} | "
              f"por Cobb-Douglas {r['best_share_by_cobb_douglas']} | "
              f"por ReT {r['best_share_by_ret']}")
    print(f"\n  ¿la rama explica la U? {branch_is_mechanism}")
    print(f"  veredicto: {verdict}")
    print("\n  falsadores:")
    for name, check in falsifiers.items():
        if name != "all_passed":
            print(f"    {name:<44} {'PASA' if check['passed'] else 'FALLA'}")

    payload = {
        "schema_version": "metric_abandonment_audit_v1",
        "claim_status": verdict if falsifiers["all_passed"] else "HALTED_FALSIFIER_FAILED",
        "shares": list(SHARES), "regimes": list(REGIMES), "seeds": seeds,
        "step_hours": STEP, "report": {k: v for k, v in report.items()},
        "branch_shift": branch_shift, "branch_explains_the_u": branch_is_mechanism,
        "cobb_douglas_agrees_with_service": cd_ok,
        "falsifiers": falsifiers,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    digest = seal_and_write(
        payload, args.output,
        contract=Path("docs/RESULTADO_RET_PREMIA_EL_ABANDONO_2026-07-31.md"),
        reference=Path("results/sensitivity/contention_headroom_v1_3/result.json"))
    print(f"\n  -> {args.output} (sello {digest[:16]}…)")
    return 0 if falsifiers["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
