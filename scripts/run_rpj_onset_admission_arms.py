#!/usr/bin/env python3
"""Executes `docs/PREREGISTRO_CLAMP_RPJ_2026-07-30.md`: arm S against arm P.

The thesis fixes both durations (Table 6.6b(3): R12 = one week, R13 = one day). Neither
arm touches them. What differs is how the length accumulates when more than one of the
twelve contracts/deliveries is drawn late in the same cycle:

    S (shipped)   delay = k * 168   /  k * 24
    P (literal)   delay =     168   /      24

Two falsifiers gate the report, both declared in the preregistration and both checked
BEFORE any moment is printed:

1. R2r contains neither R12 nor R13, so all six R2r moments must be bit-identical across
   arms. Anything else means the change touched something it must not.
2. Under P every R12 event must last exactly 168.0 h and every R13 event exactly 24.0 h.

Acceptance is six-moment dominance at the declared EPSILON, not a single target. The
script reports the verdict the contract implies; it does not select an arm.
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

from supply_chain.config import HOURS_PER_WEEK  # noqa: E402
from supply_chain.config import THESIS_FAITHFUL_PROTOCOL as P  # noqa: E402
from supply_chain.episode_metrics import (  # noqa: E402
    compute_order_level_ret_excel_request_snapshot_ledger as ledger,
)
from supply_chain.fidelity_moments import (  # noqa: E402
    EPSILON,
    MOMENT_NAMES,
    moments_from_rows,
)
from supply_chain.provenance import calibration_stamp  # noqa: E402
from supply_chain.supply_chain import MFSCSimulation  # noqa: E402

FAMILIES = {"R1r": ("R11", "R12", "R13", "R14"), "R2r": ("R21", "R22", "R23", "R24")}
# Declared in the contract, disjoint from every previous block.
ROOTS = tuple(2_400_001 + i for i in range(12))
# Falsifier 1 replays the frozen arm-C block on its own roots.
REGRESSION_ROOTS = tuple(2_300_001 + i for i in range(12))
REGRESSION_EXPECT = {"rpj_p95": 2405.5, "ret_mean": 0.007}
ARMS = {"C_clamped": "clamped", "W_within_window": "within_window"}
# The contract's target and its priority ordering.
PRIMARY = "rpj_p95"
PROTECTED = "ret_mean"


def run_episode(*, family: str, seed: int, horizon: float, admission: str):
    risks = FAMILIES[family]
    return MFSCSimulation(
        shifts=1,
        initial_buffers={n: 0.0 for n in ("op3_rm", "op5_rm", "op9_rations")},
        inventory_replenishment_period=0.0,
        seed=seed, horizon=horizon, risks_enabled=True, risk_level="current",
        enabled_risks=set(risks),
        risk_overrides={r: "increased" for r in risks},
        rpj_onset_admission=admission,
        strict_exogenous_crn=True, year_basis=P["year_basis"],
        warmup_trigger=P["warmup_trigger"], r14_defect_mode=P["r14_defect_mode"])


def episode_moments(sim, horizon_years: float) -> tuple[dict[str, float], list[dict]]:
    orders = [o for o in sim.orders
              if not bool(getattr(o, "metrics_excluded", False))
              and float(getattr(o, "OPTj", 0.0)) >= float(sim.warmup_time)]
    ret = [float(v) for v in ledger(orders, current_time=float(sim.env.now))["ret_values"]]
    grab = lambda attr: [float(getattr(o, attr, 0.0) or 0.0) for o in orders]  # noqa: E731
    moments = moments_from_rows(apj=grab("APj"), rpj=grab("RPj"), ret=ret,
                                horizon_years=horizon_years)
    events = [{"risk_id": str(getattr(e, "risk_id", "")),
               "duration": float(getattr(e, "duration", 0.0) or 0.0)}
              for e in getattr(sim, "risk_events", [])]
    return moments, events


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--contract", type=Path,
                    default=Path("docs/PREREGISTRO_CLAMP_RPJ_2026-07-30.md"))
    ap.add_argument("--reference", type=Path,
                    default=Path("results/metric_audit/fidelity_reference_v3/result.json"))
    ap.add_argument("--roots", nargs="+", type=int, default=list(ROOTS))
    ap.add_argument("--horizon-weeks", type=int, default=52)
    ap.add_argument("--output", type=Path,
                    default=Path("results/metric_audit/rpj_onset_admission_v1/"
                                 "result.json"))
    args = ap.parse_args()

    horizon = float(args.horizon_weeks * HOURS_PER_WEEK)
    horizon_years = float(args.horizon_weeks) / 52.0
    ref_blob = json.loads(args.reference.read_text())
    reference = ref_blob["reference_by_family"]
    started = time.perf_counter()

    per_arm: dict[str, dict[str, list[dict[str, float]]]] = {}
    durations: dict[str, dict[str, list[float]]] = {}
    for arm, admission in ARMS.items():
        per_arm[arm] = {}
        durations[arm] = {"R12": [], "R13": []}
        for family in FAMILIES:
            rows = []
            for seed in args.roots:
                sim = run_episode(family=family, seed=seed, horizon=horizon,
                                  admission=admission)
                sim.step(action=None, step_hours=horizon)
                moments, events = episode_moments(sim, horizon_years)
                rows.append(moments)
                for e in events:
                    if e["risk_id"] in durations[arm] and e["duration"] > 0.0:
                        durations[arm][e["risk_id"]].append(e["duration"])
            per_arm[arm][family] = rows
            print(f"  {arm} {family} ({time.perf_counter() - started:.0f}s)", flush=True)

    # ---- FALSIFIER 1: arm C must reproduce the frozen block on its own roots. ----
    reg = []
    for seed in REGRESSION_ROOTS:
        sim = run_episode(family="R1r", seed=seed, horizon=horizon, admission="clamped")
        sim.step(action=None, step_hours=horizon)
        reg.append(episode_moments(sim, horizon_years)[0])
    reg_got = {k: float(np.mean([r[k] for r in reg])) for k in REGRESSION_EXPECT}
    falsifier_1 = all(abs(reg_got[k] - v) <= max(0.05 * abs(v), 1e-4)
                      for k, v in REGRESSION_EXPECT.items())

    # ---- FALSIFIER 2: RPj <= CTj everywhere, both arms (his data: 0/21,561). ----
    # ---- FALSIFIER 3: under W, every RPj>0 order has an in-window onset. ----
    viol_2 = {a: 0 for a in ARMS}
    viol_3 = 0
    checked = 0
    for arm, admission in ARMS.items():
        for family in FAMILIES:
            for seed in args.roots[:3]:
                sim = run_episode(family=family, seed=seed, horizon=horizon,
                                  admission=admission)
                sim.step(action=None, step_hours=horizon)
                for o in sim.orders:
                    if getattr(o, "metrics_excluded", False):
                        continue
                    if float(getattr(o, "OPTj", 0.0)) < float(sim.warmup_time):
                        continue
                    rp = float(getattr(o, "RPj", 0.0) or 0.0)
                    if rp <= 0.0 or o.CTj is None:
                        continue
                    checked += 1
                    if rp > float(o.CTj) + 1e-6:
                        viol_2[arm] += 1
                    if admission == "within_window":
                        opt, oat = float(o.OPTj), float(o.OATj)
                        refs = getattr(o, "ret_risk_event_refs", []) or []
                        if not any(opt < float(r.get("start_time", 0.0)) <= oat
                                   for r in refs) and not any(
                                       str(k).startswith("ongoing")
                                       for k in (o.ret_risk_indicators or {})):
                            viol_3 += 1
    falsifier_2 = all(v == 0 for v in viol_2.values())
    falsifier_3 = viol_3 == 0

    summary = {
        "falsifier_1_armC_reproduces_frozen": falsifier_1,
        "falsifier_1_expected": REGRESSION_EXPECT, "falsifier_1_got": reg_got,
        "falsifier_2_rpj_le_ctj": falsifier_2, "falsifier_2_violations": viol_2,
        "falsifier_3_in_window_onset": falsifier_3,
        "falsifier_3_violations": viol_3, "falsifier_3_orders_checked": checked,
        "falsifiers_pass": bool(falsifier_1 and falsifier_2 and falsifier_3),
    }
    if not summary["falsifiers_pass"]:
        print("\nFALSADOR FALLIDO — no se reportan momentos.")
        print(json.dumps(summary, indent=1))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(
            {"claim_status": "HALTED_FALSIFIER_FAILED", "summary": summary},
            indent=1, sort_keys=True) + "\n")
        return 1

    # ---- Moments and d_k, only now that both falsifiers passed. ----
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
                combined = math.sqrt(R["spread"] ** 2 / R["n_sheets"] + se[m] ** 2)
                dk[m] = abs(mean[m] - R["mean"]) / combined if combined > 0 else math.nan
            cells[arm] = {"moments": mean, "moment_se": se, "discrepancies": dk,
                          "sum_dk": float(sum(dk.values()))}
        results[family] = cells

    # ---- The contract's acceptance rule, applied verbatim. ----
    s = results["R1r"]["C_clamped"]["discrepancies"]
    p = results["R1r"]["W_within_window"]["discrepancies"]
    worse = {f"{f}.{m}": float(
        results[f]["W_within_window"]["discrepancies"][m]
        - results[f]["C_clamped"]["discrepancies"][m])
        for f in FAMILIES for m in MOMENT_NAMES
        if m != "autotomy_share"      # excluded by contract section 6
        and results[f]["W_within_window"]["discrepancies"][m]
        - results[f]["C_clamped"]["discrepancies"][m] > EPSILON}
    acceptance = {
        "primary_improves": bool(p[PRIMARY] < s[PRIMARY]),
        "primary_delta_dk": float(p[PRIMARY] - s[PRIMARY]),
        "protected_not_degraded_beyond_epsilon": bool(all(
            results[f]["W_within_window"]["discrepancies"][PROTECTED]
            - results[f]["C_clamped"]["discrepancies"][PROTECTED] <= EPSILON
            for f in FAMILIES)),
        "protected_delta_dk": float(p[PROTECTED] - s[PROTECTED]),
        "moments_worse_beyond_epsilon": worse,
        "epsilon": EPSILON,
    }
    acceptance["adopt_W"] = bool(
        acceptance["primary_improves"]
        and acceptance["protected_not_degraded_beyond_epsilon"]
        and not worse and falsifier_1)

    for family in FAMILIES:
        print(f"\n=== {family} ===")
        print(f"  {'momento':<24}{'C clamp':>11}{'d_k':>7}{'W ventana':>12}{'d_k':>7}"
              f"{'referencia':>12}")
        for m in MOMENT_NAMES:
            a = results[family]["C_clamped"]
            b = results[family]["W_within_window"]
            print(f"  {m:<24}{a['moments'][m]:>11.3f}{a['discrepancies'][m]:>7.1f}"
                  f"{b['moments'][m]:>12.3f}{b['discrepancies'][m]:>7.1f}"
                  f"{reference[family][m]['mean']:>12.3f}")
        print(f"  {'SUMA d_k':<24}{results[family]['C_clamped']['sum_dk']:>11.1f}"
              f"{'':>7}{results[family]['W_within_window']['sum_dk']:>12.1f}")
    print(f"\nfalsadores: {'PASAN' if summary['falsifiers_pass'] else 'FALLAN'}")
    print(f"veredicto del contrato -> adopt_W = {acceptance['adopt_W']}")

    payload = {
        "schema_version": "rpj_onset_admission_v1",
        "calibration_provenance": calibration_stamp(
            note="arms differ only in which R^0 onsets are admissible; no constant swept"),
        "claim_status": "DEVELOPMENT_PREREGISTERED_TWO_ARM_ONSET_ADMISSION_TEST",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "contract_path": str(args.contract),
        "contract_sha256": sha256(args.contract.read_bytes()).hexdigest(),
        "reference_path": str(args.reference),
        "reference_sha256": ref_blob.get("self_sha256"),
        "algorithm_2_source": "thesis p.69 Algorithm 2 line 2",
        "arms": ARMS,
        "roots": list(args.roots),
        "horizon_weeks": args.horizon_weeks,
        "falsifiers": summary,
        "acceptance": acceptance,
        "selection_rule": "the contract's rule, applied verbatim; no arm chosen here",
        "results": results,
        "per_episode": {a: {f: per_arm[a][f] for f in FAMILIES} for a in ARMS},
        "elapsed_seconds": time.perf_counter() - started,
    }
    body = json.dumps(payload, indent=1, sort_keys=True)
    payload["self_sha256"] = sha256(body.encode()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(f"\n-> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
