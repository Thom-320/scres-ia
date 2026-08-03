#!/usr/bin/env python3
"""Executes `docs/PREREGISTRO_DELAY_DISTRIBUCION_2026-07-30.md`: arms A, D1, D2, D3.

    A   constant 54.0                       (status quo)
    D1  48.0074 + Exp(beta)                 beta from p50 alone
    D2  48.0074 + Lognormal(mu, sigma)      from p25 and p50
    D3  48.0074 + Weibull(k, lam)           from p25 and p50

Parameters are DERIVED here by moment matching from Garrido's own {min, p25, p50}
= {48.0074, 75.00, 101.45}. Nothing is searched. p1, p5 and p95 are RESERVED: they
never enter estimation and exist to falsify the shape.

Four falsifiers gate the report:

1. arm A reproduces the frozen autotomy-arms block on roots 2,500,001-12;
2. in D1-D3 min(CTj) >= 48.0074 and no order has CTj < LT;
3. in D1-D3 CTj shows more than 500 distinct values -- the direct test that it
   stopped being a point mass (46 today, 69.2% at one value);
4. each arm's realised p50(CTj) is within +-10% of the 101.45 target.
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
from scipy.stats import norm

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
ROOTS = tuple(2_600_001 + i for i in range(12))
REGRESSION_ROOTS = tuple(2_500_001 + i for i in range(12))

# Estimation quantiles, declared in the contract. RESERVED: p1 = 48.41, p5 = 50.42.
FLOOR, Q25, Q50 = 48.0074, 75.00, 101.45
TARGET_P50 = 101.45
PRIMARY, PROTECTED = "autotomy_share", "ret_mean"


def derive_params() -> dict[str, dict]:
    """Moment matching on {min, p25, p50}. No search, no free parameter."""
    m50, m25 = Q50 - FLOOR, Q25 - FLOOR
    mu = math.log(m50)
    sigma = (mu - math.log(m25)) / (-norm.ppf(0.25))
    k = math.log(math.log(2) / math.log(4 / 3)) / math.log(m50 / m25)
    lam = m50 / (math.log(2) ** (1 / k))
    return {
        "A_constant": {"dist": "constant", "params": {}, "floor_const": 54.0},
        "D1_exponential": {"dist": "exponential",
                           "params": {"floor": FLOOR, "beta": m50 / math.log(2)}},
        "D2_lognormal": {"dist": "lognormal",
                         "params": {"floor": FLOOR, "mu": mu, "sigma": sigma}},
        "D3_weibull": {"dist": "weibull",
                       "params": {"floor": FLOOR, "k": k, "lam": lam}},
    }


ARMS = derive_params()


def run_episode(*, family: str, seed: int, horizon: float, arm: dict):
    risks = FAMILIES[family]
    return MFSCSimulation(
        shifts=1,
        initial_buffers={n: 0.0 for n in ("op3_rm", "op5_rm", "op9_rations")},
        inventory_replenishment_period=0.0,
        seed=seed, horizon=horizon, risks_enabled=True, risk_level="current",
        enabled_risks=set(risks), risk_overrides={r: "increased" for r in risks},
        demand_on_hand_fulfillment_delay=float(arm.get("floor_const", FLOOR)),
        fulfillment_delay_distribution=str(arm["dist"]),
        fulfillment_delay_params=dict(arm["params"]),
        strict_exogenous_crn=True, year_basis=P["year_basis"],
        warmup_trigger=P["warmup_trigger"], r14_defect_mode=P["r14_defect_mode"])


def episode(sim, horizon_years: float):
    orders = [o for o in sim.orders
              if not bool(getattr(o, "metrics_excluded", False))
              and float(getattr(o, "OPTj", 0.0)) >= float(sim.warmup_time)]
    ret = [float(v) for v in ledger(orders, current_time=float(sim.env.now))["ret_values"]]
    g = lambda a: [float(getattr(o, a, 0.0) or 0.0) for o in orders]  # noqa: E731
    m = moments_from_rows(apj=g("APj"), rpj=g("RPj"), ret=ret,
                          horizon_years=horizon_years)
    ctj = [float(o.CTj) for o in orders if o.CTj is not None]
    return m, ctj


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    # --contract is REQUIRED: a default is how three artifacts got sealed against
    # the wrong document. Previous default was Path("docs/PREREGISTRO_DELAY_DISTRIBUCION_2026-07-30.md")
    ap.add_argument("--contract", type=Path,
                    required=True)
    ap.add_argument("--reference", type=Path,
                    default=Path("results/metric_audit/fidelity_reference_v3/result.json"))
    ap.add_argument("--roots", nargs="+", type=int, default=list(ROOTS))
    ap.add_argument("--horizon-weeks", type=int, default=52)
    ap.add_argument("--output", type=Path,
                    default=Path("results/metric_audit/"
                                 "fulfillment_delay_distribution_v1/result.json"))
    args = ap.parse_args()

    horizon = float(args.horizon_weeks * HOURS_PER_WEEK)
    horizon_years = float(args.horizon_weeks) / 52.0
    ref_blob = json.loads(args.reference.read_text())
    reference = ref_blob["reference_by_family"]
    started = time.perf_counter()

    per_arm: dict[str, dict[str, list]] = {}
    ctj_stats: dict[str, dict] = {}
    for arm, spec in ARMS.items():
        per_arm[arm] = {}
        allct: list[float] = []
        for family in FAMILIES:
            rows = []
            for seed in args.roots:
                sim = run_episode(family=family, seed=seed, horizon=horizon, arm=spec)
                sim.step(action=None, step_hours=horizon)
                m, ctj = episode(sim, horizon_years)
                rows.append(m)
                allct += ctj
            per_arm[arm][family] = rows
        a = np.array(allct)
        ctj_stats[arm] = {
            "n": int(a.size), "min": float(a.min()),
            "n_distinct": int(np.unique(a.round(4)).size),
            "modal_share": float(np.bincount(
                np.unique(a.round(4), return_inverse=True)[1]).max() / a.size),
            "p1": float(np.percentile(a, 1)), "p5": float(np.percentile(a, 5)),
            "p25": float(np.percentile(a, 25)), "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)),
            "n_below_lt": int((a < float(LEAD_TIME_PROMISE)).sum()),
        }
        print(f"  {arm} ({time.perf_counter() - started:.0f}s)", flush=True)

    # ---- FALSIFIERS ----
    reg = []
    for seed in REGRESSION_ROOTS:
        sim = run_episode(family="R1r", seed=seed, horizon=horizon,
                          arm=ARMS["A_constant"])
        sim.step(action=None, step_hours=horizon)
        reg.append(episode(sim, horizon_years)[0])
    reg_got = {k: float(np.mean([r[k] for r in reg]))
               for k in ("rpj_p95", "ret_mean", "autotomy_share")}
    frozen = json.loads(Path("results/metric_audit/autotomy_arms_v1/"
                             "result.json").read_text())
    expect = frozen["results"]["R1r"]["A_status_quo"]["moments"]
    f1 = all(abs(reg_got[k] - float(expect[k])) <= max(0.05 * abs(float(expect[k])), 1e-4)
             for k in reg_got)

    dists = [a for a in ARMS if a != "A_constant"]
    f2 = all(ctj_stats[a]["min"] >= FLOOR - 1e-6 and ctj_stats[a]["n_below_lt"] == 0
             for a in dists)
    f3 = all(ctj_stats[a]["n_distinct"] > 500 for a in dists)
    f4 = all(abs(ctj_stats[a]["p50"] - TARGET_P50) <= 0.10 * TARGET_P50 for a in dists)

    summary = {
        "falsifier_1_armA_reproduces_frozen": f1,
        "falsifier_1_expected": {k: float(expect[k]) for k in reg_got},
        "falsifier_1_got": reg_got,
        "falsifier_2_support_and_no_order_below_lt": f2,
        "falsifier_3_ctj_not_a_point_mass": f3,
        "falsifier_4_realised_p50_within_10pct": f4,
        "ctj_stats": ctj_stats,
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

    d = lambda f, a, m: results[f][a]["discrepancies"][m]  # noqa: E731
    qual = {}
    for arm in dists:
        worse = {f"{f}.{m}": float(d(f, arm, m) - d(f, "A_constant", m))
                 for f in FAMILIES for m in MOMENT_NAMES
                 if d(f, arm, m) - d(f, "A_constant", m) > EPSILON}
        qual[arm] = {
            "primary_improves_both": bool(all(
                d(f, arm, PRIMARY) < d(f, "A_constant", PRIMARY) for f in FAMILIES)),
            "protected_ok": bool(all(
                d(f, arm, PROTECTED) - d(f, "A_constant", PROTECTED) <= EPSILON
                for f in FAMILIES)),
            "moments_worse_beyond_epsilon": worse,
            "sum_dk_both_families": float(sum(results[f][arm]["sum_dk"] for f in FAMILIES)),
        }
        qual[arm]["qualifies"] = bool(qual[arm]["primary_improves_both"]
                                      and qual[arm]["protected_ok"] and not worse)
    winners = [a for a in dists if qual[a]["qualifies"]]
    acceptance = {
        "per_arm": qual, "qualifying_arms": winners,
        "adopted": (min(winners, key=lambda a: qual[a]["sum_dk_both_families"])
                    if winners else None),
        "epsilon": EPSILON,
    }

    print("\n=== CTj realizado (falsadores 2-4) ===")
    print(f"  {'brazo':<16}{'min':>9}{'distintos':>11}{'modal%':>9}"
          f"{'p1':>8}{'p5':>8}{'p25':>8}{'p50':>8}")
    print(f"  {'Garrido':<16}{48.0074:>9.3f}{'—':>11}{'—':>9}{48.41:>8.2f}"
          f"{50.42:>8.2f}{75.00:>8.2f}{101.45:>8.2f}")
    for a in ARMS:
        c = ctj_stats[a]
        print(f"  {a:<16}{c['min']:>9.3f}{c['n_distinct']:>11}{100*c['modal_share']:>9.1f}"
              f"{c['p1']:>8.2f}{c['p5']:>8.2f}{c['p25']:>8.2f}{c['p50']:>8.2f}")

    for family in FAMILIES:
        print(f"\n=== {family} ===")
        print(f"  {'momento (d_k)':<24}" + "".join(f"{a.split('_')[0]:>12}" for a in ARMS)
              + f"{'referencia':>12}")
        for m in MOMENT_NAMES:
            print(f"  {m:<24}" + "".join(f"{d(family, a, m):>12.1f}" for a in ARMS)
                  + f"{reference[family][m]['mean']:>12.3f}")
        print(f"  {'autotomy_share cruda':<24}"
              + "".join(f"{results[family][a]['moments']['autotomy_share']:>12.5f}"
                        for a in ARMS))
        print(f"  {'SUMA d_k':<24}"
              + "".join(f"{results[family][a]['sum_dk']:>12.1f}" for a in ARMS))
    print("\nfalsadores: PASAN")
    print(f"brazos que califican: {winners or 'ninguno'}")
    print(f"veredicto del contrato -> adoptado = {acceptance['adopted']}")

    payload = {
        "schema_version": "fulfillment_delay_distribution_v1",
        "calibration_provenance": calibration_stamp(
            note="arms differ only in the fulfilment-delay shape; LT untouched"),
        "claim_status": "DEVELOPMENT_PREREGISTERED_FOUR_ARM_DELAY_SHAPE_TEST",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "contract_path": str(args.contract),
        "contract_sha256": sha256(args.contract.read_bytes()).hexdigest(),
        "reference_path": str(args.reference),
        "reference_sha256": ref_blob.get("self_sha256"),
        "lead_time_untouched": float(LEAD_TIME_PROMISE),
        "estimation_quantiles": {"min": FLOOR, "p25": Q25, "p50": Q50},
        "reserved_quantiles": {"p1": 48.41, "p5": 50.42},
        "derivation": "moment matching, closed form; no search",
        "arms": ARMS, "roots": list(args.roots),
        "falsifiers": summary, "acceptance": acceptance,
        "selection_rule": "the contract's rule, applied verbatim",
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
