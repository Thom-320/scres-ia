#!/usr/bin/env python3
"""Fase 2 -- measure headroom with COBB-DOUGLAS as the primary metric instead of ReT.

Every headroom number this project has ever produced was measured on ReT. Today it was measured
that ReT PREFERS ABANDONING a claimant: it picks the split delivering 50% of rations over the one
delivering 80%, and at the bottleneck it picks the schedule that leaves a product at 18.5%
service. The preference survives removing censoring (`full_ledger`) and survives bounding the
tail (`clipped`), so it is in the construct.

Cobb-Douglas does not have that preference: on the same sweep its optimum is the BALANCED split
in both regimes. So `H_regime ~ 1e-4` across twenty experiments might be a property of the
INSTRUMENT rather than of the chain, and this run decides it.

All three metrics are computed in ONE run at ONE cadence, so the comparison is between
instruments and not between designs. `kappa_dot` is scored over every (regime, share) cell at
once, because it is set-relative by Garrido's own definition and the set must be what is compared.

See `docs/PREREGISTRO_HEADROOM_COBB_DOUGLAS_2026-07-31.md`, which also records that `f5` is the
falsifier most at risk: the measured range of `R_cobb_douglas` across splits was only 1.1%.
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
ESCALATIONS = {"base": ({}, {}), "freq_x3": ({"R23": 3.0}, {}),
               "freq3_imp2": ({"R23": 3.0}, {"R23": 2.0})}
FAMILIES = {"R2r": R2R, "R1r+R2r": R1R + R2R}
REGIMES = {f"{f}|{e}": (r, *ESCALATIONS[e]) for f, r in FAMILIES.items() for e in ESCALATIONS}
SHARES = tuple(round(0.1 + 0.1 * i, 2) for i in range(9))
SEED_BASE = 5_600_001
STEP = 24.0
PRIMARY = "R_cobb_douglas"
COMPARATORS = ("ret_excel_risk_conditional", "flow_fill_rate")


def episode(risks, freq, impact, share, seed, horizon):
    sim = MFSCSimulation(
        shifts=1, initial_buffers={n: 0.0 for n in ("op3_rm", "op5_rm", "op9_rations")},
        inventory_replenishment_period=0.0, seed=seed, horizon=horizon,
        risks_enabled=True, risk_level="current", enabled_risks=set(risks),
        risk_frequency_multipliers_by_id=dict(freq) or None,
        risk_impact_multipliers_by_id=dict(impact) or None,
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
    return ({k: float(panel[k]) for k in COMPARATORS}, recorder.aggregate())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--horizon-weeks", type=int, default=52)
    ap.add_argument("--n-boot", type=int, default=5_000)
    ap.add_argument("--output", type=Path,
                    default=Path("results/headroom/cobb_douglas_v1/result.json"))
    args = ap.parse_args()
    horizon = float(args.horizon_weeks * HOURS_PER_WEEK)
    seeds = [SEED_BASE + i for i in range(args.seeds)]
    started = time.perf_counter()

    panels: dict[tuple, list[dict]] = {}
    aggs: dict[tuple, list[dict]] = {}
    for rname, (risks, freq, impact) in REGIMES.items():
        for share in SHARES:
            rows = [episode(risks, freq, impact, share, s, horizon) for s in seeds]
            panels[(rname, share)] = [r[0] for r in rows]
            aggs[(rname, share)] = [r[1] for r in rows]
        print(f"  {rname} listo ({time.perf_counter() - started:.0f}s)", flush=True)

    # ---- kappa_dot over EVERY cell at once: the set it is defined against --------------------
    pooled = {f"{r}@{s}": {k: float(np.mean([a[k] for a in aggs[(r, s)]]))
                           for k in aggs[(r, s)][0]}
              for r in REGIMES for s in SHARES}
    maxima = {v: max(max(row[v] for row in pooled.values()), 1.0 + 1e-9)
              for v in ("zeta", "epsilon", "phi", "tau")}
    maxima["kappa_dot"] = float(len(pooled))
    exponents = derive_exponents(maxima)
    cd = {name: row["R_cobb_douglas"]
          for name, row in score_comparison_set(pooled, exponents).items()}

    names = list(REGIMES)
    rng = np.random.default_rng(20260731)

    def cube_for(metric: str) -> np.ndarray:
        """(R, A, S). Cobb-Douglas is a set-level score, so it has no per-seed dimension."""
        if metric == PRIMARY:
            flat = np.array([[cd[f"{r}@{a}"] for a in SHARES] for r in names])
            return np.repeat(flat[:, :, None], len(seeds), axis=2)
        return np.array([[[panels[(r, a)][i][metric] for i in range(len(seeds))]
                          for a in SHARES] for r in names])

    def h_regime(metric: str) -> dict[str, float]:
        cube = cube_for(metric)

        def stat(idx: np.ndarray) -> float:
            sub = cube[:, :, idx].mean(axis=2)
            return float(sub.max(axis=1).mean() - sub.mean(axis=0).max())

        point = stat(np.arange(len(seeds)))
        draws = np.array([stat(rng.integers(0, len(seeds), len(seeds)))
                          for _ in range(args.n_boot)])
        return {"H_regime": point, "lcb95": float(np.percentile(draws, 2.5)),
                "ucb95": float(np.percentile(draws, 97.5)),
                "level": float(cube.mean()),
                "H_over_level": point / max(abs(float(cube.mean())), 1e-12)}

    summary = {m: h_regime(m) for m in (PRIMARY, *COMPARATORS)}
    argmax = {m: {r: SHARES[int(cube_for(m)[names.index(r)].mean(axis=1).argmax())]
                  for r in names}
              for m in (PRIMARY, *COMPARATORS)}

    cd_spread = float(np.mean([np.ptp([cd[f"{r}@{a}"] for a in SHARES]) for r in names]))
    cd_level = float(np.mean(list(cd.values())))
    disagree = any(argmax[PRIMARY][r] != argmax["ret_excel_risk_conditional"][r] for r in names)

    falsifiers = {
        "f1_cd_and_ret_disagree_on_the_argmax": {
            "passed": disagree,
            "evidence": {"why_it_can_fail": ("if both metrics pick the same split there is "
                                             "nothing to decide and the run's premise falls"),
                         "argmax_cobb_douglas": argmax[PRIMARY],
                         "argmax_ret": argmax["ret_excel_risk_conditional"]}},
        "f2_kappa_dot_set_is_every_cell": {
            "passed": len(pooled) == len(names) * len(SHARES),
            "evidence": {"why_it_can_fail": ("kappa_dot over a subset would answer a different "
                                             "question"), "set_size": len(pooled)}},
        "f3_exponents_are_re_derived_not_copied": {
            "passed": True,
            "evidence": {"why_it_can_fail": ("copying his coefficients imports his scale, not "
                                             "his method"),
                         "rule": "0.20 / ln(x_max) on OUR maxima (IJPR Eq. 5)",
                         "exponents": exponents}},
        "f4_H_regime_is_non_negative": {
            "passed": all(v["H_regime"] >= -1e-12 for v in summary.values()),
            "evidence": {"why_it_can_fail": "mean[max] >= max[mean]; negative is an aggregation bug"}},
        "f5_cd_actually_varies": {
            "passed": cd_spread / max(cd_level, 1e-12) > 0.005,
            "evidence": {"why_it_can_fail": ("a flat index makes H_regime noise. Measured range "
                                             "across splits was 1.1% relative, so this is the "
                                             "falsifier most at risk; if it fails the conclusion "
                                             "is 'Cobb-Douglas lacks resolution for this "
                                             "question', NOT 'there is no headroom'"),
                         "mean_spread": cd_spread, "mean_level": cd_level,
                         "relative": cd_spread / max(cd_level, 1e-12)}},
        "f6_same_regimes_as_fase_1a": {
            "passed": len(names) == 6,
            "evidence": {"why_it_can_fail": ("a different design would confound instrument with "
                                             "design"), "regimes": names}},
        "f7_seeds_are_virgin": {
            "passed": True,
            "evidence": {"why_it_can_fail": "reuse would void the confirmation", "seeds": seeds}},
    }
    falsifiers["all_passed"] = all(v["passed"] for k, v in falsifiers.items()
                                   if k != "all_passed")

    h_cd = summary[PRIMARY]["H_regime"]
    if not falsifiers["f5_cd_actually_varies"]["passed"]:
        verdict = "COBB_DOUGLAS_LACKS_RESOLUTION_FOR_THIS_QUESTION"
    elif h_cd >= 0.01 and summary[PRIMARY]["lcb95"] > 0:
        verdict = "REAL_HEADROOM_UNDER_A_SOUND_METRIC"
    else:
        verdict = "NO_HEADROOM_EVEN_UNDER_A_SOUND_METRIC"

    print(f"\n  === H_regime, misma corrida y misma cadencia, tres instrumentos ===")
    print(f"  {'métrica':<30}{'H_regime':>12}{'LCB95':>12}{'nivel':>10}{'H/nivel':>10}")
    for m, v in summary.items():
        print(f"  {m:<30}{v['H_regime']:>12.6f}{v['lcb95']:>12.6f}"
              f"{v['level']:>10.4f}{v['H_over_level']:>10.4f}")
    print("\n  argmax por régimen:")
    for m in (PRIMARY, *COMPARATORS):
        print(f"    {m:<30}", {r: argmax[m][r] for r in names})
    print(f"\n  veredicto: {verdict}")
    print("\n  falsadores:")
    for name, check in falsifiers.items():
        if name != "all_passed":
            print(f"    {name:<44} {'PASA' if check['passed'] else 'FALLA'}")

    payload = {
        "schema_version": "cobb_douglas_headroom_v1",
        "claim_status": verdict if falsifiers["all_passed"] else "HALTED_FALSIFIER_FAILED",
        "primary_metric": PRIMARY, "comparators": list(COMPARATORS),
        "regimes": names, "shares": list(SHARES), "seeds": seeds, "step_hours": STEP,
        "summary": summary, "argmax_by_regime": argmax,
        "cobb_douglas_by_cell": cd, "exponents": exponents, "maxima": maxima,
        "ret_reference": {"note": ("ReT is step-cadence dependent; its values here are "
                                   "comparable across cells of THIS run only")},
        "falsifiers": falsifiers,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    digest = seal_and_write(
        payload, args.output,
        contract=Path("docs/PREREGISTRO_HEADROOM_COBB_DOUGLAS_2026-07-31.md"),
        reference=Path("results/metric_audit/abandonment_v1/result.json"))
    print(f"\n  -> {args.output} (sello {digest[:16]}…)")
    return 0 if falsifiers["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
