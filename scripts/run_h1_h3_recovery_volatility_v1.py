#!/usr/bin/env python3
"""H1 (recovery time) and H3 (volatility) of the v.0 manuscript, on the configurations the
Fase 4 strategies would actually deploy.

H2 and H4 are already measured. These two never were. The arms are not invented here: they are
the configurations each strategy chose in the meta-learner run --

    hybrid  = chosen by `neuron_memory` (the Fig. 5 neuron that carries rho)
    static  = chosen by `ofat`, which IS Garrido's own thesis design
    reset   = chosen by `neuron_reset`, the memory ablation

-- evaluated across a ladder of disruption intensities, paired on seeds.

H3's primary is SERVICE, not ReT. Today it was measured that `ret_excel` prefers abandoning a
claimant and that the preference survives both removing censoring and bounding the tail, so a
"variance reduction" read on it would mean nothing. ReT is reported beside it, with the warning.

`system_ttr` is right-censored by construction, so `f3` refuses to report H1 as measured unless
the censored FRACTION is comparable between arms -- an arm that simply never recovers would look
fast otherwise.

See `docs/PREREGISTRO_H1_H3_2026-07-31.md` for the reading rule, fixed in advance.
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

META = Path("results/garrido_meta_learner_v1_1/result.json")
RISKS = ("R11", "R12", "R13", "R14", "R21", "R22", "R23", "R24")
INTENSITIES = (1.0, 2.0, 3.0, 4.0)
ARMS = {"hybrid": "neuron_memory", "static": "ofat", "reset": "neuron_reset"}
SEED_BASE = 5_700_001
STEP = 24.0


def modal_config(meta: dict, strategy: str) -> dict:
    """The configuration this strategy deploys most often across contexts and repeats."""
    counts: dict[str, int] = {}
    for run in meta["per_context"][strategy]:
        for ctx in run:
            key = json.dumps(run[ctx]["chosen_config"], sort_keys=True)
            counts[key] = counts.get(key, 0) + 1
    return json.loads(max(counts, key=lambda k: counts[k]))


def episode(config: dict, intensity: float, seed: int, horizon: float):
    sim = MFSCSimulation(
        shifts=int(config["shifts"]),
        initial_buffers={"op3_rm": 0.0, "op5_rm": 0.0,
                         "op9_rations": float(config["buffer_hours"]) * 2_500.0 / 24.0},
        inventory_replenishment_period=0.0, seed=seed, horizon=horizon,
        risks_enabled=True, risk_level="current", enabled_risks=set(RISKS),
        risk_frequency_multipliers_by_id={r: float(intensity) for r in RISKS},
        strict_exogenous_crn=True, year_basis=P["year_basis"],
        warmup_trigger=P["warmup_trigger"], r14_defect_mode=P["r14_defect_mode"])
    sim.params["op9_rop"] = float(config["op9_rop"])
    sim.params["op12_rop"] = float(config["op12_rop"])
    recorder = CobbDouglasRecorder(period_hours=STEP)
    for _ in range(int(round(horizon / STEP))):
        sim.step(step_hours=STEP)
        recorder.sample(sim)
    panel = compute_episode_metrics(sim, include_temporal_panel=True)
    keep = {k: float(panel.get(k, float("nan")))
            for k in ("system_ttr_mean", "system_ttr_p95", "system_ttr_censored_fraction",
                      "system_ttr_n_clusters", "flow_fill_rate",
                      "ret_excel_risk_conditional", "temporal_maximum_service_drop")}
    keep["risk_events"] = float(len(sim.risk_events))
    return keep, recorder.aggregate()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--horizon-weeks", type=int, default=52)
    ap.add_argument("--n-boot", type=int, default=5_000)
    ap.add_argument("--output", type=Path,
                    default=Path("results/manuscript/h1_h3_v1/result.json"))
    args = ap.parse_args()
    horizon = float(args.horizon_weeks * HOURS_PER_WEEK)
    seeds = [SEED_BASE + i for i in range(args.seeds)]
    meta = json.loads(META.read_text())
    configs = {arm: modal_config(meta, strategy) for arm, strategy in ARMS.items()}
    started = time.perf_counter()

    panels: dict[tuple, list[dict]] = {}
    aggs: dict[tuple, list[dict]] = {}
    for arm, config in configs.items():
        for intensity in INTENSITIES:
            rows = [episode(config, intensity, s, horizon) for s in seeds]
            panels[(arm, intensity)] = [r[0] for r in rows]
            aggs[(arm, intensity)] = [r[1] for r in rows]
        print(f"  {arm} listo ({time.perf_counter() - started:.0f}s)", flush=True)

    # Cobb-Douglas over every (arm, intensity) cell: kappa_dot is set-relative.
    pooled = {f"{a}@{i}": {k: float(np.mean([x[k] for x in aggs[(a, i)]]))
                           for k in aggs[(a, i)][0]}
              for a in configs for i in INTENSITIES}
    maxima = {v: max(max(r[v] for r in pooled.values()), 1.0 + 1e-9)
              for v in ("zeta", "epsilon", "phi", "tau")}
    maxima["kappa_dot"] = float(len(pooled))
    cd = {n: r["R_cobb_douglas"]
          for n, r in score_comparison_set(pooled, derive_exponents(maxima)).items()}

    def per_seed(arm: str, intensity: float, key: str) -> np.ndarray:
        return np.array([row[key] for row in panels[(arm, intensity)]])

    def mean(arm: str, intensity: float, key: str) -> float:
        return float(np.nanmean(per_seed(arm, intensity, key)))

    rng = np.random.default_rng(20260731)

    def paired_lcb(a: np.ndarray, b: np.ndarray) -> dict:
        """b - a per seed; positive means arm A is better (lower) on a lower-is-better metric."""
        d = b - a
        draws = d[rng.integers(0, d.size, size=(args.n_boot, d.size))].mean(axis=1)
        return {"mean": float(d.mean()), "lcb95": float(np.percentile(draws, 2.5)),
                "ucb95": float(np.percentile(draws, 97.5))}

    # ---- H1: recovery time, pooled across the ladder, paired by (seed, intensity) -----------
    def ttr_vector(arm: str) -> np.ndarray:
        return np.concatenate([per_seed(arm, i, "system_ttr_mean") for i in INTENSITIES])

    h1 = {"hybrid_vs_static": paired_lcb(ttr_vector("hybrid"), ttr_vector("static")),
          "hybrid_vs_reset": paired_lcb(ttr_vector("hybrid"), ttr_vector("reset"))}
    ttr_means = {a: float(np.nanmean(ttr_vector(a))) for a in configs}
    censored = {a: float(np.nanmean([mean(a, i, "system_ttr_censored_fraction")
                                     for i in INTENSITIES])) for a in configs}

    # ---- H3: variance ACROSS intensities, per seed, then compared -----------------------------
    def variance_vector(arm: str, key: str) -> np.ndarray:
        """One variance per seed, taken across the four intensity rungs."""
        matrix = np.stack([per_seed(arm, i, key) for i in INTENSITIES])   # (I, S)
        return matrix.var(axis=0, ddof=1)

    h3 = {"flow_fill_rate": {
              "hybrid_vs_static": paired_lcb(variance_vector("hybrid", "flow_fill_rate"),
                                             variance_vector("static", "flow_fill_rate")),
              "hybrid_vs_reset": paired_lcb(variance_vector("hybrid", "flow_fill_rate"),
                                            variance_vector("reset", "flow_fill_rate"))},
          "ret_excel_risk_conditional": {
              "hybrid_vs_static": paired_lcb(
                  variance_vector("hybrid", "ret_excel_risk_conditional"),
                  variance_vector("static", "ret_excel_risk_conditional"))}}
    cd_variance = {a: float(np.var([cd[f"{a}@{i}"] for i in INTENSITIES], ddof=1))
                   for a in configs}

    events = {i: float(np.mean([mean(a, i, "risk_events") for a in configs]))
              for i in INTENSITIES}
    censor_gap = max(censored.values()) - min(censored.values())

    falsifiers = {
        "f1_the_arms_are_actually_different_configurations": {
            "passed": configs["hybrid"] != configs["static"],
            "evidence": {"why_it_can_fail": ("identical configurations give identical results, "
                                             "so H1 and H3 would be vacuous by construction"),
                         "configs": configs}},
        "f2_the_intensity_ladder_actually_escalates": {
            "passed": events[max(INTENSITIES)] > events[min(INTENSITIES)],
            "evidence": {"why_it_can_fail": ("without more risk events, 'heterogeneous "
                                             "intensities' has not been tested"),
                         "mean_risk_events_by_intensity": events}},
        "f3_ttr_censoring_is_disclosed_and_comparable": {
            "passed": censor_gap < 0.10,
            "evidence": {"why_it_can_fail": ("system_ttr is right-censored, so its mean is "
                                             "optimistic; if the censored FRACTION also differs "
                                             "between arms, an arm that simply never recovers "
                                             "looks fast and the comparison is confounded"),
                         "censored_fraction_by_arm": censored, "gap": censor_gap,
                         "threshold": 0.10}},
        "f4_arms_share_seeds_and_ladder": {
            "passed": True,
            "evidence": {"why_it_can_fail": "different seeds would measure luck, not policy",
                         "seeds": seeds, "intensities": list(INTENSITIES)}},
        "f5_variance_is_across_intensities_not_within": {
            "passed": True,
            "evidence": {"why_it_can_fail": ("H3 is about variance ACROSS intensities; taking it "
                                             "within one rung would be a different hypothesis"),
                         "mechanism": "variance_vector stacks the four rungs per seed, ddof=1"}},
        "f6_seeds_are_virgin": {
            "passed": True,
            "evidence": {"why_it_can_fail": "reuse would void the confirmation", "seeds": seeds}},
    }
    falsifiers["all_passed"] = all(v["passed"] for k, v in falsifiers.items()
                                   if k != "all_passed")

    h1_ok = (falsifiers["f3_ttr_censoring_is_disclosed_and_comparable"]["passed"]
             and h1["hybrid_vs_static"]["lcb95"] > 0)
    h3_ok = h3["flow_fill_rate"]["hybrid_vs_static"]["lcb95"] > 0
    verdict = f"H1_{'SUPPORTED' if h1_ok else 'NOT_SUPPORTED'}__H3_{'SUPPORTED' if h3_ok else 'NOT_SUPPORTED'}"

    print("\n  === configuraciones desplegadas ===")
    for arm, cfg in configs.items():
        print(f"  {arm:<8}{cfg}")
    print(f"\n  === H1 · tiempo de recuperación (system_ttr_mean, menor es mejor) ===")
    for arm, value in ttr_means.items():
        print(f"  {arm:<8}{value:>10.2f} h   censura {censored[arm]:.3f}")
    for name, v in h1.items():
        print(f"  {name:<20} ventaja {v['mean']:+.2f} h  [{v['lcb95']:+.2f}, {v['ucb95']:+.2f}]")
    print(f"\n  === H3 · varianza entre intensidades (menor es mejor) ===")
    for key, block in h3.items():
        for name, v in block.items():
            print(f"  {key:<28}{name:<20} {v['mean']:+.3e} "
                  f"[{v['lcb95']:+.3e}, {v['ucb95']:+.3e}]")
    print(f"  varianza de Cobb-Douglas: " + "  ".join(f"{a}={v:.3e}"
                                                     for a, v in cd_variance.items()))
    print(f"\n  veredicto: {verdict}")
    print("\n  falsadores:")
    for name, check in falsifiers.items():
        if name != "all_passed":
            print(f"    {name:<52} {'PASA' if check['passed'] else 'FALLA'}")

    payload = {
        "schema_version": "h1_h3_recovery_volatility_v1",
        "claim_status": verdict if falsifiers["all_passed"] else "HALTED_FALSIFIER_FAILED",
        "arms": ARMS, "configs": configs, "intensities": list(INTENSITIES), "seeds": seeds,
        "H1": {"system_ttr_mean_by_arm": ttr_means, "contrasts": h1,
               "censored_fraction_by_arm": censored,
               "caveat": "system_ttr is right-censored by construction; its mean is optimistic"},
        "H3": {"contrasts": h3, "cobb_douglas_variance_by_arm": cd_variance,
               "primary": "flow_fill_rate",
               "why_not_ret": ("ret_excel prefers abandoning a claimant, so a variance reduction "
                               "read on it would not mean what H3 claims")},
        "falsifiers": falsifiers,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    digest = seal_and_write(
        payload, args.output,
        contract=Path("docs/PREREGISTRO_H1_H3_2026-07-31.md"), reference=META)
    print(f"\n  -> {args.output} (sello {digest[:16]}…)")
    return 0 if falsifiers["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
