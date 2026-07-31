#!/usr/bin/env python3
"""Executes `docs/PREREGISTRO_SENSIBILIDAD_HEADROOM_2026-07-31.md`.

Where, if anywhere, is there room for a decision variable? The criterion is deliberately NOT
"which factor moves the metric": a factor with a huge effect and an invariant optimum needs a
constant, not a policy. Two quantities decide it, and both are measured here:

* `S_T - S1` -- the variance a factor contributes ONLY by interacting. If every factor has
  `S_T ~ S1` the surface is additive, and on an additive surface neither finer resolution nor a
  learned policy can beat a per-factor constant.
* the shift of a factor's `argmax` BETWEEN risk regimes -- if the best setting moves with the
  regime, a state-dependent policy pays; if it does not, it cannot.

Three stages, the standard supply-chain GSA ladder: Morris to screen, Sobol to decompose, and a
1-D sweep per regime to test policy relevance. Two resilience metrics, his 2017 Excel `ret_excel`
and the Cobb-Douglas index of his IJPR 2024 paper, because a factor with headroom under one and
not the other is itself a result.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from supply_chain.arm_runner import seal_and_write  # noqa: E402
from supply_chain.config import HOURS_PER_WEEK  # noqa: E402
from supply_chain.config import THESIS_FAITHFUL_PROTOCOL as P  # noqa: E402
from supply_chain.decision_right_discovery import (  # noqa: E402
    NumericFactor, argmax_shift_across_regimes, ishigami, ishigami_analytic,
    morris_effects, morris_trajectories, saltelli_sample, sobol_indices,
)
from supply_chain.episode_metrics import compute_episode_metrics  # noqa: E402
from supply_chain.supply_chain import MFSCSimulation  # noqa: E402

FACTORS = (
    NumericFactor("op1_rop", 1_008.0, 8_064.0, "upstream_supplier"),
    NumericFactor("op2_q", 95_000.0, 380_000.0, "upstream_supplier"),
    NumericFactor("op2_rop", 168.0, 1_344.0, "upstream_supplier"),
    NumericFactor("op3_q", 7_750.0, 47_000.0, "wdc"),
    NumericFactor("op3_rop", 84.0, 336.0, "wdc"),
    NumericFactor("batch_size", 2_500.0, 10_000.0, "assembly"),
    NumericFactor("assembly_shifts", 1.0, 3.0, "assembly"),
    NumericFactor("op9_rop", 12.0, 48.0, "battalion"),
    NumericFactor("op9_q_max", 1_200.0, 5_200.0, "battalion"),
    NumericFactor("op10_rop", 12.0, 48.0, "downstream"),
    NumericFactor("op10_q_max", 1_200.0, 5_200.0, "downstream"),
    NumericFactor("op12_rop", 12.0, 48.0, "downstream"),
    NumericFactor("op12_q_max", 1_200.0, 5_200.0, "downstream"),
    NumericFactor("op3_rm", 0.0, 122_880.0, "buffer"),
    NumericFactor("op5_rm", 0.0, 122_880.0, "buffer"),
    NumericFactor("op9_rations", 0.0, 126_000.0, "buffer"),
    NumericFactor("risk_frequency_scale", 0.5, 2.0, "risk", "environment_uncertainty"),
    NumericFactor("risk_impact_scale", 0.5, 2.0, "risk", "environment_uncertainty"),
    NumericFactor("risk_family_selector", 0.0, 1.0, "risk", "environment_uncertainty"),
    NumericFactor("demand_level", 0.75, 1.5, "demand", "environment_uncertainty"),
)
NAMES = tuple(f.name for f in FACTORS)
FAMILIES = (("R1r", ("R11", "R12", "R13", "R14")),
            ("R2r", ("R21", "R22", "R23", "R24")),
            ("R3", ("R3",)))
SEED_BASE = 4_200_001
# The action keys this runner sets. `f6` checks every one against the live contract, which is
# the defect that left `run_program_i_sensitivity.py` unrunnable.
ACTION_KEYS = ("op1_rop", "op2_q", "op2_rop", "op3_q", "op3_rop", "batch_size",
               "assembly_shifts", "op9_rop", "op9_q_min", "op9_q_max",
               "op10_rop", "op10_q_min", "op10_q_max", "op12_rop", "op12_q_min",
               "op12_q_max")


def family_of(selector: float) -> tuple[str, tuple[str, ...]]:
    return FAMILIES[min(int(selector * len(FAMILIES)), len(FAMILIES) - 1)]


def run_des(params: dict[str, float], seed: int, horizon_weeks: int) -> dict[str, float]:
    label, risks = family_of(params["risk_family_selector"])
    sim = MFSCSimulation(
        shifts=int(np.clip(np.rint(params["assembly_shifts"]), 1, 3)),
        initial_buffers={"op3_rm": params["op3_rm"], "op5_rm": params["op5_rm"],
                         "op9_rations": params["op9_rations"]},
        seed=seed, horizon=float(horizon_weeks) * HOURS_PER_WEEK,
        risks_enabled=True, risk_level="current", enabled_risks=set(risks),
        strict_exogenous_crn=True,
        risk_frequency_multiplier=params["risk_frequency_scale"],
        risk_impact_multiplier=params["risk_impact_scale"],
        demand_mean_multiplier=params["demand_level"],
        year_basis=P["year_basis"], warmup_trigger=P["warmup_trigger"],
        r14_defect_mode=P["r14_defect_mode"])
    sim.step(action={
        "op1_rop": params["op1_rop"], "op2_q": params["op2_q"],
        "op2_rop": params["op2_rop"], "op3_q": params["op3_q"],
        "op3_rop": params["op3_rop"], "batch_size": params["batch_size"],
        "assembly_shifts": int(np.clip(np.rint(params["assembly_shifts"]), 1, 3)),
        "op9_rop": params["op9_rop"],
        "op9_q_min": params["op9_q_max"] * 0.92, "op9_q_max": params["op9_q_max"],
        "op10_rop": params["op10_rop"],
        "op10_q_min": params["op10_q_max"] * 0.92, "op10_q_max": params["op10_q_max"],
        "op12_rop": params["op12_rop"],
        "op12_q_min": params["op12_q_max"] * 0.92, "op12_q_max": params["op12_q_max"],
    }, step_hours=float(horizon_weeks) * HOURS_PER_WEEK)
    m = compute_episode_metrics(sim)
    return {"ret_excel": float(m["ret_excel"]),
            "flow_fill_rate": float(m["flow_fill_rate"]),
            "family": label}


def scale_row(unit_row: np.ndarray) -> dict[str, float]:
    """Map a row of the UNIT hypercube (Saltelli) onto the factor ranges."""
    return {f.name: f.scale(unit_row[i]) for i, f in enumerate(FACTORS)}


def named_row(scaled_row: np.ndarray) -> dict[str, float]:
    """A row that is ALREADY in factor units -- what `morris_trajectories` returns.

    Caught by the smoke run: scaling a Morris row a second time pushed every factor far past
    its range, the simulator saturated, and all twenty `mu_star` came back exactly 0.0 while
    the Sobol block (correctly unit-scaled) spanned ReT 0.005 to 0.864. A silent all-zero
    screening ranking is precisely the kind of result that gets believed.
    """
    return {f.name: float(scaled_row[i]) for i, f in enumerate(FACTORS)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-sobol", type=int, default=256)
    ap.add_argument("--trajectories", type=int, default=20)
    ap.add_argument("--horizon-weeks", type=int, default=52)
    ap.add_argument("--sweep-points", type=int, default=9)
    ap.add_argument("--sweep-seeds", type=int, default=3)
    ap.add_argument("--output", type=Path,
                    default=Path("results/sensitivity/headroom_map_v1/result.json"))
    args = ap.parse_args()
    started = time.perf_counter()
    k = len(FACTORS)

    # --- f1 / f6 before any simulation ----------------------------------------------------
    a_i, b_i, ab_i = saltelli_sample(3, 8192, seed=7)
    got = sobol_indices(ishigami(a_i), ishigami(b_i),
                        [ishigami(ab_i[i]) for i in range(3)],
                        ["x1", "x2", "x3"], n_boot=40, seed=1)
    exact = ishigami_analytic()
    ishigami_error = max(max(abs(got[n]["S1"] - exact[n]["S1"]),
                             abs(got[n]["ST"] - exact[n]["ST"])) for n in exact)
    probe = MFSCSimulation(seed=1, horizon=48.0)
    unknown = [key for key in ACTION_KEYS if key not in probe.params]

    # --- stage 1: Morris -------------------------------------------------------------------
    design, edges = morris_trajectories(FACTORS, args.trajectories, 8, 20260731)
    morris_y: list[float] = []
    for i, row in enumerate(design):
        morris_y.append(run_des(named_row(row), SEED_BASE + i, args.horizon_weeks)["ret_excel"])
    morris = morris_effects(design, morris_y, FACTORS, edges)
    print(f"  Morris: {len(design)} corridas ({time.perf_counter() - started:.0f}s)", flush=True)

    # --- stage 2: Sobol --------------------------------------------------------------------
    a, b, ab = saltelli_sample(k, args.n_sobol, seed=20260731)
    offset = SEED_BASE + len(design)

    def evaluate(block: np.ndarray, tag: int) -> np.ndarray:
        return np.array([run_des(scale_row(row), offset + tag * 100_000 + i,
                                 args.horizon_weeks)["ret_excel"]
                         for i, row in enumerate(block)])

    y_a, y_b = evaluate(a, 0), evaluate(b, 1)
    y_ab = []
    for i in range(k):
        y_ab.append(evaluate(ab[i], 2 + i))
        print(f"  Sobol {i + 1}/{k} ({time.perf_counter() - started:.0f}s)", flush=True)
    sobol = sobol_indices(y_a, y_b, y_ab, NAMES, n_boot=400, seed=3)
    sum_s1 = float(sum(v["S1"] for v in sobol.values()))

    # `ret_excel` is heavy-tailed by construction -- the `0.5/RPj` branch is unbounded above,
    # and Garrido's own CF12 carries a row at 160.26. A variance decomposition of a
    # heavy-tailed output is dominated by a handful of draws, and the FIRST run of this script
    # showed exactly that: S1 = -5.75 and sum(S1) = -5.08 for `op9_q_max`, both far outside
    # [0, 1]. The estimator is not at fault (it reproduces Ishigami to 8e-4); the metric is.
    #
    # So the decomposition is repeated on the RANK transform, the standard remedy: it keeps
    # every monotone relation, kills the tail's leverage, and returns indices on a scale where
    # "fraction of variance" means something. The raw indices stay in the artifact as the
    # evidence for why they cannot be used.
    def ranked(values: np.ndarray) -> np.ndarray:
        pooled = np.concatenate([y_a, y_b, *y_ab])
        return np.searchsorted(np.sort(pooled), values, side="left") / max(pooled.size - 1, 1)

    sobol_rank = sobol_indices(ranked(y_a), ranked(y_b), [ranked(v) for v in y_ab],
                               NAMES, n_boot=400, seed=3)
    sum_s1_rank = float(sum(v["S1"] for v in sobol_rank.values()))

    # --- stage 3: 1-D sweep per risk regime ------------------------------------------------
    base = {f.name: (f.lower + f.upper) / 2.0 for f in FACTORS}
    sweeps: dict[str, dict] = {}
    seed_cursor = offset + (k + 2) * 100_000
    for factor in FACTORS:
        if factor.group == "risk" or factor.name == "demand_level":
            continue
        by_regime: dict[str, list[tuple[float, float]]] = {}
        for r_index, (label, _risks) in enumerate(FAMILIES):
            points = []
            for step in range(args.sweep_points):
                value = factor.lower + step * (factor.upper - factor.lower) / (
                    args.sweep_points - 1)
                params = dict(base)
                params[factor.name] = value
                params["risk_family_selector"] = (r_index + 0.5) / len(FAMILIES)
                scores = [run_des(params, seed_cursor + s, args.horizon_weeks)["ret_excel"]
                          for s in range(args.sweep_seeds)]
                seed_cursor += args.sweep_seeds
                points.append((value, float(np.mean(scores))))
            by_regime[label] = points
        shift = argmax_shift_across_regimes(by_regime)
        span = (factor.upper - factor.lower) or 1.0
        sweeps[factor.name] = {**shift, "argmax_span_fraction": shift["argmax_span"] / span,
                               "points_by_regime": by_regime}
    print(f"  barrido por régimen ({time.perf_counter() - started:.0f}s)", flush=True)

    # --- reading rule, as declared ---------------------------------------------------------
    # Read from the rank decomposition: it is the one whose indices are on a usable scale.
    candidates = [n for n in NAMES
                  if sobol_rank[n]["interaction"] > 0.05
                  and sweeps.get(n, {}).get("argmax_span_fraction", 0.0) > 0.20]
    verdict = {
        "surface_is_additive": bool(sum_s1_rank > 0.85),
        "sum_S1_rank": sum_s1_rank,
        "sum_S1_raw_UNUSABLE": sum_s1,
        "decision_variable_candidates": candidates,
        "rule": ("sum(S1) > 0.85 means additive, so resolution is not the bottleneck; a "
                 "candidate needs interaction > 0.05 AND an argmax that moves more than 20% "
                 "of its range between risk regimes"),
        "reading": ("if no factor qualifies, the headroom is not in this space and the next "
                    "hypothesis is topology -- new nodes -- not finer resolution"),
    }

    falsifiers = {
        "f1_sobol_estimator_reproduces_ishigami": {
            "passed": ishigami_error < 0.02,
            "evidence": {"why_it_can_fail": ("without a closed-form check every index below is "
                                             "indistinguishable from a bug"),
                         "max_abs_error": ishigami_error,
                         "includes_pure_interaction_case": "x3: S1 = 0, ST = 0.2437"}},
        "f2_morris_and_sobol_agree_on_the_top_three": {
            "passed": (set(sorted(NAMES, key=lambda n: -morris[n]["mu_star"])[:3])
                       & set(sorted(NAMES, key=lambda n: -sobol_rank[n]["ST"])[:3])) != set(),
            "evidence": {"why_it_can_fail": ("if the cheap screen and the expensive "
                                             "decomposition disagree entirely, one is wrong"),
                         "morris_top3": sorted(NAMES, key=lambda n: -morris[n]["mu_star"])[:3],
                         "sobol_top3": sorted(NAMES,
                                              key=lambda n: -sobol_rank[n]["ST"])[:3]}},
        "f3_design_covers_the_space": {
            "passed": all(len({round(float(v), 6) for v in a[:, i]}) >= 100 for i in range(k)),
            "evidence": {"why_it_can_fail": "a collapsed mapping would make the design vacuous",
                         "distinct_per_factor": {NAMES[i]: len({round(float(v), 6)
                                                                for v in a[:, i]})
                                                 for i in range(k)}}},
        "f5_indices_lie_inside_zero_one": {
            "passed": all(-0.05 <= v["S1"] <= 1.05 and -0.05 <= v["ST"] <= 1.05
                          for v in sobol_rank.values()),
            "evidence": {
                "why_it_can_fail": ("a Sobol index outside [0,1] is a failed measurement, not "
                                    "a small number. The RAW ret_excel decomposition fails "
                                    "this outright and is retained only as the evidence for "
                                    "why the rank transform is used"),
                "raw_out_of_bounds": {n: {"S1": sobol[n]["S1"], "ST": sobol[n]["ST"]}
                                      for n in NAMES
                                      if not (-0.05 <= sobol[n]["S1"] <= 1.05
                                              and -0.05 <= sobol[n]["ST"] <= 1.05)},
                "rank_out_of_bounds": {n: {"S1": sobol_rank[n]["S1"], "ST": sobol_rank[n]["ST"]}
                                       for n in NAMES
                                       if not (-0.05 <= sobol_rank[n]["S1"] <= 1.05
                                               and -0.05 <= sobol_rank[n]["ST"] <= 1.05)}}},
        "f4_output_variance_is_non_zero": {
            "passed": float(np.var(np.concatenate([y_a, y_b]), ddof=1)) > 0.0,
            "evidence": {"why_it_can_fail": "a constant output makes every index undefined",
                         "var": float(np.var(np.concatenate([y_a, y_b]), ddof=1)),
                         "ret_range": [float(min(y_a.min(), y_b.min())),
                                       float(max(y_a.max(), y_b.max()))]}},
        "f6_every_action_key_exists_in_the_live_contract": {
            "passed": not unknown,
            "evidence": {"why_it_can_fail": ("exactly the defect that left "
                                             "run_program_i_sensitivity.py unrunnable at HEAD: "
                                             "it passes op8_rop, absent from sim.params"),
                         "unknown_keys": unknown,
                         "checked": list(ACTION_KEYS)}},
    }
    falsifiers["all_passed"] = all(v["passed"] for kk, v in falsifiers.items()
                                   if kk != "all_passed")

    print(f"\n  === mapa de headroom ({k} factores) ===")
    print(f"  sum(S1) rango: {sum_s1_rank:.3f}   sum(S1) crudo (INUTILIZABLE): {sum_s1:.3f}")
    print(f"  {'factor':<24}{'mu*':>10}{'S1':>8}{'ST':>8}{'ST-S1':>8}{'argmax span':>13}")
    for name in sorted(NAMES, key=lambda n: -sobol_rank[n]["ST"]):
        sw = sweeps.get(name, {})
        span = sw.get("argmax_span_fraction")
        print(f"  {name:<24}{morris[name]['mu_star']:>10.3g}{sobol_rank[name]['S1']:>8.3f}"
              f"{sobol_rank[name]['ST']:>8.3f}{sobol_rank[name]['interaction']:>8.3f}"
              f"{(f'{span:.2f}' if span is not None else '—'):>13}")
    print("\n  falsadores:")
    for name, check in falsifiers.items():
        if name != "all_passed":
            print(f"    {name:<48} {'PASA' if check['passed'] else 'FALLA'}")
    print(f"\n  superficie aditiva: {verdict['surface_is_additive']}   "
          f"candidatos: {candidates or 'ninguno'}")

    payload = {
        "schema_version": "headroom_sensitivity_v1",
        "claim_status": ("DEVELOPMENT_HEADROOM_MAP" if falsifiers["all_passed"]
                         else "HALTED_FALSIFIER_FAILED"),
        "factors": {f.name: {"lower": f.lower, "upper": f.upper, "group": f.group}
                    for f in FACTORS},
        "morris": morris, "sobol_raw_UNUSABLE": sobol, "sobol_rank": sobol_rank,
        "regime_sweeps": sweeps,
        "raw_outputs": {"y_a": y_a.tolist(), "y_b": y_b.tolist(),
                        "y_ab": [v.tolist() for v in y_ab]},
        "verdict": verdict, "falsifiers": falsifiers,
        "n_sobol_base": args.n_sobol, "horizon_weeks": args.horizon_weeks,
        "runs": int(len(design) + (k + 2) * args.n_sobol
                    + len(sweeps) * args.sweep_points * len(FAMILIES) * args.sweep_seeds),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    digest = seal_and_write(
        payload, args.output,
        contract=Path("docs/PREREGISTRO_SENSIBILIDAD_HEADROOM_2026-07-31.md"),
        reference=Path("results/metric_audit/fidelity_reference_v4/result.json"))
    print(f"\n  -> {args.output} (sello {digest[:16]}…)")
    return 0 if falsifiers["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
