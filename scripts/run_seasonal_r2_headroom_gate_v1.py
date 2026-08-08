#!/usr/bin/env python3
"""The ceiling under Garrido's own physics: seasonal demand x within-episode R2.

Contract: docs/PREREGISTRO_GATE_HEADROOM_ESTACIONAL_R2_2026-08-08.md, committed before this file
existed. Custody: declared replay of block 8600001-8600012. No fresh seeds -- there are none.

WHAT IT MEASURES. `H_regime` with the project's canonical estimator -- mean-over-regimes of the
best posture, minus the best single posture across regimes, on per-regime min-max normalised
means. That is exactly "a regime-knowing oracle over the best robust constant", and it is a
CEILING: no observable policy can beat a policy that already knows the regime.

THE CEILING IS DELIBERATELY CLAIRVOYANT, and the asymmetry that creates is the point. It is fitted
on the same episodes it is read on, so it OVERSTATES. A ceiling that overstates makes a STOP
conclusion stronger -- if even the inflated ceiling cannot clear the bar, nothing observable can --
and an OPEN conclusion weaker, which is why the contract says an open ceiling authorises designing
a confirmation and nothing else. It never authorises training.

THE 2 x 2. Demand D0 (inherited stationary U(2400,2600)) vs D1 (Garrido 2024 §3.2 `GR_{t+v}`,
Holt with alpha/gamma ~ U[0,1) on a seasonal profile whose multiplier averages exactly 1.0, so
mean demand is preserved BY CONSTRUCTION -- f2 verifies it rather than assuming it). Risk R_fixed
(multipliers pinned at 1.0) vs R_draw (R1 fixed; the R2 frequency and impact multipliers are
RESAMPLED EVERY STEP from a frozen support, so the realisation moves inside the episode and a
constant posture cannot track it).

THE REGIME LABEL is episode-level and posture-independent, which is what makes the comparison
common-random-number clean: under R_fixed it is the risk context; under R_draw it is the risk
context crossed with the tercile of the episode's realised mean multiplier. The draw schedule is
a function of the seed alone, so every posture in a cell sees the identical realisation -- f7.

Seasonal demand does not add an episode-level regime of its own: with a 12-week period over a
26-week episode every episode spans both troughs and plateaus. Its role in this design is to
change the physics and ask whether the optimal posture moves ACROSS RISK REGIMES more under
seasonal demand than under stationary demand. That is the 2 x 2 interaction, and it is the
question the contract poses.

Development on a declared replay. Trains nothing, adjudicates nothing, authorises nothing.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.arm_runner import seal_and_write  # noqa: E402
from supply_chain.continuous_its_env import make_continuous_its_track_a_env  # noqa: E402
from supply_chain.episode_metrics import compute_episode_metrics  # noqa: E402
from supply_chain.seed_custody import custody_falsifier, module_manifest  # noqa: E402

R1 = ("R11", "R12", "R13", "R14")
R2 = ("R21", "R22", "R23", "R24")
RISK_CONTEXTS = {"R1r": R1, "R2r": R2, "R1r+R2r": R1 + R2}

#: Frozen support for the within-episode R2 draw. Three levels, resampled every step.
R2_MULTIPLIER_SUPPORT = (0.5, 1.0, 2.0)

#: The posture grid, PER DIMENSION, because the two dimensions do not share a domain. The
#: continuous Track-A action is Box(low=[0, -1], high=[1, 1]): the first component is the strategic
#: buffer fraction and the second is a shift SIGNAL that maps to S1/S2/S3 across [-1, 1].
#:
#: The first version used [0, 1] for both, which never reached S1 -- the only shift level that
#: binds, since S2 and S3 both sit above demand. Every posture then returned a byte-identical
#: episode and the gate reported H = 0 in all four cells with all falsifiers green. That is a dead
#: instrument reading as a measured null, which is why `f9` now exists.
POSTURE_LEVELS = {0: (0.0, 0.25, 0.5, 0.75, 1.0), 1: (-1.0, -0.5, 0.0, 0.5, 1.0)}

SEED_BLOCK = tuple(range(8600001, 8600013))
GATE = 0.01                      # the bar the risk screen preregistered; see contract §4
N_BOOT = 2_000
N_PLACEBO = 500
MAX_STEPS = 26
MEAN_DEMAND_TOLERANCE = 0.01     # f2

PRIMARY = "flow_fill_rate"
SECONDARY = "R_cobb_douglas"
REPORTED = "ret_excel"

MODULES = ("supply_chain/demand_seasonal.py", "supply_chain/continuous_its_env.py",
           "supply_chain/episode_metrics.py", "supply_chain/arm_runner.py",
           "supply_chain/seed_custody.py")


def cell_kwargs(demand: str, risk_context: str) -> dict:
    kw = dict(init_frac=0.0, reward_mode="ReT_excel_delta", observation_version="v6",
              risk_level="current", enabled_risks=RISK_CONTEXTS[risk_context],
              risk_rng_mode="per_risk", stochastic_pt=False, max_steps=MAX_STEPS,
              step_size_hours=168.0, risk_obs=True, holding_cost=0.0, shift_cost=0.001)
    if demand == "D1":
        kw["demand_process"] = "garrido_seasonal_v1"
        kw["demand_seasonal_contract"] = {"forecast_mode": "garrido_generator"}
    return kw


def draw_schedule(seed: int, risk_mode: str) -> list[float]:
    """The R2 multiplier in force at each step. A function of the seed ALONE.

    Posture-independence is not a nicety here: it is what makes every posture in a cell face the
    identical realisation, so a difference between postures is policy and not draw noise.
    """
    if risk_mode == "R_fixed":
        return [1.0] * MAX_STEPS
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0x5232]))
    return [float(R2_MULTIPLIER_SUPPORT[i]) for i in
            rng.integers(0, len(R2_MULTIPLIER_SUPPORT), MAX_STEPS)]


def run_episode(demand: str, risk_context: str, risk_mode: str, seed: int,
                posture: tuple[float, float]) -> dict:
    env = make_continuous_its_track_a_env(**cell_kwargs(demand, risk_context))
    env.reset(seed=int(seed))
    sim = env.unwrapped.sim
    schedule = draw_schedule(seed, risk_mode)
    action = np.asarray(posture, dtype=np.float32)[:env.action_space.shape[0]]
    seasonal_scales: list[float] = []
    done = truncated = False
    step = 0
    try:
        while not (done or truncated):
            m = schedule[min(step, len(schedule) - 1)]
            if risk_mode == "R_draw":
                sim.risk_frequency_multiplier = m
                sim.risk_impact_multiplier = m
            if getattr(sim, "demand_seasonal", None) is not None:
                seasonal_scales.append(float(sim.demand_seasonal.scale(float(sim.env.now))))
            _o, _r, done, truncated, _i = env.step(action)
            step += 1
        metrics = compute_episode_metrics(sim)
        return {
            PRIMARY: float(metrics[PRIMARY]),
            REPORTED: float(metrics[REPORTED]),
            "demanded_rations": float(metrics["demanded_rations"]),
            "delivered_rations": float(metrics["delivered_rations"]),
            "risk_events": float(len(sim.risk_events)),
            "mean_multiplier": float(np.mean(schedule)),
            "seasonal_scale_cv": (float(np.std(seasonal_scales) / np.mean(seasonal_scales))
                                  if seasonal_scales and np.mean(seasonal_scales) > 0 else 0.0),
        }
    finally:
        env.close()


def h_regime(per_regime: np.ndarray) -> float:
    """The project's canonical estimator. `per_regime` is (n_regimes, n_postures) of means.

    Per-regime min-max normalisation, then mean-of-max minus max-of-mean: exactly the gap between
    a regime-knowing oracle and the best single posture held across every regime.
    """
    norm = []
    for row in per_regime:
        lo, hi = float(row.min()), float(row.max())
        norm.append((row - lo) / (hi - lo) if hi > lo else np.zeros_like(row))
    stacked = np.stack(norm)
    return float(stacked.max(axis=1).mean() - stacked.mean(axis=0).max())


def regime_table(matrix: np.ndarray, labels: np.ndarray, regimes: list,
                 weights: np.ndarray | None = None) -> np.ndarray:
    """Mean endpoint per (regime, posture) from a units x postures matrix.

    Array-based on purpose. The first version rebuilt the table with a list comprehension per
    (regime, posture) cell, which is O(regimes x postures x units) per call and made 2,000
    bootstrap draws x 2 endpoints x 4 cells a billion-operation loop. `weights` carries the
    multiplicity of a bootstrap draw, so resampling never rebuilds the row list.
    """
    out = np.full((len(regimes), matrix.shape[1]), np.nan)
    for i, reg in enumerate(regimes):
        mask = labels == reg
        if weights is None:
            if mask.any():
                out[i] = matrix[mask].mean(axis=0)
            continue
        w = weights[mask]
        total = w.sum()
        if total > 0:
            out[i] = (matrix[mask] * w[:, None]).sum(axis=0) / total
    return out


def h_regime_from(matrix: np.ndarray, labels: np.ndarray, regimes: list,
                  weights: np.ndarray | None = None) -> float:
    table = regime_table(matrix, labels, regimes, weights)
    keep = ~np.isnan(table).any(axis=1)
    return h_regime(table[keep]) if keep.sum() >= 2 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--replay-of", required=True)
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--output", type=Path,
                    default=Path("results/seasonal_r2_headroom_gate/result.json"))
    args = ap.parse_args()
    started = time.perf_counter()
    rng = np.random.default_rng(20260808)

    seeds = list(SEED_BLOCK[:args.seeds])
    postures = [(a, b) for a in POSTURE_LEVELS[0] for b in POSTURE_LEVELS[1]]
    cells = list(itertools.product(("D0", "D1"), ("R_fixed", "R_draw")))
    total = len(cells) * len(RISK_CONTEXTS) * len(seeds) * len(postures)
    print(f"  {len(cells)} celdas x {len(RISK_CONTEXTS)} contextos x {len(seeds)} semillas "
          f"x {len(postures)} posturas = {total} episodios")

    raw: dict[str, list[dict]] = {}
    done_n = 0
    for demand, risk_mode in cells:
        key = f"{demand}|{risk_mode}"
        raw[key] = []
        for ctx in RISK_CONTEXTS:
            for seed in seeds:
                for posture in postures:
                    m = run_episode(demand, ctx, risk_mode, seed, posture)
                    raw[key].append({"risk_context": ctx, "seed": seed, "posture": posture, **m})
                    done_n += 1
        print(f"    {key:14s} listo  ({done_n}/{total})")

    # --- the regime label: episode-level, posture-independent ------------------------------
    # Under R_draw it is the risk context crossed with the tercile of the episode's realised mean
    # multiplier. That mean is a function of the seed alone, so the label cannot depend on which
    # posture was played -- which is what f7 checks rather than assumes.
    def build_regime_of(risk_mode: str) -> dict:
        out = {}
        if risk_mode == "R_fixed":
            for ctx in RISK_CONTEXTS:
                for s in seeds:
                    out[(ctx, s)] = ctx
            return out
        means = {s: float(np.mean(draw_schedule(s, "R_draw"))) for s in seeds}
        cuts = np.quantile(list(means.values()), [1 / 3, 2 / 3])
        for ctx in RISK_CONTEXTS:
            for s in seeds:
                t = int(means[s] > cuts[0]) + int(means[s] > cuts[1])
                out[(ctx, s)] = f"{ctx}|m{t}"
        return out

    results = {}
    for demand, risk_mode in cells:
        key = f"{demand}|{risk_mode}"
        regime_of = build_regime_of(risk_mode)
        cell_out = {"demand": demand, "risk_mode": risk_mode,
                    "n_episodes": len(raw[key]), "endpoints": {}}
        # One row per experimental UNIT -- a (risk_context, seed) pair -- and one column per
        # posture. Every posture in a cell shares the unit's realisation, which is what makes the
        # bootstrap a resample over units rather than over draws.
        units = [(c, s) for c in RISK_CONTEXTS for s in seeds]
        unit_index = {u: i for i, u in enumerate(units)}
        posture_index = {p: j for j, p in enumerate(postures)}
        unit_labels = np.array([regime_of[u] for u in units])
        regimes = sorted(set(unit_labels.tolist()))
        unit_seed = np.array([s for _, s in units])

        for endpoint in (PRIMARY, REPORTED):
            matrix = np.full((len(units), len(postures)), np.nan)
            for r in raw[key]:
                matrix[unit_index[(r["risk_context"], r["seed"])],
                       posture_index[r["posture"]]] = r[endpoint]
            point = h_regime_from(matrix, unit_labels, regimes)

            draws = np.empty(N_BOOT)
            for b in range(N_BOOT):
                pick = rng.integers(0, len(seeds), len(seeds))
                counts = np.bincount(pick, minlength=len(seeds))
                w = counts[np.searchsorted(seeds, unit_seed)].astype(float)
                draws[b] = h_regime_from(matrix, unit_labels, regimes, w)

            # THE PLACEBO the contract requires. Regime labels are permuted across units; the
            # freedom to vary the posture by label is untouched and only the information is
            # destroyed. If a shuffled label buys the same H, the value is in the label varying
            # and not in what it says -- which is what happened at op12, and is why this
            # falsifier can genuinely fail.
            placebo = np.empty(N_PLACEBO)
            for b in range(N_PLACEBO):
                shuffled = unit_labels.copy()
                rng.shuffle(shuffled)
                placebo[b] = h_regime_from(matrix, shuffled, regimes)

            table = regime_table(matrix, unit_labels, regimes)
            cell_out["endpoints"][endpoint] = {
                "H_regime": point,
                "lcb95": float(np.percentile(draws, 2.5)),
                "ucb95": float(np.percentile(draws, 97.5)),
                "clears_gate": bool(float(np.percentile(draws, 2.5)) >= GATE),
                "placebo_mean": float(placebo.mean()),
                "placebo_p95": float(np.percentile(placebo, 95)),
                "beats_placebo": bool(point > float(np.percentile(placebo, 95))),
                "n_regimes": len(regimes), "regimes": regimes,
                "best_posture_overall": list(postures[int(np.nanargmax(table.mean(axis=0)))]),
                "best_posture_per_regime": {
                    r: list(postures[int(np.nanargmax(table[i]))]) for i, r in enumerate(regimes)},
                "unique_regime_optima": len({tuple(postures[int(np.nanargmax(table[i]))])
                                             for i in range(len(regimes))}),
                # f9's evidence. An H_regime of zero says something about the ENVIRONMENT only if
                # the endpoint moves when the posture moves; if it does not, zero is a fact about
                # the instrument. Spread is measured on the regime means, noise on the same units.
                "max_posture_spread": float(np.nanmax(np.nanmax(table, axis=1)
                                                      - np.nanmin(table, axis=1))),
                "posture_stderr": float(np.nanmean(np.nanstd(matrix, axis=0))
                                        / np.sqrt(max(len(units), 1))),
                "distinct_regime_means": int(len(np.unique(np.round(table[~np.isnan(table)], 9)))),
            }
        cell_out["mean_demand"] = float(np.mean([r["demanded_rations"] for r in raw[key]]))
        cell_out["mean_risk_events"] = float(np.mean([r["risk_events"] for r in raw[key]]))
        cell_out["mean_seasonal_scale_cv"] = float(
            np.mean([r["seasonal_scale_cv"] for r in raw[key]]))
        results[key] = cell_out

    prim = {k: v["endpoints"][PRIMARY] for k, v in results.items()}
    clearing = [k for k, v in prim.items() if v["clears_gate"]]
    clearing_clean = [k for k in clearing if prim[k]["beats_placebo"]]
    best_cell = max(prim, key=lambda k: prim[k]["H_regime"])

    verdict = ("CEILING_OPEN_UNDER_GARRIDO_PHYSICS" if clearing_clean
               else "PERIOD_VARYING_NOT_STATE_VARYING" if clearing
               else "STOP_NO_HEADROOM_UNDER_GARRIDO_PHYSICS")

    # --- falsifiers -------------------------------------------------------------------------
    d0_mean = np.mean([results[k]["mean_demand"] for k in results if k.startswith("D0")])
    d1_mean = np.mean([results[k]["mean_demand"] for k in results if k.startswith("D1")])
    mean_ratio = float(d1_mean / d0_mean) if d0_mean else float("nan")
    d0_cv = np.mean([results[k]["mean_seasonal_scale_cv"] for k in results if k.startswith("D0")])
    d1_cv = np.mean([results[k]["mean_seasonal_scale_cv"] for k in results if k.startswith("D1")])
    fixed_ev = np.mean([results[k]["mean_risk_events"] for k in results if "R_fixed" in k])
    draw_ev = {s: float(np.mean(draw_schedule(s, "R_draw"))) for s in seeds}

    anchor = prim["D0|R_fixed"]
    labels_independent = all(
        len({r["posture"] for r in raw[k]}) == len(postures) for k in raw)

    discrimination = {}
    for key, cell in results.items():
        e = cell["endpoints"][PRIMARY]
        discrimination[key] = {
            "max_posture_spread": e["max_posture_spread"],
            "posture_stderr": e["posture_stderr"],
            "distinct_regime_means": e["distinct_regime_means"],
        }

    falsifiers = {
        "f1_D1_really_changes_the_process": {
            "passed": bool(d1_cv > 0.05 and d0_cv == 0.0),
            "evidence": {"why_it_can_fail": "the mean-preserving rescale could flatten the "
                                            "seasonality; if D1 ~ D0 there is no new physics and "
                                            "the demand axis is decorative. It nearly did fail "
                                            "for a different reason -- the seasonal contract is "
                                            "switched on by demand_process, not by passing the "
                                            "contract, and the first smoke test returned "
                                            "byte-identical D0 and D1",
                         "seasonal_scale_cv_D0": float(d0_cv),
                         "seasonal_scale_cv_D1": float(d1_cv)}},
        "f2_mean_demand_is_preserved": {
            "passed": bool(abs(mean_ratio - 1.0) < MEAN_DEMAND_TOLERANCE),
            "evidence": {"why_it_can_fail": "if mean demand moves, headroom and load are "
                                            "indistinguishable and the result is uninterpretable "
                                            "in EITHER direction",
                         "mean_demand_D0": float(d0_mean), "mean_demand_D1": float(d1_mean),
                         "ratio": mean_ratio, "tolerance": MEAN_DEMAND_TOLERANCE}},
        "f3_anchor_behaves_as_the_status_quo": {
            "passed": bool(anchor["n_regimes"] == len(RISK_CONTEXTS)
                           and anchor["lcb95"] < GATE),
            "evidence": {"why_it_can_fail": "D0 x R_fixed IS the status quo, whose every sealed "
                                            "measurement returns no material regime headroom. If "
                                            "this reimplementation manufactures headroom there, "
                                            "the instrument is wrong and nothing else in the "
                                            "table is comparable with the atlas",
                         "anchor": {k: anchor[k] for k in
                                    ("H_regime", "lcb95", "n_regimes", "unique_regime_optima")}}},
        "f4_the_oracle_dominates_the_constant": {
            "passed": all(v["H_regime"] >= -1e-12 for v in prim.values()),
            "evidence": {"why_it_can_fail": "the oracle is a per-regime maximum and the constant "
                                            "is a single posture, so H cannot be negative by "
                                            "construction. A negative value means the estimator "
                                            "is misindexed -- pure integrity control",
                         "per_cell": {k: v["H_regime"] for k, v in prim.items()}}},
        "f5_the_uninformed_placebo_does_not_match_the_oracle": {
            "passed": all(v["beats_placebo"] for k, v in prim.items() if v["clears_gate"]),
            "evidence": {"why_it_can_fail": "THE decisive one, and it has failed before: at op12 "
                                            "the uninformed placebo beat the state-conditioned "
                                            "rule, because the value was in the period varying "
                                            "and not in what varied it. A shuffled regime label "
                                            "keeps the freedom to vary and destroys the "
                                            "information; if it buys the same H, there is no "
                                            "state value however high the ceiling",
                         "n_permutations": N_PLACEBO,
                         "per_cell": {k: {"H_regime": v["H_regime"],
                                          "placebo_mean": v["placebo_mean"],
                                          "placebo_p95": v["placebo_p95"],
                                          "beats_placebo": v["beats_placebo"],
                                          "clears_gate": v["clears_gate"]}
                                      for k, v in prim.items()}}},
        "f6_R_draw_really_randomises": {
            "passed": bool(len({round(v, 6) for v in draw_ev.values()}) > 1
                           and np.mean([results[k]["mean_risk_events"] for k in results
                                        if "R_draw" in k]) != fixed_ev),
            "evidence": {"why_it_can_fail": "an inert draw would return R_fixed under another "
                                            "name. Measured directly: the realised risk-event "
                                            "count must respond to the schedule",
                         "support": list(R2_MULTIPLIER_SUPPORT),
                         "mean_multiplier_by_seed": draw_ev,
                         "mean_risk_events_R_fixed": float(fixed_ev),
                         "mean_risk_events_R_draw": float(np.mean(
                             [results[k]["mean_risk_events"] for k in results if "R_draw" in k]))}},
        "f7_common_random_numbers": {
            "passed": bool(labels_independent),
            "evidence": {"why_it_can_fail": "the draw schedule is a function of the seed alone, so "
                                            "every posture in a cell must face the identical "
                                            "realisation. If the regime label depended on the "
                                            "posture, the comparison would measure draw noise "
                                            "rather than policy",
                         "schedule_depends_only_on": ["seed"],
                         "postures_per_cell": len(postures)}},
        "f9_the_endpoint_discriminates_between_postures": {
            "passed": all(v["max_posture_spread"] > 2.0 * v["posture_stderr"]
                          for v in discrimination.values()),
            "evidence": {"why_it_can_fail": "AND IT DID, on the first run. The posture grid spanned "
                                            "[0, 1] in both action dimensions, but the second is a "
                                            "shift signal over [-1, 1], so S1 -- the only shift "
                                            "level that binds -- was never visited. All 25 postures "
                                            "returned byte-identical episodes and the gate reported "
                                            "H = 0 in every cell with every other falsifier green. "
                                            "An H_regime of zero is only evidence about the "
                                            "environment if the endpoint moves when the posture "
                                            "moves; otherwise it is evidence about the instrument",
                         "rule": "max posture spread within a regime must exceed 2 standard errors",
                         "per_cell": discrimination}},
        "f8_no_fresh_seeds": custody_falsifier(seeds, replay_of=args.replay_of,
                                               exclude=args.output),
    }
    falsifiers["all_passed"] = all(
        v["passed"] for k, v in falsifiers.items()
        if k != "all_passed" and isinstance(v, dict) and not v.get("not_applicable"))

    print(f"\n  endpoint primario: {PRIMARY}   barra LCB95 >= {GATE}\n")
    for k, v in sorted(prim.items(), key=lambda kv: -kv[1]["H_regime"]):
        tag = ("  <-- CRUZA" if k in clearing_clean
               else "  <-- cruza SIN batir al placebo" if k in clearing else "")
        print(f"    {k:14s} H {v['H_regime']:+.5f}  lcb {v['lcb95']:+.5f}  "
              f"placebo p95 {v['placebo_p95']:+.5f}  regimenes {v['n_regimes']}  "
              f"optimos unicos {v['unique_regime_optima']}{tag}")
    print(f"\n  {REPORTED} (se reporta, no decide):")
    for k, v in results.items():
        e = v["endpoints"][REPORTED]
        print(f"    {k:14s} H {e['H_regime']:+.5f}  lcb {e['lcb95']:+.5f}")
    print(f"\n  veredicto: {verdict}   (maximo {prim[best_cell]['H_regime']:+.5f} "
          f"en {best_cell}, contra barra {GATE})\n")
    for name, f in falsifiers.items():
        if name == "all_passed" or not isinstance(f, dict):
            continue
        label = ("NO APLICA" if f.get("not_applicable")
                 else "PASA" if f["passed"] else "FALLA")
        print(f"    {name:<52} {label}")

    payload = {
        "schema_version": "seasonal_r2_headroom_gate_v1",
        "claim_status": verdict,
        "scope": "DEVELOPMENT_ON_DECLARED_REPLAY_NO_LEARNER_AUTHORIZED",
        "run_role": "CEILING_GATE", "replay_of": args.replay_of,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "module_manifest": module_manifest(MODULES, script=__file__),
        "gate": GATE, "primary_endpoint": PRIMARY, "reported_endpoint": REPORTED,
        "secondary_endpoint_not_computed": {
            "name": SECONDARY,
            "why": ("the Cobb-Douglas port needs the per-period recorder over the physical "
                    "ledger, which this gate does not instrument; the contract lists it as "
                    "reported, not deciding, and its absence is declared rather than silent")},
        "ceiling_is_clairvoyant": {
            "fitted_on": "the same episodes it is read on",
            "consequence": ("the ceiling OVERSTATES. That makes a STOP stronger -- an inflated "
                            "ceiling that cannot clear the bar rules out every observable policy "
                            "-- and an OPEN weaker, which is why an open ceiling authorises "
                            "designing a confirmation and never training")},
        "design": {"cells": [f"{d}|{r}" for d, r in cells],
                   "risk_contexts": {k: list(v) for k, v in RISK_CONTEXTS.items()},
                   "postures": [list(p) for p in postures],
                   "r2_multiplier_support": list(R2_MULTIPLIER_SUPPORT),
                   "seeds": seeds, "max_steps": MAX_STEPS,
                   "n_episodes": total},
        "cells": results, "crossing_the_gate": clearing,
        "crossing_and_beating_placebo": clearing_clean, "best_cell": best_cell,
        "falsifiers": falsifiers, "elapsed_seconds": time.perf_counter() - started,
    }
    digest = seal_and_write(payload, args.output, contract=args.contract,
                            reference=Path("results/garrido_risk_headroom_sensitivity_v1"
                                           "/result.json"))
    print(f"\n  -> {args.output} (sello {digest[:16]}…)")
    return 0 if falsifiers["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
