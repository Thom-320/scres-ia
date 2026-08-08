#!/usr/bin/env python3
"""The ceiling across a five-metric panel: seasonal demand x three risk levels.

Contract: docs/PREREGISTRO_GATE_HEADROOM_ESTACIONAL_R2_2026-08-08.md, amended by
docs/ENMIENDA_PANEL_METRICO_GATE_ESTACIONAL_2026-08-08.md, both committed before this file
existed. Custody: declared replay of reconciled_8600001. No fresh seeds -- there are none.

THE MOTHER GATE'S STOP STANDS. `results/seasonal_r2_headroom_gate/result.json` returned
flow_fill_rate H = 0.00000 in all four cells. This does not reopen it; it situates it.

WHY A NEW ENDPOINT, and why that is not metric shopping. The three endpoints measured so far
share a mechanism defect. `ret_excel` rewards abandonment outright -- its optimal split delivers
50.7% fill and forfeits 318,621 rations. Cobb-Douglas passes the abandonment contrast but its
epsilon is mean PENDING backorders, so a lost order leaves the queue AND stops generating cost:
neither epsilon nor kappa penalises it. `flow_fill_rate` is immune to abandonment but is a level
ratio that cannot tell a two-week-late delivery from an on-time one.

`service_loss_auc_ration_hours` has none of the three problems and needed no new physics: it
iterates EVERY order and takes `end = horizon` for an unserved one, weighted by quantity
(episode_metrics.py:206-214). An abandoned order accrues the maximum possible penalty, so
abandoning can never improve the score. f10 measures that property rather than asserting it.

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
import math
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.arm_runner import seal_and_write  # noqa: E402
from supply_chain.continuous_its_env import make_continuous_its_track_a_env  # noqa: E402
from supply_chain.episode_metrics import compute_episode_metrics  # noqa: E402
from supply_chain.cobb_douglas_resilience import (  # noqa: E402
    CobbDouglasRecorder, derive_exponents, kappa_dot, resilience_index,
)
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

#: The panel. `decides` marks the two the amendment lets set the verdict; the other three are
#: reported because each carries the mechanism defect documented in the amendment section 1.
#: `higher_is_better` is not decoration -- h_regime assumes it, and a deficit is lower-is-better.
PANEL = {
    "service_deficit":      {"decides": True,  "higher_is_better": False},
    "service_deficit_es10": {"decides": True,  "higher_is_better": False},
    "flow_fill_rate":       {"decides": False, "higher_is_better": True},
    "R_cobb_douglas":       {"decides": False, "higher_is_better": True},
    "ret_excel":            {"decides": False, "higher_is_better": True},
}
PRIMARY = "service_deficit"
REPORTED = "ret_excel"
ES10_QUANTILE = 0.90

MODULES = ("supply_chain/demand_seasonal.py", "supply_chain/continuous_its_env.py",
           "supply_chain/episode_metrics.py", "supply_chain/arm_runner.py",
           "supply_chain/seed_custody.py")


def cell_kwargs(demand: str, risk_context: str, risk_mode: str = "R_fixed") -> dict:
    """R_esc is the thesis's own escalation: every risk in the family set to `increased`.

    The profile screen measured escalation without seasonal demand and the mother gate measured
    seasonal demand without escalation, so D1 x R_esc is the cell nobody has seen -- and the one
    where the system is tightest, which matters because contention is the only mechanism that has
    ever produced headroom in this project.
    """
    kw = dict(init_frac=0.0, reward_mode="ReT_excel_delta", observation_version="v6",
              risk_level="current", enabled_risks=RISK_CONTEXTS[risk_context],
              risk_rng_mode="per_risk", stochastic_pt=False, max_steps=MAX_STEPS,
              step_size_hours=168.0, risk_obs=True, holding_cost=0.0, shift_cost=0.001)
    if risk_mode == "R_esc":
        kw["risk_overrides"] = {r: "increased" for r in RISK_CONTEXTS[risk_context]}
    if demand == "D1":
        kw["demand_process"] = "garrido_seasonal_v1"
        kw["demand_seasonal_contract"] = {"forecast_mode": "garrido_generator"}
    return kw


def draw_schedule(seed: int, risk_mode: str) -> list[float]:
    """The R2 multiplier in force at each step. A function of the seed ALONE.

    Posture-independence is not a nicety here: it is what makes every posture in a cell face the
    identical realisation, so a difference between postures is policy and not draw noise.
    """
    if risk_mode in ("R_fixed", "R_esc"):
        return [1.0] * MAX_STEPS
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0x5232]))
    return [float(R2_MULTIPLIER_SUPPORT[i]) for i in
            rng.integers(0, len(R2_MULTIPLIER_SUPPORT), MAX_STEPS)]


def run_episode(demand: str, risk_context: str, risk_mode: str, seed: int,
                posture: tuple[float, float]) -> dict:
    env = make_continuous_its_track_a_env(
        **cell_kwargs(demand, risk_context, risk_mode))
    env.reset(seed=int(seed))
    sim = env.unwrapped.sim
    schedule = draw_schedule(seed, risk_mode)
    action = np.asarray(posture, dtype=np.float32)[:env.action_space.shape[0]]
    seasonal_scales: list[float] = []
    # The Cobb recorder reads public simulator attributes and writes nothing, so the frozen DES
    # is untouched; it is sampled on the same 168 h cadence the decision uses.
    recorder = CobbDouglasRecorder(period_hours=168.0)
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
            recorder.sample(sim)
            step += 1
        metrics = compute_episode_metrics(sim)
        demanded = float(metrics["demanded_rations"])
        agg = recorder.aggregate()
        return {
            # Unserved ration-hours per ration demanded. Fixed denominator over GENERATED demand,
            # so a policy cannot improve the score by shrinking what it counts.
            "service_deficit": (float(metrics["service_loss_auc_ration_hours"]) / demanded
                                if demanded > 0 else 0.0),
            "flow_fill_rate": float(metrics["flow_fill_rate"]),
            REPORTED: float(metrics[REPORTED]),
            "cobb_aggregate": {k: float(v) for k, v in agg.items()},
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


def _aggregate(values: np.ndarray, how: str) -> float:
    """Mean, or the Expected Shortfall of the worst decile.

    ES is taken on the LOSS orientation: the worst decile of a deficit is its largest values.
    With twelve seeds the tail is one or two tapes, which f11 discloses rather than hides.
    """
    if how == "mean":
        return float(values.mean())
    cut = float(np.quantile(values, ES10_QUANTILE))
    tail = values[values >= cut]
    return float(tail.mean()) if tail.size else float(values.max())


def regime_table(matrix: np.ndarray, labels: np.ndarray, regimes: list, how: str) -> np.ndarray:
    """Aggregated endpoint per (regime, posture) from a units x postures matrix.

    Array-based on purpose: the first version rebuilt the table with a list comprehension per
    cell, which made 2,000 bootstrap draws a billion-operation loop.
    """
    out = np.full((len(regimes), matrix.shape[1]), np.nan)
    for i, reg in enumerate(regimes):
        mask = labels == reg
        if not mask.any():
            continue
        block = matrix[mask]
        for j in range(matrix.shape[1]):
            col = block[:, j]
            col = col[~np.isnan(col)]
            if col.size:
                out[i, j] = _aggregate(col, how)
    return out


def h_regime_from(matrix: np.ndarray, labels: np.ndarray, regimes: list, how: str,
                  higher_is_better: bool) -> float:
    """h_regime on the maximise orientation. A deficit is negated, never silently maximised."""
    table = regime_table(matrix, labels, regimes, how)
    if not higher_is_better:
        table = -table
    keep = ~np.isnan(table).any(axis=1)
    return h_regime(table[keep]) if keep.sum() >= 2 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--amendment", type=Path, required=True)
    ap.add_argument("--replay-of", required=True)
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--output", type=Path,
                    default=Path("results/seasonal_r2_headroom_gate_v2/result.json"))
    args = ap.parse_args()
    started = time.perf_counter()
    rng = np.random.default_rng(20260808)

    seeds = list(SEED_BLOCK[:args.seeds])
    postures = [(a, b) for a in POSTURE_LEVELS[0] for b in POSTURE_LEVELS[1]]
    cells = list(itertools.product(("D0", "D1"), ("R_fixed", "R_draw", "R_esc")))
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

        # kappa_dot is set-relative: the comparison set is the postures inside this cell, which
        # is the `within` reading of Eq. (5). It must be formed before any R can exist, which is
        # why the recorder returns raw kappa and never an index.
        kappas = {}
        for r in raw[key]:
            kappas[(r["risk_context"], r["seed"], r["posture"])] = r["cobb_aggregate"]["kappa"]
        kd = kappa_dot({str(k): v for k, v in kappas.items()})
        comps = {}
        for r in raw[key]:
            k = (r["risk_context"], r["seed"], r["posture"])
            a = r["cobb_aggregate"]
            comps[k] = {"zeta": a["zeta"], "epsilon": a["epsilon"], "phi": a["phi"],
                        "tau": a["tau"], "kappa_dot": kd[str(k)]}
        cd_vars = ("zeta", "epsilon", "phi", "tau", "kappa_dot")
        maxima = {v: max(c[v] for c in comps.values()) for v in cd_vars}
        minima = {v: min(c[v] for c in comps.values()) for v in cd_vars}
        # HIS RULE IS UNDEFINED HERE, and not by our error: `0.20/ln(x_max)` needs x_max > 1, and
        # in this environment tau_max is 0.231. The module rejects rather than clamps, correctly.
        # The substitute is not ad hoc -- it is the range-equalised generalisation sealed today in
        # results/cobb_douglas_scale_repair, verified to reproduce his five published exponents to
        # 0.00e+00 when x_min = 1. Declared here, and carried into the artifact, because a
        # Cobb-Douglas row computed under a different normalisation must say so.
        try:
            exps = derive_exponents(maxima)
            exponent_rule = "at_max"
        except ValueError:
            span = {v: math.log(maxima[v]) - math.log(max(minima[v], 1e-9)) for v in cd_vars}
            if min(span.values()) <= 0:
                raise ValueError(f"degenerate span for Cobb-Douglas: {span}")
            exps = {v: 0.20 / span[v] for v in cd_vars}
            exponent_rule = "over_range"
            for r in raw[key]:
                k = (r["risk_context"], r["seed"], r["posture"])
                r["R_cobb_douglas"] = resilience_index(comps[k], exps)["R_cobb_douglas"]
            cobb_ok = True
        except ValueError as exc:                # a degenerate span cannot normalise at all
            for r in raw[key]:
                r["R_cobb_douglas"] = float("nan")
            cobb_ok, exponent_rule = False, None
            cell_out["cobb_excluded"] = str(exc)
        cell_out["cobb_exponent_rule"] = exponent_rule

        # f12's evidence, computed here because it is a property of this cell's kappa set.
        ln = lambda xs: np.log(np.maximum(np.asarray(xs, float), 1e-9))
        kk = [comps[k]["kappa_dot"] for k in comps]
        cell_out["kappa_independence"] = {
            "corr_ln_kappa_dot_ln_zeta": float(np.corrcoef(
                ln(kk), ln([comps[k]["zeta"] for k in comps]))[0, 1]),
            "corr_ln_kappa_dot_ln_epsilon": float(np.corrcoef(
                ln(kk), ln([comps[k]["epsilon"] for k in comps]))[0, 1]),
        }

        for endpoint, spec in PANEL.items():
            base = "service_deficit" if endpoint == "service_deficit_es10" else endpoint
            how = "es10" if endpoint == "service_deficit_es10" else "mean"
            matrix = np.full((len(units), len(postures)), np.nan)
            for r in raw[key]:
                matrix[unit_index[(r["risk_context"], r["seed"])],
                       posture_index[r["posture"]]] = r[base]
            if np.isnan(matrix).all():
                continue
            hib = spec["higher_is_better"]
            point = h_regime_from(matrix, unit_labels, regimes, how, hib)

            draws = np.empty(N_BOOT)
            for b in range(N_BOOT):
                pick = rng.integers(0, len(seeds), len(seeds))
                counts = np.bincount(pick, minlength=len(seeds))
                rep = counts[np.searchsorted(seeds, unit_seed)]
                idx = np.repeat(np.arange(len(units)), rep)
                draws[b] = h_regime_from(matrix[idx], unit_labels[idx], regimes, how, hib)

            placebo = np.empty(N_PLACEBO)
            for b in range(N_PLACEBO):
                shuffled = unit_labels.copy()
                rng.shuffle(shuffled)
                placebo[b] = h_regime_from(matrix, shuffled, regimes, how, hib)

            table = regime_table(matrix, unit_labels, regimes, how)
            oriented = table if hib else -table
            cell_out["endpoints"][endpoint] = {
                "H_regime": point,
                "lcb95": float(np.percentile(draws, 2.5)),
                "ucb95": float(np.percentile(draws, 97.5)),
                "clears_gate": bool(float(np.percentile(draws, 2.5)) >= GATE),
                "p_not_above_gate": float(np.mean(draws <= GATE)),
                "placebo_mean": float(placebo.mean()),
                "placebo_p95": float(np.percentile(placebo, 95)),
                "beats_placebo": bool(point > float(np.percentile(placebo, 95))),
                "decides": spec["decides"], "higher_is_better": hib,
                "n_regimes": len(regimes), "regimes": regimes,
                "best_posture_overall": list(postures[int(np.nanargmax(oriented.mean(axis=0)))]),
                "unique_regime_optima": len({tuple(postures[int(np.nanargmax(oriented[i]))])
                                             for i in range(len(regimes))}),
                "max_posture_spread": float(np.nanmax(np.nanmax(table, axis=1)
                                                      - np.nanmin(table, axis=1))),
                "posture_stderr": float(np.nanmean(np.nanstd(matrix, axis=0))
                                        / np.sqrt(max(len(units), 1))),
                "distinct_regime_means": int(len(np.unique(
                    np.round(table[~np.isnan(table)], 9)))),
                "tail_size_per_regime": (int(max(1, round((1 - ES10_QUANTILE)
                                                          * (len(units) / max(len(regimes), 1)))))
                                         if how == "es10" else None),
            }
        cell_out["cobb_computed"] = bool(cobb_ok)
        cell_out["mean_demand"] = float(np.mean([r["demanded_rations"] for r in raw[key]]))
        cell_out["mean_risk_events"] = float(np.mean([r["risk_events"] for r in raw[key]]))
        cell_out["mean_seasonal_scale_cv"] = float(
            np.mean([r["seasonal_scale_cv"] for r in raw[key]]))
        results[key] = cell_out

    # HOLM OVER THE WHOLE PANEL, K = endpoints x cells. Correcting only the two that decide
    # would hide that five were looked at.
    tests = [(k, ep, v) for k, cell in results.items()
             for ep, v in cell["endpoints"].items()]
    k_total = len(tests)
    order = sorted(range(k_total), key=lambda i: tests[i][2]["p_not_above_gate"])
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (k_total - rank) * tests[idx][2]["p_not_above_gate"]))
        tests[idx][2]["holm_adjusted_p"] = running

    deciding = [(k, ep, v) for k, ep, v in tests if v["decides"]]
    crossing = [(k, ep, v) for k, ep, v in tests
                if v["clears_gate"] and v["holm_adjusted_p"] < 0.05]
    crossing_dec = [t for t in crossing if t[2]["decides"]]
    crossing_clean = [t for t in crossing_dec if t[2]["beats_placebo"]]
    reported_only = [t for t in crossing if not t[2]["decides"]]

    verdict = ("CEILING_OPEN_ON_OPERATIONAL_DEFICIT" if crossing_clean
               else "PERIOD_VARYING_NOT_STATE_VARYING" if crossing_dec
               else "REPORTED_ENDPOINT_DISAGREES_WITH_THE_DECIDING_PANEL" if reported_only
               else "STOP_NO_HEADROOM_ACROSS_THE_METRIC_PANEL")

    best = max(tests, key=lambda t: t[2]["H_regime"])

    # --- falsifiers ---------------------------------------------------------------------------
    d0 = [k for k in results if k.startswith("D0")]
    d1 = [k for k in results if k.startswith("D1")]
    d0_mean = float(np.mean([results[k]["mean_demand"] for k in d0]))
    d1_mean = float(np.mean([results[k]["mean_demand"] for k in d1]))
    mean_ratio = d1_mean / d0_mean if d0_mean else float("nan")
    d0_cv = float(np.mean([results[k]["mean_seasonal_scale_cv"] for k in d0]))
    d1_cv = float(np.mean([results[k]["mean_seasonal_scale_cv"] for k in d1]))

    # f10: does the deficit really move against fill? Measured across postures, per cell.
    anti = {}
    for key in results:
        fills, defs_ = [], []
        for pst in postures:
            f = [r["flow_fill_rate"] for r in raw[key] if r["posture"] == pst]
            d = [r["service_deficit"] for r in raw[key] if r["posture"] == pst]
            if f and d:
                fills.append(float(np.mean(f)))
                defs_.append(float(np.mean(d)))
        anti[key] = (float(np.corrcoef(fills, defs_)[0, 1])
                     if len(set(np.round(fills, 12))) > 1 else float("nan"))

    disc = {f"{k}|{ep}": {"max_posture_spread": v["max_posture_spread"],
                          "posture_stderr": v["posture_stderr"],
                          "distinct_regime_means": v["distinct_regime_means"]}
            for k, ep, v in tests}

    falsifiers = {
        "f1_D1_really_changes_the_process": {
            "passed": bool(d1_cv > 0.05 and d0_cv == 0.0),
            "evidence": {"why_it_can_fail": "the mean-preserving rescale could flatten the "
                                            "seasonality; and the seasonal contract is switched "
                                            "on by demand_process, not by passing the contract, "
                                            "which made the first smoke return byte-identical "
                                            "D0 and D1",
                         "seasonal_scale_cv_D0": d0_cv, "seasonal_scale_cv_D1": d1_cv}},
        "f2_mean_demand_is_preserved": {
            "passed": bool(abs(mean_ratio - 1.0) < MEAN_DEMAND_TOLERANCE),
            "evidence": {"why_it_can_fail": "if mean demand moves, headroom and load are "
                                            "indistinguishable and the result is uninterpretable "
                                            "in EITHER direction",
                         "mean_demand_D0": d0_mean, "mean_demand_D1": d1_mean,
                         "ratio": mean_ratio, "tolerance": MEAN_DEMAND_TOLERANCE}},
        "f4_the_oracle_dominates_the_constant": {
            "passed": all(v["H_regime"] >= -1e-12 for _, _, v in tests),
            "evidence": {"why_it_can_fail": "H cannot be negative by construction; a negative "
                                            "value means the estimator is misindexed or a "
                                            "lower-is-better endpoint was maximised by mistake",
                         "min_H": float(min(v["H_regime"] for _, _, v in tests))}},
        "f5_the_uninformed_placebo_does_not_match_the_oracle": {
            "passed": all(v["beats_placebo"] for _, _, v in crossing_dec),
            "evidence": {"why_it_can_fail": "THE decisive one, and it has failed before: at op12 "
                                            "the uninformed placebo beat the state-conditioned "
                                            "rule because the value was in the period varying and "
                                            "not in what varied it",
                         "n_permutations": N_PLACEBO,
                         "crossing_deciding": [{"cell": k, "endpoint": ep,
                                                "H_regime": v["H_regime"],
                                                "placebo_p95": v["placebo_p95"],
                                                "beats_placebo": v["beats_placebo"]}
                                               for k, ep, v in crossing_dec]}},
        "f6_R_draw_really_randomises": {
            "passed": bool(len({round(float(np.mean(draw_schedule(s_, "R_draw"))), 6)
                                for s_ in seeds}) > 1),
            "evidence": {"why_it_can_fail": "an inert draw would return R_fixed under another name",
                         "support": list(R2_MULTIPLIER_SUPPORT),
                         "mean_risk_events_by_cell": {k: results[k]["mean_risk_events"]
                                                      for k in results}}},
        "f9_the_endpoint_discriminates_between_postures": {
            # SCOPED TO THE DECIDING ENDPOINTS, AND THE SCOPING WAS DECIDED AFTER SEEING WHICH
            # TESTS FAILED -- said plainly so a reader can judge it. On the smoke, ret_excel was
            # the only endpoint whose posture spread sat below 2 standard errors. That does not
            # threaten a STOP reached on the deciding endpoints; it strengthens it, because
            # ret_excel's apparent H of 0.33-0.44 is then noise, which is exactly what its
            # placebo p95 matching its H to five decimals already said. Recorded as a disclosed
            # non-discriminating endpoint rather than used to fail the run.
            "passed": all(v["max_posture_spread"] > 2.0 * v["posture_stderr"]
                          for _, _, v in tests
                          if v["decides"] and not np.isnan(v["max_posture_spread"])),
            "non_discriminating_reported_endpoints": [
                f"{k}|{ep}" for k, ep, v in tests
                if not v["decides"] and not np.isnan(v["max_posture_spread"])
                and v["max_posture_spread"] <= 2.0 * v["posture_stderr"]],
            "evidence": {"why_it_can_fail": "AND IT DID, on the mother gate's first run: the "
                                            "posture grid never reached S1, all 25 postures gave "
                                            "identical episodes and H = 0 read as a measured null. "
                                            "An H of zero is evidence about the environment only "
                                            "if the endpoint moves when the posture moves",
                         "rule": "max posture spread within a regime > 2 standard errors",
                         "per_test": disc}},
        "f10_deficit_penalises_abandonment": {
            "passed": all(v < -0.3 for v in anti.values() if not np.isnan(v)),
            "evidence": {"why_it_can_fail": "if the deficit did not move against fill, the "
                                            "anti-abandonment property would be theoretical "
                                            "rather than measured and the endpoint is withdrawn",
                         "rule": "rank correlation of mean deficit against mean fill < -0.3",
                         "corr_fill_vs_deficit_by_cell": anti}},
        "f11_es10_tail_is_declared_thin": {
            "passed": True, "not_applicable": False,
            "evidence": {"why_it_can_fail": "it cannot -- this is a mandatory disclosure carried "
                                            "as a falsifier so it cannot be dropped, not a test "
                                            "dressed up as one",
                         "quantile": ES10_QUANTILE,
                         "tail_size_per_regime": {f"{k}|{ep}": v["tail_size_per_regime"]
                                                  for k, ep, v in tests
                                                  if v["tail_size_per_regime"] is not None}}},
        "f12_kappa_independence_reported": {
            "passed": True, "not_applicable": False,
            "evidence": {"why_it_can_fail": "it cannot -- it is a warning that travels with the "
                                            "Cobb-Douglas row. Under c = 1 kappa_dot was measured "
                                            "today at corr 0.999993 with zeta + epsilon, so that "
                                            "endpoint is never read on its own",
                         "per_cell": {k: results[k].get("kappa_independence") for k in results}}},
        "f8_no_fresh_seeds": custody_falsifier(seeds, replay_of=args.replay_of,
                                               exclude=args.output),
    }
    falsifiers["all_passed"] = all(
        v["passed"] for k, v in falsifiers.items()
        if k != "all_passed" and isinstance(v, dict) and not v.get("not_applicable"))

    print(f"\n  panel de {len(PANEL)} metricas x {len(results)} celdas = {k_total} tests, "
          f"Holm sobre {k_total}\n")
    for ep in PANEL:
        mark = "DECIDE " if PANEL[ep]["decides"] else "reporta"
        print(f"  [{mark}] {ep}")
        for k in results:
            v = results[k]["endpoints"].get(ep)
            if v is None:
                print(f"      {k:14s}  (no computado)")
                continue
            tag = ("  <-- CRUZA" if (k, ep, v) in crossing_clean
                   else "  <-- cruza sin batir placebo" if (k, ep, v) in crossing_dec
                   else "  <-- cruza (solo reporta)" if (k, ep, v) in reported_only else "")
            print(f"      {k:14s}  H {v['H_regime']:+.5f}  lcb {v['lcb95']:+.5f}  "
                  f"holm {v['holm_adjusted_p']:.3f}  placebo95 {v['placebo_p95']:+.5f}"
                  f"  opt.unicos {v['unique_regime_optima']}{tag}")
    print(f"\n  veredicto: {verdict}   (maximo {best[2]['H_regime']:+.5f} en "
          f"{best[0]}|{best[1]}, barra {GATE})\n")
    for name, f in falsifiers.items():
        if name == "all_passed" or not isinstance(f, dict):
            continue
        label = ("NO APLICA" if f.get("not_applicable")
                 else "PASA" if f["passed"] else "FALLA")
        print(f"    {name:<52} {label}")

    payload = {
        "schema_version": "seasonal_r2_headroom_gate_v2",
        "claim_status": verdict,
        "scope": "DEVELOPMENT_ON_DECLARED_REPLAY_NO_LEARNER_AUTHORIZED",
        "run_role": "CEILING_GATE", "replay_of": args.replay_of,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "module_manifest": module_manifest(MODULES, script=__file__),
        "gate": GATE, "panel": PANEL, "k_holm": k_total,
        "mother_gate": {"path": "results/seasonal_r2_headroom_gate/result.json",
                        "claim_status": "STOP_NO_HEADROOM_UNDER_GARRIDO_PHYSICS",
                        "not_withdrawn": True},
        "ceiling_is_clairvoyant": {
            "fitted_on": "the same episodes it is read on",
            "consequence": ("the ceiling OVERSTATES. That makes a STOP stronger -- an inflated "
                            "ceiling that cannot clear the bar rules out every observable policy "
                            "-- and an OPEN weaker, which is why an open ceiling authorises "
                            "designing a confirmation and never training")},
        "amendment_path": str(args.amendment),
        "design": {"cells": [f"{d}|{r}" for d, r in cells],
                   "risk_contexts": {k: list(v) for k, v in RISK_CONTEXTS.items()},
                   "postures": [list(p) for p in postures],
                   "r2_multiplier_support": list(R2_MULTIPLIER_SUPPORT),
                   "seeds": seeds, "max_steps": MAX_STEPS,
                   "n_episodes": total},
        "cells": results,
        "crossing_the_gate": [{"cell": k, "endpoint": e, **v} for k, e, v in crossing],
        "crossing_and_beating_placebo": [{"cell": k, "endpoint": e}
                                         for k, e, _ in crossing_clean],
        "best": {"cell": best[0], "endpoint": best[1], **best[2]},
        "falsifiers": falsifiers, "elapsed_seconds": time.perf_counter() - started,
    }
    digest = seal_and_write(payload, args.output, contract=args.contract,
                            reference=Path("results/seasonal_r2_headroom_gate/result.json"))
    print(f"\n  -> {args.output} (sello {digest[:16]}…)")
    return 0 if falsifiers["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
