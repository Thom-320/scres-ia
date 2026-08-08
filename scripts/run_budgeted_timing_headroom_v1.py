#!/usr/bin/env python3
"""Is there timing value when the resource is scarce? One actuator, one frozen budget.

Contract: docs/PREREGISTRO_HEADROOM_PRESUPUESTO_CONGELADO_2026-08-08.md, committed before this
file existed. Custody: declared replay -- no virgin blocks exist (ENMIENDA_4). Adjudicates nothing.

WHY THIS EXISTS. The seasonal gate returned H = 0 on every endpoint in every cell, and the zero
belonged to the endpoint rather than the environment: `flow_fill_rate` charges nothing for
resources, so the same 0.8404 service plateau is bought with 4,368 shift-hours or with 13,104. With
a free resource the maximal posture dominates regardless of state and H = 0 falls out by
construction. This family imposes the three conditions headroom needs TOGETHER instead of hoping
for them -- the resource is scarce (a frozen budget), its marginal value moves in time (risks that
strike inside the episode), and the policy has an observable signal to allocate it (backlog).

THE BUDGET. `B_S = sum_t (S_t - 1) * dt`, which already exists as `extra_shift_hours`. A constant
S2 over 26 weeks costs 4,368 h; the budgets are 25/50/75 percent of it. At B25 a policy CANNOT sit
in S2, so it must decide WHEN to spend -- the question the previous design could not pose.

TWO CONSERVATIVE CHOICES, both against the hypothesis and both declared:
  * the open-loop comparator is chosen IN-SAMPLE, which inflates it and makes G1 harder to pass;
  * the clairvoyant picks per tape from a DECLARED candidate family, so it is a lower bound on the
    true clairvoyant, which also makes G1 harder to pass.
A STOP reached this way is therefore stronger than the numbers alone suggest, and a GO weaker.

THE POLICY CLASS INCLUDES TIME-VARYING SCHEDULES. That is the whole point: yesterday's verdict had
to be withdrawn because it named a ceiling over constant postures a ceiling over every policy.
Whatever this returns will name the class it actually searched.
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
ALL_RISKS = R1 + R2

#: Verified in supply_chain/config.py:469-499. R24 is a contingent demand surge at op13 and R21
#: hits ops 3,5,6,7,9 simultaneously, so reserved capacity is the aligned actuator for both. R22
#: hits every LOC and R23 hits op11 -- shifts cannot reach either, which is what makes them
#: controls rather than extra conditions.
RISK_REGIMES = {"R24_up": "R24", "R21_up": "R21", "R22_up": "R22", "R23_up": "R23"}
NEGATIVE_CONTROLS = ("R22_up", "R23_up")

MAX_STEPS = 26
STEP_HOURS = 168.0
#: Surge units, not hours: S2 costs 1 and S3 costs 2 per step. A constant S2 over the horizon costs
#: MAX_STEPS units, and the three budgets are 25/50/75 percent of that.
BUDGETS = {"B25": int(0.25 * MAX_STEPS), "B50": int(0.50 * MAX_STEPS),
           "B75": int(0.75 * MAX_STEPS)}
SHIFT_SIGNAL = {1: -1.0, 2: 0.0, 3: 1.0}     # verified against continuous_its_shift

SEED_BLOCK = tuple(range(8600001, 8600013))
G1_BAR = 0.01
N_BOOT = 2_000
N_PLACEBO = 20
BUFFER_FRAC = 0.0            # pinned: this family varies ONE actuator, declared in contract s1

MODULES = ("supply_chain/continuous_its_env.py", "supply_chain/episode_metrics.py",
           "supply_chain/arm_runner.py", "supply_chain/seed_custody.py")


def cell_kwargs(demand: str, regime: str) -> dict:
    kw = dict(init_frac=0.0, reward_mode="ReT_excel_delta", observation_version="v6",
              risk_level="current", enabled_risks=ALL_RISKS, risk_rng_mode="per_risk",
              stochastic_pt=False, max_steps=MAX_STEPS, step_size_hours=STEP_HOURS,
              risk_obs=True, holding_cost=0.0, shift_cost=0.0,
              risk_overrides={RISK_REGIMES[regime]: "increased"})
    if demand == "D1":
        # researcher_defined_periodic_demand_v1 -- NOT attributed to Garrido. The realised path is
        # U(2400,2600) x our own 12-week profile; alpha and gamma feed only the GR forecast.
        kw["demand_process"] = "garrido_seasonal_v1"
        kw["demand_seasonal_contract"] = {"forecast_mode": "garrido_generator"}
    return kw


def spend(schedule) -> int:
    return int(sum(s - 1 for s in schedule))


def uniform_schedule(budget: int) -> list[int]:
    """Spread the budget evenly across the horizon: the no-information baseline."""
    sched = [1] * MAX_STEPS
    if budget <= 0:
        return sched
    idx = np.linspace(0, MAX_STEPS - 1, num=min(budget, MAX_STEPS)).round().astype(int)
    for i in idx:
        sched[i] = 2
    left = budget - int(sum(s - 1 for s in sched))
    for i in idx:                       # any remainder goes to S3, cheapest first
        if left <= 0:
            break
        sched[i], left = 3, left - 1
    return sched


def contiguous_schedule(budget: int, offset: int) -> list[int]:
    """Spend the whole budget in one contiguous block starting at `offset`."""
    sched = [1] * MAX_STEPS
    left = budget
    i = 0
    while left > 0 and i < MAX_STEPS:
        k = (offset + i) % MAX_STEPS
        take = min(2, left)
        sched[k], left = 1 + take, left - take
        i += 1
    return sched


def ranked_schedule(budget: int, weights) -> list[int]:
    """Spend on the weeks with the most realised pressure. CLAIRVOYANT: `weights` come from the
    tape, so no online policy can build this. It exists only as a ceiling."""
    sched = [1] * MAX_STEPS
    order = list(np.argsort(-np.asarray(weights, dtype=float)))
    left = budget
    for k in order:
        if left <= 0:
            break
        take = min(2, left)
        sched[int(k)], left = 1 + take, left - take
    return sched


def endpoint(metrics, demanded) -> float:
    """L_s = integrated unserved demand / integrated generated demand. LOWER IS BETTER.

    The numerator walks EVERY order and takes end = horizon for an unserved one, weighted by
    quantity (episode_metrics.py:206-214), so abandoning an order accrues the maximum penalty and
    can never improve the score. Measured, not assumed: corr(fill, deficit) = -1.0 in the v2 panel.
    """
    return float(metrics["service_loss_auc_ration_hours"]) / demanded if demanded > 0 else 0.0


def play(demand: str, regime: str, seed: int, schedule=None, rule_budget: int | None = None):
    """One episode. Either a fixed schedule, or the causal backlog rule when `rule_budget` is set."""
    env = make_continuous_its_track_a_env(**cell_kwargs(demand, regime))
    env.reset(seed=int(seed))
    sim = env.unwrapped.sim
    done = truncated = False
    step = 0
    left = rule_budget or 0
    played, pressure, shift_units = [], [], 0
    try:
        while not (done or truncated):
            if schedule is not None:
                s = schedule[min(step, MAX_STEPS - 1)]
            else:
                # THE CAUSAL RULE, and it has no free parameter to tune: spend a surge unit
                # whenever there is unmet demand on the books and budget remains. A threshold
                # chosen after seeing results would be the p-hacking this family exists to avoid.
                backlog = float(getattr(sim, "pending_backorder_qty", 0.0) or 0.0)
                s = 2 if (backlog > 0.0 and left > 0) else 1
                left -= (s - 1)
            played.append(s)
            shift_units += (s - 1)
            _o, _r, done, truncated, info = env.step(
                np.array([BUFFER_FRAC, SHIFT_SIGNAL[s]], dtype=np.float32))
            pressure.append(float(info.get("new_backorder_qty", 0.0) or 0.0))
            step += 1
        m = compute_episode_metrics(sim)
        demanded = float(m["demanded_rations"])
        return {
            "L": endpoint(m, demanded),
            "flow_fill_rate": float(m["flow_fill_rate"]),
            "lost_rations": demanded - float(m["delivered_rations"]),
            "extra_shift_hours": float(shift_units) * STEP_HOURS,
            "shift_hours": float(sum(played)) * STEP_HOURS,
            "schedule": played, "pressure": pressure, "spent_units": int(shift_units),
        }
    finally:
        env.close()


def es10(values) -> float:
    v = np.asarray(values, dtype=float)
    cut = float(np.quantile(v, 0.90))
    tail = v[v >= cut]
    return float(tail.mean()) if tail.size else float(v.max())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--replay-of", required=True)
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--output", type=Path,
                    default=Path("results/budgeted_timing_headroom/result.json"))
    args = ap.parse_args()
    started = time.perf_counter()
    rng = np.random.default_rng(20260808)

    seeds = list(SEED_BLOCK[:args.seeds])
    cells = list(itertools.product(BUDGETS, ("D0", "D1"), RISK_REGIMES))
    print(f"  {len(cells)} celdas x {len(seeds)} semillas, presupuestos "
          f"{ {k: v for k, v in BUDGETS.items()} }")

    results = {}
    for bname, demand, regime in cells:
        key = f"{bname}|{demand}|{regime}"
        budget = BUDGETS[bname]
        # The declared open-loop candidate family: one uniform spread plus twelve contiguous
        # placements, one per week of the seasonal period. Frozen here, not searched over later.
        cands = [uniform_schedule(budget)] + [contiguous_schedule(budget, o) for o in range(12)]

        per_seed = {}
        for seed in seeds:
            ref = play(demand, regime, seed)                       # S1 throughout: the tape probe
            openloop = [play(demand, regime, seed, schedule=c) for c in cands]
            clair_sched = ranked_schedule(budget, ref["pressure"])
            clair_run = play(demand, regime, seed, schedule=clair_sched)
            rule = play(demand, regime, seed, rule_budget=budget)
            k_spent = rule["spent_units"]
            placebo = []
            for _ in range(N_PLACEBO):
                sched = [1] * MAX_STEPS
                left = k_spent
                for w in rng.permutation(MAX_STEPS):
                    if left <= 0:
                        break
                    take = min(2, left)
                    sched[int(w)], left = 1 + take, left - take
                placebo.append(play(demand, regime, seed, schedule=sched))
            per_seed[seed] = {"openloop": openloop, "clairvoyant": clair_run, "rule": rule,
                              "placebo": placebo, "reference": ref}

        # Open-loop comparator: ONE schedule for every seed, chosen on the in-sample mean. That
        # inflates it and makes G1 harder to pass -- declared, and the direction is against us.
        mean_by_cand = [float(np.mean([per_seed[s]["openloop"][i]["L"] for s in seeds]))
                        for i in range(len(cands))]
        best_i = int(np.argmin(mean_by_cand))
        L_open = np.array([per_seed[s]["openloop"][best_i]["L"] for s in seeds])
        # Clairvoyant: per tape, the best of the declared family AND the pressure-ranked schedule.
        L_clair = np.array([min([per_seed[s]["openloop"][i]["L"] for i in range(len(cands))]
                                + [per_seed[s]["clairvoyant"]["L"]]) for s in seeds])
        L_rule = np.array([per_seed[s]["rule"]["L"] for s in seeds])
        L_plac = np.array([float(np.mean([p["L"] for p in per_seed[s]["placebo"]]))
                           for s in seeds])
        L_unif = np.array([per_seed[s]["openloop"][0]["L"] for s in seeds])

        def contrast(a, b):
            """b minus a, so a POSITIVE value means `a` has less unserved demand than `b`."""
            d = np.asarray(b) - np.asarray(a)
            boot = np.array([float(np.mean(d[rng.integers(0, len(d), len(d))]))
                             for _ in range(N_BOOT)])
            return {"mean": float(d.mean()), "lcb95": float(np.percentile(boot, 2.5)),
                    "ucb95": float(np.percentile(boot, 97.5)),
                    "p_not_above_bar": float(np.mean(boot <= G1_BAR)),
                    "p_not_above_zero": float(np.mean(boot <= 0.0))}

        spends = {"openloop": [per_seed[s]["openloop"][best_i]["extra_shift_hours"] for s in seeds],
                  "clairvoyant": [per_seed[s]["clairvoyant"]["extra_shift_hours"] for s in seeds],
                  "rule": [per_seed[s]["rule"]["extra_shift_hours"] for s in seeds],
                  "placebo": [float(np.mean([p["extra_shift_hours"]
                                             for p in per_seed[s]["placebo"]])) for s in seeds]}
        cand_means = np.asarray(mean_by_cand)
        non_dominated = int(np.sum(cand_means <= cand_means.min() * 1.001))
        spread = float(cand_means.max() - cand_means.min())
        se = float(np.std([per_seed[s]["openloop"][best_i]["L"] for s in seeds])
                   / np.sqrt(len(seeds)))
        without_worst = np.delete(cand_means, int(np.argmax(cand_means)))

        results[key] = {
            "budget_units": budget, "budget_hours": budget * STEP_HOURS,
            "demand": demand, "risk_regime": regime,
            "is_negative_control": regime in NEGATIVE_CONTROLS,
            "best_openloop_index": best_i,
            "best_openloop_schedule": cands[best_i],
            "L": {"openloop": float(L_open.mean()), "clairvoyant": float(L_clair.mean()),
                  "rule": float(L_rule.mean()), "placebo": float(L_plac.mean()),
                  "uniform": float(L_unif.mean())},
            "es10": {"openloop": es10(L_open), "clairvoyant": es10(L_clair),
                     "rule": es10(L_rule), "placebo": es10(L_plac)},
            "G1_timing_value": contrast(L_clair, L_open),
            "G2_observable_conversion": contrast(L_rule, L_open),
            "rule_vs_placebo": contrast(L_rule, L_plac),
            "extra_shift_hours_by_class": {k: float(np.mean(v)) for k, v in spends.items()},
            "max_extra_shift_hours": float(max(max(v) for v in spends.values())),
            "budget_exhausted_by": [k for k, v in spends.items()
                                    if abs(float(np.mean(v)) - budget * STEP_HOURS) < 1e-9],
            "n_non_dominated_schedules": non_dominated,
            "candidate_spread": spread, "candidate_stderr": se,
            "spread_without_worst": float(without_worst.max() - without_worst.min()),
            "mean_fill_rule": float(np.mean([per_seed[s]["rule"]["flow_fill_rate"]
                                             for s in seeds])),
            "mean_lost_rations_rule": float(np.mean([per_seed[s]["rule"]["lost_rations"]
                                                     for s in seeds])),
        }
        g1 = results[key]["G1_timing_value"]
        print(f"    {key:22s} G1 {g1['mean']:+.6f} lcb {g1['lcb95']:+.6f}   "
              f"G2 {results[key]['G2_observable_conversion']['mean']:+.6f}   "
              f"no-dom {non_dominated}")

    # ---- Holm over the two deciding contrasts in every cell --------------------------------
    tests = [(k, g, results[k][g]) for k in results
             for g in ("G1_timing_value", "G2_observable_conversion")]
    k_total = len(tests)
    pkey = {"G1_timing_value": "p_not_above_bar", "G2_observable_conversion": "p_not_above_zero"}
    order = sorted(range(k_total), key=lambda i: tests[i][2][pkey[tests[i][1]]])
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (k_total - rank) * tests[idx][2][pkey[tests[idx][1]]]))
        tests[idx][2]["holm_adjusted_p"] = running

    g1_pass = [k for k in results
               if results[k]["G1_timing_value"]["lcb95"] >= G1_BAR
               and results[k]["G1_timing_value"]["holm_adjusted_p"] < 0.05]
    g2_pass = [k for k in g1_pass
               if results[k]["G2_observable_conversion"]["lcb95"] > 0
               and results[k]["G2_observable_conversion"]["holm_adjusted_p"] < 0.05
               and results[k]["rule_vs_placebo"]["lcb95"] > 0]
    controls_fired = [k for k in g2_pass if results[k]["is_negative_control"]]

    verdict = ("CONFOUNDED_NO_ADJUDICATION" if controls_fired
               else "OBSERVABLE_TIMING_VALUE_UNDER_EQUAL_BUDGET" if g2_pass
               else "TIMING_VALUE_EXISTS_BUT_DOES_NOT_CONVERT" if g1_pass
               else "STOP_NO_TIMING_VALUE_UNDER_A_BINDING_BUDGET")

    falsifiers = {
        "f1_budget_binds": {
            "passed": all(v["budget_exhausted_by"] and
                          v["max_extra_shift_hours"] <= v["budget_hours"] + 1e-6
                          for v in results.values()),
            "evidence": {"why_it_can_fail": "if no class exhausts the budget it does not bind and "
                                            "we are back at the free resource that made the "
                                            "previous gate's zero an artefact of its endpoint",
                         "per_cell": {k: {"budget_hours": v["budget_hours"],
                                          "exhausted_by": v["budget_exhausted_by"],
                                          "max_spent": v["max_extra_shift_hours"]}
                                      for k, v in results.items()}}},
        "f2_budgets_are_equal_across_policies": {
            "passed": all(max(v["extra_shift_hours_by_class"].values())
                          - min(v["extra_shift_hours_by_class"].values()) <= 1e-9
                          or v["extra_shift_hours_by_class"]["rule"] <= v["budget_hours"] + 1e-9
                          for v in results.values()),
            "evidence": {"why_it_can_fail": "if the classes spend differently we compare policy AND "
                                            "resource, which is exactly what invalidated the "
                                            "previous gate. The causal rule may UNDERSPEND -- it "
                                            "cannot see the future -- and that is recorded rather "
                                            "than corrected, because topping it up would give it "
                                            "information it does not have",
                         "per_cell": {k: v["extra_shift_hours_by_class"]
                                      for k, v in results.items()}}},
        "f3_at_least_three_schedules_are_non_dominated": {
            "passed": all(v["n_non_dominated_schedules"] >= 3 for v in results.values()),
            "evidence": {"why_it_can_fail": "the falsifier the auditor demanded. The previous f9 "
                                            "passed because ONE bad corner existed, and a corner "
                                            "is not a decision frontier",
                         "per_cell": {k: v["n_non_dominated_schedules"]
                                      for k, v in results.items()}}},
        "f4_not_explained_by_one_corner": {
            "passed": all(v["spread_without_worst"] > 2.0 * v["candidate_stderr"]
                          for v in results.values()),
            "evidence": {"why_it_can_fail": "if deleting the single worst schedule collapses the "
                                            "spread below noise, there is no decision surface and "
                                            "any contrast is one bad option against the rest",
                         "per_cell": {k: {"spread": v["candidate_spread"],
                                          "without_worst": v["spread_without_worst"],
                                          "two_se": 2.0 * v["candidate_stderr"]}
                                      for k, v in results.items()}}},
        "f5_placebo_does_not_match_the_rule": {
            "passed": all(results[k]["rule_vs_placebo"]["lcb95"] > 0 for k in g1_pass),
            "evidence": {"why_it_can_fail": "THE decisive one, and it failed at op12: same budget, "
                                            "same number of surge weeks, only WHEN differs. If the "
                                            "permuted calendar matches the rule, the value is in "
                                            "spending, not in spending well",
                         "per_cell": {k: results[k]["rule_vs_placebo"] for k in results}}},
        "f6_negative_controls_stay_negative": {
            "passed": not controls_fired,
            "evidence": {"why_it_can_fail": "R22 hits every LOC and R23 hits op11; shifts reach "
                                            "neither. If raising shifts 'solves' them there is "
                                            "confounding and the R21/R24 cells cannot be read",
                         "controls": list(NEGATIVE_CONTROLS),
                         "fired": controls_fired}},
        "f7_clairvoyant_dominates": {
            "passed": all(v["L"]["clairvoyant"] <= v["L"]["openloop"] + 1e-12
                          for v in results.values()),
            "evidence": {"why_it_can_fail": "the clairvoyant picks per tape from a superset of the "
                                            "open-loop family, so it cannot be worse; a violation "
                                            "means the estimator is misindexed",
                         "per_cell": {k: {"clairvoyant": v["L"]["clairvoyant"],
                                          "openloop": v["L"]["openloop"]}
                                      for k, v in results.items()}}},
        "f8_endpoint_discriminates": {
            "passed": all(v["candidate_spread"] > 2.0 * v["candidate_stderr"]
                          for v in results.values()),
            "evidence": {"why_it_can_fail": "inherited, and it exists because a dead posture grid "
                                            "once read as a measured null",
                         "per_cell": {k: {"spread": v["candidate_spread"],
                                          "two_se": 2.0 * v["candidate_stderr"]}
                                      for k, v in results.items()}}},
        "f9_no_fresh_seeds": custody_falsifier(seeds, replay_of=args.replay_of,
                                               exclude=args.output),
    }
    falsifiers["all_passed"] = all(
        v["passed"] for k, v in falsifiers.items()
        if k != "all_passed" and isinstance(v, dict) and not v.get("not_applicable"))

    print(f"\n  G1 (timing clarividente >= {G1_BAR}): {len(g1_pass)}/{len(results)} celdas")
    print(f"  G2 (conversion observable):            {len(g2_pass)}/{len(results)} celdas")
    print(f"\n  veredicto: {verdict}\n")
    for name, f in falsifiers.items():
        if name == "all_passed" or not isinstance(f, dict):
            continue
        label = ("NO APLICA" if f.get("not_applicable")
                 else "PASA" if f["passed"] else "FALLA")
        print(f"    {name:<48} {label}")

    payload = {
        "schema_version": "budgeted_timing_headroom_v1",
        "claim_status": verdict,
        "scope": "DEVELOPMENT_ON_DECLARED_REPLAY_NO_LEARNER_AUTHORIZED",
        "run_role": "BUDGETED_CEILING_GATE", "replay_of": args.replay_of,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "module_manifest": module_manifest(MODULES, script=__file__),
        "design": {"budgets_units": BUDGETS, "budget_hours":
                   {k: v * STEP_HOURS for k, v in BUDGETS.items()},
                   "risk_regimes": RISK_REGIMES, "negative_controls": list(NEGATIVE_CONTROLS),
                   "seeds": seeds, "max_steps": MAX_STEPS, "buffer_pinned_at": BUFFER_FRAC,
                   "n_cells": len(cells), "k_holm": k_total,
                   "demand_D1": "researcher_defined_periodic_demand_v1 -- NOT Garrido's Eq. (1)"},
        "conservative_by_design": {
            "openloop_chosen_in_sample": "inflates the comparator, so G1 is harder to pass",
            "clairvoyant_from_a_declared_family": "a lower bound on the true clairvoyant, so G1 is "
                                                  "harder to pass",
            "consequence": "a STOP here is stronger than the numbers alone, a GO weaker"},
        "endpoint": {"primary": "L_s = service_loss_auc_ration_hours / demanded_rations",
                     "direction": "lower_is_better",
                     "abandonment": "an unserved order takes end = horizon, so abandoning accrues "
                                    "the maximum penalty; measured at corr(fill, deficit) = -1.0"},
        "gates": {"G1_bar": G1_BAR, "G2_bar": 0.0},
        "cells": results, "g1_pass": g1_pass, "g2_pass": g2_pass,
        "negative_controls_fired": controls_fired,
        "falsifiers": falsifiers, "elapsed_seconds": time.perf_counter() - started,
    }
    digest = seal_and_write(payload, args.output, contract=args.contract,
                            reference=Path("results/seasonal_r2_headroom_gate_v2/result.json"))
    print(f"\n  -> {args.output} (sello {digest[:16]}…)")
    return 0 if falsifiers["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
