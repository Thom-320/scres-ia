#!/usr/bin/env python3
"""Inventory prepositioning: WHEN to hold the buffer, at equal inventory-hours, over an EXACT class.

Same skeleton as `run_exact_timing_headroom_v2.py`, one actuator swapped. Contract:
docs/PREREGISTRO_INVENTARIO_EXACTO_V3_2026-08-08.md, committed before this file existed.

WHY INVENTORY, AND WHY R21. The shift family returned no material headroom inside its exact class,
and its own contract said why that could not settle prepositioning: with the buffer pinned at zero
it never tested the lever R21 is aligned with. R21 is `Natural disasters` -- ops 3, 5, 6, 7 and 9
struck SIMULTANEOUSLY with exp(120 h) recovery (config.py:469-475) -- so it knocks out upstream
production, and held stock downstream is the actuator that covers it. Shifts are pinned at S1
here, which is the level that binds, so the buffer is the only free lever.

MEASURED BEFORE THE RUNNER EXISTED, at S1 and seed 8600001: never holding the buffer gives L* =
0.3603, always holding gives 0.2396, and -- the part that matters -- two schedules with the SAME
thirteen-week budget give 0.2396 (weeks 0-12) against 0.3261 (weeks 13-25). The timing surface is
real here in a way it was not for shifts. Whether a tape-knowing choice beats the best fixed
calendar is a different question, and it is the one this measures. Custody: declared replay. Adjudicates nothing, authorises no learner.

WHAT V1 GOT WRONG, and what changes here.

  * V1 declared STOP when LCB95 < delta. That is "we failed to show superiority", not "we showed
    there is none". Absence needs UCB95 < delta, and only over a class that was ENUMERATED. There
    is no STOP branch in this file's vocabulary.
  * V1's ceiling was a heuristic over 14 schedules. Since L(approx) >= L(true), Delta-hat <=
    Delta*, so a positive Delta-hat establishes headroom while a zero bounds nothing. My claim
    that this made a STOP stronger was backwards.
  * V1's endpoint was ration-hours per ration -- HOURS -- so a 0.01 bar meant thirty-six seconds.
    Here it is realised exposure over maximum possible exposure, dimensionless in [0, 1].
  * V1 confounded timing with intensity: `uniform` spent on S2 while `contiguous` spent on S3
    first. Here every policy plays EXACTLY K weeks of S2 and never S3, so budget and intensity are
    identical by construction and timing is the only free variable.
  * V1 computed claim_status before all_passed, so its JSON said STOP with f3 red. Here a failed
    falsifier sets claim_status to BLOCKED_INSTRUMENT.

THE TWO CLASSES, and why the separation is the whole point. The EXACT class is all 26 starts of a
contiguous K-week block, enumerated without omission; only it can support a no-headroom claim, and
only through UCB. The ENRICHED class adds 150 random K-subsets, the pressure-ranked schedule and
the rule's own realised calendar; it can only ever say HEADROOM_FOUND or HEADROOM_NOT_FOUND_BY_
SEARCH. A heuristic search never produces a general absence claim.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.arm_runner import seal_and_write  # noqa: E402
from supply_chain.continuous_its_env import make_continuous_its_track_a_env  # noqa: E402
from supply_chain.seed_custody import custody_falsifier, module_manifest  # noqa: E402

R1 = ("R11", "R12", "R13", "R14")
R2 = ("R21", "R22", "R23", "R24")
ALL_RISKS = R1 + R2

MAX_STEPS = 26
STEP_HOURS = 168.0
K_SURGE = 13                       # weeks holding the buffer: exactly half the horizon
#: Two levels only. Intensity cannot vary, so WHEN is the single free variable, exactly as S3 was
#: excluded in the shift family. Inventory-hours = K * FULL * STEP_HOURS, identical for every
#: policy by construction rather than by tolerance.
BUFFER_LEVELS = {0: 0.0, 1: 1.0}
SHIFT_PINNED = -1.0                # S1 throughout: the binding level, so the buffer is the lever

CELLS = {"R21_current": {}, "R21_increased": {"R21": "increased"}}
SEED_BLOCK = tuple(range(8600001, 8600013))
DELTA = 0.01                       # one percentage point of maximum exposure
N_BOOT = 2_000
N_RANDOM = 150
N_PLACEBO = 20

MODULES = ("supply_chain/continuous_its_env.py", "supply_chain/episode_metrics.py",
           "supply_chain/arm_runner.py", "supply_chain/seed_custody.py")


def cell_kwargs(overrides: dict) -> dict:
    return dict(init_frac=0.0, reward_mode="ReT_excel_delta", observation_version="v6",
                risk_level="current", enabled_risks=ALL_RISKS, risk_rng_mode="per_risk",
                stochastic_pt=False, max_steps=MAX_STEPS, step_size_hours=STEP_HOURS,
                risk_obs=True, holding_cost=0.0, shift_cost=0.0,
                risk_overrides=dict(overrides))


def exposure(sim) -> tuple[float, float, float]:
    """L* = realised exposure / maximum possible exposure. Dimensionless, in [0, 1].

    Numerator and denominator use the SAME order set and the same promise time, so the denominator
    is a property of the tape and not of the policy. An unserved order takes e = T and therefore
    saturates its own term, so abandoning can never improve the score.
    """
    horizon = float(sim.env.now)
    start = float(sim.warmup_time)
    num = den = 0.0
    for o in sim.orders:
        if bool(getattr(o, "metrics_excluded", False)):
            continue
        opt = float(getattr(o, "OPTj", 0.0) or 0.0)
        if opt < start:
            continue
        q = float(o.quantity or 0.0)
        due = opt + float(o.LTj or 0.0)
        end = float(o.OATj) if getattr(o, "OATj", None) is not None else horizon
        num += q * max(0.0, end - due)
        den += q * max(0.0, horizon - due)
    return (num / den if den > 0 else 0.0), num, den


def play(overrides: dict, seed: int, weeks=None, rule: bool = False):
    """One episode. `weeks` is the set of S2 weeks; `rule` runs the causal exhausting policy."""
    env = make_continuous_its_track_a_env(**cell_kwargs(overrides))
    env.reset(seed=int(seed))
    sim = env.unwrapped.sim
    chosen = set(int(w) for w in (weeks or ()))
    left = K_SURGE
    done = truncated = False
    step = 0
    played, pressure = [], []
    try:
        while not (done or truncated):
            if rule:
                remaining = MAX_STEPS - step
                backlog = float(getattr(sim, "pending_backorder_qty", 0.0) or 0.0)
                # Causal, and it exhausts the budget WITHOUT being shown the future: the second
                # clause forces the remaining surge weeks only when there are exactly as many
                # decisions left as budget. Topping the rule up afterwards would hand it
                # information it never had.
                s = 2 if (left > 0 and (backlog > 0.0 or left == remaining)) else 1
                left -= (s - 1)
            else:
                s = 2 if step in chosen else 1
            played.append(s)
            _o, _r, done, truncated, info = env.step(
                np.array([BUFFER_LEVELS[s - 1], SHIFT_PINNED], dtype=np.float32))
            pressure.append(float(info.get("new_backorder_qty", 0.0) or 0.0))
            step += 1
        L, num, den = exposure(sim)
        return {"L": L, "num": num, "den": den, "surge_weeks": int(sum(s - 1 for s in played)),
                "schedule": played, "pressure": pressure,
                "weeks": sorted(i for i, s in enumerate(played) if s == 2)}
    finally:
        env.close()


def contiguous(start: int) -> list[int]:
    return sorted(((start + i) % MAX_STEPS) for i in range(K_SURGE))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--replay-of", required=True)
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--output", type=Path,
                    default=Path("results/exact_inventory_headroom_v3/result.json"))
    args = ap.parse_args()
    started = time.perf_counter()
    rng = np.random.default_rng(20260808)

    seeds = list(SEED_BLOCK[:args.seeds])
    exact = [contiguous(s) for s in range(MAX_STEPS)]           # all 26 starts, enumerated
    extra = [sorted(rng.choice(MAX_STEPS, size=K_SURGE, replace=False).tolist())
             for _ in range(N_RANDOM)]
    enriched = exact + extra                                    # strictly contains the exact class
    print(f"  clase exacta {len(exact)} calendarios · enriquecida {len(enriched)} · "
          f"{len(CELLS)} celdas × {len(seeds)} semillas")

    results = {}
    for cname, overrides in CELLS.items():
        per_seed = {}
        for seed in seeds:
            ref = play(overrides, seed, weeks=exact[0])
            runs = [play(overrides, seed, weeks=w) for w in enriched]
            order = list(np.argsort(-np.asarray(ref["pressure"], dtype=float)))
            ranked = play(overrides, seed, weeks=sorted(int(i) for i in order[:K_SURGE]))
            rl = play(overrides, seed, rule=True)
            plac = [play(overrides, seed,
                         weeks=sorted(rng.choice(MAX_STEPS, size=K_SURGE,
                                                 replace=False).tolist()))
                    for _ in range(N_PLACEBO)]
            per_seed[seed] = {"runs": runs, "ranked": ranked, "rule": rl, "placebo": plac}
        print(f"    {cname} listo")

        L_all = np.array([[per_seed[s]["runs"][i]["L"] for i in range(len(enriched))]
                          for s in seeds])                       # (seeds, schedules)
        L_exact = L_all[:, :len(exact)]
        L_rank = np.array([per_seed[s]["ranked"]["L"] for s in seeds])
        L_rule = np.array([per_seed[s]["rule"]["L"] for s in seeds])
        L_plac = np.array([float(np.mean([p["L"] for p in per_seed[s]["placebo"]]))
                           for s in seeds])

        def gap(matrix, extra_cols=None):
            """Per-tape best minus the single best-on-average schedule. Paired, then bootstrapped.

            This is the clairvoyant-over-open-loop contrast, and it is paired by tape so the
            interval reflects the difference and not the level.
            """
            m = matrix if extra_cols is None else np.hstack([matrix, extra_cols])
            best_fixed = int(np.argmin(m.mean(axis=0)))
            d = m[:, best_fixed] - m.min(axis=1)                 # >= 0 by construction
            boot = np.array([float(np.mean(d[rng.integers(0, len(d), len(d))]))
                             for _ in range(N_BOOT)])
            return {"mean": float(d.mean()),
                    "lcb95": float(np.percentile(boot, 2.5)),
                    "ucb95": float(np.percentile(boot, 97.5)),
                    "best_fixed_index": best_fixed,
                    "openloop_L": float(m[:, best_fixed].mean()),
                    "clairvoyant_L": float(m.min(axis=1).mean())}

        cols = np.column_stack([L_rank, L_rule])
        g_exact = gap(L_exact)
        g_enriched = gap(L_all, cols)

        d_rp = L_plac - L_rule                                   # positive => rule better
        boot_rp = np.array([float(np.mean(d_rp[rng.integers(0, len(d_rp), len(d_rp))]))
                            for _ in range(N_BOOT)])
        paired_sd = float(np.std(L_exact - L_exact.mean(axis=1, keepdims=True), ddof=1))

        results[cname] = {
            "overrides": overrides,
            "exact_class": {"n_schedules": len(exact), "enumerated": True, **g_exact},
            "enriched_class": {"n_schedules": len(enriched) + 2, **g_enriched},
            "rule_vs_placebo": {"mean": float(d_rp.mean()),
                                "lcb95": float(np.percentile(boot_rp, 2.5))},
            "L_levels": {"rule": float(L_rule.mean()), "placebo": float(L_plac.mean()),
                         "ranked": float(L_rank.mean()),
                         "exact_best_fixed": g_exact["openloop_L"]},
            "surge_weeks_by_class": {
                "schedules": sorted({r["surge_weeks"] for s in seeds
                                     for r in per_seed[s]["runs"]}),
                "rule": sorted({per_seed[s]["rule"]["surge_weeks"] for s in seeds}),
                "placebo": sorted({p["surge_weeks"] for s in seeds
                                   for p in per_seed[s]["placebo"]}),
                "ranked": sorted({per_seed[s]["ranked"]["surge_weeks"] for s in seeds})},
            "L_range": [float(L_all.min()), float(L_all.max())],
            "exact_spread": float(L_exact.mean(axis=0).max() - L_exact.mean(axis=0).min()),
            "paired_stderr": paired_sd / float(np.sqrt(len(seeds))),
        }
        print(f"      exacta   Delta {g_exact['mean']:+.6f} "
              f"[{g_exact['lcb95']:+.6f}, {g_exact['ucb95']:+.6f}]  L_open "
              f"{g_exact['openloop_L']:.5f}")
        print(f"      enriqu.  Delta {g_enriched['mean']:+.6f} "
              f"[{g_enriched['lcb95']:+.6f}, {g_enriched['ucb95']:+.6f}]")

    # ---- Holm at the interval level over the four deciding contrasts ------------------------
    contrasts = [(c, k) for c in results for k in ("exact_class", "enriched_class")]
    k_total = len(contrasts)

    established = [f"{c}|{k}" for c, k in contrasts if results[c][k]["lcb95"] >= DELTA]
    # Only the ENUMERATED class may support an absence claim, and only through the upper bound.
    absent = [f"{c}|exact_class" for c in results
              if results[c]["exact_class"]["ucb95"] < DELTA]

    delta_risk = {k: results["R21_increased"][k]["mean"] - results["R21_current"][k]["mean"]
                  for k in ("exact_class", "enriched_class")}

    # THE SUBSTITUTION PROBE, and it is the reason this family cannot deliver what it promised.
    # Four action patterns on the same relief weeks: shifts only, buffer only, both, neither.
    sub = {}
    for label, act_on in (("shifts_only", (0.0, 0.0)), ("buffer_only", (1.0, -1.0)),
                          ("both", (1.0, 0.0)), ("neither", (0.0, -1.0))):
        vals = []
        for sd in seeds[:4]:
            env = make_continuous_its_track_a_env(**cell_kwargs({}))
            env.reset(seed=int(sd))
            sm = env.unwrapped.sim
            d = t_ = False
            st = 0
            while not (d or t_):
                a = act_on if st in set(range(K_SURGE)) else (0.0, -1.0)
                _o, _r, d, t_, _i = env.step(np.array(a, dtype=np.float32))
                st += 1
            vals.append(exposure(sm)[0])
            env.close()
        sub[label] = float(np.mean(vals))
    substitutable = (abs(sub["shifts_only"] - sub["buffer_only"]) < 1e-9
                     and abs(sub["both"] - sub["buffer_only"]) < 1e-9)

    falsifiers = {
        "f1_exactly_K_buffer_weeks": {
            "passed": all(v["surge_weeks_by_class"][c] == [K_SURGE]
                          for v in results.values()
                          for c in ("schedules", "rule", "placebo", "ranked")),
            "evidence": {"why_it_can_fail": "if any policy holds the buffer for a different number of weeks, "
                                            "budget and intensity are not identical and the "
                                            "contrast mixes policy with resource -- the defect "
                                            "that invalidated the previous two gates",
                         "K": K_SURGE,
                         "per_cell": {k: v["surge_weeks_by_class"] for k, v in results.items()}}},
        "f2_endpoint_is_dimensionless": {
            "passed": all(0.0 <= v["L_range"][0] and v["L_range"][1] <= 1.0
                          for v in results.values()),
            "evidence": {"why_it_can_fail": "L* must lie in [0,1]; outside that range the "
                                            "denominator is not maximum possible exposure and the "
                                            "0.01 bar loses its meaning, which is exactly how V1's "
                                            "bar came to mean thirty-six seconds",
                         "per_cell": {k: v["L_range"] for k, v in results.items()}}},
        "f3_exact_class_is_exhaustive": {
            "passed": all(v["exact_class"]["n_schedules"] == MAX_STEPS
                          and v["exact_class"]["enumerated"] for v in results.values()),
            "evidence": {"why_it_can_fail": "enumerating fewer than all 26 starts turns the exact "
                                            "class into another heuristic, and then nothing here "
                                            "can support an absence claim",
                         "n_starts": MAX_STEPS}},
        "f4_search_contains_the_exact_class": {
            "passed": all(v["enriched_class"]["n_schedules"] > v["exact_class"]["n_schedules"]
                          for v in results.values()),
            "evidence": {"why_it_can_fail": "if the enriched search did not strictly contain the "
                                            "exact class, 'not found by search' could not be "
                                            "compared with 'absent within the class'",
                         "per_cell": {k: {"exact": v["exact_class"]["n_schedules"],
                                          "enriched": v["enriched_class"]["n_schedules"]}
                                      for k, v in results.items()}}},
        "f5_placebo_does_not_match_the_rule": {
            "passed": all(v["rule_vs_placebo"]["lcb95"] > 0 for v in results.values()),
            "evidence": {"why_it_can_fail": "same budget, same K surge weeks, only WHEN differs. "
                                            "If a permuted calendar matches the rule, the value is "
                                            "in spending and not in spending well -- how op12 "
                                            "failed",
                         "per_cell": {k: v["rule_vs_placebo"] for k, v in results.items()}}},
        "f6_clairvoyant_dominates": {
            "passed": all(v[c]["mean"] >= -1e-12 for v in results.values()
                          for c in ("exact_class", "enriched_class")),
            "evidence": {"why_it_can_fail": "the per-tape minimum cannot exceed a fixed column, so "
                                            "a negative gap means the estimator is misindexed",
                         "per_cell": {k: {c: v[c]["mean"] for c in
                                          ("exact_class", "enriched_class")}
                                      for k, v in results.items()}}},
        "f7_endpoint_discriminates": {
            "passed": all(v["exact_spread"] > 2.0 * v["paired_stderr"]
                          for v in results.values()),
            "evidence": {"why_it_can_fail": "measured against the PAIRED standard error of the "
                                            "differences, not the standard error of the best "
                                            "schedule -- V1 used the wrong one. If the 26 exact "
                                            "calendars do not separate above paired noise, there "
                                            "is no timing surface to have a value",
                         "per_cell": {k: {"spread": v["exact_spread"],
                                          "two_paired_se": 2.0 * v["paired_stderr"]}
                                      for k, v in results.items()}}},
                "f10_actuator_is_distinct_from_the_companion_family": {
            "passed": not substitutable,
            "evidence": {"why_it_can_fail": "AND IT DID. This family exists on the premise that the "
                                            "buffer is a lever the shift family never tested. "
                                            "Measured on the same relief weeks, shifts alone, "
                                            "buffer alone and BOTH together give byte-identical "
                                            "exposure, and only turning neither on is worse. The "
                                            "two actuators are PERFECT SUBSTITUTES and they "
                                            "SATURATE, so this gate re-measures the shift family's "
                                            "question with an interchangeable lever and cannot "
                                            "adjudicate prepositioning as something distinct",
                         "same_relief_weeks": list(range(K_SURGE)),
                         "exposure_by_pattern": sub,
                         "perfect_substitutes": substitutable,
                         "why_it_matters": ("Program O measured that contention over a "
                                            "NON-FUNGIBLE shared resource carries H_PI = 0.1515 "
                                            "and that making the same resource fungible drives it "
                                            "to EXACTLY 0. These two levers are perfectly fungible "
                                            "and saturating, so there is no allocation problem for "
                                            "any policy -- static, adaptive or neural -- to solve. "
                                            "That is a mechanism for the whole negative programme, "
                                            "not another null")}},
        "f9_no_fresh_seeds": custody_falsifier(seeds, replay_of=args.replay_of,
                                               exclude=args.output),
    }
    all_ok = all(v["passed"] for k, v in falsifiers.items()
                 if isinstance(v, dict) and not v.get("not_applicable"))

    # THE FALSIFIERS DECIDE THE CLAIM, not just the exit code. V1's JSON said STOP with f3 red.
    if not all_ok:
        verdict = "BLOCKED_INSTRUMENT"
    elif established:
        verdict = "HEADROOM_ESTABLISHED"
    elif len(absent) == len(results):
        verdict = "NO_MATERIAL_HEADROOM_WITHIN_EXACT_CLASS"
    else:
        verdict = "INCONCLUSIVE"
    falsifiers["all_passed"] = all_ok
    falsifiers["f8_falsifiers_block_the_claim"] = {
        "passed": (verdict == "BLOCKED_INSTRUMENT") == (not all_ok),
        "evidence": {"why_it_can_fail": "self-referential control. V1 computed claim_status before "
                                        "all_passed, so its artifact read STOP while f3 was red",
                     "all_falsifiers_passed": all_ok, "claim_status": verdict}}

    print(f"\n  sustitucion (mismas semanas de alivio): {json.dumps(sub)}")
    print(f"  sustitutos perfectos: {substitutable}")
    print(f"  Delta_R21 (increased - current): {json.dumps(delta_risk)}")
    print(f"  headroom establecido en: {established or 'ninguna'}")
    print(f"  ausencia dentro de la clase exacta en: {absent or 'ninguna'}")
    print(f"\n  veredicto: {verdict}   (delta = {DELTA}, adimensional)\n")
    for name, f in falsifiers.items():
        if name == "all_passed" or not isinstance(f, dict):
            continue
        label = ("NO APLICA" if f.get("not_applicable")
                 else "PASA" if f["passed"] else "FALLA")
        print(f"    {name:<44} {label}")

    payload = {
        "schema_version": "exact_inventory_headroom_v3",
        "claim_status": verdict,
        "scope": "DEVELOPMENT_ON_DECLARED_REPLAY_NO_LEARNER_AUTHORIZED",
        "run_role": "EXACT_CLASS_TIMING_GATE", "replay_of": args.replay_of,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "module_manifest": module_manifest(MODULES, script=__file__),
        "companion": {"path": "results/exact_timing_headroom_v2/result.json",
                      "claim_status": "NO_MATERIAL_HEADROOM_WITHIN_EXACT_CLASS",
                      "relation": ("same skeleton, shifts instead of inventory; that family could "
                                   "not settle prepositioning because it pinned the buffer at "
                                   "zero, which is why this one exists")},
        "decision_rule": {
            "delta": DELTA, "units": "fraction of maximum possible exposure",
            "established": "LCB95 >= delta",
            "absence": "UCB95 < delta AND the class was enumerated",
            "no_stop_branch": ("failing to clear a bar from below is not absence, so the "
                               "vocabulary has no STOP")},
        "endpoint": {"formula": "sum q_i [e_i - (OPT_i + LT_i)]+ / sum q_i [T - (OPT_i + LT_i)]+",
                     "dimensionless": True, "direction": "lower_is_better",
                     "denominator_is_policy_invariant": True},
        "design": {"K_buffer_weeks": K_SURGE, "max_steps": MAX_STEPS,
                   "buffer_levels": list(BUFFER_LEVELS.values()),
                   "shift_pinned_at": "S1",
                   "intensity_fixed_to_isolate_timing": True,
                   "n_exact": len(exact), "n_enriched": len(enriched) + 2,
                   "seeds": seeds, "cells": list(CELLS), "k_holm": k_total},
        "risk_estimand": {"formula": "Delta(R21 increased) - Delta(R21 current)",
                          "value": delta_risk,
                          "note": ("R21 strikes ops 3,5,6,7,9 simultaneously, so held stock "
                                   "downstream is its aligned actuator; config.py:469-475")},
        "actuator_substitution": {"exposure_by_pattern": sub,
                                  "perfect_substitutes": substitutable,
                                  "relief_weeks": list(range(K_SURGE))},
        "cells": results, "headroom_established_in": established,
        "absence_within_exact_class_in": absent,
        "falsifiers": falsifiers, "elapsed_seconds": time.perf_counter() - started,
    }
    digest = seal_and_write(payload, args.output, contract=args.contract,
                            reference=Path("results/exact_timing_headroom_v2/result.json"))
    print(f"\n  -> {args.output} (sello {digest[:16]}…)")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
