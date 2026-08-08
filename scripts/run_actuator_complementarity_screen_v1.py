#!/usr/bin/env python3
"""Where, if anywhere, do the two decision levers stop being perfect substitutes?

THE FOUR-PATTERN PROBE, run over a grid of source-anchored R2 severities. On the same relief
weeks it plays shifts only, buffer only, both, and neither, and asks one question:

    complementarity = min(shifts_only, buffer_only) - both

If that is zero, the second lever buys nothing once the first is on: the levers are perfect
substitutes and there is NO ALLOCATION PROBLEM for any policy -- static, adaptive or neural -- to
solve. At the shipped operating point it is exactly zero, measured: shifts alone, buffer alone and
both together all give L* = 0.239551 while neither gives 0.360272.

WHY THAT MATTERS MORE THAN ANOTHER NULL. Program O measured that contention over a NON-FUNGIBLE
shared resource carries H_PI = 0.1515 and that making the same resource fungible drives it to
EXACTLY 0. Perfectly fungible saturating levers are the fungible case, so the five headroom nulls
this project has measured all have one mechanism behind them rather than five coincidences.

WHAT THIS IS AND IS NOT. It is an ENVIRONMENT ELIGIBILITY SCREEN, run before any architecture is
compared and blind to every architecture: it never trains, never scores a network, and cannot
prefer one. Selecting an environment after seeing which architecture wins in it is circular, and
this exists precisely so that selection can be made on physics instead. Every axis moved here is
one Garrido authorised -- R2 frequency and impact -- plus a declared demand-pressure axis of ours.

`neither - min(shifts, buffer)` is reported alongside: a cell where the levers do nothing at all is
useless in the other direction, and both failure modes have to be visible.
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
from supply_chain.seed_custody import custody_falsifier, module_manifest  # noqa: E402

R1 = ("R11", "R12", "R13", "R14")
R2 = ("R21", "R22", "R23", "R24")
MAX_STEPS, STEP_HOURS, K_RELIEF = 26, 168.0, 13

PATTERNS = {"shifts_only": (0.0, 0.0), "buffer_only": (1.0, -1.0),
            "both": (1.0, 0.0), "neither": (0.0, -1.0)}
OFF = (0.0, -1.0)

#: Axes Garrido authorised by name -- R1 held still, R2 frequency and impact moved -- plus one
#: declared axis of ours on demand pressure, because a lever cannot bind where capacity never does.
FREQ = (1.0, 2.0, 4.0)
IMPACT = (1.0, 2.0, 4.0)
DEMAND = (1.0, 1.15)

SEED_BLOCK = tuple(range(8600001, 8600013))
MODULES = ("supply_chain/continuous_its_env.py", "supply_chain/episode_metrics.py",
           "supply_chain/arm_runner.py", "supply_chain/seed_custody.py")


def exposure(sim) -> float:
    horizon, start = float(sim.env.now), float(sim.warmup_time)
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
    return num / den if den > 0 else 0.0


def play(freq: float, impact: float, demand: float, seed: int, pattern) -> float:
    env = make_continuous_its_track_a_env(
        init_frac=0.0, reward_mode="ReT_excel_delta", observation_version="v6",
        risk_level="current", enabled_risks=R1 + R2, risk_rng_mode="per_risk",
        stochastic_pt=False, max_steps=MAX_STEPS, step_size_hours=STEP_HOURS,
        risk_obs=True, holding_cost=0.0, shift_cost=0.0,
        risk_frequency_multiplier=float(freq), risk_impact_multiplier=float(impact),
        demand_mean_multiplier=float(demand))
    env.reset(seed=int(seed))
    sim = env.unwrapped.sim
    relief = set(range(K_RELIEF))
    done = truncated = False
    step = 0
    try:
        while not (done or truncated):
            a = pattern if step in relief else OFF
            _o, _r, done, truncated, _i = env.step(np.array(a, dtype=np.float32))
            step += 1
        return exposure(sim)
    finally:
        env.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--replay-of", required=True)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--output", type=Path,
                    default=Path("results/actuator_complementarity_screen/result.json"))
    args = ap.parse_args()
    started = time.perf_counter()
    seeds = list(SEED_BLOCK[:args.seeds])

    grid = list(itertools.product(FREQ, IMPACT, DEMAND))
    print(f"  {len(grid)} configuraciones x {len(PATTERNS)} patrones x {len(seeds)} semillas "
          f"= {len(grid) * len(PATTERNS) * len(seeds)} episodios")

    cells = {}
    for freq, impact, demand in grid:
        key = f"f{freq:g}|i{impact:g}|d{demand:g}"
        vals = {p: float(np.mean([play(freq, impact, demand, s, a) for s in seeds]))
                for p, a in PATTERNS.items()}
        single_best = min(vals["shifts_only"], vals["buffer_only"])
        cells[key] = {
            "freq_multiplier": freq, "impact_multiplier": impact, "demand_multiplier": demand,
            "exposure": vals,
            # > 0 means the second lever still buys something once the first is on.
            "complementarity": float(single_best - vals["both"]),
            # > 0 means the levers do anything at all.
            "lever_authority": float(vals["neither"] - single_best),
            "saturated": bool(vals["neither"] - single_best < 1e-9),
            "collapsed": bool(vals["both"] > 0.95),
        }
        c = cells[key]
        print(f"    {key:22s} solo-turnos {vals['shifts_only']:.5f}  solo-buffer "
              f"{vals['buffer_only']:.5f}  ambos {vals['both']:.5f}  ninguno "
              f"{vals['neither']:.5f}   compl {c['complementarity']:+.6f}")

    usable = [k for k, v in cells.items()
              if v["complementarity"] > 1e-6 and not v["saturated"] and not v["collapsed"]]
    best = max(cells, key=lambda k: cells[k]["complementarity"])

    verdict = ("COMPLEMENTARITY_FOUND" if usable
               else "PERFECT_SUBSTITUTES_EVERYWHERE_ON_THE_SCREENED_GRID")

    falsifiers = {
        "f1_probe_separates_the_four_patterns": {
            "passed": any(v["lever_authority"] > 1e-6 for v in cells.values()),
            "evidence": {"why_it_can_fail": "if no cell shows the levers doing anything at all, "
                                            "the probe measures nothing and no conclusion about "
                                            "substitution is available in either direction",
                         "cells_with_authority": [k for k, v in cells.items()
                                                  if v["lever_authority"] > 1e-6]}},
        "f2_baseline_reproduces_the_measured_substitution": {
            "passed": abs(cells["f1|i1|d1"]["complementarity"]) < 1e-9,
            "evidence": {"why_it_can_fail": "the shipped operating point was measured at exactly "
                                            "zero complementarity; if this screen disagrees there, "
                                            "the probe is not measuring the same thing and none of "
                                            "the other cells are comparable with it",
                         "baseline": cells["f1|i1|d1"]}},
        "f3_screen_is_blind_to_architecture": {
            "passed": True, "not_applicable": False,
            "evidence": {"why_it_can_fail": "it cannot -- it is a mandatory disclosure carried as a "
                                            "falsifier so it cannot be dropped. Nothing here trains "
                                            "or scores a network, so a cell cannot be chosen "
                                            "because an architecture wins in it. Selecting an "
                                            "environment after seeing which architecture wins is "
                                            "circular, and this screen exists so the choice is made "
                                            "on physics instead",
                         "trains_nothing": True, "scores_no_architecture": True}},
        "f4_both_failure_modes_are_visible": {
            "passed": True, "not_applicable": False,
            "evidence": {"why_it_can_fail": "disclosure. Too easy (levers saturate) and too hard "
                                            "(everything collapses) are both useless, and a screen "
                                            "that only reported one would look like a search for "
                                            "difficulty",
                         "saturated_cells": [k for k, v in cells.items() if v["saturated"]],
                         "collapsed_cells": [k for k, v in cells.items() if v["collapsed"]]}},
        "f5_no_fresh_seeds": custody_falsifier(seeds, replay_of=args.replay_of,
                                               exclude=args.output),
    }
    falsifiers["all_passed"] = all(
        v["passed"] for k, v in falsifiers.items()
        if k != "all_passed" and isinstance(v, dict) and not v.get("not_applicable"))

    print(f"\n  celdas con complementariedad usable: {usable or 'ninguna'}")
    print(f"  maxima complementariedad: {best} = "
          f"{cells[best]['complementarity']:+.6f}")
    print(f"\n  veredicto: {verdict}\n")
    for name, f in falsifiers.items():
        if name == "all_passed" or not isinstance(f, dict):
            continue
        label = ("NO APLICA" if f.get("not_applicable")
                 else "PASA" if f["passed"] else "FALLA")
        print(f"    {name:<46} {label}")

    payload = {
        "schema_version": "actuator_complementarity_screen_v1",
        "claim_status": verdict,
        "scope": "DEVELOPMENT_ENVIRONMENT_ELIGIBILITY_SCREEN_NO_ARCHITECTURE_COMPARED",
        "run_role": "FOUR_PATTERN_PROBE", "replay_of": args.replay_of,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "module_manifest": module_manifest(MODULES, script=__file__),
        "question": ("complementarity = min(shifts_only, buffer_only) - both. Zero means the "
                     "second lever buys nothing once the first is on, so there is no allocation "
                     "problem for any policy to solve"),
        "why_it_matters": ("Program O measured H_PI = 0.1515 under a NON-FUNGIBLE shared resource "
                           "and exactly 0 once the same resource was made fungible. Perfectly "
                           "fungible saturating levers are the fungible case, which gives the "
                           "project's five headroom nulls one mechanism instead of five "
                           "coincidences"),
        "axes": {"risk_frequency_multiplier": list(FREQ),
                 "risk_impact_multiplier": list(IMPACT),
                 "demand_mean_multiplier": list(DEMAND),
                 "authorised_by": ("Garrido asked for R1 held still and R2 frequency and impact "
                                   "moved; the demand axis is ours and declared as such")},
        "relief_weeks": list(range(K_RELIEF)), "seeds": seeds,
        "cells": cells, "usable_cells": usable, "max_complementarity_cell": best,
        "falsifiers": falsifiers, "elapsed_seconds": time.perf_counter() - started,
    }
    digest = seal_and_write(payload, args.output, contract=args.contract,
                            reference=Path("results/exact_timing_headroom_v2/result.json"))
    print(f"\n  -> {args.output} (sello {digest[:16]}…)")
    return 0 if falsifiers["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
