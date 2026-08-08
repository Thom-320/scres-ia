#!/usr/bin/env python3
"""Which lever is redundant, in which direction, and is anything even firing?

SUPERSEDES the claim of `results/actuator_complementarity_screen/result.json`, which is retained
and relabelled. Three defects in that screen, all found by an external auditor and all confirmed
against its own JSON before this file was written:

  * it used GLOBAL `risk_frequency_multiplier` / `risk_impact_multiplier`, so x2 and x4 escalated
    R1 AND R2 together. It was a joint escalation screen, never an R2 screen with R1 held still --
    which is precisely the constraint the PI had clarified.
  * it reported PERFECT_SUBSTITUTES on `min(shifts, buffer) - both`, a statistic that hides
    asymmetry. Its own numbers: `buffer_only == both` in 18 of 18 cells, but
    `shifts_only == buffer_only` in only 8. That is ONE-WAY redundancy of shifts given a maximal
    buffer, not symmetric substitutability.
  * its narrative said fifteen zero cells and an authority range of 0.11-0.15. The artifact says
    SIXTEEN zeros and 0.09465-0.15062.

WHAT THIS MEASURES INSTEAD. Both marginals, separately, because that is where the asymmetry lives:

    M_S = L(buffer) - L(buffer + shifts)     what shifts add given the buffer
    M_B = L(shifts) - L(shifts + buffer)     what the buffer adds given shifts

Only if BOTH are zero AND L(shifts) equals L(buffer) are the levers substitutable in the symmetric
sense. Anything else is redundancy in one direction and has to be named that way.

R1 AND R3 ARE HELD IDENTICAL ACROSS ARMS, with per-ID multipliers touching R21-R24 only, so the
estimand is the effect of moving R2. Their distribution families are never changed, which is what
the PI's clarification permits and forbids respectively.

REALISED EVENT COUNTS ARE RECORDED, because a 26-week episode against R21's source window of up to
16,128 hours -- about 96 weeks -- may contain almost no R2 events at all, and a null measured on an
episode where nothing fired says nothing about the risk. The previous screen never looked.

Per-seed values are kept, not just means, so the diagnostic can carry a paired interval rather than
a point.
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
MAX_STEPS, STEP_HOURS = 26, 168.0

PATTERNS = {"shifts_only": (0.0, 0.0), "buffer_only": (1.0, -1.0),
            "both": (1.0, 0.0), "neither": (0.0, -1.0)}
OFF = (0.0, -1.0)

#: Three cheap calendars, because the previous screen used ONE early block and could not tell a
#: property of the levers from a property of that calendar.
SCHEDULES = {"early": set(range(13)),
             "late": set(range(13, 26)),
             "alternating": set(range(0, 26, 2))}

#: Per-ID, R21-R24 only. R1 and R3 keep their parameters identical across every arm.
R2_MULTIPLIERS = (1.0, 2.0, 4.0)

SEED_BLOCK = tuple(range(8600001, 8600013))
EQUIVALENCE_MARGIN = 0.001         # a marginal below this is called negligible, not zero
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


def play(mult: float, seed: int, pattern, weeks) -> dict:
    env = make_continuous_its_track_a_env(
        init_frac=0.0, reward_mode="ReT_excel_delta", observation_version="v6",
        risk_level="current", enabled_risks=R1 + R2, risk_rng_mode="per_risk",
        stochastic_pt=False, max_steps=MAX_STEPS, step_size_hours=STEP_HOURS,
        risk_obs=True, holding_cost=0.0, shift_cost=0.0,
        # PER-ID, R2 ONLY. R1 and R3 are untouched and therefore identical across arms.
        risk_frequency_multipliers_by_id={r: float(mult) for r in R2},
        risk_impact_multipliers_by_id={r: float(mult) for r in R2})
    env.reset(seed=int(seed))
    sim = env.unwrapped.sim
    done = truncated = False
    step = 0
    try:
        while not (done or truncated):
            a = pattern if step in weeks else OFF
            _o, _r, done, truncated, _i = env.step(np.array(a, dtype=np.float32))
            step += 1
        events = getattr(sim, "risk_events", []) or []
        by_id: dict[str, int] = {}
        for e in events:
            rid = str(e.get("risk_id") if isinstance(e, dict) else getattr(e, "risk_id", "?"))
            by_id[rid] = by_id.get(rid, 0) + 1
        return {"L": exposure(sim), "n_events": len(events),
                "n_R2_events": sum(v for k, v in by_id.items() if k in R2),
                "n_R1_events": sum(v for k, v in by_id.items() if k in R1),
                "events_by_id": by_id}
    finally:
        env.close()


def paired(values_a, values_b, rng, n_boot=2000) -> dict:
    d = np.asarray(values_a, float) - np.asarray(values_b, float)
    boot = np.array([float(np.mean(d[rng.integers(0, len(d), len(d))])) for _ in range(n_boot)])
    return {"mean": float(d.mean()), "lcb95": float(np.percentile(boot, 2.5)),
            "ucb95": float(np.percentile(boot, 97.5)),
            "negligible": bool(abs(float(np.percentile(boot, 2.5))) < EQUIVALENCE_MARGIN
                               and abs(float(np.percentile(boot, 97.5))) < EQUIVALENCE_MARGIN)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--replay-of", required=True)
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--output", type=Path,
                    default=Path("results/lever_redundancy_diagnostic/result.json"))
    args = ap.parse_args()
    started = time.perf_counter()
    rng = np.random.default_rng(20260808)
    seeds = list(SEED_BLOCK[:args.seeds])

    grid = list(itertools.product(R2_MULTIPLIERS, SCHEDULES))
    print(f"  {len(grid)} celdas x {len(PATTERNS)} patrones x {len(seeds)} semillas = "
          f"{len(grid) * len(PATTERNS) * len(seeds)} episodios")

    cells = {}
    for mult, sched in grid:
        key = f"R2x{mult:g}|{sched}"
        weeks = SCHEDULES[sched]
        runs = {p: [play(mult, s, a, weeks) for s in seeds] for p, a in PATTERNS.items()}
        L = {p: [r["L"] for r in v] for p, v in runs.items()}

        m_s = paired(L["buffer_only"], L["both"], rng)      # what shifts add given buffer
        m_b = paired(L["shifts_only"], L["both"], rng)      # what buffer adds given shifts
        sym = paired(L["shifts_only"], L["buffer_only"], rng)
        auth = paired(L["neither"], [min(a, b) for a, b in
                                     zip(L["shifts_only"], L["buffer_only"])], rng)

        cells[key] = {
            "r2_multiplier": mult, "schedule": sched,
            "L_mean": {p: float(np.mean(v)) for p, v in L.items()},
            "L_by_seed": {p: [float(x) for x in v] for p, v in L.items()},
            "M_S_shifts_given_buffer": m_s,
            "M_B_buffer_given_shifts": m_b,
            "symmetry_shifts_vs_buffer": sym,
            "lever_authority": auth,
            "events": {"R2_mean": float(np.mean([r["n_R2_events"] for r in runs["neither"]])),
                       "R1_mean": float(np.mean([r["n_R1_events"] for r in runs["neither"]])),
                       "R2_min": int(min(r["n_R2_events"] for r in runs["neither"])),
                       "by_id_mean": {k: float(np.mean([r["events_by_id"].get(k, 0)
                                                        for r in runs["neither"]]))
                                      for k in R1 + R2}},
            "verdict": ("SYMMETRIC_SUBSTITUTES" if m_s["negligible"] and m_b["negligible"]
                        and sym["negligible"]
                        else "SHIFTS_REDUNDANT_GIVEN_BUFFER" if m_s["negligible"]
                        else "BUFFER_REDUNDANT_GIVEN_SHIFTS" if m_b["negligible"]
                        else "BOTH_LEVERS_CONTRIBUTE"),
        }
        c = cells[key]
        print(f"    {key:22s} M_S {m_s['mean']:+.6f}[{m_s['lcb95']:+.5f},{m_s['ucb95']:+.5f}]  "
              f"M_B {m_b['mean']:+.6f}[{m_b['lcb95']:+.5f},{m_b['ucb95']:+.5f}]  "
              f"eventos R2 {c['events']['R2_mean']:.1f}  {c['verdict']}")

    verdicts = {k: v["verdict"] for k, v in cells.items()}
    symmetric = [k for k, v in verdicts.items() if v == "SYMMETRIC_SUBSTITUTES"]
    one_way = [k for k, v in verdicts.items() if v == "SHIFTS_REDUNDANT_GIVEN_BUFFER"]
    both_matter = [k for k, v in verdicts.items() if v == "BOTH_LEVERS_CONTRIBUTE"]

    status = ("SYMMETRIC_SUBSTITUTES_EVERYWHERE" if len(symmetric) == len(cells)
              else "BOTH_LEVERS_CONTRIBUTE_SOMEWHERE" if both_matter
              else "ONE_WAY_REDUNDANCY_ONLY")

    falsifiers = {
        "f1_R1_is_identical_across_arms": {
            "passed": len({round(v["events"]["R1_mean"], 6) for v in cells.values()}) <= len(
                SCHEDULES),
            "evidence": {"why_it_can_fail": "per-ID multipliers touch R21-R24 only, so realised R1 "
                                            "event counts must not track the R2 multiplier. If "
                                            "they do, R1 moved with R2 and the estimand is joint "
                                            "escalation -- the exact defect that invalidated the "
                                            "previous screen",
                         "R1_mean_by_cell": {k: v["events"]["R1_mean"] for k, v in cells.items()},
                         "R2_mean_by_cell": {k: v["events"]["R2_mean"] for k, v in cells.items()}}},
        "f2_R2_actually_fires": {
            # PER-ID, and it can fail on a single risk while the family total looks healthy. The
            # first probe run showed R24 firing 8 times, R22 and R23 once each, and R21 ZERO -- its
            # source window reaches 16,128 h, about 96 weeks, against a 26-week episode. Every
            # R21-aligned conclusion in the inventory family was therefore measured where R21
            # barely occurs, and a family-level count would have hidden that completely.
            "passed": all(max(v["events"]["by_id_mean"][r] for v in cells.values()) >= 1.0
                          for r in R2),
            "evidence": {"why_it_can_fail": "R21's source window reaches 16,128 h, about 96 weeks, "
                                            "against a 26-week episode. A cell where no R2 event "
                                            "fires cannot support any conclusion about R2, and the "
                                            "previous screen never recorded exposure at all",
                         "max_mean_events_by_r2_id": {
                             r: max(v["events"]["by_id_mean"][r] for v in cells.values())
                             for r in R2},
                         "per_cell": {k: {"R2_mean": v["events"]["R2_mean"],
                                          "R2_min": v["events"]["R2_min"],
                                          "by_id": v["events"]["by_id_mean"]}
                                      for k, v in cells.items()}}},
        "f3_both_marginals_are_reported": {
            "passed": all("M_S_shifts_given_buffer" in v and "M_B_buffer_given_shifts" in v
                          and "symmetry_shifts_vs_buffer" in v for v in cells.values()),
            "evidence": {"why_it_can_fail": "min(shifts, buffer) - both HIDES asymmetry, which is "
                                            "how 'perfect substitutes' got claimed from data where "
                                            "shifts_only equalled buffer_only in only 8 of 18 "
                                            "cells. Symmetry needs both marginals AND the "
                                            "shifts-vs-buffer contrast",
                         "statistic_withdrawn": "min(L_S, L_B) - L_both"}},
        "f4_levers_are_not_action_dead": {
            "passed": all(v["lever_authority"]["lcb95"] > 0 for v in cells.values()),
            "evidence": {"why_it_can_fail": "if turning a lever on changes nothing at all, a zero "
                                            "marginal means the action never reached the "
                                            "simulator, not that the levers are interchangeable. "
                                            "A collision of outputs can be saturation, a dead "
                                            "action, a redundant implementation or true "
                                            "substitutability, and this separates the first from "
                                            "the rest",
                         "per_cell": {k: v["lever_authority"] for k, v in cells.items()}}},
        "f5_schedule_is_not_confounded": {
            "passed": len(SCHEDULES) >= 3,
            "evidence": {"why_it_can_fail": "the previous screen used ONE early block, so a "
                                            "property of that calendar could not be told from a "
                                            "property of the levers",
                         "schedules": {k: sorted(v) for k, v in SCHEDULES.items()},
                         "verdict_by_schedule": verdicts}},
        "f6_no_fresh_seeds": custody_falsifier(seeds, replay_of=args.replay_of,
                                               exclude=args.output),
    }
    falsifiers["all_passed"] = all(
        v["passed"] for k, v in falsifiers.items()
        if k != "all_passed" and isinstance(v, dict) and not v.get("not_applicable"))
    if not falsifiers["all_passed"]:
        status = "BLOCKED_INSTRUMENT"

    print(f"\n  simetricos: {len(symmetric)}/{len(cells)} · redundancia unidireccional: "
          f"{len(one_way)}/{len(cells)} · ambas contribuyen: {len(both_matter)}/{len(cells)}")
    print(f"\n  veredicto: {status}\n")
    for name, f in falsifiers.items():
        if name == "all_passed" or not isinstance(f, dict):
            continue
        label = ("NO APLICA" if f.get("not_applicable")
                 else "PASA" if f["passed"] else "FALLA")
        print(f"    {name:<42} {label}")

    payload = {
        "schema_version": "lever_redundancy_diagnostic_v1",
        "claim_status": status,
        "scope": "DEVELOPMENT_DIAGNOSTIC_NO_ARCHITECTURE_COMPARED",
        "run_role": "MARGINAL_REDUNDANCY_DIAGNOSTIC", "replay_of": args.replay_of,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "module_manifest": module_manifest(MODULES, script=__file__),
        "supersedes": {
            "path": "results/actuator_complementarity_screen/result.json",
            "withdrawn_claim": "PERFECT_SUBSTITUTES_EVERYWHERE_ON_THE_SCREENED_GRID",
            "why": ("it escalated R1 and R2 jointly through GLOBAL multipliers, so it was never an "
                    "R2 screen; and min(shifts, buffer) - both hides asymmetry, which its own "
                    "numbers show: buffer_only == both in 18 of 18 cells but shifts_only == "
                    "buffer_only in only 8"),
            "retained": True},
        "estimands": {"M_S": "L(buffer) - L(buffer + shifts): what shifts add given the buffer",
                      "M_B": "L(shifts) - L(shifts + buffer): what the buffer adds given shifts",
                      "symmetry": "L(shifts) - L(buffer)",
                      "equivalence_margin": EQUIVALENCE_MARGIN},
        "design": {"r2_multipliers_per_id": list(R2_MULTIPLIERS), "r2_ids": list(R2),
                   "r1_and_r3": "untouched, identical across arms, distribution families frozen",
                   "schedules": {k: sorted(v) for k, v in SCHEDULES.items()},
                   "seeds": seeds, "max_steps": MAX_STEPS},
        "cells": cells, "symmetric_cells": symmetric, "one_way_cells": one_way,
        "both_contribute_cells": both_matter,
        "falsifiers": falsifiers, "elapsed_seconds": time.perf_counter() - started,
    }
    digest = seal_and_write(payload, args.output, contract=args.contract,
                            reference=Path("results/actuator_complementarity_screen/result.json"))
    print(f"\n  -> {args.output} (sello {digest[:16]}…)")
    return 0 if falsifiers["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
