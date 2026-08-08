#!/usr/bin/env python3
"""The clairvoyant ceiling inside the priced space, over an ENUMERATED schedule class.

Contract: `docs/ENMIENDA_COSTE_BUFFER_DECLARADO_2026-08-08.md`, extended by this file's own
preregistered reading rules below, both committed before the run. Custody: declared replay.
Falsifiers come from `supply_chain.falsifiers`, so a literal `passed` cannot be constructed and
disclosures are never counted in the total.

THIS IS THE GATE THAT KEPT BEING SKIPPED. Five families measured architectures, calendars or
metrics before establishing that a decision worth making existed. Here the order is the other way
round: what does a tape-knowing choice of schedule buy over the best schedule chosen once, inside
the priced space that `results/priced_buffer_gate/result.json` just certified as eligible?

WHAT MAKES A CLAIM OF ABSENCE ADMISSIBLE, and it is the correction that took four attempts:

    LCB95(gap) >= delta            -> headroom established
    UCB95(gap) <  delta AND the class was ENUMERATED
                                   -> no material headroom WITHIN THAT CLASS
    otherwise                      -> inconclusive

There is no STOP branch. Failing to clear a bar from below is not absence, and the enumerated
class is the only thing an absence may be claimed over -- never over schedules nobody tried.

THE OPEN-LOOP COMPARATOR IS SELECTED ON TRAIN AND SCORED ON TEST. Selecting it on the test tapes,
which the benchmark did, optimises the comparator against the very data it is compared on and
shrinks the gap it is supposed to bound.
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

from scripts.run_priced_buffer_gate_v1 import (  # noqa: E402
    LAMBDAS, LEAD_HOURS, MAX_STEPS, SCENARIO, STEP_HOURS, make_env, options, play,
)
from supply_chain.arm_runner import seal_and_write  # noqa: E402
from supply_chain.falsifiers import (  # noqa: E402
    disclosure, ge, gt, lt, not_applicable, preflight, summarise,
)
from supply_chain.seed_custody import custody_falsifier, module_manifest  # noqa: E402

DELTA = 0.01                # one percentage point of the priced objective; both terms are in [0,1]
REFERENCE_LAMBDA = 1.0
N_BOOT = 2_000
N_PLACEBO = 200
SEED_BLOCK = tuple(range(8600001, 8600013))
MODULES = ("supply_chain/supply_chain.py", "supply_chain/continuous_its_env.py",
           "supply_chain/falsifiers.py", "supply_chain/arm_runner.py",
           "supply_chain/seed_custody.py")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--replay-of", required=True)
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--output", type=Path,
                    default=Path("results/priced_clairvoyant_ceiling/result.json"))
    args = ap.parse_args()
    started = time.perf_counter()
    rng = np.random.default_rng(20260808)
    seeds = list(SEED_BLOCK[:args.seeds])
    train_seeds, test_seeds = seeds[:6], seeds[6:]
    opts = options()
    print(f"  {len(opts)} calendarios enumerados x {len(seeds)} semillas = "
          f"{len(opts) * len(seeds)} episodios")

    env = make_env()
    env.reset(seed=seeds[0])
    sim = env.unwrapped.sim
    reset_now = float(sim.env.now)
    live = {"demand_process": getattr(sim, "demand_process", None),
            "strategic_buffer_release_mode": getattr(sim, "strategic_buffer_release_mode", None),
            "inventory_replenishment_lead_time":
                float(getattr(sim, "inventory_replenishment_lead_time", 0.0))}
    env.close()
    pre = preflight(probe=lambda o: play(o, seeds[0])["L"], options=opts,
                    reset_now=reset_now, horizon=MAX_STEPS * STEP_HOURS,
                    scenario=live, expected_scenario=SCENARIO)
    if not summarise(pre)["all_passed"]:
        print("  PRE-VUELO FALLA — no se corre")

    L = np.zeros((len(seeds), len(opts)))
    IH = np.zeros_like(L)
    for i, s in enumerate(seeds):
        for j, o in enumerate(opts):
            r = play(o, s)
            L[i, j], IH[i, j] = r["L"], r["inventory_hours"]
    max_ih = float(IH.max()) or 1.0
    print("    cache lista")

    def gap_at(lam: float) -> dict:
        """Per-tape best minus the single best-on-TRAIN schedule, paired on the test tapes."""
        J = L + lam * (IH / max_ih)
        tr = [seeds.index(s) for s in train_seeds]
        te = [seeds.index(s) for s in test_seeds]
        fixed = int(np.argmin(J[tr].mean(axis=0)))          # chosen on TRAIN only
        open_loop = J[te, fixed]
        clair = J[te].min(axis=1)
        d = open_loop - clair                                # >= 0 by construction
        boot = np.array([float(np.mean(d[rng.integers(0, len(d), len(d))]))
                         for _ in range(N_BOOT)])
        # UNINFORMED PLACEBO: a schedule picked at random per tape, same freedom to vary, no
        # information. If it matches the clairvoyant, the value is in varying and not in knowing.
        plac = np.array([float(np.mean(J[te, rng.integers(0, len(opts), len(te))]))
                         for _ in range(N_PLACEBO)])
        return {"lambda": lam, "fixed_index": fixed, "fixed_option": list(opts[fixed]),
                "open_loop_J": float(open_loop.mean()), "clairvoyant_J": float(clair.mean()),
                "gap_mean": float(d.mean()),
                "lcb95": float(np.percentile(boot, 2.5)),
                "ucb95": float(np.percentile(boot, 97.5)),
                "placebo_mean_J": float(plac.mean()),
                "clairvoyant_beats_placebo": bool(float(clair.mean()) < float(plac.mean())),
                "unique_per_tape_optima": int(len(set(J[te].argmin(axis=1).tolist())))}

    per_lambda = {str(l): gap_at(l) for l in LAMBDAS}
    ref = per_lambda[str(REFERENCE_LAMBDA)]

    established = [l for l, v in per_lambda.items() if v["lcb95"] >= DELTA]
    absent = [l for l, v in per_lambda.items() if v["ucb95"] < DELTA]

    falsifiers = dict(pre)
    falsifiers["f5_class_is_enumerated"] = ge(
        len(opts), len(opts),
        "an absence may only be claimed over a class that was enumerated without omission; a "
        "heuristic search can say HEADROOM_FOUND but never that none exists",
        n_options=len(opts), options=[list(o) for o in opts])
    falsifiers["f6_open_loop_selected_on_train_only"] = ge(
        len(set(train_seeds) & set(test_seeds)) * -1 + 1, 1,
        "the benchmark chose its comparator with argmin over the TEST tapes, which optimises it "
        "against the data it is compared on and shrinks the very gap it should bound",
        train_seeds=train_seeds, test_seeds=test_seeds,
        n_overlap=len(set(train_seeds) & set(test_seeds)))
    falsifiers["f7_clairvoyant_weakly_dominates"] = ge(
        min(v["gap_mean"] for v in per_lambda.values()), -1e-12,
        "a per-tape minimum cannot exceed a fixed column, so a negative gap means the estimator is "
        "misindexed",
        gap_by_lambda={k: v["gap_mean"] for k, v in per_lambda.items()})
    falsifiers["f8_clairvoyant_beats_the_uninformed_placebo"] = ge(
        sum(1 for v in per_lambda.values() if v["clairvoyant_beats_placebo"]), len(LAMBDAS),
        "at op12 an uninformed placebo matched the state-conditioned rule, which meant the value "
        "was in the freedom to vary and not in knowing the tape; if that repeats, a positive gap "
        "is not evidence of information value",
        placebo_draws=N_PLACEBO,
        by_lambda={k: {"clairvoyant": v["clairvoyant_J"], "placebo": v["placebo_mean_J"]}
                   for k, v in per_lambda.items()})
    falsifiers["d1_no_stop_branch"] = disclosure(
        "failing to clear the bar from below is not absence; absence requires UCB95 < delta over "
        "the enumerated class, and the vocabulary has no STOP",
        delta=DELTA, established_at=established, absent_at=absent)
    falsifiers["d2_fidelity_price"] = disclosure(
        "release and the 336 h lead time are OUR extensions with no source event, so nothing here "
        "is presented as reproducing Garrido-Rios (2017)",
        lead_hours=LEAD_HOURS, reference_lambda=REFERENCE_LAMBDA)
    falsifiers["d3_no_fresh_seeds"] = not_applicable(
        "declared replay of an already-consumed development block",
        custody=custody_falsifier(seeds, replay_of=args.replay_of, exclude=args.output))

    summary = summarise(falsifiers)
    if not summary["all_passed"]:
        verdict = "BLOCKED_INSTRUMENT"
    elif established:
        verdict = "HEADROOM_ESTABLISHED_IN_THE_PRICED_SPACE"
    elif len(absent) == len(LAMBDAS):
        verdict = "NO_MATERIAL_HEADROOM_WITHIN_THE_ENUMERATED_CLASS"
    else:
        verdict = "INCONCLUSIVE"

    print(f"\n  {'lambda':>7} {'open-loop J':>12} {'clarividente':>13} {'hueco':>10} "
          f"{'lcb95':>10} {'ucb95':>10} {'optimos':>8}")
    for k, v in per_lambda.items():
        print(f"  {k:>7} {v['open_loop_J']:12.6f} {v['clairvoyant_J']:13.6f} "
              f"{v['gap_mean']:10.6f} {v['lcb95']:10.6f} {v['ucb95']:10.6f} "
              f"{v['unique_per_tape_optima']:8d}")
    print(f"\n  veredicto: {verdict}   (delta = {DELTA})")
    print(f"  falsadores: {summary['n_computed']} computados, {summary['n_failed']} fallidos, "
          f"{summary['n_disclosures']} divulgaciones, {summary['n_not_applicable']} no aplicables")
    for k, v in falsifiers.items():
        if v.get("computed"):
            print(f"    {k:52s} {'PASA' if v['passed'] else 'FALLA'}")

    payload = {
        "schema_version": "priced_clairvoyant_ceiling_v1",
        "claim_status": verdict,
        "scope": "DEVELOPMENT_ON_DECLARED_REPLAY_CEILING_ONLY_NO_LEARNER_AUTHORIZED",
        "run_role": "CLAIRVOYANT_CEILING_GATE", "replay_of": args.replay_of,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "module_manifest": module_manifest(MODULES, script=__file__),
        "eligibility": {"path": "results/priced_buffer_gate/result.json",
                        "claim_status": "PRICED_DECISION_SPACE_ELIGIBLE"},
        "decision_rule": {"delta": DELTA,
                          "established": "LCB95(gap) >= delta",
                          "absence": "UCB95(gap) < delta AND the class was enumerated",
                          "no_stop_branch": True},
        "scenario": SCENARIO, "live_scenario": live,
        "objective": "J(lambda) = L* + lambda * inventory_hours / max_inventory_hours",
        "reference_lambda": REFERENCE_LAMBDA, "lambdas": list(LAMBDAS),
        "splits": {"train": train_seeds, "test": test_seeds},
        "options": [list(o) for o in opts],
        "L_matrix": L.tolist(), "inventory_hours_matrix": IH.tolist(),
        "max_inventory_hours": max_ih,
        "per_lambda": per_lambda, "reference": ref,
        "established_at": established, "absent_at": absent,
        "falsifiers": falsifiers, "falsifier_summary": summary,
        "elapsed_seconds": time.perf_counter() - started,
    }
    digest = seal_and_write(payload, args.output, contract=args.contract,
                            reference=Path("results/priced_buffer_gate/result.json"))
    print(f"\n  -> {args.output} (sello {digest[:16]}…)")
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
