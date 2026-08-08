#!/usr/bin/env python3
"""Seasonal phase as the signal: one arm that cannot convert, and one that could.

Contract: `docs/ENMIENDA_SENAL_FASE_ESTACIONAL_2026-08-08.md`, committed before this file. Custody:
declared replay. Falsifiers from `supply_chain.falsifiers`.

MEASURED BEFORE THE DESIGN, and it is what the design turns on: the seasonal phase is `week mod 12`
and IDENTICAL on every tape, with the trough at phase 11 (scale 0.35 against 1.059). A policy
reading only the phase is therefore a deterministic function of time -- open-loop -- and the
clairvoyant gap is by construction `best-per-tape minus best-fixed`, the part that requires knowing
the tape. So arm A CANNOT convert; it can only be a better fixed schedule, which answers the
different and still legitimate question of whether my contiguous-block class was too narrow.

Arm B adds the one thing the phase lacks: a signal that VARIES ACROSS TAPES AT THE SAME t. It holds
only when last week's realised demand exceeded its seasonal expectation, which is exactly the
property the backlog rule failed to exploit.
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

from scripts.run_priced_buffer_gate_v1 import (  # noqa: E402
    MAX_STEPS, SCENARIO, STEP_HOURS, exposure, make_env, options,
)
from supply_chain.arm_runner import seal_and_write  # noqa: E402
from supply_chain.falsifiers import (  # noqa: E402
    disclosure, ge, gt, not_applicable, summarise,
)
from supply_chain.seed_custody import custody_falsifier, module_manifest  # noqa: E402

LAMBDA_HEADLINE = 0.35
BAND = (0.275, 0.30, 0.325, 0.35, 0.375, 0.40, 0.425, 0.45, 0.475, 0.50)
PHASE_PERIOD = 12
WIDTHS = (2, 4, 6, 8)
N_BOOT = 4_000
N_PLACEBO = 40
SEED_BLOCK = tuple(range(8600001, 8600013))
CEILING = Path("results/priced_clairvoyant_ceiling/result.json")
MODULES = ("supply_chain/supply_chain.py", "supply_chain/continuous_its_env.py",
           "supply_chain/falsifiers.py", "supply_chain/arm_runner.py",
           "supply_chain/seed_custody.py")


def window(width: int, offset: int) -> set:
    return {(offset + i) % PHASE_PERIOD for i in range(width)}


POLICIES = [(w, o) for w in WIDTHS for o in range(PHASE_PERIOD)]


def play_phase(policy, seed: int, arm: str, placebo_weeks=None) -> dict:
    """Arm A holds on phase alone. Arm B also requires last week's demand above its expectation."""
    width, offset = policy
    phases = window(width, offset)
    env = make_env()
    env.reset(seed=int(seed))
    sim = env.unwrapped.sim
    sd = sim.demand_seasonal
    done = truncated = False
    step, inv_hours, prev_demanded = 0, 0.0, 0.0
    held, phase_seen, deviations = [], [], []
    try:
        while not (done or truncated):
            now = float(sim.env.now)
            phase = int(sd.phase(now)) if sd is not None else 0
            phase_seen.append(phase)
            demanded = float(getattr(sim, "total_demanded", 0.0))
            last_week = demanded - prev_demanded
            # Expectation for the week just observed, from the seasonal profile alone.
            expect = float(sd.scale(max(now - STEP_HOURS, 0.0))) if sd is not None else 1.0
            dev = (last_week / (expect + 1e-9)) if step > 0 else 0.0
            deviations.append(dev)
            if placebo_weeks is not None:
                on = step in placebo_weeks
            elif arm == "A":
                on = phase in phases
            else:
                # STATE: hold only when the phase window is open AND last week ran hot relative to
                # its own seasonal expectation. `dev` differs across tapes at the same t; the phase
                # does not.
                ref = float(np.median(deviations[:-1])) if step > 1 else dev
                on = (phase in phases) and (dev > ref)
            held.append(int(on))
            prev_demanded = demanded
            _o, _r, done, truncated, _i = env.step(
                np.array([1.0 if on else 0.0, -1.0], dtype=np.float32))
            inv_hours += (1.0 if on else 0.0) * STEP_HOURS
            step += 1
        return {"L": exposure(sim), "inventory_hours": inv_hours,
                "weeks_held": int(sum(held)), "schedule": held,
                "phases": phase_seen, "deviations": deviations}
    finally:
        env.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--replay-of", required=True)
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--output", type=Path,
                    default=Path("results/phase_signal_conversion/result.json"))
    args = ap.parse_args()
    started = time.perf_counter()
    rng = np.random.default_rng(20260808)
    seeds = list(SEED_BLOCK[:args.seeds])
    train_seeds, test_seeds = seeds[:6], seeds[6:]
    opts = options()

    ceil = json.loads(CEILING.read_text())
    L_s = np.asarray(ceil["L_matrix"], dtype=float)
    IH_s = np.asarray(ceil["inventory_hours_matrix"], dtype=float)
    max_ih = float(ceil["max_inventory_hours"])
    order = list(ceil["splits"]["train"]) + list(ceil["splits"]["test"])
    idx = {s: order.index(s) for s in order}
    print(f"  {len(POLICIES)} politicas de fase x 2 brazos x {len(seeds)} semillas")

    runs = {arm: {p: {s: play_phase(p, s, arm) for s in seeds} for p in POLICIES}
            for arm in ("A", "B")}
    print("    brazos listos")

    def J_sched(seed, j, lam):
        i = idx[seed]
        return L_s[i, j] + lam * (IH_s[i, j] / max_ih)

    def J(r, lam):
        return r["L"] + lam * (r["inventory_hours"] / max_ih)

    results = {}
    for lam in BAND:
        fixed = int(np.argmin([np.mean([J_sched(s, j, lam) for s in train_seeds])
                               for j in range(len(opts))]))
        open_loop = np.array([J_sched(s, fixed, lam) for s in test_seeds])
        clair = np.array([min(J_sched(s, j, lam) for j in range(len(opts))) for s in test_seeds])
        cell = {"lambda": lam, "fixed_option": list(opts[fixed]),
                "open_loop_J": float(open_loop.mean()),
                "clairvoyant_J": float(clair.mean()),
                "ceiling_gap": float((open_loop - clair).mean()), "arms": {}}
        for arm in ("A", "B"):
            best = min(POLICIES,
                       key=lambda p: np.mean([J(runs[arm][p][s], lam) for s in train_seeds]))
            arm_J = np.array([J(runs[arm][best][s], lam) for s in test_seeds])
            held = [runs[arm][best][s]["weeks_held"] for s in test_seeds]
            plac = []
            for s, k in zip(test_seeds, held):
                vals = [J(play_phase(best, s, arm, placebo_weeks=set(
                    rng.choice(MAX_STEPS, size=min(k, MAX_STEPS), replace=False).tolist())), lam)
                    for _ in range(N_PLACEBO)]
                plac.append(float(np.mean(vals)))
            plac = np.array(plac)

            def boot(d):
                b = np.array([float(np.mean(d[rng.integers(0, len(d), len(d))]))
                              for _ in range(N_BOOT)])
                return {"mean": float(d.mean()), "lcb95": float(np.percentile(b, 2.5)),
                        "ucb95": float(np.percentile(b, 97.5))}
            cell["arms"][arm] = {
                "policy": list(best), "arm_J": float(arm_J.mean()),
                "placebo_J": float(plac.mean()), "weeks_held": held,
                "vs_open_loop": boot(open_loop - arm_J),
                "vs_placebo": boot(plac - arm_J),
                "schedules_identical_across_tapes": len(
                    {tuple(runs[arm][best][s]["schedule"]) for s in seeds}) == 1,
            }
        results[str(lam)] = cell
        a, b = cell["arms"]["A"], cell["arms"]["B"]
        print(f"    lambda {lam:.3f} techo {cell['ceiling_gap']:+.6f} | A {a['vs_open_loop']['mean']:+.6f}"
              f" [{a['vs_open_loop']['lcb95']:+.6f}] | B {b['vs_open_loop']['mean']:+.6f}"
              f" [{b['vs_open_loop']['lcb95']:+.6f}]")

    head = results[str(LAMBDA_HEADLINE)]
    dev_spread = float(np.mean([
        np.std([runs["B"][POLICIES[0]][s]["deviations"][t] for s in seeds])
        for t in range(1, MAX_STEPS)]))
    phase_identical = len({tuple(runs["A"][POLICIES[0]][s]["phases"]) for s in seeds}) == 1
    a_identical = all(v["arms"]["A"]["schedules_identical_across_tapes"] for v in results.values())

    converts = [k for k, v in results.items()
                if v["arms"]["B"]["vs_open_loop"]["lcb95"] > 0
                and v["arms"]["B"]["vs_placebo"]["lcb95"] > 0]
    a_beats_fixed = [k for k, v in results.items() if v["arms"]["A"]["vs_open_loop"]["lcb95"] > 0]

    falsifiers = {
        "f7_phase_is_deterministic_across_tapes": ge(
            float(phase_identical), 1.0,
            "if the phase varied across tapes, arm A would not be open-loop and the whole framing "
            "of the amendment collapses; it is checked, not assumed",
            phase_sequence=runs["A"][POLICIES[0]][seeds[0]]["phases"][:13]),
        "f8_arm_A_cannot_convert_per_tape_headroom": ge(
            float(a_identical), 1.0,
            "arm A's realised calendar must be IDENTICAL on every tape. If it differs, either the "
            "phase is not deterministic or the runner leaked state into a policy declared "
            "open-loop",
            identical_at_every_lambda=a_identical),
        "f9_arm_B_signal_varies_across_tapes": gt(
            dev_spread, 0.0,
            "if the demand deviation were constant across tapes, arm B would be open-loop in "
            "disguise and could not convert either -- the property backlog had and failed to use",
            mean_cross_tape_sd=dev_spread),
        "f10_policy_does_not_exceed_the_ceiling": ge(
            min(v["ceiling_gap"] - v["arms"][a]["vs_open_loop"]["mean"]
                for v in results.values() for a in ("A", "B")), -1e-9,
            "no policy may beat a tape-knowing chooser; exceeding the ceiling means it saw "
            "something it should not have",
            slack=min(v["ceiling_gap"] - v["arms"][a]["vs_open_loop"]["mean"]
                      for v in results.values() for a in ("A", "B"))),
        "f11_ceiling_still_positive_here": gt(
            head["ceiling_gap"], 0.0,
            "conversion is undefined where there is nothing to convert",
            ceiling_by_lambda={k: v["ceiling_gap"] for k, v in results.items()}),
        "d1_arm_A_is_open_loop_by_construction": disclosure(
            "arm A reads only week mod 12, so it is a deterministic schedule; a win by arm A says "
            "the contiguous-block class was too narrow, NOT that observable control converts",
            trough_phase=11, period=PHASE_PERIOD),
        "d2_lambda_was_selected_on_these_tapes": disclosure(
            "lambda = 0.35 is the peak of a sweep run on these same tapes; the whole band is "
            "reported beside it", headline=LAMBDA_HEADLINE, band=list(BAND)),
        "d3_no_fresh_seeds": not_applicable(
            "declared replay of an already-consumed development block",
            custody=custody_falsifier(seeds, replay_of=args.replay_of, exclude=args.output)),
    }
    summary = summarise(falsifiers)
    if not summary["all_passed"]:
        verdict = "BLOCKED_INSTRUMENT"
    elif str(LAMBDA_HEADLINE) in converts:
        verdict = "PHASE_PLUS_STATE_CONVERTS_AT_THE_HEADLINE_PRICE"
    elif converts:
        verdict = "PHASE_PLUS_STATE_CONVERTS_ONLY_OUTSIDE_THE_HEADLINE_PRICE"
    elif a_beats_fixed:
        verdict = "OPEN_LOOP_PHASE_SCHEDULE_BEATS_THE_ENUMERATED_CLASS_NO_CONVERSION"
    else:
        verdict = "NEITHER_PHASE_ARM_CONVERTS"

    print(f"\n  lambda {LAMBDA_HEADLINE}: techo {head['ceiling_gap']:+.6f}")
    for arm in ("A", "B"):
        v = head["arms"][arm]
        print(f"    brazo {arm}: politica {v['policy']}  J {v['arm_J']:.6f}  "
              f"vs open-loop {v['vs_open_loop']['mean']:+.6f} "
              f"[{v['vs_open_loop']['lcb95']:+.6f}]  vs placebo "
              f"{v['vs_placebo']['mean']:+.6f} [{v['vs_placebo']['lcb95']:+.6f}]")
    print(f"\n  veredicto: {verdict}")
    print(f"  falsadores: {summary['n_computed']} computados, {summary['n_failed']} fallidos, "
          f"{summary['n_disclosures']} divulgaciones, {summary['n_not_applicable']} no aplicables")
    for k, v in falsifiers.items():
        if v.get("computed"):
            print(f"    {k:52s} {'PASA' if v['passed'] else 'FALLA'}")

    payload = {
        "schema_version": "phase_signal_conversion_v1",
        "claim_status": verdict,
        "scope": "DEVELOPMENT_ON_DECLARED_REPLAY_NO_LEARNER_AUTHORIZED",
        "run_role": "PHASE_SIGNAL_CONVERSION", "replay_of": args.replay_of,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "module_manifest": module_manifest(MODULES, script=__file__),
        "ceiling_source": {"path": str(CEILING), "self_sha256": ceil.get("self_sha256")},
        "arms": {"A": "phase only -- deterministic in t, OPEN-LOOP, cannot convert",
                 "B": "phase AND last week's demand vs its seasonal expectation -- reads state"},
        "policy_family": {"widths": list(WIDTHS), "offsets": list(range(PHASE_PERIOD)),
                          "selected_on": "train tapes only"},
        "headline_lambda": LAMBDA_HEADLINE, "band": list(BAND),
        "splits": {"train": train_seeds, "test": test_seeds},
        "results": results, "headline": head,
        "converts_at": converts, "arm_A_beats_fixed_at": a_beats_fixed,
        "falsifiers": falsifiers, "falsifier_summary": summary,
        "elapsed_seconds": time.perf_counter() - started,
    }
    digest = seal_and_write(payload, args.output, contract=args.contract, reference=CEILING)
    print(f"\n  -> {args.output} (sello {digest[:16]}…)")
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
