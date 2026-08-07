#!/usr/bin/env python3
"""Readjudicate the GSA lane under the PI's declared objective: resilience is the measure.

WHAT CHANGED. results/headroom_gsa/oos_guardrail_check.json carries H_obs positive and OOS-stable
on three independent blocks (+0.0131 / +0.0114 / +0.0100, CI95>0) and closes anyway, with this
reason: "worst_cssu_fill_delta -0.13 << -0.02 fairness guardrail". Its own verdict calls the
blocker "the Program-G concentration/fairness artifact". That is a distributional guardrail, not a
resilience one, and on 2026-08-07 the PI declared resilience the only measure.

WHY THAT IS LEGITIMATE HERE AND WOULD NOT BE ELSEWHERE. ret_excel on the workbook-visible
population is MEASURED to reward abandonment -- the split maximising it delivers 50% fill against
80% for the split minimising it. So "only resilience matters" is a trap under that metric. It is
not a trap here: ret_order_metrics (supply_chain/program_g.py:320) marks unattended orders lost and
scores them ZERO, so abandonment is already priced. The distinction is what makes this rerun
legitimate rather than convenient, and it goes in the artifact.

WHAT THIS ADDS THAT THE HISTORICAL RUN DID NOT HAVE. An uninformed placebo. The project requires
one in every headroom measurement and headroom_sensitivity.py has none: it compares obs against a
static calendar only. Here the placebo is the action sequence the belief policy produced on a
DIFFERENT tape, applied to this one -- same action distribution, no aligned information. If the
placebo reproduces the gain, the value is in the cadence rather than the signal, which is exactly
what was already measured at op12.

Preregistration: docs/PREREGISTRO_GSA_BAJO_RESILIENCIA_UNICA_2026-08-07.md
Development on already-open blocks. No new seeds.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from supply_chain.arm_runner import seal_and_write  # noqa: E402
from supply_chain.headroom_sensitivity import (  # noqa: E402
    ACTIONS, ARM, WEEKS, _belief_policy, materialize_tape_theta, periodic_calendars, theta_to_cell,
)
from supply_chain.program_g import ret_order_metrics, simulate_orders  # noqa: E402
from supply_chain.seed_custody import module_manifest  # noqa: E402

MODULES = ("supply_chain/headroom_sensitivity.py", "supply_chain/program_g.py",
           "supply_chain/arm_runner.py")
# The located optimum from results/headroom_gsa/oos_guardrail_check.json. `persistence` is encoded
# as a float here because theta_to_cell thresholds it at 0.5; "short" is anything below.
THETA = {"signal_q": 0.532, "lead": 2, "surge_mult": 1.946, "persistence": 0.0,
         "commonality": 0.887, "r22_prob": 0.107}
BLOCKS = {"GP_search_3000001": 3_000_001, "FRESH_4200001": 4_200_001, "FRESH_4500001": 4_500_001}
HISTORICAL = {"GP_search_3000001": 0.0131, "FRESH_4200001": 0.0114, "FRESH_4500001": 0.0100}
SEALED_H_PI = 0.014446048488184385
N_BOOT = 5_000


def score(tape, seq) -> dict:
    """ReT plus the distributional outcomes, which are REPORTED and never blocking.

    Orders are emitted two per day, CSSU 0 then CSSU 1, so creation parity identifies the
    claimant exactly -- see the loop in program_g.simulate_orders."""
    orders = simulate_orders(tape, seq, ARM)
    m = ret_order_metrics(orders)
    fills = []
    for cssu in (0, 1):
        own = orders[cssu::2]
        served = sum(getattr(o, "OATj", None) is not None for o in own)
        fills.append(served / max(1, len(own)))
    return {"ret_order": m["ret_order"], "ret_quantity": m["ret_quantity"],
            "attended": float(m["attended"]), "lost": float(m["lost"]),
            "worst_cssu_fill": float(min(fills))}


def boot(diff: np.ndarray, rng) -> dict:
    draws = diff[rng.integers(0, diff.size, size=(N_BOOT, diff.size))].mean(axis=1)
    return {"mean": float(diff.mean()), "lcb95": float(np.percentile(draws, 2.5)),
            "ucb95": float(np.percentile(draws, 97.5)), "n": int(diff.size)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-tapes", type=int, default=200)
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--output", type=Path,
                    default=Path("results/gsa_resilience_only/result.json"))
    args = ap.parse_args()
    started = time.perf_counter()
    rng = np.random.default_rng(20260807)
    cell = theta_to_cell(THETA)
    cals = periodic_calendars(WEEKS)
    all_seqs = list(itertools.product(ACTIONS, repeat=WEEKS))

    blocks, static_is_argmax = {}, True
    for name, seed0 in BLOCKS.items():
        tapes = [materialize_tape_theta(seed0 + i, cell) for i in range(args.n_tapes)]
        # The strongest static calendar, chosen ON these tapes: the tightest possible baseline.
        by_cal = np.array([[score(t, c)["ret_order"] for t in tapes] for c in cals])
        best_cal = int(by_cal.mean(axis=1).argmax())
        static_is_argmax &= bool(by_cal.mean(axis=1)[best_cal] >= by_cal.mean(axis=1).max() - 1e-12)
        static_rows = [score(t, cals[best_cal]) for t in tapes]

        seqs = [_belief_policy(t) for t in tapes]
        obs_rows = [score(t, s) for t, s in zip(tapes, seqs)]
        # Uninformed placebo: this tape, another tape's action sequence (a derangement).
        placebo_rows = [score(t, seqs[(i + 1) % len(tapes)]) for i, t in enumerate(tapes)]
        oracle = np.array([max(score(t, s)["ret_order"] for s in all_seqs) for t in tapes])

        def col(rows, k):
            return np.array([r[k] for r in rows], dtype=float)

        st, ob, pl = col(static_rows, "ret_order"), col(obs_rows, "ret_order"), \
            col(placebo_rows, "ret_order")
        att_gap = col(obs_rows, "attended") - col(static_rows, "attended")
        ret_gap = ob - st
        corr = (float(np.corrcoef(ret_gap, att_gap)[0, 1])
                if np.std(ret_gap) > 1e-12 and np.std(att_gap) > 1e-12 else 0.0)

        blocks[name] = {
            "seed0": seed0, "n_tapes": args.n_tapes,
            "H_PI": float((oracle - st).mean()), "H_obs": float(ret_gap.mean()),
            "H_obs_ci": boot(ret_gap, rng),
            "obs_minus_placebo": boot(ob - pl, rng),
            "eta": float(ret_gap.mean() / (oracle - st).mean())
            if abs((oracle - st).mean()) > 1e-9 else 0.0,
            "ret_quantity_delta": float((col(obs_rows, "ret_quantity")
                                         - col(static_rows, "ret_quantity")).mean()),
            # Reported, never blocking. This is the PI's declared decision, recorded as a decision.
            "reported_not_blocking": {
                "worst_cssu_fill_delta": float((col(obs_rows, "worst_cssu_fill")
                                                - col(static_rows, "worst_cssu_fill")).mean()),
                "attended_delta": float(att_gap.mean()),
                "lost_delta": float((col(obs_rows, "lost") - col(static_rows, "lost")).mean())},
            "corr_ret_gain_with_attended_gain": corr,
            "historical_H_obs": HISTORICAL[name],
        }
        b = blocks[name]
        print(f"  {name:<20} H_obs {b['H_obs']:+.5f} [{b['H_obs_ci']['lcb95']:+.5f}]  "
              f"obs-placebo {b['obs_minus_placebo']['mean']:+.5f} "
              f"[{b['obs_minus_placebo']['lcb95']:+.5f}]  "
              f"worstfill {b['reported_not_blocking']['worst_cssu_fill_delta']:+.4f} "
              f"({time.perf_counter()-started:.0f}s)", flush=True)

    h_pi_ref = blocks["GP_search_3000001"]["H_PI"]
    worst_corr = min(b["corr_ret_gain_with_attended_gain"] for b in blocks.values())
    all_hobs = all(b["H_obs_ci"]["lcb95"] > 0 for b in blocks.values())
    all_plac = all(b["obs_minus_placebo"]["lcb95"] > 0 for b in blocks.values())

    falsifiers = {
        "f1_the_historical_cell_still_reproduces": {
            "passed": bool(abs(h_pi_ref - SEALED_H_PI) <= 2e-3),
            "evidence": {"why_it_can_fail": "if supply_chain drifted since 2026-07-14 the sealed "
                                            "numbers are not comparable and nothing below means "
                                            "anything -- gap A2 applied to this lane",
                         "sealed_H_PI": SEALED_H_PI, "recomputed_H_PI": h_pi_ref,
                         "tolerance": 2e-3}},
        "f2_the_gain_is_not_bought_by_attending_fewer": {
            "passed": bool(worst_corr >= -0.30),
            "evidence": {"why_it_can_fail": "a strongly negative correlation between the ReT gain "
                                            "and the change in orders attended would mean the "
                                            "resilience gain is bought by abandonment, and the "
                                            "lane dies whether or not the fairness guardrail "
                                            "still blocks",
                         "per_block": {k: v["corr_ret_gain_with_attended_gain"]
                                       for k, v in blocks.items()},
                         "threshold": -0.30}},
        "f3_an_uninformed_placebo_does_not_reproduce_it": {
            "passed": bool(all_plac),
            "evidence": {"why_it_can_fail": "the same action distribution without aligned "
                                            "information reproducing the gain would put the value "
                                            "in the cadence, not the signal -- exactly what was "
                                            "measured at op12",
                         "per_block": {k: v["obs_minus_placebo"] for k, v in blocks.items()}}},
        "f4_the_static_baseline_is_the_argmax": {
            "passed": bool(static_is_argmax),
            "evidence": {"why_it_can_fail": "a weaker static comparator would inflate every "
                                            "headroom reported here",
                         "n_calendars": len(cals)}},
        "f5_no_new_seeds": {
            "passed": True,
            "evidence": {"why_it_can_fail": "a seed outside the already-open blocks would consume "
                                            "custody this run never declared",
                         "blocks": {k: [v, v + args.n_tapes - 1] for k, v in BLOCKS.items()}}},
    }
    falsifiers["all_passed"] = all(v["passed"] for k, v in falsifiers.items()
                                   if k != "all_passed" and isinstance(v, dict))

    if not falsifiers["f1_the_historical_cell_still_reproduces"]["passed"]:
        verdict = "HALTED_PHYSICS_DRIFTED_SINCE_THE_HISTORICAL_CELL"
    elif all_hobs and all_plac and falsifiers["f2_the_gain_is_not_bought_by_attending_fewer"][
            "passed"]:
        verdict = "GSA_QUALIFIES_UNDER_RESILIENCE_ONLY"
    else:
        verdict = "GSA_DOES_NOT_QUALIFY_EVEN_UNDER_RESILIENCE_ONLY"

    print(f"\n  veredicto: {verdict}\n")
    for name, f in falsifiers.items():
        if name == "all_passed" or not isinstance(f, dict):
            continue
        print(f"    {name:<48} {'PASA' if f['passed'] else 'FALLA'}")

    payload = {
        "schema_version": "gsa_resilience_only_v1",
        "claim_status": verdict,
        "scope": "DEVELOPMENT_ALREADY_OPEN_BLOCKS_NO_NEW_SEEDS_NO_TRAINING_AUTHORISED",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "module_manifest": module_manifest(MODULES, script=__file__),
        "preregistration": str(args.contract),
        "pi_decision_recorded": (
            "2026-08-07: resilience is the measure. The worst-CSSU-fill guardrail that closed the "
            "historical run is distributional, not a resilience criterion, so it is reported and "
            "no longer blocking. Recorded as a decision, not as a finding."),
        "why_this_is_not_the_abandonment_trap": (
            "ret_excel on the workbook-visible population rewards abandonment because abandoned "
            "orders leave the scored population. The metric here does not: ret_order_metrics "
            "marks unattended orders lost and scores them zero (program_g.py:320), so abandonment "
            "is already priced. f2 tests this empirically rather than trusting the docstring."),
        "placebo_note": ("headroom_sensitivity.py has no uninformed placebo; it compares obs "
                         "against a static calendar only. The placebo here is another tape's "
                         "belief-policy action sequence applied to this tape."),
        "theta": THETA, "cell": cell, "n_tapes": args.n_tapes,
        "blocks": blocks, "falsifiers": falsifiers,
        "elapsed_seconds": time.perf_counter() - started,
    }
    digest = seal_and_write(payload, args.output, contract=args.contract,
                            reference=Path("results/headroom_gsa/all_cells_reconstruction.json"))
    print(f"\n  -> {args.output} (sello {digest[:16]}…)")
    return 0 if falsifiers["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
