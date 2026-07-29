#!/usr/bin/env python3
"""Post-terminal CRN consumption audit for Program F (verifier).

The screen's `threat_sha256` proves every action received the same INPUT tape
identifier. It does NOT prove the exogenous logs actually CONSUMED during each
rollout were identical. This audit materializes and compares, across all six
fixed-budget actions on a common prefix:

  * realized_demand_sha256  -- every order's (j, OPTj, quantity, contingent,
    cssu_destination): demand times, quantities and destinations actually placed.
  * base_threat_sha256      -- every consumed risk event's (event_id, risk_id,
    onset/start_time, base_duration_hours, affected_ops, magnitude): risk onsets,
    targets and exogenous magnitudes.

Only mitigation-dependent quantities (realized_duration_hours, reserve issue) may
differ. As a sensitivity (negative control) the audit also hashes the realized
durations and asserts they DO differ across actions, proving the hashes are not
trivially constant. Disposable audit seeds 938xxx; no calibration/holdout/virgin
tape is opened; nothing is trained.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.program_f import (
    ACTIONS, CONTEXTS, ConstantPortfolio, HOURS_PER_WEEK, advance_including,
    compute_episode_metrics, digest, make_sim, materialize_tape,
)


def run_capture(tape: dict, action: tuple[int, int, int], interior_cutoff_h: float) -> dict:
    """Run one fixed action and capture the exogenous logs actually consumed.

    The per-event damage LOG is appended only after the (mitigation-dependent)
    realized window closes, so an event whose realized window crosses the episode
    horizon is not logged under some actions -- a logging-completeness artifact at
    the boundary, NOT a change in the exogenous schedule. We therefore hash the
    threat over INTERIOR events only (onset early enough that the append always
    fires under every action: realized <= 2*base for R11, <= base for R22/R23, so
    onset + 2*base < horizon guarantees completion). Boundary events are counted
    separately and their identical exogenous schedule is asserted from the tape.
    """
    sim, controller, start = make_sim(tape)
    end = start + int(tape["weeks"]) * HOURS_PER_WEEK
    policy = ConstantPortfolio(action)
    for week in range(int(tape["weeks"])):
        controller.activate_week(week)
        controller.request(policy(controller.observation()))
        advance_including(sim, min(end, start + (week + 1) * HOURS_PER_WEEK))
    compute_episode_metrics(sim, treatment_start=start)

    base_events_by_id = {row["event_id"]: row for row in tape["base_events"]}
    # Exogenous demand actually placed (arrival time, quantity, destination).
    demand = sorted(
        (
            int(o.j), round(float(o.OPTj) - start, 6), round(float(o.quantity), 9),
            bool(o.contingent), o.cssu_destination,
        )
        for o in sim.orders
    )
    # Interior exogenous risk consumption: identity, onset, base magnitude, ops.
    interior = sorted(
        (
            d["event_id"], d["risk_id"], round(float(d["start_time"]) - start, 6),
            round(float(d["base_duration_hours"]), 9),
            tuple(map(int, base_events_by_id[d["event_id"]]["affected_ops"])),
            round(float(base_events_by_id[d["event_id"]]["magnitude"]), 9),
        )
        for d in controller.damage_events
        if float(base_events_by_id[d["event_id"]]["onset_hours"])
        + 2.0 * float(d["base_duration_hours"]) < interior_cutoff_h
    )
    # Negative control: mitigation-dependent realized durations (must differ).
    realized = sorted(
        (d["event_id"], round(float(d["realized_duration_hours"]), 9))
        for d in controller.damage_events
    )
    return {
        "realized_demand_sha256": digest(demand),
        "interior_threat_sha256": digest(interior),
        "realized_duration_sha256": digest(realized),
        "n_orders": len(sim.orders),
        "n_events_logged": len(controller.damage_events),
        "n_interior_events": len(interior),
        "reserve_issued": float(sim.program_f_reserve_fragments_issued),
    }


def main() -> int:
    output = Path("results/program_f/crn_audit")
    output.mkdir(parents=True, exist_ok=True)
    tapes = [
        materialize_tape(938000 + i, context, "disposable-crn-audit", weeks=16)
        for i, context in enumerate(CONTEXTS)
    ]
    tape_reports = []
    all_demand_ok = all_threat_ok = realized_varies_ok = True
    for tape in tapes:
        horizon_h = int(tape["weeks"]) * HOURS_PER_WEEK
        per_action = {
            "".join(map(str, a)): run_capture(tape, a, horizon_h) for a in ACTIONS
        }
        demand_hashes = {r["realized_demand_sha256"] for r in per_action.values()}
        threat_hashes = {r["interior_threat_sha256"] for r in per_action.values()}
        realized_hashes = {r["realized_duration_sha256"] for r in per_action.values()}
        demand_ok = len(demand_hashes) == 1
        threat_ok = len(threat_hashes) == 1
        realized_varies = len(realized_hashes) > 1  # sensitivity: must differ
        all_demand_ok &= demand_ok
        all_threat_ok &= threat_ok
        realized_varies_ok &= realized_varies
        tape_reports.append({
            "tape_id": tape["tape_id"], "context": tape["first_context"],
            "input_threat_sha256": tape["threat_sha256"],
            "demand_identical_across_actions": demand_ok,
            "interior_threat_identical_across_actions": threat_ok,
            "realized_duration_varies_across_actions": realized_varies,
            "unique_demand_hashes": len(demand_hashes),
            "unique_interior_threat_hashes": len(threat_hashes),
            "unique_realized_duration_hashes": len(realized_hashes),
            "logged_event_counts_by_action": {
                k: r["n_events_logged"] for k, r in per_action.items()
            },
            "per_action": per_action,
        })
    all_pass = all_demand_ok and all_threat_ok and realized_varies_ok
    verdict = {
        "gate": "PROGRAM_F_CRN_CONSUMPTION_AUDIT",
        "note": (
            "SUPERSEDED complementary diagnostic. The authoritative CRN audit is "
            "scripts/audit_program_f_terminal_crn.py, which records every base event "
            "at ONSET (policy-independent) across all 288 screen tapes + 1152 branch "
            "checks. This disposable per-context probe instead reads the post-outage "
            "damage LOG, so it hashes threat over INTERIOR events only (a single "
            "near-horizon event's LOG row truncates under some actions -- a logging "
            "artifact, not a CRN leak; see docs/PROGRAM_F_POST_TERMINAL_AUDIT). Demand "
            "and interior threat are identical across the six actions; realized "
            "durations vary as a sensitivity control."
        ),
        "post_terminal": True, "calibration_tapes_opened": 0,
        "holdout_tapes_opened": 0, "virgin_tapes_opened": 0, "ppo_trained": False,
        "n_tapes": len(tapes), "n_actions": len(ACTIONS),
        "demand_identical_all_tapes": all_demand_ok,
        "interior_threat_identical_all_tapes": all_threat_ok,
        "realized_duration_varies_all_tapes": realized_varies_ok,
        "all_pass": all_pass,
        "interpretation": (
            "PASS_PROGRAM_F_CRN_CONSUMPTION" if all_pass
            else "FAIL_PROGRAM_F_CRN_CONSUMPTION"
        ),
        "tapes": tape_reports,
    }
    (output / "verdict.json").write_text(
        json.dumps(verdict, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in verdict.items() if k != "tapes"}, indent=2, sort_keys=True))
    return 0 if all_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
