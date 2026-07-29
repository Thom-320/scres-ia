#!/usr/bin/env python3
"""Count an exact tape-specific effect quotient for the M/T/R calendar family.

This script does not estimate H_PI.  It proves and measures an exact reduction
in the number of DES executions needed for the full calendar frontier and
per-tape oracle.  The reduction is valid only for the frozen
``paper2_bottleneck_migration_v1`` implementation.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.paper2_bottleneck import (  # noqa: E402
    ACTIONS,
    CONTEXTS,
    materialize_tape,
    run_policy,
)


ROOT = Path(__file__).resolve().parent.parent
WEEKS = 24
SPLITS = {
    "calibration": (1_100_001, 60),
    "locked": (1_110_001, 120),
}
ACTION_INDEX = {action: index for index, action in enumerate(ACTIONS)}
OUTCOME_KEYS = (
    "ret_excel",
    "ration_ret_excel",
    "ret_excel_cvar05",
    "ret_excel_cvar10",
    "service_loss_auc_ration_hours",
    "n_lost",
    "lost_orders",
    "backorder_qty_final",
    "backlog_age_max",
    "mass_residual",
    "reserve_units_issued",
    "consumed_base_threat_sha256",
    "realized_demand_sha256",
)


def event_masks(tape: dict[str, Any]) -> list[tuple[bool, bool, bool]]:
    masks = [[False, False, False] for _ in range(WEEKS)]
    for event in tape["base_events"]:
        week = min(WEEKS - 1, int(float(event["onset_hours"]) // 168))
        risk_id = str(event["risk_id"])
        if risk_id == "R11":
            masks[week][0] = True
        elif risk_id in {"R22", "R23"}:
            masks[week][1] = True
        elif risk_id == "R24":
            masks[week][2] = True
    return [tuple(mask) for mask in masks]


def effect_codes(tape: dict[str, Any]) -> list[tuple[int, int, int]]:
    """Return the packed 3-bit effect label for M, T and R in each week."""
    return [
        (1 if m else 0, 2 if t else 0, 4 if r else 0)
        for m, t, r in event_masks(tape)
    ]


def count_effect_words(tape: dict[str, Any]) -> int:
    """Subset-determinize the no-adjacent-switch action automaton."""
    codes = effect_codes(tape)
    # NFA state is (last active action index, switched at this week). A switch
    # in consecutive weeks is forbidden. The DFA state is the subset of NFA
    # states reachable by one physical-effect word. Counts record how many
    # distinct effect words reach each subset; no calendar tuples are stored.
    counts: dict[frozenset[tuple[int, bool]], int] = {
        frozenset({(0, False)}): 1,
    }
    for week in range(1, WEEKS):
        next_counts: defaultdict[frozenset[tuple[int, bool]], int] = defaultdict(int)
        for subset, n_words in counts.items():
            grouped: dict[int, set[tuple[int, bool]]] = {}
            for last, switched_previous_week in subset:
                choices = (last,) if switched_previous_week else (0, 1, 2)
                for action in choices:
                    label = codes[week][action]
                    grouped.setdefault(label, set()).add((action, action != last))
            for destination in grouped.values():
                next_counts[frozenset(destination)] += n_words
        counts = dict(next_counts)
    return sum(counts.values())


def full_calendar_count() -> int:
    return sum(
        math.comb(WEEKS - switches, switches) * 2**switches
        for switches in range(WEEKS // 2 + 1)
    )


def week_from_observation(observation: dict[str, float]) -> int:
    return int(round(float(observation["week_phase"]) * (WEEKS - 1)))


def active_calendar_policy(sequence: tuple[tuple[int, int, int], ...]):
    if len(sequence) != WEEKS or sequence[0] != ACTIONS[0]:
        raise ValueError("Active calendar must contain 24 weeks and start at M")

    def policy(observation: dict[str, float]):
        week = week_from_observation(observation)
        return sequence[min(week + 1, WEEKS - 1)]

    return policy


def digest_selected(row: dict[str, Any]) -> str:
    payload = {key: row[key] for key in OUTCOME_KEYS}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def validate_known_collision() -> dict[str, Any]:
    tape = materialize_tape(1_110_001, CONTEXTS[0], "locked", weeks=WEEKS)
    # Weeks 1-2 contain only R11 events. T and R both leave R11 unmitigated;
    # they are physically equivalent on this tape although their posture logs
    # differ. Switches occur at weeks 1 and 3, so the dwell rule is respected.
    t_sequence = (ACTIONS[0], ACTIONS[1], ACTIONS[1]) + (ACTIONS[0],) * 21
    r_sequence = (ACTIONS[0], ACTIONS[2], ACTIONS[2]) + (ACTIONS[0],) * 21
    t_row = run_policy(tape, active_calendar_policy(t_sequence))
    r_row = run_policy(tape, active_calendar_policy(r_sequence))
    t_digest = digest_selected(t_row)
    r_digest = digest_selected(r_row)
    return {
        "seed": 1_110_001,
        "differing_active_weeks": [1, 2],
        "week_1_2_event_masks": [list(mask) for mask in event_masks(tape)[1:3]],
        "calendar_a": "MTT" + "M" * 21,
        "calendar_b": "MRR" + "M" * 21,
        "selected_outcome_keys": list(OUTCOME_KEYS),
        "calendar_a_digest": t_digest,
        "calendar_b_digest": r_digest,
        "selected_outcomes_exactly_equal": t_digest == r_digest,
        "allocation_destination_hours_differ": any(
            t_row[key] != r_row[key]
            for key in ("token_hours_m", "token_hours_t", "token_hours_r")
        ),
        "scope_warning": "Validates the implementation quotient for selected terminal outcomes on one burned tape; it is not itself an H_PI result.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "paper2_bottleneck" / "effect_quotient_audit.json",
    )
    parser.add_argument(
        "--split",
        choices=("all", *SPLITS),
        default="all",
        help="Count one frozen split or both.",
    )
    args = parser.parse_args()

    selected_splits = SPLITS if args.split == "all" else {args.split: SPLITS[args.split]}
    split_results: dict[str, Any] = {}
    total_distinct = 0
    total_tapes = 0
    for split, (start, count) in selected_splits.items():
        per_tape = []
        for offset in range(count):
            seed = start + offset
            tape = materialize_tape(seed, CONTEXTS[offset % len(CONTEXTS)], split, weeks=WEEKS)
            distinct = count_effect_words(tape)
            per_tape.append({
                "seed": seed,
                "tape_sha256": tape["threat_sha256"],
                "distinct_effect_words": distinct,
            })
        split_total = sum(row["distinct_effect_words"] for row in per_tape)
        split_results[split] = {
            "seed_start": start,
            "n_tapes": count,
            "distinct_effect_executions": split_total,
            "per_tape": per_tape,
        }
        total_distinct += split_total
        total_tapes += count

    corrective_path = ROOT / "results" / "paper2_bottleneck" / "corrective_completeness_audit.json"
    corrective = json.loads(corrective_path.read_text())
    seconds_per_run = float(corrective["calendar_frontier"]["measured_seconds_per_run"])
    calendars = full_calendar_count()
    brute_runs = calendars * total_tapes
    result = {
        "schema_version": "paper2_bottleneck_effect_quotient_v1",
        "scientific_status": "EXACT_ACCELERATION_DESIGN_NOT_H_PI_RESULT",
        "contract_id": "paper2_bottleneck_migration_v1",
        "weeks": WEEKS,
        "full_calendar_count": calendars,
        "proof": {
            "effect_word": "e_tau,w(a)=(I[R11_w and a=M], I[R22_or_R23_w and a=T], I[R24_w and a=R])",
            "implementation_scope": "BottleneckController has no other action-dependent transition: condition is action-independent; reserve target is fixed; R11 reads M, R22/R23 read T, and R24 reads R.",
            "equivalence": "For a fixed tape, feasible calendars with the same effect word have identical DES outcomes and reserve use; allocation-destination logs may differ.",
            "dwell_rule": "No active-action switches in adjacent weeks; week 0 is M.",
        },
        "splits": split_results,
        "totals": {
            "n_tapes": total_tapes,
            "brute_force_des_runs": brute_runs,
            "effect_quotient_des_runs": total_distinct,
            "run_reduction_factor": brute_runs / total_distinct,
            "estimated_serial_cpu_days": total_distinct * seconds_per_run / 86400.0,
            "measured_seconds_per_full_run": seconds_per_run,
        },
        "collision_validation": validate_known_collision(),
        "execution_protocol": [
            "Enumerate one representative calendar for every tape-specific effect word.",
            "Run the canonical DES once per representative and store all endpoints, guardrails, reserve ledgers and hashes.",
            "Enumerate all 11,184,811 calendars using table lookups across the 60 calibration tapes to obtain the exact open-loop frontier.",
            "Take the per-tape maximum across each locked tape table for the resource-restricted PI result.",
            "Apply the frozen reserve-resource rule before optimization and rerun selected winners through run_policy as an integrity check.",
        ],
        "not_done": [
            "No effect table DES executions have been run.",
            "No open-loop frontier or PI ceiling has been computed.",
            "Reserve issue/replenishment semantics and Garrido validation of one fungible cross-stage team remain unresolved.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["totals"], indent=2, sort_keys=True))
    print(json.dumps(result["collision_validation"], indent=2, sort_keys=True))
    return 0 if result["collision_validation"]["selected_outcomes_exactly_equal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
