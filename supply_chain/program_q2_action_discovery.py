"""Burned-only action-right diagnostics for a prospective Program Q2."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from supply_chain.program_o_full_des_transducer import FullDESSkeleton, simulate_full_des_frontier


SEQUENCES = tuple(
    tuple("P_C" if (action >> (2 - position)) & 1 else "P_H" for position in range(3))
    for action in range(8)
)
CENTERED_ACTIONS = (0, 2, 5, 7)
COUNT4_TO_SEQUENCE8 = {0: 0, 1: 2, 2: 5, 3: 7}


def sequence8_scheduler() -> dict[str, tuple[str, ...]]:
    return {str(index): labels for index, labels in enumerate(SEQUENCES)}


def count4_calendar_to_sequence8(calendar: Sequence[int]) -> tuple[int, ...]:
    values = tuple(int(value) for value in calendar)
    if any(value not in COUNT4_TO_SEQUENCE8 for value in values):
        raise ValueError("count4 calendar contains an invalid action")
    return tuple(COUNT4_TO_SEQUENCE8[value] for value in values)


def pulse_branching(
    *, skeleton: FullDESSkeleton, baseline_sequence8: Sequence[int]
) -> dict[str, Any]:
    baseline = tuple(map(int, baseline_sequence8))
    if len(baseline) != int(skeleton.decision_weeks):
        raise ValueError("baseline length differs from skeleton")
    calendars: list[tuple[int, ...]] = []
    identities: list[tuple[int, int]] = []
    for week in range(int(skeleton.decision_weeks)):
        for action in range(8):
            candidate = list(baseline)
            candidate[week] = action
            calendars.append(tuple(candidate))
            identities.append((week, action))
    matrix = simulate_full_des_frontier(
        skeleton=skeleton,
        scheduler=sequence8_scheduler(),
        calendars=np.asarray(calendars, dtype=np.uint8),
    )
    rows = {
        identity: {key: float(values[index]) for key, values in matrix.items()}
        for index, identity in enumerate(identities)
    }
    states = []
    for week in range(int(skeleton.decision_weeks)):
        best8 = max(range(8), key=lambda action: (rows[(week, action)]["ret_visible"], -action))
        best4 = max(
            CENTERED_ACTIONS,
            key=lambda action: (rows[(week, action)]["ret_visible"], -action),
        )
        states.append(
            {
                "week": week,
                "best8_action": best8,
                "best4_action": best4,
                "best8_sequence": list(SEQUENCES[best8]),
                "best4_sequence": list(SEQUENCES[best4]),
                "best8_ret": rows[(week, best8)]["ret_visible"],
                "best4_ret": rows[(week, best4)]["ret_visible"],
                "ret_gain": rows[(week, best8)]["ret_visible"] - rows[(week, best4)]["ret_visible"],
                "omitted_action_optimal": best8 not in CENTERED_ACTIONS,
                "action_ret": [rows[(week, action)]["ret_visible"] for action in range(8)],
            }
        )
    return {"states": states, "rows": rows}


def greedy_full_rollout(
    *,
    skeleton: FullDESSkeleton,
    baseline_sequence8: Sequence[int],
    allowed_actions: Sequence[int],
) -> tuple[tuple[int, ...], dict[str, float]]:
    calendar = list(map(int, baseline_sequence8))
    scheduler = sequence8_scheduler()
    for week in range(int(skeleton.decision_weeks)):
        candidates = []
        for action in allowed_actions:
            row = list(calendar)
            row[week] = int(action)
            candidates.append(row)
        matrix = simulate_full_des_frontier(
            skeleton=skeleton,
            scheduler=scheduler,
            calendars=np.asarray(candidates, dtype=np.uint8),
        )
        best = max(
            range(len(candidates)),
            key=lambda index: (float(matrix["ret_visible"][index]), -int(allowed_actions[index])),
        )
        calendar = candidates[best]
    final = simulate_full_des_frontier(
        skeleton=skeleton,
        scheduler=scheduler,
        calendars=np.asarray([calendar], dtype=np.uint8),
    )
    return tuple(calendar), {key: float(values[0]) for key, values in final.items()}


def perfect_information_timing_identity() -> dict[str, Any]:
    """Certify the feasible-set identity behind the timing-oracle correction."""
    weekly = {
        tuple(int(label == "P_C") for label in action) for action in SEQUENCES
    }
    # A weekly sequence8 action chooses every binary slot in that week. Across
    # eight weeks its Cartesian product is exactly all 24-bit calendars.
    return {
        "weekly_sequence_action_count": 8,
        "weeks": 8,
        "batch_binary_decisions": 24,
        "weekly_clairvoyant_feasible_calendars": 8**8,
        "batch_clairvoyant_feasible_calendars": 2**24,
        "counts_equal": 8**8 == 2**24,
        "one_week_sequences_unique": len(set(SEQUENCES)) == 8,
        "representative_flattened_sequence_count": len(weekly),
        "conclusion": "Perfect-information timing headroom is zero by feasible-set identity; observable between-batch information is required to define timing value.",
    }
