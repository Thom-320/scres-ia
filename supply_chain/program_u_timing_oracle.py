"""Exact calendar classes and paired oracle gate for endogenous review timing."""

from __future__ import annotations

from itertools import product
from typing import Mapping, Sequence

import numpy as np

from supply_chain.program_t_t0_gate import paired_bootstrap_lcb


Calendar = tuple[int, ...]


def variable_review_calendars(
    *,
    horizon: int = 8,
    review_budget: int = 4,
    dwell_options: Sequence[int] = (1, 2, 4),
    action_count: int = 4,
) -> tuple[Calendar, ...]:
    """All calendars using exactly ``review_budget`` state-review decisions.

    The last review holds its selected action through the remaining horizon,
    matching :class:`EndogenousReviewProgramORetEnv`.
    """
    if horizon < review_budget or review_budget <= 0:
        raise ValueError("horizon must allow every review right to be exercised")
    dwell = tuple(sorted(set(map(int, dwell_options))))
    calendars: set[Calendar] = set()

    def recurse(prefix: tuple[int, ...], reviews_used: int) -> None:
        remaining = horizon - len(prefix)
        rights_left = review_budget - reviews_used
        if rights_left == 1:
            for action in range(action_count):
                calendars.add(prefix + (action,) * remaining)
            return
        for selected_dwell in dwell:
            # Leave at least one week for each remaining review decision.
            if selected_dwell > remaining - (rights_left - 1):
                continue
            for action in range(action_count):
                recurse(prefix + (action,) * selected_dwell, reviews_used + 1)

    recurse((), 0)
    return tuple(sorted(calendars))


def fixed_cadence_calendars(
    *, horizon: int = 8, review_budget: int = 4, action_count: int = 4
) -> tuple[Calendar, ...]:
    if horizon % review_budget:
        raise ValueError("primary fixed-cadence comparator requires an even partition")
    dwell = horizon // review_budget
    return tuple(
        tuple(action for block_action in actions for action in (block_action,) * dwell)
        for actions in product(range(action_count), repeat=review_budget)
    )


def weekly_calendars(*, horizon: int = 8, action_count: int = 4) -> tuple[Calendar, ...]:
    return tuple(product(range(action_count), repeat=horizon))


def timing_oracle_gate(
    *,
    score_by_calendar: Mapping[Calendar, Sequence[float]],
    bootstrap_seed: int = 20260720,
) -> dict[str, object]:
    variable = variable_review_calendars()
    fixed = fixed_cadence_calendars()
    weekly = weekly_calendars()
    missing = [calendar for calendar in weekly if calendar not in score_by_calendar]
    if missing:
        raise ValueError("score map must contain the complete 4^8 answer matrix")

    def oracle(calendars: Sequence[Calendar]) -> np.ndarray:
        matrix = np.asarray([score_by_calendar[row] for row in calendars], dtype=float)
        if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
            raise ValueError("calendar scores must be a finite calendar-by-tape matrix")
        return np.max(matrix, axis=0)

    variable_value = oracle(variable)
    fixed_value = oracle(fixed)
    weekly_value = oracle(weekly)
    timing_delta = variable_value - fixed_value
    weekly_regret = variable_value - weekly_value
    lcb = paired_bootstrap_lcb(
        variable_value, fixed_value, seed=bootstrap_seed
    )
    subset_check = bool(np.all(weekly_regret <= 1e-12))
    passed = lcb >= 0.015 and subset_check
    return {
        "status": "PASS_ENDOGENOUS_REVIEW_VALUE__LEARNER_FIT_AUTHORIZED"
        if passed
        else "STOP_NO_ENDOGENOUS_REVIEW_VALUE",
        "variable_review_calendar_count": len(variable),
        "fixed_cadence_calendar_count": len(fixed),
        "weekly_calendar_count": len(weekly),
        "mean_timing_value": float(np.mean(timing_delta)),
        "timing_value_lcb95": lcb,
        "mean_regret_to_unconstrained_weekly_oracle": float(np.mean(weekly_regret)),
        "variable_class_is_bounded_by_weekly_oracle": subset_check,
        "review_rights_matched_in_primary_estimand": True,
        "new_scientific_seeds_authorized": False,
        "hybrid_confirmation_authorized": False,
    }

