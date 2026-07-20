from __future__ import annotations

from supply_chain.program_u_timing_oracle import (
    fixed_cadence_calendars,
    timing_oracle_gate,
    variable_review_calendars,
    weekly_calendars,
)


def test_timing_classes_have_correct_set_relation() -> None:
    variable = set(variable_review_calendars())
    fixed = set(fixed_cadence_calendars())
    weekly = set(weekly_calendars())
    assert len(fixed) == 4**4
    assert fixed <= variable <= weekly


def test_timing_gate_never_claims_to_beat_weekly_oracle() -> None:
    weekly = weekly_calendars()
    # Make early switching valuable so variable timing beats two-week cadence.
    score_map = {
        calendar: (float(calendar[0] == 0) + float(calendar[1] == 1),) * 20
        for calendar in weekly
    }
    result = timing_oracle_gate(score_by_calendar=score_map)
    assert result["variable_class_is_bounded_by_weekly_oracle"] is True
    assert result["mean_regret_to_unconstrained_weekly_oracle"] <= 0

