import math

import pytest

from supply_chain.program_o_hobs import (
    calendar_index,
    observable_calendar,
    posterior_after_week,
    predicted_request_share_c,
    transition_belief,
)


def weekly_events(labels):
    times = []
    products = []
    for week, week_labels in enumerate(labels):
        for day, product in enumerate(week_labels):
            times.append(30.0 + 168.0 * week + 24.0 * day)
            products.append(product)
    return times, products


def test_exact_bayes_update_matches_closed_form():
    posterior = posterior_after_week(
        0.5, ["P_C"] * 5 + ["P_H"], dominant_share=0.9
    )
    expected_odds = 9.0 ** 4
    assert posterior == pytest.approx(expected_odds / (1.0 + expected_odds))
    predicted = transition_belief(posterior, regime_persistence=0.75)
    assert predicted == pytest.approx(0.75 * posterior + 0.25 * (1-posterior))
    assert predicted_request_share_c(predicted, dominant_share=0.9) > 0.5


def test_week_action_never_reads_current_or_future_requests():
    times, products = weekly_events(
        [["P_C"] * 6, ["P_H"] * 6, ["P_C"] * 6]
    )
    calendar, decisions = observable_calendar(
        request_times=times,
        request_products=products,
        decision_start=0.0,
        decision_weeks=3,
        policy_id="belief_extreme_v1",
        initial_action=2,
    )
    assert calendar[0] == 2
    assert decisions[0].prior_request_products == ()
    assert decisions[1].prior_request_products == ("P_C",) * 6
    assert decisions[2].prior_request_products == ("P_C",) * 6 + ("P_H",) * 6
    assert all(
        time < decision.decision_time
        for decision in decisions
        for time in decision.prior_request_times
    )


def test_future_labels_cannot_change_earlier_actions_or_hashes():
    times, products = weekly_events([["P_C"] * 6, ["P_H"] * 6])
    first = observable_calendar(
        request_times=times,
        request_products=products,
        decision_start=0.0,
        decision_weeks=2,
        policy_id="belief_extreme_v1",
        initial_action=2,
    )
    changed = products[:6] + ["P_C"] * 6
    second = observable_calendar(
        request_times=times,
        request_products=changed,
        decision_start=0.0,
        decision_weeks=2,
        policy_id="belief_extreme_v1",
        initial_action=2,
    )
    assert first[0] == second[0]
    assert [row.observation_sha256 for row in first[1]] == [
        row.observation_sha256 for row in second[1]
    ]


def test_swapped_history_reverses_nontie_extreme_action():
    times, products = weekly_events([["P_C"] * 6, ["P_C"] * 6])
    original, original_rows = observable_calendar(
        request_times=times,
        request_products=products,
        decision_start=0.0,
        decision_weeks=2,
        policy_id="belief_extreme_v1",
        initial_action=2,
    )
    swapped, swapped_rows = observable_calendar(
        request_times=times,
        request_products=products,
        decision_start=0.0,
        decision_weeks=2,
        policy_id="belief_extreme_v1",
        initial_action=2,
        swap_observed_labels=True,
    )
    assert not original_rows[1].tie_state
    assert not swapped_rows[1].tie_state
    assert swapped[1] == 3 - original[1]


def test_extra_week_delay_does_not_use_immediate_prior_week():
    times, products = weekly_events([["P_C"] * 6, ["P_H"] * 6, ["P_H"] * 6])
    delayed, rows = observable_calendar(
        request_times=times,
        request_products=products,
        decision_start=0.0,
        decision_weeks=3,
        policy_id="belief_extreme_v1",
        initial_action=2,
        history_delay_weeks=1,
    )
    assert rows[1].prior_request_products == ()
    assert rows[2].prior_request_products == ("P_C",) * 6
    assert delayed[2] == 3


def test_no_history_policy_is_a_fixed_sequence():
    times, products = weekly_events([["P_C"] * 6, ["P_H"] * 6])
    calendar, _ = observable_calendar(
        request_times=times,
        request_products=products,
        decision_start=0.0,
        decision_weeks=2,
        policy_id="no_history_v1",
        initial_action=1,
        ignore_history=True,
    )
    assert calendar == (1, 1)


def test_calendar_index_is_base4_lexicographic():
    assert calendar_index((0, 0, 0)) == 0
    assert calendar_index((0, 0, 1)) == 1
    assert calendar_index((3, 3, 3)) == 4**3 - 1


def test_parameter_validation_is_fail_closed():
    with pytest.raises(ValueError):
        posterior_after_week(0.0, ["P_C"], dominant_share=0.9)
    with pytest.raises(ValueError):
        transition_belief(0.5, regime_persistence=0.49)
