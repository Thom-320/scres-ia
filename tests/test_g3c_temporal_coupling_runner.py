"""Instrument tests for the G3c burned-only runner.

The tests deliberately mutate the compact evidence consumed by the falsifiers.  A check that only
compares an implementation with itself is not evidence that the runner can reject a defect.
"""
from __future__ import annotations

import numpy as np
import pytest

from scripts.run_g3c_temporal_coupling import (
    DWELL_LEVELS,
    TREATMENT_LEVELS,
    GUARDRAIL_MARGINS,
    check_myopic_trace,
    guardrail_harm,
    hysteresis_target,
    myopic_target,
    paired_bootstrap,
    required_n,
    validate_guardrail_margins,
)


def _row(*, fill=0.8, flow=0.8, lost=0.0, backorder=10.0):
    return {"metrics": {
        "worst_claimant_fill": fill,
        "flow_fill_rate": flow,
        "lost_orders": lost,
        "backorder_qty_final": backorder,
    }}


def test_myopic_rule_is_directional_and_wrong_control_is_its_inverse():
    assert myopic_target(10.0, 1.0) == 0.9
    assert myopic_target(1.0, 10.0) == 0.1
    assert myopic_target(4.0, 4.0) == 0.5
    assert myopic_target(10.0, 1.0, wrong=True) == pytest.approx(0.1)


def test_v2_grid_contains_only_the_frozen_live_levels():
    assert DWELL_LEVELS == (1, 6, 12)
    assert TREATMENT_LEVELS == (6, 12)


def test_hysteresis_has_entry_and_release_thresholds():
    state, target = hysteresis_target(0, 0.11)
    assert (state, target) == (1, 0.9)
    state, target = hysteresis_target(state, 0.05)
    assert (state, target) == (1, 0.9)
    state, target = hysteresis_target(state, 0.01)
    assert (state, target) == (0, 0.5)


def test_myopic_trace_rejects_a_future_or_direction_mutation():
    good = [{"time": 0.0, "unmet_a": 10.0, "unmet_b": 1.0, "target": 0.9}]
    bad = [{"time": 0.0, "unmet_a": 10.0, "unmet_b": 1.0, "target": 0.1}]
    assert check_myopic_trace(good)["passed"] is True
    assert check_myopic_trace(bad)["passed"] is False


def test_guardrail_margins_are_signed_and_zero_is_not_accepted_for_stochastic_outcomes():
    assert validate_guardrail_margins(GUARDRAIL_MARGINS)["passed"] is True
    assert validate_guardrail_margins({"flow_fill_rate": 0.0,
                                       "lost_orders": 0.5,
                                       "backorder_qty_final_relative": 0.01})["passed"] is False


def test_guardrail_harm_uses_the_declared_direction_and_denominator():
    reference = [_row(flow=0.8, lost=0.0, backorder=10.0)]
    candidate = [_row(flow=0.7, lost=1.0, backorder=20.0)]
    np.testing.assert_allclose(guardrail_harm(reference, candidate, "flow_fill_rate"), [0.1])
    np.testing.assert_allclose(guardrail_harm(reference, candidate, "lost_orders"), [1.0])
    np.testing.assert_allclose(
        guardrail_harm(reference, candidate, "backorder_qty_final_relative"), [1.0])


def test_power_requirement_grows_when_the_target_gets_smaller():
    assert required_n(0.1, 0.005) > required_n(0.1, 0.010)
    assert required_n(0.0, 0.010) == 1


def test_bootstrap_is_paired_and_reports_a_finite_interval():
    stat = paired_bootstrap(np.asarray([0.1, 0.2, 0.3]), 200, np.random.default_rng(7))
    assert stat["n"] == 3
    assert stat["lcb95"] <= stat["mean"] <= stat["ucb95"]
