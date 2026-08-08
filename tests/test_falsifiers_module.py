"""Mutation tests: every guard must be shown to FAIL, or it is not a guard.

Each test here reintroduces the exact defect that shipped on 2026-08-08 and asserts the module
catches it. A guard that has never been seen to fail is a comment.
"""
from __future__ import annotations

import numpy as np
import pytest

from supply_chain.falsifiers import (
    FalsifierConstructionError, check, disclosure, ge, gt, lt, not_applicable, preflight,
    summarise,
)


def test_a_verdict_without_operands_is_rejected():
    """The defect: `passed: True` hardcoded and then counted in 'N falsifiers pass'.

    Identity cannot catch it -- Python interns True -- so the guard is that the operands must be
    handed over. A literal cannot satisfy that without fabricating numbers.
    """
    with pytest.raises(FalsifierConstructionError):
        check(True, "cannot fail", computed_from={})
    with pytest.raises(FalsifierConstructionError):
        check(True, "cannot fail", computed_from={"note": "no numeric operand"})


def test_a_falsifier_must_say_why_it_can_fail():
    with pytest.raises(FalsifierConstructionError):
        check(1 > 0, "   ", computed_from={"x": 1.0})


def test_comparison_helpers_record_both_operands():
    out = gt(3.0, 1.0, "why")
    assert out["passed"] is True
    assert out["evidence"]["computed_from"] == {"observed": 3.0, "threshold": 1.0}
    assert lt(3.0, 1.0, "why")["passed"] is False


def test_summarise_excludes_disclosures_and_not_applicable():
    """The defect: 'nine falsifiers pass' counted four literals and one not-applicable."""
    f = {"a": gt(1.0, 0.0, "why"),
         "b": disclosure("a fact that cannot fail"),
         "c": not_applicable("declared replay")}
    s = summarise(f)
    assert s["n_computed"] == 1 and s["n_disclosures"] == 1 and s["n_not_applicable"] == 1
    assert s["all_passed"] is True


def test_summarise_fails_when_a_computed_check_fails():
    f = {"a": gt(1.0, 0.0, "why"), "b": gt(1.0, 2.0, "why")}
    s = summarise(f)
    assert s["all_passed"] is False and s["failed"] == ["b"]


def test_summarise_is_not_vacuously_true_with_no_computed_checks():
    assert summarise({"d": disclosure("only a disclosure")})["all_passed"] is False


def _pf(values, reset_now=100.0, horizon=4368.0, scenario=None, expected=None):
    scenario = scenario if scenario is not None else {"demand": "garrido_seasonal_v1"}
    expected = expected if expected is not None else {"demand": "garrido_seasonal_v1"}
    return preflight(probe=lambda o: o, options=values, reset_now=reset_now,
                     horizon=horizon, scenario=scenario, expected_scenario=expected)


def test_preflight_catches_a_dead_action():
    """The defect: the posture grid never reached S1, so all 25 options were identical."""
    out = _pf([0.5, 0.5, 0.5])
    assert out["p1_endpoint_responds_to_the_action"]["passed"] is False
    assert _pf([0.1, 0.5, 0.9])["p1_endpoint_responds_to_the_action"]["passed"] is True


def test_preflight_catches_a_one_bit_decision_space():
    """The defect: eleven schedules tied, so the benchmark compared choosers on a single bit."""
    out = _pf([0.2, 0.2, 0.2, 0.9])          # two distinct levels only
    assert out["p2_decision_space_has_more_than_one_effective_level"]["passed"] is False
    assert _pf([0.1, 0.2, 0.3])[
        "p2_decision_space_has_more_than_one_effective_level"]["passed"] is True


def test_preflight_catches_a_reset_that_consumed_the_horizon():
    """The defect: R13 at x16 left env.now at 161,280 h against a 4,368 h horizon."""
    out = _pf([0.1, 0.2, 0.3], reset_now=161_280.0)
    assert out["p3_reset_leaves_time_inside_the_horizon"]["passed"] is False


def test_preflight_catches_a_silent_scenario_fallback():
    """The defect: the sensitivity reverted to thesis_uniform while naming the seasonal scenario."""
    out = _pf([0.1, 0.2, 0.3], scenario={"demand": "thesis_uniform"})
    assert out["p4_scenario_is_the_declared_one"]["passed"] is False
