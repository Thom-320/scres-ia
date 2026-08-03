"""Unit tests for the no-rerun G3-obs f2 audit."""

from scripts.audit_g3_obs_f2 import compare_order


def test_f2_requires_the_delayed_arm_between_real_and_placebo():
    result = compare_order({
        "threshold_windowed": 0.20,
        "threshold_delayed": 0.10,
        "uninformed_placebo": 0.00,
        "wrong_claimant": -0.20,
    })
    assert result["passed"] is True
    assert [item["strictly_greater"] for item in result["comparisons"]] == [True, True, True]


def test_f2_fails_when_delayed_beats_the_real_signal():
    result = compare_order({
        "threshold_windowed": 0.10,
        "threshold_delayed": 0.20,
        "uninformed_placebo": 0.00,
        "wrong_claimant": -0.20,
    })
    assert result["passed"] is False
    assert result["comparisons"][0]["strictly_greater"] is False
