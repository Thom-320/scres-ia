from __future__ import annotations

import math

import pytest

from supply_chain.garrido_v0_recovery import (
    CONTEXT_ORDER,
    RECOVERY_WINDOW_HOURS,
    context_descriptor,
    placebo_event_rows,
    recovery_utility,
    restricted_recovery_summary,
    risk_event_rows,
)


def panel(*, auc: float, drop: float, ttr: float | None) -> dict:
    return {
        "temporal_cluster_records": [
            {
                "service_loss_auc_ration_hours": auc,
                "maximum_service_drop": drop,
                "system_ttr_hours": ttr,
            }
        ]
    }


def test_censoring_maps_to_tau_not_zero() -> None:
    out = restricted_recovery_summary(
        panel(auc=100.0, drop=0.2, ttr=None),
        panel(auc=0.0, drop=0.0, ttr=24.0),
    )
    assert out["restricted_ttr_hours"] == RECOVERY_WINDOW_HOURS
    assert out["right_censored_at_tau"] is True
    assert out["recovered_within_tau"] is False


def test_absorbed_shock_has_zero_recovery_time() -> None:
    out = restricted_recovery_summary(
        panel(auc=25.0, drop=0.05, ttr=240.0),
        panel(auc=25.0, drop=0.05, ttr=240.0),
    )
    assert out["impacted"] is False
    assert out["absorbed"] is True
    assert out["restricted_ttr_hours"] == 0.0


def test_observed_recovery_is_preserved() -> None:
    out = restricted_recovery_summary(
        panel(auc=100.0, drop=0.2, ttr=216.0),
        panel(auc=10.0, drop=0.0, ttr=24.0),
    )
    assert out["restricted_ttr_hours"] == 216.0
    assert out["right_censored_at_tau"] is False


def test_recovery_utility_is_recovery_first() -> None:
    fast_bad_tie = {
        "restricted_ttr_hours": 24.0,
        "excess_service_loss_auc_ration_hours": 1e12,
    }
    slow_good_tie = {
        "restricted_ttr_hours": 48.0,
        "excess_service_loss_auc_ration_hours": 0.0,
    }
    assert recovery_utility(
        fast_bad_tie, demanded_rations=1.0, flow_fill_rate=0.0
    ) > recovery_utility(slow_good_tie, demanded_rations=1e9, flow_fill_rate=1.0)


def test_contexts_and_placebo_are_source_safe() -> None:
    assert CONTEXT_ORDER == ("R11", "R12", "R13", "R14", "R21", "R22", "R23", "R24")
    assert placebo_event_rows()[0]["affected_ops"] == []
    assert len(risk_event_rows("R11")) == 8
    assert len(risk_event_rows("R22")) == 4
    for context in CONTEXT_ORDER:
        descriptor = context_descriptor(context)
        assert descriptor.ndim == 1
        assert math.isfinite(float(descriptor.sum()))


def test_recovery_summary_requires_exactly_one_cluster() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        restricted_recovery_summary(
            {"temporal_cluster_records": []},
            panel(auc=0.0, drop=0.0, ttr=0.0),
        )
