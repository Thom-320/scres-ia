from __future__ import annotations

from types import SimpleNamespace

import pytest

from supply_chain.ret_repair import repaired_ret_mean, repaired_ret_values


def _order(
    *,
    j: int = 1,
    rpj: float,
    ctj: float = 240.0,
    ltj: float = 48.0,
    indicators: dict[str, float] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        j=j,
        OPTj=0.0,
        OATj=ctj,
        CTj=ctj,
        LTj=ltj,
        APj=0.0,
        RPj=rpj,
        quantity=2500.0,
        lost=False,
        ret_bt_at_request=0,
        ret_ut_at_request=0,
        ret_ledger_event_sequence=j,
        ret_risk_indicators=indicators or {"R23": rpj},
    )


def test_clip_is_minimal_range_projection() -> None:
    order = _order(rpj=0.01)
    assert repaired_ret_mean([order], current_time=240.0, mode="canonical") == 50.0
    assert repaired_ret_mean([order], current_time=240.0, mode="clip_0_1") == 1.0
    assert order.RPj == 0.01


def test_quantity_time_is_bounded_and_does_not_mutate_order() -> None:
    order = _order(rpj=0.01, indicators={"R23": 0.01, "R24": 2516.0})
    values = repaired_ret_values(
        [order],
        current_time=240.0,
        mode="quantity_time_clip_0_1",
    )
    assert values.tolist() == pytest.approx([0.5 / 192.0])
    assert order.RPj == 0.01


def test_quantity_time_does_not_reassign_pure_timing_risk() -> None:
    order = _order(rpj=0.25, indicators={"R23": 0.25})
    assert repaired_ret_mean(
        [order],
        current_time=240.0,
        mode="quantity_time_clip_0_1",
    ) == 1.0


def test_unknown_mode_fails_closed() -> None:
    with pytest.raises(ValueError, match="unknown ReT repair mode"):
        repaired_ret_values([], current_time=0.0, mode="invented")
