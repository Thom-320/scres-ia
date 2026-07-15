"""Sanity tests for the time-resolved resilience metric.

Run: .venv/bin/python -m pytest research/thesis_native_timing/test_resilience_timeresolved.py -q
"""

from __future__ import annotations

from types import SimpleNamespace

from research.thesis_native_timing.resilience_timeresolved import (
    resilience_from_sim,
    DEFAULT_REF_WINDOW_HOURS,
)


def _order(opt, lt, oat, qty=1000.0):
    return SimpleNamespace(OPTj=opt, LTj=lt, OATj=oat, quantity=qty, metrics_excluded=False)


def _sim(orders, horizon=100_000.0, warmup=0.0):
    return SimpleNamespace(env=SimpleNamespace(now=horizon), warmup_time=warmup, orders=orders)


def test_all_on_time_gives_triangle_one():
    # every order delivered exactly at its promise (OAT == OPT+LT) -> zero lateness -> triangle 1
    orders = [_order(opt=i * 100.0, lt=48.0, oat=i * 100.0 + 48.0) for i in range(10)]
    r = resilience_from_sim(_sim(orders))
    assert abs(r["resilience_triangle_v1"] - 1.0) < 1e-9


def test_lateness_lowers_triangle():
    on_time = [_order(0.0, 48.0, 48.0) for _ in range(10)]
    late = [_order(0.0, 48.0, 48.0 + 300.0) for _ in range(10)]  # 300h late each
    assert (
        resilience_from_sim(_sim(late))["resilience_triangle_v1"]
        < resilience_from_sim(_sim(on_time))["resilience_triangle_v1"]
    )


def test_equal_count_different_recovery_speed_differs():
    """THE key property: same #late (=> same fill-rate/ret count) but different recovery
    DURATION must give different time-resolved resilience. A count metric cannot see this."""
    ref = DEFAULT_REF_WINDOW_HOURS
    # Set A: 5 on-time + 5 late-by-100h.  Set B: 5 on-time + 5 late-by-500h.
    # Both have identical on-time COUNT (5/10) -> identical ret_excel/fill-rate.
    fast = [_order(0.0, 48.0, 48.0) for _ in range(5)] + [_order(0.0, 48.0, 48.0 + 100.0) for _ in range(5)]
    slow = [_order(0.0, 48.0, 48.0) for _ in range(5)] + [_order(0.0, 48.0, 48.0 + 500.0) for _ in range(5)]
    rf = resilience_from_sim(_sim(fast), ref_window_hours=ref)
    rs = resilience_from_sim(_sim(slow), ref_window_hours=ref)
    # same number of late orders, but slow recovery -> strictly lower triangle
    assert rf["n_recovered_late"] == rs["n_recovered_late"] == 5.0
    assert rs["resilience_triangle_v1"] < rf["resilience_triangle_v1"] - 1e-6


def test_unfulfilled_capped_not_horizon_sensitive():
    """An order never fulfilled contributes exactly one window of loss, independent of horizon."""
    never = [_order(0.0, 48.0, None)]
    a = resilience_from_sim(_sim(never, horizon=50_000.0))["resilience_triangle_v1"]
    b = resilience_from_sim(_sim(never, horizon=500_000.0))["resilience_triangle_v1"]
    assert abs(a - b) < 1e-9  # horizon-independent
    assert abs(a - 0.0) < 1e-9  # single order down the whole window -> triangle 0


def test_triangle_bounded():
    orders = [_order(0.0, 48.0, 48.0 + 10_000.0) for _ in range(3)]
    t = resilience_from_sim(_sim(orders))["resilience_triangle_v1"]
    assert 0.0 <= t <= 1.0
