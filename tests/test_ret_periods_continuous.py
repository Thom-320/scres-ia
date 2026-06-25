"""Tests for the continuous AP/RP/DP resilience (faithful limits, smooth)."""

from __future__ import annotations

from types import SimpleNamespace

from supply_chain.ret_thesis import (
    compute_ret_periods_continuous_per_order as ret_c,
    compute_periods_continuous_ret,
)


def _order(*, OATj, APj=0.0, RPj=0.0, DPj=0.0, LTj=48.0):
    return SimpleNamespace(OATj=OATj, APj=APj, RPj=RPj, DPj=DPj, LTj=LTj)


def test_no_disruption_is_one():
    assert ret_c(_order(OATj=100.0)) == 1.0


def test_autotomy_absorbed_is_one():
    # delivered on time despite disruption (AP>0, RP=DP=0)
    assert ret_c(_order(OATj=100.0, APj=30.0)) == 1.0


def test_recovery_decays_with_rp():
    fast = ret_c(_order(OATj=100.0, RPj=24.0), tau_hours=48.0)
    slow = ret_c(_order(OATj=100.0, RPj=240.0), tau_hours=48.0)
    assert 0.0 < slow < fast < 1.0


def test_non_recovery_long_disruption_is_small():
    assert ret_c(_order(OATj=100.0, DPj=2000.0), tau_hours=48.0) < 0.05


def test_unfulfilled_is_zero():
    assert ret_c(_order(OATj=None, RPj=10.0)) == 0.0


def test_bounded_in_unit_interval():
    for kw in [dict(OATj=1.0), dict(OATj=1.0, APj=5, RPj=300, DPj=300),
               dict(OATj=None), dict(OATj=1.0, APj=48)]:
        v = ret_c(_order(**kw))
        assert 0.0 <= v <= 1.0


def test_aggregate_excludes_unfulfilled():
    orders = [_order(OATj=1.0), _order(OATj=None, DPj=99), _order(OATj=2.0, RPj=48.0)]
    res = compute_periods_continuous_ret(orders, tau_hours=48.0)
    assert res["n_served"] == 2
    assert 0.5 < res["mean_ret_continuous"] < 1.0
