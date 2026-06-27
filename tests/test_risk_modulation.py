"""Tests for Garrido-authorized risk modulation (frequency phi, impact psi) and the
strategic-buffer rebuild lead time. Defaults (1.0, 1.0, 0.0) reproduce the thesis."""

from __future__ import annotations

from supply_chain.supply_chain import MFSCSimulation


def _sim(**kw):
    base = dict(shifts=2, seed=7, horizon=16_128, risks_enabled=True, risk_level="current",
               risk_occurrence_mode="thesis_window", warmup_trigger="op9_arrival")
    base.update(kw)
    return MFSCSimulation(**base)


def test_defaults_are_thesis_baseline():
    s = _sim()
    assert s.risk_frequency_multiplier == 1.0
    assert s.risk_impact_multiplier == 1.0
    assert s.inventory_replenishment_lead_time == 0.0


def test_frequency_multiplier_increases_events():
    base = _sim(risk_frequency_multiplier=1.0)
    base.run()
    more = _sim(risk_frequency_multiplier=3.0)
    more.run()
    assert len(more.risk_events) > len(base.risk_events)


def test_impact_multiplier_lengthens_recovery():
    s = _sim()
    base = s._get_risk_recovery_mean("R21")
    s2 = _sim(risk_impact_multiplier=2.5)
    assert s2._get_risk_recovery_mean("R21") == base * 2.5
    # bigger surge too
    lo0, hi0 = _sim()._get_risk_surge()
    lo2, hi2 = _sim(risk_impact_multiplier=2.0)._get_risk_surge()
    assert hi2 >= hi0


def test_frequency_shortens_uniform_window():
    s1 = _sim(risk_frequency_multiplier=1.0)
    s3 = _sim(risk_frequency_multiplier=3.0)
    assert s3._get_risk_b("R11") < s1._get_risk_b("R11")


def test_frequency_multiplier_does_not_change_black_swan_window():
    s1 = _sim(risk_frequency_multiplier=1.0)
    s3 = _sim(risk_frequency_multiplier=3.0)

    assert s3._get_risk_b("R3") == s1._get_risk_b("R3")
