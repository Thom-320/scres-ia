from __future__ import annotations

import pytest

from scripts.audit_garrido_event_delayed_quantity import (
    _sim_kwargs,
    delayed_quantity_metrics,
)
from supply_chain.supply_chain import MFSCSimulation


def test_delayed_quantity_curve_measures_peak_area_and_recovery() -> None:
    factual = [(10.0, 5.0), (30.0, 5.0)]
    no_event = [(10.0, 10.0)]

    result = delayed_quantity_metrics(factual, no_event, horizon=40.0)

    assert result["delayed_quantity_peak"] == pytest.approx(5.0)
    assert result["delay_onset"] == pytest.approx(10.0)
    assert result["first_temporary_recovery_at"] == pytest.approx(30.0)
    assert result["debt_recovered_at"] == pytest.approx(30.0)
    assert result["delayed_unit_hours"] == pytest.approx(100.0)
    assert result["terminal_delayed_quantity"] == pytest.approx(0.0)


def test_unrecovered_quantity_is_reported_at_horizon() -> None:
    result = delayed_quantity_metrics([], [(4.0, 7.0)], horizon=20.0)

    assert result["delayed_quantity_peak"] == pytest.approx(7.0)
    assert result["debt_recovered_at"] is None
    assert result["terminal_delayed_quantity"] == pytest.approx(7.0)
    assert result["delayed_unit_hours"] == pytest.approx(112.0)


def test_availability_recording_is_observational_only() -> None:
    sim = MFSCSimulation(risks_enabled=False)
    before = float(sim.raw_material_wdc.level)

    sim._record_material_availability("raw_material_wdc", 12.0)

    assert float(sim.raw_material_wdc.level) == before
    assert sim.material_availability_events["raw_material_wdc"] == [(0.0, 12.0)]


def test_leave_one_event_out_detects_r13_supplier_availability_debt() -> None:
    kwargs = _sim_kwargs(cfi=1, seed=375, horizon=1_000.0)
    event = {
        "risk_id": "R13",
        "start_time": 672.0,
        "end_time": 816.0,
        "duration": 144.0,
        "affected_ops": [2],
        "magnitude": 6.0,
        "unit": "delayed_deliveries",
    }
    factual = MFSCSimulation(**kwargs, risk_event_tape=[event]).run()
    no_event = MFSCSimulation(**kwargs, risk_event_tape=[]).run()

    result = delayed_quantity_metrics(
        factual.material_availability_events["raw_material_wdc"],
        no_event.material_availability_events["raw_material_wdc"],
        horizon=1_000.0,
    )

    assert result["delayed_quantity_peak"] > 0.0
    assert result["delay_onset"] == pytest.approx(696.0)
