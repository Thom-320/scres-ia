from __future__ import annotations

import pytest
from types import SimpleNamespace

from scripts.audit_garrido_event_delayed_quantity import (
    _sim_kwargs,
    delayed_quantity_metrics,
    match_fifo_release_opportunities,
    match_order_release_counterfactual,
)
from supply_chain.supply_chain import MFSCSimulation, OrderRecord


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


def test_order_matching_uses_same_order_identity_not_neighboring_window() -> None:
    factual_orders = [
        OrderRecord(j=1, OPTj=0.0, quantity=10.0, op9_release_time=30.0),
        OrderRecord(j=2, OPTj=1.0, quantity=20.0, op9_release_time=20.0),
    ]
    no_event_orders = [
        OrderRecord(j=1, OPTj=0.0, quantity=10.0, op9_release_time=10.0),
        OrderRecord(j=2, OPTj=1.0, quantity=20.0, op9_release_time=20.0),
    ]

    rows = match_order_release_counterfactual(
        SimpleNamespace(orders=factual_orders),
        SimpleNamespace(orders=no_event_orders),
        event_ref="R13@0",
        horizon=100.0,
    )

    assert [row["j"] for row in rows] == [1]
    assert rows[0]["release_delay_hours"] == pytest.approx(20.0)
    assert rows[0]["delayed_unit_hours"] == pytest.approx(200.0)


def test_order_matching_reports_horizon_censored_release() -> None:
    factual = OrderRecord(j=7, OPTj=0.0, quantity=5.0)
    counterfactual = OrderRecord(j=7, OPTj=0.0, quantity=5.0, op9_release_time=40.0)

    rows = match_order_release_counterfactual(
        SimpleNamespace(orders=[factual]),
        SimpleNamespace(orders=[counterfactual]),
        event_ref="R13@0",
        horizon=100.0,
    )

    assert rows[0]["status"] == "censored_not_released_factual"
    assert rows[0]["release_delay_hours"] == pytest.approx(60.0)


def test_fifo_matching_allocates_only_incremental_release_debt() -> None:
    factual_orders = [
        OrderRecord(j=1, OPTj=0.0, quantity=5.0, op9_release_time=20.0),
        OrderRecord(j=2, OPTj=1.0, quantity=5.0, op9_release_time=30.0),
    ]
    no_event_orders = [
        OrderRecord(j=1, OPTj=0.0, quantity=5.0, op9_release_time=10.0),
        OrderRecord(j=2, OPTj=1.0, quantity=5.0, op9_release_time=30.0),
    ]

    rows = match_fifo_release_opportunities(
        SimpleNamespace(orders=factual_orders),
        SimpleNamespace(orders=no_event_orders),
        event_ref="R13@0",
    )

    assert len(rows) == 1
    assert rows[0]["j"] == 1
    assert rows[0]["direct_fifo_quantity"] == pytest.approx(5.0)
    assert rows[0]["release_debt_after"] == pytest.approx(5.0)
