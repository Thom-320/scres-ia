from __future__ import annotations

import pytest

from supply_chain.supply_chain import MFSCSimulation


def _run(*, event: dict | None = None, horizon: float = 2_500.0) -> MFSCSimulation:
    tape = [] if event is None else [event]
    sim = MFSCSimulation(
        seed=375,
        horizon=horizon,
        risks_enabled=True,
        enabled_risks=set(),
        risk_event_tape=tape,
        risk_occurrence_mode="thesis_window",
        seed_stream_mode="split",
        raw_material_order_up_to_multiplier=1.0,
        demand_on_hand_fulfillment_delay=0.0,
        procurement_contract_mode="causal_coupled",
    )
    return sim.run()


def _event(risk_id: str, start: float, duration: float, ops: list[int]) -> dict:
    return {
        "risk_id": risk_id,
        "start_time": start,
        "end_time": start + duration,
        "duration": duration,
        "affected_ops": ops,
    }


def test_op1_contract_is_live_and_r12_delays_supplier_material() -> None:
    baseline = _run(horizon=1_800.0)
    disrupted = _run(event=_event("R12", 100.0, 700.0, [1]), horizon=1_800.0)

    assert baseline.contract_completion_events[0][0] == pytest.approx(672.0)
    assert baseline.supplier_delivery_events[0][0] == pytest.approx(696.0)
    assert disrupted.contract_completion_events[0][0] > baseline.contract_completion_events[0][0]
    assert disrupted.supplier_delivery_events[0][0] > baseline.supplier_delivery_events[0][0]
    assert disrupted.supplier_delivery_events != baseline.supplier_delivery_events


def test_r13_delays_physical_supplier_delivery() -> None:
    baseline = _run(horizon=1_200.0)
    disrupted = _run(event=_event("R13", 680.0, 200.0, [2]), horizon=1_200.0)

    assert disrupted.supplier_delivery_events[0][0] > baseline.supplier_delivery_events[0][0]
    assert disrupted.raw_material_wdc.level != pytest.approx(baseline.raw_material_wdc.level)


@pytest.mark.parametrize(
    ("risk_id", "start", "duration", "ops", "metric"),
    [
        ("R11", 900.0, 240.0, [5], "total_produced"),
        ("R21", 850.0, 400.0, [3, 5, 6, 7, 9], "total_produced"),
        ("R22", 1_000.0, 400.0, [10], "total_theatre_inflow"),
        ("R23", 1_000.0, 400.0, [11], "total_theatre_inflow"),
        ("R3", 850.0, 672.0, [5, 6, 7, 9], "total_produced"),
    ],
)
def test_forced_duration_risk_changes_an_intended_physical_kpi(
    risk_id: str,
    start: float,
    duration: float,
    ops: list[int],
    metric: str,
) -> None:
    baseline = _run(horizon=2_500.0)
    disrupted = _run(event=_event(risk_id, start, duration, ops), horizon=2_500.0)

    terminal_kpi_changed = getattr(disrupted, metric) != pytest.approx(
        getattr(baseline, metric)
    )
    delivery_timing_changed = disrupted.delivery_events != baseline.delivery_events
    assert terminal_kpi_changed or delivery_timing_changed
    event = next(item for item in disrupted.risk_events if item.risk_id == risk_id)
    assert event.affected_ops == ops


def test_r24_changes_demand_but_not_unrelated_production_capacity() -> None:
    baseline = _run(horizon=2_500.0)
    surge = {
        **_event("R24", 1_000.0, 0.0, [13]),
        "magnitude": 2_500.0,
        "unit": "rations",
    }
    disrupted = _run(event=surge, horizon=2_500.0)

    assert disrupted.total_demanded == pytest.approx(baseline.total_demanded + 2_500.0)
    assert disrupted.total_produced == pytest.approx(baseline.total_produced)


def test_r14_defects_change_good_output_and_preserve_ration_mass() -> None:
    baseline = MFSCSimulation(
        seed=375,
        horizon=2_500.0,
        risks_enabled=False,
        seed_stream_mode="split",
        raw_material_order_up_to_multiplier=1.0,
        demand_on_hand_fulfillment_delay=0.0,
        procurement_contract_mode="causal_coupled",
    ).run()
    disrupted = MFSCSimulation(
        seed=375,
        horizon=2_500.0,
        risks_enabled=True,
        enabled_risks={"R14"},
        risk_frequency_multipliers_by_id={"R14": 20.0},
        seed_stream_mode="split",
        raw_material_order_up_to_multiplier=1.0,
        demand_on_hand_fulfillment_delay=0.0,
        procurement_contract_mode="causal_coupled",
    ).run()

    assert any(event.risk_id == "R14" for event in disrupted.risk_events)
    assert disrupted.total_produced < baseline.total_produced
    assert disrupted.flow_ledger()["ration_residual"] == pytest.approx(0.0, abs=1e-6)


def test_default_thesis_strict_flow_ledger_conserves_both_material_classes() -> None:
    sim = _run(horizon=2_500.0)
    ledger = sim.flow_ledger()

    assert ledger["raw_residual"] == pytest.approx(0.0, abs=1e-6)
    assert ledger["ration_residual"] == pytest.approx(0.0, abs=1e-6)
    assert sim.total_delivered == pytest.approx(sim.total_theatre_inflow)
    assert sim.total_order_fulfilled <= sim.total_demanded


def test_op9_linked_orders_get_ct_from_transport_not_fixed_delay() -> None:
    sim = MFSCSimulation(
        seed=375,
        horizon=2_500.0,
        risks_enabled=False,
        seed_stream_mode="split",
        raw_material_order_up_to_multiplier=1.0,
        demand_on_hand_fulfillment_delay=999.0,
        procurement_contract_mode="causal_coupled",
        order_fulfillment_mode="op9_linked",
        demand_start_after_warmup=True,
    ).run()
    completed = [order for order in sim.orders if order.CTj is not None]

    assert completed
    assert completed[0].CTj == pytest.approx(48.0)
    assert completed[0].CTj != pytest.approx(999.0)
    assert sim.flow_ledger()["ration_residual"] == pytest.approx(0.0, abs=1e-6)


def test_r22_delays_an_op9_linked_order_beyond_the_48h_promise() -> None:
    event = _event("R22", 1_000.0, 240.0, [10])
    sim = MFSCSimulation(
        seed=375,
        horizon=2_500.0,
        risks_enabled=True,
        risk_event_tape=[event],
        seed_stream_mode="split",
        raw_material_order_up_to_multiplier=1.0,
        procurement_contract_mode="causal_coupled",
        order_fulfillment_mode="op9_linked",
        demand_start_after_warmup=True,
    ).run()

    assert any(order.CTj and order.CTj > order.LTj for order in sim.orders)
    assert sim.total_backorders > 0
