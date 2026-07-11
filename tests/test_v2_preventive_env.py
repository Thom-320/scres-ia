from __future__ import annotations

import numpy as np

from supply_chain.l_program_env import CampaignTape
from supply_chain.supply_chain import MFSCSimulation, OrderRecord
from supply_chain.v2_preventive_env import (
    WARNING_LEAD_HOURS,
    build_warning_schedule,
)


def _materialized_tape() -> CampaignTape:
    return CampaignTape(
        campaign_id="v2-test",
        family="R2",
        risk_level="increased",
        base_seed=123,
        horizon_weeks=20,
        split="calibration",
        calendar_materialized=True,
        risk_events=(
            {
                "risk_id": "R22",
                "start_time": 800.0,
                "end_time": 920.0,
                "duration": 120.0,
                "affected_ops": [10],
                "description": "test corridor outage",
                "magnitude": 0.0,
                "unit": "hours",
            },
            {
                "risk_id": "R21",
                "start_time": 1_600.0,
                "end_time": 1_720.0,
                "duration": 120.0,
                "affected_ops": [5],
                "description": "not warning eligible",
                "magnitude": 0.0,
                "unit": "hours",
            },
        ),
    )


def test_warning_schedule_is_deterministic_and_targets_only_downstream_risks() -> None:
    tape = _materialized_tape()
    first = build_warning_schedule(tape, seed=44, mode="perfect")
    second = build_warning_schedule(tape, seed=44, mode="perfect")
    assert first.payload() == second.payload()
    assert len(first.intervals) == 1
    warning = first.intervals[0]
    assert warning.start_time == 800.0 - WARNING_LEAD_HOURS
    assert warning.event_key.startswith("R22:")


def test_shuffled_placebo_preserves_count_but_breaks_timing() -> None:
    tape = _materialized_tape()
    imperfect = build_warning_schedule(tape, seed=99, mode="imperfect")
    placebo = build_warning_schedule(tape, seed=99, mode="shuffled_placebo")
    assert len(placebo.intervals) == len(imperfect.intervals)
    if imperfect.intervals:
        assert [row.start_time for row in placebo.intervals] != [
            row.start_time for row in imperfect.intervals
        ]


def test_emergency_reserve_serves_only_while_corridor_is_down() -> None:
    sim = MFSCSimulation(
        risks_enabled=False,
        order_fulfillment_mode="op9_linked",
        seed=7,
        horizon=100.0,
    )
    sim.configure_emergency_theatre_reserve(
        capacity=5_000.0,
        initial_stock=5_000.0,
        target=5_000.0,
        issue_delay=1.0,
    )
    calm = OrderRecord(j=1, OPTj=0.0, quantity=2_500.0, remaining_qty=2_500.0)
    sim.env.process(sim._place_demand_order(calm))
    sim.env.run(until=0.1)
    assert np.isclose(sim.emergency_theatre_reserve.level, 5_000.0)
    assert calm in sim.pending_backorders

    disrupted = OrderRecord(
        j=2,
        OPTj=float(sim.env.now),
        quantity=2_500.0,
        remaining_qty=2_500.0,
    )
    sim._take_down(10)
    sim.env.process(sim._place_demand_order(disrupted))
    sim.env.run(until=0.2)
    assert np.isclose(sim.flow_ledger()["ration_residual"], 0.0)
    sim.env.run(until=2.0)
    assert np.isclose(sim.emergency_theatre_reserve.level, 2_500.0)
    assert disrupted.OATj is not None
    assert disrupted not in sim.pending_backorders


def test_replenishment_moves_existing_stock_and_conserves_mass() -> None:
    sim = MFSCSimulation(risks_enabled=False, seed=8, horizon=500.0)
    sim.rations_sb.put(5_000.0)
    sim.total_strategic_rations_injected += 5_000.0
    sim.configure_emergency_theatre_reserve(
        capacity=5_000.0,
        initial_stock=0.0,
        target=0.0,
        replenishment_lead_time=24.0,
    )
    before = sim.flow_ledger()["ration_residual"]
    sim.request_emergency_reserve_target(2_500.0)
    sim.env.run(until=25.0)
    after = sim.flow_ledger()["ration_residual"]
    assert np.isclose(sim.rations_sb.level, 2_500.0)
    assert np.isclose(sim.emergency_theatre_reserve.level, 2_500.0)
    assert np.isclose(before, 0.0)
    assert np.isclose(after, 0.0)


def test_physical_replenishment_uses_two_downstream_legs() -> None:
    sim = MFSCSimulation(risks_enabled=False, seed=9, horizon=100.0)
    sim.rations_sb.put(5_000.0)
    sim.total_strategic_rations_injected += 5_000.0
    sim.configure_emergency_theatre_reserve(
        capacity=5_000.0,
        transport_mode="physical_downstream",
    )
    sim.request_emergency_reserve_target(2_500.0)
    sim.env.run(until=47.9)
    assert np.isclose(sim.emergency_theatre_reserve.level, 0.0)
    sim.env.run(until=48.1)
    assert np.isclose(sim.emergency_theatre_reserve.level, 2_500.0)
    assert np.isclose(sim.flow_ledger()["ration_residual"], 0.0)
