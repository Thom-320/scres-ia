from __future__ import annotations

import pytest

from supply_chain.supply_chain import MFSCSimulation, OrderRecord, RiskEvent


def _completed_order(
    *,
    j: int = 1,
    opt: float = 0.0,
    oat: float = 100.0,
    causal_blocks: list[dict[str, float | int | str]] | None = None,
) -> OrderRecord:
    order = OrderRecord(
        j=j,
        OPTj=opt,
        OATj=oat,
        CTj=oat - opt,
        quantity=2_500.0,
    )
    # Deliberately explicit interface for the causal lane.  Aggregate wait
    # hours are insufficient because they cannot identify event or interval.
    order.causal_block_intervals = list(causal_blocks or [])
    return order


def _causal_sim() -> MFSCSimulation:
    return MFSCSimulation(
        horizon=200.0,
        risks_enabled=False,
        risk_attribution_source="causal_exposure",
        ret_recovery_period_mode="elapsed",
    )


def test_unrelated_temporal_overlap_is_not_causal_attribution() -> None:
    sim = _causal_sim()
    order = _completed_order()
    sim.risk_events = [
        RiskEvent("R23", 10.0, 20.0, 10.0, [11]),
    ]

    sim._set_order_ret_indicators(order)

    assert "R23" not in order.ret_risk_indicators
    assert order.RPj == pytest.approx(0.0)


def test_direct_operation_block_records_responsible_event() -> None:
    sim = _causal_sim()
    order = _completed_order(
        causal_blocks=[
            {
                "op_id": 11,
                "start_time": 10.0,
                "end_time": 20.0,
                "reason": "operation_down",
            }
        ]
    )
    sim.risk_events = [
        RiskEvent("R23", 10.0, 20.0, 10.0, [11]),
    ]

    sim._set_order_ret_indicators(order)

    assert order.ret_risk_indicators["R23"] == pytest.approx(10.0)
    assert order.RPj == pytest.approx(90.0)


def test_block_on_different_operation_does_not_inherit_event() -> None:
    sim = _causal_sim()
    order = _completed_order(
        causal_blocks=[
            {
                "op_id": 11,
                "start_time": 10.0,
                "end_time": 20.0,
                "reason": "operation_down",
            }
        ]
    )
    sim.risk_events = [
        RiskEvent("R22", 10.0, 20.0, 10.0, [10]),
    ]

    sim._set_order_ret_indicators(order)

    assert "R22" not in order.ret_risk_indicators


def test_r24_exposure_end_recomputes_when_backlog_history_advances() -> None:
    sim = _causal_sim()
    event = RiskEvent(
        "R24",
        10.0,
        10.0,
        0.0,
        [13],
        magnitude=2_500.0,
        unit="rations",
    )
    sim._queue_len_history = [(0.0, 0), (10.0, 1)]
    sim.env.run(until=20.0)

    assert sim._exposure_end_for(event) == pytest.approx(20.0)

    # The episode later resolves at t=30. A value cached while it was open may
    # not truncate the episode permanently.
    sim._queue_len_history.append((30.0, 0))
    sim.env.run(until=40.0)
    assert sim._exposure_end_for(event) == pytest.approx(30.0)


def test_causal_attribution_mode_does_not_change_physical_trajectory() -> None:
    common = dict(
        horizon=3_000.0,
        seed=21,
        risks_enabled=True,
        enabled_risks={"R22", "R23", "R24"},
        risk_occurrence_mode="thesis_window",
        seed_stream_mode="split",
        risk_rng_mode="per_risk",
        procurement_contract_mode="causal_coupled",
        order_fulfillment_mode="op9_linked",
        demand_start_after_warmup=True,
    )
    overlap = MFSCSimulation(
        **common,
        risk_attribution_source="des_events",
    ).run()
    causal = MFSCSimulation(
        **common,
        risk_attribution_source="causal_exposure",
    ).run()

    overlap_physics = [
        (order.j, order.OPTj, order.OATj, order.CTj, order.lost)
        for order in overlap.orders
    ]
    causal_physics = [
        (order.j, order.OPTj, order.OATj, order.CTj, order.lost)
        for order in causal.orders
    ]
    assert causal_physics == overlap_physics
    assert causal.flow_ledger() == overlap.flow_ledger()


def test_r24_contingent_priority_propagates_only_to_orders_behind_it() -> None:
    sim = _causal_sim()
    event = RiskEvent("R24", 10.0, 10.0, 0.0, [13], magnitude=2_500.0)
    sim._r24_causal_episodes["R24-000001"] = {
        "episode_id": "R24-000001",
        "event": event,
        "surge_qty": 2_500.0,
        "assigned_order_j": 2,
        "closed_at": None,
    }
    regular = OrderRecord(
        j=1, OPTj=10.0, quantity=2_600.0, remaining_qty=2_600.0
    )
    contingent = OrderRecord(
        j=2,
        OPTj=11.0,
        quantity=5_000.0,
        remaining_qty=5_000.0,
        contingent=True,
        causal_r24_event_ids={"R24-000001"},
    )

    sim._enqueue_backorder(regular)
    sim._enqueue_backorder(contingent)

    assert sim.pending_backorders[0] is contingent
    assert "R24-000001" in regular.causal_r24_event_ids


def test_r24_episode_closes_after_tagged_queue_wave_resolves() -> None:
    sim = _causal_sim()
    event = RiskEvent("R24", 10.0, 10.0, 0.0, [13], magnitude=2_500.0)
    episode_id = "R24-000001"
    sim._r24_causal_episodes[episode_id] = {
        "episode_id": episode_id,
        "event": event,
        "surge_qty": 2_500.0,
        "assigned_order_j": 1,
        "closed_at": None,
    }
    contingent = OrderRecord(
        j=1,
        OPTj=10.0,
        quantity=5_000.0,
        remaining_qty=5_000.0,
        contingent=True,
        causal_r24_event_ids={episode_id},
    )
    sim.orders.append(contingent)
    sim._enqueue_backorder(contingent)
    sim.env.run(until=20.0)
    contingent.OATj = 20.0
    contingent.CTj = 10.0
    contingent.remaining_qty = 0.0

    sim._remove_pending_backorder(contingent)

    assert sim._r24_causal_episodes[episode_id]["closed_at"] == pytest.approx(
        20.0
    )
    later = OrderRecord(
        j=2, OPTj=21.0, quantity=2_500.0, remaining_qty=2_500.0
    )
    sim._enqueue_backorder(later)
    assert not later.causal_r24_event_ids
