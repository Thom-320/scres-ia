from __future__ import annotations

import pytest

from supply_chain.supply_chain import MFSCSimulation, OrderRecord, RiskEvent


def _sim(**kwargs) -> MFSCSimulation:
    return MFSCSimulation(
        risks_enabled=False,
        risk_attribution_source="causal_exposure",
        material_lineage_mode="tagged_lots",
        order_fulfillment_mode="op9_linked",
        **kwargs,
    )


def test_lineage_fifo_split_and_quantity_conservation() -> None:
    sim = _sim()
    sim._lineage_put(
        "raw_material_wdc", 12.0, risk_event_refs=("R13@1",), source_stage="op2_output"
    )
    sim._lineage_put("raw_material_wdc", 8.0, source_stage="op2_output")

    taken = sim._lineage_take("raw_material_wdc", 15.0)

    assert sum(row.quantity for row in taken) == pytest.approx(15.0)
    assert taken[0].quantity == pytest.approx(12.0)
    assert taken[0].risk_event_refs == ("R13@1",)
    assert taken[1].quantity == pytest.approx(3.0)
    assert sim._lineage_snapshot()["raw_material_wdc"] == pytest.approx(5.0)


def test_raw_to_ration_conversion_preserves_event_reference() -> None:
    sim = _sim(raw_material_flow_mode="bom_total_units_order_up_to")
    sim._lineage_put(
        "raw_material_al", 120.0, risk_event_refs=("R12@0",), source_stage="op4_output"
    )

    ration_slices = sim._lineage_take(
        "raw_material_al", 120.0, output_scale=1.0 / sim._raw_units_per_ration
    )
    sim._lineage_forward("pending_batch", ration_slices, source_stage="op7_work")

    assert sim._lineage_snapshot()["raw_material_al"] == pytest.approx(0.0)
    assert sim._lineage_snapshot()["pending_batch"] == pytest.approx(10.0)
    assert ration_slices[0].risk_event_refs == ("R12@0",)


def test_upstream_event_is_attached_to_correct_next_stage_only() -> None:
    sim = _sim()
    event = RiskEvent("R13", 10.0, 34.0, 24.0, [2])

    sim._register_upstream_scarcity_debt(event)

    ref = sim._lineage_event_ref(event)
    assert sim._pending_lineage_events_by_stage["op2_output"] == [ref]
    assert sim._pending_lineage_events_by_stage["op4_output"] == []
    assert sim._consume_pending_stage_refs("op2_output") == (ref,)
    assert sim._consume_pending_stage_refs("op2_output") == ()


def test_stockout_records_open_debt_but_unrelated_inventory_consumption_does_not() -> None:
    sim = _sim()
    event = RiskEvent("R12", 0.0, 168.0, 168.0, [1])
    sim._register_upstream_scarcity_debt(event)
    order = OrderRecord(j=1, OPTj=200.0, quantity=10.0, remaining_qty=10.0)

    sim._record_active_upstream_stockout_causes(order, duration=24.0)

    assert [row["risk_id"] for row in order.lineage_shortage_refs] == ["R12"]
    assert order.consumed_material_lineage == []
    assert order.causal_block_intervals[0]["propagation_source"] == "upstream_scarcity_debt"


def test_debt_closes_only_when_matching_tagged_lot_reaches_sb() -> None:
    sim = _sim()
    event_a = RiskEvent("R12", 0.0, 168.0, 168.0, [1])
    event_b = RiskEvent("R21", 20.0, 30.0, 10.0, [5])
    sim._register_upstream_scarcity_debt(event_a)
    sim._register_upstream_scarcity_debt(event_b)
    arrived_refs = {sim._lineage_event_ref(event_a)}

    sim._upstream_scarcity_debts = [
        event
        for event in sim._upstream_scarcity_debts
        if sim._lineage_event_ref(event) not in arrived_refs
    ]

    assert sim._upstream_scarcity_debts == [event_b]


def test_lineage_off_is_a_noop_and_cannot_change_physical_state() -> None:
    sim = MFSCSimulation(risks_enabled=False, material_lineage_mode="off")
    before = sim._lineage_snapshot()

    sim._lineage_put(
        "raw_material_wdc", 100.0, risk_event_refs=("R13@1",), source_stage="test"
    )
    assert sim._lineage_take("raw_material_wdc", 50.0) == []

    assert sim._lineage_snapshot() == before

