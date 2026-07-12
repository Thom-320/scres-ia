from __future__ import annotations

import pytest

from supply_chain.supply_chain import MFSCSimulation, OrderRecord


def _queued_sim() -> MFSCSimulation:
    sim = MFSCSimulation(risks_enabled=False, seed=17)
    sim.pending_backorders = [
        OrderRecord(j=1, OPTj=0.0, quantity=3000.0, remaining_qty=3000.0),
        OrderRecord(j=2, OPTj=100.0, quantity=1000.0, remaining_qty=1000.0),
    ]
    sim.pending_backorder_qty = 4000.0
    return sim


def test_priority_setter_reorders_only_membership_and_records_hashes() -> None:
    sim = _queued_sim()
    before_rng = repr(sim.demand_rng.bit_generator.state)
    event = sim.set_backorder_priority_rule("fifo_flat")
    assert [order.j for order in sim.pending_backorders] == [1, 2]
    assert event["queue_membership_unchanged"] is True
    assert event["previous_rule"] == "spt_contingent"
    assert event["new_rule"] == "fifo_flat"
    assert sim.pending_backorder_qty == 4000.0
    assert repr(sim.demand_rng.bit_generator.state) == before_rng


def test_priority_setter_rejects_unknown_rule() -> None:
    with pytest.raises(ValueError, match="Invalid backorder_priority_rule"):
        _queued_sim().set_backorder_priority_rule("clairvoyant")


def test_step_priority_action_does_not_enter_mutable_physical_params() -> None:
    sim = MFSCSimulation(risks_enabled=False, seed=19, horizon=1.0)
    original = dict(sim.params)
    sim.step({"backorder_priority_rule": "age_threshold"}, step_hours=0.5)
    assert sim.backorder_priority_rule == "age_threshold"
    assert sim.params == original
    assert len(sim.backorder_priority_rule_events) == 1


def test_default_construction_has_no_dynamic_priority_event() -> None:
    sim = MFSCSimulation(risks_enabled=False, seed=23)
    assert sim.backorder_priority_rule == "spt_contingent"
    assert sim.backorder_priority_rule_events == []
