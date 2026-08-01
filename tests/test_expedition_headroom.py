import pytest

from supply_chain.supply_chain import MFSCSimulation


def test_expedition_charges_budget_and_shortens_only_next_leg():
    sim = MFSCSimulation(
        seed=101,
        expedite_budget_hours=24.0,
        expedite_reduction_hours=12.0,
        expedite_charge_hours=24.0,
    )

    armed = sim.arm_expedition("op8")
    assert armed["status"] == "armed"
    assert sim.expedite_budget_remaining == pytest.approx(0.0)

    assert sim._pt("op8_pt") == pytest.approx(12.0)
    assert sim._pt("op8_pt") == pytest.approx(24.0)
    applied = [event for event in sim.expedite_events if event["status"] == "applied"]
    assert len(applied) == 1
    assert applied[0]["leg"] == "op8"


def test_expedition_budget_rejection_is_explicit_and_inert():
    sim = MFSCSimulation(seed=102, expedite_budget_hours=0.0)

    rejected = sim.arm_expedition("op12")

    assert rejected["status"] == "rejected_budget"
    assert sim.expedite_budget_remaining == pytest.approx(0.0)
    assert sim._pt("op12_pt") == pytest.approx(24.0)


def test_expedition_queues_duplicate_pending_leg_and_rejects_unknown_leg():
    sim = MFSCSimulation(seed=103, expedite_budget_hours=48.0)
    sim.arm_expedition("op10")
    queued = sim.arm_expedition("op10")
    assert queued["status"] == "armed"
    assert queued["queue_depth_after_arm"] == 2
    assert sim._pt("op10_pt") == pytest.approx(12.0)
    assert sim._pt("op10_pt") == pytest.approx(12.0)
    with pytest.raises(ValueError, match="leg must be one of"):
        sim.arm_expedition("op9")


def test_expedition_zero_budget_does_not_change_baseline_processing_time():
    baseline = MFSCSimulation(seed=104)
    disabled = MFSCSimulation(seed=104, expedite_budget_hours=0.0)

    assert disabled._pt("op8_pt") == pytest.approx(baseline._pt("op8_pt"))
    assert disabled._pt("op10_pt") == pytest.approx(baseline._pt("op10_pt"))
    assert disabled._pt("op12_pt") == pytest.approx(baseline._pt("op12_pt"))
