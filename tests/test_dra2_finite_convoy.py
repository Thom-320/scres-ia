import pytest

from supply_chain.dra2_convoy import ConvoyThresholdPolicy, static_policies
from supply_chain.supply_chain import MFSCSimulation


def make_sim() -> MFSCSimulation:
    return MFSCSimulation(
        seed=909,
        horizon=10_000,
        risks_enabled=False,
        op8_dispatch_mode="finite_convoy_v1",
    )


def seed_staging(sim: MFSCSimulation, qty: float) -> None:
    sim.rations_al.put(qty)
    sim.op8_staging_first_ready_at = float(sim.env.now)


def test_dispatch_consumes_one_convoy_and_returns_after_48_hours():
    sim = make_sim()
    seed_staging(sim, 3_000)
    event = sim.apply_op8_convoy_action("DISPATCH_NOW")
    assert event["departed"]
    assert event["quantity"] == 3_000
    assert not sim.op8_convoy_available

    sim.env.run(until=24.001)
    assert sim.rations_sb.level == pytest.approx(3_000)
    assert not sim.op8_convoy_available
    sim.env.run(until=48.001)
    assert sim.op8_convoy_available
    assert sim.op8_convoy_departures == 1
    assert sim.op8_convoy_metrics()["op8_convoy_load_factor"] == pytest.approx(0.6)


def test_dispatch_and_hold_change_future_feasibility_and_resource_state():
    """G-A: the action must change tomorrow, not merely today's flow label."""
    dispatch = make_sim()
    hold = make_sim()
    seed_staging(dispatch, 3_000)
    seed_staging(hold, 3_000)

    dispatch.apply_op8_convoy_action("DISPATCH_NOW")
    hold.apply_op8_convoy_action("HOLD")

    dispatch.env.run(until=24.001)
    hold.env.run(until=24.001)
    assert not dispatch.op8_convoy_dispatch_feasible()
    assert hold.op8_convoy_dispatch_feasible()
    assert dispatch.op8_convoy_departures > hold.op8_convoy_departures

    dispatch.env.run(until=48.001)
    hold.env.run(until=48.001)
    assert dispatch.op8_convoy_available
    assert hold.op8_convoy_available
    # The slot reconverges after its contracted cycle, but the resource and
    # inventory histories do not: HOLD preserved the staged load.
    assert dispatch.rations_al.level == pytest.approx(0.0)
    assert hold.rations_al.level == pytest.approx(3_000.0)
    assert dispatch.op8_convoy_departures == 1
    assert hold.op8_convoy_departures == 0


def test_nominal_and_actual_return_match_when_route_is_up():
    """Regression for the reported 54 h artifact: no hidden 6 h delay."""
    sim = make_sim()
    seed_staging(sim, 5_000)
    departed_at = float(sim.env.now)
    sim.apply_op8_convoy_action("DISPATCH_NOW")
    nominal = float(sim.op8_convoy_nominal_return_at)

    sim.env.run(until=departed_at + 48.001)

    assert nominal == pytest.approx(departed_at + 48.0)
    assert sim.op8_convoy_actual_return_at == pytest.approx(nominal)
    assert sim.op8_convoy_route_wait_hours == pytest.approx(0.0)


def test_partial_departure_loses_unused_lift_and_cannot_overlap():
    sim = make_sim()
    seed_staging(sim, 2_500)
    sim.apply_op8_convoy_action(1)
    second = sim.apply_op8_convoy_action(1)
    assert not second["departed"]
    assert second["mask_reason"] == "convoy_away"
    assert sim.op8_convoy_capacity_committed == 5_000
    assert sim.op8_convoy_dispatched_rations == 2_500


def test_hold_preserves_staging_and_vehicle():
    sim = make_sim()
    seed_staging(sim, 2_000)
    event = sim.apply_op8_convoy_action("HOLD")
    sim.env.run(until=24.0)
    assert not event["departed"]
    assert sim.rations_al.level == pytest.approx(2_000)
    assert sim.op8_convoy_available
    assert sim.op8_convoy_departures == 0


def test_route_outage_pauses_an_in_flight_convoy():
    sim = make_sim()
    seed_staging(sim, 5_000)
    sim.apply_op8_convoy_action("DISPATCH_NOW")
    sim.env.run(until=5.0)
    sim._take_down(8)
    sim.env.run(until=30.0)
    assert sim.rations_sb.level == 0
    sim._bring_up(8)
    sim.env.run(until=50.001)
    assert sim.rations_sb.level == pytest.approx(5_000)
    assert sim.op8_convoy_route_wait_hours == pytest.approx(25.0)


def test_convoy_mass_and_resource_conservation():
    sim = make_sim()
    seed_staging(sim, 4_000)
    sim.apply_op8_convoy_action("DISPATCH_NOW")
    sim.env.run(until=12.0)
    assert sim.rations_al.level + sim._in_transit + sim.rations_sb.level == pytest.approx(4_000)
    assert sim.op8_convoy_metrics()["op8_convoy_resource_residual"] == 0.0
    sim.env.run(until=60.0)
    assert sim.rations_al.level + sim._in_transit + sim.rations_sb.level == pytest.approx(4_000)
    assert sim.op8_convoy_metrics()["op8_convoy_resource_residual"] == 0.0


def test_static_policy_contract_and_wait_trigger():
    assert len(static_policies()) == 9
    policy = ConvoyThresholdPolicy(5_000, 48)
    obs = {
        "convoy_available": 1.0,
        "op8_route_up": 1.0,
        "op7_staged_inventory": 2_500.0,
        "staging_age": 47.0,
    }
    assert policy.action(obs) == "HOLD"
    obs["staging_age"] = 48.0
    assert policy.action(obs) == "DISPATCH_NOW"


def test_observation_contains_no_privileged_future_fields():
    sim = make_sim()
    obs = sim.get_op8_convoy_observation()
    forbidden = ("future", "regime", "repair_duration", "next_risk")
    assert not any(token in key for key in obs for token in forbidden)


def test_historical_mode_is_default_and_rejects_convoy_actions():
    sim = MFSCSimulation(seed=1)
    assert sim.op8_dispatch_mode == "thesis_full_batch"
    with pytest.raises(RuntimeError):
        sim.apply_op8_convoy_action("DISPATCH_NOW")


def test_explicit_historical_mode_preserves_default_trajectory():
    kwargs = {"seed": 919, "horizon": 5_000, "risks_enabled": False}
    default = MFSCSimulation(**kwargs)
    explicit = MFSCSimulation(**kwargs, op8_dispatch_mode="thesis_full_batch")
    default.run(); explicit.run()
    assert explicit.total_produced == default.total_produced
    assert explicit.total_delivered == default.total_delivered
    assert explicit.total_demanded == default.total_demanded
    assert [order.CTj for order in explicit.orders] == [order.CTj for order in default.orders]
