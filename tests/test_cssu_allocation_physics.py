import pytest

from supply_chain.cssu_allocation import (
    ALLOCATION_LEVELS,
    allocate_shared_capacity,
    stable_cssu_destination,
)
from supply_chain.supply_chain import MFSCSimulation


def test_destination_assignment_is_deterministic_and_rng_free():
    first = [stable_cssu_destination(simulation_seed=101, order_id=j) for j in range(100)]
    second = [stable_cssu_destination(simulation_seed=101, order_id=j) for j in range(100)]
    assert first == second
    assert set(first) == {"A", "B"}


@pytest.mark.parametrize("alpha", ALLOCATION_LEVELS)
def test_allocation_never_creates_stock_or_capacity(alpha):
    result = allocate_shared_capacity(
        stock=2_200,
        daily_capacity=2_500,
        allocation_a=alpha,
        requested={"A": 5_000, "B": 5_000},
    )
    assert result.total_dispatched == pytest.approx(2_200)
    assert result.total_dispatched <= result.available
    assert result.dispatched_a == pytest.approx(2_200 * alpha)
    assert result.dispatched_b == pytest.approx(2_200 * (1 - alpha))


def test_unused_share_can_be_reallocated_without_enlarging_pool():
    result = allocate_shared_capacity(
        stock=10_000,
        daily_capacity=2_500,
        allocation_a=0.75,
        requested={"A": 100, "B": 5_000},
    )
    assert result.dispatched_a == pytest.approx(100)
    assert result.dispatched_b == pytest.approx(2_400)
    assert result.unused == pytest.approx(0)


def test_invalid_action_and_negative_physics_are_rejected():
    with pytest.raises(ValueError):
        allocate_shared_capacity(
            stock=1, daily_capacity=1, allocation_a=0.6, requested={"A": 1, "B": 1}
        )
    with pytest.raises(ValueError):
        allocate_shared_capacity(
            stock=-1, daily_capacity=1, allocation_a=0.5, requested={"A": 1, "B": 1}
        )


def test_explicit_aggregate_mode_is_bitwise_identical_to_default():
    kwargs = {
        "seed": 123,
        "horizon": 5_000,
        "initial_buffers": {"op9_rations": 10_000},
        "order_fulfillment_mode": "op9_linked",
        "op9_dispatch_policy": "fixed_clock_daily",
        "downstream_transport_capacity_mode": "parallel",
    }
    default = MFSCSimulation(**kwargs)
    aggregate = MFSCSimulation(**kwargs, cssu_topology_mode="aggregate")
    default.run()
    aggregate.run()

    assert aggregate.total_demanded == default.total_demanded
    assert aggregate.total_delivered == default.total_delivered
    assert aggregate._in_transit == default._in_transit
    assert [order.CTj for order in aggregate.orders] == [
        order.CTj for order in default.orders
    ]


def test_split_episode_conserves_destination_mass():
    sim = MFSCSimulation(
        seed=123,
        horizon=5_000,
        initial_buffers={"op9_rations": 10_000},
        order_fulfillment_mode="op9_linked",
        op9_dispatch_policy="fixed_clock_daily",
        downstream_transport_capacity_mode="parallel",
        cssu_topology_mode="split_v1",
        cssu_service_rule="FIFO_PARTIAL",
    )
    sim.run()

    assert sum(sim.cssu_demanded.values()) == sim.total_demanded
    assert sum(sim.cssu_delivered.values()) == sim.total_delivered
    assert all(order.cssu_destination in {"A", "B"} for order in sim.orders)
    for cssu in ("A", "B"):
        unresolved = sum(
            order.remaining_qty
            for order in sim.orders
            if order.cssu_destination == cssu
        )
        assert sim.cssu_demanded[cssu] == pytest.approx(
            sim.cssu_delivered[cssu]
            + sim.cssu_in_transit[cssu]
            + unresolved
        )
        assert sim.cssu_dispatched[cssu] == pytest.approx(
            sim.cssu_delivered[cssu] + sim.cssu_in_transit[cssu]
        )
        assert sim.cssu_in_transit[cssu] == pytest.approx(
            sim.cssu_inbound_in_transit[cssu]
            + sim.cssu_inventory[cssu]
            + sim.cssu_outbound_in_transit[cssu]
        )


def test_joint_scarcity_makes_allocation_actuator_live():
    kwargs = {
        "seed": 321,
        "horizon": 2_000,
        "initial_buffers": {"op9_rations": 1_000_000},
        "order_fulfillment_mode": "op9_linked",
        "op9_dispatch_policy": "fixed_clock_daily",
        "downstream_transport_capacity_mode": "parallel",
        "cssu_topology_mode": "split_v1",
        "cssu_service_rule": "FIFO_PARTIAL",
        "demand_mean_multiplier": 2.0,
    }
    low_a = MFSCSimulation(**kwargs, cssu_allocation_a=0.25)
    high_a = MFSCSimulation(**kwargs, cssu_allocation_a=0.75)
    low_a.run()
    high_a.run()
    assert low_a.total_demanded == high_a.total_demanded
    assert low_a.cssu_allocation_live_epochs > 0
    assert high_a.cssu_allocation_live_epochs > 0
    assert high_a.cssu_dispatched["A"] > low_a.cssu_dispatched["A"]
    assert high_a.cssu_dispatched["B"] < low_a.cssu_dispatched["B"]


def test_dynamic_action_obeys_one_day_activation_latency():
    sim = MFSCSimulation(cssu_topology_mode="split_v1")
    sim.set_cssu_allocation_action(0.75, "FIFO_PARTIAL")
    sim._activate_due_cssu_action()
    assert sim.cssu_allocation_a == 0.50
    sim.env.run(until=24.0)
    sim._activate_due_cssu_action()
    assert sim.cssu_allocation_a == 0.75
    assert sim.cssu_service_rule == "FIFO_PARTIAL"


def test_localized_cssu_outage_does_not_take_down_other_destination():
    sim = MFSCSimulation(cssu_topology_mode="split_v1")
    sim._take_down_cssu(11, "A")
    assert sim._is_cssu_path_down(11, "A")
    assert not sim._is_cssu_path_down(11, "B")
    assert not sim._is_down(11)
    sim._bring_up_cssu(11, "A")
    assert not sim._is_cssu_path_down(11, "A")


def test_r23_generator_records_one_local_target_and_restores_it():
    sim = MFSCSimulation(seed=77, cssu_topology_mode="split_v1")
    sim.env.process(sim._risk_R23_event(beta=1.0))
    sim.env.run()
    event = sim.risk_events[-1]
    assert event.risk_id == "R23"
    assert event.affected_ops == [11]
    assert event.affected_cssu in {"A", "B"}
    assert not sim._is_cssu_path_down(11, event.affected_cssu)
    other = "B" if event.affected_cssu == "A" else "A"
    assert not sim._is_cssu_path_down(11, other)


def test_cssu_observation_contains_current_state_but_no_future_truth():
    sim = MFSCSimulation(seed=88, cssu_topology_mode="split_v1")
    obs = sim.get_cssu_observation()
    assert obs["cssu_A_op11_up"] == 1.0
    assert obs["cssu_B_op11_up"] == 1.0
    forbidden_fragments = ("future", "next_risk", "repair_duration", "regime")
    assert not any(
        fragment in key
        for key in obs
        for fragment in forbidden_fragments
    )
