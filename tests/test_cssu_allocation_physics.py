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


def test_opt_in_split_preserves_aggregate_trajectory_and_accounts_destinations():
    kwargs = {
        "seed": 123,
        "horizon": 5_000,
        "initial_buffers": {"op9_rations": 10_000},
        "order_fulfillment_mode": "op9_linked",
        "op9_dispatch_policy": "fixed_clock_daily",
        "downstream_transport_capacity_mode": "parallel",
    }
    aggregate = MFSCSimulation(**kwargs, cssu_topology_mode="aggregate")
    split = MFSCSimulation(**kwargs, cssu_topology_mode="split_v1")
    aggregate.run()
    split.run()

    # Adding destinations consumes no RNG and changes no aggregate physics.
    assert split.total_demanded == aggregate.total_demanded
    assert split.total_delivered == aggregate.total_delivered
    assert split._in_transit == aggregate._in_transit
    assert sum(split.cssu_demanded.values()) == split.total_demanded
    assert sum(split.cssu_delivered.values()) == split.total_delivered
    assert all(order.cssu_destination in {"A", "B"} for order in split.orders)
