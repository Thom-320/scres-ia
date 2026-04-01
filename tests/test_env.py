from __future__ import annotations

import numpy as np
import pytest

from supply_chain.config import HOURS_PER_YEAR_GREGORIAN, HOURS_PER_YEAR_THESIS
from supply_chain.env import MFSCGymEnv
from supply_chain.supply_chain import MFSCSimulation, OrderRecord


def test_env_reset_returns_obs_info() -> None:
    env = MFSCGymEnv(step_size_hours=168, max_steps=3)
    obs, info = env.reset(seed=42)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (15,)
    assert obs.dtype == np.float32
    assert isinstance(info, dict)
    assert "time" in info


def test_env_step_returns_five_values() -> None:
    env = MFSCGymEnv(step_size_hours=168, max_steps=2)
    env.reset(seed=42)
    action = np.zeros(4, dtype=np.float32)
    result = env.step(action)
    assert len(result) == 5
    obs, reward, terminated, truncated, info = result
    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_env_action_bounds_and_clipping() -> None:
    env = MFSCGymEnv(step_size_hours=24, max_steps=2)
    env.reset(seed=42)
    raw_action = np.array([2.5, -5.0, 0.4, 1.2], dtype=np.float32)
    _, _, _, _, info = env.step(raw_action)
    clipped = np.asarray(info["clipped_action"], dtype=np.float32)
    assert np.all(clipped <= 1.0)
    assert np.all(clipped >= -1.0)


def test_env_observation_shape_dtype() -> None:
    env = MFSCGymEnv(step_size_hours=24, max_steps=3)
    obs, _ = env.reset(seed=42)
    assert obs.shape == (15,)
    assert obs.dtype == np.float32
    for _ in range(2):
        obs, _, _, _, _ = env.step(np.zeros(4, dtype=np.float32))
        assert obs.shape == (15,)
        assert obs.dtype == np.float32


def test_seed_reproducibility_fixed_seed() -> None:
    env_a = MFSCGymEnv(step_size_hours=24, max_steps=2)
    env_b = MFSCGymEnv(step_size_hours=24, max_steps=2)
    obs_a, _ = env_a.reset(seed=123)
    obs_b, _ = env_b.reset(seed=123)
    assert np.allclose(obs_a, obs_b)

    action = np.array([0.1, -0.2, 0.0, 0.8], dtype=np.float32)
    out_a = env_a.step(action)
    out_b = env_b.step(action)
    assert np.allclose(out_a[0], out_b[0])
    assert out_a[1] == out_b[1]
    assert out_a[2] == out_b[2]
    assert out_a[3] == out_b[3]


def test_year_basis_thesis_vs_gregorian_outputs() -> None:
    horizon = HOURS_PER_YEAR_THESIS * 2
    sim_thesis = MFSCSimulation(
        shifts=1,
        risks_enabled=False,
        seed=42,
        horizon=horizon,
        year_basis="thesis",
    ).run()
    sim_gregorian = MFSCSimulation(
        shifts=1,
        risks_enabled=False,
        seed=42,
        horizon=horizon,
        year_basis="gregorian",
    ).run()
    th = sim_thesis.get_annual_throughput()
    gr = sim_gregorian.get_annual_throughput()
    assert th["hours_per_year"] == HOURS_PER_YEAR_THESIS
    assert gr["hours_per_year"] == HOURS_PER_YEAR_GREGORIAN
    assert th["year_basis"] == "thesis"
    assert gr["year_basis"] == "gregorian"


def test_deterministic_baseline_is_seed_invariant() -> None:
    horizon = HOURS_PER_YEAR_THESIS * 2
    sim_a = MFSCSimulation(
        shifts=1,
        risks_enabled=False,
        seed=42,
        horizon=horizon,
        year_basis="thesis",
        deterministic_baseline=True,
    ).run()
    sim_b = MFSCSimulation(
        shifts=1,
        risks_enabled=False,
        seed=7,
        horizon=horizon,
        year_basis="thesis",
        deterministic_baseline=True,
    ).run()
    out_a = sim_a.get_annual_throughput(start_time=sim_a.warmup_time)
    out_b = sim_b.get_annual_throughput(start_time=sim_b.warmup_time)
    assert out_a["produced_by_year"] == out_b["produced_by_year"]
    assert out_a["avg_annual_delivery"] == out_b["avg_annual_delivery"]


def test_post_warmup_yearly_production_removes_first_year_startup_dip() -> None:
    horizon = HOURS_PER_YEAR_THESIS * 3
    sim = MFSCSimulation(
        shifts=1,
        risks_enabled=False,
        seed=42,
        horizon=horizon,
        year_basis="thesis",
        deterministic_baseline=True,
    ).run()
    raw = sim.get_annual_throughput()
    aligned = sim.get_annual_throughput(start_time=sim.warmup_time, num_years=2)
    assert raw["produced_by_year"][1] < aligned["produced_by_year"][1]


def test_step_info_contains_rt_metrics() -> None:
    sim = MFSCSimulation(
        shifts=1,
        risks_enabled=True,
        seed=42,
        horizon=24 * 30,
        year_basis="thesis",
    )
    _, _, _, info = sim.step(step_hours=24)
    assert "new_demanded" in info
    assert "step_disruption_hours" in info
    assert "total_inventory" in info
    assert "inventory_detail" in info


def test_backorder_queue_enforces_sixty_order_cap_and_unattended_count() -> None:
    sim = MFSCSimulation(seed=42, deterministic_baseline=True)

    for order_idx in range(61):
        order = OrderRecord(
            j=order_idx + 1,
            OPTj=0.0,
            quantity=100.0 + order_idx,
            remaining_qty=100.0 + order_idx,
            backorder=True,
        )
        sim.orders.append(order)
        sim._enqueue_backorder(order)

    assert len(sim.pending_backorders) == 60
    assert sim.total_unattended_orders == 1
    assert sim.pending_backorders[-1].quantity == pytest.approx(159.0)
    assert any(order.lost for order in sim.orders)


def test_backorder_queue_prioritizes_contingent_then_shortest_orders() -> None:
    sim = MFSCSimulation(seed=42, deterministic_baseline=True)

    large_regular = OrderRecord(
        j=1, OPTj=0.0, quantity=2000.0, remaining_qty=2000.0, backorder=True
    )
    small_regular = OrderRecord(
        j=2, OPTj=0.0, quantity=1000.0, remaining_qty=1000.0, backorder=True
    )
    contingent = OrderRecord(
        j=3,
        OPTj=0.0,
        quantity=2500.0,
        remaining_qty=2500.0,
        backorder=True,
        contingent=True,
    )

    sim._enqueue_backorder(large_regular)
    sim._enqueue_backorder(small_regular)
    sim._enqueue_backorder(contingent)

    assert [order.j for order in sim.pending_backorders] == [3, 2, 1]


def test_pending_backorders_are_served_once_theatre_inventory_arrives() -> None:
    sim = MFSCSimulation(seed=42, deterministic_baseline=True)
    delayed_order = OrderRecord(
        j=1,
        OPTj=0.0,
        quantity=1200.0,
        remaining_qty=1200.0,
        backorder=True,
    )
    sim.orders.append(delayed_order)
    sim._enqueue_backorder(delayed_order)

    def deliver_and_serve():
        yield sim.env.timeout(1.0)
        yield sim.rations_theatre.put(1500.0)
        yield from sim._serve_pending_backorders()

    sim.env.process(deliver_and_serve())
    sim.env.run()

    assert delayed_order.backorder is False
    assert delayed_order.OATj == pytest.approx(1.0)
    assert delayed_order.CTj == pytest.approx(1.0)
    assert delayed_order.remaining_qty == pytest.approx(0.0)
    assert len(sim.pending_backorders) == 0
    assert sim.pending_backorder_qty == pytest.approx(0.0)


def test_zero_qty_pending_backorders_are_dropped_without_container_get(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sim = MFSCSimulation(seed=42, deterministic_baseline=True)
    delayed_order = OrderRecord(
        j=1,
        OPTj=0.0,
        quantity=1200.0,
        remaining_qty=0.0,
        backorder=True,
    )
    sim.orders.append(delayed_order)
    sim._enqueue_backorder(delayed_order)

    def _unexpected_get(amount: float):
        raise AssertionError(f"Container.get should not be called for amount={amount}")

    monkeypatch.setattr(sim.rations_theatre, "get", _unexpected_get)

    sim.env.process(sim._serve_pending_backorders())
    sim.env.run()

    assert delayed_order.backorder is False
    assert delayed_order.remaining_qty == pytest.approx(0.0)
    assert len(sim.pending_backorders) == 0
    assert sim.pending_backorder_qty == pytest.approx(0.0)


def test_pending_backorder_removal_is_idempotent_during_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sim = MFSCSimulation(seed=42, deterministic_baseline=True)
    delayed_order = OrderRecord(
        j=1,
        OPTj=0.0,
        quantity=1200.0,
        remaining_qty=1200.0,
        backorder=True,
    )
    sim.orders.append(delayed_order)
    sim._enqueue_backorder(delayed_order)
    sim.rations_theatre._level = 1500.0

    def _fake_get(amount: float):
        assert amount == pytest.approx(1200.0)

        def _event():
            if delayed_order in sim.pending_backorders:
                sim.pending_backorders.remove(delayed_order)
                sim._refresh_pending_backorder_qty()
            yield sim.env.timeout(0.0)

        return sim.env.process(_event())

    monkeypatch.setattr(sim.rations_theatre, "get", _fake_get)

    sim.env.process(sim._serve_pending_backorders())
    sim.env.run()

    assert delayed_order.backorder is False
    assert delayed_order.remaining_qty == pytest.approx(0.0)
    assert len(sim.pending_backorders) == 0
    assert sim.pending_backorder_qty == pytest.approx(0.0)


def test_env_rt_v0_reward_mode_emits_components() -> None:
    env = MFSCGymEnv(step_size_hours=24, max_steps=2, reward_mode="rt_v0")
    env.reset(seed=42)
    _, reward, _, _, info = env.step(np.zeros(4, dtype=np.float32))
    assert isinstance(reward, float)
    assert info["reward_mode"] == "rt_v0"
    assert "rt_components" in info
