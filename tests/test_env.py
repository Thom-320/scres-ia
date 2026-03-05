from __future__ import annotations

import numpy as np

from supply_chain.config import HOURS_PER_YEAR_GREGORIAN, HOURS_PER_YEAR_THESIS
from supply_chain.env import MFSCGymEnv
from supply_chain.supply_chain import MFSCSimulation


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


def test_env_rt_v0_reward_mode_emits_components() -> None:
    env = MFSCGymEnv(step_size_hours=24, max_steps=2, reward_mode="rt_v0")
    env.reset(seed=42)
    _, reward, _, _, info = env.step(np.zeros(4, dtype=np.float32))
    assert isinstance(reward, float)
    assert info["reward_mode"] == "rt_v0"
    assert "rt_components" in info
