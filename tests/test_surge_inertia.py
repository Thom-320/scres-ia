"""Tests for capacity inertia (Ed.2): activation lag + finite surge budget."""

from __future__ import annotations

from supply_chain.env_experimental_shifts import MFSCGymEnvShifts


def _env(**kw):
    base = dict(reward_mode="control_v1", observation_version="v5",
               action_mode="shift_only", risk_level="current", max_steps=10)
    base.update(kw)
    return MFSCGymEnvShifts(**base)


def test_inertia_off_by_default_allows_instant_s3():
    env = _env()
    env.reset(seed=1)
    env.step([1.0])  # request S3
    assert env.sim.params["assembly_shifts"] == 3


def test_activation_lag_ramps_one_level_per_step():
    env = _env(surge_inertia=True, surge_ramp_per_step=1, surge_budget_hours=1e9)
    env.reset(seed=1)
    env.step([1.0])  # request S3 from S1 -> ramp to S2
    assert env._effective_shift == 2
    env.step([1.0])  # -> S3
    assert env._effective_shift == 3


def test_finite_surge_budget_caps_to_s1_when_exhausted():
    # Budget for ~3 steps of S3 surge (2 levels x 168h).
    env = _env(surge_inertia=True, surge_ramp_per_step=3, surge_budget_hours=672.0)
    env.reset(seed=1)
    levels = []
    for _ in range(5):
        env.step([1.0])  # always request S3
        levels.append(env._effective_shift)
    # Eventually the budget runs out and capacity is forced to S1.
    assert levels[-1] == 1
    assert env._surge_budget_remaining == 0.0


def test_demobilises_instantly():
    env = _env(surge_inertia=True, surge_ramp_per_step=1, surge_budget_hours=1e9)
    env.reset(seed=1)
    env.step([1.0]); env.step([1.0])      # ramp up to S3
    assert env._effective_shift == 3
    env.step([-1.0])                       # request S1 -> instant drop
    assert env._effective_shift == 1
