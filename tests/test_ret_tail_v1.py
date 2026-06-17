"""Tests for the ReT_tail_v1 tail/recovery-aligned reward.

ReT_tail_v1 exists to fix ReT_ladder_v1's mean-tracking: its cost term is
UN-gated, so holding a large buffer / running extra shifts always costs reward.
These tests lock in (a) registration, (b) bounded finite reward, and (c) that the
cost actually bites (higher cap/inv kappa strictly lowers reward for a
high-buffer + high-shift action) — the property the reward_surface_audit relies on.
"""
from __future__ import annotations

import numpy as np

from supply_chain.env_experimental_shifts import REWARD_MODE_OPTIONS
from supply_chain.external_env_interface import make_dkana_thesis_faithful_env


def _avg_reward(*, inv_kappa: float, cap_kappa: float, action) -> float:
    env = make_dkana_thesis_faithful_env(
        reward_mode="ReT_tail_v1",
        action_space_mode="thesis_factorized",
        risk_level="increased",
        risk_occurrence_mode="thesis_periodic",
        raw_material_flow_mode="kit_equivalent_order_up_to",
        raw_material_order_up_to_multiplier=2.0,
        ret_tail_inv_kappa=inv_kappa,
        ret_tail_cap_kappa=cap_kappa,
        max_steps=20,
    )
    env.reset(seed=7)
    total = 0.0
    steps = 0
    for _ in range(15):
        _, reward, terminated, truncated, _ = env.step(np.asarray(action))
        total += float(reward)
        steps += 1
        if terminated or truncated:
            break
    return total / max(1, steps)


def test_ret_tail_v1_registered() -> None:
    assert "ReT_tail_v1" in REWARD_MODE_OPTIONS


def test_ret_tail_v1_reward_is_finite_and_bounded() -> None:
    env = make_dkana_thesis_faithful_env(
        reward_mode="ReT_tail_v1",
        action_space_mode="thesis_factorized",
        risk_level="increased",
        risk_occurrence_mode="thesis_periodic",
        raw_material_flow_mode="kit_equivalent_order_up_to",
        max_steps=20,
    )
    _, info = env.reset(seed=1)
    for _ in range(10):
        _, reward, terminated, truncated, info = env.step(np.array([3, 0]))  # I504, S1
        assert np.isfinite(reward)
        assert 0.0 < reward <= 1.0
        assert "ret_tail_step" in info
        if terminated or truncated:
            break


def test_ret_tail_v1_cost_bites_monotonically() -> None:
    """For a max-buffer + max-shift action, raising the cost kappas must strictly
    lower the reward — otherwise the un-gated cost lever does nothing."""
    high_action = np.array([5, 2])  # I1344 (max buffer) + S3 (max shift)
    low_cost = _avg_reward(inv_kappa=0.1, cap_kappa=0.1, action=high_action)
    mid_cost = _avg_reward(inv_kappa=0.5, cap_kappa=0.25, action=high_action)
    high_cost = _avg_reward(inv_kappa=1.5, cap_kappa=0.5, action=high_action)
    assert low_cost > mid_cost > high_cost
