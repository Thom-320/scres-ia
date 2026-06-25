from __future__ import annotations

import numpy as np
import pytest

from supply_chain.env_experimental_shifts import (
    RET_TAIL_BETA,
    RET_TAIL_BOOST,
    RET_TAIL_CAP_KAPPA,
    RET_TAIL_GAMMA,
    RET_TAIL_INV_KAPPA,
    RET_TAIL_TRANSFORM,
    RET_TAIL_W_CE,
    RET_TAIL_W_RC,
    RET_TAIL_W_SC,
    REWARD_MODE_OPTIONS,
)
from supply_chain.thesis_decision_env import make_thesis_factorized_track_a_env


def _make_env(**overrides):
    kwargs = {
        "reward_mode": "ReT_tail_v2",
        "risk_level": "increased",
        "risk_occurrence_mode": "thesis_window",
        "raw_material_flow_mode": "kit_equivalent_order_up_to",
        "raw_material_order_up_to_multiplier": 2.0,
        "priming_enabled": False,
        "max_steps": 20,
    }
    kwargs.update(overrides)
    return make_thesis_factorized_track_a_env(**kwargs)


def _avg_reward(*, inv_kappa: float, cap_kappa: float, action) -> float:
    env = _make_env(
        ret_tail_inv_kappa=inv_kappa,
        ret_tail_cap_kappa=cap_kappa,
    )
    try:
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
    finally:
        env.close()


def test_ret_tail_v2_registered() -> None:
    assert "ReT_tail_v2" in REWARD_MODE_OPTIONS


def test_ret_tail_v2_defaults_are_branch_audit_defaults() -> None:
    assert RET_TAIL_W_SC == pytest.approx(0.30)
    assert RET_TAIL_W_RC == pytest.approx(0.60)
    assert RET_TAIL_W_CE == pytest.approx(0.10)
    assert RET_TAIL_CAP_KAPPA == pytest.approx(0.40)
    assert RET_TAIL_INV_KAPPA == pytest.approx(0.25)
    assert RET_TAIL_BOOST == pytest.approx(0.0)
    assert RET_TAIL_TRANSFORM == "identity"
    assert RET_TAIL_GAMMA == pytest.approx(1.0)
    assert RET_TAIL_BETA == pytest.approx(2.0)


def test_ret_tail_v2_reward_is_finite_bounded_and_reported() -> None:
    env = _make_env()
    try:
        env.reset(seed=1)
        for _ in range(10):
            _, reward, terminated, truncated, info = env.step(
                np.array([3, 0], dtype=np.int64)
            )
            assert np.isfinite(reward)
            assert 0.0 < reward <= 1.0
            assert info["ret_tail_step"] == pytest.approx(reward)
            assert info["ret_tail_transform"] == "identity"
            assert info["ret_tail_step"] == pytest.approx(info["ret_tail_base_step"])
            assert 0.0 < info["ret_tail_cost_efficiency"] <= 1.0
            if terminated or truncated:
                break
    finally:
        env.close()


def test_ret_tail_v2_cost_bites_monotonically() -> None:
    high_action = np.array([5, 2], dtype=np.int64)  # I1344 + S3
    low_cost = _avg_reward(inv_kappa=0.1, cap_kappa=0.1, action=high_action)
    mid_cost = _avg_reward(inv_kappa=0.5, cap_kappa=0.25, action=high_action)
    high_cost = _avg_reward(inv_kappa=1.5, cap_kappa=0.5, action=high_action)
    assert low_cost > mid_cost > high_cost


def test_ret_tail_v2_power_transform_steepens_without_reordering() -> None:
    info = {
        "new_demanded": 10_000.0,
        "new_backorder_qty": 1_000.0,
        "pending_backorder_qty": 20_000.0,
        "step_disruption_hours": 0.0,
    }
    identity_env = _make_env(max_steps=5)
    power_env = _make_env(ret_tail_transform="power", ret_tail_gamma=2.0, max_steps=5)
    try:
        identity_env.reset(seed=3)
        power_env.reset(seed=3)
        identity = identity_env.unwrapped._compute_ret_tail_v2(info, shifts=1)
        powered = power_env.unwrapped._compute_ret_tail_v2(info, shifts=1)

        assert 0.0 < identity["ret_tail_step"] < 1.0
        assert powered["ret_tail_base_step"] == pytest.approx(identity["ret_tail_step"])
        assert powered["ret_tail_step"] == pytest.approx(identity["ret_tail_step"] ** 2)
        assert powered["ret_tail_step"] < identity["ret_tail_step"]
        assert powered["ret_tail_transform"] == "power"
    finally:
        identity_env.close()
        power_env.close()


def test_ret_tail_v2_exp_norm_transform_is_bounded_and_monotone() -> None:
    env = _make_env(ret_tail_transform="exp_norm", ret_tail_beta=4.0, max_steps=5)
    try:
        env.reset(seed=5)
        low = env.unwrapped._compute_ret_tail_v2(
            {
                "new_demanded": 10_000.0,
                "new_backorder_qty": 3_000.0,
                "pending_backorder_qty": 30_000.0,
                "step_disruption_hours": 0.0,
            },
            shifts=1,
        )
        high = env.unwrapped._compute_ret_tail_v2(
            {
                "new_demanded": 10_000.0,
                "new_backorder_qty": 0.0,
                "pending_backorder_qty": 0.0,
                "step_disruption_hours": 0.0,
            },
            shifts=1,
        )

        assert 0.0 < low["ret_tail_step"] < high["ret_tail_step"] <= 1.0
        assert low["ret_tail_transform"] == "exp_norm"
    finally:
        env.close()
