from __future__ import annotations

import numpy as np
import pytest

from supply_chain.env_experimental_shifts import REWARD_MODE_OPTIONS
from supply_chain.thesis_decision_env import make_thesis_factorized_track_a_env


def _make_env(**overrides):
    kwargs = {
        "risk_level": "current",
        "risk_occurrence_mode": "thesis_window",
        "raw_material_flow_mode": "kit_equivalent_order_up_to",
        "raw_material_order_up_to_multiplier": 2.0,
        "priming_enabled": False,
        "max_steps": 5,
    }
    kwargs.update(overrides)
    return make_thesis_factorized_track_a_env(**kwargs)


def test_new_reward_modes_are_registered() -> None:
    assert "ReT_cd_balanced" in REWARD_MODE_OPTIONS
    assert "ReT_excel_plus_cvar" in REWARD_MODE_OPTIONS


def test_ret_cd_balanced_is_finite_and_reports_balanced_cost_weight() -> None:
    env = _make_env(
        reward_mode="ReT_cd_balanced",
        ret_cd_balanced_n_kappa=0.05,
    )
    try:
        env.reset(seed=11)
        _, reward, _terminated, _truncated, info = env.step(
            np.array([1, 0], dtype=np.int64)
        )
        assert np.isfinite(reward)
        assert 0.0 < reward < 1.0
        assert info["ret_cd_balanced_step"] == pytest.approx(reward)
        assert info["ret_cd_balanced_n_kappa"] == pytest.approx(0.05)
        assert info["ret_cd_balanced_components"]["exponents"]["n_kappa"] == pytest.approx(
            0.05
        )
    finally:
        env.close()


def test_ret_excel_plus_cvar_is_finite_and_reports_tail_penalty() -> None:
    env = _make_env(
        reward_mode="ReT_excel_plus_cvar",
        ret_excel_cvar_alpha=0.5,
        ret_excel_cvar_window=4,
    )
    try:
        env.reset(seed=13)
        _, reward, _terminated, _truncated, info = env.step(
            np.array([1, 0], dtype=np.int64)
        )
        assert np.isfinite(reward)
        assert "ret_excel_plus_cvar_step" in info
        assert info["ret_excel_plus_cvar_step"] == pytest.approx(reward)
        assert info["ret_excel_plus_cvar_alpha"] == pytest.approx(0.5)
        assert info["ret_excel_plus_cvar_penalty"] >= 0.0
    finally:
        env.close()


def test_ret_excel_plus_cvar_penalizes_worse_tail_loss() -> None:
    low_loss_env = _make_env(
        reward_mode="ReT_excel_plus_cvar",
        ret_excel_cvar_alpha=0.8,
        max_steps=1,
    )
    high_loss_env = _make_env(
        reward_mode="ReT_excel_plus_cvar",
        ret_excel_cvar_alpha=0.8,
        max_steps=1,
    )
    try:
        low_loss_env.reset(seed=17)
        high_loss_env.reset(seed=17)
        low = low_loss_env.unwrapped._compute_ret_excel_plus_cvar(
            {"new_backorder_qty": 0.0, "new_demanded": 10_000.0}
        )
        high = high_loss_env.unwrapped._compute_ret_excel_plus_cvar(
            {"new_backorder_qty": 10_000.0, "new_demanded": 10_000.0}
        )
        assert high["ret_excel_plus_cvar_step"] < low["ret_excel_plus_cvar_step"]
        assert high["ret_excel_plus_cvar_penalty"] > low["ret_excel_plus_cvar_penalty"]
    finally:
        low_loss_env.close()
        high_loss_env.close()
