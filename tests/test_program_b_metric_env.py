from __future__ import annotations

import numpy as np
import pytest

from supply_chain.program_o_ret_env import compute_service_safe_reward


def test_service_safe_reward_assigns_zero_to_omitted_quantity() -> None:
    metrics = {"ration_ret_visible": 0.8, "omitted_quantity": 20.0}
    assert compute_service_safe_reward(metrics, demanded_quantity=100.0) == pytest.approx(0.64)


def test_service_safe_reward_clips_visible_endpoint() -> None:
    metrics = {"ration_ret_visible": 1.4, "omitted_quantity": 0.0}
    assert compute_service_safe_reward(metrics, demanded_quantity=100.0) == pytest.approx(1.0)


def test_service_safe_reward_is_zero_when_all_demand_is_unresolved() -> None:
    metrics = {"ration_ret_visible": 0.8, "omitted_quantity": 100.0}
    assert compute_service_safe_reward(metrics, demanded_quantity=100.0) == pytest.approx(0.0)


def test_service_safe_reward_rejects_nonpositive_demand() -> None:
    with pytest.raises(ValueError):
        compute_service_safe_reward({"ration_ret_visible": 0.8}, demanded_quantity=0.0)


def test_service_safe_reward_returns_finite_scalar() -> None:
    result = compute_service_safe_reward(
        {"ration_ret_visible": float(np.nan_to_num(0.5)), "omitted_quantity": 5.0},
        demanded_quantity=10.0,
    )
    assert np.isfinite(result)
