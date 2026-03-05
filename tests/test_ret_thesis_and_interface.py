from __future__ import annotations

import pytest

from supply_chain.external_env_interface import (
    get_shift_control_env_spec,
    make_shift_control_env,
    spec_to_dict,
)
from supply_chain.env_experimental_shifts import MFSCGymEnvShifts, NUM_TRACKED_OPS


def test_ret_thesis_components_cover_all_cases() -> None:
    env = MFSCGymEnvShifts(reward_mode="ReT_thesis")
    step_hours = 10.0

    no_demand = env._compute_ret_thesis_components({}, step_hours)
    assert no_demand["ret_case"] == "no_demand"
    assert no_demand["ret_value"] == 1.0

    fill_only = env._compute_ret_thesis_components(
        {
            "new_demanded": 100.0,
            "new_backorder_qty": 10.0,
            "step_disruption_hours": 0.0,
        },
        step_hours,
    )
    assert fill_only["ret_case"] == "fill_rate_only"
    assert fill_only["ret_value"] == pytest.approx(0.9)

    autotomy = env._compute_ret_thesis_components(
        {
            "new_demanded": 100.0,
            "new_backorder_qty": 2.0,
            "step_disruption_hours": 13.0,
        },
        step_hours,
    )
    assert autotomy["ret_case"] == "autotomy"
    assert autotomy["ret_value"] == pytest.approx(0.9)

    recovery = env._compute_ret_thesis_components(
        {
            "new_demanded": 100.0,
            "new_backorder_qty": 10.0,
            "step_disruption_hours": 26.0,
        },
        step_hours,
    )
    assert recovery["ret_case"] == "recovery"
    assert recovery["ret_value"] == pytest.approx(1.0 / (1.0 + 0.2))

    non_recovery = env._compute_ret_thesis_components(
        {
            "new_demanded": 100.0,
            "new_backorder_qty": 60.0,
            "step_disruption_hours": step_hours * NUM_TRACKED_OPS,
        },
        step_hours,
    )
    assert non_recovery["ret_case"] == "non_recovery"
    assert non_recovery["ret_value"] == 0.0


def test_shift_env_ret_thesis_step_exposes_component_metadata() -> None:
    env = MFSCGymEnvShifts(step_size_hours=24, max_steps=2, reward_mode="ReT_thesis")
    env.reset(seed=7)
    _, _, _, _, info = env.step([0.0, 0.0, 0.0, 0.0, 0.0])
    assert "ret_components" in info
    assert "ret_case" in info["ret_components"]
    assert "reward_total" in info["ret_components"]
    assert (
        info["ret_components"]["thresholds_source"] == "configurable_repo_approximation"
    )


def test_external_interface_matches_shift_env_contract() -> None:
    spec = get_shift_control_env_spec()
    env = make_shift_control_env(max_steps=1)

    assert spec.env_variant == "shift_control"
    assert spec.reward_mode == "ReT_thesis"
    assert len(spec.observation_fields) == env.observation_space.shape[0]
    assert len(spec.action_fields) == env.action_space.shape[0]

    payload = spec_to_dict(spec)
    assert payload["shift_mapping"]["signal_ge_0.33"] == 3
