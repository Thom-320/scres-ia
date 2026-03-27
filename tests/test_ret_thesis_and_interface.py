from __future__ import annotations

import pytest

import numpy as np

from supply_chain.external_env_interface import (
    get_episode_terminal_metrics,
    get_observation_fields,
    get_shift_control_constraint_context,
    get_shift_control_env_spec,
    make_shift_control_env,
    run_episodes,
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
    assert spec.reward_mode == "ReT_seq_v1"
    assert len(spec.observation_fields) == env.observation_space.shape[0]
    assert len(spec.action_fields) == env.action_space.shape[0]

    payload = spec_to_dict(spec)
    assert payload["shift_mapping"]["signal_ge_0.33"] == 3


def test_v2_observation_contract_exposes_augmented_state() -> None:
    spec = get_shift_control_env_spec(
        reward_mode="control_v1", observation_version="v2"
    )
    env = make_shift_control_env(
        max_steps=2,
        reward_mode="control_v1",
        observation_version="v2",
        step_size_hours=24,
    )
    obs, info = env.reset(seed=7)

    assert spec.observation_version == "v2"
    assert tuple(spec.observation_fields) == get_observation_fields("v2")
    assert len(spec.observation_fields) == 18
    assert obs.shape == (18,)
    assert info["observation_version"] == "v2"
    assert obs[-3:].tolist() == pytest.approx([0.0, 0.0, 0.0])

    next_obs, _, _, _, step_info = env.step([0.0, 0.0, 0.0, 0.0, 0.0])
    assert step_info["observation_version"] == "v2"
    assert next_obs.shape == (18,)
    assert next_obs[-3] == pytest.approx(float(step_info["new_demanded"]) / 18_200.0)
    assert next_obs[-2] == pytest.approx(
        float(step_info["new_backorder_qty"]) / 18_200.0
    )
    assert next_obs[-1] == pytest.approx(
        float(step_info["step_disruption_hours"]) / 312.0
    )


def test_v3_observation_contract_exposes_normalized_cumulative_history() -> None:
    spec = get_shift_control_env_spec(
        reward_mode="control_v1", observation_version="v3"
    )
    env = make_shift_control_env(
        max_steps=2,
        reward_mode="control_v1",
        observation_version="v3",
        step_size_hours=24,
        risk_level="increased",
        stochastic_pt=True,
    )
    obs, info = env.reset(seed=7)

    assert spec.observation_version == "v3"
    assert tuple(spec.observation_fields) == get_observation_fields("v3")
    assert len(spec.observation_fields) == 20
    assert obs.shape == (20,)
    assert info["observation_version"] == "v3"
    assert obs[-5:].tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0, 0.0])

    next_obs, _, _, _, step_info = env.step([0.0, 0.0, 0.0, 0.0, 0.0])
    assert step_info["observation_version"] == "v3"
    assert next_obs.shape == (20,)
    assert 0.0 <= next_obs[-2] <= 1.0
    assert 0.0 <= next_obs[-1] <= 1.0
    # Cumulative backorder rate: with 48h deferred backorder counting (D-16),
    # exact step-level equality no longer holds. Check non-negative and bounded.
    assert next_obs[-2] >= 0.0
    max_op_hours = 24.0 * NUM_TRACKED_OPS
    assert next_obs[-1] == pytest.approx(
        min(1.0, float(step_info["step_disruption_hours"]) / max_op_hours)
    )


def test_v4_observation_contract_exposes_shift_and_upstream_disruption_state() -> None:
    spec = get_shift_control_env_spec(
        reward_mode="control_v1", observation_version="v4"
    )
    env = make_shift_control_env(
        max_steps=2,
        reward_mode="control_v1",
        observation_version="v4",
        step_size_hours=24,
        risk_level="increased",
        stochastic_pt=True,
    )
    obs, info = env.reset(seed=7)

    assert spec.observation_version == "v4"
    assert tuple(spec.observation_fields) == get_observation_fields("v4")
    assert len(spec.observation_fields) == 24
    assert obs.shape == (24,)
    assert info["observation_version"] == "v4"
    assert obs[15:24].tolist() == pytest.approx(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 / 3.0, 0.0, 0.0]
    )

    next_obs, _, _, _, step_info = env.step([0.0, 0.0, 0.0, 0.0, 0.0])
    assert step_info["observation_version"] == "v4"
    assert next_obs.shape == (24,)
    assert next_obs[16] == pytest.approx(
        float(step_info["new_backorder_qty"]) / 18_200.0
    )
    assert next_obs[21] == pytest.approx(float(step_info["shifts_active"]) / 3.0)
    assert 0.0 <= next_obs[22] <= 1.0
    assert 0.0 <= next_obs[23] <= 1.0


def test_reset_info_exposes_action_constraints() -> None:
    env = make_shift_control_env(max_steps=1)
    _, info = env.reset(seed=7)
    assert "action_constraints" in info
    constraints = info["action_constraints"]
    assert constraints["inventory_multiplier_range"]["min"] == pytest.approx(0.5)
    assert constraints["shift_signal_bands"]["signal_ge_0.33"] == 3
    assert constraints["base_control_parameters"]["op3_q"] > 0
    assert info["observation_version"] == "v1"


def test_external_constraint_context_exposes_base_parameters() -> None:
    context = get_shift_control_constraint_context()
    assert context["inventory_multiplier_range"]["max"] == pytest.approx(2.0)
    assert context["shift_signal_bands"]["signal_lt_-0.33"] == 1
    assert (
        context["base_control_parameters"]["op9_q_max"]
        >= context["base_control_parameters"]["op9_q_min"]
    )


def test_state_constraint_context_is_exposed_on_reset_and_step() -> None:
    env = make_shift_control_env(max_steps=1, reward_mode="control_v1")
    _, info = env.reset(seed=7)
    assert "state_constraint_context" in info
    state_context = info["state_constraint_context"]
    assert state_context["op3_total_dispatch_cap"] >= 0.0
    assert "inventory_detail" in state_context
    assert "cumulative_backorder_rate_by_inventory_node" in state_context
    assert "cumulative_disruption_fraction_by_operation" in state_context
    assert "pending_backorders_count" in state_context
    assert "pending_backorder_qty" in state_context
    assert "unattended_orders_total" in state_context
    assert state_context["cumulative_backorder_rate_by_inventory_node"][
        "rations_theatre"
    ] == pytest.approx(0.0)

    _, _, _, _, step_info = env.step([0.0, 0.0, 0.0, 0.0, 0.0])
    assert "state_constraint_context" in step_info
    assert step_info["state_constraint_context"]["total_inventory"] >= 0.0
    assert step_info["state_constraint_context"]["pending_backorders_count"] >= 0.0
    disruption_vector = step_info["state_constraint_context"][
        "cumulative_disruption_fraction_by_operation"
    ]
    assert len(disruption_vector) == 13


# ---------------------------------------------------------------------------
# run_episodes: generic callable-policy evaluation
# ---------------------------------------------------------------------------


def test_run_episodes_with_neutral_policy() -> None:
    """A neutral (all-zeros) policy runs and returns structured metrics."""

    def neutral_policy(obs: np.ndarray, info: dict) -> np.ndarray:
        return np.zeros(5, dtype=np.float32)

    results = run_episodes(
        neutral_policy,
        n_episodes=2,
        seed=1,
        env_kwargs={
            "reward_mode": "control_v1",
            "step_size_hours": 24,
            "max_steps": 4,
            "w_bo": 2.0,
            "w_cost": 0.06,
            "w_disr": 0.0,
        },
        policy_name="neutral_test",
    )
    assert len(results) == 2
    for row in results:
        assert row["policy"] == "neutral_test"
        assert row["steps"] == 4
        assert "reward_total" in row
        assert "fill_rate" in row
        assert 0.0 <= row["fill_rate"] <= 1.0
        assert "flow_fill_rate" in row
        assert 0.0 <= row["flow_fill_rate"] <= 1.0
        assert "order_level_ret_mean" in row
        assert "pct_steps_S1" in row
        assert "pct_steps_S2" in row
        assert "pct_steps_S3" in row


def test_run_episodes_with_custom_callable() -> None:
    """Any callable that maps (obs, info) -> action works."""

    class MyPolicy:
        def __init__(self) -> None:
            self.call_count = 0

        def __call__(self, obs: np.ndarray, info: dict) -> np.ndarray:
            self.call_count += 1
            # shift to S3 based on backorder_rate
            action = np.zeros(5, dtype=np.float32)
            if obs[7] > 0.1:
                action[4] = 1.0
            return action

    policy = MyPolicy()
    results = run_episodes(
        policy,
        n_episodes=1,
        seed=42,
        env_kwargs={
            "reward_mode": "control_v1",
            "risk_level": "increased",
            "step_size_hours": 24,
            "max_steps": 5,
            "w_bo": 2.0,
            "w_cost": 0.06,
            "w_disr": 0.0,
            "stochastic_pt": True,
        },
        policy_name="my_custom",
    )
    assert len(results) == 1
    assert results[0]["policy"] == "my_custom"
    assert policy.call_count == 5


def test_run_episodes_collect_trajectories() -> None:
    """When collect_trajectories=True, per-step data is captured."""
    results = run_episodes(
        lambda obs, info: np.zeros(5, dtype=np.float32),
        n_episodes=1,
        seed=1,
        env_kwargs={
            "reward_mode": "control_v1",
            "step_size_hours": 24,
            "max_steps": 3,
            "w_bo": 1.0,
            "w_cost": 0.02,
            "w_disr": 0.0,
        },
        collect_trajectories=True,
    )
    assert "trajectory" in results[0]
    traj = results[0]["trajectory"]
    assert len(traj) == 3
    assert "obs" in traj[0]
    assert "action" in traj[0]
    assert "reward" in traj[0]
    assert "info" in traj[0]


def test_terminal_metrics_match_env_order_level_definition() -> None:
    env = make_shift_control_env(
        reward_mode="control_v1",
        observation_version="v1",
        step_size_hours=168,
        max_steps=32,
        risk_level="increased",
        stochastic_pt=True,
        year_basis="thesis",
        w_bo=4.0,
        w_cost=0.02,
        w_disr=0.0,
    )
    obs, info = env.reset(seed=123)
    terminated = False
    truncated = False
    while not (terminated or truncated):
        obs, _, terminated, truncated, info = env.step(np.zeros(5, dtype=np.float32))

    metrics = get_episode_terminal_metrics(env)
    assert metrics["fill_rate_order_level"] == pytest.approx(
        env.sim._order_level_fill_rate()
    )
    assert metrics["fill_rate_state_terminal"] == pytest.approx(env.sim._fill_rate())
    assert metrics["backorder_rate_order_level"] == pytest.approx(
        1.0 - metrics["fill_rate_order_level"]
    )
    env.close()


def test_run_episodes_uses_terminal_fill_rate_not_flow_ratio() -> None:
    results = run_episodes(
        lambda obs, info: np.zeros(5, dtype=np.float32),
        n_episodes=1,
        seed=123,
        env_kwargs={
            "reward_mode": "control_v1",
            "observation_version": "v1",
            "step_size_hours": 168,
            "max_steps": 260,
            "risk_level": "increased",
            "stochastic_pt": True,
            "year_basis": "thesis",
            "w_bo": 4.0,
            "w_cost": 0.02,
            "w_disr": 0.0,
        },
        policy_name="static_s2_like",
    )
    row = results[0]
    assert row["fill_rate"] == pytest.approx(1.0 - row["backorder_rate"], rel=1e-6)
    assert row["fill_rate"] == pytest.approx(0.8137193203272498)
    assert row["fill_rate_state_terminal"] == pytest.approx(0.8143486469477659)
    assert row["fill_rate"] > row["flow_fill_rate"]
