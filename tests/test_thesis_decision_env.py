from __future__ import annotations

import numpy as np
import pytest

from supply_chain.config import (
    INVENTORY_BUFFERS,
    THESIS_ROBUSTNESS_DOWNSTREAM_Q_SOURCE,
    TRACK_A_TRAINING_DOWNSTREAM_Q_SOURCE,
)
from supply_chain.thesis_decision_env import (
    Discrete18TrackAEnv,
    make_discrete18_track_a_env,
    make_thesis_factorized_track_a_env,
)


def test_thesis_factorized_track_a_env_applies_initial_decision() -> None:
    env = make_thesis_factorized_track_a_env(
        max_steps=1,
        reward_mode="control_v1",
        observation_version="v4",
        priming_enabled=False,
        initial_action=np.asarray([1, 2], dtype=np.int64),
    )

    assert env.action_space.nvec.tolist() == [6, 3]
    _obs, info = env.reset(seed=7)

    decision = info["thesis_decision"]
    assert info["action_contract"] == "track_a_thesis_factorized_v1"
    assert info["action_space_mode"] == "thesis_factorized"
    assert info["downstream_q_source"] == TRACK_A_TRAINING_DOWNSTREAM_Q_SOURCE
    assert info["downstream_q_lane"] == "thesis_replication_training"
    assert decision["inventory_period_hours"] == pytest.approx(168.0)
    assert decision["assembly_shifts"] == 3
    assert decision["inventory_buffer_targets"]["op3_rm"] == pytest.approx(
        INVENTORY_BUFFERS[168]["op3_rm"]
    )
    assert env.unwrapped.sim.params["assembly_shifts"] == 3
    assert env.unwrapped.sim.inventory_buffer_targets["op9_rations"] == pytest.approx(
        INVENTORY_BUFFERS[168]["op9_rations"]
    )

    _obs, _reward, _terminated, _truncated, step_info = env.step(
        np.asarray([0, 1], dtype=np.int64)
    )
    assert step_info["downstream_q_source"] == TRACK_A_TRAINING_DOWNSTREAM_Q_SOURCE
    assert step_info["thesis_decision"]["inventory_period_hours"] is None
    assert step_info["thesis_decision"]["assembly_shifts"] == 2
    assert env.unwrapped.sim.inventory_buffer_targets == {}
    env.close()


def test_thesis_factorized_track_a_env_marks_table_620_as_robustness_lane() -> None:
    env = make_thesis_factorized_track_a_env(
        max_steps=1,
        reward_mode="control_v1",
        observation_version="v4",
        priming_enabled=False,
        downstream_q_source=THESIS_ROBUSTNESS_DOWNSTREAM_Q_SOURCE,
    )

    _obs, info = env.reset(seed=13)
    assert info["downstream_q_source"] == THESIS_ROBUSTNESS_DOWNSTREAM_Q_SOURCE
    assert info["downstream_q_lane"] == "robustness_sensitivity"
    assert env.unwrapped.downstream_q_source == THESIS_ROBUSTNESS_DOWNSTREAM_Q_SOURCE
    env.close()


def test_discrete18_track_a_env_maps_to_same_factorized_surface() -> None:
    assert Discrete18TrackAEnv.encode_discrete_action(1, 2) == 5
    assert Discrete18TrackAEnv.decode_discrete_action(5).tolist() == [1, 2]

    env = make_discrete18_track_a_env(
        max_steps=1,
        reward_mode="control_v1",
        observation_version="v4",
        priming_enabled=False,
        initial_action=5,
    )

    assert env.action_space.n == 18
    _obs, info = env.reset(seed=11)
    assert info["action_contract"] == "track_a_discrete18_v1"
    assert info["action_space_mode"] == "discrete_18"
    assert info["thesis_decision"]["inventory_period_hours"] == pytest.approx(168.0)
    assert info["thesis_decision"]["assembly_shifts"] == 3

    _obs, _reward, _terminated, _truncated, step_info = env.step(5)
    assert step_info["discrete_action"] == 5
    assert step_info["thesis_factorized_action"] == [1, 2]
    assert step_info["thesis_decision"]["assembly_shifts"] == 3
    env.close()
