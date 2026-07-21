from __future__ import annotations

import gymnasium as gym
import numpy as np

from supply_chain.program_q2_reward import (
    ProgramQ2RewardWrapper,
    TerminalRewardCalibration,
)


class DelayedRewardEnv(gym.Env):
    def __init__(self) -> None:
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(0.0, 3.0, shape=(1,), dtype=np.float32)
        self.period = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.period = 0
        return np.asarray([0.0], dtype=np.float32), {}

    def step(self, action):
        self.period += 1
        terminal = self.period == 3
        reward = 0.8 if terminal else 0.0
        return np.asarray([self.period], dtype=np.float32), reward, terminal, False, {}


def _episode(wrapper: ProgramQ2RewardWrapper) -> tuple[list[float], dict]:
    wrapper.reset()
    rewards = []
    final = {}
    for _ in range(3):
        _, reward, terminated, _, final = wrapper.step(0)
        rewards.append(reward)
        if terminated:
            break
    return rewards, final


def test_raw_reward_is_byte_for_byte_semantically_unchanged() -> None:
    rewards, final = _episode(ProgramQ2RewardWrapper(DelayedRewardEnv()))
    assert rewards == [0.0, 0.0, 0.8]
    assert final["canonical_episode_return"] == 0.8
    assert final["learning_episode_return"] == 0.8


def test_standardization_changes_only_terminal_signal() -> None:
    wrapper = ProgramQ2RewardWrapper(
        DelayedRewardEnv(),
        mode="standardized_terminal",
        calibration=TerminalRewardCalibration(mean=0.5, standard_deviation=0.1),
    )
    rewards, final = _episode(wrapper)
    np.testing.assert_allclose(rewards, [0.0, 0.0, 3.0])
    assert final["canonical_episode_return"] == 0.8


def test_pbrs_telescopes_and_preserves_terminal_objective() -> None:
    wrapper = ProgramQ2RewardWrapper(
        DelayedRewardEnv(), mode="pbrs_terminal", potential=lambda obs: float(obs[0]) / 10.0
    )
    rewards, final = _episode(wrapper)
    np.testing.assert_allclose(rewards, [0.1, 0.1, 0.6])
    np.testing.assert_allclose(sum(rewards), 0.8)
    assert final["canonical_episode_return"] == 0.8


def test_pbrs_nonzero_initial_potential_differs_by_policy_independent_constant() -> None:
    wrapper = ProgramQ2RewardWrapper(
        DelayedRewardEnv(), mode="pbrs_terminal", potential=lambda obs: 0.25 + float(obs[0])
    )
    rewards, _ = _episode(wrapper)
    np.testing.assert_allclose(sum(rewards), 0.8 - 0.25)
