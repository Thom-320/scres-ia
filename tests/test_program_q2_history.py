from __future__ import annotations

import gymnasium as gym
import numpy as np

from supply_chain.program_q2_history import CausalFrameStack


class CounterEnv(gym.Env):
    def __init__(self) -> None:
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(0.0, 10.0, shape=(2,), dtype=np.float32)
        self.value = 0

    def reset(self, *, seed=None, options=None):
        self.value = 1
        return np.asarray([1.0, 2.0], dtype=np.float32), {}

    def step(self, action):
        self.value += 1
        return np.asarray([self.value, 2 * self.value], dtype=np.float32), 0.0, False, False, {}


def test_frame_stack_is_right_aligned_and_contains_no_future_state() -> None:
    env = CausalFrameStack(CounterEnv(), frames=3)
    observation, _ = env.reset()
    np.testing.assert_array_equal(observation.reshape(3, 2)[:-1], 0.0)
    np.testing.assert_array_equal(observation.reshape(3, 2)[-1], [1.0, 2.0])
    observation, *_ = env.step(0)
    np.testing.assert_array_equal(
        observation.reshape(3, 2),
        [[0.0, 0.0], [1.0, 2.0], [2.0, 4.0]],
    )
