"""Causal fixed-window observations for non-recurrent Q2 learners."""

from __future__ import annotations

from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np


class CausalFrameStack(gym.ObservationWrapper):
    """Right-align the current and past observations in a flat vector."""

    def __init__(self, env: gym.Env, *, frames: int = 8) -> None:
        super().__init__(env)
        if frames <= 0:
            raise ValueError("frames must be positive")
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise TypeError("CausalFrameStack requires a Box observation space")
        self.frames = int(frames)
        self._shape = int(np.prod(env.observation_space.shape))
        low = np.tile(np.asarray(env.observation_space.low).reshape(-1), self.frames)
        high = np.tile(np.asarray(env.observation_space.high).reshape(-1), self.frames)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self._history: deque[np.ndarray] = deque(maxlen=self.frames)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        current = np.asarray(observation, dtype=np.float32).reshape(-1)
        if current.shape != (self._shape,):
            raise ValueError("underlying observation shape drift")
        self._history.append(current.copy())
        output = np.zeros((self.frames, self._shape), dtype=np.float32)
        values = list(self._history)
        output[-len(values) :] = values
        return output.reshape(-1)

    def reset(self, **kwargs: Any):
        self._history.clear()
        return super().reset(**kwargs)
