"""Reward contracts for prospective Program Q2 experiments.

The wrapped environment remains the frozen Program O/Q physical contract.  The
wrapper changes only the learning signal and always exposes the unmodified
canonical terminal ReT in ``info`` for evaluation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import gymnasium as gym
import numpy as np


RewardMode = Literal["raw_terminal", "standardized_terminal", "pbrs_terminal"]
Potential = Callable[[np.ndarray], float]


@dataclass(frozen=True)
class TerminalRewardCalibration:
    """Frozen affine calibration estimated on development data only."""

    mean: float
    standard_deviation: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.mean):
            raise ValueError("reward mean must be finite")
        if not np.isfinite(self.standard_deviation) or self.standard_deviation <= 0:
            raise ValueError("reward standard deviation must be finite and positive")


def zero_potential(_observation: np.ndarray) -> float:
    return 0.0


class ProgramQ2RewardWrapper(gym.Wrapper):
    """Apply an auditable Q2 learning reward without changing evaluation.

    For PBRS, ``gamma`` is deliberately fixed to one.  The terminal potential
    is forced to zero, so the episode return is canonical ReT minus the
    policy-independent initial potential.  This preserves action rankings.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        mode: RewardMode = "raw_terminal",
        calibration: TerminalRewardCalibration | None = None,
        potential: Potential = zero_potential,
    ) -> None:
        super().__init__(env)
        if mode not in ("raw_terminal", "standardized_terminal", "pbrs_terminal"):
            raise ValueError(f"unknown reward mode: {mode}")
        if mode == "standardized_terminal" and calibration is None:
            raise ValueError("standardized terminal reward requires frozen calibration")
        self.mode = mode
        self.calibration = calibration
        self.potential = potential
        self._previous_potential: float | None = None
        self._canonical_return = 0.0
        self._learning_return = 0.0

    def reset(self, **kwargs: Any):
        observation, info = self.env.reset(**kwargs)
        self._previous_potential = float(self.potential(np.asarray(observation)))
        if not np.isfinite(self._previous_potential):
            raise ValueError("potential returned a non-finite initial value")
        self._canonical_return = 0.0
        self._learning_return = 0.0
        return observation, dict(info)

    def step(self, action: Any):
        observation, canonical_reward, terminated, truncated, info = self.env.step(action)
        canonical = float(canonical_reward)
        self._canonical_return += canonical
        if self.mode == "raw_terminal":
            learning = canonical
        elif self.mode == "standardized_terminal":
            assert self.calibration is not None
            # Only the terminal reward is transformed.  Intermediate zeros
            # remain zero instead of receiving a repeated negative offset.
            learning = (
                (canonical - self.calibration.mean) / self.calibration.standard_deviation
                if terminated or truncated
                else 0.0
            )
        else:
            if self._previous_potential is None:
                raise RuntimeError("reset() must be called before step()")
            next_potential = (
                0.0
                if terminated or truncated
                else float(self.potential(np.asarray(observation)))
            )
            if not np.isfinite(next_potential):
                raise ValueError("potential returned a non-finite value")
            learning = canonical + next_potential - self._previous_potential
            self._previous_potential = next_potential
        self._learning_return += float(learning)
        enriched = dict(info)
        enriched.update(
            canonical_reward=canonical,
            learning_reward=float(learning),
            canonical_episode_return=float(self._canonical_return),
            learning_episode_return=float(self._learning_return),
            reward_mode=self.mode,
        )
        return observation, float(learning), terminated, truncated, enriched
