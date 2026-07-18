"""Exact finite Bayes-adaptive benchmark for Program T.

The hidden state is (model class, demand regime).  The model class is fixed for
an episode and the regime follows a model-specific Markov chain.  Weekly demand
is the observation, so the reachable continuous beliefs can be enumerated
exactly for the short frozen horizon.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import comb
from typing import Iterable

import numpy as np


Latent = tuple[int, int]
Physical = tuple[int, int, int, int]  # inv C, inv H, backlog C, backlog H


@dataclass(frozen=True)
class ExactPOMDPConfig:
    horizon: int = 5
    weekly_units: int = 3
    persistence_by_model: tuple[float, ...] = (0.65, 0.80, 0.92)
    dominant_share_by_model: tuple[float, ...] = (0.70, 0.80, 0.90)
    backlog_penalty: float = 1.0
    worst_product_penalty: float = 0.5

    def __post_init__(self) -> None:
        if not 1 <= self.horizon <= 6:
            raise ValueError("exact benchmark horizon must be in [1, 6]")
        if self.weekly_units != 3:
            raise ValueError("four actions require exactly three weekly production units")
        if len(self.persistence_by_model) != 3 or len(self.dominant_share_by_model) != 3:
            raise ValueError("Program T exact benchmark freezes three model classes")


class ExactProductMixPOMDP:
    def __init__(self, config: ExactPOMDPConfig = ExactPOMDPConfig()) -> None:
        self.config = config
        self.latents: tuple[Latent, ...] = tuple((model, regime) for model in range(3) for regime in range(2))

    @property
    def initial_belief(self) -> tuple[float, ...]:
        return (1 / len(self.latents),) * len(self.latents)

    def transition_probability(self, old: Latent, new: Latent) -> float:
        old_model, old_regime = old
        new_model, new_regime = new
        if old_model != new_model:
            return 0.0
        persistence = self.config.persistence_by_model[old_model]
        return persistence if old_regime == new_regime else 1 - persistence

    def demand_probability(self, latent: Latent, demand_c: int) -> float:
        model, regime = latent
        share = self.config.dominant_share_by_model[model]
        probability_c = share if regime == 1 else 1 - share
        n = self.config.weekly_units
        return comb(n, demand_c) * probability_c**demand_c * (1 - probability_c) ** (n - demand_c)

    def predictive_latent(self, belief: tuple[float, ...]) -> np.ndarray:
        result = np.zeros(len(self.latents), dtype=float)
        for old_index, old in enumerate(self.latents):
            for new_index, new in enumerate(self.latents):
                result[new_index] += belief[old_index] * self.transition_probability(old, new)
        return result

    def observation_probability(self, belief: tuple[float, ...], demand_c: int) -> float:
        predictive = self.predictive_latent(belief)
        return float(sum(
            predictive[index] * self.demand_probability(latent, demand_c)
            for index, latent in enumerate(self.latents)
        ))

    def update_belief(self, belief: tuple[float, ...], demand_c: int) -> tuple[float, ...]:
        predictive = self.predictive_latent(belief)
        posterior = np.asarray([
            predictive[index] * self.demand_probability(latent, demand_c)
            for index, latent in enumerate(self.latents)
        ])
        total = float(posterior.sum())
        if total <= 0:
            raise RuntimeError("reachable observation has zero probability")
        return tuple(float(round(value / total, 12)) for value in posterior)

    def physical_transition(self, state: Physical, action: int, demand_c: int) -> Physical:
        if action not in (0, 1, 2, 3):
            raise ValueError("action must be in {0,1,2,3}")
        inv_c, inv_h, backlog_c, backlog_h = state
        supply_c = inv_c + action
        supply_h = inv_h + (3 - action)
        need_c = backlog_c + demand_c
        need_h = backlog_h + (3 - demand_c)
        served_c = min(supply_c, need_c)
        served_h = min(supply_h, need_h)
        return (
            supply_c - served_c,
            supply_h - served_h,
            need_c - served_c,
            need_h - served_h,
        )

    def reward(self, state: Physical) -> float:
        backlog_c, backlog_h = state[2], state[3]
        return -(
            self.config.backlog_penalty * (backlog_c + backlog_h)
            + self.config.worst_product_penalty * max(backlog_c, backlog_h)
        )

    def solve(self, initial_state: Physical = (0, 0, 0, 0)) -> dict:
        visited: set[tuple] = set()

        @lru_cache(maxsize=None)
        def value(time: int, state: Physical, belief: tuple[float, ...]) -> tuple[float, int]:
            visited.add((time, state, belief))
            if time == self.config.horizon:
                return 0.0, 0
            action_values: list[float] = []
            for action in range(4):
                expected = 0.0
                for demand_c in range(4):
                    probability = self.observation_probability(belief, demand_c)
                    if probability == 0:
                        continue
                    next_state = self.physical_transition(state, action, demand_c)
                    next_belief = self.update_belief(belief, demand_c)
                    continuation, _ = value(time + 1, next_state, next_belief)
                    expected += probability * (self.reward(next_state) + continuation)
                action_values.append(expected)
            best_action = int(np.argmax(action_values))
            return float(action_values[best_action]), best_action

        optimum, first_action = value(0, initial_state, self.initial_belief)
        return {
            "schema_version": "program_t_exact_pomdp_result_v1",
            "horizon": self.config.horizon,
            "latent_states": len(self.latents),
            "actions": 4,
            "optimal_value": optimum,
            "first_action": first_action,
            "reachable_belief_physical_states": len(visited),
            "cache": value.cache_info()._asdict(),
        }


def solve_grid(horizons: Iterable[int] = (4, 5, 6)) -> dict:
    rows = [ExactProductMixPOMDP(ExactPOMDPConfig(horizon=horizon)).solve() for horizon in horizons]
    return {
        "schema_version": "program_t_exact_pomdp_grid_v1",
        "solver": "exact_reachable_belief_dynamic_programming",
        "rows": rows,
    }
