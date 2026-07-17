"""Incremental Program O-R environment with canonical ReT terminal reward."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from supply_chain.program_o_full_des_transducer import (
    FullDESSkeleton,
    extract_full_des_skeleton,
    simulate_full_des_frontier,
)
from supply_chain.program_o_state_rich import (
    StateRichConfiguration,
    StateRichObservation,
    state_rich_calendar,
)


@dataclass(frozen=True)
class ProgramORetCell:
    cell_id: str
    regime_persistence: float
    dominant_share: float


CONFIRMED_RET_CELLS = (
    ProgramORetCell("rho75_share90", 0.75, 0.90),
    ProgramORetCell("rho90_share75", 0.90, 0.75),
    ProgramORetCell("rho90_share90", 0.90, 0.90),
)
OBSERVATION_DIM = 21
_REPLAY_CONFIG = StateRichConfiguration("belief_mpc", 3)


def normalized_state_rich_observation(
    observation: StateRichObservation,
) -> np.ndarray:
    """Map the frozen state-rich observation to a bounded learner vector."""
    values: list[float] = []
    for field in (
        observation.on_hand,
        observation.locked_pipeline,
        observation.backlog_quantity,
    ):
        values.extend(float(value) / 120_000.0 for value in field)
    values.extend(float(value) / 48.0 for value in observation.backlog_orders)
    values.extend(float(value) / 1_344.0 for value in observation.max_backlog_age)
    values.extend(float(value) / 120_000.0 for value in observation.in_flight_quantity)
    values.extend((float(observation.belief_c), float(observation.predicted_share_c)))
    previous = np.zeros(5, dtype=np.float32)
    previous[4 if observation.previous_action is None else int(observation.previous_action)] = 1.0
    values.extend(previous.tolist())
    values.extend(
        (float(observation.week) / 7.0, float(observation.remaining_decisions) / 8.0)
    )
    vector = np.asarray(values, dtype=np.float32)
    if vector.shape != (OBSERVATION_DIM,):
        raise AssertionError(f"Program O-R observation shape drift: {vector.shape}")
    return np.clip(vector, 0.0, 1.0)


SkeletonFactory = Callable[[int, ProgramORetCell], FullDESSkeleton]


class ProgramORetOnlyEnv(gym.Env[np.ndarray, int]):
    """Eight-decision Program O-R environment; only terminal ReT is rewarded."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        scheduler: Mapping[str, Sequence[str]],
        tape_seed_start: int,
        tape_seed_end: int,
        cells: Sequence[ProgramORetCell] = CONFIRMED_RET_CELLS,
        belief_model_persistence: float = 0.75,
        belief_model_dominant_share: float = 0.90,
        skeleton_factory: SkeletonFactory | None = None,
    ) -> None:
        super().__init__()
        if int(tape_seed_end) < int(tape_seed_start):
            raise ValueError("tape_seed_end must be >= tape_seed_start")
        if not cells:
            raise ValueError("at least one Program O-R cell is required")
        self.scheduler = {str(key): tuple(value) for key, value in scheduler.items()}
        self.tape_seed_start = int(tape_seed_start)
        self.tape_seed_end = int(tape_seed_end)
        self.cells = tuple(cells)
        self.belief_model_persistence = float(belief_model_persistence)
        self.belief_model_dominant_share = float(belief_model_dominant_share)
        self._skeleton_factory = skeleton_factory or self._direct_skeleton
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBSERVATION_DIM,), dtype=np.float32
        )
        self._episode_index = 0
        self._actions: list[int] = []
        self._cell: ProgramORetCell | None = None
        self._skeleton: FullDESSkeleton | None = None

    def _direct_skeleton(self, seed: int, cell: ProgramORetCell) -> FullDESSkeleton:
        skeleton, _sim = extract_full_des_skeleton(
            seed=int(seed),
            scheduler=self.scheduler,
            regime_persistence=float(cell.regime_persistence),
            dominant_share=float(cell.dominant_share),
            downstream_freight_physics_mode="fixed_clock_physical_v1",
        )
        return skeleton

    def _tape_seed(self) -> int:
        count = self.tape_seed_end - self.tape_seed_start + 1
        if self._episode_index >= count:
            raise RuntimeError("Program O-R training tape namespace exhausted")
        return self.tape_seed_start + self._episode_index

    def _decisions(self) -> tuple[Any, ...]:
        if self._skeleton is None:
            raise RuntimeError("reset() must be called before step()")
        weeks = int(self._skeleton.decision_weeks)
        padded = tuple(self._actions) + (0,) * (weeks - len(self._actions))
        _calendar, decisions = state_rich_calendar(
            skeleton=self._skeleton.as_dict(),
            scheduler=self.scheduler,
            config=_REPLAY_CONFIG,
            regime_persistence=self.belief_model_persistence,
            dominant_share=self.belief_model_dominant_share,
            action_overrides=padded,
        )
        return decisions

    def _current_observation(self) -> tuple[np.ndarray, str]:
        decisions = self._decisions()
        index = len(self._actions)
        if index >= len(decisions):
            raise RuntimeError("no observation exists after terminal action")
        observation = decisions[index].observation
        return normalized_state_rich_observation(observation), observation.observation_sha256

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}
        cell_index = int(options.get("cell_index", self._episode_index % len(self.cells)))
        if cell_index not in range(len(self.cells)):
            raise ValueError("cell_index is outside the frozen cell set")
        self._cell = self.cells[cell_index]
        tape_seed = int(options["tape_seed"]) if "tape_seed" in options else self._tape_seed()
        supplied = options.get("skeleton")
        self._skeleton = (
            supplied
            if isinstance(supplied, FullDESSkeleton)
            else self._skeleton_factory(tape_seed, self._cell)
        )
        self._actions = []
        self._episode_index += 1
        observation, digest = self._current_observation()
        return observation, {"observation_sha256": digest}

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._skeleton is None:
            raise RuntimeError("reset() must be called before step()")
        action_value = int(action)
        if not self.action_space.contains(action_value):
            raise ValueError("Program O-R action must be in {0,1,2,3}")
        self._actions.append(action_value)
        terminated = len(self._actions) == int(self._skeleton.decision_weeks)
        if not terminated:
            observation, digest = self._current_observation()
            return observation, 0.0, False, False, {"observation_sha256": digest}

        metrics = simulate_full_des_frontier(
            skeleton=self._skeleton,
            scheduler=self.scheduler,
            calendars=np.asarray([self._actions], dtype=np.uint8),
        )
        terminal_metrics = {key: float(value[0]) for key, value in metrics.items()}
        reward = float(terminal_metrics["ret_visible"])
        observation = np.zeros(self.observation_space.shape, dtype=np.float32)
        info = {
            "calendar": list(self._actions),
            "metrics": terminal_metrics,
            "skeleton_sha256": self._skeleton.skeleton_sha256,
        }
        return observation, reward, True, False, info
