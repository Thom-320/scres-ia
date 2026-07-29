"""Observable closed-loop Program E environment over the frozen convoy physics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import gymnasium as gym
import numpy as np

from .dra2_experiment import advance_including, make_sim
from .episode_metrics import compute_episode_metrics


OBSERVATION_KEYS = (
    "op7_staged_inventory", "pending_assembly_quantity", "sb_inventory",
    "downstream_backlog_qty", "downstream_backlog_count", "oldest_backlog_age",
    "recent_7d_demand", "recent_7d_production", "recent_7d_delivery",
    "convoy_available", "convoy_return_eta", "time_since_departure",
    "staging_age", "op8_route_up", "previous_action_dispatch", "day_phase",
    "departures_to_date", "unavailable_hours_to_date",
)


@dataclass(frozen=True)
class ObservableHeuristic:
    staging_threshold: float = 2_500.0
    maximum_wait_hours: float = 48.0
    urgent_backlog_age_hours: float = 96.0

    def action(self, observation: Mapping[str, float], mask: Sequence[bool]) -> int:
        if not bool(mask[1]):
            return 0
        dispatch = (
            float(observation["op7_staged_inventory"]) >= self.staging_threshold
            or float(observation["staging_age"]) >= self.maximum_wait_hours
            or float(observation["oldest_backlog_age"]) >= self.urgent_backlog_age_hours
        )
        return int(dispatch)


class ProgramEConvoyEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        tapes: Sequence[dict[str, Any]],
        normalizers: Mapping[str, Any],
        *,
        episode_days: int = 56,
        random_tapes: bool = True,
    ) -> None:
        super().__init__()
        if not tapes:
            raise ValueError("Program E requires at least one tape")
        self.tapes = list(tapes)
        self.normalizers = dict(normalizers)
        self.episode_days = int(episode_days)
        self.random_tapes = bool(random_tapes)
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(
            low=-10.0, high=10.0, shape=(len(OBSERVATION_KEYS),), dtype=np.float32
        )
        self._tape_cursor = 0
        self.sim = None
        self.start = 0.0
        self.end = 0.0
        self.steps = 0
        self._resource_start: dict[str, float] = {}

    def _select_tape(self) -> dict[str, Any]:
        if self.random_tapes:
            index = int(self.np_random.integers(0, len(self.tapes)))
        else:
            index = self._tape_cursor % len(self.tapes)
            self._tape_cursor += 1
        return self.tapes[index]

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        tape = (options or {}).get("tape") or self._select_tape()
        self.sim, self.start = make_sim(tape)
        self.end = self.start + self.episode_days * 24.0
        self.steps = 0
        self._resource_start = dict(self.sim.op8_convoy_metrics())
        return self._normalized_observation(), {
            "tape_id": tape["tape_id"], "family": tape["family"]
        }

    def raw_observation(self) -> dict[str, float]:
        obs = dict(self.sim.get_op8_convoy_observation())
        obs["departures_to_date"] = float(
            self.sim.op8_convoy_departures
            - self._resource_start.get("op8_convoy_departures", 0.0)
        )
        obs["unavailable_hours_to_date"] = float(
            self.sim.op8_convoy_vehicle_hours
            - self._resource_start.get("op8_convoy_unavailable_hours", 0.0)
        )
        return {key: float(obs[key]) for key in OBSERVATION_KEYS}

    def _normalized_observation(self) -> np.ndarray:
        raw = self.raw_observation()
        scales = self.normalizers["observation_scales"]
        return np.asarray([
            np.clip(raw[key] / max(float(scales[key]), 1e-9), -10.0, 10.0)
            for key in OBSERVATION_KEYS
        ], dtype=np.float32)

    def action_masks(self) -> np.ndarray:
        return np.asarray([True, bool(self.sim.op8_convoy_dispatch_feasible())])

    def _daily_loss_terms(self) -> tuple[float, float]:
        now = float(self.sim.env.now)
        service = float(self.sim.pending_backorder_qty) * 24.0
        backlog_age = 0.0
        for order in self.sim.pending_backorders:
            backlog_age += (
                float(order.remaining_qty)
                * max(0.0, now - float(order.OPTj))
                * 24.0
            )
        return service, backlog_age

    def step(self, action: int):
        requested = int(action)
        mask = self.action_masks()
        effective = requested if mask[requested] else 0
        event = self.sim.apply_op8_convoy_action(effective, source="program_e_policy")
        advance_including(self.sim, min(self.end, float(self.sim.env.now) + 24.0))
        self.steps += 1
        service, backlog_age = self._daily_loss_terms()
        reward_scales = self.normalizers["reward_scales"]
        reward = -(
            service / max(float(reward_scales["daily_service_loss_p95"]), 1.0)
            + 0.1 * backlog_age
            / max(float(reward_scales["daily_backlog_age_p95"]), 1.0)
        )
        terminated = float(self.sim.env.now) >= self.end - 1e-9
        info = {
            "requested_action": requested,
            "effective_action": effective,
            "masked": requested != effective,
            "departed": bool(event["departed"]),
        }
        if terminated:
            metrics = compute_episode_metrics(self.sim, treatment_start=self.start)
            resources = self.sim.op8_convoy_metrics()
            info.update(metrics)
            info.update({
                "episode_departures": float(
                    resources["op8_convoy_departures"]
                    - self._resource_start.get("op8_convoy_departures", 0.0)
                ),
                "episode_unavailable_hours": float(
                    resources["op8_convoy_unavailable_hours"]
                    - self._resource_start.get("op8_convoy_unavailable_hours", 0.0)
                ),
            })
        return self._normalized_observation(), float(reward), terminated, False, info


def make_identity_normalizers() -> dict[str, Any]:
    return {
        "observation_scales": {key: 1.0 for key in OBSERVATION_KEYS},
        "reward_scales": {
            "daily_service_loss_p95": 1.0,
            "daily_backlog_age_p95": 1.0,
        },
    }
