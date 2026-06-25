from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from .config import (
    CAPACITY_BY_SHIFTS,
    INVENTORY_BUFFERS,
    OPERATIONS,
    THESIS_REPLICATION_DOWNSTREAM_Q_SOURCE,
    THESIS_ROBUSTNESS_DOWNSTREAM_Q_SOURCE,
    TRACK_A_TRAINING_DOWNSTREAM_Q_SOURCE,
)
from .external_env_interface import (
    THESIS_DECISION_ACTION_FIELDS,
    THESIS_FACTORIZED_ACTION_FIELDS,
    THESIS_INVENTORY_PERIODS,
    make_thesis_aligned_training_env,
)


class ThesisFactorizedTrackAEnv(gym.Wrapper):
    """Expose Garrido-Rios Track A decisions without importing DKANA/torch.

    The trainable action surface is `MultiDiscrete([6, 3])`:
    inventory level 0 means no strategic buffer, levels 1..5 map to the thesis
    inventory periods, and shift index 0..2 maps to S1..S3.
    """

    action_contract = "track_a_thesis_factorized_v1"
    action_space_mode = "thesis_factorized"
    inventory_period_mode = "thesis_strict"

    def __init__(self, env: gym.Env, *, initial_action: Any | None = None) -> None:
        super().__init__(env)
        self.initial_action = initial_action
        self.action_fields = THESIS_DECISION_ACTION_FIELDS
        self.factorized_action_fields = THESIS_FACTORIZED_ACTION_FIELDS
        self.action_space = gym.spaces.MultiDiscrete([6, 3])
        self.observation_space = env.observation_space
        self._realized_decision = self._realized_vector({}, 1)

    @staticmethod
    def _validate_factorized_action(action: Any) -> np.ndarray:
        action_array = np.asarray(action, dtype=np.int64)
        if action_array.shape != (2,):
            raise ValueError(f"Action must have shape (2,), got {action_array.shape}.")
        if np.any(action_array < 0) or np.any(action_array > np.asarray([5, 2])):
            raise ValueError("Thesis-factorized action values are out of bounds.")
        return action_array

    @classmethod
    def decode_thesis_factorized_action(
        cls, action: Any
    ) -> tuple[dict[str, int], int, np.ndarray]:
        discrete = cls._validate_factorized_action(action)
        periods_by_node: dict[str, int] = {}
        if int(discrete[0]) > 0:
            period = int(THESIS_INVENTORY_PERIODS[int(discrete[0]) - 1])
            periods_by_node = {
                node_name: period for node_name in ("op3", "op5", "op9")
            }
        shifts = int(discrete[1]) + 1
        return periods_by_node, shifts, cls._realized_vector(periods_by_node, shifts)

    @staticmethod
    def thesis_buffer_targets(periods_by_node: dict[str, int]) -> dict[str, float]:
        key_by_node = {
            "op3": "op3_rm",
            "op5": "op5_rm",
            "op9": "op9_rations",
        }
        targets: dict[str, float] = {}
        for node_name, period in periods_by_node.items():
            target_key = key_by_node[node_name]
            targets[target_key] = float(INVENTORY_BUFFERS[int(period)][target_key])
        return targets

    @staticmethod
    def _realized_vector(periods_by_node: dict[str, int], shifts: int) -> np.ndarray:
        realized = np.zeros(len(THESIS_DECISION_ACTION_FIELDS), dtype=np.float32)
        for node_index, node_name in enumerate(("op3", "op5", "op9")):
            period = periods_by_node.get(node_name)
            if period is None:
                continue
            period_index = THESIS_INVENTORY_PERIODS.index(int(period))
            realized[node_index * 5 + period_index] = 1.0
        realized[15 + int(shifts) - 1] = 1.0
        return realized

    def _set_inventory_targets(
        self, periods_by_node: dict[str, int]
    ) -> dict[str, float]:
        sim = getattr(self.unwrapped, "sim", None)
        if sim is None:
            return {}
        if not periods_by_node:
            sim.inventory_buffer_targets = {}
            sim.inventory_replenishment_period = None
            return {}

        targets = self.thesis_buffer_targets(periods_by_node)
        if hasattr(sim, "_normalize_inventory_buffer_targets"):
            internal_targets = sim._normalize_inventory_buffer_targets(targets)
        else:
            internal_targets = dict(targets)
        sim.inventory_buffer_targets = dict(internal_targets)
        sim.inventory_replenishment_period = float(min(periods_by_node.values()))
        for key, target in internal_targets.items():
            sim._top_up_inventory_buffer(key, float(target))
        return targets

    def _action_dict(self, shifts: int) -> dict[str, float | int]:
        sim = getattr(self.unwrapped, "sim", None)
        cap = CAPACITY_BY_SHIFTS[int(shifts)]
        op9_q_min = float(OPERATIONS[9]["q"][0])
        op9_q_max = float(OPERATIONS[9]["q"][1])
        if sim is not None:
            op9_q_min = float(sim.params.get("op9_q_min", op9_q_min))
            op9_q_max = float(sim.params.get("op9_q_max", op9_q_max))
        return {
            "assembly_shifts": int(shifts),
            "op3_q": float(cap["op3_q"]),
            "op3_rop": float(OPERATIONS[3]["rop"]),
            "op9_q_min": op9_q_min,
            "op9_q_max": op9_q_max,
            "op9_rop": float(OPERATIONS[9]["rop"]),
            "batch_size": float(cap["op7_q"]),
        }

    def _decision_payload(
        self,
        *,
        periods_by_node: dict[str, int],
        shifts: int,
        targets: dict[str, float],
        action: Any,
    ) -> dict[str, Any]:
        unique_periods = set(periods_by_node.values())
        common_period = unique_periods.pop() if len(unique_periods) == 1 else None
        discrete = self._validate_factorized_action(action)
        return {
            "inventory_period_hours": (
                None if common_period is None else float(common_period)
            ),
            "inventory_period_hours_by_node": {
                node_name: float(period)
                for node_name, period in periods_by_node.items()
            },
            "inventory_buffer_targets": dict(targets),
            "assembly_shifts": int(shifts),
            "common_inventory_level": int(discrete[0]),
            "assembly_shift_level": int(discrete[1]),
        }

    def _attach_info(
        self,
        info: dict[str, Any],
        *,
        periods_by_node: dict[str, int],
        shifts: int,
        targets: dict[str, float],
        action: Any,
        phase: str,
    ) -> dict[str, Any]:
        enriched = dict(info)
        base_env = getattr(self.env, "unwrapped", self.env)
        downstream_q_source = str(
            getattr(base_env, "downstream_q_source", TRACK_A_TRAINING_DOWNSTREAM_Q_SOURCE)
        )
        if downstream_q_source == THESIS_REPLICATION_DOWNSTREAM_Q_SOURCE:
            downstream_q_lane = "thesis_replication_training"
        elif downstream_q_source == THESIS_ROBUSTNESS_DOWNSTREAM_Q_SOURCE:
            downstream_q_lane = "robustness_sensitivity"
        else:
            downstream_q_lane = "custom"
        decision = self._decision_payload(
            periods_by_node=periods_by_node,
            shifts=shifts,
            targets=targets,
            action=action,
        )
        enriched["action_contract"] = self.action_contract
        enriched["action_space_mode"] = self.action_space_mode
        enriched["inventory_period_mode"] = self.inventory_period_mode
        enriched["downstream_q_source"] = downstream_q_source
        enriched["downstream_q_lane"] = downstream_q_lane
        enriched["thesis_decision_action_fields"] = list(self.action_fields)
        enriched["thesis_factorized_action_fields"] = list(
            self.factorized_action_fields
        )
        enriched["thesis_decision"] = decision
        sim = getattr(base_env, "sim", None)
        enriched["thesis_decision"]["inventory_buffer_targets_internal"] = dict(
            getattr(sim, "inventory_buffer_targets", {})
        )
        enriched["thesis_factorized_action"] = (
            self._validate_factorized_action(action).astype(int).tolist()
        )
        enriched["thesis_decision_action_vector"] = (
            self._realized_decision.astype(float).tolist()
        )
        enriched["action_phase"] = phase
        if phase == "reset":
            enriched["initial_decision"] = {
                **decision,
                "applied_before_warmup": bool(periods_by_node or shifts != 1),
            }
        else:
            enriched["weekly_decision"] = decision
        return enriched

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        reset_options = dict(options or {})
        initial_action = reset_options.pop("initial_action", self.initial_action)
        if initial_action is None:
            initial_action = np.asarray([0, 0], dtype=np.int64)
        periods_by_node, shifts, realized_action = (
            self.decode_thesis_factorized_action(initial_action)
        )
        targets = self.thesis_buffer_targets(periods_by_node)
        self._realized_decision = realized_action
        reset_options["initial_buffers"] = dict(targets)
        reset_options["initial_shifts"] = int(shifts)
        reset_options["inventory_replenishment_period"] = (
            None if not periods_by_node else float(min(periods_by_node.values()))
        )
        obs, info = self.env.reset(seed=seed, options=reset_options)
        return obs, self._attach_info(
            info,
            periods_by_node=periods_by_node,
            shifts=shifts,
            targets=targets,
            action=initial_action,
            phase="reset",
        )

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        periods_by_node, shifts, realized_action = (
            self.decode_thesis_factorized_action(action)
        )
        targets = self._set_inventory_targets(periods_by_node)
        self._realized_decision = realized_action
        obs, reward, terminated, truncated, info = self.env.step(
            self._action_dict(shifts)
        )
        return (
            obs,
            float(reward),
            bool(terminated),
            bool(truncated),
            self._attach_info(
                info,
                periods_by_node=periods_by_node,
                shifts=shifts,
                targets=targets,
                action=action,
                phase="weekly_decision",
            ),
        )


class Discrete18TrackAEnv(gym.Wrapper):
    """Flatten the Track A thesis-factorized surface to `Discrete(18)`."""

    action_contract = "track_a_discrete18_v1"
    action_space_mode = "discrete_18"

    def __init__(self, env: ThesisFactorizedTrackAEnv) -> None:
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(18)
        self.observation_space = env.observation_space

    @staticmethod
    def decode_discrete_action(action: int) -> np.ndarray:
        action_int = int(action)
        if action_int < 0 or action_int >= 18:
            raise ValueError("Discrete Track A action must be in [0, 17].")
        return np.asarray([action_int // 3, action_int % 3], dtype=np.int64)

    @staticmethod
    def encode_discrete_action(level: int, shift_index: int) -> int:
        level_int = int(level)
        shift_int = int(shift_index)
        if level_int < 0 or level_int > 5 or shift_int < 0 or shift_int > 2:
            raise ValueError("Expected level in [0, 5] and shift_index in [0, 2].")
        return level_int * 3 + shift_int

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        reset_options = dict(options or {})
        if "initial_discrete_action" in reset_options:
            reset_options["initial_action"] = self.decode_discrete_action(
                int(reset_options.pop("initial_discrete_action"))
            )
        elif isinstance(reset_options.get("initial_action"), (int, np.integer)):
            reset_options["initial_action"] = self.decode_discrete_action(
                int(reset_options["initial_action"])
            )
        obs, info = self.env.reset(seed=seed, options=reset_options)
        enriched = dict(info)
        enriched["action_contract"] = self.action_contract
        enriched["action_space_mode"] = self.action_space_mode
        return obs, enriched

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        discrete_action = int(action)
        factorized_action = self.decode_discrete_action(discrete_action)
        obs, reward, terminated, truncated, info = self.env.step(factorized_action)
        enriched = dict(info)
        enriched["action_contract"] = self.action_contract
        enriched["action_space_mode"] = self.action_space_mode
        enriched["discrete_action"] = discrete_action
        enriched["thesis_factorized_action"] = factorized_action.astype(int).tolist()
        return obs, float(reward), terminated, truncated, enriched


def make_thesis_factorized_track_a_env(**env_overrides: Any) -> ThesisFactorizedTrackAEnv:
    """Build the torch-free Track A `MultiDiscrete([6, 3])` wrapper."""
    initial_action = env_overrides.pop("initial_action", None)
    action_space_mode = str(env_overrides.pop("action_space_mode", "thesis_factorized"))
    if action_space_mode != "thesis_factorized":
        raise ValueError(
            "Track A factorized wrapper expects "
            "action_space_mode='thesis_factorized'."
        )
    env_overrides.pop("inventory_period_mode", None)
    env_overrides.pop("observation_mode", None)
    env_overrides.pop("learn_initial_decision", None)
    env_overrides.setdefault(
        "downstream_q_source",
        TRACK_A_TRAINING_DOWNSTREAM_Q_SOURCE,
    )
    env = make_thesis_aligned_training_env(**env_overrides)
    return ThesisFactorizedTrackAEnv(env, initial_action=initial_action)


def make_discrete18_track_a_env(**env_overrides: Any) -> Discrete18TrackAEnv:
    """Build the torch-free Track A `Discrete(18)` wrapper for DQN-style methods."""
    initial_action = env_overrides.pop("initial_action", None)
    if isinstance(initial_action, (int, np.integer)):
        initial_action = Discrete18TrackAEnv.decode_discrete_action(int(initial_action))
    factorized = make_thesis_factorized_track_a_env(
        **env_overrides,
        initial_action=initial_action,
    )
    return Discrete18TrackAEnv(factorized)
