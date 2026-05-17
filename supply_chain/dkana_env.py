from __future__ import annotations

from collections import deque
from typing import Any, Sequence

import gymnasium as gym
import numpy as np

from .config import CAPACITY_BY_SHIFTS, INVENTORY_BUFFERS, OPERATIONS
from .dkana import (
    RELATION_MODES,
    build_dkana_config_fields,
    build_enumeration_map,
    build_mfsc_relational_state,
    build_prefixed_variable_names,
)
from .external_env_interface import (
    ACTION_FIELDS,
    ACTION_FIELDS_TRACK_B_V1,
    STATE_CONSTRAINT_FIELDS,
    THESIS_DECISION_ACTION_FIELDS,
    THESIS_DECISION_OBSERVATION_FIELDS,
    THESIS_INVENTORY_PERIODS,
    build_shift_control_constraint_vector,
    build_shift_control_state_constraint_vector,
    get_observation_fields,
    get_shift_control_constraint_context,
    make_thesis_aligned_training_env,
    make_track_b_env,
)


class DKANAContextEnvWrapper(gym.Wrapper):
    """
    Add the DKANA SMS context window directly to env ``info``.

    The wrapped env still returns the normal flat observation for PPO/SB3
    compatibility. DKANA consumers can read ``info["dkana_row_matrices"]``,
    ``info["dkana_config_context"]``, and ``info["dkana_time_mask"]`` on every
    reset/step without maintaining a separate adapter.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        window_size: int = 12,
        relation_mode: str = "temporal_delta",
        observation_fields: Sequence[str] | None = None,
        state_constraint_fields: Sequence[str] = STATE_CONSTRAINT_FIELDS,
        action_dim: int | None = None,
        include_prev_reward: bool = False,
    ) -> None:
        super().__init__(env)
        if window_size <= 0:
            raise ValueError("window_size must be > 0.")
        if relation_mode not in RELATION_MODES:
            raise ValueError(f"relation_mode must be one of {RELATION_MODES}.")

        inferred_observation_version = str(getattr(env, "observation_version", "v7"))
        inferred_action_contract = str(getattr(env, "action_contract", "track_b_v1"))
        if action_dim is None:
            action_dim = (
                len(ACTION_FIELDS_TRACK_B_V1)
                if inferred_action_contract == "track_b_v1"
                else len(ACTION_FIELDS)
            )
        if action_dim not in (len(ACTION_FIELDS), len(ACTION_FIELDS_TRACK_B_V1)):
            raise ValueError("Unsupported action_dim for DKANA context wrapper.")

        self.window_size = int(window_size)
        self.relation_mode = relation_mode
        self.observation_fields = tuple(
            observation_fields or get_observation_fields(inferred_observation_version)
        )
        self.state_constraint_fields = tuple(state_constraint_fields)
        self.action_dim = int(action_dim)
        self.include_prev_reward = bool(include_prev_reward)
        self.config_fields = build_dkana_config_fields(
            self.action_dim,
            include_prev_reward=self.include_prev_reward,
        )
        self.constraint_context_vector = build_shift_control_constraint_vector(
            get_shift_control_constraint_context()
        ).astype(np.float32)
        self.variable_names = build_prefixed_variable_names(
            self.observation_fields,
            self.state_constraint_fields,
        )
        self.enumeration_map = build_enumeration_map(self.variable_names)
        self.relation_to_index = dict(self.enumeration_map.relation_to_index)

        self._rows: deque[np.ndarray] = deque(maxlen=self.window_size)
        self._configs: deque[np.ndarray] = deque(maxlen=self.window_size)
        self._previous_observation: np.ndarray | None = None
        self._previous_state_constraint_vector: np.ndarray | None = None

    def _reset_context(self) -> None:
        self._rows.clear()
        self._configs.clear()
        self._previous_observation = None
        self._previous_state_constraint_vector = None

    def _state_vector_from_info(self, info: dict[str, Any]) -> np.ndarray:
        state_context = info.get("state_constraint_context")
        if not isinstance(state_context, dict):
            raise ValueError("DKANA env requires info['state_constraint_context'].")
        return build_shift_control_state_constraint_vector(state_context).astype(
            np.float32
        )

    def _action_vector_from_payload(
        self,
        action: Any | None,
        info: dict[str, Any] | None = None,
    ) -> np.ndarray:
        payload = None if info is None else info.get("clipped_action")
        if payload is None:
            payload = action
        if isinstance(payload, np.ndarray | list | tuple):
            action_array = np.asarray(payload, dtype=np.float32)
            if action_array.shape == (self.action_dim,):
                return np.clip(action_array, -1.0, 1.0).astype(np.float32)
        return np.zeros(self.action_dim, dtype=np.float32)

    def _append_state(
        self,
        obs: np.ndarray,
        info: dict[str, Any],
        *,
        previous_action: np.ndarray,
        previous_reward: float = 0.0,
    ) -> None:
        obs_array = np.asarray(obs, dtype=np.float32)
        state_vector = self._state_vector_from_info(info)
        row_matrix = build_mfsc_relational_state(
            obs_array,
            state_vector,
            observation_fields=self.observation_fields,
            state_constraint_fields=self.state_constraint_fields,
            enumeration_map=self.enumeration_map,
            previous_observation=self._previous_observation,
            previous_state_constraint_vector=self._previous_state_constraint_vector,
            relation_mode=self.relation_mode,
        )
        config_parts = [self.constraint_context_vector, previous_action]
        if self.include_prev_reward:
            config_parts.append(
                np.asarray([float(previous_reward)], dtype=np.float32),
            )
        config_context = np.concatenate(config_parts, axis=0).astype(np.float32)
        self._rows.append(row_matrix)
        self._configs.append(config_context)
        self._previous_observation = obs_array.copy()
        self._previous_state_constraint_vector = state_vector.copy()

    def _window_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self._rows:
            raise RuntimeError("No DKANA rows available.")
        row_shape = self._rows[0].shape
        config_dim = self._configs[0].shape[0]
        row_window = np.zeros(
            (self.window_size, row_shape[0], row_shape[1]),
            dtype=np.float32,
        )
        config_window = np.zeros((self.window_size, config_dim), dtype=np.float32)
        time_mask = np.zeros((self.window_size,), dtype=bool)
        valid_count = len(self._rows)
        pad_length = self.window_size - valid_count
        row_window[pad_length:] = np.stack(list(self._rows), axis=0)
        config_window[pad_length:] = np.stack(list(self._configs), axis=0)
        time_mask[pad_length:] = True
        return row_window, config_window, time_mask

    def _attach_dkana_info(self, info: dict[str, Any]) -> dict[str, Any]:
        row_window, config_window, time_mask = self._window_arrays()
        dkana_context = {
            "row_matrices": row_window,
            "config_context": config_window,
            "time_mask": time_mask,
            "window_size": self.window_size,
            "relation_mode": self.relation_mode,
            "relation_to_index": self.relation_to_index,
            "variable_names": self.variable_names,
            "config_fields": self.config_fields,
            "include_prev_reward": self.include_prev_reward,
        }
        enriched_info = dict(info)
        enriched_info["dkana_context"] = dkana_context
        enriched_info["dkana_row_matrices"] = row_window
        enriched_info["dkana_config_context"] = config_window
        enriched_info["dkana_time_mask"] = time_mask
        return enriched_info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        self._reset_context()
        self._append_state(
            obs,
            info,
            previous_action=np.zeros(self.action_dim, dtype=np.float32),
            previous_reward=0.0,
        )
        enriched_info = dict(info)
        enriched_info["previous_reward"] = 0.0
        return obs, self._attach_dkana_info(enriched_info)

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        previous_action = self._action_vector_from_payload(action, info)
        self._append_state(
            obs,
            info,
            previous_action=previous_action,
            previous_reward=float(reward),
        )
        enriched_info = dict(info)
        enriched_info["previous_reward"] = float(reward)
        return (
            obs,
            reward,
            terminated,
            truncated,
            self._attach_dkana_info(enriched_info),
        )


def make_dkana_track_b_env(
    *,
    dkana_window_size: int = 12,
    relation_mode: str = "temporal_delta",
    include_prev_reward: bool = False,
    **env_overrides: Any,
) -> DKANAContextEnvWrapper:
    """Build Track B with DKANA context windows emitted by the environment."""
    env = make_track_b_env(**env_overrides)
    return DKANAContextEnvWrapper(
        env,
        window_size=dkana_window_size,
        relation_mode=relation_mode,
        include_prev_reward=include_prev_reward,
        observation_fields=get_observation_fields(
            str(env_overrides.get("observation_version", "v7"))
        ),
        action_dim=len(ACTION_FIELDS_TRACK_B_V1),
    )


class DKANAThesisFaithfulDecisionEnvWrapper(gym.Wrapper):
    """
    Expose Garrido-Rios thesis decision variables as a 18D DKANA contract.

    The action vector is ordered as Table 6.16 inventory-buffer choices
    (Op3, Op5, Op9 crossed with I168,1...I1344,1) followed by Table 6.20
    capacity choices (S1, S2, S3). The observation mirrors the realized 18D
    decision vector and appends the latest reward.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.action_fields = THESIS_DECISION_ACTION_FIELDS
        self.observation_fields = THESIS_DECISION_OBSERVATION_FIELDS
        self.action_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(len(self.action_fields),),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=-1_000_000.0,
            high=1_000_000.0,
            shape=(len(self.observation_fields),),
            dtype=np.float32,
        )
        self._realized_decision = np.zeros(len(self.action_fields), dtype=np.float32)
        self._realized_decision[-3] = 1.0
        self._last_reward = 0.0

    def _build_observation(self) -> np.ndarray:
        return np.concatenate(
            [
                self._realized_decision.astype(np.float32),
                np.asarray([self._last_reward], dtype=np.float32),
            ],
            axis=0,
        )

    def _select_inventory_period(self, action: np.ndarray) -> int | None:
        inventory_scores = action[:15].reshape(3, 5)
        period_scores = inventory_scores.mean(axis=0)
        if float(period_scores.max()) <= 0.0:
            return None
        return int(THESIS_INVENTORY_PERIODS[int(period_scores.argmax())])

    @staticmethod
    def _select_shifts(action: np.ndarray) -> int:
        return int(action[15:18].argmax()) + 1

    def _set_inventory_targets(self, period: int | None) -> dict[str, float]:
        base_env = self.unwrapped
        sim = getattr(base_env, "sim", None)
        if sim is None:
            return {}
        if period is None:
            sim.inventory_buffer_targets = {}
            return {}

        targets = {
            key: float(value) for key, value in INVENTORY_BUFFERS[int(period)].items()
        }
        sim.inventory_buffer_targets = dict(targets)
        sim.inventory_replenishment_period = float(period)
        for key, target in targets.items():
            sim._top_up_inventory_buffer(key, target)
        return targets

    def _realized_vector(self, period: int | None, shifts: int) -> np.ndarray:
        realized = np.zeros(len(self.action_fields), dtype=np.float32)
        if period is not None:
            period_index = THESIS_INVENTORY_PERIODS.index(int(period))
            for node_index in range(3):
                realized[node_index * 5 + period_index] = 1.0
        realized[15 + shifts - 1] = 1.0
        return realized

    def _action_dict(self, shifts: int) -> dict[str, float | int]:
        base_env = self.unwrapped
        sim = getattr(base_env, "sim", None)
        cap = CAPACITY_BY_SHIFTS[shifts]
        op9_q_min = float(OPERATIONS[9]["q"][0])
        op9_q_max = float(OPERATIONS[9]["q"][1])
        if sim is not None:
            op9_q_min = float(sim.params.get("op9_q_min", op9_q_min))
            op9_q_max = float(sim.params.get("op9_q_max", op9_q_max))
        return {
            "assembly_shifts": shifts,
            "op3_q": float(cap["op3_q"]),
            "op3_rop": float(OPERATIONS[3]["rop"]),
            "op9_q_min": op9_q_min,
            "op9_q_max": op9_q_max,
            "op9_rop": float(OPERATIONS[9]["rop"]),
            "batch_size": float(cap["op7_q"]),
        }

    def _attach_info(
        self,
        info: dict[str, Any],
        *,
        period: int | None,
        shifts: int,
        targets: dict[str, float],
    ) -> dict[str, Any]:
        enriched = dict(info)
        enriched["action_contract"] = "thesis_faithful_dkana_v1"
        enriched["observation_contract"] = "thesis_decision_reward_v1"
        enriched["thesis_decision_action_fields"] = list(self.action_fields)
        enriched["thesis_decision_observation_fields"] = list(self.observation_fields)
        enriched["thesis_decision"] = {
            "inventory_period_hours": None if period is None else float(period),
            "inventory_buffer_targets": dict(targets),
            "assembly_shifts": int(shifts),
        }
        return enriched

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        _, info = self.env.reset(seed=seed, options=options)
        self._last_reward = 0.0
        self._realized_decision = self._realized_vector(None, 1)
        return self._build_observation(), self._attach_info(
            info,
            period=None,
            shifts=1,
            targets={},
        )

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action_array = np.asarray(action, dtype=np.float32)
        if action_array.shape != (len(self.action_fields),):
            raise ValueError(
                f"Action must have shape ({len(self.action_fields)},), "
                f"got {action_array.shape}."
            )
        clipped = np.clip(action_array, 0.0, 1.0)
        period = self._select_inventory_period(clipped)
        shifts = self._select_shifts(clipped)
        targets = self._set_inventory_targets(period)
        self._realized_decision = self._realized_vector(period, shifts)

        _, reward, terminated, truncated, info = self.env.step(
            self._action_dict(shifts)
        )
        self._last_reward = float(reward)
        return (
            self._build_observation(),
            float(reward),
            terminated,
            truncated,
            self._attach_info(info, period=period, shifts=shifts, targets=targets),
        )


def make_dkana_thesis_faithful_env(
    **env_overrides: Any,
) -> DKANAThesisFaithfulDecisionEnvWrapper:
    """Build thesis-aligned Gym with the 18D/19D DKANA decision-vector contract."""
    env = make_thesis_aligned_training_env(**env_overrides)
    return DKANAThesisFaithfulDecisionEnvWrapper(env)
