from __future__ import annotations

from collections import deque
from typing import Any, Sequence

import gymnasium as gym
import numpy as np

from supply_chain.dkana import (
    RELATION_MODES,
    build_dkana_config_fields,
    build_enumeration_map,
    build_mfsc_relational_state,
    build_prefixed_variable_names,
)
from supply_chain.external_env_interface import (
    ACTION_FIELDS,
    ACTION_FIELDS_TRACK_B_V1,
    STATE_CONSTRAINT_FIELDS,
    build_shift_control_constraint_vector,
    build_shift_control_state_constraint_vector,
    get_observation_fields,
    get_shift_control_constraint_context,
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
        self.config_fields = build_dkana_config_fields(self.action_dim)
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
        config_context = np.concatenate(
            [self.constraint_context_vector, previous_action],
            axis=0,
        ).astype(np.float32)
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
        )
        return obs, self._attach_dkana_info(info)

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        previous_action = self._action_vector_from_payload(action, info)
        self._append_state(obs, info, previous_action=previous_action)
        return obs, reward, terminated, truncated, self._attach_dkana_info(info)


def make_dkana_track_b_env(
    *,
    dkana_window_size: int = 12,
    relation_mode: str = "temporal_delta",
    **env_overrides: Any,
) -> DKANAContextEnvWrapper:
    """Build Track B with DKANA context windows emitted by the environment."""
    env = make_track_b_env(**env_overrides)
    return DKANAContextEnvWrapper(
        env,
        window_size=dkana_window_size,
        relation_mode=relation_mode,
        observation_fields=get_observation_fields(
            str(env_overrides.get("observation_version", "v7"))
        ),
        action_dim=len(ACTION_FIELDS_TRACK_B_V1),
    )
