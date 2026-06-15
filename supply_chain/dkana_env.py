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
    SDM_HISTORY_FIELDS,
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

    def __init__(
        self,
        env: gym.Env,
        *,
        observation_mode: str = "decision_reward",
        action_space_mode: str = "onehot_18d",
        inventory_period_mode: str = "thesis_strict",
        initial_action: Any | None = None,
        learn_initial_decision: bool = False,
    ) -> None:
        super().__init__(env)
        if observation_mode not in (
            "decision_reward",
            "env_reward",
            "env_state_reward",
            "env_sdm_history_reward",
        ):
            raise ValueError(
                "observation_mode must be 'decision_reward', 'env_reward', "
                "'env_state_reward', or 'env_sdm_history_reward'."
            )
        if action_space_mode not in ("onehot_18d", "factorized", "thesis_factorized"):
            raise ValueError(
                "action_space_mode must be 'onehot_18d', 'factorized', "
                "or 'thesis_factorized'."
            )
        if inventory_period_mode not in ("thesis_strict", "per_node"):
            raise ValueError(
                "inventory_period_mode must be 'thesis_strict' or 'per_node'."
            )
        self.action_fields = THESIS_DECISION_ACTION_FIELDS
        self.observation_mode = observation_mode
        self.action_space_mode = action_space_mode
        self.inventory_period_mode = inventory_period_mode
        self.initial_action = initial_action
        self.learn_initial_decision = bool(learn_initial_decision)
        self._awaiting_initial_decision = False
        self._pending_reset_seed: int | None = None
        self._pending_reset_options: dict[str, Any] = {}
        self.base_observation_version = str(getattr(env, "observation_version", "v4"))
        if observation_mode == "decision_reward":
            self.observation_fields = THESIS_DECISION_OBSERVATION_FIELDS
            obs_shape = (len(self.observation_fields),)
            obs_low = -1_000_000.0
            obs_high = 1_000_000.0
        else:
            base_fields = get_observation_fields(self.base_observation_version)
            self.observation_fields = base_fields + ("reward",)
            obs_shape = (len(self.observation_fields),)
            obs_low = 0.0
            obs_high = 20.0
        if observation_mode == "env_state_reward":
            base_fields = get_observation_fields(self.base_observation_version)
            self.observation_fields = (
                base_fields + STATE_CONSTRAINT_FIELDS + ("reward",)
            )
            obs_shape = (len(self.observation_fields),)
            obs_low = -1_000_000.0
            obs_high = 1_000_000.0
        if observation_mode == "env_sdm_history_reward":
            base_fields = get_observation_fields(self.base_observation_version)
            self.observation_fields = base_fields + SDM_HISTORY_FIELDS + ("reward",)
            obs_shape = (len(self.observation_fields),)
            obs_low = -1_000_000.0
            obs_high = 1_000_000.0
        if action_space_mode == "thesis_factorized":
            # Direct thesis decision variables: common I_{t,S} level plus S.
            # 0 means no strategic inventory buffer; 1..5 map to I168,1..I1344,1.
            # The final categorical variable maps 0/1/2 to S1/S2/S3.
            self.action_space = gym.spaces.MultiDiscrete([6, 3])
        elif action_space_mode == "factorized":
            # 0 means no strategic buffer for a node; 1..5 map to I168,1..I1344,1.
            # The final categorical variable maps 0/1/2 to S1/S2/S3.
            self.action_space = gym.spaces.MultiDiscrete([6, 6, 6, 3])
        else:
            self.action_space = gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(len(self.action_fields),),
                dtype=np.float32,
            )
        self.observation_space = gym.spaces.Box(
            low=obs_low,
            high=obs_high,
            shape=obs_shape,
            dtype=np.float32,
        )
        self._realized_decision = np.zeros(len(self.action_fields), dtype=np.float32)
        self._realized_decision[-3] = 1.0
        self._latest_env_observation = np.zeros(
            len(get_observation_fields(self.base_observation_version)), dtype=np.float32
        )
        self._latest_state_constraint_vector = np.zeros(
            len(STATE_CONSTRAINT_FIELDS), dtype=np.float32
        )
        self._latest_sdm_history_vector = np.zeros(
            len(SDM_HISTORY_FIELDS), dtype=np.float32
        )
        self._last_reward = 0.0

    def _empty_phase_info(self, phase: str) -> dict[str, Any]:
        info = {
            "action_phase": phase,
            "action_contract": "thesis_faithful_dkana_v1",
            "observation_contract": (
                "thesis_decision_reward_v1"
                if self.observation_mode == "decision_reward"
                else f"{self.observation_mode}_{self.base_observation_version}"
            ),
            "action_space_mode": self.action_space_mode,
            "inventory_period_mode": self.inventory_period_mode,
            "thesis_decision_action_fields": list(self.action_fields),
            "thesis_decision_observation_fields": list(self.observation_fields),
            "thesis_decision": {
                "inventory_period_hours": None,
                "inventory_period_hours_by_node": {},
                "inventory_buffer_targets": {},
                "assembly_shifts": 1,
            },
        }
        info["initial_decision"] = {
            **info["thesis_decision"],
            "applied_before_warmup": False,
        }
        return info

    def _build_observation(self) -> np.ndarray:
        if self.observation_mode == "env_sdm_history_reward":
            return np.concatenate(
                [
                    self._latest_env_observation.astype(np.float32),
                    self._latest_sdm_history_vector.astype(np.float32),
                    np.asarray([self._last_reward], dtype=np.float32),
                ],
                axis=0,
            )
        if self.observation_mode == "env_state_reward":
            return np.concatenate(
                [
                    self._latest_env_observation.astype(np.float32),
                    self._latest_state_constraint_vector.astype(np.float32),
                    np.asarray([self._last_reward], dtype=np.float32),
                ],
                axis=0,
            )
        if self.observation_mode == "env_reward":
            return np.concatenate(
                [
                    self._latest_env_observation.astype(np.float32),
                    np.asarray([self._last_reward], dtype=np.float32),
                ],
                axis=0,
            )
        return np.concatenate(
            [
                self._realized_decision.astype(np.float32),
                np.asarray([self._last_reward], dtype=np.float32),
            ],
            axis=0,
        )

    def _update_state_constraint_vector(self, info: dict[str, Any]) -> None:
        state_context = info.get("state_constraint_context")
        if isinstance(state_context, dict):
            self._latest_state_constraint_vector = (
                build_shift_control_state_constraint_vector(state_context).astype(
                    np.float32
                )
            )

    def _update_sdm_history_vector(self) -> None:
        sim = getattr(self.unwrapped, "sim", None)
        if sim is None or not hasattr(sim, "get_sdm_history_context"):
            return
        context = sim.get_sdm_history_context()
        self._latest_sdm_history_vector = np.asarray(
            [float(context.get(field, 0.0)) for field in SDM_HISTORY_FIELDS],
            dtype=np.float32,
        )

    def _select_inventory_period_by_node(self, action: np.ndarray) -> dict[str, int]:
        inventory_scores = action[:15].reshape(3, 5)
        selected: dict[str, int] = {}
        for node_index, node_name in enumerate(("op3", "op5", "op9")):
            node_scores = inventory_scores[node_index]
            if float(node_scores.max()) > 0.0:
                selected[node_name] = int(
                    THESIS_INVENTORY_PERIODS[int(node_scores.argmax())]
                )
        return self._normalize_periods_by_node(selected)

    def _normalize_periods_by_node(
        self, periods_by_node: dict[str, int]
    ) -> dict[str, int]:
        if self.inventory_period_mode == "per_node" or not periods_by_node:
            return dict(periods_by_node)
        counts = {
            period: list(periods_by_node.values()).count(period)
            for period in set(periods_by_node.values())
        }
        chosen_period = max(
            counts,
            key=lambda period: (counts[period], THESIS_INVENTORY_PERIODS.index(period)),
        )
        return {node_name: int(chosen_period) for node_name in ("op3", "op5", "op9")}

    @staticmethod
    def _select_shifts(action: np.ndarray) -> int:
        return int(action[15:18].argmax()) + 1

    def _decode_action(self, action: Any) -> tuple[dict[str, int], int, np.ndarray]:
        action_array = np.asarray(action)
        if self.action_space_mode == "thesis_factorized":
            if action_array.shape != (2,):
                raise ValueError(
                    f"Action must have shape (2,), got {action_array.shape}."
                )
            discrete = np.asarray(action_array, dtype=np.int64)
            if np.any(discrete < 0) or np.any(discrete > np.asarray([5, 2])):
                raise ValueError("Thesis-factorized action values are out of bounds.")
            periods_by_node = {}
            if int(discrete[0]) > 0:
                period = int(THESIS_INVENTORY_PERIODS[int(discrete[0]) - 1])
                periods_by_node = {
                    node_name: period for node_name in ("op3", "op5", "op9")
                }
            shifts = int(discrete[1]) + 1
            return (
                periods_by_node,
                shifts,
                self._realized_vector(periods_by_node, shifts),
            )

        if self.action_space_mode == "factorized":
            if action_array.shape != (4,):
                raise ValueError(
                    f"Action must have shape (4,), got {action_array.shape}."
                )
            discrete = np.asarray(action_array, dtype=np.int64)
            if np.any(discrete < 0) or np.any(discrete > np.asarray([5, 5, 5, 2])):
                raise ValueError("Factorized action values are out of bounds.")
            periods_by_node = {}
            for node_name, level in zip(
                ("op3", "op5", "op9"), discrete[:3], strict=True
            ):
                if int(level) > 0:
                    periods_by_node[node_name] = int(
                        THESIS_INVENTORY_PERIODS[int(level) - 1]
                    )
            periods_by_node = self._normalize_periods_by_node(periods_by_node)
            shifts = int(discrete[3]) + 1
            return (
                periods_by_node,
                shifts,
                self._realized_vector(periods_by_node, shifts),
            )

        action_array = np.asarray(action, dtype=np.float32)
        if action_array.shape != (len(self.action_fields),):
            raise ValueError(
                f"Action must have shape ({len(self.action_fields)},), "
                f"got {action_array.shape}."
            )
        clipped = np.clip(action_array, 0.0, 1.0)
        periods_by_node = self._select_inventory_period_by_node(clipped)
        shifts = self._select_shifts(clipped)
        return periods_by_node, shifts, self._realized_vector(periods_by_node, shifts)

    @staticmethod
    def _buffer_targets_from_periods(
        periods_by_node: dict[str, int],
    ) -> dict[str, float]:
        key_by_node = {
            "op3": "op3_rm",
            "op5": "op5_rm",
            "op9": "op9_rations",
        }
        targets = {}
        for node_name, period in periods_by_node.items():
            target_key = key_by_node[node_name]
            targets[target_key] = float(INVENTORY_BUFFERS[int(period)][target_key])
        return targets

    def _set_inventory_targets(
        self, periods_by_node: dict[str, int]
    ) -> dict[str, float]:
        base_env = self.unwrapped
        sim = getattr(base_env, "sim", None)
        if sim is None:
            return {}
        if not periods_by_node:
            sim.inventory_buffer_targets = {}
            return {}

        targets = self._buffer_targets_from_periods(periods_by_node)
        if hasattr(sim, "_normalize_inventory_buffer_targets"):
            internal_targets = sim._normalize_inventory_buffer_targets(targets)
        else:
            internal_targets = dict(targets)
        sim.inventory_buffer_targets = dict(internal_targets)
        sim.inventory_replenishment_period = float(min(periods_by_node.values()))
        for key, target in internal_targets.items():
            sim._top_up_inventory_buffer(key, target)
        return internal_targets

    def _realized_vector(
        self, periods_by_node: dict[str, int], shifts: int
    ) -> np.ndarray:
        realized = np.zeros(len(self.action_fields), dtype=np.float32)
        for node_index, node_name in enumerate(("op3", "op5", "op9")):
            period = periods_by_node.get(node_name)
            if period is None:
                continue
            period_index = THESIS_INVENTORY_PERIODS.index(int(period))
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
        periods_by_node: dict[str, int],
        shifts: int,
        targets: dict[str, float],
    ) -> dict[str, Any]:
        enriched = dict(info)
        enriched["action_contract"] = "thesis_faithful_dkana_v1"
        enriched["observation_contract"] = (
            "thesis_decision_reward_v1"
            if self.observation_mode == "decision_reward"
            else f"{self.observation_mode}_{self.base_observation_version}"
        )
        enriched["action_space_mode"] = self.action_space_mode
        enriched["inventory_period_mode"] = self.inventory_period_mode
        enriched["thesis_decision_action_fields"] = list(self.action_fields)
        enriched["thesis_decision_observation_fields"] = list(self.observation_fields)
        unique_periods = set(periods_by_node.values())
        common_period = unique_periods.pop() if len(unique_periods) == 1 else None
        enriched["thesis_decision"] = {
            "inventory_period_hours": (
                None if common_period is None else float(common_period)
            ),
            "inventory_period_hours_by_node": {
                node_name: float(period)
                for node_name, period in periods_by_node.items()
            },
            "inventory_buffer_targets": dict(targets),
            "assembly_shifts": int(shifts),
        }
        return enriched

    def _reset_underlying_with_initial_action(
        self,
        *,
        seed: int | None,
        wrapper_options: dict[str, Any],
        initial_action: Any | None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        initial_periods_by_node: dict[str, int] = {}
        initial_shifts = 1
        initial_targets: dict[str, float] = {}
        if initial_action is not None:
            initial_periods_by_node, initial_shifts, realized_action = (
                self._decode_action(initial_action)
            )
            initial_targets = self._buffer_targets_from_periods(initial_periods_by_node)
            self._realized_decision = realized_action
            wrapper_options["initial_buffers"] = dict(initial_targets)
            wrapper_options["initial_shifts"] = int(initial_shifts)
            wrapper_options["inventory_replenishment_period"] = (
                None
                if not initial_periods_by_node
                else float(min(initial_periods_by_node.values()))
            )
        else:
            self._realized_decision = self._realized_vector({}, 1)

        obs, info = self.env.reset(seed=seed, options=wrapper_options)
        self._latest_env_observation = np.asarray(obs, dtype=np.float32)
        self._update_state_constraint_vector(info)
        self._update_sdm_history_vector()
        self._last_reward = 0.0
        enriched_info = self._attach_info(
            info,
            periods_by_node=initial_periods_by_node,
            shifts=initial_shifts,
            targets=initial_targets,
        )
        enriched_info["action_phase"] = "weekly_decision"
        enriched_info["initial_decision"] = dict(enriched_info["thesis_decision"])
        enriched_info["initial_decision"]["applied_before_warmup"] = bool(
            initial_action is not None
        )
        enriched_info["thesis_decision_action_vector"] = (
            self._realized_decision.tolist()
        )
        return self._build_observation(), enriched_info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        wrapper_options = dict(options or {})
        initial_action = wrapper_options.pop("initial_action", self.initial_action)
        self._awaiting_initial_decision = (
            self.learn_initial_decision and initial_action is None
        )
        if self._awaiting_initial_decision:
            self._pending_reset_seed = seed
            self._pending_reset_options = wrapper_options
            self._realized_decision = self._realized_vector({}, 1)
            self._latest_env_observation = np.zeros_like(self._latest_env_observation)
            self._latest_state_constraint_vector = np.zeros_like(
                self._latest_state_constraint_vector
            )
            self._latest_sdm_history_vector = np.zeros_like(
                self._latest_sdm_history_vector
            )
            self._last_reward = 0.0
            return self._build_observation(), self._empty_phase_info("initial_decision")

        return self._reset_underlying_with_initial_action(
            seed=seed,
            wrapper_options=wrapper_options,
            initial_action=initial_action,
        )

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._awaiting_initial_decision:
            self._awaiting_initial_decision = False
            obs, info = self._reset_underlying_with_initial_action(
                seed=self._pending_reset_seed,
                wrapper_options=dict(self._pending_reset_options),
                initial_action=action,
            )
            info["action_phase"] = "initial_decision"
            return obs, 0.0, False, False, info

        periods_by_node, shifts, realized_action = self._decode_action(action)
        targets = self._set_inventory_targets(periods_by_node)
        self._realized_decision = realized_action

        obs, reward, terminated, truncated, info = self.env.step(
            self._action_dict(shifts)
        )
        self._latest_env_observation = np.asarray(obs, dtype=np.float32)
        self._update_state_constraint_vector(info)
        self._update_sdm_history_vector()
        self._last_reward = float(reward)
        enriched_info = self._attach_info(
            info, periods_by_node=periods_by_node, shifts=shifts, targets=targets
        )
        enriched_info["action_phase"] = "weekly_decision"
        enriched_info["weekly_decision"] = dict(enriched_info["thesis_decision"])
        enriched_info["thesis_decision_action_vector"] = (
            self._realized_decision.tolist()
        )
        return (
            self._build_observation(),
            float(reward),
            terminated,
            truncated,
            enriched_info,
        )


def make_dkana_thesis_faithful_env(
    **env_overrides: Any,
) -> DKANAThesisFaithfulDecisionEnvWrapper:
    """Build thesis-aligned Gym with the 18D/19D DKANA decision-vector contract."""
    observation_mode = str(env_overrides.pop("observation_mode", "decision_reward"))
    action_space_mode = str(env_overrides.pop("action_space_mode", "onehot_18d"))
    inventory_period_mode = str(
        env_overrides.pop("inventory_period_mode", "thesis_strict")
    )
    initial_action = env_overrides.pop("initial_action", None)
    learn_initial_decision = bool(env_overrides.pop("learn_initial_decision", False))
    env = make_thesis_aligned_training_env(**env_overrides)
    return DKANAThesisFaithfulDecisionEnvWrapper(
        env,
        observation_mode=observation_mode,
        action_space_mode=action_space_mode,
        inventory_period_mode=inventory_period_mode,
        initial_action=initial_action,
        learn_initial_decision=learn_initial_decision,
    )
