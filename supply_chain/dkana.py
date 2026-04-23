from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from supply_chain.external_env_interface import (
    ACTION_FIELDS,
    ACTION_FIELDS_TRACK_B_V1,
    CONTROL_CONTEXT_FIELDS,
    OBSERVATION_FIELDS_V3,
    STATE_CONSTRAINT_FIELDS,
    build_shift_control_constraint_vector,
    build_shift_control_state_constraint_vector,
    get_shift_control_constraint_context,
)

RELATION_TOKENS: tuple[str, ...] = ("=", "<", ">")
RELATION_MODES: tuple[str, ...] = ("equality", "temporal_delta")
DKANA_CONFIG_FIELDS: tuple[str, ...] = CONTROL_CONTEXT_FIELDS + tuple(
    f"prev_{field_name}" for field_name in ACTION_FIELDS
)
MIN_STD = 1e-4


def build_dkana_config_fields(action_dim: int) -> tuple[str, ...]:
    """Return control-context field names for the action contract width."""
    if action_dim == len(ACTION_FIELDS):
        action_fields = ACTION_FIELDS
    elif action_dim == len(ACTION_FIELDS_TRACK_B_V1):
        action_fields = ACTION_FIELDS_TRACK_B_V1
    else:
        raise ValueError(
            f"action_dim {action_dim} does not match any known action contract "
            f"(Track A: {len(ACTION_FIELDS)}, Track B: {len(ACTION_FIELDS_TRACK_B_V1)})."
        )
    return CONTROL_CONTEXT_FIELDS + tuple(
        f"prev_{field_name}" for field_name in action_fields
    )


@dataclass(frozen=True)
class EnumerationMap:
    """Stable integer indexing for symbolic variables and relation tokens."""

    variable_to_index: dict[str, int]
    relation_to_index: dict[str, int]

    def variable_index(self, variable_name: str) -> int:
        """Return the numeric id associated with a symbolic variable."""
        if variable_name not in self.variable_to_index:
            raise KeyError(f"Unknown variable {variable_name!r}.")
        return self.variable_to_index[variable_name]

    def relation_index(self, relation_token: str) -> int:
        """Return the numeric id associated with a relation token."""
        if relation_token not in self.relation_to_index:
            raise KeyError(f"Unknown relation token {relation_token!r}.")
        return self.relation_to_index[relation_token]


@dataclass(frozen=True)
class DKANADataset:
    """Sliding-window dataset for offline DKANA training on MFSC rollouts."""

    row_matrices: np.ndarray
    config_context: np.ndarray
    action_targets: np.ndarray
    time_mask: np.ndarray
    reward_targets: np.ndarray | None
    variable_names: tuple[str, ...]
    config_fields: tuple[str, ...]
    relation_to_index: dict[str, int]
    relation_mode: str


def build_prefixed_variable_names(
    observation_fields: Sequence[str],
    state_constraint_fields: Sequence[str],
) -> tuple[str, ...]:
    """Disambiguate repeated names across observation and state-constraint views."""
    return tuple(f"obs::{field_name}" for field_name in observation_fields) + tuple(
        f"state::{field_name}" for field_name in state_constraint_fields
    )


def build_enumeration_map(
    variable_names: Sequence[str],
    relation_tokens: Sequence[str] = RELATION_TOKENS,
) -> EnumerationMap:
    """Build zero-based ids for symbolic variables and relation operators."""
    unique_variable_names = tuple(variable_names)
    if len(unique_variable_names) != len(set(unique_variable_names)):
        raise ValueError("variable_names must be unique.")
    unique_relation_tokens = tuple(relation_tokens)
    if len(unique_relation_tokens) != len(set(unique_relation_tokens)):
        raise ValueError("relation_tokens must be unique.")
    return EnumerationMap(
        variable_to_index={
            variable_name: index
            for index, variable_name in enumerate(unique_variable_names)
        },
        relation_to_index={
            relation_token: index
            for index, relation_token in enumerate(unique_relation_tokens)
        },
    )


def build_equality_row_matrix(
    values: np.ndarray | Sequence[float],
    field_names: Sequence[str],
    enumeration_map: EnumerationMap,
    *,
    relation_token: str = "=",
) -> np.ndarray:
    """Encode a numeric feature vector as row-wise symbolic equality triplets."""
    values_array = np.asarray(values, dtype=np.float32)
    if values_array.ndim != 1:
        raise ValueError("values must be a 1D array.")
    if values_array.shape[0] != len(field_names):
        raise ValueError(
            "values length must match field_names length: "
            f"{values_array.shape[0]} != {len(field_names)}."
        )
    relation_index = float(enumeration_map.relation_index(relation_token))
    row_matrix = np.empty((len(field_names), 3), dtype=np.float32)
    for row_index, field_name in enumerate(field_names):
        row_matrix[row_index, 0] = float(enumeration_map.variable_index(field_name))
        row_matrix[row_index, 1] = relation_index
        row_matrix[row_index, 2] = float(values_array[row_index])
    return row_matrix


def build_temporal_relation_row_matrix(
    current_values: np.ndarray | Sequence[float],
    previous_values: np.ndarray | Sequence[float],
    field_names: Sequence[str],
    enumeration_map: EnumerationMap,
    *,
    tolerance: float = 1e-6,
) -> np.ndarray:
    """Encode current-vs-previous feature relations as symbolic triplets."""
    current_array = np.asarray(current_values, dtype=np.float32)
    previous_array = np.asarray(previous_values, dtype=np.float32)
    if current_array.ndim != 1 or previous_array.ndim != 1:
        raise ValueError("current_values and previous_values must be 1D arrays.")
    if current_array.shape != previous_array.shape:
        raise ValueError("current_values and previous_values must have the same shape.")
    if current_array.shape[0] != len(field_names):
        raise ValueError("values length must match field_names length.")

    row_matrix = np.empty((len(field_names), 3), dtype=np.float32)
    for row_index, field_name in enumerate(field_names):
        difference = float(current_array[row_index] - previous_array[row_index])
        if difference > tolerance:
            relation_token = ">"
        elif difference < -tolerance:
            relation_token = "<"
        else:
            relation_token = "="
        row_matrix[row_index, 0] = float(enumeration_map.variable_index(field_name))
        row_matrix[row_index, 1] = float(enumeration_map.relation_index(relation_token))
        row_matrix[row_index, 2] = float(previous_array[row_index])
    return row_matrix


def build_mfsc_relational_state(
    observation: np.ndarray | Sequence[float],
    state_constraint_vector: np.ndarray | Sequence[float],
    *,
    observation_fields: Sequence[str] = OBSERVATION_FIELDS_V3,
    state_constraint_fields: Sequence[str] = STATE_CONSTRAINT_FIELDS,
    enumeration_map: EnumerationMap | None = None,
    previous_observation: np.ndarray | Sequence[float] | None = None,
    previous_state_constraint_vector: np.ndarray | Sequence[float] | None = None,
    relation_mode: str = "equality",
) -> np.ndarray:
    """
    Build the MRC output for a single MFSC state.

    The current repo exposes normalized numeric observations plus live state
    constraints. As a first DKANA-compatible representation, each feature is
    encoded as a triplet [variable_id, relation_id, value] using the equality
    relation. This preserves row-wise structure without inventing unsupported
    symbolic comparisons.
    """
    observation_array = np.asarray(observation, dtype=np.float32)
    state_constraint_array = np.asarray(state_constraint_vector, dtype=np.float32)
    variable_names = build_prefixed_variable_names(
        observation_fields,
        state_constraint_fields,
    )
    if enumeration_map is None:
        enumeration_map = build_enumeration_map(variable_names)
    if relation_mode not in RELATION_MODES:
        raise ValueError(f"relation_mode must be one of {RELATION_MODES}.")
    all_values = np.concatenate([observation_array, state_constraint_array], axis=0)
    equality_rows = build_equality_row_matrix(
        all_values, variable_names, enumeration_map
    )
    if relation_mode == "equality":
        return equality_rows

    if previous_observation is None or previous_state_constraint_vector is None:
        temporal_rows = np.zeros_like(equality_rows)
    else:
        previous_observation_array = np.asarray(previous_observation, dtype=np.float32)
        previous_state_constraint_array = np.asarray(
            previous_state_constraint_vector, dtype=np.float32
        )
        previous_values = np.concatenate(
            [previous_observation_array, previous_state_constraint_array],
            axis=0,
        )
        temporal_rows = build_temporal_relation_row_matrix(
            all_values,
            previous_values,
            variable_names,
            enumeration_map,
        )
    return np.concatenate([equality_rows, temporal_rows], axis=0)


def build_previous_action_context(
    actions: np.ndarray | Sequence[Sequence[float]],
    episode_ids: np.ndarray | Sequence[int],
) -> np.ndarray:
    """
    Shift actions within each episode to approximate theta_SC_(i-1).

    The first observation in each episode receives a zero vector because no
    previous control configuration exists yet.
    """
    actions_array = np.asarray(actions, dtype=np.float32)
    episode_ids_array = np.asarray(episode_ids, dtype=np.int32)
    if actions_array.ndim != 2:
        raise ValueError("actions must be a 2D array.")
    if episode_ids_array.ndim != 1:
        raise ValueError("episode_ids must be a 1D array.")
    if actions_array.shape[0] != episode_ids_array.shape[0]:
        raise ValueError("actions and episode_ids must have the same length.")

    previous_actions = np.zeros_like(actions_array)
    for episode_id in np.unique(episode_ids_array):
        indices = np.flatnonzero(episode_ids_array == episode_id)
        if indices.size <= 1:
            continue
        previous_actions[indices[1:]] = actions_array[indices[:-1]]
    return previous_actions


def build_dkana_windows(
    *,
    observations: np.ndarray,
    actions: np.ndarray,
    episode_ids: np.ndarray,
    constraint_context: np.ndarray,
    state_constraint_context: np.ndarray,
    rewards: np.ndarray | None = None,
    window_size: int,
    observation_fields: Sequence[str] = OBSERVATION_FIELDS_V3,
    state_constraint_fields: Sequence[str] = STATE_CONSTRAINT_FIELDS,
    relation_mode: str = "equality",
) -> DKANADataset:
    """Create fixed-length causal windows for offline DKANA training."""
    if window_size <= 0:
        raise ValueError("window_size must be > 0.")
    if relation_mode not in RELATION_MODES:
        raise ValueError(f"relation_mode must be one of {RELATION_MODES}.")

    observations_array = np.asarray(observations, dtype=np.float32)
    actions_array = np.asarray(actions, dtype=np.float32)
    episode_ids_array = np.asarray(episode_ids, dtype=np.int32)
    constraint_context_array = np.asarray(constraint_context, dtype=np.float32)
    state_constraint_array = np.asarray(state_constraint_context, dtype=np.float32)
    rewards_array = None if rewards is None else np.asarray(rewards, dtype=np.float32)

    num_steps = observations_array.shape[0]
    if observations_array.ndim != 2:
        raise ValueError("observations must be a 2D array.")
    if actions_array.ndim != 2:
        raise ValueError("actions must be a 2D array.")
    if constraint_context_array.ndim != 2:
        raise ValueError("constraint_context must be a 2D array.")
    if state_constraint_array.ndim != 2:
        raise ValueError("state_constraint_context must be a 2D array.")
    if episode_ids_array.ndim != 1:
        raise ValueError("episode_ids must be a 1D array.")
    if num_steps != actions_array.shape[0] or num_steps != episode_ids_array.shape[0]:
        raise ValueError("observations, actions, and episode_ids must align.")
    if num_steps != constraint_context_array.shape[0]:
        raise ValueError("constraint_context must align with observations.")
    if num_steps != state_constraint_array.shape[0]:
        raise ValueError("state_constraint_context must align with observations.")
    if rewards_array is not None and rewards_array.shape[0] != num_steps:
        raise ValueError("rewards must align with observations.")
    if observations_array.shape[1] != len(observation_fields):
        raise ValueError("observations width does not match observation_fields.")
    if state_constraint_array.shape[1] != len(state_constraint_fields):
        raise ValueError(
            "state_constraint_context width does not match state_constraint_fields."
        )
    if constraint_context_array.shape[1] != len(CONTROL_CONTEXT_FIELDS):
        raise ValueError("constraint_context width does not match repo contract.")
    if actions_array.shape[1] not in (
        len(ACTION_FIELDS),
        len(ACTION_FIELDS_TRACK_B_V1),
    ):
        raise ValueError(
            f"actions width {actions_array.shape[1]} does not match any known "
            f"action contract (Track A: {len(ACTION_FIELDS)}, "
            f"Track B: {len(ACTION_FIELDS_TRACK_B_V1)})."
        )

    variable_names = build_prefixed_variable_names(
        observation_fields,
        state_constraint_fields,
    )
    enumeration_map = build_enumeration_map(variable_names)
    row_count = len(variable_names) * (2 if relation_mode == "temporal_delta" else 1)
    # config_dim is constraint_context + previous_actions (dynamic for Track B)
    config_dim = constraint_context_array.shape[1] + actions_array.shape[1]
    row_matrices_by_step: list[tuple[int, np.ndarray]] = []
    for episode_id in np.unique(episode_ids_array):
        episode_indices = np.flatnonzero(episode_ids_array == episode_id)
        for offset, step_index in enumerate(episode_indices):
            previous_index = None if offset == 0 else int(episode_indices[offset - 1])
            row_matrices_by_step.append(
                (
                    int(step_index),
                    build_mfsc_relational_state(
                        observations_array[step_index],
                        state_constraint_array[step_index],
                        observation_fields=observation_fields,
                        state_constraint_fields=state_constraint_fields,
                        enumeration_map=enumeration_map,
                        previous_observation=(
                            None
                            if previous_index is None
                            else observations_array[previous_index]
                        ),
                        previous_state_constraint_vector=(
                            None
                            if previous_index is None
                            else state_constraint_array[previous_index]
                        ),
                        relation_mode=relation_mode,
                    ),
                )
            )
    row_matrices_by_step.sort(key=lambda item: item[0])
    row_matrices = np.stack(
        [row_matrix for _, row_matrix in row_matrices_by_step], axis=0
    )
    previous_actions = build_previous_action_context(actions_array, episode_ids_array)
    config_context_array = np.concatenate(
        [constraint_context_array, previous_actions],
        axis=1,
    ).astype(np.float32)

    window_rows = np.zeros(
        (num_steps, window_size, row_count, 3),
        dtype=np.float32,
    )
    window_config = np.zeros(
        (num_steps, window_size, config_dim),
        dtype=np.float32,
    )
    time_mask = np.zeros((num_steps, window_size), dtype=bool)

    for episode_id in np.unique(episode_ids_array):
        episode_indices = np.flatnonzero(episode_ids_array == episode_id)
        for offset, step_index in enumerate(episode_indices):
            window_start = max(0, offset - window_size + 1)
            source_indices = episode_indices[window_start : offset + 1]
            pad_length = window_size - source_indices.shape[0]
            window_rows[step_index, pad_length:] = row_matrices[source_indices]
            window_config[step_index, pad_length:] = config_context_array[
                source_indices
            ]
            time_mask[step_index, pad_length:] = True

    return DKANADataset(
        row_matrices=window_rows,
        config_context=window_config,
        action_targets=actions_array.astype(np.float32),
        time_mask=time_mask,
        reward_targets=rewards_array,
        variable_names=variable_names,
        config_fields=build_dkana_config_fields(actions_array.shape[1]),
        relation_to_index=dict(enumeration_map.relation_to_index),
        relation_mode=relation_mode,
    )


class LearnedPositionalEncoding(nn.Module):
    """Learned positional embeddings for row or temporal ordering."""

    def __init__(self, max_length: int, embedding_dim: int) -> None:
        super().__init__()
        self.max_length = int(max_length)
        self.position_embeddings = nn.Embedding(self.max_length, embedding_dim)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 3:
            raise ValueError("inputs must have shape (batch, sequence, embedding_dim).")
        sequence_length = int(inputs.shape[1])
        if sequence_length > self.max_length:
            raise ValueError(
                f"sequence length {sequence_length} exceeds max_length {self.max_length}."
            )
        positions = torch.arange(sequence_length, device=inputs.device)
        positions = positions.unsqueeze(0).expand(inputs.shape[0], -1)
        return inputs + self.position_embeddings(positions)


def build_causal_attention_mask(length: int, device: torch.device) -> Tensor:
    """Create a boolean mask that blocks attention to future positions."""
    if length <= 0:
        raise ValueError("length must be > 0.")
    return torch.triu(
        torch.ones((length, length), dtype=torch.bool, device=device),
        diagonal=1,
    )


class DKANAPolicy(nn.Module):
    """Distributional DKANA starter policy for MFSC control actions."""

    def __init__(
        self,
        *,
        config_dim: int,
        action_dim: int,
        row_dim: int = 3,
        latent_dim: int = 128,
        num_heads: int = 8,
        local_layers: int = 2,
        global_layers: int = 2,
        dim_feedforward: int = 256,
        max_rows: int = len(OBSERVATION_FIELDS_V3) + len(STATE_CONSTRAINT_FIELDS) + 1,
        max_sequence_length: int = 52,
    ) -> None:
        super().__init__()
        if latent_dim % num_heads != 0:
            raise ValueError("latent_dim must be divisible by num_heads.")
        self.action_dim = int(action_dim)
        self.latent_dim = int(latent_dim)
        self.num_heads = int(num_heads)
        self.row_encoder = nn.Sequential(
            nn.Linear(row_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
        )
        self.config_encoder = nn.Sequential(
            nn.Linear(config_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
        )
        self.local_position_encoding = LearnedPositionalEncoding(max_rows, latent_dim)
        local_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            activation="gelu",
            batch_first=True,
        )
        self.local_attention = nn.TransformerEncoder(
            local_layer, num_layers=local_layers
        )
        self.global_position_encoding = LearnedPositionalEncoding(
            max_sequence_length,
            latent_dim,
        )
        global_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            activation="gelu",
            batch_first=True,
        )
        self.global_attention = nn.TransformerEncoder(
            global_layer,
            num_layers=global_layers,
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, action_dim * 2),
        )

    def forward(
        self,
        row_matrices: Tensor,
        config_context: Tensor,
        time_mask: Tensor | None = None,
    ) -> torch.distributions.Normal:
        """
        Return an action distribution from causal row/state attention.

        Parameters
        ----------
        row_matrices :
            Tensor with shape ``(batch, sequence, rows, 3)``.
        config_context :
            Tensor with shape ``(batch, sequence, config_dim)``.
        time_mask :
            Optional boolean tensor with shape ``(batch, sequence)`` where
            ``True`` marks valid timesteps and ``False`` marks left padding.
        """
        if row_matrices.ndim != 4:
            raise ValueError("row_matrices must have shape (batch, sequence, rows, 3).")
        if config_context.ndim != 3:
            raise ValueError(
                "config_context must have shape (batch, sequence, config_dim)."
            )
        if row_matrices.shape[0] != config_context.shape[0]:
            raise ValueError("row_matrices and config_context batch sizes must match.")
        if row_matrices.shape[1] != config_context.shape[1]:
            raise ValueError(
                "row_matrices and config_context sequence lengths must match."
            )
        if row_matrices.shape[-1] != 3:
            raise ValueError("row_matrices last dimension must be 3.")
        if time_mask is not None:
            if time_mask.shape != row_matrices.shape[:2]:
                raise ValueError("time_mask must match (batch, sequence).")
            time_mask = time_mask.to(dtype=torch.bool, device=row_matrices.device)
            if torch.any(time_mask.sum(dim=1) == 0):
                raise ValueError(
                    "Each batch item must contain at least one valid step."
                )

        batch_size, sequence_length, row_count, _ = row_matrices.shape
        row_latent = self.row_encoder(row_matrices)
        config_latent = self.config_encoder(config_context).unsqueeze(2)
        local_tokens = torch.cat([row_latent, config_latent], dim=2)
        local_tokens = local_tokens.reshape(
            batch_size * sequence_length,
            row_count + 1,
            self.latent_dim,
        )
        local_tokens = self.local_position_encoding(local_tokens)
        local_mask = build_causal_attention_mask(
            local_tokens.shape[1], local_tokens.device
        )
        local_context = self.local_attention(local_tokens, mask=local_mask)

        state_embeddings = local_context[:, -1, :].reshape(
            batch_size,
            sequence_length,
            self.latent_dim,
        )
        state_embeddings = self.global_position_encoding(state_embeddings)
        global_mask = build_causal_attention_mask(
            sequence_length, state_embeddings.device
        )
        if time_mask is not None:
            valid_query = time_mask.unsqueeze(2)
            valid_key = time_mask.unsqueeze(1)
            self_attention = torch.eye(
                sequence_length,
                dtype=torch.bool,
                device=state_embeddings.device,
            ).unsqueeze(0)
            combined_mask = global_mask.unsqueeze(0).expand(batch_size, -1, -1).clone()
            combined_mask |= ~valid_key
            combined_mask = torch.where(valid_query, combined_mask, ~self_attention)
            global_mask = combined_mask.repeat_interleave(self.num_heads, dim=0)
        global_context = self.global_attention(
            state_embeddings,
            mask=global_mask,
        )

        if time_mask is None:
            policy_embedding = global_context[:, -1, :]
        else:
            position_index = torch.arange(sequence_length, device=time_mask.device)
            position_index = position_index.unsqueeze(0).expand(batch_size, -1)
            last_valid_index = (
                position_index.masked_fill(~time_mask, -1).max(dim=1).values
            )
            gather_index = last_valid_index.view(-1, 1, 1).expand(
                -1,
                1,
                self.latent_dim,
            )
            policy_embedding = global_context.gather(1, gather_index).squeeze(1)

        decoder_output = self.decoder(policy_embedding)
        mean, std_logits = torch.chunk(decoder_output, chunks=2, dim=-1)
        std = torch.nn.functional.softplus(std_logits) + MIN_STD
        return torch.distributions.Normal(mean, std)


class DKANAOnlinePolicyAdapter:
    """Stateful callable that supplies DKANA with an online SMS context window."""

    def __init__(
        self,
        model: DKANAPolicy,
        *,
        window_size: int,
        observation_fields: Sequence[str],
        state_constraint_fields: Sequence[str] = STATE_CONSTRAINT_FIELDS,
        action_dim: int,
        relation_mode: str = "equality",
        device: str | torch.device = "cpu",
        deterministic: bool = True,
    ) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be > 0.")
        if action_dim not in (len(ACTION_FIELDS), len(ACTION_FIELDS_TRACK_B_V1)):
            raise ValueError("Unsupported action_dim for DKANA online adapter.")
        if relation_mode not in RELATION_MODES:
            raise ValueError(f"relation_mode must be one of {RELATION_MODES}.")
        self.model = model.to(device)
        self.model.eval()
        self.window_size = int(window_size)
        self.observation_fields = tuple(observation_fields)
        self.state_constraint_fields = tuple(state_constraint_fields)
        self.action_dim = int(action_dim)
        self.relation_mode = relation_mode
        self.device = torch.device(device)
        self.deterministic = bool(deterministic)
        self.constraint_context_vector = build_shift_control_constraint_vector(
            get_shift_control_constraint_context()
        ).astype(np.float32)
        self.variable_names = build_prefixed_variable_names(
            self.observation_fields,
            self.state_constraint_fields,
        )
        self.enumeration_map = build_enumeration_map(self.variable_names)
        self._rows: deque[np.ndarray] = deque(maxlen=self.window_size)
        self._configs: deque[np.ndarray] = deque(maxlen=self.window_size)
        self._previous_observation: np.ndarray | None = None
        self._previous_state_constraint_vector: np.ndarray | None = None
        self._previous_action = np.zeros(self.action_dim, dtype=np.float32)

    def reset(self) -> None:
        """Clear the per-episode SMS and previous-action context."""
        self._rows.clear()
        self._configs.clear()
        self._previous_observation = None
        self._previous_state_constraint_vector = None
        self._previous_action = np.zeros(self.action_dim, dtype=np.float32)

    def _state_vector_from_info(self, info: dict[str, Any]) -> np.ndarray:
        state_context = info.get("state_constraint_context")
        if not isinstance(state_context, dict):
            raise ValueError("DKANA policy requires info['state_constraint_context'].")
        return build_shift_control_state_constraint_vector(state_context).astype(
            np.float32
        )

    def _append_current_state(self, obs: np.ndarray, info: dict[str, Any]) -> None:
        state_vector = self._state_vector_from_info(info)
        row_matrix = build_mfsc_relational_state(
            obs,
            state_vector,
            observation_fields=self.observation_fields,
            state_constraint_fields=self.state_constraint_fields,
            enumeration_map=self.enumeration_map,
            previous_observation=self._previous_observation,
            previous_state_constraint_vector=self._previous_state_constraint_vector,
            relation_mode=self.relation_mode,
        )
        config_context = np.concatenate(
            [self.constraint_context_vector, self._previous_action],
            axis=0,
        ).astype(np.float32)
        self._rows.append(row_matrix)
        self._configs.append(config_context)
        self._previous_observation = obs.copy()
        self._previous_state_constraint_vector = state_vector.copy()

    def _build_window_tensors(self) -> tuple[Tensor, Tensor, Tensor]:
        if not self._rows:
            raise RuntimeError("No DKANA context rows available.")
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
        return (
            torch.from_numpy(row_window[None, :]).to(self.device),
            torch.from_numpy(config_window[None, :]).to(self.device),
            torch.from_numpy(time_mask[None, :]).to(self.device),
        )

    def __call__(self, obs: np.ndarray, info: dict[str, Any]) -> np.ndarray:
        """Return a clipped DKANA action for the current environment state."""
        obs_array = np.asarray(obs, dtype=np.float32)
        self._append_current_state(obs_array, info)
        row_tensor, config_tensor, mask_tensor = self._build_window_tensors()
        with torch.no_grad():
            distribution = self.model(row_tensor, config_tensor, mask_tensor)
            action_tensor = (
                distribution.mean if self.deterministic else distribution.sample()
            )
        action = action_tensor.squeeze(0).detach().cpu().numpy().astype(np.float32)
        action = np.clip(action, -1.0, 1.0)
        self._previous_action = action.copy()
        return action
