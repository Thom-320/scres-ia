"""David's DMLPA architectures — VERBATIM as proposed by David (2026-07-17).

Two variants, selectable from the lab notebook via MODEL_KIND:
  * ``PPO_DMLPA_FAITHFUL``   -> FriendDMLPAFaithful (his original: no positional encoding)
  * ``PPO_DMLPA_POSITIONAL`` -> FriendDMLPAPositional (sinusoidal PE + pre-LayerNorm)

The classes below are David's code unchanged (imports adapted to module scope). The
``factor`` argument partitions the flattened causal-history observation into ``factor``
tokens; with the lab's HISTORY_LEN=8 stacked observations of 21 features, factor=8 gives
the transformer one token per history step.
"""
from __future__ import annotations

import math
from typing import Any

import torch
from einops import rearrange
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class FriendDMLPAFaithful(BaseFeaturesExtractor):
    """David's original DMLPA-style feature extractor.

    It intentionally does not add explicit positional encoding. Use this as the
    faithful architecture baseline when comparing against David's proposed model.
    """

    def __init__(self, observation_space, factor: int = 1, features_dim: int = 120, hidden_dim: int = 100, nhead: int = 12, num_layers: int = 4):
        super().__init__(observation_space, features_dim)
        flat_dim = int(observation_space.shape[0])
        if flat_dim % factor != 0:
            raise ValueError(f"Observation dimension {flat_dim} is not divisible by factor={factor}")
        if features_dim % nhead != 0:
            raise ValueError("features_dim must be divisible by nhead")
        self.obs_dimension = flat_dim // factor
        self.factor = factor
        self.latent_rw = torch.nn.Sequential(
            torch.nn.Linear(self.obs_dimension, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, features_dim),
        )
        layer = torch.nn.TransformerEncoderLayer(d_model=features_dim, nhead=nhead, batch_first=True)
        self.accumulated = torch.nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        observations = rearrange(observations, "b (d k) -> b d k", d=self.factor)
        observations = self.latent_rw(observations)
        observations = self.accumulated(observations)
        return observations[:, -1, :]


class FriendDMLPAPositional(BaseFeaturesExtractor):
    """DMLPA variant with sinusoidal positional encoding and LayerNorm."""

    def __init__(self, observation_space, factor: int = 1, features_dim: int = 120, hidden_dim: int = 100, nhead: int = 12, num_layers: int = 4):
        super().__init__(observation_space, features_dim)
        flat_dim = int(observation_space.shape[0])
        if flat_dim % factor != 0:
            raise ValueError(f"Observation dimension {flat_dim} is not divisible by factor={factor}")
        if features_dim % nhead != 0:
            raise ValueError("features_dim must be divisible by nhead")
        self.obs_dimension = flat_dim // factor
        self.factor = factor
        self.latent_rw = torch.nn.Sequential(
            torch.nn.Linear(self.obs_dimension, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, features_dim),
        )
        self.pre_norm = torch.nn.LayerNorm(features_dim)
        layer = torch.nn.TransformerEncoderLayer(d_model=features_dim, nhead=nhead, batch_first=True)
        self.accumulated = torch.nn.TransformerEncoder(layer, num_layers=num_layers)
        self.register_buffer("pos_encoding", self.build_sinusoidal_pe(factor, features_dim))

    @staticmethod
    def build_sinusoidal_pe(seq_len: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        observations = rearrange(observations, "b (d k) -> b d k", d=self.factor)
        observations = self.latent_rw(observations)
        observations = observations + self.pos_encoding
        observations = self.pre_norm(observations)
        observations = self.accumulated(observations)
        return observations[:, -1, :]


# Backward-compatible alias used by older cells/notebooks.
DMLPA = FriendDMLPAPositional

DMLPA_MODEL_KINDS = ("PPO_DMLPA_FAITHFUL", "PPO_DMLPA_POSITIONAL")


def build_dmlpa_policy_kwargs(
    model_kind: str,
    *,
    factor: int,
    features_dim: int = 120,
    hidden_dim: int = 100,
    nhead: int = 12,
    num_layers: int = 4,
) -> dict[str, Any]:
    """David's build_policy_kwargs, parameterized by the lab config cell."""
    if model_kind == "PPO_DMLPA_FAITHFUL":
        extractor = FriendDMLPAFaithful
    elif model_kind == "PPO_DMLPA_POSITIONAL":
        extractor = FriendDMLPAPositional
    else:
        raise ValueError(f"Unknown DMLPA MODEL_KIND={model_kind}")
    return {
        "features_extractor_class": extractor,
        "features_extractor_kwargs": {
            "factor": factor,
            "features_dim": features_dim,
            "hidden_dim": hidden_dim,
            "nhead": nhead,
            "num_layers": num_layers,
        },
        "net_arch": dict(pi=[128, 64], vf=[128, 64]),
    }
