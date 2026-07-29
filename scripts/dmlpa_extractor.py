"""DMLPA features extractor (Transformer-over-history) for SB3 PPO — einops-free.

Recreates the user's DMLPA: project each of `factor` stacked observation frames to a latent,
add sinusoidal positional encoding, run a multi-layer TransformerEncoder over the frame sequence,
and pool the last token. Use with VecFrameStack(n_stack=factor): the policy then sees the last
`factor` frames and can attend over disruption history (Garrido "learn the disruptions").
"""

from __future__ import annotations

import math

import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class DMLPA(BaseFeaturesExtractor):
    def __init__(self, observation_space, factor: int, features_dim: int = 120, nhead: int = 12,
                 num_layers: int = 4):
        super().__init__(observation_space, features_dim)
        self.factor = int(factor)
        self.obs_dimension = observation_space.shape[0] // self.factor
        self.latent_rw = torch.nn.Sequential(
            torch.nn.Linear(self.obs_dimension, 100),
            torch.nn.GELU(),
            torch.nn.Linear(100, features_dim),
        )
        self.pre_norm = torch.nn.LayerNorm(features_dim)
        layer = torch.nn.TransformerEncoderLayer(
            d_model=features_dim, nhead=nhead, batch_first=True)
        self.accumulated = torch.nn.TransformerEncoder(layer, num_layers=num_layers)
        self.register_buffer("pos_encoding", self._build_pe(self.factor, features_dim))

    @staticmethod
    def _build_pe(seq_len: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        b = observations.shape[0]
        # VecFrameStack concatenates frames on the last axis -> (b, factor, obs_dim)
        x = observations.reshape(b, self.factor, self.obs_dimension)
        x = self.latent_rw(x)
        x = x + self.pos_encoding
        x = self.pre_norm(x)
        x = self.accumulated(x)
        return x[:, -1, :]  # last-token pooling
