"""Real KAN (pykan, Liu et al. 2024) feature extractor for SB3 policies.

Unlike ``scripts/kan_extractor.py`` (an RBF/KAN-inspired layer with a linear
skip -- explicitly not a KAN, see docs/KAN_REAL_DEMO_2026-07-02.md), this
wraps the official ``pykan`` ``KAN`` class -- learnable B-spline univariate
edge functions, the actual Kolmogorov-Arnold mechanism -- as a genuine SB3
``BaseFeaturesExtractor`` for online PPO training.

Two pykan defaults are disabled here because they are supervised-fitting /
interpretability conveniences that are ruinously slow in an online RL loop
(measured ~160x forward-pass slowdown at batch=1 on this machine):

- ``save_act=False``: skip caching per-layer activations for spline plots.
- ``symbolic_enabled=False``: skip the symbolic-regression bookkeeping.

Both can be re-enabled after training (on a frozen copy) if a spline plot is
wanted for a paper figure -- they do not need to be on during training.
"""

from __future__ import annotations

import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from kan import KAN


class RealKANFeaturesExtractor(BaseFeaturesExtractor):
    """Official pykan ``KAN`` as an SB3 feature extractor."""

    def __init__(
        self,
        observation_space,
        features_dim: int = 32,
        hidden_width: int = 32,
        grid: int = 3,
        k: int = 3,
        grid_range: tuple[float, float] = (-6.0, 6.0),
        clamp_input: float = 8.0,
        seed: int = 0,
    ) -> None:
        super().__init__(observation_space, features_dim)
        if len(observation_space.shape) != 1:
            raise ValueError("RealKANFeaturesExtractor expects a flat vector observation.")
        self.input_dim = int(observation_space.shape[0])
        self.clamp_input = float(clamp_input)
        self.kan = KAN(
            width=[self.input_dim, int(hidden_width), int(features_dim)],
            grid=int(grid),
            k=int(k),
            grid_range=list(grid_range),
            seed=int(seed),
            auto_save=False,
            save_act=False,
            symbolic_enabled=False,
        )
        self.norm = torch.nn.LayerNorm(features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(observations.float(), -self.clamp_input, self.clamp_input)
        features = self.kan(x)
        return self.norm(features)
