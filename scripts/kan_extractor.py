"""KAN-style feature extractor for SB3 policies.

This is a lightweight Kolmogorov-Arnold inspired layer: each input dimension is
expanded through learned univariate radial basis functions, and the resulting
edge functions are summed into a latent feature vector. It is intentionally
small enough for Track B smoke tests and avoids adding a fragile third-party
KAN dependency to the paper pipeline.
"""

from __future__ import annotations

import math

import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class RBFKANFeaturesExtractor(BaseFeaturesExtractor):
    """Univariate-basis additive feature map for vector observations."""

    def __init__(
        self,
        observation_space,
        features_dim: int = 64,
        num_centers: int = 9,
        center_min: float = -2.5,
        center_max: float = 2.5,
        use_linear_skip: bool = True,
    ) -> None:
        super().__init__(observation_space, features_dim)
        if len(observation_space.shape) != 1:
            raise ValueError("RBFKANFeaturesExtractor expects a flat vector observation.")
        self.input_dim = int(observation_space.shape[0])
        self.num_centers = int(num_centers)
        self.use_linear_skip = bool(use_linear_skip)

        centers = torch.linspace(float(center_min), float(center_max), self.num_centers)
        spacing = float(centers[1] - centers[0]) if self.num_centers > 1 else 1.0
        self.register_buffer("centers", centers.view(1, 1, self.num_centers))
        self.log_width = torch.nn.Parameter(torch.tensor(math.log(max(spacing, 1e-3))))
        self.coefficients = torch.nn.Parameter(
            0.02 * torch.randn(self.input_dim, self.num_centers, features_dim)
        )
        self.bias = torch.nn.Parameter(torch.zeros(features_dim))
        self.norm = torch.nn.LayerNorm(features_dim)
        self.skip = (
            torch.nn.Linear(self.input_dim, features_dim)
            if self.use_linear_skip
            else None
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(observations.float(), -10.0, 10.0)
        width = torch.exp(self.log_width).clamp_min(1e-3)
        basis = torch.exp(-0.5 * ((x.unsqueeze(-1) - self.centers) / width) ** 2)
        features = torch.einsum("bng,ngf->bf", basis, self.coefficients)
        features = features / math.sqrt(float(self.input_dim))
        features = features + self.bias
        if self.skip is not None:
            features = features + self.skip(x)
        return torch.tanh(self.norm(features))
