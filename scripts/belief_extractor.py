"""Belief feature extractors for Track B PPO policies (Ruta A).

Both extractors are plain ``BaseFeaturesExtractor`` subclasses -- structurally
identical in role to ``scripts/real_kan_extractor.py``'s
``RealKANFeaturesExtractor`` -- meant to be:

1. wrapped with a temporary linear prediction head and trained with a
   supervised BCE loss on (v10 observation, future-risk label) pairs by
   ``scripts/pretrain_risk_belief_encoder.py`` (no PPO loss, no RL);
2. have the prediction head dropped, and the trunk's ``state_dict()`` loaded
   into a fresh PPO policy's ``features_extractor`` before ``model.learn()``.

The PPO reward and evaluation metric are unchanged (Garrido Excel ReT). This
tests whether a pretrained *shared representation* -- not just a frozen scalar
probability appended to the observation, which
``scripts/run_track_b_risk_belief_sidecar.py`` already tried and found did not
help PPO+MLP -- lets the policy make better use of observed risk memory.
"""

from __future__ import annotations

import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from kan import KAN


class MLPBeliefExtractor(BaseFeaturesExtractor):
    """Plain MLP trunk over the full v10 observation vector."""

    def __init__(
        self,
        observation_space,
        features_dim: int = 64,
        hidden_width: int = 64,
    ) -> None:
        super().__init__(observation_space, features_dim)
        if len(observation_space.shape) != 1:
            raise ValueError("MLPBeliefExtractor expects a flat vector observation.")
        input_dim = int(observation_space.shape[0])
        self.trunk = torch.nn.Sequential(
            torch.nn.Linear(input_dim, int(hidden_width)),
            torch.nn.ReLU(),
            torch.nn.Linear(int(hidden_width), int(features_dim)),
            torch.nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.trunk(observations.float())


class RealKANBeliefExtractor(BaseFeaturesExtractor):
    """Real pykan KAN trunk over the full v10 observation vector.

    Mirrors ``scripts/real_kan_extractor.py``'s ``RealKANFeaturesExtractor``
    construction exactly, so a pretrained trunk's ``state_dict()`` loads
    directly into a fresh instance built the same way for RL training.
    """

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
            raise ValueError("RealKANBeliefExtractor expects a flat vector observation.")
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
