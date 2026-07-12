"""Ruta B: feature extractor with a live auxiliary belief head.

Pairs with ``scripts/ruta_b_risk_label_wrapper.py``: the wrapper appends one
extra dimension (the true future-risk label) to the observation. This
extractor splits it back off before computing the policy/value features, and
also runs an auxiliary linear head on the shared trunk features to predict
that same label -- the auxiliary loss on this head is what
``scripts/ruta_b_aux_ppo.py`` adds to the PPO loss every gradient step, so the
trunk cannot "forget" to predict the way it did in Ruta A.

The trunk architecture matches ``scripts/belief_extractor.py::MLPBeliefExtractor``
exactly, so a Ruta-A-pretrained trunk can be loaded as a warm start (optional).
"""

from __future__ import annotations

import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from kan import KAN


class RutaBAuxFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space,
        features_dim: int = 64,
        hidden_width: int = 64,
    ) -> None:
        super().__init__(observation_space, features_dim)
        if len(observation_space.shape) != 1:
            raise ValueError("RutaBAuxFeaturesExtractor expects a flat vector observation.")
        # Last column is the injected future-risk label, not a real feature.
        input_dim = int(observation_space.shape[0]) - 1
        self.trunk = torch.nn.Sequential(
            torch.nn.Linear(input_dim, int(hidden_width)),
            torch.nn.ReLU(),
            torch.nn.Linear(int(hidden_width), int(features_dim)),
            torch.nn.ReLU(),
        )
        self.aux_head = torch.nn.Linear(int(features_dim), 1)
        self.last_aux_logit: torch.Tensor | None = None
        self.last_aux_label: torch.Tensor | None = None

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        observations = observations.float()
        real_obs = observations[:, :-1]
        label = observations[:, -1]
        features = self.trunk(real_obs)
        self.last_aux_logit = self.aux_head(features).squeeze(-1)
        self.last_aux_label = label
        return features

    def load_pretrained_trunk(self, state_dict: dict) -> None:
        self.trunk.load_state_dict(state_dict)


class RutaBRealKANAuxFeaturesExtractor(BaseFeaturesExtractor):
    """Ruta B auxiliary extractor with an official pykan KAN trunk.

    The final observation column is still the raw future-risk label used only
    for the auxiliary BCE objective. The KAN trunk receives the real observation
    vector without that label, matching the MLP Ruta B contract.
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
            raise ValueError("RutaBRealKANAuxFeaturesExtractor expects a flat vector observation.")
        self.input_dim = int(observation_space.shape[0]) - 1
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
        self.aux_head = torch.nn.Linear(int(features_dim), 1)
        self.last_aux_logit: torch.Tensor | None = None
        self.last_aux_label: torch.Tensor | None = None

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        observations = observations.float()
        real_obs = observations[:, :-1]
        label = observations[:, -1]
        x = torch.clamp(real_obs, -self.clamp_input, self.clamp_input)
        features = self.norm(self.kan(x))
        self.last_aux_logit = self.aux_head(features).squeeze(-1)
        self.last_aux_label = label
        return features
