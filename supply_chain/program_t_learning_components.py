"""Trainable components for the prospective Paper 2 hybrid.

The module defines the causal recurrent representation and its decision-focused
loss, but performs no training and opens no seed. Architecture sizes remain
unfrozen until the T0 and U2 mechanism gates pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


FORBIDDEN_FEATURE_TOKENS = (
    "true_regime",
    "transition_matrix",
    "future_demand",
    "future_risk",
    "tape_id",
    "seed",
    "oracle_action",
    "oracle_score",
)


def validate_deployable_features(names: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(str(name) for name in names)
    if len(set(normalized)) != len(normalized):
        raise ValueError("feature names must be unique")
    violations = [
        name
        for name in normalized
        if any(token in name.lower() for token in FORBIDDEN_FEATURE_TOKENS)
    ]
    if violations:
        raise ValueError(f"nondeployable features are forbidden: {violations}")
    return normalized


@dataclass(frozen=True)
class HybridLossWeights:
    decision_regret: float = 1.0
    belief_likelihood: float = 0.20
    terminal_quantile: float = 0.50
    timing_imitation: float = 0.10

    def __post_init__(self) -> None:
        if min(
            self.decision_regret,
            self.belief_likelihood,
            self.terminal_quantile,
            self.timing_imitation,
        ) < 0:
            raise ValueError("loss weights must be nonnegative")


@dataclass(frozen=True)
class HybridOutput:
    belief_logits: Tensor
    ret_quantiles: Tensor
    action_logits: Tensor
    timing_logits: Tensor


class CausalBeliefValueGRU(nn.Module):
    """Causal history encoder with belief, value, mix, and timing heads."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        regime_count: int,
        quantile_levels: Sequence[float],
        action_count: int = 4,
        dwell_count: int = 3,
    ) -> None:
        super().__init__()
        levels = tuple(map(float, quantile_levels))
        if input_dim <= 0 or hidden_dim <= 0 or regime_count <= 1:
            raise ValueError("invalid recurrent geometry")
        if not levels or levels != tuple(sorted(levels)) or not all(0 < q < 1 for q in levels):
            raise ValueError("quantile levels must be increasing inside (0,1)")
        self.quantile_levels = levels
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.belief_head = nn.Linear(hidden_dim, regime_count)
        self.quantile_base = nn.Linear(hidden_dim, 1)
        self.quantile_increments = nn.Linear(hidden_dim, len(levels) - 1)
        self.action_head = nn.Linear(hidden_dim, action_count)
        self.timing_head = nn.Linear(hidden_dim, dwell_count)

    def forward(self, history: Tensor, lengths: Tensor) -> HybridOutput:
        if history.ndim != 3 or lengths.ndim != 1 or len(history) != len(lengths):
            raise ValueError("history must be [batch,time,features] with aligned lengths")
        packed = nn.utils.rnn.pack_padded_sequence(
            history,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _sequence, hidden = self.gru(packed)
        representation = hidden[-1]
        base = self.quantile_base(representation)
        if len(self.quantile_levels) == 1:
            quantiles = base
        else:
            increments = F.softplus(self.quantile_increments(representation))
            quantiles = torch.cat((base, base + torch.cumsum(increments, dim=-1)), dim=-1)
        return HybridOutput(
            belief_logits=self.belief_head(representation),
            ret_quantiles=quantiles,
            action_logits=self.action_head(representation),
            timing_logits=self.timing_head(representation),
        )

    def loss(
        self,
        output: HybridOutput,
        *,
        realized_regime: Tensor,
        realized_terminal_ret: Tensor,
        regret_by_action: Tensor,
        selected_dwell: Tensor,
        weights: HybridLossWeights = HybridLossWeights(),
    ) -> dict[str, Tensor]:
        if regret_by_action.shape != output.action_logits.shape:
            raise ValueError("regret_by_action must align with the four mix logits")
        probabilities = F.softmax(output.action_logits, dim=-1)
        decision_regret = torch.mean(torch.sum(probabilities * regret_by_action, dim=-1))
        belief = F.cross_entropy(output.belief_logits, realized_regime.long())
        target = realized_terminal_ret.reshape(-1, 1)
        errors = target - output.ret_quantiles
        levels = torch.as_tensor(
            self.quantile_levels, dtype=errors.dtype, device=errors.device
        ).reshape(1, -1)
        quantile = torch.mean(torch.maximum(levels * errors, (levels - 1.0) * errors))
        timing = F.cross_entropy(output.timing_logits, selected_dwell.long())
        total = (
            weights.decision_regret * decision_regret
            + weights.belief_likelihood * belief
            + weights.terminal_quantile * quantile
            + weights.timing_imitation * timing
        )
        return {
            "total": total,
            "decision_regret": decision_regret,
            "belief_likelihood": belief,
            "terminal_quantile": quantile,
            "timing_imitation": timing,
        }


def quantile_coverage(
    predicted_quantiles: Tensor,
    realized: Tensor,
    levels: Sequence[float],
) -> tuple[float, ...]:
    if predicted_quantiles.ndim != 2 or predicted_quantiles.shape[1] != len(levels):
        raise ValueError("predicted quantiles and levels do not align")
    target = realized.reshape(-1, 1)
    return tuple(
        float(value)
        for value in torch.mean((target <= predicted_quantiles).float(), dim=0).tolist()
    )

