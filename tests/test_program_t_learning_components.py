from __future__ import annotations

import pytest
import torch

from supply_chain.program_t_learning_components import (
    CausalBeliefValueGRU,
    quantile_coverage,
    validate_deployable_features,
)


def test_feature_audit_rejects_privileged_information() -> None:
    assert validate_deployable_features(("inventory_c", "backlog_h"))
    with pytest.raises(ValueError, match="nondeployable"):
        validate_deployable_features(("inventory_c", "true_regime"))


def test_recurrent_heads_and_quantiles_are_well_formed() -> None:
    model = CausalBeliefValueGRU(
        input_dim=5,
        hidden_dim=8,
        regime_count=3,
        quantile_levels=(0.1, 0.5, 0.9),
    )
    history = torch.randn(4, 6, 5)
    output = model(history, torch.tensor([6, 5, 4, 3]))
    assert output.belief_logits.shape == (4, 3)
    assert output.action_logits.shape == (4, 4)
    assert output.timing_logits.shape == (4, 3)
    assert torch.all(output.ret_quantiles[:, 1:] >= output.ret_quantiles[:, :-1])


def test_decision_focused_loss_is_differentiable() -> None:
    model = CausalBeliefValueGRU(
        input_dim=3,
        hidden_dim=6,
        regime_count=3,
        quantile_levels=(0.1, 0.5, 0.9),
    )
    output = model(torch.randn(3, 4, 3), torch.tensor([4, 4, 3]))
    losses = model.loss(
        output,
        realized_regime=torch.tensor([0, 1, 2]),
        realized_terminal_ret=torch.tensor([0.7, 0.8, 0.9]),
        regret_by_action=torch.tensor(
            [[0.0, 0.1, 0.2, 0.3], [0.2, 0.0, 0.1, 0.3], [0.3, 0.2, 0.1, 0.0]]
        ),
        selected_dwell=torch.tensor([0, 1, 2]),
    )
    losses["total"].backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_quantile_coverage_returns_one_value_per_level() -> None:
    coverage = quantile_coverage(
        torch.tensor([[0.2, 0.5], [0.8, 0.9]]),
        torch.tensor([0.4, 0.7]),
        (0.1, 0.9),
    )
    assert coverage == (0.5, 1.0)

