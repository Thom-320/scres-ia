from __future__ import annotations

import pytest

from supply_chain.program_u_static_search import (
    BudgetExhausted,
    BudgetedPanelObjective,
    autoregressive_policy_gradient_search,
    bayesian_optimization_search,
    cma_es_search,
    cross_entropy_search,
    random_search,
)


def _evaluator(calendar: tuple[int, ...], tape: int) -> float:
    target = (0, 1, 2, 3)
    return float(sum(left == right for left, right in zip(calendar, target)) + tape * 0.0)


def _objective(budget: int = 24) -> BudgetedPanelObjective:
    return BudgetedPanelObjective(
        evaluator=_evaluator,
        tape_ids=(11, 12),
        horizon=4,
        action_count=4,
        call_budget=budget,
    )


def test_objective_enforces_panel_budget_and_caches() -> None:
    objective = _objective(budget=4)
    assert objective.evaluate((0, 1, 2, 3)) == 4.0
    assert objective.calls == 2
    assert objective.evaluate((0, 1, 2, 3)) == 4.0
    assert objective.calls == 2
    objective.evaluate((3, 3, 3, 3))
    with pytest.raises(BudgetExhausted):
        objective.evaluate((2, 2, 2, 2))


@pytest.mark.parametrize(
    "search",
    [
        random_search,
        cross_entropy_search,
        autoregressive_policy_gradient_search,
        cma_es_search,
        bayesian_optimization_search,
    ],
)
def test_searches_respect_budget_and_return_valid_calendar(search) -> None:
    objective = _objective()
    result = search(objective, seed=7)
    assert result.calendar_tape_calls <= objective.call_budget
    assert result.calendar_tape_calls % len(objective.tape_ids) == 0
    assert len(result.best_calendar) == objective.horizon
    assert all(0 <= value < objective.action_count for value in result.best_calendar)
    assert result.trace[-1].best_score == result.best_score


def test_ppo_search_uses_panel_rewards_and_respects_budget() -> None:
    pytest.importorskip("stable_baselines3")
    from supply_chain.program_u_static_search import ppo_calendar_search

    objective = BudgetedPanelObjective(
        evaluator=lambda calendar, tape: float(sum(calendar) + tape),
        tape_ids=(11, 12),
        horizon=2,
        action_count=2,
        call_budget=8,
    )
    result = ppo_calendar_search(objective, seed=7)
    assert 0 < result.calendar_tape_calls <= 8
    assert result.calendar_tape_calls % 2 == 0
    assert len(result.best_calendar) == 2
