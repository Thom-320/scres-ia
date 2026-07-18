from __future__ import annotations

import math

import pytest

from research.paper2_exhaustive_search.program_t_exact_pomdp import (
    ExactPOMDPConfig,
    ExactProductMixPOMDP,
    solve_grid,
)


def test_unknown_model_belief_and_observation_probabilities_conserve_probability() -> None:
    model = ExactProductMixPOMDP(ExactPOMDPConfig(horizon=2))
    probabilities = [
        model.observation_probability(model.initial_belief, demand_c)
        for demand_c in range(4)
    ]
    assert sum(probabilities) == pytest.approx(1.0)
    for demand_c in range(4):
        posterior = model.update_belief(model.initial_belief, demand_c)
        assert sum(posterior) == pytest.approx(1.0, abs=1e-10)


def test_unknown_model_exact_grid_is_finite_and_nontrivial() -> None:
    payload = solve_grid((2, 3, 4))
    assert payload["solver"] == "exact_reachable_belief_dynamic_programming"
    for row in payload["rows"]:
        assert row["first_action"] in (0, 1, 2, 3)
        assert math.isfinite(row["optimal_value"])
        assert row["reachable_belief_physical_states"] > 4


def test_unknown_model_physics_conserves_shared_weekly_capacity() -> None:
    model = ExactProductMixPOMDP(ExactPOMDPConfig(horizon=2))
    for action in range(4):
        for demand_c in range(4):
            inventory_c, inventory_h, backlog_c, backlog_h = model.physical_transition(
                (0, 0, 0, 0), action, demand_c
            )
            assert inventory_c + inventory_h == backlog_c + backlog_h
