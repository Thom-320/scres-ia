from __future__ import annotations

import pytest

from supply_chain.estar_planners import DirectDESMPC


def test_direct_des_mpc_evaluates_frozen_candidates_and_selects_best() -> None:
    seen: list[int] = []
    planner = DirectDESMPC(candidate_multiplier=2)
    result = planner.plan(lambda index: seen.append(index) or -abs(index - 3), 3)
    assert seen == [0, 1, 2, 3, 4, 5]
    assert result.best_score == pytest.approx(0.0)
    assert result.rollouts == 6
    assert result.des_calls == 6
    assert result.solver_iterations == 6


def test_direct_des_mpc_rejects_empty_candidate_set() -> None:
    with pytest.raises(ValueError, match="candidate_count"):
        DirectDESMPC().plan(lambda _: 0.0, 0)
