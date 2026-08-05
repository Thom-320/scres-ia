"""Small explicit structured planners used by the E* engineering preflight.

These planners are deliberately modest.  They do not establish a scientific
frontier or an optimality result; they make the measured backend explicit and
auditable before any fresh data or learner is authorized.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class PlannerResult:
    best_score: float
    rollouts: int
    des_calls: int
    solver_iterations: int


class DirectDESMPC:
    """Enumerative receding-horizon controller over a frozen candidate set.

    Each candidate is evaluated by a fresh DES rollout.  The objective is
    supplied by the caller so the engineering preflight does not silently
    choose a paper metric.  The class is a compute backend, not a claim of
    global optimality.
    """

    name = "direct_DES_MPC"

    def __init__(self, *, candidate_multiplier: int = 3) -> None:
        if int(candidate_multiplier) < 1:
            raise ValueError("candidate_multiplier must be positive")
        self.candidate_multiplier = int(candidate_multiplier)

    def plan(
        self,
        evaluate_candidate: Callable[[int], float],
        candidate_count: int,
    ) -> PlannerResult:
        count = int(candidate_count) * self.candidate_multiplier
        if count < 1:
            raise ValueError("candidate_count must be positive")
        best = float("-inf")
        for index in range(count):
            score = float(evaluate_candidate(index))
            if score > best:
                best = score
        return PlannerResult(
            best_score=best,
            rollouts=count,
            des_calls=count,
            solver_iterations=count,
        )
