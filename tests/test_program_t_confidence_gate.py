from __future__ import annotations

from dataclasses import replace

import pytest

from supply_chain.program_t_confidence_gated_mpc import (
    BeliefValueEstimate,
    ConfidenceGatedBeliefValueMPC,
    PlannerBudget,
    PlannerProposal,
    SplitConformalBounds,
)


class FixedPlanner:
    def __init__(self, proposal: PlannerProposal) -> None:
        self.proposal = proposal

    def propose(self, history, physical_state, budget):
        return self.proposal


class FixedEstimator:
    def __init__(self, estimate: BeliefValueEstimate) -> None:
        self.value = estimate

    def estimate(self, history, physical_state, nominal, augmented):
        return self.value


def estimate(**changes) -> BeliefValueEstimate:
    base = BeliefValueEstimate(
        corrected_belief=(0.4, 0.6),
        ret_quantiles=tuple(float(index) for index in range(19)),
        improvement_lcb=0.01,
        worst_product_risk_ucb=0.01,
        in_calibration_support=True,
    )
    return replace(base, **changes)


def controller(value: BeliefValueEstimate, *, augmented_runtime: float = 2, feasible=True):
    return ConfidenceGatedBeliefValueMPC(
        nominal_planner=FixedPlanner(PlannerProposal(1, 0.5, 0.01, 2)),
        augmented_planner=FixedPlanner(
            PlannerProposal(3, 0.6, 0.01, augmented_runtime, feasible=feasible)
        ),
        estimator=FixedEstimator(value),
    )


def test_augmented_action_requires_every_gate() -> None:
    budget = PlannerBudget(10, 4, 256, 1e-9)
    assert controller(estimate()).act([], [], budget).action == 3
    assert controller(estimate()).act([], [], budget).mode == "augmented"
    assert controller(estimate(improvement_lcb=0)).act([], [], budget).action == 1
    assert controller(estimate(worst_product_risk_ucb=0.021)).act([], [], budget).action == 1
    assert controller(estimate(in_calibration_support=False)).act([], [], budget).action == 1
    assert controller(estimate(), augmented_runtime=20).act([], [], budget).action == 1
    assert controller(estimate(), feasible=False).act([], [], budget).action == 1


def test_nominal_infeasibility_is_terminal() -> None:
    value = estimate()
    model = ConfidenceGatedBeliefValueMPC(
        nominal_planner=FixedPlanner(PlannerProposal(1, 0, 0, 1, feasible=False)),
        augmented_planner=FixedPlanner(PlannerProposal(2, 0, 0, 1)),
        estimator=FixedEstimator(value),
    )
    with pytest.raises(RuntimeError, match="nominal fail-closed"):
        model.act([], [], PlannerBudget(10, 4, 256, 1e-9))


def test_split_conformal_bounds_are_conservative_and_deterministic() -> None:
    bounds = SplitConformalBounds(alpha=0.2).fit(
        predicted_improvement=[0.2, 0.2, 0.2, 0.2],
        realized_improvement=[0.1, 0.15, 0.18, 0.2],
        predicted_risk=[0.01, 0.01, 0.01, 0.01],
        realized_risk=[0.01, 0.02, 0.03, 0.04],
    )
    assert bounds.fitted
    assert bounds.improvement_lcb(0.2) == pytest.approx(0.1)
    assert bounds.risk_ucb(0.01) == pytest.approx(0.04)


def test_quantile_count_is_not_prematurely_frozen() -> None:
    short = estimate(ret_quantiles=(0.1, 0.2, 0.3))
    assert short.ret_quantiles == (0.1, 0.2, 0.3)
