"""Confidence-gated interfaces for Program T learning-augmented MPC.

This module deliberately contains no DES or neural-network implementation.  It
defines the deployable boundary between a learned estimator and a physically
feasible planner, and keeps the classical proposal as the fail-closed fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, Sequence

import numpy as np


ObservationHistory = Sequence[Sequence[float]]
PhysicalState = Sequence[float]


@dataclass(frozen=True)
class PlannerBudget:
    max_online_ms: int
    horizon: int
    scenario_limit: int
    solver_tolerance: float

    def __post_init__(self) -> None:
        if self.max_online_ms <= 0 or self.horizon <= 0 or self.scenario_limit <= 0:
            raise ValueError("planner budget values must be positive")
        if self.solver_tolerance < 0:
            raise ValueError("solver_tolerance must be nonnegative")


@dataclass(frozen=True)
class BeliefValueEstimate:
    corrected_belief: tuple[float, ...]
    ret_quantiles: tuple[float, ...]
    improvement_lcb: float
    worst_product_risk_ucb: float
    in_calibration_support: bool

    def __post_init__(self) -> None:
        if not self.corrected_belief or not np.isclose(sum(self.corrected_belief), 1.0):
            raise ValueError("corrected_belief must be a nonempty probability vector")
        if any(value < 0 or value > 1 for value in self.corrected_belief):
            raise ValueError("corrected_belief probabilities must lie in [0, 1]")
        if not self.ret_quantiles:
            raise ValueError("ret_quantiles must be nonempty")
        if tuple(sorted(self.ret_quantiles)) != self.ret_quantiles:
            raise ValueError("ret_quantiles must be nondecreasing")


@dataclass(frozen=True)
class PlannerProposal:
    action: int
    predicted_value: float
    predicted_worst_product_shortfall: float
    runtime_ms: float
    feasible: bool = True

    def __post_init__(self) -> None:
        if self.action not in (0, 1, 2, 3):
            raise ValueError("Program T action must be in {0,1,2,3}")
        if self.runtime_ms < 0:
            raise ValueError("runtime_ms must be nonnegative")


@dataclass(frozen=True)
class TrustDecision:
    action: int
    mode: Literal["classical", "augmented", "fallback"]
    improvement_lcb: float
    constraint_risk_ucb: float
    runtime_ms: float
    fallback_reason: str | None


class Planner(Protocol):
    def propose(
        self,
        history: ObservationHistory,
        physical_state: PhysicalState,
        budget: PlannerBudget,
    ) -> PlannerProposal: ...


class BeliefValueEstimator(Protocol):
    def estimate(
        self,
        history: ObservationHistory,
        physical_state: PhysicalState,
        nominal: PlannerProposal,
        augmented: PlannerProposal,
    ) -> BeliefValueEstimate: ...


class LearningAugmentedPlanner(Protocol):
    def act(
        self,
        history: ObservationHistory,
        physical_state: PhysicalState,
        budget: PlannerBudget,
    ) -> TrustDecision: ...


@dataclass(frozen=True)
class ConfidenceGateConfig:
    worst_product_margin: float = 0.02
    require_strict_improvement: bool = True

    def __post_init__(self) -> None:
        if self.worst_product_margin < 0:
            raise ValueError("worst_product_margin must be nonnegative")


class SplitConformalBounds:
    """Finite-sample residual bounds with the conservative `higher` quantile."""

    def __init__(self, alpha: float = 0.05) -> None:
        if not 0 < alpha < 1:
            raise ValueError("alpha must lie strictly between zero and one")
        self.alpha = alpha
        self._improvement_error: float | None = None
        self._risk_error: float | None = None

    @staticmethod
    def _finite_sample_quantile(values: Sequence[float], alpha: float) -> float:
        array = np.asarray(values, dtype=float)
        if array.ndim != 1 or len(array) == 0 or not np.all(np.isfinite(array)):
            raise ValueError("calibration residuals must be a finite nonempty vector")
        rank = min(len(array), int(np.ceil((len(array) + 1) * (1 - alpha))))
        return float(np.partition(array, rank - 1)[rank - 1])

    def fit(
        self,
        *,
        predicted_improvement: Sequence[float],
        realized_improvement: Sequence[float],
        predicted_risk: Sequence[float],
        realized_risk: Sequence[float],
    ) -> "SplitConformalBounds":
        pred_i = np.asarray(predicted_improvement, dtype=float)
        real_i = np.asarray(realized_improvement, dtype=float)
        pred_r = np.asarray(predicted_risk, dtype=float)
        real_r = np.asarray(realized_risk, dtype=float)
        if pred_i.shape != real_i.shape or pred_r.shape != real_r.shape:
            raise ValueError("predicted and realized calibration vectors must align")
        self._improvement_error = self._finite_sample_quantile(pred_i - real_i, self.alpha)
        self._risk_error = self._finite_sample_quantile(real_r - pred_r, self.alpha)
        return self

    @property
    def fitted(self) -> bool:
        return self._improvement_error is not None and self._risk_error is not None

    def improvement_lcb(self, predicted: float) -> float:
        if self._improvement_error is None:
            raise RuntimeError("conformal bounds are not fitted")
        return float(predicted - self._improvement_error)

    def risk_ucb(self, predicted: float) -> float:
        if self._risk_error is None:
            raise RuntimeError("conformal bounds are not fitted")
        return float(predicted + self._risk_error)


class ConfidenceGatedBeliefValueMPC:
    """Select an augmented proposal only when every local gate passes.

    This is an abstention mechanism, not a proof of episode-level safety.
    Simultaneous worst-product and lost-demand guarantees still require frozen
    end-to-end evaluation because repeated locally admissible actions can have
    persistent joint consequences.
    """

    def __init__(
        self,
        *,
        nominal_planner: Planner,
        augmented_planner: Planner,
        estimator: BeliefValueEstimator,
        config: ConfidenceGateConfig = ConfidenceGateConfig(),
    ) -> None:
        self.nominal_planner = nominal_planner
        self.augmented_planner = augmented_planner
        self.estimator = estimator
        self.config = config

    def act(
        self,
        history: ObservationHistory,
        physical_state: PhysicalState,
        budget: PlannerBudget,
    ) -> TrustDecision:
        nominal = self.nominal_planner.propose(history, physical_state, budget)
        if not nominal.feasible:
            raise RuntimeError("nominal fail-closed planner returned an infeasible action")
        augmented = self.augmented_planner.propose(history, physical_state, budget)
        runtime = nominal.runtime_ms + augmented.runtime_ms
        if runtime > budget.max_online_ms:
            return self._fallback(nominal, runtime, "online_budget_exceeded")
        if not augmented.feasible:
            return self._fallback(nominal, runtime, "augmented_infeasible")
        estimate = self.estimator.estimate(history, physical_state, nominal, augmented)
        if not estimate.in_calibration_support:
            return self._fallback(nominal, runtime, "outside_calibration_support", estimate)
        if estimate.worst_product_risk_ucb > self.config.worst_product_margin:
            return self._fallback(nominal, runtime, "worst_product_risk", estimate)
        improvement_passes = (
            estimate.improvement_lcb > 0
            if self.config.require_strict_improvement
            else estimate.improvement_lcb >= 0
        )
        if not improvement_passes:
            return self._fallback(nominal, runtime, "no_certified_improvement", estimate)
        return TrustDecision(
            action=augmented.action,
            mode="augmented",
            improvement_lcb=estimate.improvement_lcb,
            constraint_risk_ucb=estimate.worst_product_risk_ucb,
            runtime_ms=runtime,
            fallback_reason=None,
        )

    @staticmethod
    def _fallback(
        nominal: PlannerProposal,
        runtime: float,
        reason: str,
        estimate: BeliefValueEstimate | None = None,
    ) -> TrustDecision:
        return TrustDecision(
            action=nominal.action,
            mode="fallback",
            improvement_lcb=float("-inf") if estimate is None else estimate.improvement_lcb,
            constraint_risk_ucb=(
                float("inf") if estimate is None else estimate.worst_product_risk_ucb
            ),
            runtime_ms=runtime,
            fallback_reason=reason,
        )
