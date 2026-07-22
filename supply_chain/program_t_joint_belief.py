"""Exact causal belief over Program Q demand-model parameters and latent regime.

The state is ``p(theta, Z_t | y_{1:t-1})`` at the start of decision week ``t``.
Only completed prior-week product counts update the posterior.  True parameters
may initialize the oracle arm, but the current latent regime is never exposed.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping

import numpy as np


THETA_GRID: tuple[tuple[float, float], ...] = (
    (0.75, 0.90),
    (0.90, 0.75),
    (0.90, 0.90),
)


def _binomial_probability(count_c: int, *, regime_c: bool, share: float) -> float:
    probability_c = float(share) if regime_c else 1.0 - float(share)
    return math.comb(6, int(count_c)) * probability_c**count_c * (1.0 - probability_c) ** (6 - count_c)


@dataclass
class ExactJointBelief:
    """Six-state exact filter with theta support fixed before evaluation."""

    probability: np.ndarray

    @classmethod
    def uniform(cls) -> "ExactJointBelief":
        return cls(np.full((len(THETA_GRID), 2), 1.0 / (2 * len(THETA_GRID))))

    @classmethod
    def oracle_parameters(cls, theta: tuple[float, float]) -> "ExactJointBelief":
        if theta not in THETA_GRID:
            raise ValueError("oracle theta must belong to the frozen grid")
        probability = np.zeros((len(THETA_GRID), 2), dtype=float)
        probability[THETA_GRID.index(theta), :] = 0.5
        return cls(probability)

    @classmethod
    def from_theta_marginal(
        cls,
        marginal: Iterable[float],
        *,
        probability_regime_c: float = 0.5,
    ) -> "ExactJointBelief":
        """Build a campaign-start belief without leaking the new regime.

        Q-R1 retains only knowledge about the campaign parameter at the first
        gate.  The physical reset gives the new latent regime its public prior.
        """
        theta = np.asarray(tuple(marginal), dtype=float)
        if theta.shape != (len(THETA_GRID),):
            raise ValueError("theta marginal must have length three")
        if np.any(theta < 0.0) or not np.isfinite(theta).all() or theta.sum() <= 0.0:
            raise ValueError("invalid theta marginal")
        probability_c = float(probability_regime_c)
        if not 0.0 <= probability_c <= 1.0:
            raise ValueError("regime probability must be in [0, 1]")
        theta /= theta.sum()
        probability = np.column_stack(
            (theta * (1.0 - probability_c), theta * probability_c)
        )
        return cls(probability)

    def __post_init__(self) -> None:
        self.probability = np.asarray(self.probability, dtype=float).copy()
        if self.probability.shape != (len(THETA_GRID), 2):
            raise ValueError("joint belief must have shape (3,2)")
        if np.any(self.probability < 0.0) or not np.isfinite(self.probability).all():
            raise ValueError("invalid joint belief")
        total = float(self.probability.sum())
        if total <= 0.0:
            raise ValueError("joint belief has zero support")
        self.probability /= total

    def copy(self) -> "ExactJointBelief":
        return ExactJointBelief(self.probability)

    def observe_previous_week(self, count_c: int) -> None:
        """Condition on y_(t-1), then transition Z_(t-1) to Z_t."""
        if not 0 <= int(count_c) <= 6:
            raise ValueError("weekly C count must be in 0..6")
        posterior_previous = self.probability.copy()
        for theta_index, (_rho, share) in enumerate(THETA_GRID):
            for regime_index in (0, 1):
                posterior_previous[theta_index, regime_index] *= _binomial_probability(
                    int(count_c), regime_c=bool(regime_index), share=share
                )
        normalizer = float(posterior_previous.sum())
        if normalizer <= 0.0:
            raise RuntimeError("observation has zero probability under theta grid")
        posterior_previous /= normalizer
        current = np.zeros_like(posterior_previous)
        for theta_index, (rho, _share) in enumerate(THETA_GRID):
            current[theta_index, 0] = rho * posterior_previous[theta_index, 0] + (1.0 - rho) * posterior_previous[theta_index, 1]
            current[theta_index, 1] = (1.0 - rho) * posterior_previous[theta_index, 0] + rho * posterior_previous[theta_index, 1]
        self.probability = current / current.sum()

    def observe_campaign(self, counts_c: Iterable[int]) -> None:
        """Consume every causal weekly count before exporting knowledge."""
        for count_c in counts_c:
            self.observe_previous_week(int(count_c))

    def between_campaign_transition(self, persistence: float) -> "ExactJointBelief":
        """Propagate theta knowledge and reset the new physical regime to 0.5.

        With three theta states, the non-stay mass is distributed uniformly
        over the other two states.  ``persistence=1/3`` is therefore the iid
        uniform transition, not 0.5.
        """
        kappa = float(persistence)
        if not 0.0 <= kappa <= 1.0:
            raise ValueError("persistence must be in [0, 1]")
        old = np.asarray(self.theta_marginal, dtype=float)
        count = len(old)
        if count < 2:
            raise RuntimeError("theta grid must contain at least two states")
        new = kappa * old + ((1.0 - kappa) / (count - 1)) * (1.0 - old)
        return ExactJointBelief.from_theta_marginal(new, probability_regime_c=0.5)

    @property
    def probability_regime_c(self) -> float:
        return float(self.probability[:, 1].sum())

    @property
    def theta_marginal(self) -> tuple[float, ...]:
        return tuple(map(float, self.probability.sum(axis=1)))

    def sample_states(self, *, count: int, seed: int) -> tuple[tuple[float, float, bool], ...]:
        if count <= 0:
            raise ValueError("count must be positive")
        rng = np.random.default_rng(int(seed))
        flat = self.probability.reshape(-1)
        indices = rng.choice(len(flat), size=int(count), p=flat)
        return tuple(
            (*THETA_GRID[int(index) // 2], bool(int(index) % 2))
            for index in indices
        )

    def enumerate_states(self) -> tuple[tuple[float, float, bool, float], ...]:
        """Return all six latent states with their exact posterior weights."""
        return tuple(
            (
                float(rho),
                float(share),
                bool(regime_index),
                float(self.probability[theta_index, regime_index]),
            )
            for theta_index, (rho, share) in enumerate(THETA_GRID)
            for regime_index in (0, 1)
        )

    def as_dict(self) -> Mapping[str, object]:
        return {
            "theta_grid": [list(theta) for theta in THETA_GRID],
            "joint": self.probability.tolist(),
            "theta_marginal": list(self.theta_marginal),
            "probability_regime_c": self.probability_regime_c,
        }


def weekly_product_counts(
    *, order_times: Iterable[float], order_products: Iterable[str], decision_start: float, weeks: int
) -> tuple[int, ...]:
    counts = [0 for _ in range(int(weeks))]
    for time_value, product in zip(order_times, order_products):
        week = int((float(time_value) - float(decision_start)) // 168.0)
        if 0 <= week < weeks and str(product) == "P_C":
            counts[week] += 1
    return tuple(counts)
