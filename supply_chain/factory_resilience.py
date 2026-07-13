"""Pure Garrido et al. (2024) factory-resilience transformation.

This module deliberately contains no MFSC/DES variable mapping.  It reproduces
Equations 3--6 only when callers supply positive factory-APP components with the
paper's semantics.  A caller that maps military-distribution state into these
components is performing a researcher adaptation and must label it as such.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable


PUBLISHED_EXPONENTS = {
    "zeta": 0.024,
    "epsilon": 0.026,
    "phi": 0.040,
    "tau": 0.060,
    "kappa_dot": 0.1771,
}


@dataclass(frozen=True)
class FactoryResilienceComponents:
    """Positive five-variable APP ledger used by the published index."""

    zeta: float
    epsilon: float
    phi: float
    tau: float
    kappa_dot: float

    def validated(self) -> "FactoryResilienceComponents":
        for name, value in vars(self).items():
            numeric = float(value)
            if not math.isfinite(numeric) or numeric <= 0.0:
                raise ValueError(f"{name} must be finite and strictly positive")
        return self


def normalize_strategy_cost(
    strategy_cost: float,
    all_strategy_costs: Iterable[float],
) -> float:
    """Return the paper's ``N * kappa_i / sum(kappa)`` cost normalization.

    The paper uses ``N=7`` because it evaluates seven pure APP substrategies.
    Keeping ``N`` equal to the supplied ensemble size makes the transformation
    explicit and prevents silently substituting a Monte-Carlo mean-cost proxy.
    """

    costs = [float(value) for value in all_strategy_costs]
    if not costs:
        raise ValueError("all_strategy_costs must contain at least one strategy")
    if any(not math.isfinite(value) or value <= 0.0 for value in costs):
        raise ValueError("all strategy costs must be finite and strictly positive")
    selected = float(strategy_cost)
    if not math.isfinite(selected) or selected <= 0.0:
        raise ValueError("strategy_cost must be finite and strictly positive")
    if selected not in costs:
        raise ValueError("strategy_cost must be a member of all_strategy_costs")
    return float(len(costs) * selected / math.fsum(costs))


def published_log_score(components: FactoryResilienceComponents) -> float:
    """Reproduce the published Equation 5 log-linear score exactly."""

    c = components.validated()
    return float(
        PUBLISHED_EXPONENTS["zeta"] * math.log(c.zeta)
        - PUBLISHED_EXPONENTS["epsilon"] * math.log(c.epsilon)
        + PUBLISHED_EXPONENTS["phi"] * math.log(c.phi)
        - PUBLISHED_EXPONENTS["tau"] * math.log(c.tau)
        - PUBLISHED_EXPONENTS["kappa_dot"] * math.log(c.kappa_dot)
    )


def published_raw_product(components: FactoryResilienceComponents) -> float:
    """Reproduce the multiplicative Cobb-Douglas form in Equation 3."""

    return float(math.exp(published_log_score(components)))


def published_sigmoid_index(components: FactoryResilienceComponents) -> float:
    """Reproduce the bounded Equation 6 factory-resilience index."""

    score = published_log_score(components)
    if score >= 0.0:
        return float(1.0 / (1.0 + math.exp(-score)))
    exp_score = math.exp(score)
    return float(exp_score / (1.0 + exp_score))


def compute_published_factory_resilience(
    components: FactoryResilienceComponents,
) -> dict[str, float | dict[str, float] | str]:
    """Return an audit-friendly exact-formula payload without DES adaptation."""

    score = published_log_score(components)
    return {
        "construct": "Garrido_et_al_2024_factory_APP",
        "formula_scope": "published_equations_3_to_6_only",
        "log_score": score,
        "raw_product": float(math.exp(score)),
        "sigmoid_index": published_sigmoid_index(components),
        "published_exponents": dict(PUBLISHED_EXPONENTS),
    }
