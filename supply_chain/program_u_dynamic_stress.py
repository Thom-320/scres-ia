"""Policy-independent dynamic stress tapes for Program U.

The tape is a researcher-introduced scenario generator.  It does not claim to
estimate MFSC risk frequencies.  Regime and potential processing-time streams
are generated before a policy is evaluated and are therefore common random
numbers across every comparator.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class StressRegime:
    name: str
    risk_frequency_multipliers: Mapping[str, float]
    risk_impact_multipliers: Mapping[str, float]
    pt_log_sigma: float


@dataclass(frozen=True)
class DynamicStressTape:
    seed: int
    cadence_hours: float
    regimes: tuple[str, ...]
    potential_pt_multipliers: tuple[float, ...]
    sha256: str


def _digest(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def validate_transition_matrix(
    names: Sequence[str], matrix: Sequence[Sequence[float]]
) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    if values.shape != (len(names), len(names)):
        raise ValueError("transition matrix shape does not match regime names")
    if np.any(values < 0.0) or not np.allclose(values.sum(axis=1), 1.0, atol=1e-12):
        raise ValueError("transition rows must be nonnegative and sum to one")
    return values


def generate_dynamic_stress_tape(
    *,
    seed: int,
    regimes: Sequence[StressRegime],
    transition_matrix: Sequence[Sequence[float]],
    periods: int,
    potential_service_draws_per_period: int,
    cadence_hours: float = 168.0,
    initial_regime: str = "normal",
) -> DynamicStressTape:
    """Generate latent regimes and mean-one potential processing times.

    The processing multiplier is lognormal with
    ``exp(sigma*z - sigma**2/2)``, hence expectation one for every sigma.  The
    complete potential stream is generated independent of policy consumption.
    A later DES adapter must index this stream by immutable operation/event
    identity rather than consuming a mutable global RNG.
    """
    if periods <= 0 or potential_service_draws_per_period <= 0 or cadence_hours <= 0:
        raise ValueError("periods, service draws and cadence must be positive")
    by_name = {row.name: row for row in regimes}
    if len(by_name) != len(regimes) or initial_regime not in by_name:
        raise ValueError("regime names must be unique and include initial_regime")
    names = tuple(row.name for row in regimes)
    matrix = validate_transition_matrix(names, transition_matrix)
    regime_ss, pt_ss = np.random.SeedSequence(int(seed)).spawn(2)
    regime_rng = np.random.default_rng(regime_ss)
    pt_rng = np.random.default_rng(pt_ss)
    current = names.index(initial_regime)
    path: list[str] = []
    multipliers: list[float] = []
    for _ in range(int(periods)):
        name = names[current]
        path.append(name)
        sigma = float(by_name[name].pt_log_sigma)
        if sigma < 0.0:
            raise ValueError("pt_log_sigma must be nonnegative")
        z = pt_rng.normal(size=int(potential_service_draws_per_period))
        values = np.exp(sigma * z - 0.5 * sigma * sigma)
        multipliers.extend(map(float, values))
        current = int(regime_rng.choice(len(names), p=matrix[current]))
    payload = {
        "seed": int(seed),
        "cadence_hours": float(cadence_hours),
        "regimes": path,
        "potential_pt_multipliers": multipliers,
    }
    return DynamicStressTape(
        seed=int(seed),
        cadence_hours=float(cadence_hours),
        regimes=tuple(path),
        potential_pt_multipliers=tuple(multipliers),
        sha256=_digest(payload),
    )

