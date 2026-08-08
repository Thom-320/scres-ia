"""Seasonal pseudo-stochastic demand, per Garrido, Pongutá & García-Reyes (2024) IJPR §3.2.

WHAT THE SOURCE FIXES, and what it leaves to us. Equation (1) of that paper is

    GR_{t+v} = a*D_t + (1-a)(F_t + d_t) + d_{t+1} + g(F_{t+1} - F_t) + (1-g)*d_t

with `a` (level) and `g` (trend) drawn U[0,1] per Monte-Carlo run, seeded by a 36-value seasonal
series, over a 36-week horizon. Figure 3 reports mean 819.13, sd 174.51 -- CV 21.3% -- min 0 and
max 1335.

READING OF EQ (1), declared. Garrido uses GR as a gross-requirements input/generator for the
decision variables, not as a forecast that is scored against a separately realised series. The
source equation is therefore implemented here as a trajectory generator. It is not adjudicated by
forecast correlation. A separate, explicitly researcher-defined Holt-Winters mode is available
when an observable forecast signal is required.

The last three terms are Holt's trend update written out:

    d_{t+1} = g(F_{t+1} - F_t) + (1-g)*d_t

so taking Eq (1) at face value can sum the trend update twice. The default
`forecast_mode="garrido_generator"` keeps the standard single-update trajectory used by the
existing characterisation; `double_trend=True` exposes the literal doubled term for source audit.
Neither mode is called a validated forecast.

WHERE THE SEASONALITY LIVES, and why that matters. The source generator inherits shape from its
seasonal seed profile. The optional `forecast_mode="holt_winters_observable"` adds a seasonal
component to a researcher-defined observable signal. That extension is our instrument, not a
repair of Garrido's equation, and its forecast skill must be scored at the declared horizon.

THE SCALE IS OURS, NEVER HIS. His mean is 819 and ours is 2500/day. Copying his level would break
the calibration against Garrido-Ríos (2017) that validates this DES. We transplant the CV and the
shape and keep our own level -- the same rule that governs the Cobb-Douglas port.

Preregistration: docs/PREREGISTRO_DEMANDA_ESTACIONAL_P2_2026-08-07.md
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

HOURS_PER_WEEK = 168.0


@dataclass(frozen=True)
class SeasonalDemandContract:
    """The seed profile and the smoothing ranges. Every field is OUR declared assumption.

    The profile is a multiplier on the thesis daily draw, so the native U(2400, 2600) machinery
    keeps producing the base series and this only reshapes its envelope. `plateau_scale` is not
    stored: it is DERIVED so the profile averages exactly 1.0, which is what keeps the mean demand
    equal to the thesis mean and the 2017 calibration intact.
    """

    period_weeks: int = 12
    trough_weeks: int = 1
    trough_scale: float = 0.35
    alpha_range: tuple[float, float] = (0.0, 1.0)
    gamma_range: tuple[float, float] = (0.0, 1.0)
    double_trend: bool = False
    forecast_mode: str = "garrido_generator"
    seasonal_beta: float = 0.20
    # Garrido's Figure 3 reaches 0. We do NOT force zero-demand weeks: a week of literally zero
    # rations is a strong domain claim we have no basis for, and it would make ret_excel's
    # censoring pathological. Declared as a departure, not an oversight.
    forecast_seed_periods: int = 36

    def __post_init__(self) -> None:
        if self.forecast_mode not in {"garrido_generator", "holt_winters_observable"}:
            raise ValueError(
                "forecast_mode must be 'garrido_generator' or "
                "'holt_winters_observable'"
            )
        if not 0.0 <= self.seasonal_beta <= 1.0:
            raise ValueError("seasonal_beta must lie in [0, 1]")
        if self.period_weeks < 2:
            raise ValueError("period_weeks must be at least 2")

    def plateau_scale(self) -> float:
        n_trough = max(0, min(self.trough_weeks, self.period_weeks))
        n_plateau = self.period_weeks - n_trough
        if n_plateau <= 0:
            return 1.0
        return (self.period_weeks - n_trough * self.trough_scale) / n_plateau

    def profile(self) -> np.ndarray:
        """One seasonal cycle of multipliers, mean exactly 1.0 by construction."""
        p = np.full(self.period_weeks, self.plateau_scale(), dtype=float)
        n_trough = max(0, min(self.trough_weeks, self.period_weeks))
        if n_trough:
            # Trough placed at the end of the cycle so week 0 starts on the plateau, matching
            # Figure 3 where the first drop appears well after t=0.
            p[-n_trough:] = self.trough_scale
        return p

    def profile_cv(self) -> float:
        p = self.profile()
        return float(p.std() / p.mean())


class SeasonalDemandProcess:
    """One episode's demand path and declared GR instrument.

    `alpha` and `gamma` are drawn ONCE per episode from U[0,1], which is Garrido's Monte-Carlo
    device: the family of paths comes from the smoothing parameters varying across runs, not
    within one.
    """

    def __init__(self, contract: SeasonalDemandContract, rng: np.random.Generator) -> None:
        self.contract = contract
        self._profile = contract.profile()
        lo, hi = contract.alpha_range
        self.alpha = float(rng.uniform(lo, hi))
        lo, hi = contract.gamma_range
        self.gamma = float(rng.uniform(lo, hi))
        # Holt state. The source-faithful mode is a generator; only the explicit observable mode
        # adds a learned seasonal component. Both use the same alpha/gamma draw for comparability.
        self._F: Optional[float] = None
        self._delta: float = 0.0
        self._seasonal = np.zeros(contract.period_weeks, dtype=float)
        self._last_week: Optional[int] = None
        self.forecast_history: list[dict[str, float]] = []

    # -- the state -------------------------------------------------------------------------
    def week_index(self, hours: float) -> int:
        return int(hours // HOURS_PER_WEEK)

    def scale(self, hours: float) -> float:
        """Seasonal multiplier in force at `hours`. Deterministic given the week."""
        return float(self._profile[self.week_index(hours) % self.contract.period_weeks])

    def phase(self, hours: float) -> int:
        return int(self.week_index(hours) % self.contract.period_weeks)

    # -- the forecast ----------------------------------------------------------------------
    def observe(self, hours: float, realised: float) -> None:
        """Feed one realised weekly demand and advance the Holt state.

        Called at most once per week; a second call inside the same week is ignored so the
        smoother cannot be advanced six times by six daily orders, which would make its effective
        alpha depend on the ordering calendar rather than on the contract.
        """
        w = self.week_index(hours)
        if self._last_week is not None and w <= self._last_week:
            return
        self._last_week = w
        if self._F is None:
            self._F = float(realised)
            self._delta = 0.0
            phase = w % self.contract.period_weeks
            self._seasonal[phase] = 0.0
            self.forecast_history.append(
                {"week": float(w), "realised": float(realised), "F": self._F,
                 "delta": self._delta, "phase": float(phase),
                 "gr": self.gross_requirements()})
            return
        F_prev = self._F
        phase = w % self.contract.period_weeks
        seasonal_prev = self._seasonal[phase]
        F_new = self.alpha * (float(realised) - seasonal_prev) + (1.0 - self.alpha) * (
            F_prev + self._delta
        )
        delta_new = self.gamma * (F_new - F_prev) + (1.0 - self.gamma) * self._delta
        if self.contract.forecast_mode == "holt_winters_observable":
            beta = self.contract.seasonal_beta
            self._seasonal[phase] = beta * (float(realised) - F_new) + (
                1.0 - beta
            ) * seasonal_prev
        self._delta = delta_new
        self._F = F_new
        self.forecast_history.append(
            {"week": float(w), "realised": float(realised), "F": self._F,
             "delta": self._delta, "phase": float(phase),
             "gr": self.gross_requirements()})

    def gross_requirements(self) -> float:
        """Return the next-period GR value for the selected source/instrument mode.

        In ``garrido_generator`` this is a trajectory input. In
        ``holt_winters_observable`` it is our researcher-defined observable forecast signal.
        """
        if self._F is None:
            return float("nan")
        trend = 2.0 * self._delta if self.contract.double_trend else self._delta
        seasonal = 0.0
        if self.contract.forecast_mode == "holt_winters_observable":
            phase = (self._last_week + 1) % self.contract.period_weeks
            seasonal = float(self._seasonal[phase])
        return float(self._F + trend + seasonal)

    @property
    def forecast_mode(self) -> str:
        return self.contract.forecast_mode
