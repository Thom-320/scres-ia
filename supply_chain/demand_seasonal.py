"""Seasonal pseudo-stochastic demand, per Garrido, Pongutá & García-Reyes (2024) IJPR §3.2.

WHAT THE SOURCE FIXES, and what it leaves to us. Equation (1) of that paper is

    GR_{t+v} = a*D_t + (1-a)(F_t + d_t) + d_{t+1} + g(F_{t+1} - F_t) + (1-g)*d_t

with `a` (level) and `g` (trend) drawn U[0,1] per Monte-Carlo run, seeded by a 36-value seasonal
series, over a 36-week horizon. Figure 3 reports mean 819.13, sd 174.51 -- CV 21.3% -- min 0 and
max 1335.

READING OF EQ (1), declared. The last three terms are Holt's trend update written out:

    d_{t+1} = g(F_{t+1} - F_t) + (1-g)*d_t

so taking Eq (1) at face value sums d_{t+1} twice and yields F_{t+1} + 2*d_{t+1}, which is not a
forecasting method anyone uses. We implement the standard reading, GR = F_{t+1} + d_{t+1} -- the
one-step-ahead Holt forecast -- and record the ambiguity as a question for Garrido rather than
silently picking. `double_trend=True` reproduces the literal sum for anyone who wants to check.

WHERE THE SEASONALITY LIVES, and why that matters. Eq (1) is Holt's LINEAR TREND method: it has a
level and a trend term and NO seasonal term. The seasonality is in the seed series, not in the
smoother. So the forecast is structurally unable to track a seasonal turn -- it always lags the
trough and overshoots the recovery. That is not a defect of this implementation; it is what makes
GR an *imperfect* observable signal of a state that genuinely exists, which is precisely the
structure the preregistration needs and the one op12 never had.

THE SCALE IS OURS, NEVER HIS. His mean is 819 and ours is 2500/day. Copying his level would break
the calibration against Garrido-Ríos (2017) that validates this DES. We transplant the CV and the
shape and keep our own level -- the same rule that governs the Cobb-Douglas port.

Preregistration: docs/PREREGISTRO_DEMANDA_ESTACIONAL_P2_2026-08-07.md
"""
from __future__ import annotations

from dataclasses import dataclass, field
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
    # Garrido's Figure 3 reaches 0. We do NOT force zero-demand weeks: a week of literally zero
    # rations is a strong domain claim we have no basis for, and it would make ret_excel's
    # censoring pathological. Declared as a departure, not an oversight.
    forecast_seed_periods: int = 36

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
    """One episode's demand path: a seasonal multiplier plus a Holt forecast of it.

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
        # Holt state, initialised on the profile's own mean so the forecast starts unbiased rather
        # than warming up through a transient that would masquerade as forecast error.
        self._F: Optional[float] = None
        self._delta: float = 0.0
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
            self.forecast_history.append(
                {"week": float(w), "realised": float(realised), "F": self._F,
                 "delta": self._delta, "gr": self.gross_requirements()})
            return
        F_prev = self._F
        F_new = self.alpha * float(realised) + (1.0 - self.alpha) * (F_prev + self._delta)
        self._delta = self.gamma * (F_new - F_prev) + (1.0 - self.gamma) * self._delta
        self._F = F_new
        self.forecast_history.append(
            {"week": float(w), "realised": float(realised), "F": self._F,
             "delta": self._delta, "gr": self.gross_requirements()})

    def gross_requirements(self) -> float:
        """`GR_{t+v}`: the one-step-ahead Holt forecast. NaN until the first week is observed."""
        if self._F is None:
            return float("nan")
        trend = 2.0 * self._delta if self.contract.double_trend else self._delta
        return float(self._F + trend)
