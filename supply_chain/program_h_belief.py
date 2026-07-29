"""Program H — exact augmented semi-Markov belief filter + belief-aware policies (O0).

Per the frozen amendment (docs/PROGRAM_H_BOUND_METHOD_AMENDMENT_2026-07-12.md): tempo is
independent by CSSU, so the joint belief factorises b_A x b_B, each over the augmented
state (tempo, dwell_age) — a Markov-sufficient statistic for the semi-Markov generator.
Ages 0..5 (short dwell {4,5,6}) / 0..7 (long {6,7,8}). Observations (O0): the imperfect
signal and the realized weekly demand (revealed by the observable inventory trajectory in
Program G). We report BOTH a signal-only and a signal+demand belief to bracket the O0
reading. J*_obs is estimated by belief-MPC; QMDP (act knowing current tempo) is a diagnostic.
"""
from __future__ import annotations

import numpy as np

from .program_g import DEMAND_DAYS, MULT, STATIONARY, TEMPO

DWELL = {"short": (4, 5, 6), "long": (6, 7, 8)}


def _augmented_states(persistence: str):
    max_age = max(DWELL[persistence]) - 1          # last age before forced transition
    return [(t, a) for t in range(3) for a in range(max_age + 1)], max_age


def _dwell_hazard(persistence: str):
    """P(transition after this week | survived to age a) from the dwell distribution."""
    ds = DWELL[persistence]; n = len(ds)
    haz = {}
    for a in range(max(ds)):
        d = a + 1                                   # weeks completed if it ends now
        p_end = sum(1 for x in ds if x == d) / n
        p_surv = sum(1 for x in ds if x >= d) / n
        haz[a] = (p_end / p_surv) if p_surv > 0 else 1.0
    return haz


def transition_matrix(persistence: str) -> tuple[np.ndarray, list]:
    states, _ = _augmented_states(persistence)
    idx = {s: i for i, s in enumerate(states)}
    haz = _dwell_hazard(persistence)
    T = np.zeros((len(states), len(states)))
    for (t, a), i in idx.items():
        h = haz.get(a, 1.0)
        # survive: same tempo, age+1
        if (t, a + 1) in idx:
            T[i, idx[(t, a + 1)]] += (1 - h)
        else:
            h = 1.0                                 # must transition at max age
        # transition (age->0): routine(1)->low(0)/surge(2) 0.5/0.5; low/surge->routine
        if TEMPO[t] == "routine":
            T[i, idx[(0, 0)]] += h * 0.5
            T[i, idx[(2, 0)]] += h * 0.5
        else:
            T[i, idx[(1, 0)]] += h
    return T, states


def initial_belief(persistence: str) -> np.ndarray:
    states, _ = _augmented_states(persistence)
    b = np.array([STATIONARY[TEMPO[t]] if a == 0 else 0.0 for (t, a) in states])
    return b / b.sum()


def _demand_loglik(states, demand_obs: float, surge_mult: float) -> np.ndarray:
    """log P(weekly demand | tempo). demand = sum_6 round(base/2 * mult), base~U{2400,2600}."""
    means = {"low": 2500 / 2 * MULT["low"] * DEMAND_DAYS,
             "routine": 2500 / 2 * MULT["routine"] * DEMAND_DAYS,
             "surge": 2500 / 2 * surge_mult * DEMAND_DAYS}
    sd = (200 / 2) * DEMAND_DAYS / np.sqrt(DEMAND_DAYS)   # rough spread of the daily-base noise
    ll = []
    for (t, _a) in states:
        mu = means[TEMPO[t]]
        ll.append(-0.5 * ((demand_obs - mu) / max(sd, 1.0)) ** 2)
    return np.array(ll)


def _signal_loglik(states, T: np.ndarray, sig: int, q: float, lead: int) -> np.ndarray:
    """log P(signal | state): signal predicts surge `lead` weeks ahead (sens=spec=q)."""
    surge_idx = np.array([1.0 if TEMPO[t] == "surge" else 0.0 for (t, _a) in states])
    Tl = np.linalg.matrix_power(T, lead)
    p_future_surge = Tl @ surge_idx                       # P(surge at +lead | current state)
    p_sig1 = p_future_surge * q + (1 - p_future_surge) * (1 - q)
    p = p_sig1 if sig == 1 else (1 - p_sig1)
    return np.log(np.clip(p, 1e-9, 1.0))


class CSSUFilter:
    def __init__(self, persistence: str, surge_mult: float, q: float, lead: int):
        self.T, self.states = transition_matrix(persistence)
        self.b = initial_belief(persistence)
        self.persistence, self.surge_mult, self.q, self.lead = persistence, surge_mult, q, lead

    def predict(self):
        self.b = self.b @ self.T
        return self

    def update(self, sig: int | None = None, demand: float | None = None):
        ll = np.zeros(len(self.states))
        if sig is not None:
            ll = ll + _signal_loglik(self.states, self.T, sig, self.q, self.lead)
        if demand is not None:
            ll = ll + _demand_loglik(self.states, demand, self.surge_mult)
        w = np.exp(ll - ll.max()) * self.b
        self.b = w / max(w.sum(), 1e-12)
        return self

    def tempo_marginal(self) -> np.ndarray:
        m = np.zeros(3)
        for (t, _a), p in zip(self.states, self.b):
            m[t] += p
        return m
