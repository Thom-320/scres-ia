"""A contention testbed whose headroom is known by construction, not estimated.

WHY IT EXISTS. Every headroom number in this project is an estimate compared against another
estimate. That is how a clairvoyant ceiling measured on twelve reused tapes survived three weeks
before a virgin block found it smaller than its own null. A method for certifying learning
eligibility cannot be validated that way: it needs at least one instance where the answer is known
before the audit runs.

THE MECHANISM IS THE ONE WE MEASURED, not a convenient invention. The only physics in which this
project ever found material headroom is contention over a scarce, non-fungible shared resource
(H_PI 0.1515, with the fungible null at exactly 0). This reproduces it in miniature with a dial
that turns it off.

THE NULL IS ALGEBRA. With `alpha = 1` every unit of unused capacity flows to the other class at
full efficiency, so delivered service is `min(d_A + d_B, C)` for EVERY split. The action cannot
change the outcome -- not approximately, bit-for-bit -- so `H_PI = 0` by construction and any audit
reporting headroom there is producing a false positive.

THE DWELL IS THE FINE POINT. With `min_dwell > 1` the regime is semi-Markov: the true state
includes time-since-switch. A first-order two-state Bayes filter -- the model-based controller a
practitioner actually writes -- is MISSPECIFIED. That is the gap where a learner can pay, and the
result has to say so rather than quietly bank it as a premium over optimality.

Contract: docs/PREREGISTRO_VALIDACION_POSITIVA_AUDIT_2026-08-08.md
Synthetic. Carries no claim about the MFSC or about Garrido-Rios (2017).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Regime labels. 0 = class A heavy, 1 = class B heavy.
A_HEAVY, B_HEAVY = 0, 1


@dataclass(frozen=True)
class BenchSpec:
    """One cell of the design. Frozen so a cell cannot be edited between arms."""

    alpha: float                 # capacity fungibility in [0, 1]; 1.0 makes the action inert
    rho: float                   # P(stay) once the minimum dwell has elapsed
    min_dwell: int               # periods a regime must last; > 1 makes the truth semi-Markov
    signal_accuracy: float       # P(signal == regime)
    periods: int = 52
    capacity: float = 100.0
    lam_high: float = 70.0
    lam_low: float = 30.0
    label: str = ""

    def as_dict(self) -> dict:
        return {"alpha": self.alpha, "rho": self.rho, "min_dwell": self.min_dwell,
                "signal_accuracy": self.signal_accuracy, "periods": self.periods,
                "capacity": self.capacity, "lam_high": self.lam_high, "lam_low": self.lam_low,
                "label": self.label}


@dataclass
class Tape:
    """One episode's exogenous realisation, drawn ONCE and shared by every arm.

    Common random numbers are not a nicety here. The whole design compares policies by their
    difference on the same world; drawing fresh demand per arm would put the comparison's noise
    floor above the effect it is meant to resolve.
    """

    regimes: np.ndarray          # (T,) int
    demand_a: np.ndarray         # (T,) float
    demand_b: np.ndarray         # (T,) float
    signals: np.ndarray          # (T,) int -- observable BEFORE the action
    seed: int = 0
    spec: BenchSpec | None = field(default=None, repr=False)


def draw_tape(spec: BenchSpec, seed: int) -> Tape:
    rng = np.random.default_rng(seed)
    regimes = np.empty(spec.periods, dtype=int)
    z = int(rng.integers(0, 2))
    dwell = 0
    for t in range(spec.periods):
        if dwell >= spec.min_dwell and rng.random() > spec.rho:
            z = 1 - z
            dwell = 0
        regimes[t] = z
        dwell += 1
    hi = np.where(regimes == A_HEAVY, spec.lam_high, spec.lam_low)
    lo = np.where(regimes == A_HEAVY, spec.lam_low, spec.lam_high)
    demand_a = rng.poisson(hi).astype(float)
    demand_b = rng.poisson(lo).astype(float)
    flip = rng.random(spec.periods) > spec.signal_accuracy
    signals = np.where(flip, 1 - regimes, regimes)
    return Tape(regimes=regimes, demand_a=demand_a, demand_b=demand_b, signals=signals,
                seed=seed, spec=spec)


def serve(spec: BenchSpec, actions: np.ndarray, tape: Tape) -> float:
    """Fill rate over the episode. `actions[t]` is the share of capacity reserved for class A.

    Spill is symmetric and simultaneous: each class's unused reservation serves the other at
    efficiency `alpha`. At `alpha = 1` this collapses to `min(d_A + d_B, C)` for every split,
    which is what makes the null cell a theorem rather than a finding.
    """
    a = np.clip(np.asarray(actions, dtype=float), 0.0, 1.0)
    cap_a, cap_b = a * spec.capacity, (1.0 - a) * spec.capacity
    d_a, d_b = tape.demand_a, tape.demand_b

    direct_a = np.minimum(d_a, cap_a)
    direct_b = np.minimum(d_b, cap_b)
    spare_a = np.maximum(cap_a - d_a, 0.0)
    spare_b = np.maximum(cap_b - d_b, 0.0)
    spill_a = np.minimum(d_a - direct_a, spec.alpha * spare_b)
    spill_b = np.minimum(d_b - direct_b, spec.alpha * spare_a)

    served = direct_a + direct_b + spill_a + spill_b
    demanded = d_a + d_b
    return float(served.sum() / max(demanded.sum(), 1e-9))


# --------------------------------------------------------------------------------------------
# Policies. Every one of these returns a full action vector for a tape; none may read `regimes`
# except the two that are labelled for it.
# --------------------------------------------------------------------------------------------

def fixed_policy(level: float, tape: Tape) -> np.ndarray:
    return np.full(tape.demand_a.shape, float(level))


def signal_threshold_policy(action_when_a: float, action_when_b: float, tape: Tape) -> np.ndarray:
    return np.where(tape.signals == A_HEAVY, action_when_a, action_when_b).astype(float)


def _myopic_split(spec: BenchSpec, p_a_heavy: float) -> float:
    """Split that a one-period-ahead planner would choose given belief `p_a_heavy`."""
    exp_a = p_a_heavy * spec.lam_high + (1.0 - p_a_heavy) * spec.lam_low
    exp_b = p_a_heavy * spec.lam_low + (1.0 - p_a_heavy) * spec.lam_high
    return float(np.clip(exp_a / max(exp_a + exp_b, 1e-9), 0.0, 1.0))


def belief_mpc_policy(spec: BenchSpec, tape: Tape) -> np.ndarray:
    """First-order two-state Bayes filter + myopic split. MISSPECIFIED when min_dwell > 1.

    This is the model-based controller a practitioner writes from the stated dynamics: it knows the
    switch probability and the signal accuracy, and it does not represent time-since-switch. The
    misspecification is the honest one -- it is what you get from believing your own model.
    """
    q = spec.signal_accuracy
    belief = 0.5
    out = np.empty(spec.periods)
    for t in range(spec.periods):
        like_a = q if tape.signals[t] == A_HEAVY else (1.0 - q)
        like_b = (1.0 - q) if tape.signals[t] == A_HEAVY else q
        post = (belief * like_a) / max(belief * like_a + (1.0 - belief) * like_b, 1e-12)
        out[t] = _myopic_split(spec, post)
        belief = post * spec.rho + (1.0 - post) * (1.0 - spec.rho)
    return out


def oracle_model_mpc_policy(spec: BenchSpec, tape: Tape) -> np.ndarray:
    """DISCLOSURE ARM. A filter over the TRUE semi-Markov state (regime, time-since-switch).

    It is not a comparator the learner must beat. It exists so that an advantage over the
    misspecified filter cannot be reported as an advantage over decision-theoretic optimality.
    """
    q = spec.signal_accuracy
    states = [(z, d) for z in (A_HEAVY, B_HEAVY) for d in range(spec.min_dwell + 1)]
    index = {s: i for i, s in enumerate(states)}
    belief = np.full(len(states), 1.0 / len(states))
    out = np.empty(spec.periods)
    for t in range(spec.periods):
        like = np.array([q if (s[0] == tape.signals[t]) else (1.0 - q) for s in states])
        post = belief * like
        post /= max(post.sum(), 1e-12)
        p_a = float(sum(post[index[s]] for s in states if s[0] == A_HEAVY))
        out[t] = _myopic_split(spec, p_a)
        nxt = np.zeros_like(post)
        for s, i in index.items():
            z, d = s
            if d < spec.min_dwell:                      # dwell not yet served: cannot switch
                nxt[index[(z, d + 1)]] += post[i]
            else:
                nxt[index[(z, spec.min_dwell)]] += post[i] * spec.rho
                nxt[index[(1 - z, 0)]] += post[i] * (1.0 - spec.rho)
        belief = nxt / max(nxt.sum(), 1e-12)
    return out


def clairvoyant_actions(spec: BenchSpec, tape: Tape, grid: np.ndarray) -> np.ndarray:
    """Per-period best split with the regime KNOWN. Upper bound on contingent decision value.

    Chosen per period on this tape's own realisation, which is exactly the selection that makes a
    clairvoyant gap biased upward under noise -- the bias the interaction null exists to price.
    """
    out = np.empty(spec.periods)
    for t in range(spec.periods):
        best, best_v = grid[0], -np.inf
        d_a, d_b = tape.demand_a[t], tape.demand_b[t]
        for level in grid:
            cap_a, cap_b = level * spec.capacity, (1.0 - level) * spec.capacity
            direct_a, direct_b = min(d_a, cap_a), min(d_b, cap_b)
            spill_a = min(d_a - direct_a, spec.alpha * max(cap_b - d_b, 0.0))
            spill_b = min(d_b - direct_b, spec.alpha * max(cap_a - d_a, 0.0))
            v = direct_a + direct_b + spill_a + spill_b
            if v > best_v:
                best, best_v = level, v
        out[t] = best
    return out


def history_features(spec: BenchSpec, tape: Tape, window: int) -> np.ndarray:
    """(T, F) features available BEFORE each action: current signal plus a lagged window.

    Lagged demands are what a first-order filter throws away and what time-since-switch is visible
    in, so this is the information the learner could use and the belief filter structurally cannot.
    """
    sig = tape.signals.astype(float) * 2.0 - 1.0
    da = tape.demand_a / spec.lam_high
    db = tape.demand_b / spec.lam_high
    cols = [sig]
    for lag in range(1, window + 1):
        cols.append(np.concatenate([np.zeros(lag), sig[:-lag]]))
        cols.append(np.concatenate([np.zeros(lag), (da - db)[:-lag]]))
    cols.append(np.ones(spec.periods))
    return np.stack(cols, axis=1)
