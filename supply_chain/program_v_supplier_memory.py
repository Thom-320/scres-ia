"""Program V: source-conserving supplier-capacity commitments with retained belief.

The kernel is event driven at weekly procurement-arrival and demand epochs.  A
frozen tape owns regimes, warnings, supplier yields and demand, so every policy
receives common random numbers.  Ordered capacity that a supplier fails to
deliver never enters inventory and is recorded as rejected source capacity.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import math
from typing import Callable

import numpy as np


SUPPLIERS = ("S0", "S1", "S2")
HORIZON = 32
WEEKLY_ORDER = 4200.0
INITIAL_INVENTORY = 4200.0
ACTIONS = tuple(
    tuple(v / 4.0 for v in counts)
    for counts in itertools.product(range(3), repeat=3)
    if sum(counts) == 4 and max(counts) <= 2
)


def _u01(*parts: object) -> float:
    digest = hashlib.sha256(":".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


@dataclass(frozen=True)
class Tape:
    seed: int
    regimes: tuple[int, ...]
    warnings: tuple[int, ...]
    shuffled_warnings: tuple[int, ...]
    yields: tuple[tuple[float, float, float], ...]
    demand: tuple[float, ...]
    sha256: str


def make_tape(seed: int) -> Tape:
    regimes: list[int] = []
    warnings: list[int] = []
    yields: list[tuple[float, float, float]] = []
    demand: list[float] = []
    previous = int(_u01("program-v", seed, "initial") * 3) % 3
    for week in range(HORIZON):
        u = _u01("program-v", seed, week, "regime")
        if week == 0 or u >= 0.88:
            if week == 0:
                regime = previous
            else:
                alternatives = [i for i in range(3) if i != previous]
                regime = alternatives[int(_u01("program-v", seed, week, "switch") >= 0.5)]
        else:
            regime = previous
        regimes.append(regime)
        previous = regime
        if _u01("program-v", seed, week, "warning-correct") < 0.65:
            warning = regime
        else:
            alternatives = [i for i in range(3) if i != regime]
            warning = alternatives[int(_u01("program-v", seed, week, "warning-wrong") >= 0.5)]
        warnings.append(warning)
        factors = []
        for supplier in range(3):
            base = 0.10 if supplier == regime else 1.00
            noise = 0.04 * (2.0 * _u01("program-v", seed, week, supplier, "yield") - 1.0)
            factors.append(float(np.clip(base + noise, 0.02, 1.0)))
        yields.append(tuple(factors))
        seasonal = 3600.0 * (1.0 + 0.18 * math.sin(2.0 * math.pi * week / 13.0))
        noise = 240.0 * (2.0 * _u01("program-v", seed, week, "demand") - 1.0)
        demand.append(float(max(2400.0, seasonal + noise)))
    offset = 1 + int(_u01("program-v", seed, "shuffle") * (HORIZON - 1))
    shuffled = tuple(warnings[offset:] + warnings[:offset])
    payload = repr((regimes, warnings, yields, demand)).encode("utf-8")
    return Tape(seed, tuple(regimes), tuple(warnings), shuffled, tuple(yields), tuple(demand),
                hashlib.sha256(payload).hexdigest())


@dataclass
class Observation:
    week: int
    warning: int
    delayed_warning: int
    shuffled_warning: int
    true_regime: int
    inventory: float
    backlog: float
    last_realized_yields: np.ndarray
    posterior: np.ndarray


PolicyFn = Callable[[Observation], tuple[float, float, float]]


@dataclass(frozen=True)
class Policy:
    name: str
    family: str
    deployable: bool
    fn: PolicyFn


def avoid_action(supplier: int) -> tuple[float, float, float]:
    return tuple(0.0 if i == supplier else 0.5 for i in range(3))


def _nearest_action(weights: np.ndarray) -> tuple[float, float, float]:
    return min(ACTIONS, key=lambda action: (float(np.sum((np.asarray(action) - weights) ** 2)), action))


def policy_library() -> tuple[Policy, ...]:
    rows = [Policy(f"constant_{i}", "constant", True, lambda _o, a=a: a)
            for i, a in enumerate(ACTIONS)]
    rows.extend([
        Policy("warning_lookup", "warning", True, lambda o: avoid_action(o.warning)),
        Policy("placebo_delayed", "placebo", True, lambda o: avoid_action(o.delayed_warning)),
        Policy("placebo_shuffled", "placebo", True, lambda o: avoid_action(o.shuffled_warning)),
        Policy("last_yield", "last_yield", True,
               lambda o: avoid_action(int(np.argmin(o.last_realized_yields)))),
        Policy("bayes_retained", "belief", True,
               lambda o: _nearest_action((1.0 - o.posterior) / (1.0 - o.posterior).sum())),
        Policy("bayes_reset", "belief_reset", True,
               lambda o: avoid_action(o.warning)),
        Policy("privileged_true_state", "privileged", False,
               lambda o: avoid_action(o.true_regime)),
    ])
    if len(ACTIONS) != 6 or len(rows) != 13:
        raise AssertionError("Program V requires six constant actions and thirteen policies")
    return tuple(rows)


def update_posterior(prior: np.ndarray, warning: int,
                     observed_yields: np.ndarray, observed_mask: np.ndarray) -> np.ndarray:
    transition = np.full((3, 3), 0.06, dtype=float)
    np.fill_diagonal(transition, 0.88)
    # An arrival at week t reveals production yields from the order committed at
    # t-1.  Condition the t-1 belief on that delayed evidence before predicting
    # the t regime; using it after the transition would shift evidence one week.
    corrected = np.asarray(prior, dtype=float).copy()
    if bool(np.any(observed_mask)):
        yield_likelihood = np.ones(3, dtype=float)
        for state in range(3):
            expected = np.array([0.10 if supplier == state else 1.00 for supplier in range(3)])
            error = ((observed_yields - expected) / 0.10) ** 2
            yield_likelihood[state] = math.exp(-0.5 * float(np.sum(error[observed_mask])))
        corrected *= yield_likelihood
        if not np.isfinite(corrected).all() or corrected.sum() <= 0:
            corrected = np.asarray(prior, dtype=float).copy()
        corrected /= corrected.sum()
    predicted = corrected @ transition
    warning_likelihood = np.full(3, 0.175)
    warning_likelihood[warning] = 0.65
    posterior = predicted * warning_likelihood
    if not np.isfinite(posterior).all() or posterior.sum() <= 0:
        posterior = warning_likelihood
    return posterior / posterior.sum()


def simulate(tape: Tape, policy: Policy) -> dict[str, float | int | str | list[float]]:
    inventory = INITIAL_INVENTORY
    backlog = 0.0
    pipeline: tuple[float, np.ndarray, np.ndarray] | None = None
    posterior = np.full(3, 1.0 / 3.0)
    last_yields = np.ones(3)
    last_mask = np.zeros(3, dtype=bool)
    demanded = delivered = ordered = received = rejected = 0.0
    backlog_auc = 0.0
    recovery_durations: list[int] = []
    active_recovery: int | None = None
    previous_regime: int | None = None
    action_switches = 0
    previous_action = None
    action_trace: list[tuple[float, float, float]] = []
    posterior_trace: list[float] = []

    for week in range(HORIZON):
        if pipeline is not None:
            quantity, yield_vector, allocation = pipeline
            arrivals = quantity * allocation * yield_vector
            inventory += float(arrivals.sum())
            received += float(arrivals.sum())
            rejected += float(quantity - arrivals.sum())
            last_mask = allocation > 0
            # The policy observes only yields attached to actual deliveries.
            # A supplier with zero allocation is neutral rather than leaked.
            last_yields = np.where(last_mask, yield_vector, 1.0)
        posterior = update_posterior(posterior, tape.warnings[week], last_yields, last_mask)
        observation = Observation(
            week=week, warning=tape.warnings[week],
            delayed_warning=tape.warnings[week - 1] if week else tape.warnings[week],
            shuffled_warning=tape.shuffled_warnings[week], true_regime=tape.regimes[week],
            inventory=inventory, backlog=backlog, last_realized_yields=last_yields.copy(),
            posterior=posterior.copy())
        action = tuple(float(v) for v in policy.fn(observation))
        if action not in ACTIONS:
            raise ValueError(f"policy {policy.name} emitted infeasible action {action}")
        if previous_action is not None and action != previous_action:
            action_switches += 1
        previous_action = action
        action_trace.append(action)
        posterior_trace.append(float(posterior.max()))
        allocation = np.asarray(action)
        pipeline = (WEEKLY_ORDER, np.asarray(tape.yields[week]), allocation)
        ordered += WEEKLY_ORDER

        demand = float(tape.demand[week]) + backlog
        demanded += float(tape.demand[week])
        served = min(inventory, demand)
        inventory -= served
        backlog = demand - served
        delivered += served
        backlog_auc += backlog

        regime_changed = previous_regime is not None and tape.regimes[week] != previous_regime
        if regime_changed:
            active_recovery = 0
        if active_recovery is not None:
            active_recovery += 1
            if backlog <= 1e-9:
                recovery_durations.append(active_recovery)
                active_recovery = None
        previous_regime = tape.regimes[week]
    if active_recovery is not None:
        recovery_durations.append(HORIZON - (HORIZON - active_recovery))
    # Received material is either present or served. Rejected supplier capacity never entered.
    mass_residual = INITIAL_INVENTORY + received - inventory - delivered
    return {
        "service": float(delivered / demanded),
        "delivered": float(delivered), "demanded": float(demanded),
        "inventory_final": float(inventory), "backlog_final": float(backlog),
        "backlog_auc": float(backlog_auc),
        "mean_recovery_weeks": float(np.mean(recovery_durations) if recovery_durations else 0.0),
        "ordered": float(ordered), "received": float(received), "rejected": float(rejected),
        "mass_residual": float(mass_residual), "action_switches": int(action_switches),
        "unique_actions": int(len(set(action_trace))),
        "posterior_confidence_mean": float(np.mean(posterior_trace)),
        "tape_sha256": tape.sha256,
    }
