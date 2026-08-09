"""Forensic reconstruction of the lost G3a recurrent-risk allocation benchmark.

This is intentionally a small event ledger rather than a claim of recovering the
lost code.  Every modelling choice not stated in the surviving manuscript is
visible here and in the forensic contract.  The purpose is to restore an
executable, row-preserving development benchmark without pretending that its
numbers are the original numbers.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Callable, Iterable

import numpy as np


STATES = ("B", "N", "A")
WEEKS = 16
DAYS_PER_WEEK = 7
CAPACITY_PER_DAY = 2500.0
DEADLINE_HOURS = 48.0
DELIVERY_DELAY_HOURS = 54.0
MEASUREMENT_CLOSE_HOURS = WEEKS * DAYS_PER_WEEK * 24.0 + DELIVERY_DELAY_HOURS


def _u01(*parts: object) -> float:
    digest = hashlib.sha256(":".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


@dataclass(frozen=True)
class DemandOrder:
    order_id: int
    claimant: str
    created_at: float
    quantity: float
    deadline_at: float


@dataclass(frozen=True)
class Tape:
    seed: int
    process: str
    states: tuple[str, ...]
    warnings: tuple[str, ...]
    shuffled_warnings: tuple[str, ...]
    demand_a: tuple[float, ...]
    demand_b: tuple[float, ...]
    r24_count: int
    payload_sha256: str


def _draw_state(seed: int, week: int, previous: str, persistent: bool) -> str:
    u = _u01("g3a-forensic", seed, week, "state")
    if not persistent or week == 0:
        return STATES[min(int(u * 3), 2)]
    if u < 0.78:
        return previous
    alternatives = [state for state in STATES if state != previous]
    return alternatives[0] if u < 0.89 else alternatives[1]


def _warning(seed: int, week: int, state: str) -> str:
    if _u01("g3a-forensic", seed, week, "warning-correct") < 0.72:
        return state
    alternatives = [candidate for candidate in STATES if candidate != state]
    return alternatives[int(_u01("g3a-forensic", seed, week, "warning-wrong") >= 0.5)]


def make_tape(seed: int, process: str) -> Tape:
    if process not in {"iid_uniform", "persistent_uniform", "persistent_seasonal"}:
        raise ValueError(f"unknown G3a process {process!r}")
    persistent = process != "iid_uniform"
    seasonal = process == "persistent_seasonal"
    states: list[str] = []
    warnings: list[str] = []
    demand_a: list[float] = []
    demand_b: list[float] = []
    r24_count = 0
    previous = "N"
    for week in range(WEEKS):
        state = _draw_state(seed, week, previous, persistent)
        states.append(state)
        warnings.append(_warning(seed, week, state))
        previous = state
        for day in range(DAYS_PER_WEEK):
            index = week * DAYS_PER_WEEK + day
            if day == 6:  # source system serves six demand days per week
                total = 0.0
            else:
                total = 2400.0 + 200.0 * _u01("g3a-forensic", seed, index, "demand")
                if seasonal:
                    phase = 2.0 * math.pi * index / 56.0
                    total *= 1.0 + 0.18 * math.sin(phase) + 0.08 * math.cos(phase / 2.0)
            share_a = {"A": 0.75, "N": 0.50, "B": 0.23}[state]
            qa, qb = total * share_a, total * (1.0 - share_a)
            if day == 2 and state != "N" and _u01("g3a-forensic", seed, week, "r24") < 0.55:
                if state == "A":
                    qa += 2500.0
                else:
                    qb += 2500.0
                r24_count += 1
            demand_a.append(float(qa))
            demand_b.append(float(qb))
    offset = 1 + int(_u01("g3a-forensic", seed, "shuffle") * (WEEKS - 1))
    shuffled = tuple(warnings[offset:] + warnings[:offset])
    payload = repr((states, warnings, demand_a, demand_b, r24_count)).encode("utf-8")
    return Tape(
        seed=int(seed), process=process, states=tuple(states), warnings=tuple(warnings),
        shuffled_warnings=shuffled, demand_a=tuple(demand_a), demand_b=tuple(demand_b),
        r24_count=r24_count, payload_sha256=hashlib.sha256(payload).hexdigest())


@dataclass
class PolicyContext:
    week: int
    warning: str
    delayed_warning: str
    shuffled_warning: str
    true_state: str
    backlog_a: float
    backlog_b: float
    last_demand_a: float
    last_demand_b: float
    belief: np.ndarray


PolicyFn = Callable[[PolicyContext], float]


@dataclass(frozen=True)
class Policy:
    name: str
    family: str
    deployable: bool
    fn: PolicyFn


def _state_action(state: str, amplitude: float) -> float:
    return float(np.clip(0.5 + {"A": amplitude, "N": 0.0, "B": -amplitude}[state], 0.1, 0.9))


def policies() -> tuple[Policy, ...]:
    rows: list[Policy] = []
    for level in np.linspace(0.1, 0.9, 9):
        rows.append(Policy(f"constant_{level:.1f}", "constant", True,
                           lambda _c, level=float(level): level))
    for amplitude in (0.1, 0.2, 0.3, 0.4):
        rows.append(Policy(f"warning_lookup_{amplitude:.1f}", "warning_lookup", True,
                           lambda c, a=amplitude: _state_action(c.warning, a)))
    rows.extend([
        Policy("placebo_delayed_warning", "placebo", True,
               lambda c: _state_action(c.delayed_warning, 0.3)),
        Policy("placebo_shuffled_warning", "placebo", True,
               lambda c: _state_action(c.shuffled_warning, 0.3)),
        Policy("placebo_wrong_claimant", "placebo", True,
               lambda c: 1.0 - _state_action(c.warning, 0.3)),
        Policy("belief_stateful", "belief", True,
               lambda c: float(np.clip(0.5 + 0.35 * (c.belief[2] - c.belief[0]), 0.1, 0.9))),
        Policy("belief_reset", "belief_reset", True,
               lambda c: _state_action(c.warning, 0.22)),
    ])
    for gain in (0.25, 0.5, 0.75, 1.0, 1.5):
        rows.append(Policy(f"lagged_demand_{gain:g}", "lagged_demand", True,
                           lambda c, g=gain: float(np.clip(
                               0.5 + g * (c.last_demand_a - c.last_demand_b)
                               / max(c.last_demand_a + c.last_demand_b, 1.0), 0.1, 0.9))))
    for gain in (0.25, 0.5, 0.75, 1.0, 1.5):
        rows.append(Policy(f"backlog_pressure_{gain:g}", "backlog", True,
                           lambda c, g=gain: float(np.clip(
                               0.5 + g * (c.backlog_a - c.backlog_b)
                               / max(c.backlog_a + c.backlog_b, CAPACITY_PER_DAY), 0.1, 0.9))))
    for weight in (0.2, 0.35, 0.5, 0.65, 0.8):
        rows.append(Policy(f"belief_backlog_{weight:g}", "belief_backlog", True,
                           lambda c, w=weight: float(np.clip(
                               w * (0.5 + 0.35 * (c.belief[2] - c.belief[0]))
                               + (1.0 - w) * (0.5 + 0.75 * (c.backlog_a - c.backlog_b)
                                  / max(c.backlog_a + c.backlog_b, CAPACITY_PER_DAY)),
                               0.1, 0.9))))
    rows.append(Policy("privileged_true_state", "privileged", False,
                       lambda c: _state_action(c.true_state, 0.4)))
    if len(rows) != 34 or len({row.name for row in rows}) != 34:
        raise AssertionError("G3a forensic policy library must contain 34 unique policies")
    return tuple(rows)


def _belief_update(prior: np.ndarray, warning: str) -> np.ndarray:
    transition = np.full((3, 3), 0.11, dtype=float)
    np.fill_diagonal(transition, 0.78)
    predicted = np.asarray(prior, dtype=float) @ transition
    likelihood = np.full(3, 0.14, dtype=float)
    likelihood[STATES.index(warning)] = 0.72
    posterior = predicted * likelihood
    return posterior / posterior.sum()


def _allocate_fifo(queues: dict[str, list[list[float]]], capacity: float) -> tuple[float, float]:
    delivered = {"A": 0.0, "B": 0.0}
    while capacity > 1e-9:
        heads = [(rows[0][0], claimant) for claimant, rows in queues.items() if rows]
        if not heads:
            break
        _created, claimant = min(heads, key=lambda item: (item[0], item[1]))
        row = queues[claimant][0]
        qty = min(capacity, row[2])
        row[2] -= qty
        capacity -= qty
        delivered[claimant] += qty
        if row[2] <= 1e-9:
            queues[claimant].pop(0)
    return delivered["A"], delivered["B"]


def _serve_queue(queue: list[list[float]], capacity: float) -> float:
    delivered = 0.0
    while capacity > 1e-9 and queue:
        qty = min(capacity, queue[0][2])
        queue[0][2] -= qty
        capacity -= qty
        delivered += qty
        if queue[0][2] <= 1e-9:
            queue.pop(0)
    return delivered


def simulate(tape: Tape, capacity_contract: str, policy: Policy) -> dict[str, float | int | str]:
    if capacity_contract not in {"hard_quota", "spare_reallocation", "global_fifo"}:
        raise ValueError(f"unknown capacity contract {capacity_contract!r}")
    queues: dict[str, list[list[float]]] = {"A": [], "B": []}
    total_demand = {"A": 0.0, "B": 0.0}
    total_dispatched = {"A": 0.0, "B": 0.0}
    late_exposure = {"A": 0.0, "B": 0.0}
    max_exposure = {"A": 0.0, "B": 0.0}
    belief = np.full(3, 1.0 / 3.0)
    last_week_demand = {"A": 0.0, "B": 0.0}
    forfeited = 0.0
    switches = 0
    previous_action: float | None = None
    action_sum = 0.0
    actions: list[float] = []
    order_id = 0

    for day in range(WEEKS * DAYS_PER_WEEK):
        week = day // DAYS_PER_WEEK
        now = float(day * 24)
        if day % DAYS_PER_WEEK == 0:
            belief = _belief_update(belief, tape.warnings[week])
            context = PolicyContext(
                week=week, warning=tape.warnings[week],
                delayed_warning=tape.warnings[week - 1] if week else "N",
                shuffled_warning=tape.shuffled_warnings[week], true_state=tape.states[week],
                backlog_a=sum(row[2] for row in queues["A"]),
                backlog_b=sum(row[2] for row in queues["B"]),
                last_demand_a=last_week_demand["A"], last_demand_b=last_week_demand["B"],
                belief=belief.copy())
            action = float(np.clip(policy.fn(context), 0.1, 0.9))
            actions.append(action)
            if previous_action is not None and abs(action - previous_action) > 1e-9:
                switches += 1
            previous_action = action
            last_week_demand = {"A": 0.0, "B": 0.0}
        action_sum += action
        for claimant, quantity in (("A", tape.demand_a[day]), ("B", tape.demand_b[day])):
            quantity = float(quantity)
            if quantity <= 0:
                continue
            queues[claimant].append([now, float(order_id), quantity])
            total_demand[claimant] += quantity
            last_week_demand[claimant] += quantity
            order_id += 1

        before = {claimant: sum(row[2] for row in rows) for claimant, rows in queues.items()}
        if capacity_contract == "global_fifo":
            delivered_a, delivered_b = _allocate_fifo(queues, CAPACITY_PER_DAY)
            used = delivered_a + delivered_b
        else:
            cap_a, cap_b = CAPACITY_PER_DAY * action, CAPACITY_PER_DAY * (1.0 - action)
            delivered_a = _serve_queue(queues["A"], cap_a)
            delivered_b = _serve_queue(queues["B"], cap_b)
            used = delivered_a + delivered_b
            if capacity_contract == "spare_reallocation":
                spare = CAPACITY_PER_DAY - used
                if spare > 1e-9:
                    # The action remains a priority; residual capacity is work-conserving.
                    pressure_a = sum(row[2] for row in queues["A"])
                    pressure_b = sum(row[2] for row in queues["B"])
                    first, second = (("A", "B") if pressure_a >= pressure_b else ("B", "A"))
                    extra_first = _serve_queue(queues[first], spare)
                    spare -= extra_first
                    extra_second = _serve_queue(queues[second], spare)
                    if first == "A":
                        delivered_a += extra_first; delivered_b += extra_second
                    else:
                        delivered_b += extra_first; delivered_a += extra_second
                    used = delivered_a + delivered_b
        forfeited += max(0.0, min(CAPACITY_PER_DAY, before["A"] + before["B"]) - used)
        total_dispatched["A"] += delivered_a
        total_dispatched["B"] += delivered_b

        # Exposure is integrated from the contractual deadline. Dispatch today arrives 54 h later.
        for claimant, delivered in (("A", delivered_a), ("B", delivered_b)):
            late_exposure[claimant] += delivered * max(0.0, now + DELIVERY_DELAY_HOURS
                                                       - (now + DEADLINE_HOURS))

    # Unresolved quantities remain exposed until measurement close. The denominator is the
    # maximum possible post-deadline exposure for every demanded unit.
    for claimant, demand_series in (("A", tape.demand_a), ("B", tape.demand_b)):
        max_exposure[claimant] = sum(
            float(q) * max(0.0, MEASUREMENT_CLOSE_HOURS - (day * 24.0 + DEADLINE_HOURS))
            for day, q in enumerate(demand_series))
        for created_at, _oid, quantity in queues[claimant]:
            late_exposure[claimant] += quantity * max(
                0.0, MEASUREMENT_CLOSE_HOURS - (created_at + DEADLINE_HOURS))

    service = {
        claimant: 1.0 - late_exposure[claimant] / max(max_exposure[claimant], 1.0)
        for claimant in ("A", "B")
    }
    unresolved = {claimant: sum(row[2] for row in queues[claimant]) for claimant in ("A", "B")}
    demanded = total_demand["A"] + total_demand["B"]
    dispatched = total_dispatched["A"] + total_dispatched["B"]
    residual = demanded - dispatched - unresolved["A"] - unresolved["B"]
    return {
        "primary_service": float(min(service.values())),
        "service_a": float(service["A"]), "service_b": float(service["B"]),
        "aggregate_fill": float(dispatched / demanded if demanded else 1.0),
        "demand_a": float(total_demand["A"]), "demand_b": float(total_demand["B"]),
        "dispatched_a": float(total_dispatched["A"]), "dispatched_b": float(total_dispatched["B"]),
        "unresolved_a": float(unresolved["A"]), "unresolved_b": float(unresolved["B"]),
        "forfeited_capacity": float(forfeited), "switches": int(switches),
        "mean_allocation_a": float(action_sum / (WEEKS * DAYS_PER_WEEK)),
        "unique_actions": int(len(set(round(value, 9) for value in actions))),
        "flow_residual": float(residual), "r24_count": int(tape.r24_count),
        "tape_sha256": tape.payload_sha256,
    }


def all_cells() -> Iterable[tuple[str, str, str]]:
    for process in ("iid_uniform", "persistent_uniform", "persistent_seasonal"):
        for contract in ("hard_quota", "spare_reallocation", "global_fifo"):
            yield f"{process}__{contract}", process, contract
