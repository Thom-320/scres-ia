"""Finite storage capacity per node, and the budget that makes it a decision.

This removes the second assumption Garrido declares in his own words (WRAP 2017, Section 6.5.5):
*"storage capacities of WDC, SBs and CSSBs are assumed to be unlimited along the simulation
horizon of the model."* It is listed among eight simplifications made "to avoid including
unnecessary details, to reduce the execution time" -- not as physics. The shipped code says the
same thing: every `simpy.Container` is built with `capacity=INF`.

And it is what he asked for directly. On 2 July: expand decision variables from the CDC downward,
adding buffer variables PER NODE, and prefer continuous over discrete. On 28 July: add nodes and
decision variables, including buffers in unconsidered nodes, rather than lengthening the episode.

WHY A BUDGET, AND NOT JUST FINITE NUMBERS.  Capping each node independently only makes the chain
worse; nothing has to be traded off, so there is no decision and no headroom. A fixed TOTAL
capacity split across nodes turns storage into a scarce, non-fungible resource -- which is the one
mechanism this project has ever measured material headroom from: Program O reached H_PI = 0.1515
under a non-fungible share, and EXACTLY 0 once the same resource was made fungible.

BLOCKING, NOT SPILLING.  When a node is full the surplus stays where it is; it is never destroyed.
Spilling would silently delete rations and flatter every downstream metric, and this project has
already measured one metric that rewards abandoning demand. Mass conservation is asserted by test,
not assumed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

INF = float("inf")

#: Storage nodes, downstream-of-CDC first, in Garrido's own vocabulary.
CAPACITY_NODES = ("wdc", "al", "sb", "cssu_a", "cssu_b")

#: The shipped model: unlimited everywhere. Using this is the null arm.
UNLIMITED: dict[str, float] = {node: INF for node in CAPACITY_NODES}


def validate_capacities(capacities: Mapping[str, float]) -> dict[str, float]:
    """Every node named, non-negative, and no unknown nodes invented."""
    unknown = set(capacities) - set(CAPACITY_NODES)
    if unknown:
        raise ValueError(f"unknown capacity nodes: {sorted(unknown)}")
    out = dict(UNLIMITED)
    for node, value in capacities.items():
        v = float(value)
        if v < 0.0 or v != v:                       # NaN is not a capacity
            raise ValueError(f"{node}: capacity must be non-negative and finite-or-inf, got {value!r}")
        out[node] = v
    return out


def budget_split(total: float, shares: Mapping[str, float]) -> dict[str, float]:
    """Split a finite TOTAL storage budget across nodes by continuous shares.

    Shares are normalised, so the caller states relative priority rather than absolute rations and
    the budget is conserved exactly. Continuous by design: Garrido asked for continuous decision
    variables on 2 July, and a three-point grid is a resolution limit rather than physics -- the
    same mistake the CSSU allocation lever carried until it was widened.
    """
    if not (total > 0.0) or total == INF:
        raise ValueError("budget total must be finite and positive")
    unknown = set(shares) - set(CAPACITY_NODES)
    if unknown:
        raise ValueError(f"unknown capacity nodes: {sorted(unknown)}")
    values = {n: float(shares.get(n, 0.0)) for n in CAPACITY_NODES}
    if any(v < 0.0 for v in values.values()):
        raise ValueError("shares must be non-negative")
    mass = sum(values.values())
    if mass <= 0.0:
        raise ValueError("shares must have positive total mass")
    return {n: total * v / mass for n, v in values.items()}


@dataclass
class NodeCapacityLedger:
    """Admission control with the accounting that makes conservation checkable."""
    capacities: dict[str, float] = field(default_factory=lambda: dict(UNLIMITED))
    blocked_qty: dict[str, float] = field(default_factory=lambda: {n: 0.0 for n in CAPACITY_NODES})
    blocked_events: dict[str, int] = field(default_factory=lambda: {n: 0 for n in CAPACITY_NODES})
    admitted_qty: dict[str, float] = field(default_factory=lambda: {n: 0.0 for n in CAPACITY_NODES})

    def __post_init__(self) -> None:
        self.capacities = validate_capacities(self.capacities)

    @property
    def is_inert(self) -> bool:
        """True when every node is unlimited, i.e. this is the shipped model."""
        return all(c == INF for c in self.capacities.values())

    def headroom(self, node: str, level: float) -> float:
        if node not in self.capacities:
            raise ValueError(f"unknown node {node!r}")
        cap = self.capacities[node]
        return INF if cap == INF else max(0.0, cap - float(level))

    def admit(self, node: str, level: float, arriving: float) -> dict[str, float]:
        """How much of `arriving` fits. The rest is BLOCKED, never destroyed.

        Returns both halves so a caller cannot accidentally drop the remainder: the conservation
        test asserts `admitted + blocked == arriving` on every call.
        """
        if arriving < 0.0:
            raise ValueError("arriving quantity must be non-negative")
        room = self.headroom(node, level)
        admitted = float(arriving) if room == INF else min(float(arriving), room)
        blocked = float(arriving) - admitted
        self.admitted_qty[node] += admitted
        if blocked > 0.0:
            self.blocked_qty[node] += blocked
            self.blocked_events[node] += 1
        return {"admitted": admitted, "blocked": blocked}

    def binding_fraction(self) -> float:
        """Share of admissions that hit a cap. Zero means the constraint never bound, so any
        result attributed to it is measuring nothing -- the dead-actuator failure that cost G3-obs
        a full run, and the reason this is telemetry rather than a comment."""
        events = sum(self.blocked_events.values())
        total = sum(1 for n in CAPACITY_NODES if self.admitted_qty[n] > 0.0 or self.blocked_qty[n] > 0.0)
        return 0.0 if total == 0 else events / float(total)

    def as_evidence(self) -> dict[str, Any]:
        return {"capacities": {n: (None if c == INF else c) for n, c in self.capacities.items()},
                "is_inert": self.is_inert,
                "blocked_qty": dict(self.blocked_qty),
                "blocked_events": dict(self.blocked_events),
                "admitted_qty": dict(self.admitted_qty),
                "total_blocked": sum(self.blocked_qty.values()),
                "total_admitted": sum(self.admitted_qty.values())}


def capacity_is_live(ledger: NodeCapacityLedger) -> bool:
    """Whether the constraint ever bound. A capacity that never fills is decoration."""
    return sum(ledger.blocked_events.values()) > 0
