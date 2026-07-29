"""Physical primitives for the opt-in two-CSSU DRA-1 extension.

This module deliberately contains no learning code.  It defines the conserved
daily capacity split that static policies, trees, heuristics, and eventually
PPO must all share.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Mapping


CSSU_IDS = ("A", "B")
ALLOCATION_LEVELS = (0.25, 0.50, 0.75)
SERVICE_RULES = ("SPT_FULL", "FIFO_PARTIAL", "R24_AGE_PARTIAL")


@dataclass(frozen=True)
class AllocationResult:
    available: float
    dispatched_a: float
    dispatched_b: float
    unused: float

    @property
    def total_dispatched(self) -> float:
        return self.dispatched_a + self.dispatched_b


def stable_cssu_destination(*, simulation_seed: int, order_id: int) -> str:
    """Assign A/B without consuming or perturbing any simulator RNG stream."""
    digest = sha256(f"dra1-cssu-v1:{simulation_seed}:{order_id}".encode()).digest()
    return CSSU_IDS[digest[0] & 1]


def allocate_shared_capacity(
    *,
    stock: float,
    daily_capacity: float,
    allocation_a: float,
    requested: Mapping[str, float],
    reallocate_unused: bool = True,
) -> AllocationResult:
    """Allocate one fixed capacity pool; the action cannot enlarge the pool.

    Shares are binding while both destinations are capacity constrained.  If a
    destination cannot use its share, spare capacity may be assigned to unmet
    demand at the other destination.  This prevents intentional idling from
    masquerading as an allocation benefit.
    """
    if allocation_a not in ALLOCATION_LEVELS:
        raise ValueError(f"allocation_a must be one of {ALLOCATION_LEVELS}")
    if stock < 0 or daily_capacity < 0:
        raise ValueError("stock and daily_capacity must be non-negative")
    demand = {cssu: float(requested.get(cssu, 0.0)) for cssu in CSSU_IDS}
    if any(value < 0 for value in demand.values()):
        raise ValueError("requested quantities must be non-negative")

    available = min(float(stock), float(daily_capacity))
    cap_a = available * allocation_a
    cap_b = available - cap_a
    sent = {"A": min(demand["A"], cap_a), "B": min(demand["B"], cap_b)}

    if reallocate_unused:
        spare = available - sent["A"] - sent["B"]
        # Deterministic largest-unmet-first use of genuinely spare capacity.
        for cssu in sorted(CSSU_IDS, key=lambda x: (-(demand[x] - sent[x]), x)):
            add = min(spare, demand[cssu] - sent[cssu])
            sent[cssu] += add
            spare -= add

    total = sent["A"] + sent["B"]
    if total > available + 1e-9:
        raise AssertionError("allocation created dispatch capacity")
    return AllocationResult(available, sent["A"], sent["B"], available - total)
