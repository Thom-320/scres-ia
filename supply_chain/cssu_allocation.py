"""Physical primitives for the opt-in two-CSSU DRA-1 extension.

This module deliberately contains no learning code.  It defines the conserved
daily capacity split that static policies, trees, heuristics, and eventually
PPO must all share.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import isfinite
from typing import Mapping


CSSU_IDS = ("A", "B")
# The three thesis-aligned static levels. They remain the DEFAULT grid, but they are no longer
# the only admissible values: `validate_allocation_a` accepts the whole interval, because a
# three-point grid cannot express an allocation that tracks which unit is currently down.
ALLOCATION_LEVELS = (0.25, 0.50, 0.75)
SERVICE_RULES = ("SPT_FULL", "FIFO_PARTIAL", "R24_AGE_PARTIAL")


def validate_allocation_a(value: float) -> float:
    """Accept any share in [0, 1]; the old membership test was a resolution limit, not physics."""
    share = float(value)
    if not 0.0 <= share <= 1.0:
        raise ValueError(f"allocation_a must lie in [0, 1]; got {share!r}")
    return share


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


def event_keyed_uniform_u64(
    *,
    simulation_seed: int,
    event_id: int,
    namespace: str = "g3a-cssu-v2",
) -> float:
    """Return a deterministic ``[0, 1)`` uniform for one exogenous event.

    This is deliberately separate from the simulator RNG.  A weighted G3a arm
    may transform this same event key through a CDF without changing any later
    simulator draws.  The namespace is part of the key so that a future
    extension cannot silently reuse the historical DRA-1 destination stream.
    """
    digest = sha256(f"{namespace}:{simulation_seed}:{event_id}".encode()).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) / float(1 << 64)


def _normalise_cssu_weights(weights: Mapping[str, float]) -> tuple[float, float]:
    """Validate and normalize a two-claimant destination distribution."""
    if set(weights) != set(CSSU_IDS):
        raise ValueError(f"weights must contain exactly {CSSU_IDS}; got {sorted(weights)!r}")
    values = {cssu: float(weights[cssu]) for cssu in CSSU_IDS}
    if any(not isfinite(value) or value < 0.0 for value in values.values()):
        raise ValueError("weights must be finite and non-negative")
    total = values["A"] + values["B"]
    if total <= 0.0:
        raise ValueError("weights must have positive total mass")
    return values["A"] / total, values["B"] / total


def stable_cssu_destination_weighted(
    *,
    simulation_seed: int,
    event_id: int,
    weights: Mapping[str, float] | None,
    namespace: str = "g3a-cssu-v2",
) -> str:
    """Map one event to A/B while preserving the exact legacy ``None`` lane.

    ``weights=None`` delegates to :func:`stable_cssu_destination` and therefore
    preserves the historical ``dra1-cssu-v1`` mapping byte for byte.  Weighted
    arms use a new 64-bit event-keyed uniform and never consume simulator RNG.
    This helper is intentionally not wired into the DES yet; the G3a contract
    must freeze the exogenous risk tape before scientific execution.
    """
    if weights is None:
        return stable_cssu_destination(
            simulation_seed=simulation_seed,
            order_id=event_id,
        )
    share_a, _ = _normalise_cssu_weights(weights)
    return (
        "A"
        if event_keyed_uniform_u64(
            simulation_seed=simulation_seed,
            event_id=event_id,
            namespace=namespace,
        )
        < share_a
        else "B"
    )


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

    ``reallocate_unused`` only reallocates genuinely unused spare capacity.  It
    is *not* a full-pooling, action-invariant null: when both destinations can
    absorb their shares, ``allocation_a`` still changes the ledger.  A true
    action-invariant pooling arm therefore requires a separate contract and is
    not implemented by this historical primitive.  The shipped default remains
    unchanged so that prior artifacts keep their original semantics.
    """
    allocation_a = validate_allocation_a(allocation_a)
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
