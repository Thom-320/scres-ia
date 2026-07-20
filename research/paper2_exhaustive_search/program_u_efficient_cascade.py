"""Fail-closed scheduling utilities for the Program U risk screen.

This module does not simulate or open seeds.  It turns precomputed exactness and
first-tape diagnostics into a deterministic work plan so an entire mask or
design is not expanded before cheap failure gates have passed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class CandidatePoint:
    point_id: str
    mask: str
    severity: float


@dataclass(frozen=True)
class FirstTapeResult:
    point_id: str
    exact: bool
    preliminary_h_pi_safe: float
    replay_error: float


@dataclass(frozen=True)
class CascadeDecision:
    stopped_masks: tuple[str, ...]
    promoted_points: tuple[str, ...]
    certification_points: tuple[str, ...]


def decide_incremental_screen(
    *,
    candidates: Sequence[CandidatePoint],
    first_tape: Sequence[FirstTapeResult],
    preliminary_threshold: float,
    certification_limit: int = 12,
) -> CascadeDecision:
    """Apply exactness, one-tape headroom, and finalist-only certification.

    A single inexact point stops its physical mask prospectively because the
    shared transducer is not certified for that mask.  Other masks continue.
    Among exact masks, points pass only if their first-tape safe headroom clears
    the frozen threshold.  Lowest severity is preferred for certification.
    """
    by_id = {row.point_id: row for row in candidates}
    if len(by_id) != len(candidates):
        raise ValueError("candidate point ids must be unique")
    results = {row.point_id: row for row in first_tape}
    unknown = set(results) - set(by_id)
    if unknown:
        raise ValueError(f"first-tape results contain unknown points: {sorted(unknown)}")
    missing = set(by_id) - set(results)
    if missing:
        raise ValueError(f"first-tape results missing points: {sorted(missing)}")

    stopped_masks = {
        by_id[point_id].mask for point_id, row in results.items() if not row.exact
    }
    promoted = [
        row
        for row in candidates
        if row.mask not in stopped_masks
        and results[row.point_id].exact
        and results[row.point_id].preliminary_h_pi_safe >= preliminary_threshold
    ]
    promoted.sort(key=lambda row: (row.severity, row.point_id))
    certified = promoted[: int(certification_limit)]
    return CascadeDecision(
        stopped_masks=tuple(sorted(stopped_masks)),
        promoted_points=tuple(row.point_id for row in promoted),
        certification_points=tuple(row.point_id for row in certified),
    )


def batch_points(
    point_ids: Sequence[str], *, batch_size: int
) -> tuple[tuple[str, ...], ...]:
    """Group multiple points per worker process to amortize Python startup."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return tuple(
        tuple(point_ids[start : start + batch_size])
        for start in range(0, len(point_ids), batch_size)
    )


def certification_mode(
    point_id: str, certification_points: Iterable[str]
) -> str:
    """Return the frozen evaluation mode for a point."""
    return "EXACT_FRONTIER" if point_id in set(certification_points) else "APPROXIMATE_SEARCH"

