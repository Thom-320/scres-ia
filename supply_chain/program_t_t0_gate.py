"""Quality-time frontier and fail-closed residual-headroom gate for T0."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ComparatorPoint:
    controller_id: str
    mean_ret: float
    p95_online_ms: float
    feasible: bool


def quality_time_frontier(points: Sequence[ComparatorPoint]) -> tuple[ComparatorPoint, ...]:
    eligible = [point for point in points if point.feasible]
    return tuple(
        point
        for point in sorted(eligible, key=lambda row: (row.p95_online_ms, -row.mean_ret))
        if not any(
            other.p95_online_ms <= point.p95_online_ms
            and other.mean_ret >= point.mean_ret
            and (other.p95_online_ms < point.p95_online_ms or other.mean_ret > point.mean_ret)
            for other in eligible
        )
    )


def paired_bootstrap_lcb(
    candidate: Sequence[float],
    comparator: Sequence[float],
    *,
    seed: int = 20260720,
    resamples: int = 20_000,
) -> float:
    delta = np.asarray(candidate, dtype=float) - np.asarray(comparator, dtype=float)
    if delta.ndim != 1 or len(delta) < 2 or not np.all(np.isfinite(delta)):
        raise ValueError("paired finite tape vectors with at least two rows are required")
    rng = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=float)
    for start in range(0, resamples, 1000):
        width = min(1000, resamples - start)
        indices = rng.integers(0, len(delta), size=(width, len(delta)))
        means[start : start + width] = delta[indices].mean(axis=1)
    return float(np.quantile(means, 0.05, method="lower"))


def adjudicate_t0_residual(
    *,
    best_observable_ret: Sequence[float],
    reinforced_mpc_ret: Sequence[float],
    worst_product_delta: Sequence[float],
    lost_order_delta: Sequence[float],
    resource_delta: Sequence[float],
) -> dict[str, object]:
    residual_lcb = paired_bootstrap_lcb(best_observable_ret, reinforced_mpc_ret)
    checks = {
        "residual_lcb95_at_least_0_015": residual_lcb >= 0.015,
        "worst_product_lcb_at_least_minus_0_02": paired_bootstrap_lcb(
            worst_product_delta, np.zeros(len(worst_product_delta))
        ) >= -0.02,
        "lost_orders_nonincrease": float(np.mean(lost_order_delta)) <= 1e-12,
        "resources_exact": bool(np.all(np.abs(resource_delta) <= 1e-12)),
    }
    passed = all(checks.values())
    return {
        "status": "PASS_T0_RESIDUAL_HEADROOM__HYBRID_COMPONENT_FIT_AUTHORIZED"
        if passed
        else "STOP_T0_NO_SAFE_RESIDUAL_HEADROOM",
        "residual_lcb95": residual_lcb,
        "checks": checks,
        "new_scientific_seeds_authorized": False,
        "hybrid_confirmation_authorized": False,
    }

