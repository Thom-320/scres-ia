from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping

import numpy as np

DEFAULT_RET_WEIGHTS = {"max": 1.0, "mean": 0.5, "min": 0.0}


def order_counts_as_backorder_for_fill_rate(
    order: Any, *, current_time: float | None = None
) -> bool:
    """Return whether order ``j`` contributes to thesis ``B_t``.

    Garrido-Rios Eq. 5.4 uses accumulated backorders: an order that was
    delivered after ``LT_j`` still counted as a backorder for service-level
    resilience.  The live DES clears ``order.backorder`` once a delayed order is
    eventually served, so the exact audit metric must reconstruct ``B_t`` from
    ``CT_j > LT_j`` rather than from the current queue flag alone.
    """
    if bool(getattr(order, "lost", False)):
        return False

    lt = float(getattr(order, "LTj", 0.0) or 0.0)
    ct = getattr(order, "CTj", None)
    if ct is not None:
        return float(ct) > lt

    if bool(getattr(order, "backorder", False)):
        return True

    if current_time is None:
        return False

    remaining = float(getattr(order, "remaining_qty", 0.0) or 0.0)
    opt = float(getattr(order, "OPTj", 0.0) or 0.0)
    return remaining > 0.0 and (float(current_time) - opt) > lt


def compute_fill_rate_from_orders(
    orders: Iterable[Any], *, current_time: float | None = None
) -> float:
    """Compute thesis Eq. 5.4, ``Re(FR_t)=1-(B_t+U_t)/D_t``.

    ``D_t`` is the number of demanded orders. ``B_t`` counts orders delivered
    late or still pending beyond the promised lead time. ``U_t`` counts lost or
    unattended orders. Lost orders are not double-counted as backorders.
    """
    order_list = list(orders)
    dt = len(order_list)
    if dt == 0:
        return 1.0

    ut = sum(1 for order in order_list if bool(getattr(order, "lost", False)))
    bt = sum(
        1
        for order in order_list
        if order_counts_as_backorder_for_fill_rate(order, current_time=current_time)
    )
    return max(0.0, min(1.0, 1.0 - (bt + ut) / dt))


def compute_ret_per_order(
    order: Any,
    *,
    fill_rate: float,
    ret_weights: Mapping[str, float] | None = None,
) -> tuple[float, str]:
    """Compute Garrido-Rios Eq. 5.5 for one order record."""
    weights = dict(DEFAULT_RET_WEIGHTS if ret_weights is None else ret_weights)
    if getattr(order, "OATj", None) is None:
        return 0.0, "unfulfilled"

    ct = getattr(order, "CTj", None)
    lt = float(getattr(order, "LTj", 0.0) or 0.0)
    ap = float(getattr(order, "APj", 0.0) or 0.0)
    rp = float(getattr(order, "RPj", 0.0) or 0.0)
    dp = float(getattr(order, "DPj", 0.0) or 0.0)

    if ap > 0.0 and ct is not None and float(ct) <= lt:
        value = float(weights["max"]) * (ap / max(lt, 1e-9))
        return min(1.0, max(0.0, value)), "autotomy"

    if ct is not None and float(ct) > lt:
        if rp > 0.0:
            value = float(weights["mean"]) * (1.0 / rp)
            return min(1.0, max(0.0, value)), "recovery"
        value = float(weights["min"]) * ((dp - rp) / max(float(ct), 1e-9))
        return max(0.0, value), "non_recovery"

    return max(0.0, min(1.0, float(fill_rate))), "fill_rate"


def compute_ret_periods_continuous_per_order(order: Any, *, tau_hours: float = 48.0) -> float:
    """Continuous resilience on the thesis periods AP/RP/DP (smooth, no case switch).

    Re_j = (AP_j + tau) / (AP_j + RP_j + DP_j + tau)  in (0, 1];  0 if unfulfilled.

    Faithful limits, but differentiable everywhere (unlike the piecewise Eq. 5.5):
      * no disruption  (AP=RP=DP=0)        -> 1   (the fill_rate case);
      * autotomy       (AP>0, RP=DP=0)     -> 1   (absorbed within lead time);
      * recovery       (RP>0)              -> (AP+tau)/(AP+RP+DP+tau) < 1, decays with RP;
      * non-recovery   (DP large)          -> small;
      * unfulfilled    (OATj is None)      -> 0.
    ``tau`` (default = lead time, 48 h) sets how fast resilience decays with delay.
    """
    if getattr(order, "OATj", None) is None:
        return 0.0
    ap = max(0.0, float(getattr(order, "APj", 0.0) or 0.0))
    rp = max(0.0, float(getattr(order, "RPj", 0.0) or 0.0))
    dp = max(0.0, float(getattr(order, "DPj", 0.0) or 0.0))
    return float((ap + tau_hours) / (ap + rp + dp + tau_hours))


def compute_periods_continuous_ret(orders: Iterable[Any], *, tau_hours: float = 48.0) -> dict:
    """Aggregate the continuous AP/RP/DP resilience over served orders."""
    vals = [
        compute_ret_periods_continuous_per_order(o, tau_hours=tau_hours)
        for o in orders
        if getattr(o, "OATj", None) is not None
    ]
    return {
        "mean_ret_continuous": float(np.mean(vals)) if vals else 0.0,
        "n_served": len(vals),
        "tau_hours": float(tau_hours),
    }


def compute_order_level_ret(
    orders: Iterable[Any],
    *,
    fill_rate: float,
    ret_weights: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Aggregate Garrido-Rios order-level ReT values across completed orders."""
    order_list = list(orders)
    case_counts: Counter[str] = Counter()
    ret_values: list[float] = []
    for order in order_list:
        ret, case = compute_ret_per_order(
            order, fill_rate=fill_rate, ret_weights=ret_weights
        )
        case_counts[case] += 1
        ret_values.append(ret)

    return {
        "mean_ret": float(np.mean(ret_values)) if ret_values else float(fill_rate),
        "fill_rate_order_level": float(fill_rate),
        "case_counts": {
            "fill_rate": int(case_counts["fill_rate"]),
            "autotomy": int(case_counts["autotomy"]),
            "recovery": int(case_counts["recovery"]),
            "non_recovery": int(case_counts["non_recovery"]),
            "unfulfilled": int(case_counts["unfulfilled"]),
        },
        "n_orders": len(order_list),
        "n_completed": sum(
            1 for order in order_list if getattr(order, "OATj", None) is not None
        ),
    }
