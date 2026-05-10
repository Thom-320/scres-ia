from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping

import numpy as np

DEFAULT_RET_WEIGHTS = {"max": 1.0, "mean": 0.5, "min": 0.0}


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
            return max(0.0, value), "recovery"
        value = float(weights["min"]) * ((dp - rp) / max(float(ct), 1e-9))
        return max(0.0, value), "non_recovery"

    return max(0.0, min(1.0, float(fill_rate))), "fill_rate"


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
        if case != "unfulfilled":
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
