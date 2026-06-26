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


def order_has_ret_risk_indicator(order: Any) -> bool:
    """Infer the spreadsheet's per-order risk-indicator gate.

    Garrido-Rios's raw Excel files compute ReT from a risk-active switch:
    ``AVERAGE(R..)>0``.  Prefer explicit per-order DES risk indicators when
    available; fall back to APj/RPj/DPj for older synthetic tests.
    """
    indicators = getattr(order, "ret_risk_indicators", None)
    if indicators:
        return any(float(value or 0.0) > 0.0 for value in indicators.values())
    return any(
        float(getattr(order, attr, 0.0) or 0.0) > 0.0
        for attr in ("APj", "RPj", "DPj")
    )


def compute_ret_per_order_excel_formula(
    order: Any,
    *,
    j: int,
    cumulative_backorders: int,
    cumulative_unattended: int,
    risk_active: bool | None = None,
) -> tuple[float, str]:
    """Compute the operational ReT formula used in Garrido's raw Excel files.

    The workbooks implement:

    ``IF(AVERAGE(risk_cols)>0, IF(APj>0, APj/LT, 0.5*(1/RPj)), 1-((Bt+Ut)/j))``

    Notes for exactness:
    - no CTj<=LTj guard is applied before ``APj/LT``;
    - DPj is not used directly by the visible spreadsheet formula;
    - the no-risk branch uses the running order-count fill term at row ``j``;
    - values are not clipped, matching Excel.
    """
    if j <= 0:
        raise ValueError("j must be a positive 1-based order index.")

    if risk_active is None:
        risk_active = order_has_ret_risk_indicator(order)

    ap = float(getattr(order, "APj", 0.0) or 0.0)
    rp = float(getattr(order, "RPj", 0.0) or 0.0)
    lt = float(getattr(order, "LTj", 0.0) or 0.0)

    if risk_active:
        if ap > 0.0:
            return ap / max(lt, 1e-9), "excel_autotomy"
        if rp > 0.0:
            return 0.5 * (1.0 / rp), "excel_recovery"
        return 0.0, "excel_risk_no_recovery"

    value = 1.0 - ((int(cumulative_backorders) + int(cumulative_unattended)) / int(j))
    return float(value), "excel_fill_rate"


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
    """Aggregate Garrido-Rios order-level ReT values across demanded orders."""
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


def compute_order_level_ret_excel_formula(
    orders: Iterable[Any],
    *,
    current_time: float | None = None,
    j_source: str = "row_index",
) -> dict[str, Any]:
    """Aggregate the Excel-faithful Garrido ReT over demanded orders.

    Orders are processed by their thesis order index ``j`` when available.  The
    no-risk branch receives the running cumulative backorder and unattended
    counts, mirroring the raw workbook columns ``sum(Bt)`` and ``sum(Ut)``.

    ``j_source`` defaults to ``"row_index"`` for backwards compatibility with
    the full DES ledger, where visible rows are consecutive.  Garrido's raw
    workbooks preserve the original order number when rows are filtered, so
    forensic workbook-visible ledgers should use ``j_source="order_j"``.
    """
    if j_source not in {"row_index", "order_j"}:
        raise ValueError("j_source must be 'row_index' or 'order_j'.")
    order_list = sorted(
        list(orders),
        key=lambda order: (
            int(getattr(order, "j", 0) or 0),
            float(getattr(order, "OPTj", 0.0) or 0.0),
        ),
    )
    case_counts: Counter[str] = Counter()
    ret_values: list[float] = []
    cumulative_backorders = 0
    cumulative_unattended = 0

    for idx, order in enumerate(order_list, start=1):
        if bool(getattr(order, "lost", False)):
            cumulative_unattended += 1
        elif order_counts_as_backorder_for_fill_rate(
            order, current_time=current_time
        ):
            cumulative_backorders += 1

        ret, case = compute_ret_per_order_excel_formula(
            order,
            j=(
                int(getattr(order, "j", idx) or idx)
                if j_source == "order_j"
                else idx
            ),
            cumulative_backorders=cumulative_backorders,
            cumulative_unattended=cumulative_unattended,
        )
        case_counts[case] += 1
        ret_values.append(ret)

    return {
        "mean_ret_excel": float(np.mean(ret_values)) if ret_values else 1.0,
        "case_counts": {
            "excel_fill_rate": int(case_counts["excel_fill_rate"]),
            "excel_autotomy": int(case_counts["excel_autotomy"]),
            "excel_recovery": int(case_counts["excel_recovery"]),
            "excel_risk_no_recovery": int(case_counts["excel_risk_no_recovery"]),
        },
        "n_orders": len(order_list),
        "n_completed": sum(
            1 for order in order_list if getattr(order, "OATj", None) is not None
        ),
    }
