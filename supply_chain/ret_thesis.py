from __future__ import annotations

from collections import Counter
import math
from typing import Any, Iterable, Mapping

import numpy as np

from .config import BACKORDER_QUEUE_CAP

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
    - the no-risk branch uses the workbook's capped backlog-list size ``Bt``;
    - values are not clipped, matching Excel.
    """
    if j <= 0:
        raise ValueError("j must be a positive 1-based order index.")

    # Unfulfilled orders (never delivered: lost/dropped or still pending at the
    # horizon) have no OATj. Garrido serves every order eventually (very late) and
    # scores them ~0 via the recovery branch; our DES drops overflow orders, so
    # without this guard they fall into the no-risk fill-rate branch and score ~1.0,
    # inflating mean ReT. Score them 0 (unfulfilled), matching the thesis
    # compute_ret_per_order and Garrido's lost-order ReT (~0.002).
    if getattr(order, "OATj", None) is None:
        return 0.0, "excel_unfulfilled"

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

    # Garrido's backlog ledger is capped at 60 orders.  The DES accumulator can
    # grow into the thousands over long runs, but the raw workbook ``sumBt``
    # column behaves like the capped backlog-list size used in Sec. 6.5.4.
    bt = min(int(cumulative_backorders), int(BACKORDER_QUEUE_CAP))
    value = 1.0 - ((bt + int(cumulative_unattended)) / int(j))
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
    no-risk branch receives the running unattended count and a capped backlog
    count, mirroring the raw workbook columns ``sum(Bt)`` and ``sum(Ut)``.

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
            "excel_unfulfilled": int(case_counts["excel_unfulfilled"]),
        },
        "n_orders": len(order_list),
        "n_completed": sum(
            1 for order in order_list if getattr(order, "OATj", None) is not None
        ),
    }


def compute_order_level_ret_excel_visible_ledger(
    orders: Iterable[Any],
    *,
    current_time: float | None = None,
    emit_order_ids: set[int] | None = None,
) -> dict[str, Any]:
    """Reproduce the workbook's sparse visible-order ledger.

    Garrido's raw sheets preserve the original order index ``j`` but omit many
    orders that are lost or remain unfulfilled.  Their effects survive through
    the cumulative ``Bt``/``Ut`` columns.  This aggregator therefore walks the
    complete DES order history to update those ledgers, while emitting ReT only
    for completed, non-lost rows—the population actually visible in the raw
    workbooks.
    """
    order_list = sorted(
        list(orders),
        key=lambda order: (
            int(getattr(order, "j", 0) or 0),
            float(getattr(order, "OPTj", 0.0) or 0.0),
        ),
    )
    visible_values: list[float] = []
    visible_rows: list[dict[str, Any]] = []
    case_counts: Counter[str] = Counter()
    last_backorders = 0
    last_unattended = 0

    ledger_events: list[tuple[float, int, int, int]] = []
    visible_orders: list[Any] = []
    for candidate in order_list:
        lost = bool(getattr(candidate, "lost", False))
        candidate_oat = getattr(candidate, "OATj", None)
        lost_time = getattr(candidate, "lost_time", None)
        emit_candidate = (
            emit_order_ids is None
            or int(getattr(candidate, "j", 0) or 0) in emit_order_ids
        )
        if not lost and candidate_oat is not None and emit_candidate:
            visible_orders.append(candidate)

        opt = float(getattr(candidate, "OPTj", 0.0) or 0.0)
        lt = float(getattr(candidate, "LTj", 0.0) or 0.0)
        activation = opt + lt
        end_candidates = [
            float(value)
            for value in (candidate_oat, lost_time)
            if value is not None
        ]
        end_time = min(end_candidates) if end_candidates else float("inf")
        if end_time > activation:
            ledger_events.append((activation, 1, +1, 0))
            if math.isfinite(end_time):
                ledger_events.append((end_time, 0, -1, 0))
        if lost and lost_time is not None:
            ledger_events.append((float(lost_time), 0, 0, +1))

    ledger_events.sort()
    visible_orders.sort(
        key=lambda order: (
            float(getattr(order, "OATj", 0.0) or 0.0),
            int(getattr(order, "j", 0) or 0),
        )
    )
    event_index = 0
    current_backorders = 0
    cumulative_unattended = 0
    ledger_snapshots: dict[int, tuple[int, int]] = {}
    for idx, order in enumerate(visible_orders, start=1):
        row_time = float(getattr(order, "OATj", 0.0) or 0.0)
        while (
            event_index < len(ledger_events)
            and ledger_events[event_index][0] <= row_time
        ):
            _, _, bt_delta, ut_delta = ledger_events[event_index]
            current_backorders += bt_delta
            cumulative_unattended += ut_delta
            event_index += 1
        current_backorders = max(
            0, min(current_backorders, int(BACKORDER_QUEUE_CAP))
        )

        j_value = int(getattr(order, "j", idx) or idx)
        ret, case = compute_ret_per_order_excel_formula(
            order,
            j=j_value,
            cumulative_backorders=current_backorders,
            cumulative_unattended=cumulative_unattended,
        )
        visible_values.append(float(ret))
        visible_rows.append(
            {
                "j": j_value,
                "opt": float(getattr(order, "OPTj", 0.0) or 0.0),
                "quantity": float(getattr(order, "quantity", 0.0) or 0.0),
                "ret": float(ret),
                "case": str(case),
            }
        )
        case_counts[case] += 1
        ledger_snapshots[j_value] = (
            current_backorders,
            cumulative_unattended,
        )

    max_j = max(
        (int(getattr(order, "j", 0) or 0) for order in order_list),
        default=0,
    )
    max_visible_j = max(ledger_snapshots, default=0)
    if max_visible_j:
        last_backorders, last_unattended = ledger_snapshots[max_visible_j]
    return {
        "mean_ret_excel": (
            float(np.mean(visible_values)) if visible_values else 1.0
        ),
        "case_counts": {
            "excel_fill_rate": int(case_counts["excel_fill_rate"]),
            "excel_autotomy": int(case_counts["excel_autotomy"]),
            "excel_recovery": int(case_counts["excel_recovery"]),
            "excel_risk_no_recovery": int(
                case_counts["excel_risk_no_recovery"]
            ),
        },
        "n_generated_orders": len(order_list),
        "max_order_index": max_j,
        "n_visible_rows": len(visible_values),
        "n_omitted_rows": len(order_list) - len(visible_values),
        "final_backorders": last_backorders,
        "final_unattended": last_unattended,
        "ret_values": visible_values,
        "ret_rows": visible_rows,
    }


def compute_order_level_ret_excel_request_snapshot_ledger(
    orders: Iterable[Any],
    *,
    current_time: float | None = None,
    emit_order_ids: set[int] | None = None,
) -> dict[str, Any]:
    """Reconstruct the workbook-visible ledger at request generation.

    Garrido-Rios (2017), Annex B, printed page 169, describes a barrier matrix
    associated with each *request generated*.  The matrix carries the order
    number, generation time, and accumulated backorder/lost-order counts to
    Op9.  The raw workbooks likewise keep rows in increasing ``j``/``OPTj``
    order even though ``OATj`` repeatedly reverses.  Consequently Bt/Ut are
    request-time snapshots, not completion-time reconstructions.

    Native DES orders should carry ``ret_bt_at_request`` and
    ``ret_ut_at_request`` captured at ``OPTj``.  For stylized adapters that
    lack those fields, this function reconstructs the same snapshot from the
    complete order history using half-open backorder intervals
    ``[OPTj + LTj, min(OATj, lost_time))`` and lost events observed by ``OPTj``.
    The fallback is deterministic but must not be described as workbook-
    validated when the adapter omits orders or event times.
    """
    del current_time  # the request snapshot is historical, not horizon-relative
    order_list = sorted(
        list(orders),
        key=lambda order: (
            int(getattr(order, "j", 0) or 0),
            float(getattr(order, "OPTj", 0.0) or 0.0),
        ),
    )
    visible_orders = [
        order
        for order in order_list
        if not bool(getattr(order, "lost", False))
        and getattr(order, "OATj", None) is not None
        and (
            emit_order_ids is None
            or int(getattr(order, "j", 0) or 0) in emit_order_ids
        )
    ]
    visible_values: list[float] = []
    visible_rows: list[dict[str, Any]] = []
    case_counts: Counter[str] = Counter()
    last_backorders = 0
    last_unattended = 0

    for idx, order in enumerate(visible_orders, start=1):
        row_time = float(getattr(order, "OPTj", 0.0) or 0.0)
        explicit_bt = getattr(order, "ret_bt_at_request", None)
        explicit_ut = getattr(order, "ret_ut_at_request", None)
        if explicit_bt is not None and explicit_ut is not None:
            current_backorders = int(explicit_bt)
            cumulative_unattended = int(explicit_ut)
            snapshot_source = "captured_at_request"
        else:
            current_backorders = 0
            cumulative_unattended = 0
            for candidate in order_list:
                if candidate is order:
                    continue
                opt = float(getattr(candidate, "OPTj", 0.0) or 0.0)
                # Request time, not order id, is the causal boundary.  This
                # also excludes every same-time new request from the current
                # row without assuming that ``j`` is chronological.
                if opt >= row_time:
                    continue
                lt = float(getattr(candidate, "LTj", 0.0) or 0.0)
                activation = opt + lt
                candidate_oat = getattr(candidate, "OATj", None)
                lost_time = getattr(candidate, "lost_time", None)
                end_candidates = [
                    float(value)
                    for value in (candidate_oat, lost_time)
                    if value is not None
                ]
                end_time = min(end_candidates) if end_candidates else float("inf")
                if activation <= row_time < end_time:
                    current_backorders += 1
                if (
                    bool(getattr(candidate, "lost", False))
                    and lost_time is not None
                    and float(lost_time) <= row_time
                ):
                    cumulative_unattended += 1
            snapshot_source = "reconstructed_from_complete_history"

        current_backorders = max(
            0, min(current_backorders, int(BACKORDER_QUEUE_CAP))
        )
        cumulative_unattended = max(0, cumulative_unattended)
        j_value = int(getattr(order, "j", idx) or idx)
        ret, case = compute_ret_per_order_excel_formula(
            order,
            j=j_value,
            cumulative_backorders=current_backorders,
            cumulative_unattended=cumulative_unattended,
        )
        visible_values.append(float(ret))
        visible_rows.append(
            {
                "j": j_value,
                "opt": row_time,
                "quantity": float(getattr(order, "quantity", 0.0) or 0.0),
                "sum_bt": current_backorders,
                "sum_ut": cumulative_unattended,
                "snapshot_source": snapshot_source,
                "snapshot_time": getattr(
                    order, "ret_ledger_snapshot_time", None
                ),
                "snapshot_event_sequence": getattr(
                    order, "ret_ledger_event_sequence", None
                ),
                "ret": float(ret),
                "case": str(case),
            }
        )
        case_counts[case] += 1
        last_backorders = current_backorders
        last_unattended = cumulative_unattended

    max_j = max(
        (int(getattr(order, "j", 0) or 0) for order in order_list),
        default=0,
    )
    return {
        "contract_version": "ret_excel_request_snapshot_v2",
        "mean_ret_excel": (
            float(np.mean(visible_values)) if visible_values else 1.0
        ),
        "case_counts": {
            "excel_fill_rate": int(case_counts["excel_fill_rate"]),
            "excel_autotomy": int(case_counts["excel_autotomy"]),
            "excel_recovery": int(case_counts["excel_recovery"]),
            "excel_risk_no_recovery": int(
                case_counts["excel_risk_no_recovery"]
            ),
        },
        "n_generated_orders": len(order_list),
        "max_order_index": max_j,
        "n_visible_rows": len(visible_values),
        "n_omitted_rows": len(order_list) - len(visible_values),
        "final_backorders": last_backorders,
        "final_unattended": last_unattended,
        "ret_values": visible_values,
        "ret_rows": visible_rows,
    }
