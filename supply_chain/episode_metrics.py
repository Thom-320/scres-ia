"""Unified episode metrics panel — every outcome we track, in one place.

`compute_episode_metrics(sim)` returns the full ORDER-DERIVED panel (resilience, service,
order-time distributions, throughput) computed from a completed `MFSCSimulation` (after
`sim.run()`), reusing the canonical functions in `supply_chain.ret_thesis`. Resource metrics
(shift-hours, surge-hours, buffer use) depend on the action TRAJECTORY, not the order ledger, so
they are computed by the caller (static config or per-step harness) and merged via
`merge_resource_metrics`.

This is the single source for the "what to beat" static panel and the PPO-vs-static comparison,
so every policy is scored on identical metrics.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

from .ret_thesis import (
    compute_order_level_ret,
    compute_order_level_ret_excel_formula,
    compute_ret_per_order_excel_formula,
    compute_periods_continuous_ret,
    compute_fill_rate_from_orders,
    order_counts_as_backorder_for_fill_rate,
)


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    idx = min(len(xs) - 1, int(q * len(xs)))
    return float(xs[idx])


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _tail_mean(values: list[float], *, frac: float, lower_tail: bool) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    n = max(1, int((len(xs) * frac) + 0.999999))
    tail = xs[:n] if lower_tail else xs[-n:]
    return _mean(tail)


def _excel_ret_details(orders: list[Any], *, current_time: float) -> dict[str, Any]:
    """Return per-order Excel-ReT values, cases, and rolling 4w summaries.

    This mirrors ``compute_order_level_ret_excel_formula`` but keeps the
    distribution instead of only the mean/case counts.
    """
    ordered = sorted(
        orders,
        key=lambda order: (
            int(getattr(order, "j", 0) or 0),
            float(getattr(order, "OPTj", 0.0) or 0.0),
        ),
    )
    cumulative_backorders = 0
    cumulative_unattended = 0
    case_counts: Counter[str] = Counter()
    values: list[float] = []
    qty_values: list[tuple[float, float]] = []
    timed_values: list[tuple[float, float]] = []

    for idx, order in enumerate(ordered, start=1):
        if bool(getattr(order, "lost", False)):
            cumulative_unattended += 1
        elif order_counts_as_backorder_for_fill_rate(
            order, current_time=current_time
        ):
            cumulative_backorders += 1

        ret, case = compute_ret_per_order_excel_formula(
            order,
            j=idx,
            cumulative_backorders=cumulative_backorders,
            cumulative_unattended=cumulative_unattended,
        )
        ret = float(ret)
        values.append(ret)
        case_counts[str(case)] += 1
        qty_values.append((ret, float(getattr(order, "quantity", 0.0) or 0.0)))
        timed_values.append((float(getattr(order, "OPTj", 0.0) or 0.0), ret))

    # Rolling 4-week Excel ReT by order placement time, for temporal stability.
    rolling: list[float] = []
    window_hours = 24.0 * 7.0 * 4.0
    left = 0
    running_sum = 0.0
    timed_values.sort(key=lambda item: item[0])
    for right, (opt, ret) in enumerate(timed_values):
        running_sum += ret
        while left <= right and timed_values[left][0] < opt - window_hours:
            running_sum -= timed_values[left][1]
            left += 1
        denom = right - left + 1
        if denom > 0:
            rolling.append(running_sum / denom)

    qty_total = sum(qty for _, qty in qty_values)
    ration_ret = (
        sum(ret * qty for ret, qty in qty_values) / qty_total
        if qty_total > 0.0 else 0.0
    )

    return {
        "values": values,
        "case_counts": case_counts,
        "ration_ret_excel": float(ration_ret),
        "rolling_4w_values": rolling,
    }


def compute_episode_metrics(
    sim: Any,
    *,
    treatment_start: float | None = None,
) -> dict[str, float]:
    """Full order-derived metrics panel for a completed simulation.

    Orders placed before ``treatment_start`` (default = end of warm-up) are excluded so the
    metric reflects only the period the policy could influence.
    """
    horizon = float(sim.env.now)
    start = float(sim.warmup_time if treatment_start is None else treatment_start)
    orders = [
        o
        for o in sim.orders
        if not bool(getattr(o, "metrics_excluded", False))
        and float(getattr(o, "OPTj", 0.0)) >= start
    ]
    n = len(orders) or 1

    served = [o for o in orders if getattr(o, "OATj", None) is not None]
    lost = [o for o in orders if bool(getattr(o, "lost", False))]
    on_time = [
        o for o in served
        if o.CTj is not None and float(o.CTj) <= float(o.LTj or 0.0)
    ]
    late = [o for o in served if o.CTj is not None and float(o.CTj) > float(o.LTj or 0.0)]

    fill_rate = compute_fill_rate_from_orders(orders, current_time=horizon)
    excel = compute_order_level_ret_excel_formula(orders, current_time=horizon)
    excel_details = _excel_ret_details(orders, current_time=horizon)
    thesis = compute_order_level_ret(orders, fill_rate=fill_rate)
    cont = compute_periods_continuous_ret(orders)

    # order-time distributions (served orders)
    apj = [float(getattr(o, "APj", 0.0) or 0.0) for o in orders]
    apj_pos = [x for x in apj if x > 0.0]
    ctj = [float(o.CTj) for o in served if o.CTj is not None]
    rpj = [float(o.RPj) for o in orders if float(getattr(o, "RPj", 0.0) or 0.0) > 0.0]
    dpj = [float(o.DPj) for o in orders if float(getattr(o, "DPj", 0.0) or 0.0) > 0.0]
    ret_values = list(excel_details["values"])
    rolling_ret = list(excel_details["rolling_4w_values"])
    excel_cases = excel_details["case_counts"]

    # service-loss area (AUC): qty-weighted late-hours beyond the lead-time promise.
    # An order is "service loss" for max(0, end - (OPTj+LT)) hours, end = OATj or horizon.
    service_loss_auc = 0.0
    backlog_ages: list[float] = []
    for o in orders:
        opt = float(o.OPTj or 0.0)
        lt = float(o.LTj or 0.0)
        end = float(o.OATj) if getattr(o, "OATj", None) is not None else horizon
        lateness = max(0.0, end - (opt + lt))
        service_loss_auc += lateness * float(o.quantity or 0.0)
        if getattr(o, "OATj", None) is None:  # still pending / lost at horizon
            backlog_ages.append(max(0.0, horizon - opt))

    delivered = sum(float(o.quantity or 0.0) for o in served)
    demanded = sum(float(o.quantity or 0.0) for o in orders)

    return {
        # counts
        "n_orders": float(len(orders)),
        "n_served": float(len(served)),
        "n_lost": float(len(lost)),
        "n_late": float(len(late)),
        # resilience (the bars)
        "ret_excel": float(excel["mean_ret_excel"]),
        "ret_thesis": float(thesis["mean_ret"]),
        "ret_continuous": float(cont["mean_ret_continuous"]),
        "ret_excel_cvar05": _tail_mean(ret_values, frac=0.05, lower_tail=True),
        "ret_excel_p05": _pct(ret_values, 0.05),
        "ret_excel_p10": _pct(ret_values, 0.10),
        "ret_excel_p25": _pct(ret_values, 0.25),
        "ret_excel_p50": _pct(ret_values, 0.50),
        "ret_excel_p75": _pct(ret_values, 0.75),
        "ret_excel_p90": _pct(ret_values, 0.90),
        "ret_excel_p95": _pct(ret_values, 0.95),
        "ret_excel_rolling_4w_mean": _mean(rolling_ret),
        "ret_excel_rolling_4w_min": float(min(rolling_ret)) if rolling_ret else 0.0,
        "ret_excel_rolling_4w_final": rolling_ret[-1] if rolling_ret else 0.0,
        "ration_ret_excel": float(excel_details["ration_ret_excel"]),
        "excel_case_pct_fill_rate": 100.0 * excel_cases["excel_fill_rate"] / n,
        "excel_case_pct_autotomy": 100.0 * excel_cases["excel_autotomy"] / n,
        "excel_case_pct_recovery": 100.0 * excel_cases["excel_recovery"] / n,
        "excel_case_pct_risk_no_recovery": (
            100.0 * excel_cases["excel_risk_no_recovery"] / n
        ),
        "excel_case_pct_unfulfilled": 100.0 * excel_cases["excel_unfulfilled"] / n,
        # service
        "fill_rate": float(fill_rate),
        "fill_rate_on_time": float(len(on_time) / n),
        "lost_orders": float(len(lost)),
        "lost_rate": float(len(lost) / n),
        "backorder_qty_final": float(getattr(sim, "pending_backorder_qty", 0.0) or 0.0),
        "service_loss_auc_ration_hours": float(service_loss_auc),
        "service_loss_auc_per_order": float(service_loss_auc / n),
        # time-to-recovery (RPj over disrupted orders)
        "ttr_mean": _mean(rpj),
        "ttr_p95": _pct(rpj, 0.95),
        "backlog_age_mean": _mean(backlog_ages),
        "backlog_age_max": float(max(backlog_ages)) if backlog_ages else 0.0,
        # order-time distributions
        "apj_p50": _pct(apj, 0.50),
        "apj_p90": _pct(apj, 0.90),
        "apj_p99": _pct(apj, 0.99),
        "apj_positive_p50": _pct(apj_pos, 0.50),
        "apj_positive_p90": _pct(apj_pos, 0.90),
        "apj_positive_p99": _pct(apj_pos, 0.99),
        "ctj_p50": _pct(ctj, 0.50),
        "ctj_p90": _pct(ctj, 0.90),
        "ctj_p99": _pct(ctj, 0.99),
        "rpj_p50": _pct(rpj, 0.50),
        "rpj_p90": _pct(rpj, 0.90),
        "rpj_p99": _pct(rpj, 0.99),
        "dpj_p99": _pct(dpj, 0.99),
        # throughput
        "delivered_rations": float(delivered),
        "demanded_rations": float(demanded),
        "flow_fill_rate": float(delivered / demanded) if demanded > 0 else 1.0,
    }


def merge_resource_metrics(
    panel: dict[str, float],
    *,
    shift_hours: float,
    extra_shift_hours: float,
    strategic_buffer_units: float,
    end_state_inventory: float | None = None,
) -> dict[str, float]:
    """Merge action-trajectory resource metrics into an order-derived panel.

    Resources depend on the policy trajectory: ``shift_hours`` = Σ shifts·Δt,
    ``extra_shift_hours`` = Σ (shifts−1)·Δt (the surge cost beyond S1), and
    ``strategic_buffer_units`` = mean held buffer target. ``unit_cost_per_ration`` is a
    simple efficiency proxy: extra-shift-hours per delivered ration (lower = more efficient).
    """
    out = dict(panel)
    delivered = float(panel.get("delivered_rations", 0.0)) or 1.0
    out["shift_hours"] = float(shift_hours)
    out["surge_hours"] = float(extra_shift_hours)
    out["strategic_buffer_units"] = float(strategic_buffer_units)
    if end_state_inventory is not None:
        out["end_state_inventory"] = float(end_state_inventory)
    out["unit_surge_hours_per_ration"] = float(extra_shift_hours / delivered)
    out["unit_buffer_units_per_ration"] = float(strategic_buffer_units / delivered)
    return out


METRIC_KEYS: tuple[str, ...] = (
    "ret_excel", "ret_thesis", "ret_continuous",
    "ret_excel_cvar05", "ret_excel_p05", "ret_excel_p10", "ret_excel_p25",
    "ret_excel_p50", "ret_excel_p75", "ret_excel_p90", "ret_excel_p95",
    "ret_excel_rolling_4w_mean", "ret_excel_rolling_4w_min",
    "ret_excel_rolling_4w_final", "ration_ret_excel",
    "excel_case_pct_fill_rate", "excel_case_pct_autotomy",
    "excel_case_pct_recovery", "excel_case_pct_risk_no_recovery",
    "excel_case_pct_unfulfilled",
    "fill_rate", "fill_rate_on_time", "lost_orders", "lost_rate",
    "backorder_qty_final", "service_loss_auc_ration_hours", "service_loss_auc_per_order",
    "ttr_mean", "ttr_p95", "backlog_age_mean", "backlog_age_max",
    "apj_p50", "apj_p90", "apj_p99", "apj_positive_p50", "apj_positive_p90",
    "apj_positive_p99",
    "ctj_p50", "ctj_p90", "ctj_p99", "rpj_p50", "rpj_p90", "rpj_p99", "dpj_p99",
    "delivered_rations", "demanded_rations", "flow_fill_rate",
    "shift_hours", "surge_hours", "strategic_buffer_units",
    "unit_surge_hours_per_ration", "unit_buffer_units_per_ration",
)
