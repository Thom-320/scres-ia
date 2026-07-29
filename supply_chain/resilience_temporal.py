"""Secondary time-resolved resilience diagnostics for completed MFSC episodes.

These diagnostics never replace or optimize the canonical Garrido ReT.  They describe
service-loss depth, area, and system recovery around clusters of realized risk events.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from .config import HOURS_PER_DAY, HOURS_PER_WEEK


def _value(item: Any, field: str, default: float = 0.0) -> float:
    if isinstance(item, Mapping):
        return float(item.get(field, default))
    return float(getattr(item, field, default))


def cluster_risk_events(
    events: Iterable[Any],
    *,
    treatment_start: float,
    treatment_end: float,
    gap_hours: float = HOURS_PER_WEEK,
) -> list[dict[str, Any]]:
    rows = sorted(
        (
            {
                "risk_id": str(
                    event.get("risk_id", "")
                    if isinstance(event, Mapping)
                    else getattr(event, "risk_id", "")
                ),
                "start": max(treatment_start, _value(event, "start_time")),
                "end": min(treatment_end, _value(event, "end_time")),
            }
            for event in events
            if _value(event, "end_time") >= treatment_start
            and _value(event, "start_time") <= treatment_end
        ),
        key=lambda row: (row["start"], row["risk_id"]),
    )
    clusters: list[dict[str, Any]] = []
    for row in rows:
        row["end"] = max(row["start"], row["end"])
        if not clusters or row["start"] - clusters[-1]["event_end"] >= gap_hours:
            clusters.append(
                {
                    "onset": row["start"],
                    "event_end": row["end"],
                    "risk_ids": [row["risk_id"]],
                }
            )
        else:
            clusters[-1]["event_end"] = max(clusters[-1]["event_end"], row["end"])
            clusters[-1]["risk_ids"].append(row["risk_id"])
    return clusters


def _daily_service_history(
    orders: list[Any],
    *,
    start: float,
    end: float,
    step_hours: float,
    due_lookback_hours: float,
) -> list[dict[str, float]]:
    history: list[dict[str, float]] = []
    for time in np.arange(start, end + 1e-9, step_hours):
        due = []
        outstanding = []
        for order in orders:
            opt = float(getattr(order, "OPTj", 0.0) or 0.0)
            lt = float(getattr(order, "LTj", 0.0) or 0.0)
            due_time = opt + lt
            if not (time - due_lookback_hours <= due_time <= time):
                continue
            qty = float(getattr(order, "quantity", 0.0) or 0.0)
            due.append(qty)
            oat = getattr(order, "OATj", None)
            if oat is None or float(oat) > time:
                outstanding.append(qty)
        due_qty = float(sum(due))
        outstanding_qty = float(sum(outstanding))
        service = 1.0 if due_qty <= 0.0 else 1.0 - min(1.0, outstanding_qty / due_qty)
        history.append(
            {
                "time": float(time),
                "service": float(service),
                "due_qty": due_qty,
                "outstanding_overdue_qty": outstanding_qty,
            }
        )
    return history


def compute_temporal_resilience_panel(
    sim: Any,
    *,
    treatment_start: float | None = None,
    sample_hours: float = HOURS_PER_DAY,
    due_lookback_hours: float = 4.0 * HOURS_PER_WEEK,
    baseline_hours: float = 2.0 * HOURS_PER_WEEK,
    cluster_gap_hours: float = HOURS_PER_WEEK,
    cluster_window_hours: float = 8.0 * HOURS_PER_WEEK,
    recovery_fraction: float = 0.95,
    recovery_consecutive_days: int = 7,
) -> dict[str, Any]:
    """Compute a preregistered secondary panel from the completed order ledger."""
    if min(sample_hours, due_lookback_hours, baseline_hours, cluster_window_hours) <= 0:
        raise ValueError("temporal windows must be positive")
    if recovery_consecutive_days <= 0:
        raise ValueError("recovery_consecutive_days must be positive")
    if not 0.0 < recovery_fraction <= 1.0:
        raise ValueError("recovery_fraction must lie in (0, 1]")

    horizon = float(sim.env.now)
    start = float(sim.warmup_time if treatment_start is None else treatment_start)
    orders = [
        order
        for order in sim.orders
        if not bool(getattr(order, "metrics_excluded", False))
        and float(getattr(order, "OPTj", 0.0) or 0.0) >= start
    ]
    clusters = cluster_risk_events(
        sim.risk_events,
        treatment_start=start,
        treatment_end=horizon,
        gap_hours=cluster_gap_hours,
    )
    history = _daily_service_history(
        orders,
        start=start,
        end=horizon,
        step_hours=sample_hours,
        due_lookback_hours=due_lookback_hours,
    )

    records: list[dict[str, Any]] = []
    recovered_ttr: list[float] = []
    for cluster_index, cluster in enumerate(clusters):
        onset = float(cluster["onset"])
        event_end = float(cluster["event_end"])
        window_end = min(horizon, onset + cluster_window_hours)
        baseline_rows = [
            row for row in history if onset - baseline_hours <= row["time"] < onset
        ]
        baseline_service = (
            float(np.median([row["service"] for row in baseline_rows]))
            if baseline_rows
            else 1.0
        )
        baseline_backlog = (
            float(np.median([row["outstanding_overdue_qty"] for row in baseline_rows]))
            if baseline_rows
            else 0.0
        )
        window_rows = [row for row in history if onset <= row["time"] <= window_end]
        loss_auc = float(
            sum(row["outstanding_overdue_qty"] * sample_hours for row in window_rows)
        )
        max_drop = float(
            max((baseline_service - row["service"] for row in window_rows), default=0.0)
        )
        consecutive = 0
        ttr: float | None = None
        for row in window_rows:
            if row["time"] < event_end:
                continue
            healthy = (
                row["service"] >= recovery_fraction * baseline_service
                and row["outstanding_overdue_qty"] <= 1.05 * baseline_backlog + 1e-9
            )
            consecutive = consecutive + 1 if healthy else 0
            if consecutive >= recovery_consecutive_days:
                ttr = float(row["time"] - onset)
                recovered_ttr.append(ttr)
                break
        records.append(
            {
                "cluster_index": cluster_index,
                "onset": onset,
                "event_end": event_end,
                "window_end": window_end,
                "risk_ids": sorted(set(cluster["risk_ids"])),
                "baseline_service": baseline_service,
                "baseline_outstanding_overdue_qty": baseline_backlog,
                "service_loss_auc_ration_hours": loss_auc,
                "maximum_service_drop": max(0.0, max_drop),
                "system_ttr_hours": ttr,
                "right_censored": ttr is None,
            }
        )

    n_clusters = len(records)
    n_censored = sum(bool(row["right_censored"]) for row in records)
    return {
        "temporal_panel_version": "risk_cluster_daily_v1",
        "temporal_service_loss_auc_ration_hours": float(
            sum(row["service_loss_auc_ration_hours"] for row in records)
        ),
        "temporal_maximum_service_drop": float(
            max((row["maximum_service_drop"] for row in records), default=0.0)
        ),
        "system_ttr_mean": float(np.mean(recovered_ttr)) if recovered_ttr else 0.0,
        "system_ttr_p95": (
            float(np.quantile(np.asarray(recovered_ttr), 0.95)) if recovered_ttr else 0.0
        ),
        "system_ttr_n_clusters": float(n_clusters),
        "system_ttr_n_recovered": float(len(recovered_ttr)),
        "system_ttr_n_censored": float(n_censored),
        "system_ttr_censored_fraction": (
            float(n_censored / n_clusters) if n_clusters else 0.0
        ),
        "temporal_cluster_records": records,
    }


TEMPORAL_METRIC_KEYS = (
    "temporal_service_loss_auc_ration_hours",
    "temporal_maximum_service_drop",
    "system_ttr_mean",
    "system_ttr_p95",
    "system_ttr_n_clusters",
    "system_ttr_n_recovered",
    "system_ttr_n_censored",
    "system_ttr_censored_fraction",
)
