"""Recovery estimands for the Garrido-v0 repeated-campaign closure.

The historical ``system_ttr_mean`` endpoint becomes zero when every recovery
cluster is censored.  Zero then means "no recovery was observed", which is the
opposite of a fast recovery.  This module defines a bounded estimand instead:

``restricted_ttr = min(time_to_restoration, tau)``.

If a shock produces no service degradation relative to its paired placebo, the
restricted TTR is zero (the system absorbed the shock).  If degradation occurs
and restoration is not observed by ``tau``, the value is ``tau``.  Thus every
cell has a meaningful value and censoring can never manufacture a fast arm.

The eight contexts below are isolated versions of Garrido's R11--R24 risks.
Durations and magnitudes were frozen at rounded medians of the already-opened
Step-3 development tapes on 2026-08-06.  They are an instrument-development
grid, not thesis parameters and not confirmatory evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .config import HOURS_PER_WEEK


EVENT_ONSET_HOURS = 8.0 * HOURS_PER_WEEK
RECOVERY_WINDOW_HOURS = 8.0 * HOURS_PER_WEEK
RECOVERY_FRACTION = 0.95
RECOVERY_CONSECUTIVE_DAYS = 7
SERVICE_DROP_TOLERANCE = 0.01


@dataclass(frozen=True)
class RecoveryContext:
    risk_id: str
    duration_hours: float
    affected_ops: tuple[int, ...]
    magnitude: float = 1.0
    unit: str = "incidents"
    repetitions: int = 1
    spacing_hours: float = 0.0


# Rounded empirical medians from the already-open Step-3 development tapes.
# Frequent short events (R11/R22) are represented as one compound campaign
# cluster; the spacing is below the one-week clustering gap.
RECOVERY_CONTEXTS: tuple[RecoveryContext, ...] = (
    RecoveryContext("R11", 1.5, (5,), repetitions=8, spacing_hours=42.0),
    RecoveryContext("R12", 672.0, (1,), magnitude=4.0, unit="delayed_contracts"),
    RecoveryContext("R13", 120.0, (2,), magnitude=5.0, unit="delayed_deliveries"),
    RecoveryContext("R14", 0.0, (7,), magnitude=197.0, unit="defective_products"),
    RecoveryContext("R21", 249.0, (3, 5, 6, 7, 9)),
    RecoveryContext("R22", 18.0, (12,), repetitions=4, spacing_hours=72.0),
    RecoveryContext("R23", 93.0, (11,)),
    RecoveryContext("R24", 0.0, (13,), magnitude=2495.0, unit="rations"),
)
CONTEXT_BY_ID = {context.risk_id: context for context in RECOVERY_CONTEXTS}
CONTEXT_ORDER = tuple(context.risk_id for context in RECOVERY_CONTEXTS)


def risk_event_rows(
    context: str | RecoveryContext,
    *,
    onset_hours: float = EVENT_ONSET_HOURS,
) -> list[dict[str, Any]]:
    """Serialize one isolated campaign shock (or compound short-event cluster)."""
    spec = CONTEXT_BY_ID[context] if isinstance(context, str) else context
    rows: list[dict[str, Any]] = []
    for occurrence in range(spec.repetitions):
        start = float(onset_hours + occurrence * spec.spacing_hours)
        duration = float(spec.duration_hours)
        rows.append(
            {
                "risk_id": spec.risk_id,
                "start_time": start,
                "end_time": start + duration,
                "duration": duration,
                "affected_ops": list(spec.affected_ops),
                "description": "garrido_v0_recovery_gate_empirical_median",
                "magnitude": float(spec.magnitude),
                "unit": spec.unit,
                "affected_cssu": None,
            }
        )
    return rows


def placebo_event_rows(*, onset_hours: float = EVENT_ONSET_HOURS) -> list[dict[str, Any]]:
    """A zero-physics marker producing the same temporal measurement window."""
    return [
        {
            "risk_id": "PLACEBO",
            "start_time": float(onset_hours),
            "end_time": float(onset_hours),
            "duration": 0.0,
            "affected_ops": [],
            "description": "temporal-window marker; no physical intervention",
            "magnitude": 0.0,
            "unit": "none",
            "affected_cssu": None,
        }
    ]


def _single_record(panel: Mapping[str, Any]) -> Mapping[str, Any]:
    records = list(panel.get("temporal_cluster_records", ()))
    if len(records) != 1:
        raise ValueError(f"expected exactly one risk cluster, observed {len(records)}")
    return records[0]


def restricted_recovery_summary(
    risk_panel: Mapping[str, Any],
    placebo_panel: Mapping[str, Any],
    *,
    tau_hours: float = RECOVERY_WINDOW_HOURS,
    service_drop_tolerance: float = SERVICE_DROP_TOLERANCE,
) -> dict[str, float | bool]:
    """Return a bounded recovery endpoint with a paired no-shock impact check.

    ``absorbed`` is decided from the *incremental* temporal service-loss area and
    maximum service drop relative to a placebo run on the same demand tape and
    posture.  Routine backlog can therefore not masquerade as shock impact.
    """
    if tau_hours <= 0.0:
        raise ValueError("tau_hours must be positive")
    risk = _single_record(risk_panel)
    placebo = _single_record(placebo_panel)
    excess_auc = max(
        0.0,
        float(risk["service_loss_auc_ration_hours"])
        - float(placebo["service_loss_auc_ration_hours"]),
    )
    excess_drop = max(
        0.0,
        float(risk["maximum_service_drop"])
        - float(placebo["maximum_service_drop"]),
    )
    impacted = bool(excess_auc > 1e-9 or excess_drop > service_drop_tolerance)
    raw_ttr = risk.get("system_ttr_hours")
    observed = raw_ttr is not None
    if not impacted:
        restricted = 0.0
        recovered = True
        censored = False
    elif observed:
        restricted = min(float(raw_ttr), float(tau_hours))
        recovered = bool(float(raw_ttr) <= tau_hours)
        censored = not recovered
    else:
        restricted = float(tau_hours)
        recovered = False
        censored = True
    return {
        "restricted_ttr_hours": float(restricted),
        "raw_ttr_hours": float(raw_ttr) if raw_ttr is not None else float("nan"),
        "impacted": impacted,
        "absorbed": not impacted,
        "recovered_within_tau": recovered,
        "right_censored_at_tau": censored,
        "excess_service_loss_auc_ration_hours": float(excess_auc),
        "excess_maximum_service_drop": float(excess_drop),
    }


def recovery_utility(
    recovery: Mapping[str, Any],
    *,
    demanded_rations: float,
    flow_fill_rate: float,
    tau_hours: float = RECOVERY_WINDOW_HOURS,
) -> float:
    """A recovery-first scalar for search, with bounded tie breakers.

    One day of restricted-TTR improvement is larger than the combined maximum
    contribution of both tie breakers.  The search therefore cannot trade a
    slower restoration for a prettier aggregate service score.
    """
    restricted = float(recovery["restricted_ttr_hours"])
    ttr_score = 1.0 - np.clip(restricted / tau_hours, 0.0, 1.0)
    auc_denominator = max(1.0, float(demanded_rations) * float(tau_hours))
    auc_score = 1.0 - np.clip(
        float(recovery["excess_service_loss_auc_ration_hours"]) / auc_denominator,
        0.0,
        1.0,
    )
    fill = np.clip(float(flow_fill_rate), 0.0, 1.0)
    return float(ttr_score + 1e-3 * auc_score + 1e-6 * fill)


def context_descriptor(context: str) -> np.ndarray:
    """Source-safe descriptor known before a DES run; no outcome statistics."""
    spec = CONTEXT_BY_ID[context]
    one_hot = [1.0 if name == context else 0.0 for name in CONTEXT_ORDER]
    op_mask = [1.0 if op in spec.affected_ops else 0.0 for op in range(1, 14)]
    return np.asarray(
        one_hot
        + op_mask
        + [
            min(1.0, spec.duration_hours / RECOVERY_WINDOW_HOURS),
            min(1.0, spec.magnitude / 2600.0),
            min(1.0, spec.repetitions / 8.0),
        ],
        dtype=float,
    )


__all__ = [
    "CONTEXT_BY_ID",
    "CONTEXT_ORDER",
    "EVENT_ONSET_HOURS",
    "RECOVERY_CONTEXTS",
    "RECOVERY_CONSECUTIVE_DAYS",
    "RECOVERY_FRACTION",
    "RECOVERY_WINDOW_HOURS",
    "context_descriptor",
    "placebo_event_rows",
    "recovery_utility",
    "restricted_recovery_summary",
    "risk_event_rows",
]
