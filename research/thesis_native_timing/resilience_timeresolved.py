"""Time-resolved resilience metric (co-primary) for the thesis-native timing ablation.

Motivation
----------
The canonical thesis metric ``ret_excel = 1 - (Bt+Ut)/j`` is an order-count service level
collapsed to a single episode mean: it records *whether* an order was eventually on time,
never *how long* it was late. Two policies with identical ``ret_excel`` but different recovery
speeds are scored identically -- so the recovery *dynamics*, where adaptive/preventive control
adds value, are invisible.

The literature state of the art is the **resilience triangle** (Bruneau et al. 2003): the
*integral* of performance loss over the disruption->recovery window (robustness x rapidity).

This module computes a properly **bounded** resilience triangle directly from the order ledger.
Each order's lateness is capped at a fixed reference recovery window ``ref_window_hours`` so that
(a) an order that recovers within the window contributes loss proportional to its recovery time
(rapidity), (b) an order still down after the window contributes exactly one window of loss
(robustness), and (c) the score never depends on the arbitrary episode horizon. It NEVER modifies
``ret_excel``; it is reported alongside it as a co-primary, with ``ret_excel`` the primary
(non-inferiority-required) endpoint.

Key invariant (tested): on a disruption-free tape the triangle equals ~1 and tracks ``ret_excel``;
on a disrupted tape, equal-``ret_excel`` policies that recover at different speeds get different
``resilience_triangle_v1``.
"""

from __future__ import annotations

from typing import Any, Mapping

# 4-week reference recovery window (matches TRACK_B_ROLLING_WINDOW_HOURS = 4*168h). A fixed
# reference keeps the score bounded in [0,1] and horizon-independent.
DEFAULT_REF_WINDOW_HOURS: float = 4 * 168.0  # 672h


def resilience_from_sim(
    sim: Any,
    *,
    treatment_start: float | None = None,
    ref_window_hours: float = DEFAULT_REF_WINDOW_HOURS,
) -> dict[str, float]:
    """Bounded resilience-triangle scores computed directly from a completed sim's orders.

    Mirrors the order-selection of ``compute_episode_metrics`` (post-warmup, not metrics_excluded),
    then integrates *capped* qty-weighted lateness.
    """
    horizon = float(sim.env.now)
    start = float(sim.warmup_time if treatment_start is None else treatment_start)
    ref = max(1e-9, float(ref_window_hours))

    orders = [
        o
        for o in sim.orders
        if not bool(getattr(o, "metrics_excluded", False))
        and float(getattr(o, "OPTj", 0.0)) >= start
    ]

    total_qty = 0.0
    capped_loss = 0.0          # bounded resilience-triangle numerator
    recovery_times: list[float] = []  # lateness among orders that DID recover (for rapidity)
    for o in orders:
        qty = float(getattr(o, "quantity", 0.0) or 0.0)
        total_qty += qty
        opt = float(getattr(o, "OPTj", 0.0) or 0.0)
        lt = float(getattr(o, "LTj", 0.0) or 0.0)
        oat = getattr(o, "OATj", None)
        end = float(oat) if oat is not None else horizon
        lateness = max(0.0, end - (opt + lt))
        capped_loss += min(ref, lateness) * qty
        if oat is not None and lateness > 0.0:
            recovery_times.append(lateness)

    max_reference_loss = max(1e-9, total_qty * ref)
    triangle = 1.0 - capped_loss / max_reference_loss
    triangle = max(0.0, min(1.0, triangle))

    # Rapidity DIAGNOSTIC (soft, never exactly 0): ref/(ref+mean recovered lateness).
    # The triangle already subsumes rapidity (it integrates *capped* lateness, so faster
    # recovery -> less loss -> higher triangle); rapidity_v1 is reported only as a secondary lens.
    if recovery_times:
        mean_recov = sum(recovery_times) / len(recovery_times)
        rapidity = ref / (ref + mean_recov)
    else:
        rapidity = 1.0
    rapidity = max(0.0, min(1.0, rapidity))

    return {
        # HEADLINE co-primary time-resolved endpoint:
        "resilience_triangle_v1": float(triangle),
        # secondary diagnostics:
        "rapidity_v1": float(rapidity),
        "capped_service_loss_ration_hours": float(capped_loss),
        "resilience_ref_window_hours": float(ref),
        "n_recovered_late": float(len(recovery_times)),
    }


def resilience_timeresolved(
    panel: Mapping[str, float],
    sim: Any | None = None,
    *,
    treatment_start: float | None = None,
    ref_window_hours: float = DEFAULT_REF_WINDOW_HOURS,
) -> dict[str, float]:
    """Layer bounded time-resolved resilience onto a canonical episode ``panel``.

    If ``sim`` is provided, the triangle is computed exactly (capped, from the order ledger).
    Otherwise it falls back to the panel's uncapped ``service_loss_auc`` (horizon-sensitive) --
    only for quick, disruption-free sanity use.
    """
    out = dict(panel)
    if sim is not None:
        out.update(resilience_from_sim(sim, treatment_start=treatment_start, ref_window_hours=ref_window_hours))
        return out
    # panel-only fallback (uncapped; use only when no unfulfilled orders exist)
    service_loss = float(panel.get("service_loss_auc_ration_hours", 0.0))
    demanded = float(panel.get("demanded_rations", 0.0))
    ref = max(1e-9, float(ref_window_hours))
    triangle = max(0.0, min(1.0, 1.0 - service_loss / max(1e-9, demanded * ref)))
    out["resilience_triangle_v1"] = float(triangle)
    out["resilience_ref_window_hours"] = float(ref)
    return out


TIMERESOLVED_KEYS: tuple[str, ...] = (
    "resilience_triangle_v1",
    "rapidity_v1",
    "capped_service_loss_ration_hours",
)
