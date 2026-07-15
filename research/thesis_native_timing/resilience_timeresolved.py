"""Time-resolved resilience metric (co-primary) for the thesis-native timing ablation.

Motivation
----------
The canonical thesis metric ``ret_excel = 1 - (Bt+Ut)/j`` is an order-count service level
collapsed to a single episode mean: it records *whether* an order was eventually on time,
never *how long* it was late. Two policies with identical ``ret_excel`` (same on-time count)
but different recovery speeds are scored identically — so the recovery *dynamics*, where
adaptive/preventive control adds value, are invisible.

The literature state of the art is the **resilience triangle** (Bruneau et al. 2003): the
*integral* of performance loss over the disruption->recovery window (robustness x rapidity).
The repo already computes the raw material for it in the canonical panel:
``service_loss_auc_ration_hours = sum_orders max(0, end-(OPTj+LT)) * qty`` — the qty-weighted
late-hours integral (area above the performance curve) — plus ``ttr`` (recovery period).

This module normalizes those into bounded, comparable co-primary scores. It NEVER modifies
``ret_excel``; it is reported alongside it, and ``ret_excel`` remains the primary
(non-inferiority-required) endpoint.

Key invariant (tested): on a disruption-free tape the time-resolved resilience equals ~1 and
tracks ``ret_excel``; on a disrupted tape, equal-``ret_excel`` policies that recover at
different speeds get different ``resilience_triangle_v1``.
"""

from __future__ import annotations

from typing import Any, Mapping

# 4-week reference recovery window (matches TRACK_B_ROLLING_WINDOW_HOURS = 4*168h). A fixed
# reference (not horizon-dependent) keeps the score stable and comparable across policies:
# it reads as "fraction of a full 4-week ration-loss avoided".
DEFAULT_REF_WINDOW_HOURS: float = 4 * 168.0  # 672h


def resilience_timeresolved(
    panel: Mapping[str, float],
    *,
    ref_window_hours: float = DEFAULT_REF_WINDOW_HOURS,
) -> dict[str, float]:
    """Layer time-resolved resilience scores onto a canonical episode panel.

    Parameters
    ----------
    panel : the dict returned by ``compute_episode_metrics`` (must contain
        ``service_loss_auc_ration_hours``, ``demanded_rations``, ``delivered_rations``,
        ``ttr_mean``, ``ret_excel``).
    ref_window_hours : the reference full-loss window used to normalize the loss integral.

    Returns
    -------
    dict with, in addition to the input panel:
      - ``resilience_triangle_v1`` in [0,1]: 1 - (late-hours integral) / (max reference loss).
        This is the SOTA integral resilience; it *falls* as orders stay late longer, so it
        distinguishes fast vs slow recovery at equal ``ret_excel``.
      - ``mean_lateness_hours``: interpretable robustness/rapidity proxy (loss integral per
        delivered ration).
      - ``rapidity_v1`` in (0,1]: normalized recovery speed from ``ttr_mean``.
      - ``resilience_tr_combined`` in [0,1]: geometric mean of triangle x rapidity.
    """
    out = dict(panel)

    service_loss = float(panel.get("service_loss_auc_ration_hours", 0.0))
    demanded = float(panel.get("demanded_rations", 0.0))
    delivered = float(panel.get("delivered_rations", 0.0))
    ttr_mean = float(panel.get("ttr_mean", 0.0))
    ref = max(1e-9, float(ref_window_hours))

    # Integral resilience (resilience triangle), normalized by the maximum reference loss
    # (every demanded ration late for the full reference window). Clamped to [0,1].
    max_reference_loss = max(1e-9, demanded * ref)
    triangle = 1.0 - service_loss / max_reference_loss
    triangle = max(0.0, min(1.0, triangle))

    mean_lateness = service_loss / delivered if delivered > 0 else 0.0

    # Rapidity: fast recovery (small ttr) -> ~1; slow recovery -> ->0. Normalized by ref.
    rapidity = ref / (ref + ttr_mean) if ttr_mean >= 0.0 else 1.0
    rapidity = max(0.0, min(1.0, rapidity))

    combined = (triangle * rapidity) ** 0.5

    out["resilience_triangle_v1"] = float(triangle)
    out["mean_lateness_hours"] = float(mean_lateness)
    out["rapidity_v1"] = float(rapidity)
    out["resilience_tr_combined"] = float(combined)
    out["resilience_ref_window_hours"] = float(ref)
    return out


# Which keys are the time-resolved co-primary endpoints (for the ablation report).
TIMERESOLVED_KEYS: tuple[str, ...] = (
    "resilience_triangle_v1",
    "rapidity_v1",
    "resilience_tr_combined",
    "mean_lateness_hours",
)
