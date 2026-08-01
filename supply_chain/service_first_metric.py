"""Service-first endpoint for SCRES experiments.

The order-level Excel ReT variants are useful for source continuity, but the
abandonment audit showed that none of them is safe as a stand-alone objective.
This module therefore exposes a deliberately lexicographic endpoint instead of
inventing weights between service and resilience:

1. a policy with no lost order beats one with any lost order;
2. among policies tied on abandonment, higher flow fill wins;
3. among those tied on fill, lower final backorder wins;
4. only then does clipped visible ReT break a remaining tie.

The tuple is an estimand, not a scalar reward. Callers must retain its four
components and may not collapse them with an unregistered weighted sum.
"""
from __future__ import annotations

from typing import Any, Mapping


SERVICE_FIRST_METRIC_ID = "service_first_resilience_v1"
SERVICE_FIRST_COMPONENTS = (
    "no_lost_orders",
    "flow_fill_rate",
    "negative_backorder_qty_final",
    "ret_excel_visible_clipped_0_1",
)


def service_first_key(panel: Mapping[str, Any]) -> tuple[float, float, float, float]:
    """Return the frozen lexicographic key for one completed episode.

    The first component is intentionally binary. This makes an abandonment
    policy lose to a no-abandonment policy even when its visible ReT is higher.
    The remaining components preserve the operational ordering without a
    researcher-chosen exchange rate.
    """
    lost = float(panel.get("lost_orders", 0.0) or 0.0)
    fill = float(panel.get("flow_fill_rate", panel.get("fill_rate", 0.0)) or 0.0)
    backorders = float(panel.get("backorder_qty_final", 0.0) or 0.0)
    ret = float(panel.get("ret_excel_visible_clipped_0_1", 0.0) or 0.0)
    return (
        float(lost <= 0.0),
        fill,
        -backorders,
        ret,
    )


def service_first_components(panel: Mapping[str, Any]) -> dict[str, float]:
    """Return named, JSON-safe components without hiding the ordering."""
    key = service_first_key(panel)
    return dict(zip(SERVICE_FIRST_COMPONENTS, key, strict=True))


def service_first_better(
    candidate: Mapping[str, Any], comparator: Mapping[str, Any]
) -> bool:
    """Whether ``candidate`` wins under the frozen lexicographic endpoint."""
    return service_first_key(candidate) > service_first_key(comparator)


# --------------------------------------------------------------------------------------------
# v2 -- successor. `v1` is frozen and untouched; this lives beside it.
#
# The audit (docs/AUDITORIA_SERVICE_FIRST_METRIC_2026-08-01.md) found `v1`'s first component
# measures the wrong quantity. `BACKORDER_QUEUE_CAP = 60`, and an order is only labelled `lost`
# when the backlog queue OVERFLOWS that cap -- so `lost_orders` is a proxy for queue overflow,
# not for abandonment. Measured: four of five allocation splits sit pinned at exactly 60 orders
# that are neither served nor flagged lost. A policy that keeps its queue at 60 abandons up to 60
# units indefinitely and records zero losses, passing v1's gate perfectly.
#
# The first repair I considered -- unserved QUANTITY share -- turns out to be `1 - flow_fill_rate`
# exactly, so it would have collapsed components 1 and 2 into one. Recording that here because it
# is the obvious fix and it is wrong.
#
# What actually distinguishes ABANDONMENT from merely low fill is where the shortfall lands:
# abandonment concentrates it on one claimant. So the leading component is the WORST claimant's
# fill, which is continuous, cannot be gamed by the queue cap, and is not implied by aggregate
# fill. With a single claimant it degenerates to aggregate fill, which is correct -- abandoning a
# claimant is undefined when there is only one.
SERVICE_FIRST_V2_METRIC_ID = "service_first_resilience_v2"
SERVICE_FIRST_V2_COMPONENTS = (
    "worst_claimant_fill",
    "flow_fill_rate",
    "negative_backorder_qty_final",
    "ret_excel_visible_clipped_0_1",
)


def claimant_fills(sim: Any) -> dict[str, float]:
    """Delivered/demanded per claimant. Empty when the model has no claimant partition."""
    demanded = getattr(sim, "cssu_demanded", None)
    delivered = getattr(sim, "cssu_delivered", None)
    if not isinstance(demanded, Mapping) or not isinstance(delivered, Mapping):
        return {}
    return {
        name: (float(delivered.get(name, 0.0)) / float(value)) if float(value) > 0 else 1.0
        for name, value in demanded.items()
    }


def service_first_key_v2(
    panel: Mapping[str, Any], claimant_fill: Mapping[str, float] | None = None
) -> tuple[float, float, float, float]:
    """The successor key. Leading component is the worst claimant's fill, not a loss flag."""
    fill = float(panel.get("flow_fill_rate", panel.get("fill_rate", 0.0)) or 0.0)
    fills = dict(claimant_fill or {})
    worst = min(fills.values()) if fills else fill
    backorders = float(panel.get("backorder_qty_final", 0.0) or 0.0)
    ret = float(panel.get("ret_excel_visible_clipped_0_1", 0.0) or 0.0)
    return (worst, fill, -backorders, ret)


def service_first_v2_components(
    panel: Mapping[str, Any], claimant_fill: Mapping[str, float] | None = None
) -> dict[str, float]:
    return dict(zip(SERVICE_FIRST_V2_COMPONENTS,
                    service_first_key_v2(panel, claimant_fill), strict=True))


def service_first_v2_better(
    candidate: Mapping[str, Any],
    comparator: Mapping[str, Any],
    candidate_fill: Mapping[str, float] | None = None,
    comparator_fill: Mapping[str, float] | None = None,
) -> bool:
    return service_first_key_v2(candidate, candidate_fill) > service_first_key_v2(
        comparator, comparator_fill
    )
