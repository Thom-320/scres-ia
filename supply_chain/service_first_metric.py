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
