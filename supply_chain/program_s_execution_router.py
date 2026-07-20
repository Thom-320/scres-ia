"""Fail-closed execution routing for the corrected Program S1b cascade."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class ExecutionRoute:
    mask: str
    engine: str
    reason: str
    transducer_certificate_required: bool


def route_risk_mask(
    *,
    mask: str,
    risks: Sequence[str],
    r14_probability_multiplier: float,
    certified_masks: Mapping[str, bool],
    r14_action_dependence_certificate: bool = False,
) -> ExecutionRoute:
    risk_set = set(map(str, risks))
    if "R14" in risk_set and (
        abs(float(r14_probability_multiplier) - 1.0) > 1e-12
        or not r14_action_dependence_certificate
    ):
        return ExecutionRoute(
            str(mask),
            "direct_simpy",
            "R14 action-dependent production/quality timing lacks an independent extreme-calendar certificate",
            True,
        )
    if not bool(certified_masks.get(str(mask), False)):
        return ExecutionRoute(
            str(mask),
            "direct_simpy",
            "mask has no current transducer-versus-direct-SimPy exactness certificate",
            True,
        )
    return ExecutionRoute(
        str(mask),
        "certified_transducer",
        "current mask certificate passes and no uncertified action-dependent R14 modification is active",
        False,
    )

