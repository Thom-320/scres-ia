"""Pre-RL policy primitives for DRA-2 finite-convoy control."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


ACTIONS = ("HOLD", "DISPATCH_NOW")
INVENTORY_THRESHOLDS = (1_000.0, 2_500.0, 5_000.0)
MAXIMUM_WAIT_HOURS = (24.0, 48.0, 72.0)


@dataclass(frozen=True)
class ConvoyThresholdPolicy:
    inventory_threshold: float
    maximum_wait_hours: float

    def __post_init__(self) -> None:
        if float(self.inventory_threshold) not in INVENTORY_THRESHOLDS:
            raise ValueError(
                f"inventory_threshold must be one of {INVENTORY_THRESHOLDS}"
            )
        if float(self.maximum_wait_hours) not in MAXIMUM_WAIT_HOURS:
            raise ValueError(
                f"maximum_wait_hours must be one of {MAXIMUM_WAIT_HOURS}"
            )

    @property
    def policy_id(self) -> str:
        return f"threshold_{int(self.inventory_threshold)}__wait_{int(self.maximum_wait_hours)}h"

    def action(self, observation: Mapping[str, float]) -> str:
        feasible = bool(
            observation["convoy_available"] > 0.5
            and observation["op8_route_up"] > 0.5
            and observation["op7_staged_inventory"] > 0.0
        )
        if not feasible:
            return "HOLD"
        should_dispatch = bool(
            observation["op7_staged_inventory"] >= self.inventory_threshold
            or observation["staging_age"] >= self.maximum_wait_hours
        )
        return "DISPATCH_NOW" if should_dispatch else "HOLD"


def static_policies() -> tuple[ConvoyThresholdPolicy, ...]:
    return tuple(
        ConvoyThresholdPolicy(threshold, wait)
        for threshold in INVENTORY_THRESHOLDS
        for wait in MAXIMUM_WAIT_HOURS
    )
