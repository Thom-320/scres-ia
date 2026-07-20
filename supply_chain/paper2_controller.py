"""Common, auditable controller boundary for Paper 2.

Controllers receive only observable history/state and return a product-mix
commitment plus the next review dwell.  Physics, scenario generation, and
metric calculation remain outside the controller so information and compute
can be matched across classical, learned, and hybrid arms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence


@dataclass(frozen=True)
class ControllerAction:
    mix: int
    dwell_weeks: int = 1

    def __post_init__(self) -> None:
        if self.mix not in (0, 1, 2, 3):
            raise ValueError("mix must lie in {0,1,2,3}")
        if self.dwell_weeks not in (1, 2, 4):
            raise ValueError("dwell_weeks must lie in {1,2,4}")


@dataclass(frozen=True)
class ControllerDecision:
    action: ControllerAction
    controller_id: str
    mode: str
    online_ms: float
    feasible: bool
    confidence: float | None = None
    fallback_reason: str | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.online_ms < 0:
            raise ValueError("online_ms must be nonnegative")


class Paper2Controller(Protocol):
    controller_id: str

    def reset(self, *, episode_id: str) -> None: ...

    def update_history(self, observation: Sequence[float]) -> None: ...

    def select_action(
        self,
        *,
        observable_state: Mapping[str, Any],
        review_rights_remaining: int,
        online_budget_ms: float,
    ) -> ControllerDecision: ...

