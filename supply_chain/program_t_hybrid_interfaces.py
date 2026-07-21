"""Narrow interfaces for any prospectively authorized Program T hybrid."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from supply_chain.paper2_controller import ControllerDecision


class BeliefProvider(Protocol):
    def reset(self) -> None: ...
    def update(self, observation: Mapping[str, Any]) -> None: ...
    def predictive_distribution(self) -> Mapping[str, Any]: ...


class ResidualDemandModel(Protocol):
    def correction(self, history: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]: ...
    def in_support(self, history: Sequence[Mapping[str, Any]]) -> bool: ...


class ConstrainedPlanner(Protocol):
    def select_action(
        self,
        *,
        observable_state: Mapping[str, Any],
        review_rights_remaining: int,
        online_budget_ms: float,
    ) -> ControllerDecision: ...


class FallbackPlanner(ConstrainedPlanner, Protocol):
    """A fallback is valid only when every returned action is feasible."""


@dataclass
class CampaignState:
    """Keep physical carry-over separate from decision knowledge."""

    physical_state: dict[str, Any] = field(default_factory=dict)
    knowledge_state: dict[str, Any] = field(default_factory=dict)

    def reset_physical(self, initial_state: Mapping[str, Any]) -> None:
        self.physical_state = dict(initial_state)

    def reset_knowledge(self, initial_state: Mapping[str, Any] | None = None) -> None:
        self.knowledge_state = {} if initial_state is None else dict(initial_state)
