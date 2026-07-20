"""ReT-aligned finite-horizon scenario MPC used by the T0 comparator gate."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
import time
from typing import Any, Literal, Mapping, Protocol, Sequence

import numpy as np

from supply_chain.paper2_controller import ControllerAction, ControllerDecision


Mode = Literal["nominal", "scenario", "robust", "constraint_aware"]


@dataclass(frozen=True)
class RolloutOutcome:
    ret: float
    worst_product_fill: float
    lost_demand: float
    resource_use: tuple[float, ...]


class ScenarioRolloutModel(Protocol):
    def scenarios(
        self,
        *,
        history: Sequence[Sequence[float]],
        observable_state: Mapping[str, Any],
        limit: int,
    ) -> Sequence[Any]: ...

    def rollout(
        self,
        *,
        observable_state: Mapping[str, Any],
        actions: Sequence[int],
        scenario: Any,
    ) -> RolloutOutcome: ...


@dataclass(frozen=True)
class ScenarioMPCConfig:
    horizon: int
    mode: Mode
    scenario_limit: int = 64
    worst_product_floor: float = 0.0
    lost_demand_ceiling: float = 0.0
    cvar_alpha: float = 0.10

    def __post_init__(self) -> None:
        if self.horizon not in (1, 3, 4, 6, 8):
            raise ValueError("T0 horizon must be one of 1,3,4,6,8")
        if self.scenario_limit <= 0 or not 0 < self.cvar_alpha <= 1:
            raise ValueError("invalid scenario configuration")


class ReTAlignedScenarioMPC:
    """Enumerate a bounded action tree and score canonical ReT rollouts.

    The rollout model owns the belief/scenario generator. This controller never
    sees a true regime, tape id, future realized event, or answer matrix.
    """

    def __init__(self, *, rollout_model: ScenarioRolloutModel, config: ScenarioMPCConfig):
        self.rollout_model = rollout_model
        self.config = config
        self.controller_id = f"ret_mpc_{config.mode}_h{config.horizon}"
        self._history: list[tuple[float, ...]] = []

    def reset(self, *, episode_id: str) -> None:
        del episode_id
        self._history = []

    def update_history(self, observation: Sequence[float]) -> None:
        self._history.append(tuple(map(float, observation)))

    @staticmethod
    def _lower_tail(values: np.ndarray, alpha: float) -> float:
        count = max(1, int(math.ceil(alpha * len(values))))
        return float(np.mean(np.sort(values)[:count]))

    def _objective(self, outcomes: Sequence[RolloutOutcome]) -> tuple[float, ...]:
        ret = np.asarray([row.ret for row in outcomes], dtype=float)
        worst = np.asarray([row.worst_product_fill for row in outcomes], dtype=float)
        lost = np.asarray([row.lost_demand for row in outcomes], dtype=float)
        feasible = bool(
            np.min(worst) >= self.config.worst_product_floor
            and np.max(lost) <= self.config.lost_demand_ceiling
        )
        if self.config.mode == "nominal":
            value = float(ret[0])
        elif self.config.mode == "scenario":
            value = float(np.mean(ret))
        elif self.config.mode == "robust":
            value = self._lower_tail(ret, self.config.cvar_alpha)
        else:
            value = float(np.mean(ret)) if feasible else float("-inf")
        # Maximize feasibility, value, tail, worst-product service; then use the
        # caller's deterministic sequence tie-break.
        return (float(feasible), value, self._lower_tail(ret, self.config.cvar_alpha), float(np.min(worst)))

    def select_action(
        self,
        *,
        observable_state: Mapping[str, Any],
        review_rights_remaining: int,
        online_budget_ms: float,
    ) -> ControllerDecision:
        del review_rights_remaining
        start = time.perf_counter()
        scenarios = tuple(
            self.rollout_model.scenarios(
                history=self._history,
                observable_state=observable_state,
                limit=self.config.scenario_limit,
            )
        )
        if not scenarios:
            raise RuntimeError("scenario generator returned no deployable scenarios")
        rows: list[tuple[tuple[float, ...], tuple[int, ...]]] = []
        for actions in product(range(4), repeat=self.config.horizon):
            outcomes = tuple(
                self.rollout_model.rollout(
                    observable_state=observable_state,
                    actions=actions,
                    scenario=scenario,
                )
                for scenario in scenarios
            )
            rows.append((self._objective(outcomes), tuple(actions)))
            if (time.perf_counter() - start) * 1000.0 > online_budget_ms:
                return ControllerDecision(
                    action=ControllerAction(0, 1),
                    controller_id=self.controller_id,
                    mode="fallback",
                    online_ms=(time.perf_counter() - start) * 1000.0,
                    feasible=False,
                    fallback_reason="online_budget_exceeded",
                )
        objective, actions = max(rows, key=lambda row: (row[0], tuple(-x for x in row[1])))
        return ControllerDecision(
            action=ControllerAction(actions[0], 1),
            controller_id=self.controller_id,
            mode=self.config.mode,
            online_ms=(time.perf_counter() - start) * 1000.0,
            feasible=bool(objective[0]),
            diagnostics={"objective": objective, "scenario_count": len(scenarios)},
        )


def t0_comparator_grid() -> tuple[ScenarioMPCConfig, ...]:
    return tuple(
        ScenarioMPCConfig(horizon=horizon, mode=mode)
        for horizon in (1, 3, 4, 6, 8)
        for mode in ("nominal", "scenario", "robust", "constraint_aware")
    )

