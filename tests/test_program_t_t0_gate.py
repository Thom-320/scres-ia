from __future__ import annotations

from supply_chain.program_t_scenario_mpc import (
    ReTAlignedScenarioMPC,
    RolloutOutcome,
    ScenarioMPCConfig,
    t0_comparator_grid,
)
from supply_chain.program_t_t0_gate import (
    ComparatorPoint,
    adjudicate_t0_residual,
    quality_time_frontier,
)


class ToyRollout:
    def scenarios(self, *, history, observable_state, limit):
        del history, observable_state
        return tuple(range(min(2, limit)))

    def rollout(self, *, observable_state, actions, scenario):
        del observable_state
        return RolloutOutcome(
            ret=1.0 - 0.1 * abs(actions[0] - scenario),
            worst_product_fill=0.9,
            lost_demand=0.0,
            resource_use=(1.0,),
        )


def test_comparator_grid_is_complete() -> None:
    grid = t0_comparator_grid()
    assert len(grid) == 20
    assert {row.horizon for row in grid} == {1, 3, 4, 6, 8}


def test_ret_aligned_mpc_uses_deployable_interface() -> None:
    controller = ReTAlignedScenarioMPC(
        rollout_model=ToyRollout(),
        config=ScenarioMPCConfig(horizon=1, mode="scenario"),
    )
    controller.reset(episode_id="toy")
    controller.update_history((0.1, 0.2))
    decision = controller.select_action(
        observable_state={}, review_rights_remaining=4, online_budget_ms=1000
    )
    assert decision.feasible
    assert decision.action.mix in (0, 1, 2, 3)


def test_frontier_removes_dominated_points() -> None:
    points = (
        ComparatorPoint("a", 0.8, 2.0, True),
        ComparatorPoint("b", 0.9, 2.0, True),
        ComparatorPoint("c", 0.95, 5.0, True),
    )
    assert [row.controller_id for row in quality_time_frontier(points)] == ["b", "c"]


def test_t0_gate_passes_only_jointly() -> None:
    result = adjudicate_t0_residual(
        best_observable_ret=[0.9] * 20,
        reinforced_mpc_ret=[0.8] * 20,
        worst_product_delta=[0.0] * 20,
        lost_order_delta=[0.0] * 20,
        resource_delta=[0.0] * 20,
    )
    assert str(result["status"]).startswith("PASS_T0")
    assert result["hybrid_confirmation_authorized"] is False

