from __future__ import annotations

from supply_chain.replenish import central_cell, materialize_tape
from supply_chain.replenish_ret import (
    BUDGET_D0, WEEKS, paced_policy, periodic_calendars, rollout_actions,
    rollout_policy,
)


def test_every_rollout_respects_equal_budget_and_weekly_cap() -> None:
    tape = materialize_tape(6700001, central_cell(), WEEKS)
    result = rollout_actions(tape, (1.5,) * WEEKS)
    assert result.ordered_D0 <= BUDGET_D0
    assert max(result.actions) <= 1.5


def test_policy_uses_only_observation_and_emits_canonical_ret() -> None:
    tape = materialize_tape(6700002, central_cell(), WEEKS)
    result = rollout_policy(tape, paced_policy(1.0, 0.5))
    assert 0.0 <= result.ret_order <= 1.0
    assert 0.0 <= result.ret_quantity <= 1.0
    assert result.attended + result.lost == WEEKS


def test_more_supply_cannot_violate_mass_or_order_count() -> None:
    tape = materialize_tape(6700003, central_cell(), WEEKS)
    low = rollout_actions(tape, (0.0,) * WEEKS)
    high = rollout_actions(tape, (1.5,) * WEEKS)
    assert high.attended >= low.attended
    assert high.remaining_qty <= low.remaining_qty


def test_periodic_frontier_is_nontrivial_and_budget_feasible() -> None:
    rows = periodic_calendars(4)
    assert len(rows) > 1000
    assert all(sum(row) <= BUDGET_D0 + 1e-9 for row in rows)
    assert (1.5, 1.0, 1.5, 1.0, 1.5, 1.0, 1.5, 1.0) in rows
