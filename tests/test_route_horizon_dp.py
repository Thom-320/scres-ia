"""The DP must be provably optimal, not plausibly optimal.

Program L's H_PI was retracted for being a myopic rule wearing a bound's name. The defence here is
brute force: on horizons small enough to enumerate every schedule, the DP must equal the true
minimum. If that ever fails the DP is not a bound and must not be reported as one.
"""
from __future__ import annotations

import itertools
import random

import pytest

from supply_chain.route_horizon_dp import (
    ACTIONS,
    HOLD,
    INF,
    ROUTE_1,
    ROUTE_2,
    RouteTape,
    evaluate_schedule,
    myopic_schedule,
    solve,
)


def tape(n=6, down=None, degraded=None, demand=None, **kw) -> RouteTape:
    return RouteTape(
        route1_down=tuple(down or [False] * n),
        route2_degraded=tuple(degraded or [False] * n),
        demand_per_epoch=tuple(demand or [1000.0] * n), **kw)


def brute_force(t: RouteTape) -> float:
    return min(evaluate_schedule(t, s)
               for s in itertools.product(ACTIONS, repeat=t.horizon))


@pytest.mark.parametrize("trial", range(60))
def test_the_dp_equals_exhaustive_enumeration(trial):
    """THE test. A DP that merely looks optimal bounds nothing."""
    rng = random.Random(20260803 + trial)
    n = rng.randint(3, 7)
    t = tape(n,
             down=[rng.random() < 0.35 for _ in range(n)],
             degraded=[rng.random() < 0.3 for _ in range(n)],
             demand=[float(rng.randint(0, 3000)) for _ in range(n)],
             convoy_capacity=float(rng.choice([500.0, 2000.0, 5000.0])))
    assert solve(t)["optimal_cost"] == pytest.approx(brute_force(t)), (
        "the DP is not returning the true minimum, so it is not a bound")


@pytest.mark.parametrize("trial", range(40))
def test_the_dp_is_never_worse_than_the_myopic_rule(trial):
    """The comparison the audit asked for: full-horizon versus the class that was retracted."""
    rng = random.Random(777 + trial)
    n = rng.randint(4, 9)
    t = tape(n,
             down=[rng.random() < 0.4 for _ in range(n)],
             degraded=[rng.random() < 0.35 for _ in range(n)],
             demand=[float(rng.randint(0, 2500)) for _ in range(n)])
    assert solve(t)["optimal_cost"] <= evaluate_schedule(t, myopic_schedule(t)) + 1e-9


def test_there_exists_a_tape_where_waiting_beats_dispatching_now():
    """If the myopic rule were always optimal the DP would be pointless. It is not: committing the
    convoy to the slow alternate can lock it out of a fast departure one epoch later."""
    t = tape(6, down=[True, False, False, False, False, False],
             degraded=[False] * 6, demand=[0.0, 3000.0, 0.0, 0.0, 0.0, 0.0],
             route2_base_hours=60.0, route2_penalty_hours=0.0)
    assert solve(t)["optimal_cost"] < evaluate_schedule(t, myopic_schedule(t)) - 1e-9, (
        "no gap between myopic and optimal on this tape; rebuild it or the DP proves nothing")


def test_a_dead_primary_forces_the_alternate():
    t = tape(4, down=[True] * 4)
    assert solve(t)["optimal_cost"] == pytest.approx(brute_force(t))
    assert all(a != ROUTE_1 for a in myopic_schedule(t))


def test_route1_is_infeasible_while_down_and_feasible_otherwise():
    t = tape(2, down=[True, False])
    assert t.transit_hours(ROUTE_1, 0) == INF
    assert t.transit_hours(ROUTE_1, 1) == pytest.approx(48.0)


def test_the_degraded_alternate_costs_its_declared_penalty():
    t = tape(2, degraded=[False, True], route2_base_hours=36.0, route2_penalty_hours=24.0)
    assert t.transit_hours(ROUTE_2, 0) == pytest.approx(72.0)
    assert t.transit_hours(ROUTE_2, 1) == pytest.approx(120.0)


def test_holding_forever_is_always_feasible_and_never_optimal_with_demand():
    t = tape(5, demand=[1000.0] * 5)
    assert solve(t)["optimal_cost"] < evaluate_schedule(t, [HOLD] * 5)


def test_zero_demand_makes_every_schedule_free():
    t = tape(4, demand=[0.0] * 4)
    assert solve(t)["optimal_cost"] == pytest.approx(0.0)


def test_the_scope_is_declared_in_the_result():
    """The surrogate limit must travel WITH the number, not live in a document nobody opens."""
    out = solve(tape(3))
    assert out["scope"] == "ADDITIVE_SERVICE_LOSS_SURROGATE_NOT_CANONICAL_RET"
    assert out["expansions"] > 0


def test_search_cost_is_reported_so_the_dp_is_comparable_to_astar():
    small, large = solve(tape(4)), solve(tape(9))
    assert large["expansions"] > small["expansions"], (
        "DP cost must grow with the horizon, or it is not being measured")


@pytest.mark.parametrize("bad", [
    lambda: RouteTape(route1_down=(), route2_degraded=(), demand_per_epoch=()),
    lambda: RouteTape(route1_down=(False,), route2_degraded=(), demand_per_epoch=(1.0,)),
    lambda: RouteTape(route1_down=(False,), route2_degraded=(False,), demand_per_epoch=(1.0,),
                      convoy_capacity=-1.0),
])
def test_malformed_tapes_are_rejected(bad):
    with pytest.raises(ValueError):
        bad()
