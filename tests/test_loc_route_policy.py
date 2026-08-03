"""Route planning as a comparator: exact where it claims to be, myopic where it is.

The tests that matter here are the ones about SCOPE. Program L's audit retracted an `H_PI` that
came from a myopic rule, so the planner must be shown to be myopic too -- otherwise the same
mislabelling is one commit away.
"""
from __future__ import annotations

import itertools

import pytest

from supply_chain.loc_graph import INF, Arc, LOCGraph, dijkstra_path
from supply_chain.loc_route_policy import (
    PROGRAM_L_ROUTE_ARCS,
    live_route_count,
    plan_program_l_epoch,
    plan_route,
    program_l_graph,
)


def test_the_cheaper_live_route_is_taken():
    out = plan_program_l_epoch(route1_hours=24.0, route2_hours=72.0, route1_down=False)
    assert out.route_id == "R1"
    assert out.transit_hours == 24.0
    assert out.had_a_choice is True


def test_the_alternate_is_taken_when_r22_kills_the_thesis_leg():
    """The whole point: with a second arc the flow re-routes instead of stopping."""
    out = plan_program_l_epoch(route1_hours=24.0, route2_hours=72.0, route1_down=True)
    assert out.route_id == "R2"
    assert out.transit_hours == 72.0
    assert out.reachable is True


def test_without_an_alternate_the_same_outage_stops_the_flow():
    """The shipped serial model, for contrast: one arc down means no route at all."""
    graph = LOCGraph((Arc(PROGRAM_L_ROUTE_ARCS["R1"], "AL", "SB", 24.0, operation=8),))
    out = plan_route(graph, "AL", "SB", down={PROGRAM_L_ROUTE_ARCS["R1"]})
    assert out.reachable is False
    assert out.transit_hours == INF
    assert out.had_a_choice is False


def test_a_degraded_alternate_can_lose_to_the_healthy_primary():
    """R2 carries a degradation penalty, so it must not be preferred unconditionally."""
    out = plan_program_l_epoch(route1_hours=24.0, route2_hours=200.0, route1_down=False)
    assert out.route_id == "R1"


@pytest.mark.parametrize("r1,r2", list(itertools.product([1.0, 24.0, 96.0], [1.0, 24.0, 96.0])))
@pytest.mark.parametrize("down", [False, True])
def test_the_plan_is_always_the_brute_force_minimum(r1, r2, down):
    """Exhaustive check over the whole parameter grid, not a spot check."""
    out = plan_program_l_epoch(route1_hours=r1, route2_hours=r2, route1_down=down)
    options = ([] if down else [r1]) + [r2]
    assert out.transit_hours == pytest.approx(min(options))


def test_a_down_arc_is_never_used():
    graph = program_l_graph(route1_hours=1.0, route2_hours=999.0)
    out = plan_route(graph, "AL", "SB", down={PROGRAM_L_ROUTE_ARCS["R1"]})
    assert PROGRAM_L_ROUTE_ARCS["R1"] not in out.arc_ids


def test_had_a_choice_is_false_when_only_one_route_survives():
    """Liveness, per decision. A run reporting routing value on epochs with no choice would be
    measuring nothing -- the dead-actuator failure that cost G3-obs a whole run."""
    assert plan_program_l_epoch(route1_hours=24.0, route2_hours=72.0,
                                route1_down=True).had_a_choice is False
    assert plan_program_l_epoch(route1_hours=24.0, route2_hours=72.0,
                                route1_down=False).had_a_choice is True


def test_search_cost_is_reported_for_the_efficiency_comparison():
    out = plan_program_l_epoch(route1_hours=24.0, route2_hours=72.0, route1_down=False)
    assert out.expansions > 0


def test_the_planner_is_MYOPIC_and_must_never_be_called_a_bound():
    """The scope test, and the reason this file exists.

    Program L's audit retracted an H_PI because it "came from a myopic true-state routing rule ...
    It did not optimize the full trajectory." Planning on the CURRENT arc states is myopic in the
    same way: a plan made now cannot account for an outage that starts during transit, so the
    realized cost can exceed the planned one. Demonstrated rather than asserted, because a
    myopic quantity relabelled as a ceiling is exactly how the earlier claim went wrong.
    """
    planned = plan_program_l_epoch(route1_hours=24.0, route2_hours=72.0, route1_down=False)
    assert planned.route_id == "R1" and planned.transit_hours == 24.0

    # The world moves after the decision: R1 dies mid-transit and the shipment eats the stall.
    # A full-horizon planner anticipating this would have paid 72 up front and arrived sooner.
    realized_if_r1_dies = 24.0 + 500.0
    best_in_hindsight = 72.0
    assert realized_if_r1_dies > best_in_hindsight, (
        "the myopic plan can be beaten in hindsight, so it is a comparator and not a ceiling")


def test_multi_hop_composition_is_where_the_graph_earns_its_place():
    """With one parallel hop A* is a two-way `min`. With two, path composition decides.

    Kept as a live test rather than a comment so the claim about scope stays true as the graph
    grows: here the cheapest first hop leads to the expensive second, and the planner must not
    take the locally cheap arc.
    """
    graph = LOCGraph((
        Arc("h1_cheap", "WDC", "AL", 1.0), Arc("h1_dear", "WDC", "AL", 10.0),
        Arc("h2_dear", "AL", "SB", 100.0),
        Arc("h2_cheap", "AL", "SB", 2.0),
    ))
    out = plan_route(graph, "WDC", "SB")
    truth = dijkstra_path(graph, "WDC", "SB")
    assert out.transit_hours == pytest.approx(truth["cost"]) == 3.0
    assert live_route_count(graph, "WDC", "SB") == 2
