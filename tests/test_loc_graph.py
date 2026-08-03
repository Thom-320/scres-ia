"""The routing graph and A*, with admissibility CHECKED rather than argued.

An inadmissible heuristic makes A* return a sub-optimal path while still looking like it worked,
which is the exact failure shape this project keeps hitting: code that does something other than
what its name promises. So the heuristic is tested against exhaustive Dijkstra on randomised
graphs, and the shipped topology is tested to have no routing decision at all.
"""
from __future__ import annotations

import random

import pytest

from supply_chain.loc_graph import (
    INF,
    Arc,
    LOCGraph,
    admissible_heuristic,
    arc_down_tape,
    astar_path,
    baseline_graph,
    dijkstra_path,
    hops_to,
    routing_is_live,
    with_alternate_route,
)


def test_shipped_topology_has_no_routing_decision():
    """The null arm: one arc per hop, so re-routing is moot. This is the thesis's model.

    Note the target is a CSSU, not THEATRE. Reaching THEATRE via CSSU_A instead of CSSU_B is not
    an alternative route -- they are different destinations serving different demand -- and
    counting it as one would have invented a decision the model does not contain.
    """
    g = baseline_graph()
    assert g.parallel_pairs() == ()
    assert routing_is_live(g, "WDC", "CSSU_A") is False
    assert routing_is_live(g, "WDC", "CSSU_B") is False


def test_an_alternate_arc_is_what_creates_the_decision():
    g = with_alternate_route(baseline_graph(), pair=("AL", "SB"), transit_hours=36.0)
    assert g.parallel_pairs() == (("AL", "SB"),)
    assert routing_is_live(g) is True


def test_astar_uses_the_cheap_arc_and_reroutes_when_it_dies():
    g = with_alternate_route(baseline_graph(), pair=("AL", "SB"), transit_hours=48.0)
    fast = astar_path(g, "WDC", "THEATRE")
    assert "op8_al_sb" in fast["arc_ids"]

    # R22 destroys the primary arc; the surviving longer route must be found, not the flow stopped.
    degraded = astar_path(g.with_down({"op8_al_sb"}), "WDC", "THEATRE")
    assert degraded["reachable"] is True
    assert "op8_al_sb_alt" in degraded["arc_ids"]
    assert degraded["cost"] > fast["cost"], "the alternate route must cost more, or it is free"


def test_killing_the_only_arc_of_a_hop_disconnects_the_chain():
    """Without an alternate route a destroyed LOC stops the flow -- the shipped behaviour."""
    out = astar_path(baseline_graph().with_down({"op8_al_sb"}), "WDC", "THEATRE")
    assert out["reachable"] is False
    assert out["cost"] == INF


def _random_graph(rng: random.Random) -> LOCGraph:
    arcs = []
    for pair, op in (("WDC", "AL"), 4), (("AL", "SB"), 8), (("SB", "CSSU_A"), 10), \
                    (("SB", "CSSU_B"), 10), (("CSSU_A", "THEATRE"), 12), \
                    (("CSSU_B", "THEATRE"), 12):
        tail, head = pair
        for k in range(rng.randint(1, 3)):
            arcs.append(Arc(f"{tail}_{head}_{k}".lower(), tail, head,
                            transit_hours=float(rng.randint(1, 200)), operation=op))
    graph = LOCGraph(tuple(arcs))
    ids = [a.arc_id for a in graph.arcs]
    rng.shuffle(ids)
    return graph.with_down(ids[:rng.randint(0, max(0, len(ids) // 3))])


@pytest.mark.parametrize("trial", range(200))
def test_astar_returns_the_same_cost_as_dijkstra(trial):
    """Randomised counterexample search: an inadmissible heuristic shows up as a cost mismatch."""
    g = _random_graph(random.Random(20260802 + trial))
    a, d = astar_path(g, "WDC", "THEATRE"), dijkstra_path(g, "WDC", "THEATRE")
    assert a["reachable"] == d["reachable"]
    if d["reachable"]:
        assert a["cost"] == pytest.approx(d["cost"]), "A* found a worse path: heuristic not admissible"


@pytest.mark.parametrize("trial", range(200))
def test_heuristic_never_exceeds_the_true_remaining_cost(trial):
    """Admissibility, checked directly at every node instead of only through the final cost."""
    g = _random_graph(random.Random(90000 + trial))
    h = admissible_heuristic(g, "CSSU_A")
    for node in hops_to(g, "CSSU_A"):
        true_cost = dijkstra_path(g, node, "CSSU_A")["cost"]
        if true_cost < INF:
            assert h(node) <= true_cost + 1e-9, f"h overestimates at {node}"


def test_an_inadmissible_heuristic_is_caught_by_the_dijkstra_comparison():
    """The mutant: prove the admissibility check can FAIL, or it proves nothing.

    A uniform rescaling of h is NOT a valid mutant here -- h is constant per layer, so scaling it
    preserves every within-layer ordering and A* still returns the optimum. That mistake wasted a
    first attempt. Real inadmissibility has to OVERESTIMATE at a specific node, so the search
    turns away from the branch that is actually cheapest.
    """
    from supply_chain import loc_graph as lg

    g = LOCGraph((Arc("a", "WDC", "AL", 1.0), Arc("b", "AL", "SB", 1.0),
                  Arc("expensive", "SB", "CSSU_A", 100.0),
                  Arc("cheap", "SB", "CSSU_B", 1.0),
                  Arc("x", "CSSU_B", "THEATRE", 1.0),
                  Arc("y", "CSSU_A", "THEATRE", 1.0)))
    truth = dijkstra_path(g, "SB", "THEATRE")
    honest = lg._search(g, "SB", "THEATRE", lg.admissible_heuristic(g, "THEATRE"))
    assert honest["cost"] == pytest.approx(truth["cost"])

    # Overestimate ONLY on the node the optimum goes through.
    poisoned = lg._search(g, "SB", "THEATRE", lambda n: 1000.0 if n == "CSSU_B" else 0.0)
    assert poisoned["cost"] > truth["cost"] + 1e-9, (
        "an inadmissible heuristic must produce a WORSE path; if it does not, this graph no "
        "longer exercises admissibility and the mutant needs rebuilding")


def test_search_cost_is_reported_because_it_is_the_efficiency_estimand():
    """`expansions` is the per-decision cost Garrido asked to compare (28 July)."""
    g = with_alternate_route(baseline_graph(), pair=("AL", "SB"), transit_hours=36.0)
    out = astar_path(g, "WDC", "CSSU_A")
    assert out["expansions"] > 0
    assert out["expansions"] <= dijkstra_path(g, "WDC", "CSSU_A")["expansions"], \
        "A* should not expand more nodes than uninformed Dijkstra"


def test_arc_target_tape_is_event_keyed_and_stable():
    """Risk targeting must not consume simulator RNG, or a policy change shifts every later draw."""
    arcs = [a.arc_id for a in baseline_graph().arcs]
    first = [arc_down_tape(simulation_seed=7, event_id=i, arcs=arcs) for i in range(50)]
    again = [arc_down_tape(simulation_seed=7, event_id=i, arcs=arcs) for i in range(50)]
    assert first == again
    assert arc_down_tape(simulation_seed=8, event_id=0, arcs=arcs) is not None
    assert len(set(first)) > 1, "a tape that always picks one arc is not targeting anything"


def test_structure_hash_ignores_arc_order_but_not_arc_content():
    g = baseline_graph()
    shuffled = LOCGraph(tuple(reversed(g.arcs)))
    assert g.structure_sha256() == shuffled.structure_sha256()
    slower = baseline_graph(transit_hours={8: 25.0})
    assert g.structure_sha256() != slower.structure_sha256()


@pytest.mark.parametrize("bad", [
    lambda: Arc("x", "WDC", "AL", -1.0),
    lambda: Arc("x", "NOWHERE", "AL", 1.0) and LOCGraph((Arc("x", "NOWHERE", "AL", 1.0),)),
    lambda: LOCGraph((Arc("x", "WDC", "AL", 1.0), Arc("x", "AL", "SB", 1.0))),
])
def test_invalid_structures_are_rejected(bad):
    with pytest.raises(ValueError):
        bad()
