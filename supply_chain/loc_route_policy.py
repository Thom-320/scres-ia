"""Route selection over the LOC graph: the exact MYOPIC optimum, and nothing more.

Program L already built alternate-route physics on the Op8 leg and ran it. Its corrective audit
(`docs/PROGRAM_L_ROUTE_RECOURSE_VERDICT_2026-07-13.md`) is a no-go for one heuristic in two cells,
and it says explicitly that this is "not a closure of the route-recourse family". It also names
the gap: "the comparator set contained only route 1, route 2, and an alternating route rule ...
MPC, DP/belief policy, hysteresis, and exact or bounded calendars were absent."

This module supplies ONE of the missing comparators, and the label matters more than the code.

WHAT IT IS: given the arc states at the moment of decision, `plan_route` returns the cheapest
surviving path. That is exact -- it is the optimum of the single-decision routing sub-problem, and
`tests/test_loc_graph.py` checks the heuristic against Dijkstra rather than arguing admissibility.

WHAT IT IS NOT, and this is the trap Program L fell into: it is MYOPIC. The retraction states that
the quantity formerly labelled `H_PI` "came from a myopic true-state routing rule ... It did not
optimize the full trajectory. Negative values in two cells prove it was not a perfect-information
upper bound." A* over current arc states is myopic in exactly the same way. It is therefore a
COMPARATOR, never a bound, and never an H_PI. The certified bound the verdict asks for needs a
full-horizon DP over (inventory, convoy availability, arc states), which is not this.

And an honest note on scope: with a single parallel hop, A* degenerates to picking the cheaper of
two live arcs -- a `min`, dressed up. The graph machinery earns its place only once more than one
hop carries an alternate, because then path composition actually matters. Saying so is better than
implying the algorithm is doing work it is not.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .loc_graph import INF, LOCGraph, astar_path, dijkstra_path

#: Program L's two Op8 routes, expressed as arcs so a planner can reason about them.
PROGRAM_L_ROUTE_ARCS = {"R1": "op8_al_sb", "R2": "op8_al_sb_alt"}


@dataclass(frozen=True)
class RouteChoice:
    """One routing decision, with the search cost that makes it comparable to a learner."""
    route_id: str | None
    arc_ids: tuple[str, ...]
    transit_hours: float
    expansions: int
    reachable: bool
    #: True when more than one live route existed, i.e. a decision was actually taken.
    had_a_choice: bool

    def as_evidence(self) -> dict[str, Any]:
        return {"route_id": self.route_id, "arc_ids": list(self.arc_ids),
                "transit_hours": self.transit_hours, "expansions": self.expansions,
                "reachable": self.reachable, "had_a_choice": self.had_a_choice}


def _route_id_for(arc_ids: Iterable[str]) -> str | None:
    """Name the plan with Program L's vocabulary when it corresponds to one of its routes."""
    ids = set(arc_ids)
    for name, arc in PROGRAM_L_ROUTE_ARCS.items():
        if arc in ids:
            return name
    return None


def live_route_count(graph: LOCGraph, source: str, target: str) -> int:
    """How many distinct first-hop options survive. Zero means the flow is stopped."""
    return sum(1 for arc in graph.out_arcs(source)
               if dijkstra_path(graph, arc.head, target)["reachable"] or arc.head == target)


def plan_route(graph: LOCGraph, source: str, target: str, *,
               down: Iterable[str] = ()) -> RouteChoice:
    """The cheapest surviving route under the arc states given.

    `down` is applied on top of the graph's own down set, so a caller can pass the R22 outages of
    the current epoch without mutating shared state.
    """
    state = graph.with_down(set(graph.down) | set(down))
    plan = astar_path(state, source, target)
    return RouteChoice(
        route_id=_route_id_for(plan["arc_ids"]) if plan["reachable"] else None,
        arc_ids=tuple(plan["arc_ids"]),
        transit_hours=float(plan["cost"]) if plan["reachable"] else INF,
        expansions=int(plan["expansions"]),
        reachable=bool(plan["reachable"]),
        had_a_choice=live_route_count(state, source, target) > 1)


def program_l_graph(*, route1_hours: float, route2_hours: float,
                    upstream_hours: float = 24.0) -> LOCGraph:
    """Program L's two Op8 routes as a graph, so the planner sees what the env sees.

    R2 is the DISCLOSED alternate. It is our assumption, not the thesis's: Section 6.5.5 takes
    route planning as given, so it neither grants nor denies a second route, and Program L already
    flagged it as pending Garrido face validation. Nothing here changes that status.
    """
    from .loc_graph import Arc

    return LOCGraph((
        Arc("op4_wdc_al", "WDC", "AL", float(upstream_hours), operation=4),
        Arc(PROGRAM_L_ROUTE_ARCS["R1"], "AL", "SB", float(route1_hours), operation=8),
        Arc(PROGRAM_L_ROUTE_ARCS["R2"], "AL", "SB", float(route2_hours), operation=None),
    ))


def plan_program_l_epoch(*, route1_hours: float, route2_hours: float,
                         route1_down: bool) -> RouteChoice:
    """One Program L decision epoch, planned on the graph instead of by a hand-written rule.

    `route1_down` is the R22 state of the thesis Op8 leg. R2 bypasses Op8 by construction, so it
    is never taken down by that risk -- which is precisely why an alternate changes the outcome
    when the primary dies, and why the shipped serial model has nothing to decide.
    """
    graph = program_l_graph(route1_hours=route1_hours, route2_hours=route2_hours)
    down = {PROGRAM_L_ROUTE_ARCS["R1"]} if route1_down else set()
    return plan_route(graph, "AL", "SB", down=down)
