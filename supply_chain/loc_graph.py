"""Lines-of-communication as a GRAPH, so that re-routing becomes a decision.

Garrido's thesis states the simplification this module removes, in his own words (WRAP 2017,
Section 6.5.5): *"With regard to LOCs, the availability of distribution vehicles and the
planning/analysis of routes are taken for granted."* It is declared there as a modelling
convenience -- "to avoid including unnecessary details, to reduce the execution time" -- not as
physics.

That matters because R22 DESTROYS a line of communication (`_risk_R22_event` over op4/op8/op10/
op12), and the shipped LOCs are serial: taking one down simply stops the flow. There is no
alternative path, so there is no decision to take. Give the network a second arc and re-routing
becomes real, and A* is the canonical algorithm for it.

Why A* and not another comparator: it is the EXACT optimum of the routing sub-problem under a
known edge-state snapshot. Every null this project has produced was measured against a comparator
we chose ourselves; an exact planner cannot be dismissed as a straw man.

Nothing here touches the simulator. `baseline_graph()` returns one arc per pair, which is the
shipped topology, so the null arm is this module used with the shipped graph.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from heapq import heappop, heappush
from typing import Iterable, Mapping, Sequence

#: The physical chain of the thesis, Figure 6.4: suppliers -> raw-material warehouse ->
#: assembly plant -> combat-rations warehouse -> cross-docking point -> troops.
NODES = ("WDC", "AL", "SB", "CSSU_A", "CSSU_B", "THEATRE")

#: Which operation id each hop corresponds to, so risk targeting keeps speaking Garrido's
#: vocabulary instead of inventing one.
HOP_OPERATION = {("WDC", "AL"): 4, ("AL", "SB"): 8,
                 ("SB", "CSSU_A"): 10, ("SB", "CSSU_B"): 10,
                 ("CSSU_A", "THEATRE"): 12, ("CSSU_B", "THEATRE"): 12}

INF = float("inf")


@dataclass(frozen=True)
class Arc:
    """One line of communication. `arc_id` is what a risk destroys."""
    arc_id: str
    tail: str
    head: str
    transit_hours: float
    capacity: float = INF
    operation: int | None = None

    def __post_init__(self) -> None:
        if self.transit_hours < 0:
            raise ValueError(f"{self.arc_id}: transit_hours must be non-negative")
        if self.capacity < 0:
            raise ValueError(f"{self.arc_id}: capacity must be non-negative")


@dataclass
class LOCGraph:
    """A directed multigraph over NODES. Down arcs stay in the structure but are unusable."""
    arcs: tuple[Arc, ...]
    down: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for arc in self.arcs:
            if arc.arc_id in seen:
                raise ValueError(f"duplicate arc_id {arc.arc_id!r}")
            seen.add(arc.arc_id)
            for endpoint in (arc.tail, arc.head):
                if endpoint not in NODES:
                    raise ValueError(f"{arc.arc_id}: unknown node {endpoint!r}")

    def live_arcs(self) -> tuple[Arc, ...]:
        return tuple(a for a in self.arcs if a.arc_id not in self.down)

    def out_arcs(self, node: str) -> tuple[Arc, ...]:
        return tuple(a for a in self.live_arcs() if a.tail == node)

    def with_down(self, down: Iterable[str]) -> "LOCGraph":
        unknown = set(down) - {a.arc_id for a in self.arcs}
        if unknown:
            raise ValueError(f"unknown arc ids: {sorted(unknown)}")
        return LOCGraph(self.arcs, frozenset(down))

    def parallel_pairs(self) -> tuple[tuple[str, str], ...]:
        """Node pairs served by more than one arc: the only places a choice exists."""
        counts: dict[tuple[str, str], int] = {}
        for arc in self.arcs:
            counts[(arc.tail, arc.head)] = counts.get((arc.tail, arc.head), 0) + 1
        return tuple(sorted(pair for pair, n in counts.items() if n > 1))

    def structure_sha256(self) -> str:
        body = ";".join(f"{a.arc_id}|{a.tail}|{a.head}|{a.transit_hours}|{a.capacity}"
                        for a in sorted(self.arcs, key=lambda a: a.arc_id))
        return sha256(body.encode()).hexdigest()


def baseline_graph(transit_hours: Mapping[int, float] | None = None) -> LOCGraph:
    """The SHIPPED topology: exactly one arc per hop, so routing is moot.

    This is the null arm. A run over this graph must reproduce the serial-LOC model, and that is
    a testable claim rather than an assertion -- see tests/test_loc_graph.py.
    """
    hours = {4: 24.0, 8: 24.0, 10: 24.0, 12: 24.0}
    hours.update(transit_hours or {})
    return LOCGraph(tuple(
        Arc(arc_id=f"op{op}_{tail}_{head}".lower(), tail=tail, head=head,
            transit_hours=float(hours[op]), operation=op)
        for (tail, head), op in HOP_OPERATION.items()))


def with_alternate_route(graph: LOCGraph, *, pair: tuple[str, str],
                         transit_hours: float, capacity: float = INF,
                         suffix: str = "alt") -> LOCGraph:
    """Add ONE alternate arc for a pair. This is OUR assumption, and its price gets measured.

    The thesis takes route planning for granted, so it neither grants nor denies a second route.
    We declare the addition, and the fidelity price against the baseline graph is reported rather
    than waved away.
    """
    tail, head = pair
    base = next((a for a in graph.arcs if (a.tail, a.head) == pair), None)
    if base is None:
        raise ValueError(f"no existing arc for pair {pair}")
    return LOCGraph(graph.arcs + (Arc(arc_id=f"{base.arc_id}_{suffix}", tail=tail, head=head,
                                      transit_hours=float(transit_hours), capacity=capacity,
                                      operation=base.operation),), graph.down)


# --------------------------------------------------------------------------------------------
# Shortest path: A*, and the Dijkstra it must agree with.
# --------------------------------------------------------------------------------------------

def hops_to(graph: "LOCGraph", target: str) -> dict[str, int]:
    """Minimum hop count to `target` over the FULL structure, ignoring which arcs are down.

    Derived by BFS rather than hard-coded: a hand-written layer table was wrong here (it counted
    three hops from WDC when the chain is four) and a wrong table silently weakens or breaks the
    heuristic. Ignoring `down` is deliberate -- removing arcs can only make the real path longer,
    so this stays a lower bound and therefore keeps the heuristic admissible.
    """
    back: dict[str, list[str]] = {}
    for arc in graph.arcs:
        back.setdefault(arc.head, []).append(arc.tail)
    dist, frontier = {target: 0}, [target]
    while frontier:
        node = frontier.pop(0)
        for tail in back.get(node, ()):
            if tail not in dist:
                dist[tail] = dist[node] + 1
                frontier.append(tail)
    return dist


def min_hop_cost(graph: LOCGraph) -> float:
    """Cheapest live arc anywhere in the graph."""
    live = graph.live_arcs()
    return min((a.transit_hours for a in live), default=0.0)


def admissible_heuristic(graph: LOCGraph, target: str):
    """h(n) = (hops remaining to `target`) x (cheapest live arc).

    Admissible: any path from `n` traverses at least `hops(n)` arcs and none is cheaper than the
    graph minimum. The argument is not trusted -- `tests/test_loc_graph.py` checks h against
    exhaustive Dijkstra at every node on 200 randomised graphs, and carries a mutant that must
    make that check fail.
    """
    hops, cheapest = hops_to(graph, target), min_hop_cost(graph)

    def h(node: str) -> float:
        return hops.get(node, 0) * cheapest

    return h


def _search(graph: LOCGraph, source: str, target: str, heuristic=None) -> dict:
    """One implementation for both algorithms: `heuristic=None` IS Dijkstra.

    Keeping a single body means the A*-equals-Dijkstra test compares algorithms, not two
    independently buggy transcriptions of the same idea.
    """
    if source not in NODES or target not in NODES:
        raise ValueError(f"unknown node in ({source!r}, {target!r})")
    h = heuristic or (lambda _node: 0.0)
    best: dict[str, float] = {source: 0.0}
    came: dict[str, tuple[str, Arc]] = {}
    frontier: list[tuple[float, int, str]] = [(h(source), 0, source)]
    expansions = 0
    tie = 0
    closed: set[str] = set()
    while frontier:
        _, _, node = heappop(frontier)
        if node in closed:
            continue
        closed.add(node)
        expansions += 1
        if node == target:
            break
        for arc in graph.out_arcs(node):
            candidate = best[node] + arc.transit_hours
            if candidate < best.get(arc.head, INF) - 1e-12:
                best[arc.head] = candidate
                came[arc.head] = (node, arc)
                tie += 1
                heappush(frontier, (candidate + h(arc.head), tie, arc.head))
    if target not in best:
        return {"reachable": False, "cost": INF, "path": (), "arc_ids": (),
                "expansions": expansions}
    path, arc_ids, cursor = [target], [], target
    while cursor != source:
        cursor, arc = came[cursor]
        path.append(cursor)
        arc_ids.append(arc.arc_id)
    return {"reachable": True, "cost": best[target], "path": tuple(reversed(path)),
            "arc_ids": tuple(reversed(arc_ids)), "expansions": expansions}


def astar_path(graph: LOCGraph, source: str, target: str) -> dict:
    """Shortest surviving route under the current arc states, plus its search cost.

    `expansions` is reported because it is the per-decision cost that grows with the graph --
    the quantity Garrido asked to compare on 28 July (parameters, speed, convergence), and the
    only place a learned policy could plausibly earn something against an exact planner.
    """
    return _search(graph, source, target, admissible_heuristic(graph, target))


def dijkstra_path(graph: LOCGraph, source: str, target: str) -> dict:
    """Uninformed reference. A* must return the same cost on every graph."""
    return _search(graph, source, target, None)


def routing_is_live(graph: LOCGraph, source: str = "WDC", target: str = "CSSU_A") -> bool:
    """True when at least two distinct live routes exist, i.e. a decision actually exists.

    The default target is a CSSU, not THEATRE. Going through CSSU_A rather than CSSU_B is NOT a
    routing alternative: they are different destinations serving different demand, and treating
    them as interchangeable would have manufactured a decision that does not exist.

    A liveness check earns its place here: with the shipped graph this is False, and an
    experiment that reported routing value on a graph where routing is moot would be measuring
    nothing. G3-obs already cost us one run to a dead actuator.
    """
    if not _search(graph, source, target)["reachable"]:
        return False
    for arc in graph.live_arcs():
        alt = graph.with_down(set(graph.down) | {arc.arc_id})
        if _search(alt, source, target)["reachable"]:
            first = _search(graph, source, target)["arc_ids"]
            if arc.arc_id in first:
                return True
    return False


def arc_down_tape(*, simulation_seed: int, event_id: int, arcs: Sequence[str]) -> str:
    """Which arc a risk event destroys, keyed by the event and NOT by simulator RNG.

    Drawing the target from the shared RNG would shift every later draw whenever a policy changed
    the call order, which silently breaks CRN. The same event-keyed discipline is already used
    for the CSSU destination tape.
    """
    if not arcs:
        raise ValueError("no arcs to target")
    digest = sha256(f"estar-arc-v1:{simulation_seed}:{event_id}".encode()).digest()
    return sorted(arcs)[int.from_bytes(digest[:8], "big") % len(arcs)]
