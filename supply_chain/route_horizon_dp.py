"""Full-horizon DP over the route/dispatch schedule: the bound Program L's audit asked for.

Program L retracted its `H_PI` with a precise diagnosis: the quantity "came from a myopic
true-state routing rule with a fixed state-responsive dispatch trigger. It did not optimize the
full trajectory. Negative values in two cells prove it was not a perfect-information upper bound."
A myopic rule can be beaten in hindsight, so it bounds nothing.

This optimizes the WHOLE trajectory by backward induction, and therefore does bound -- but only
over a declared scope, and the scope is the load-bearing part:

STATE.  `(epoch, convoy_free_at)`. The convoy is the single scarce carrier, so when it next
becomes available is the only thing a past decision leaves behind that constrains a future one.
That makes the state a sufficient statistic and Bellman applies exactly.

CLAIRVOYANT.  Transitions read the REALIZED tape: R22 outages and the alternate's degradation are
CRN-fixed per tape, so the DP sees the future. That is deliberate -- an upper bound must be
allowed to cheat. It is a ceiling for causal policies, never a policy anyone could deploy.

SURROGATE, and this is the limit to state out loud.  The stage cost is an ADDITIVE service-loss
proxy: undelivered demand accumulating per hour. Canonical ReT is not additive over departures,
so this is an exact optimum of the surrogate and NOT automatically a bound on ReT. Claiming
otherwise would repeat Program L's error one level up. Whether the surrogate dominates ReT is an
empirical question with its own falsifier, not an assumption to smuggle in here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

INF = float("inf")

HOLD, ROUTE_1, ROUTE_2 = "HOLD", "ROUTE_1", "ROUTE_2"
ACTIONS = (HOLD, ROUTE_1, ROUTE_2)


@dataclass(frozen=True)
class RouteTape:
    """One realized tape: everything the DP is allowed to know.

    `route1_down` and `route2_degraded` are indexed by decision epoch, matching the CRN tapes the
    Program L environment already builds from the R22 event list and the Z2 semi-Markov draw.
    """
    route1_down: tuple[bool, ...]
    route2_degraded: tuple[bool, ...]
    demand_per_epoch: tuple[float, ...]
    route1_hours: float = 24.0
    route2_base_hours: float = 36.0
    route2_penalty_hours: float = 24.0
    convoy_capacity: float = 5_000.0
    epoch_hours: float = 24.0

    def __post_init__(self) -> None:
        n = len(self.route1_down)
        if not (len(self.route2_degraded) == len(self.demand_per_epoch) == n):
            raise ValueError("tape series must have equal length")
        if n == 0:
            raise ValueError("tape must have at least one epoch")
        for name in ("route1_hours", "route2_base_hours", "route2_penalty_hours",
                     "convoy_capacity", "epoch_hours"):
            if float(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")

    @property
    def horizon(self) -> int:
        return len(self.route1_down)

    def transit_hours(self, action: str, epoch: int) -> float:
        """Round-trip time. INF when the action is infeasible at this epoch."""
        if action == ROUTE_1:
            return INF if self.route1_down[epoch] else 2.0 * self.route1_hours
        if action == ROUTE_2:
            extra = self.route2_penalty_hours if self.route2_degraded[epoch] else 0.0
            return 2.0 * (self.route2_base_hours + extra)
        raise ValueError(f"{action!r} has no transit")


def _epochs_busy(tape: RouteTape, action: str, epoch: int) -> float:
    hours = tape.transit_hours(action, epoch)
    return INF if hours == INF else hours / tape.epoch_hours


def solve(tape: RouteTape) -> dict[str, Any]:
    """Backward induction over `(epoch, convoy_free_at)`.

    Returns the optimal cost, the schedule achieving it, and the number of states expanded -- the
    last so the DP's own cost is comparable with A*'s, which is the efficiency estimand Garrido
    asked to measure on 28 July.
    """
    n = tape.horizon
    # Undelivered demand is carried forward; a departure clears up to the convoy capacity when it
    # ARRIVES, so a schedule is only as good as its arrival times.
    # Value function over (epoch, convoy_free_at) with convoy_free_at clipped to the horizon.
    best: dict[tuple[int, int], float] = {}
    move: dict[tuple, str] = {}
    expansions = 0

    def value(epoch: int, free_at: int, carried: float,
              pending_qty: float, pending_arrival: int) -> float:
        """Cost-to-go. `carried` is demand not yet DELIVERED; goods in transit still count.

        Rations on a convoy have not reached the troops, so they keep costing until `arrival`.
        The first version cleared them at departure, which made a 72 h route look as good as a
        24 h one and turned the bound into a bound on a fiction.
        """
        nonlocal expansions
        if epoch >= n:
            return 0.0
        # Delivery lands first: in-transit goods arrive and only then stop costing.
        if pending_qty > 0.0 and pending_arrival <= epoch:
            carried = max(0.0, carried - pending_qty)
            pending_qty, pending_arrival = 0.0, 0
        key = (epoch, free_at, round(carried, 6), round(pending_qty, 6), pending_arrival)
        cached = best.get(key)
        if cached is not None:
            return cached
        expansions += 1
        carried_now = carried + tape.demand_per_epoch[epoch]
        stage = carried_now * tape.epoch_hours

        options: list[tuple[float, str]] = [
            (stage + value(epoch + 1, free_at, carried_now, pending_qty, pending_arrival), HOLD)]
        if free_at <= epoch and pending_qty == 0.0:
            for action in (ROUTE_1, ROUTE_2):
                busy = _epochs_busy(tape, action, epoch)
                if busy == INF:
                    continue
                arrival = epoch + max(1, int(round(busy / 2.0)))
                if arrival >= n:
                    continue
                nxt_free = epoch + max(1, int(round(busy)))
                loaded = min(carried_now, tape.convoy_capacity)
                options.append((stage + value(epoch + 1, nxt_free, carried_now, loaded, arrival),
                                action))

        chosen = min(options, key=lambda t: t[0])
        best[key] = chosen[0]
        move[key] = chosen[1]
        return chosen[0]

    total = value(0, 0, 0.0, 0.0, 0)
    return {"optimal_cost": total, "expansions": expansions,
            "horizon": n, "scope": "ARRIVAL_TIMED_ADDITIVE_SERVICE_LOSS_SURROGATE_NOT_CANONICAL_RET",
            "direction": "MINIMISED_LOSS: a lower cost is better, so the clairvoyant optimum is a LOWER bound on achievable loss"}


def evaluate_schedule(tape: RouteTape, schedule: Sequence[str]) -> float:
    """Cost of a FIXED action sequence, under the same accounting the DP uses.

    Exists so the DP can be checked against brute force rather than trusted: any schedule
    evaluated here must cost at least the DP optimum.
    """
    if len(schedule) != tape.horizon:
        raise ValueError("schedule length must match the horizon")
    carried, free_at, total = 0.0, 0, 0.0
    pending_qty, pending_arrival = 0.0, 0
    for epoch, action in enumerate(schedule):
        if pending_qty > 0.0 and pending_arrival <= epoch:
            carried = max(0.0, carried - pending_qty)
            pending_qty, pending_arrival = 0.0, 0
        carried += tape.demand_per_epoch[epoch]
        total += carried * tape.epoch_hours
        if action == HOLD or free_at > epoch or pending_qty > 0.0:
            continue
        busy = _epochs_busy(tape, action, epoch)
        if busy == INF:
            continue
        arrival = epoch + max(1, int(round(busy / 2.0)))
        if arrival >= tape.horizon:
            continue
        pending_qty, pending_arrival = min(carried, tape.convoy_capacity), arrival
        free_at = epoch + max(1, int(round(busy)))
    return total


def myopic_schedule(tape: RouteTape) -> list[str]:
    """The comparator the DP must beat or tie: dispatch as soon as possible by the cheapest
    LIVE route, deciding on current state only. This is the class Program L's retracted H_PI
    belonged to, reproduced here so the gap between myopic and full-horizon is measurable."""
    out, free_at = [], 0
    for epoch in range(tape.horizon):
        if free_at > epoch:
            out.append(HOLD)
            continue
        feasible = [(tape.transit_hours(a, epoch), a) for a in (ROUTE_1, ROUTE_2)]
        feasible = [(h, a) for h, a in feasible if h != INF]
        if not feasible:
            out.append(HOLD)
            continue
        hours, action = min(feasible)
        out.append(action)
        free_at = epoch + max(1, int(round(hours / tape.epoch_hours)))
    return out
