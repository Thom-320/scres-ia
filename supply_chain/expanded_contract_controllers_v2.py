"""Corrective controllers for the expanded strategic-buffer contract.

Version 1 is retained as a diagnostic artifact.  This module fixes the two
largest comparability defects found in that instrument:

* a posture is a three-node vector, so the static frontier contains 6^3=216
  postures rather than six coupled rungs;
* the DDMRP-compatible arm is projected onto exactly that same discrete domain.

The state-conditioned replay MPC lives in the v2 runner because it must own the
materialized exogenous tapes and verify the replay state hash at every branch.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from itertools import product
from typing import Any

from supply_chain.expanded_contract_controllers import LADDER_HOURS, NODES, level_targets

Posture = tuple[int, int, int]
ALL_POSTURES: tuple[Posture, ...] = tuple(product(LADDER_HOURS, repeat=len(NODES)))


def posture_name(posture: Posture) -> str:
    return "static_" + "_".join(
        f"{node.replace('_', '')}I{hours}" for node, hours in zip(NODES, posture)
    )


def posture_targets(posture: Posture) -> dict[str, float]:
    if len(posture) != len(NODES):
        raise ValueError(f"Expected {len(NODES)} posture entries, got {len(posture)}")
    return {
        node: float(level_targets(int(hours))[node])
        for node, hours in zip(NODES, posture)
    }


def nearest_posture(targets: dict[str, float]) -> Posture:
    """Project arbitrary targets onto the shared 6^3 thesis domain."""
    selected: list[int] = []
    for node in NODES:
        target = float(targets[node])
        selected.append(
            min(
                LADDER_HOURS,
                key=lambda hours: (
                    abs(float(level_targets(hours)[node]) - target),
                    int(hours),
                ),
            )
        )
    return tuple(selected)  # type: ignore[return-value]


@dataclass
class VectorStaticPosture:
    posture: Posture
    name: str = field(init=False)

    def __post_init__(self) -> None:
        if self.posture not in ALL_POSTURES:
            raise ValueError(f"Posture outside frozen domain: {self.posture}")
        self.name = posture_name(self.posture)

    def reset(self) -> None:
        return None

    def act(self, sim: Any, epoch: int) -> dict[str, float]:
        return posture_targets(self.posture)


@dataclass
class ProjectedDDMRPController:
    """DDMRP-compatible dynamic buffer heuristic on the common action domain.

    The simulator has no strategic purchase-order object when replenishment lead
    time is zero.  Accordingly, on-order is recorded explicitly as zero rather
    than silently approximated.  Qualified demand spikes are the outstanding
    contingent backlog at the node.  Node usage is reconstructed from the
    container balance using recorded arrivals and on-hand changes, then averaged
    over the frozen rolling window.
    """

    lead_time_factor: float = 0.5
    variability_factor: float = 0.5
    order_cycle_days: float = 7.0
    window_days: float = 28.0
    dlt_days: dict[str, float] = field(
        default_factory=lambda: {"op3_rm": 28.0, "op5_rm": 28.0, "op9_rations": 7.0}
    )
    name: str = "ddmrp_projected_v2"

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._last_time = 0.0
        self._last_on_hand: dict[str, float] = {}
        self._last_arrival_count: dict[str, int] = {}
        self._usage: dict[str, deque[tuple[float, float]]] = {
            node: deque() for node in NODES
        }
        self.last_diagnostic: dict[str, Any] = {}

    @staticmethod
    def _on_hand(sim: Any) -> dict[str, float]:
        return {
            "op3_rm": float(sim.raw_material_wdc.level),
            "op5_rm": float(sim.raw_material_al.level),
            "op9_rations": float(sim.rations_sb.level),
        }

    def _node_usage(self, sim: Any) -> dict[str, float]:
        now = float(sim.env.now)
        current = self._on_hand(sim)
        usage: dict[str, float] = {}
        for node in NODES:
            events = list(sim.material_availability_events.get(node, ()))
            start = int(self._last_arrival_count.get(node, 0))
            arrivals = sum(float(qty) for _, qty in events[start:])
            previous = float(self._last_on_hand.get(node, current[node]))
            used = max(0.0, previous + arrivals - current[node])
            self._usage[node].append((now, used))
            cutoff = now - self.window_days * 24.0
            while self._usage[node] and self._usage[node][0][0] < cutoff:
                self._usage[node].popleft()
            self._last_arrival_count[node] = len(events)
        self._last_on_hand = current
        self._last_time = now
        return {
            node: sum(qty for _, qty in self._usage[node]) / max(self.window_days, 1e-9)
            for node in NODES
        }

    @staticmethod
    def _qualified_spikes(sim: Any) -> dict[str, float]:
        contingent = sum(
            float(order.remaining_qty)
            for order in sim.pending_backorders
            if bool(getattr(order, "contingent", False))
        )
        return {
            "op3_rm": 12.0 * contingent,
            "op5_rm": 12.0 * contingent,
            "op9_rations": contingent,
        }

    def act(self, sim: Any, epoch: int) -> dict[str, float]:
        adu = self._node_usage(sim)
        on_hand = self._on_hand(sim)
        spikes = self._qualified_spikes(sim)
        continuous: dict[str, float] = {}
        diagnostics: dict[str, Any] = {}
        for node in NODES:
            node_adu = adu[node]
            if node_adu <= 0.0:
                node_adu = 30_000.0 if node.endswith("_rm") else 2_500.0
            dlt = float(self.dlt_days[node])
            red = node_adu * dlt * self.lead_time_factor * (
                1.0 + self.variability_factor
            )
            yellow = node_adu * dlt
            green = max(
                node_adu * dlt * self.lead_time_factor,
                node_adu * self.order_cycle_days,
            )
            top_of_green = red + yellow + green
            top_of_yellow = red + yellow
            on_order = 0.0
            net_flow = on_hand[node] + on_order - spikes[node]
            continuous[node] = (
                top_of_green if net_flow < top_of_yellow else max(on_hand[node], red)
            )
            diagnostics[node] = {
                "adu": node_adu,
                "on_hand": on_hand[node],
                "on_order": on_order,
                "qualified_spikes": spikes[node],
                "net_flow_position": net_flow,
                "top_of_yellow": top_of_yellow,
                "top_of_green": top_of_green,
                "continuous_target": continuous[node],
            }
        posture = nearest_posture(continuous)
        targets = posture_targets(posture)
        self.last_diagnostic = {
            "epoch": int(epoch),
            "posture": list(posture),
            "targets": targets,
            "nodes": diagnostics,
        }
        return targets

