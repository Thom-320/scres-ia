"""Source-conserving decision-contract kernel for Garrido E*.

This module is deliberately smaller than the historical DES.  It is the
canonical action/ledger layer that the E* DES bridge must implement before a
scientific run is permitted.  It is useful now for synthetic fixtures,
conservation tests, liveness tests, and planner-timing instrumentation.

The important invariant is negative capability: a target is not stock.  A
policy can request procurement or dispatch, but it cannot create inventory,
skip a lead time, spill a full buffer, or silently use a masked action.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import time
from typing import Any, Mapping


NODE_IDS: tuple[str, ...] = ("wdc", "al", "sb", "cssu_a", "cssu_b")
UPSTREAM_NODES: tuple[str, ...] = ("wdc", "al", "sb")
DOWNSTREAM_NODES: tuple[str, ...] = ("cssu_a", "cssu_b")
SUPPLIER_LANES: tuple[str, ...] = ("supplier_wdc", "supplier_al", "supplier_sb")
DISPATCH_LANES: tuple[str, ...] = ("sb_to_cssu_a", "sb_to_cssu_b")

DEFAULT_NODE_CAPACITIES: dict[str, float] = {
    "wdc": 100_000.0,
    "al": 100_000.0,
    "sb": 100_000.0,
    "cssu_a": 50_000.0,
    "cssu_b": 50_000.0,
}
DEFAULT_SUPPLIER_CAPACITY: dict[str, float] = {
    lane: 25_000.0 for lane in SUPPLIER_LANES
}
DEFAULT_LEAD_TIMES: dict[str, float] = {
    "supplier_wdc": 24.0,
    "supplier_al": 48.0,
    "supplier_sb": 72.0,
    "sb_to_cssu_a": 24.0,
    "sb_to_cssu_b": 24.0,
}
DEFAULT_TRANSPORT_CAPACITY: dict[str, float] = {
    lane: 25_000.0 for lane in (*SUPPLIER_LANES, *DISPATCH_LANES)
}

MASKS: dict[str, dict[str, bool]] = {
    "M000": {"P": False, "U": False, "D": False},
    "M100": {"P": True, "U": False, "D": False},
    "M010": {"P": False, "U": True, "D": False},
    "M001": {"P": False, "U": False, "D": True},
    "M110": {"P": True, "U": True, "D": False},
    "M101": {"P": True, "U": False, "D": True},
    "M011": {"P": False, "U": True, "D": True},
    "M111": {"P": True, "U": True, "D": True},
}


def _finite_nonnegative_map(
    values: Mapping[str, float], allowed: tuple[str, ...], name: str
) -> dict[str, float]:
    unknown = set(values) - set(allowed)
    if unknown:
        raise ValueError(f"{name}: unknown keys {sorted(unknown)}")
    output = {key: float(values.get(key, 0.0)) for key in allowed}
    if any(value < 0.0 or not math.isfinite(value) for value in output.values()):
        raise ValueError(f"{name}: values must be finite and non-negative")
    return output


@dataclass(frozen=True)
class DecisionMask:
    """The three decision rights activated by one factorial mask."""

    mask_id: str
    procurement: bool
    upstream_buffer: bool
    downstream_dispatch: bool

    @classmethod
    def from_id(cls, mask_id: str) -> "DecisionMask":
        try:
            values = MASKS[str(mask_id)]
        except KeyError as exc:
            raise ValueError(f"unknown E* mask {mask_id!r}") from exc
        return cls(
            mask_id=str(mask_id),
            procurement=bool(values["P"]),
            upstream_buffer=bool(values["U"]),
            downstream_dispatch=bool(values["D"]),
        )


@dataclass(frozen=True)
class TransitShipment:
    lane: str
    destination: str
    quantity: float
    due_at: float


@dataclass
class EStarAction:
    """The only action shape accepted by the E* kernel."""

    procurement_qty: dict[str, float] = field(default_factory=dict)
    buffer_targets: dict[str, float] = field(default_factory=dict)
    dispatch_qty: dict[str, float] = field(default_factory=dict)
    active_supplier_lanes: tuple[str, ...] = ()
    active_dispatch_lanes: tuple[str, ...] = ()

    def canonical(self) -> dict[str, Any]:
        return {
            "procurement_qty": {
                key: float(value)
                for key, value in sorted(self.procurement_qty.items())
            },
            "buffer_targets": {
                key: float(value)
                for key, value in sorted(self.buffer_targets.items())
            },
            "dispatch_qty": {
                key: float(value)
                for key, value in sorted(self.dispatch_qty.items())
            },
            "active_supplier_lanes": sorted(self.active_supplier_lanes),
            "active_dispatch_lanes": sorted(self.active_dispatch_lanes),
        }


@dataclass
class EStarState:
    time: float
    inventory: dict[str, float]
    in_transit: tuple[TransitShipment, ...]
    on_order: dict[str, float]
    buffer_targets: dict[str, float]
    backlog: dict[str, float]


@dataclass
class EStarLedger:
    """Cumulative physical and service accounting for one kernel episode."""

    initial_inventory: dict[str, float]
    source_stock_initial: dict[str, float]
    source_stock_remaining: dict[str, float]
    procurement_ordered: dict[str, float] = field(default_factory=dict)
    procurement_received: dict[str, float] = field(default_factory=dict)
    dispatch_sent: dict[str, float] = field(default_factory=dict)
    delivered: dict[str, float] = field(default_factory=dict)
    demanded: dict[str, float] = field(default_factory=dict)
    unresolved: dict[str, float] = field(default_factory=dict)
    blocked_qty: dict[str, float] = field(default_factory=dict)
    resource_usage: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in (
            "procurement_ordered",
            "procurement_received",
            "dispatch_sent",
            "delivered",
            "demanded",
            "unresolved",
            "blocked_qty",
            "resource_usage",
        ):
            getattr(self, field_name).setdefault("total", 0.0)

    def physical_residual(
        self, inventory: Mapping[str, float], in_transit: tuple[TransitShipment, ...]
    ) -> float:
        source_issued = sum(
            self.source_stock_initial[key] - self.source_stock_remaining[key]
            for key in self.source_stock_initial
        )
        transit = sum(float(shipment.quantity) for shipment in in_transit)
        delivered = sum(
            float(value)
            for key, value in self.delivered.items()
            if key != "total"
        )
        present = sum(float(value) for value in inventory.values())
        initial = sum(float(value) for value in self.initial_inventory.values())
        return initial + source_issued - present - transit - delivered

    def as_evidence(
        self, inventory: Mapping[str, float], in_transit: tuple[TransitShipment, ...]
    ) -> dict[str, Any]:
        def with_total(values: Mapping[str, float]) -> dict[str, float]:
            clean = {
                key: float(value)
                for key, value in values.items()
                if key != "total"
            }
            clean["total"] = float(sum(clean.values()))
            return clean

        return {
            "initial_inventory": dict(self.initial_inventory),
            "source_stock_initial": dict(self.source_stock_initial),
            "source_stock_remaining": dict(self.source_stock_remaining),
            "procurement_ordered": with_total(self.procurement_ordered),
            "procurement_received": with_total(self.procurement_received),
            "dispatch_sent": with_total(self.dispatch_sent),
            "delivered": with_total(self.delivered),
            "demanded": with_total(self.demanded),
            "unresolved": with_total(self.unresolved),
            "blocked_qty": with_total(self.blocked_qty),
            "resource_usage": with_total(self.resource_usage),
            "physical_residual": self.physical_residual(inventory, in_transit),
        }


@dataclass(frozen=True)
class PlannerStats:
    planner: str
    elapsed_seconds: float
    kernel_rollouts: int
    des_calls: int
    solver_iterations: int
    peak_memory_bytes: int | None = None


@dataclass
class EStarTransition:
    state: EStarState
    observation: dict[str, Any]
    ledger: dict[str, Any]
    planner_stats: PlannerStats | None


class EStarKernel:
    """Small source-conserving kernel used by the E* contract and preflight tools.

    The historical DES bridge is intentionally separate.  This class is not
    allowed to silently claim flags-off equivalence; callers must provide and
    record a bridge receipt before promoting a timing result to ``H_compute``.
    """

    def __init__(
        self,
        *,
        mask_id: str = "M111",
        node_capacities: Mapping[str, float] | None = None,
        supplier_capacity: Mapping[str, float] | None = None,
        source_stock: Mapping[str, float] | None = None,
        lead_times: Mapping[str, float] | None = None,
        transport_capacity: Mapping[str, float] | None = None,
        initial_inventory: Mapping[str, float] | None = None,
        decision_epoch_hours: float = 168.0,
        horizon_hours: float = 52.0 * 168.0,
    ) -> None:
        if (
            not math.isfinite(float(decision_epoch_hours))
            or not math.isfinite(float(horizon_hours))
            or decision_epoch_hours <= 0.0
            or horizon_hours <= 0.0
        ):
            raise ValueError("decision epoch and horizon must be positive")
        self.mask = DecisionMask.from_id(mask_id)
        self.node_capacities = _finite_nonnegative_map(
            node_capacities or DEFAULT_NODE_CAPACITIES, NODE_IDS, "node_capacities"
        )
        self.supplier_capacity = _finite_nonnegative_map(
            supplier_capacity or DEFAULT_SUPPLIER_CAPACITY,
            SUPPLIER_LANES,
            "supplier_capacity",
        )
        self.source_stock_initial = _finite_nonnegative_map(
            source_stock
            or {lane: self.supplier_capacity[lane] * 100.0 for lane in SUPPLIER_LANES},
            SUPPLIER_LANES,
            "source_stock",
        )
        self.lead_times = _finite_nonnegative_map(
            lead_times or DEFAULT_LEAD_TIMES,
            (*SUPPLIER_LANES, *DISPATCH_LANES),
            "lead_times",
        )
        self.transport_capacity = _finite_nonnegative_map(
            transport_capacity or DEFAULT_TRANSPORT_CAPACITY,
            (*SUPPLIER_LANES, *DISPATCH_LANES),
            "transport_capacity",
        )
        self.initial_inventory = _finite_nonnegative_map(
            initial_inventory or {node: 0.0 for node in NODE_IDS},
            NODE_IDS,
            "initial_inventory",
        )
        if any(
            self.initial_inventory[node] > self.node_capacities[node] + 1e-9
            for node in NODE_IDS
        ):
            raise ValueError("initial inventory exceeds a node capacity")
        self.decision_epoch_hours = float(decision_epoch_hours)
        self.horizon_hours = float(horizon_hours)
        self._time = 0.0
        self._inventory: dict[str, float] = {}
        self._buffer_targets: dict[str, float] = {}
        self._backlog: dict[str, float] = {}
        self._transit: list[TransitShipment] = []
        self._source_stock: dict[str, float] = {}
        self._ledger: EStarLedger
        self.reset()

    def reset(self) -> EStarState:
        self._time = 0.0
        self._inventory = dict(self.initial_inventory)
        self._buffer_targets = {node: 0.0 for node in NODE_IDS}
        self._backlog = {node: 0.0 for node in DOWNSTREAM_NODES}
        self._transit = []
        self._source_stock = dict(self.source_stock_initial)
        self._ledger = EStarLedger(
            initial_inventory=dict(self.initial_inventory),
            source_stock_initial=dict(self.source_stock_initial),
            source_stock_remaining=dict(self._source_stock),
            procurement_ordered={lane: 0.0 for lane in SUPPLIER_LANES},
            procurement_received={node: 0.0 for node in NODE_IDS},
            dispatch_sent={lane: 0.0 for lane in DISPATCH_LANES},
            delivered={node: 0.0 for node in DOWNSTREAM_NODES},
            demanded={node: 0.0 for node in DOWNSTREAM_NODES},
            unresolved={node: 0.0 for node in DOWNSTREAM_NODES},
            blocked_qty={node: 0.0 for node in NODE_IDS},
            resource_usage={key: 0.0 for key in (*SUPPLIER_LANES, *DISPATCH_LANES)},
        )
        return self.state()

    def clone(self) -> "EStarKernel":
        return deepcopy(self)

    def state(self) -> EStarState:
        on_order = {node: 0.0 for node in NODE_IDS}
        for shipment in self._transit:
            on_order[shipment.destination] += float(shipment.quantity)
        return EStarState(
            time=float(self._time),
            inventory=dict(self._inventory),
            in_transit=tuple(self._transit),
            on_order=on_order,
            buffer_targets=dict(self._buffer_targets),
            backlog=dict(self._backlog),
        )

    def observe(self) -> dict[str, Any]:
        """Return only current/past state; no future demand or risk is exposed."""
        state = self.state()
        return {
            "time": state.time,
            "inventory": dict(state.inventory),
            "on_order": dict(state.on_order),
            "buffer_targets": dict(state.buffer_targets),
            "backlog": dict(state.backlog),
            "mask_id": self.mask.mask_id,
        }

    def _validate_action(self, action: EStarAction) -> None:
        procurement = _finite_nonnegative_map(
            action.procurement_qty, SUPPLIER_LANES, "procurement_qty"
        )
        targets = _finite_nonnegative_map(
            action.buffer_targets, NODE_IDS, "buffer_targets"
        )
        dispatch = _finite_nonnegative_map(
            action.dispatch_qty, DISPATCH_LANES, "dispatch_qty"
        )
        if not self.mask.procurement and any(procurement.values()):
            raise ValueError(f"mask {self.mask.mask_id} does not permit procurement")
        if not self.mask.upstream_buffer:
            for node in UPSTREAM_NODES:
                if abs(targets[node] - self._buffer_targets[node]) > 1e-9:
                    raise ValueError(
                        f"mask {self.mask.mask_id} must carry upstream target {node}"
                    )
        if not self.mask.downstream_dispatch and any(dispatch.values()):
            raise ValueError(
                f"mask {self.mask.mask_id} does not permit downstream dispatch"
            )
        for node, target in targets.items():
            if target > self.node_capacities[node] + 1e-9:
                raise ValueError(f"buffer target exceeds capacity for {node}")
        supplier_lanes = set(action.active_supplier_lanes)
        dispatch_lanes = set(action.active_dispatch_lanes)
        if not supplier_lanes <= set(SUPPLIER_LANES):
            raise ValueError("unknown active supplier lane")
        if not dispatch_lanes <= set(DISPATCH_LANES):
            raise ValueError("unknown active dispatch lane")
        if not self.mask.procurement and supplier_lanes:
            raise ValueError(f"mask {self.mask.mask_id} does not permit supplier lanes")
        if not self.mask.downstream_dispatch and dispatch_lanes:
            raise ValueError(
                f"mask {self.mask.mask_id} does not permit dispatch lanes"
            )
        if any(
            procurement[lane] > 0.0 and lane not in supplier_lanes
            for lane in SUPPLIER_LANES
        ):
            raise ValueError("positive procurement requires its lane to be active")
        if any(
            dispatch[lane] > 0.0 and lane not in dispatch_lanes
            for lane in DISPATCH_LANES
        ):
            raise ValueError("positive dispatch requires its lane to be active")
        if any(procurement[lane] > 0.0 for lane in SUPPLIER_LANES) and not supplier_lanes:
            raise ValueError("positive procurement requires an active supplier lane")
        if any(dispatch[lane] > 0.0 for lane in DISPATCH_LANES) and not dispatch_lanes:
            raise ValueError("positive dispatch requires an active dispatch lane")

    def _destination_for_supplier(self, lane: str) -> str:
        return {
            "supplier_wdc": "wdc",
            "supplier_al": "al",
            "supplier_sb": "sb",
        }[lane]

    def _receive_due(self) -> None:
        remaining: list[TransitShipment] = []
        for shipment in self._transit:
            if shipment.due_at > self._time + 1e-9:
                remaining.append(shipment)
                continue
            room = max(0.0, self.node_capacities[shipment.destination] - self._inventory[shipment.destination])
            admitted = min(float(shipment.quantity), room)
            blocked = float(shipment.quantity) - admitted
            self._inventory[shipment.destination] += admitted
            if blocked > 1e-9:
                remaining.append(
                    TransitShipment(
                        lane=shipment.lane,
                        destination=shipment.destination,
                        quantity=blocked,
                        due_at=self._time + self.decision_epoch_hours,
                    )
                )
                self._ledger.blocked_qty[shipment.destination] += blocked
            if shipment.lane in SUPPLIER_LANES:
                self._ledger.procurement_received[shipment.destination] += admitted
            self._ledger.resource_usage[shipment.lane] += admitted
        self._transit = remaining

    def step(
        self,
        action: EStarAction,
        *,
        demand: Mapping[str, float] | None = None,
        planner_stats: PlannerStats | None = None,
    ) -> EStarTransition:
        started = time.perf_counter()
        self._receive_due()
        self._validate_action(action)
        procurement = _finite_nonnegative_map(
            action.procurement_qty, SUPPLIER_LANES, "procurement_qty"
        )
        targets = _finite_nonnegative_map(
            action.buffer_targets, NODE_IDS, "buffer_targets"
        )
        dispatch = _finite_nonnegative_map(
            action.dispatch_qty, DISPATCH_LANES, "dispatch_qty"
        )
        if self.mask.upstream_buffer or self.mask.downstream_dispatch:
            self._buffer_targets.update(targets)
        for lane, quantity in procurement.items():
            if quantity <= 0.0:
                continue
            available_source = self._source_stock[lane]
            if quantity > available_source + 1e-9:
                raise ValueError(f"procurement exceeds source stock on {lane}")
            if quantity > self.supplier_capacity[lane] + 1e-9:
                raise ValueError(f"procurement exceeds epoch capacity on {lane}")
            if quantity > self.transport_capacity[lane] + 1e-9:
                raise ValueError(f"procurement exceeds transport capacity on {lane}")
            self._source_stock[lane] -= quantity
            destination = self._destination_for_supplier(lane)
            self._transit.append(
                TransitShipment(
                    lane=lane,
                    destination=destination,
                    quantity=quantity,
                    due_at=self._time + self.lead_times[lane],
                )
            )
            self._ledger.procurement_ordered[lane] += quantity
            self._ledger.resource_usage[lane] += quantity
        remaining_sb = self._inventory["sb"]
        for lane, quantity in dispatch.items():
            if quantity <= 0.0:
                continue
            if quantity > self.transport_capacity[lane] + 1e-9:
                raise ValueError(f"dispatch exceeds transport capacity on {lane}")
            if quantity > remaining_sb + 1e-9:
                raise ValueError("dispatch exceeds available SB inventory")
            remaining_sb -= quantity
            destination = {
                "sb_to_cssu_a": "cssu_a",
                "sb_to_cssu_b": "cssu_b",
            }[lane]
            self._transit.append(
                TransitShipment(
                    lane=lane,
                    destination=destination,
                    quantity=quantity,
                    due_at=self._time + self.lead_times[lane],
                )
            )
            self._ledger.dispatch_sent[lane] += quantity
            self._ledger.resource_usage[lane] += quantity
        self._inventory["sb"] = remaining_sb
        next_time = min(self.horizon_hours, self._time + self.decision_epoch_hours)
        self._time = next_time
        self._receive_due()
        observed_demand = _finite_nonnegative_map(
            demand or {}, DOWNSTREAM_NODES, "demand"
        )
        for node, quantity in observed_demand.items():
            self._ledger.demanded[node] += quantity
            delivered = min(quantity, self._inventory[node])
            self._inventory[node] -= delivered
            unresolved = quantity - delivered
            self._ledger.delivered[node] += delivered
            self._ledger.unresolved[node] += unresolved
            self._backlog[node] += unresolved
        self._ledger.source_stock_remaining = dict(self._source_stock)
        elapsed = time.perf_counter() - started
        stats = planner_stats or PlannerStats(
            planner="kernel_action",
            elapsed_seconds=elapsed,
            kernel_rollouts=0,
            des_calls=0,
            solver_iterations=0,
        )
        return EStarTransition(
            state=self.state(),
            observation=self.observe(),
            ledger=self._ledger.as_evidence(
                self._inventory, tuple(self._transit)
            ),
            planner_stats=stats,
        )

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "mask_id": self.mask.mask_id,
            "state": asdict(self.state()),
            "ledger": self._ledger.as_evidence(
                self._inventory, tuple(self._transit)
            ),
        }

    def payload_sha256(self) -> str:
        body = json.dumps(
            self.canonical_payload(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(body).hexdigest()


def flags_off_bridge_descriptor() -> dict[str, Any]:
    """Return the fail-closed status of the historical DES bridge.

    The descriptor is intentionally not a PASS.  A real bridge receipt must
    be produced by the DES adapter after an independent golden-vector check.
    """
    return {
        "status": "DES_BRIDGE_PENDING",
        "kernel": "EStarKernel",
        "historical_des": "MFSCSimulation",
        "requires": [
            "M000 trajectory or payload golden vector",
            "same exogenous tape",
            "mass/capacity/WIP/resource ledger equality",
            "independent mutation test",
        ],
    }
