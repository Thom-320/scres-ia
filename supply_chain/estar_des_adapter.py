"""Source-conserving E* adapter around the historical MFSC DES.

The historical ``MFSCSimulation`` remains untouched.  ``M000`` is delegated to
that class unchanged; non-null masks use this adapter's explicit procurement
and dispatch queues.  The adapter is deliberately conservative:

* buffer targets are intentions, never stock;
* supplier actions consume an explicit source stock and arrive after a frozen
  lead time;
* dispatch actions consume available SB stock and can only serve existing
  claimant backlog;
* full buffers block arrivals instead of spilling them;
* the parent DES's material ledger must still close before a bridge receipt can
  be issued.

This is an engineering bridge, not an authority to run new science.  The
contract runner must still verify M000 against its golden payload and exercise
mutation tests before changing the E* bridge status.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Mapping

import simpy

from supply_chain.estar_kernel import (
    DOWNSTREAM_NODES,
    EStarAction,
    MASKS,
    SUPPLIER_LANES,
)
from supply_chain.supply_chain import MFSCSimulation


class EStarBridgeNotReady(RuntimeError):
    """Raised when a caller asks the adapter to use an unsupported contract."""


@dataclass(frozen=True)
class EStarTransitRecord:
    lane: str
    destination: str
    quantity: float
    requested_at: float
    due_at: float


SUPPLIER_DESTINATIONS: dict[str, tuple[str, str]] = {
    "supplier_wdc": ("wdc", "raw_material_wdc"),
    "supplier_al": ("al", "raw_material_al"),
    "supplier_sb": ("sb", "rations_sb"),
}
NODE_TO_TARGET: dict[str, str] = {
    "wdc": "op3_rm",
    "al": "op5_rm",
    "sb": "op9_rations",
}


def _finite_nonnegative(values: Mapping[str, float], allowed: tuple[str, ...], name: str) -> dict[str, float]:
    unknown = set(values) - set(allowed)
    if unknown:
        raise ValueError(f"{name}: unknown keys {sorted(unknown)}")
    result = {key: float(values.get(key, 0.0)) for key in allowed}
    if any(value < 0.0 or not math.isfinite(value) for value in result.values()):
        raise ValueError(f"{name}: values must be finite and non-negative")
    return result


class EStarDESAdapter(MFSCSimulation):
    """Historical DES plus a source-conserving P/U/D action contract.

    ``expanded=True`` is used only by burned fixtures until an independent
    bridge receipt is written.  ``expanded=False`` exists solely to prove that
    the adapter does not alter the historical M000 path.
    """

    def __init__(
        self,
        *,
        mask_id: str = "M111",
        expanded: bool = True,
        node_capacities: Mapping[str, float] | None = None,
        supplier_capacity: Mapping[str, float] | None = None,
        source_stock: Mapping[str, float] | None = None,
        e_star_lead_times: Mapping[str, float] | None = None,
        e_star_transport_capacity: Mapping[str, float] | None = None,
        **kwargs: Any,
    ) -> None:
        if mask_id not in MASKS:
            raise ValueError(f"unknown E* mask {mask_id!r}")
        if not expanded and mask_id != "M000":
            raise ValueError("non-expanded adapter mode only supports M000")
        self.e_star_mask_id = str(mask_id)
        self.e_star_expanded = bool(expanded)
        self.e_star_node_capacities = _finite_nonnegative(
            node_capacities
            or {
                "wdc": 100_000.0,
                "al": 100_000.0,
                "sb": 100_000.0,
                "cssu_a": 50_000.0,
                "cssu_b": 50_000.0,
            },
            ("wdc", "al", "sb", "cssu_a", "cssu_b"),
            "node_capacities",
        )
        self.e_star_supplier_capacity = _finite_nonnegative(
            supplier_capacity
            or {lane: 25_000.0 for lane in SUPPLIER_LANES},
            SUPPLIER_LANES,
            "supplier_capacity",
        )
        self.e_star_source_stock_initial = _finite_nonnegative(
            source_stock
            or {lane: self.e_star_supplier_capacity[lane] * 100.0 for lane in SUPPLIER_LANES},
            SUPPLIER_LANES,
            "source_stock",
        )
        self.e_star_lead_times = _finite_nonnegative(
            e_star_lead_times
            or {
                "supplier_wdc": 24.0,
                "supplier_al": 48.0,
                "supplier_sb": 72.0,
                "sb_to_cssu_a": 24.0,
                "sb_to_cssu_b": 24.0,
            },
            (*SUPPLIER_LANES, "sb_to_cssu_a", "sb_to_cssu_b"),
            "e_star_lead_times",
        )
        self.e_star_transport_capacity = _finite_nonnegative(
            e_star_transport_capacity
            or {
                lane: 25_000.0
                for lane in (
                    *SUPPLIER_LANES,
                    "sb_to_cssu_a",
                    "sb_to_cssu_b",
                )
            },
            (*SUPPLIER_LANES, "sb_to_cssu_a", "sb_to_cssu_b"),
            "e_star_transport_capacity",
        )

        if self.e_star_expanded:
            kwargs = dict(kwargs)
            kwargs.setdefault("initial_buffers", {
                "op3_rm": 0.0,
                "op5_rm": 0.0,
                "op9_rations": 0.0,
            })
            kwargs.setdefault("inventory_replenishment_period", None)
            kwargs.setdefault("order_fulfillment_mode", "op9_linked")
            kwargs.setdefault("cssu_topology_mode", "split_v1")
            kwargs.setdefault("cssu_reallocate_unused", False)
            kwargs.setdefault(
                "cssu_storage_capacity",
                {
                    "A": self.e_star_node_capacities["cssu_a"],
                    "B": self.e_star_node_capacities["cssu_b"],
                },
            )
        super().__init__(**kwargs)

        self._e_star_targets = {node: 0.0 for node in ("wdc", "al", "sb", "cssu_a", "cssu_b")}
        self._e_star_source_stock = dict(self.e_star_source_stock_initial)
        self._e_star_transit: list[EStarTransitRecord] = []
        self._e_star_procurement_ordered = {lane: 0.0 for lane in SUPPLIER_LANES}
        self._e_star_procurement_received = {node: 0.0 for node in ("wdc", "al", "sb")}
        self._e_star_dispatch_sent = {lane: 0.0 for lane in ("sb_to_cssu_a", "sb_to_cssu_b")}
        self._e_star_dispatch_in_transit = {
            lane: 0.0 for lane in ("sb_to_cssu_a", "sb_to_cssu_b")
        }
        self._e_star_external_rations = 0.0
        self.e_star_action_events: list[dict[str, Any]] = []
        self.e_star_blocked_arrivals: dict[str, float] = {
            node: 0.0 for node in ("wdc", "al", "sb", "cssu_a", "cssu_b")
        }

        if self.e_star_expanded:
            # Parent containers are intentionally INF in the historical model.
            # The expanded adapter makes only the approved E* nodes finite.
            self.raw_material_wdc = self._replace_with_finite_container(
                self.raw_material_wdc,
                self.e_star_node_capacities["wdc"],
            )
            self.raw_material_al = self._replace_with_finite_container(
                self.raw_material_al,
                self.e_star_node_capacities["al"],
            )
            self.rations_sb = self._replace_with_finite_container(
                self.rations_sb,
                self.e_star_node_capacities["sb"],
            )

    def _replace_with_finite_container(self, original: Any, capacity: float):
        level = float(original.level)
        if level > float(capacity) + 1e-9:
            raise ValueError("historical initial inventory exceeds E* capacity")
        return simpy.Container(self.env, capacity=float(capacity), init=level)

    # ------------------------------------------------------------------
    # Historical process gates
    # ------------------------------------------------------------------
    def _disabled_e_star_process(self):
        while True:
            yield self.env.timeout(max(1.0, float(self.horizon) + 1.0))

    def _op2_supplier_delivery(self):
        if not self.e_star_expanded:
            yield from super()._op2_supplier_delivery()
            return
        yield from self._disabled_e_star_process()

    def _op9_sb_dispatch(self):
        if not self.e_star_expanded:
            yield from super()._op9_sb_dispatch()
            return
        yield from self._disabled_e_star_process()

    def _op9_daily_freight_dispatch(self):
        if not self.e_star_expanded:
            yield from super()._op9_daily_freight_dispatch()
            return
        yield from self._disabled_e_star_process()

    def _op10_transport_to_cssu(self):
        if not self.e_star_expanded:
            yield from super()._op10_transport_to_cssu()
            return
        yield from self._disabled_e_star_process()

    def _op12_transport_to_theatre(self):
        if not self.e_star_expanded:
            yield from super()._op12_transport_to_theatre()
            return
        yield from self._disabled_e_star_process()

    # ------------------------------------------------------------------
    # E* actions and source-conserving queues
    # ------------------------------------------------------------------
    def _container_for_destination(self, destination: str):
        return {
            "raw_material_wdc": self.raw_material_wdc,
            "raw_material_al": self.raw_material_al,
            "rations_sb": self.rations_sb,
        }[destination]

    def _current_node_level(self, node: str) -> float:
        if node == "wdc":
            return float(self.raw_material_wdc.level)
        if node == "al":
            return float(self.raw_material_al.level)
        if node == "sb":
            return float(self.rations_sb.level)
        if node in {"cssu_a", "cssu_b"}:
            return float(self.cssu_inventory[node[-1].upper()])
        raise KeyError(node)

    def _pending_claimant_qty(self, claimant: str) -> float:
        return float(sum(
            float(order.remaining_qty)
            for order in self.pending_backorders
            if order.cssu_destination == claimant
        ))

    def _validate_e_star_action(self, action: EStarAction) -> None:
        mask = MASKS[self.e_star_mask_id]
        procurement = _finite_nonnegative(action.procurement_qty, SUPPLIER_LANES, "procurement_qty")
        targets = _finite_nonnegative(
            action.buffer_targets,
            ("wdc", "al", "sb", "cssu_a", "cssu_b"),
            "buffer_targets",
        )
        dispatch_lanes = ("sb_to_cssu_a", "sb_to_cssu_b")
        dispatch = _finite_nonnegative(action.dispatch_qty, dispatch_lanes, "dispatch_qty")
        suppliers = set(action.active_supplier_lanes)
        dispatch_active = set(action.active_dispatch_lanes)
        if not suppliers <= set(SUPPLIER_LANES):
            raise ValueError("unknown active supplier lane")
        if not dispatch_active <= set(dispatch_lanes):
            raise ValueError("unknown active dispatch lane")
        if not mask["P"] and (suppliers or any(procurement.values())):
            raise ValueError(f"mask {self.e_star_mask_id} does not permit procurement")
        if not mask["D"] and (dispatch_active or any(dispatch.values())):
            raise ValueError(f"mask {self.e_star_mask_id} does not permit dispatch")
        if not mask["U"]:
            for node in ("wdc", "al", "sb"):
                if abs(targets[node] - self._e_star_targets[node]) > 1e-9:
                    raise ValueError(f"mask {self.e_star_mask_id} must carry target {node}")
        if any(procurement[lane] > 0.0 and lane not in suppliers for lane in SUPPLIER_LANES):
            raise ValueError("positive procurement requires its lane to be active")
        if any(dispatch[lane] > 0.0 and lane not in dispatch_active for lane in dispatch_lanes):
            raise ValueError("positive dispatch requires its lane to be active")
        for node, target in targets.items():
            if target > self.e_star_node_capacities[node] + 1e-9:
                raise ValueError(f"buffer target exceeds capacity for {node}")
        for lane, quantity in procurement.items():
            if quantity <= 0.0:
                continue
            if quantity > self._e_star_source_stock[lane] + 1e-9:
                raise ValueError(f"procurement exceeds source stock on {lane}")
            if quantity > self.e_star_supplier_capacity[lane] + 1e-9:
                raise ValueError(f"procurement exceeds supplier capacity on {lane}")
            if quantity > self.e_star_transport_capacity[lane] + 1e-9:
                raise ValueError(f"procurement exceeds transport capacity on {lane}")
        committed = sum(self._e_star_dispatch_in_transit.values())
        requested_dispatch = sum(dispatch.values())
        if requested_dispatch > float(self.rations_sb.level) - committed + 1e-9:
            raise ValueError("dispatch exceeds available SB inventory")
        planned_by_claimant = {"A": 0.0, "B": 0.0}
        for lane, quantity in dispatch.items():
            if quantity <= 0.0:
                continue
            claimant = "A" if lane.endswith("cssu_a") else "B"
            if quantity > self.e_star_transport_capacity[lane] + 1e-9:
                raise ValueError(f"dispatch exceeds transport capacity on {lane}")
            planned_by_claimant[claimant] += quantity
            if planned_by_claimant[claimant] > self._pending_claimant_qty(claimant) + 1e-9:
                raise ValueError("dispatch exceeds claimant shortfall")

    def _schedule_procurement(self, lane: str, quantity: float) -> None:
        node, destination = SUPPLIER_DESTINATIONS[lane]
        now = float(self.env.now)
        self._e_star_source_stock[lane] -= quantity
        self._e_star_procurement_ordered[lane] += quantity
        record = EStarTransitRecord(
            lane=lane,
            destination=destination,
            quantity=float(quantity),
            requested_at=now,
            due_at=now + float(self.e_star_lead_times[lane]),
        )
        self._e_star_transit.append(record)
        self.env.process(self._deliver_e_star_procurement(record))

    def _deliver_e_star_procurement(self, record: EStarTransitRecord):
        yield self.env.timeout(max(0.0, record.due_at - float(self.env.now)))
        container = self._container_for_destination(record.destination)
        node = SUPPLIER_DESTINATIONS[record.lane][0]
        while True:
            room = float(container.capacity) - float(container.level)
            if room + 1e-9 >= record.quantity:
                break
            self.e_star_blocked_arrivals[node] += record.quantity
            if float(self.env.now) >= float(self.horizon):
                return
            yield self.env.timeout(1.0)
        yield container.put(record.quantity)
        self._e_star_procurement_received[node] += record.quantity
        self._e_star_transit = [item for item in self._e_star_transit if item != record]
        if record.lane in {"supplier_wdc", "supplier_al"}:
            self.total_external_raw_material += record.quantity
        else:
            self._e_star_external_rations += record.quantity

    def _schedule_dispatch(self, lane: str, quantity: float) -> None:
        claimant = "A" if lane.endswith("cssu_a") else "B"
        self._e_star_dispatch_in_transit[lane] += quantity
        self.env.process(
            self._dispatch_e_star_orders_after_lead(
                lane,
                claimant,
                quantity,
                float(self.env.now) + float(self.e_star_lead_times[lane]),
            )
        )

    def _dispatch_e_star_orders_after_lead(
        self,
        lane: str,
        claimant: str,
        quantity: float,
        due_at: float,
    ):
        yield self.env.timeout(max(0.0, due_at - float(self.env.now)))
        self._e_star_dispatch_in_transit[lane] = max(
            0.0, self._e_star_dispatch_in_transit[lane] - float(quantity)
        )
        self._e_star_dispatch_sent[lane] += float(quantity)
        yield from self._dispatch_e_star_orders(claimant, quantity)

    def _dispatch_e_star_orders(self, claimant: str, quantity: float):
        remaining_budget = float(quantity)
        orders = sorted(
            [
                order
                for order in self.pending_backorders
                if order.cssu_destination == claimant and float(order.remaining_qty) > 1e-9
            ],
            key=self._cssu_order_key,
        )
        for order in orders:
            if remaining_budget <= 1e-9:
                break
            qty = min(remaining_budget, float(order.remaining_qty))
            yield self.rations_sb.get(qty)
            self._record_material_availability("order_release", qty)
            order.remaining_qty -= qty
            order.in_flight_qty += qty
            if order.op9_release_time is None:
                order.op9_release_time = float(self.env.now)
                order.causal_wait_hours["op9_release"] = max(
                    0.0, float(self.env.now) - float(order.OPTj)
                )
            self.cssu_dispatched[claimant] += qty
            self.env.process(self._deliver_order_from_op9(order, qty))
            remaining_budget -= qty
            if order.remaining_qty <= 1e-9:
                order.remaining_qty = 0.0
                self._remove_pending_backorder(order)
        self._refresh_pending_backorder_qty()

    def apply_e_star_action(self, action: EStarAction) -> dict[str, Any]:
        if not self.e_star_expanded:
            raise EStarBridgeNotReady("M000 adapter delegates to historical step()")
        self._validate_e_star_action(action)
        for node, target in action.buffer_targets.items():
            self._e_star_targets[node] = float(target)
            if node in NODE_TO_TARGET:
                self.inventory_buffer_targets[NODE_TO_TARGET[node]] = float(target)
        for lane, quantity in action.procurement_qty.items():
            if quantity > 0.0:
                self._schedule_procurement(lane, quantity)
        for lane, quantity in action.dispatch_qty.items():
            if quantity > 0.0:
                self._schedule_dispatch(lane, quantity)
        event = {
            "time": float(self.env.now),
            "action": action.canonical(),
            "targets": dict(self._e_star_targets),
        }
        self.e_star_action_events.append(event)
        return event

    def step_e_star(
        self, action: EStarAction, *, step_hours: float | None = None
    ) -> tuple[Any, float, bool, dict[str, Any]]:
        self.apply_e_star_action(action)
        observation, reward, done, info = super().step(action=None, step_hours=step_hours)
        info = dict(info)
        info["e_star"] = self.e_star_evidence()
        return observation, reward, done, info

    def flow_ledger(self) -> dict[str, float]:
        base = dict(super().flow_ledger())
        if self.e_star_expanded:
            base["ration_sources"] = float(base.get("ration_sources", 0.0)) + self._e_star_external_rations
            base["ration_residual"] = float(base.get("ration_residual", 0.0)) + self._e_star_external_rations
        return base

    def e_star_evidence(self) -> dict[str, Any]:
        return {
            "mask_id": self.e_star_mask_id,
            "time": float(self.env.now),
            "targets": dict(self._e_star_targets),
            "source_stock_initial": dict(self.e_star_source_stock_initial),
            "source_stock_remaining": dict(self._e_star_source_stock),
            "procurement_ordered": dict(self._e_star_procurement_ordered),
            "procurement_received": dict(self._e_star_procurement_received),
            "dispatch_sent": dict(self._e_star_dispatch_sent),
            "dispatch_in_transit": dict(self._e_star_dispatch_in_transit),
            "external_rations": float(self._e_star_external_rations),
            "in_transit": [asdict(item) for item in self._e_star_transit],
            "blocked_arrivals": dict(self.e_star_blocked_arrivals),
            "flow_ledger": self.flow_ledger(),
            "actions": list(self.e_star_action_events),
        }

    def e_star_payload_sha256(self) -> str:
        payload = self.e_star_evidence()
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
        ).hexdigest()
