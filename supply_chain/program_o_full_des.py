"""Minimal two-product Program O extension of the full Op1--Op13 DES.

The aggregate SimPy containers remain the only physical mass.  This module
adds an auditable FIFO metadata partition for two nonfungible finished products
and a frozen weekly production-target contract.  No setup, product-specific
BOM, processing-time, risk, expiry, or substitution physics is introduced in
the primary cell.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from supply_chain.episode_metrics import compute_episode_metrics
from supply_chain.supply_chain import (
    DEMAND,
    HOURS_PER_WEEK,
    MFSCSimulation,
    OrderRecord,
)

PRODUCTS = ("P_C", "P_H")
PRODUCT_SET = frozenset(PRODUCTS)
DOWNSTREAM_FREIGHT_PHYSICS_MODES = frozenset(
    {"loaded_only", "fixed_clock_physical_v1"}
)


def _json_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass
class ProgramOOrderRecord(OrderRecord):
    requested_product_id: str = "P_C"
    supplied_product_quantities: dict[str, float] = field(default_factory=dict)


@dataclass
class ProductSlice:
    product_id: str
    quantity: float
    lot_id: str
    created_at: float


@dataclass
class ProductTarget:
    slot_id: str
    product_id: str
    remaining: float
    source: str
    committed_at: float
    activated_at: float
    started_at: float | None = None
    completed_at: float | None = None


class ProductTagLedger:
    """FIFO metadata partition mirroring aggregate physical nodes."""

    NODE_NAMES = (
        "pending_batch",
        "rework_op6",
        "rations_al",
        "op8_transit",
        "rations_sb",
        "order_transit",
        "delivered",
        "scrap",
    )

    def __init__(self) -> None:
        self.nodes: dict[str, list[ProductSlice]] = {
            node: [] for node in self.NODE_NAMES
        }
        self.produced = {product_id: 0.0 for product_id in PRODUCTS}

    def put(
        self,
        node: str,
        product_id: str,
        quantity: float,
        *,
        lot_id: str,
        created_at: float,
    ) -> None:
        if node not in self.nodes:
            raise KeyError(node)
        if product_id not in PRODUCT_SET:
            raise ValueError(product_id)
        qty = float(quantity)
        if qty <= 1e-12:
            return
        queue = self.nodes[node]
        if queue and queue[-1].product_id == product_id and queue[-1].lot_id == lot_id:
            queue[-1].quantity += qty
            return
        queue.append(ProductSlice(product_id, qty, str(lot_id), float(created_at)))

    def quantity(self, node: str, product_id: str | None = None) -> float:
        return float(
            sum(
                item.quantity
                for item in self.nodes[node]
                if product_id is None or item.product_id == product_id
            )
        )

    def _take(
        self, node: str, quantity: float, *, product_id: str | None
    ) -> list[ProductSlice]:
        remaining = float(quantity)
        if remaining < -1e-9:
            raise ValueError("quantity must be non-negative")
        consumed: list[ProductSlice] = []
        queue = self.nodes[node]
        index = 0
        while remaining > 1e-8 and index < len(queue):
            item = queue[index]
            if product_id is not None and item.product_id != product_id:
                index += 1
                continue
            used = min(remaining, float(item.quantity))
            consumed.append(
                ProductSlice(item.product_id, used, item.lot_id, item.created_at)
            )
            item.quantity -= used
            remaining -= used
            if item.quantity <= 1e-8:
                queue.pop(index)
            else:
                index += 1
        if remaining > 1e-6:
            qualifier = product_id or "any product"
            raise AssertionError(
                f"product metadata underflow at {node}: {remaining} of "
                f"{quantity} missing for {qualifier}"
            )
        return consumed

    def take_fifo(self, node: str, quantity: float) -> list[ProductSlice]:
        return self._take(node, quantity, product_id=None)

    def take_product(
        self, node: str, product_id: str, quantity: float
    ) -> list[ProductSlice]:
        return self._take(node, quantity, product_id=product_id)

    def put_slices(
        self, node: str, slices: Iterable[ProductSlice], *, now: float
    ) -> None:
        for item in slices:
            self.put(
                node,
                item.product_id,
                item.quantity,
                lot_id=item.lot_id,
                created_at=float(now),
            )

    def remove_tokens(self, node: str, tokens: Iterable[ProductSlice]) -> None:
        for token in tokens:
            remaining = float(token.quantity)
            queue = self.nodes[node]
            index = 0
            while remaining > 1e-8 and index < len(queue):
                item = queue[index]
                if item.product_id != token.product_id or item.lot_id != token.lot_id:
                    index += 1
                    continue
                used = min(remaining, float(item.quantity))
                item.quantity -= used
                remaining -= used
                if item.quantity <= 1e-8:
                    queue.pop(index)
                else:
                    index += 1
            if remaining > 1e-6:
                raise AssertionError(
                    f"missing token {token.lot_id}/{token.product_id}: {remaining}"
                )

    def snapshot(self) -> dict[str, dict[str, float]]:
        return {
            node: {
                product_id: self.quantity(node, product_id) for product_id in PRODUCTS
            }
            for node in self.NODE_NAMES
        }


def product_demand_tape(
    seed: int,
    *,
    regime_persistence: float,
    dominant_share: float,
    weeks: int = 8,
    initial_regime: str | None = None,
) -> dict[str, Any]:
    """Reproduce the frozen Program O event-keyed two-product tape."""
    rng = np.random.default_rng(int(seed))
    if initial_regime is not None and str(initial_regime) not in PRODUCTS:
        raise ValueError("initial_regime must be P_C, P_H, or None")
    regime = (
        str(initial_regime)
        if initial_regime is not None
        else ("P_C" if int(rng.integers(0, 2)) == 0 else "P_H")
    )
    regimes: list[str] = []
    labels: list[str] = []
    if int(weeks) <= 0 or int(weeks) > 8:
        raise ValueError("weeks must be in 1..8")
    for _week in range(int(weeks)):
        regimes.append(regime)
        for _day in range(6):
            dominant = bool(rng.random() < float(dominant_share))
            labels.append(regime if dominant else PRODUCTS[1 - PRODUCTS.index(regime)])
        if rng.random() > float(regime_persistence):
            regime = PRODUCTS[1 - PRODUCTS.index(regime)]
    raw = {
        "seed": int(seed),
        "regimes": regimes,
        "order_products": labels,
    }
    # Preserve Program Q's historical tape payload and digest byte-for-byte
    # when no researcher extension is requested.
    if initial_regime is not None:
        raw["initial_regime"] = str(initial_regime)
    payload = {
        **raw,
        "regime_persistence": float(regime_persistence),
        "dominant_share": float(dominant_share),
        "sha256": _json_digest(raw),
    }
    return payload


class ProgramOFullDESSimulation(MFSCSimulation):
    """Opt-in product-tagged full DES under the Program O H_PI contract."""

    def __init__(
        self,
        *,
        seed: int,
        calendar: Sequence[int],
        scheduler: Mapping[str, Sequence[str]],
        regime_persistence: float,
        dominant_share: float,
        complete_substitution: bool = False,
        activation_delay_hours: float = 24.0,
        demand_offsets_hours: Sequence[float] = (30, 54, 78, 102, 126, 150),
        clearance_hours: float = 1344.0,
        downstream_freight_physics_mode: str = "loaded_only",
        risks_enabled: bool = False,
        enabled_risks: set[str] | None = None,
        risk_frequency_multipliers_by_id: dict[str, float] | None = None,
        risk_impact_multipliers_by_id: dict[str, float] | None = None,
        risk_event_tape: Iterable[dict[str, Any]] | None = None,
        risk_rng_mode: str = "shared",
        initial_regime: str | None = None,
    ) -> None:
        # Relevant-risk pass-through (contract program_o_relevant_risk_sensitivity_v1):
        # defaults reproduce the historical risks-off physics BIT-EXACTLY; when enabled, the
        # parent's fidelity-verified risk machinery (exact thinning per risk id) drives events.
        normalized_scheduler = {
            int(action): tuple(str(product_id) for product_id in labels)
            for action, labels in scheduler.items()
        }
        if not normalized_scheduler or set(normalized_scheduler) != set(
            range(len(normalized_scheduler))
        ) or any(
            len(labels) != 3 or not set(labels).issubset(PRODUCT_SET)
            for labels in normalized_scheduler.values()
        ):
            raise ValueError(
                "scheduler must map consecutive actions from zero to three products"
            )
        if not 1 <= len(calendar) <= 8 or any(
            int(value) not in normalized_scheduler for value in calendar
        ):
            raise ValueError("calendar contains an action absent from scheduler")
        if len(demand_offsets_hours) != 6:
            raise ValueError("six weekly demand offsets are required")
        if downstream_freight_physics_mode not in DOWNSTREAM_FREIGHT_PHYSICS_MODES:
            raise ValueError(
                "downstream_freight_physics_mode must be loaded_only or "
                "fixed_clock_physical_v1"
            )

        super().__init__(
            shifts=1,
            seed=int(seed),
            horizon=20_000.0,
            risks_enabled=bool(risks_enabled),
            enabled_risks=enabled_risks,
            risk_frequency_multipliers_by_id=risk_frequency_multipliers_by_id,
            risk_impact_multipliers_by_id=risk_impact_multipliers_by_id,
            risk_event_tape=risk_event_tape,
            risk_rng_mode=str(risk_rng_mode),
            stochastic_pt=False,
            deterministic_baseline=False,
            warmup_trigger="op9_arrival",
            raw_material_flow_mode="kit_equivalent_order_up_to",
            procurement_contract_mode="causal_coupled",
            replenishment_route_aware=True,
            order_fulfillment_mode="op9_linked",
            assembly_flow_mode="aggregate_line",
            op9_dispatch_policy="fixed_clock_daily",
            downstream_transport_capacity_mode="tandem_capacity_one",
            op8_dispatch_mode="thesis_full_batch",
            strict_exogenous_crn=True,
            demand_start_after_warmup=True,
        )
        self.program_o_calendar = tuple(int(value) for value in calendar)
        self.program_o_decision_weeks = len(self.program_o_calendar)
        self.program_o_scheduler = normalized_scheduler
        self.program_o_complete_substitution = bool(complete_substitution)
        self.program_o_activation_delay_hours = float(activation_delay_hours)
        self.program_o_demand_offsets_hours = tuple(map(float, demand_offsets_hours))
        self.program_o_clearance_hours = float(clearance_hours)
        self.program_o_downstream_freight_physics_mode = str(
            downstream_freight_physics_mode
        )
        self.program_o_tape = product_demand_tape(
            int(seed),
            regime_persistence=float(regime_persistence),
            dominant_share=float(dominant_share),
            weeks=self.program_o_decision_weeks,
            initial_regime=initial_regime,
        )
        self.program_o_ledger = ProductTagLedger()
        self.program_o_target_queue: deque[ProductTarget] = deque()
        self.program_o_active_target: ProductTarget | None = None
        self.program_o_target_events: list[dict[str, Any]] = []
        self.program_o_product_events: list[dict[str, Any]] = []
        self.program_o_action_events: list[dict[str, Any]] = []
        self.program_o_order_route_events: list[dict[str, Any]] = []
        self.program_o_decision_start: float | None = None
        self.program_o_prefix_state_hash: str | None = None
        self.program_o_committed_action_slots = 0
        self.program_o_completed_action_slots = 0
        self.program_o_actual_loaded_departures = 0
        self.program_o_actual_payload = 0.0
        self.program_o_actual_downstream_vehicle_hours = 0.0
        self.program_o_scheduled_downstream_missions = 0
        self.program_o_empty_downstream_missions = 0
        self.program_o_scheduled_downstream_vehicle_hours = 0.0
        self.program_o_scheduled_downstream_crew_hours = 0.0
        self.program_o_scheduled_payload_capacity = 0.0
        self.program_o_downstream_mission_events: list[dict[str, Any]] = []
        self.program_o_warmup_event = self.env.event()
        self._program_o_processes_started = False
        self._queue_targets(
            ("P_C", "P_H"),
            source="neutral_prefix",
            committed_at=0.0,
            activated_at=0.0,
            week=-1,
        )

    def _queue_targets(
        self,
        labels: Sequence[str],
        *,
        source: str,
        committed_at: float,
        activated_at: float,
        week: int,
    ) -> None:
        for position, product_id in enumerate(labels):
            if product_id not in PRODUCT_SET:
                raise ValueError(product_id)
            slot_id = f"{source}:w{week}:s{position}"
            target = ProductTarget(
                slot_id=slot_id,
                product_id=product_id,
                remaining=float(self.params["batch_size"]),
                source=str(source),
                committed_at=float(committed_at),
                activated_at=float(activated_at),
            )
            self.program_o_target_queue.append(target)
            self.program_o_target_events.append(dict(asdict(target), event="activated"))

    def _ensure_active_target(self) -> ProductTarget | None:
        if self.program_o_active_target is None and self.program_o_target_queue:
            target = self.program_o_target_queue.popleft()
            target.started_at = float(self.env.now)
            self.program_o_active_target = target
            self.program_o_target_events.append(dict(asdict(target), event="started"))
        return self.program_o_active_target

    def _assembly_output_capacity(self, nominal_capacity: float) -> float:
        target = self._ensure_active_target()
        if target is None:
            return 0.0
        available_target_quantity = float(target.remaining) + sum(
            float(item.remaining) for item in self.program_o_target_queue
        )
        return min(float(nominal_capacity), available_target_quantity)

    def _record_assembly_product_output(self, quantity: float) -> None:
        remaining = float(quantity)
        while remaining > 1e-8:
            target = self._ensure_active_target()
            if target is None:
                raise AssertionError(
                    "physical production has no committed product target"
                )
            used = min(remaining, float(target.remaining))
            self.program_o_ledger.put(
                "pending_batch",
                target.product_id,
                used,
                lot_id=target.slot_id,
                created_at=float(self.env.now),
            )
            self.program_o_ledger.produced[target.product_id] += used
            target.remaining -= used
            remaining -= used
            if target.remaining <= 1e-8:
                target.remaining = 0.0
                target.completed_at = float(self.env.now)
                self.program_o_target_events.append(
                    dict(asdict(target), event="production_completed")
                )
                if target.source == "policy":
                    self.program_o_completed_action_slots += 1
                self.program_o_active_target = None

    def _record_rework_product_output(self, quantity: float) -> None:
        slices = self.program_o_ledger.take_fifo("rework_op6", float(quantity))
        self.program_o_ledger.put_slices("pending_batch", slices, now=self.env.now)
        self.program_o_product_events.append(
            {
                "time": float(self.env.now),
                "event": "r14_rework_returned_to_pending_batch",
                "quantity": float(quantity),
                "tokens": [asdict(item) for item in slices],
            }
        )

    def _record_product_rework_started(self, quantity: float) -> None:
        slices = self.program_o_ledger.take_fifo("pending_batch", float(quantity))
        if self.r14_defect_mode == "thesis_strict_op6":
            target_node = "rework_op6"
        elif self.r14_defect_mode == "discard":
            target_node = "scrap"
        else:
            raise ValueError(
                "Program O product tagging supports R14 thesis_strict_op6 or discard; "
                "generic raw-material reprocess would erase committed product identity"
            )
        self.program_o_ledger.put_slices(target_node, slices, now=self.env.now)
        self.program_o_product_events.append(
            {
                "time": float(self.env.now),
                "event": "r14_product_rework_started",
                "quantity": float(quantity),
                "target_node": target_node,
                "tokens": [asdict(item) for item in slices],
            }
        )

    def _stage_product_metadata(self, quantity: float) -> None:
        slices = self.program_o_ledger.take_fifo("pending_batch", quantity)
        products = {item.product_id for item in slices}
        if len(products) != 1:
            raise AssertionError("one physical finished batch spans product targets")
        self.program_o_ledger.put_slices("rations_al", slices, now=self.env.now)
        self.program_o_product_events.append(
            {
                "time": float(self.env.now),
                "event": "op7_staged",
                "quantity": float(quantity),
                "tokens": [asdict(item) for item in slices],
            }
        )

    def _take_op8_product_metadata(self, quantity: float) -> list[ProductSlice]:
        slices = self.program_o_ledger.take_fifo("rations_al", quantity)
        self.program_o_ledger.put_slices("op8_transit", slices, now=self.env.now)
        return slices

    def _arrive_op8_product_metadata(
        self, token: list[ProductSlice], quantity: float
    ) -> None:
        if abs(sum(item.quantity for item in token) - float(quantity)) > 1e-8:
            raise AssertionError("Op8 product token does not match physical quantity")
        self.program_o_ledger.remove_tokens("op8_transit", token)
        self.program_o_ledger.put_slices("rations_sb", token, now=self.env.now)
        self.program_o_product_events.append(
            {
                "time": float(self.env.now),
                "event": "op8_arrived_sb",
                "quantity": float(quantity),
                "tokens": [asdict(item) for item in token],
                "product_inventory_after": {
                    product_id: self.program_o_ledger.quantity("rations_sb", product_id)
                    for product_id in PRODUCTS
                },
            }
        )

    def _op8_warmup_ready(self) -> bool:
        batch = float(self.params["batch_size"])
        return all(
            self.program_o_ledger.quantity("rations_sb", product_id) + 1e-8 >= batch
            for product_id in PRODUCTS
        )

    def _mark_warmup_complete(self) -> None:
        was_complete = bool(self.warmup_complete)
        super()._mark_warmup_complete()
        if not was_complete and self.warmup_complete:
            self.program_o_decision_start = float(self.warmup_time)
            self.program_o_prefix_state_hash = self.aggregate_state_hash()
            if not self.program_o_warmup_event.triggered:
                self.program_o_warmup_event.succeed(float(self.warmup_time))

    def _activate_week(self, week: int, action: int, requested_at: float):
        yield self.env.timeout(self.program_o_activation_delay_hours)
        labels = self.program_o_scheduler[int(action)]
        self._queue_targets(
            labels,
            source="policy",
            committed_at=float(requested_at),
            activated_at=float(self.env.now),
            week=int(week),
        )
        self.program_o_action_events.append(
            {
                "week": int(week),
                "action": int(action),
                "requested_at": float(requested_at),
                "activated_at": float(self.env.now),
                "labels": list(labels),
                "status": "activated",
            }
        )

    def _program_o_action_controller(self):
        yield self.program_o_warmup_event
        for week, action in enumerate(self.program_o_calendar):
            if week:
                yield self.env.timeout(float(HOURS_PER_WEEK))
            requested_at = float(self.env.now)
            self.program_o_committed_action_slots += 3
            self.program_o_action_events.append(
                {
                    "week": int(week),
                    "action": int(action),
                    "requested_at": requested_at,
                    "effective_at": requested_at
                    + self.program_o_activation_delay_hours,
                    "status": "requested",
                }
            )
            self.env.process(self._activate_week(week, action, requested_at))

    def _op13_demand(self):
        yield self.program_o_warmup_event
        start = float(self.program_o_decision_start or self.env.now)
        labels = tuple(self.program_o_tape["order_products"])
        order_num = 0
        for week in range(self.program_o_decision_weeks):
            for day, offset in enumerate(self.program_o_demand_offsets_hours):
                target_time = start + week * float(HOURS_PER_WEEK) + float(offset)
                if target_time > self.env.now:
                    yield self.env.timeout(target_time - float(self.env.now))
                demand_qty, is_contingent, causal_ids = (
                    self._sample_calendar_demand_quantity()
                )
                product_id = labels[week * 6 + day]
                self.total_demanded += demand_qty
                order_num += 1
                order = ProgramOOrderRecord(
                    j=order_num,
                    OPTj=float(self.env.now),
                    quantity=float(demand_qty),
                    remaining_qty=float(demand_qty),
                    contingent=bool(is_contingent),
                    causal_r24_event_ids=causal_ids,
                    requested_product_id=product_id,
                )
                yield from self._place_demand_order(order)

    def _select_op9_dispatch_order(self) -> OrderRecord | None:
        if not self.pending_backorders:
            return None
        if self.program_o_complete_substitution:
            head = self.pending_backorders[0]
            return (
                head
                if self.program_o_ledger.quantity("rations_sb") + 1e-9
                >= float(head.remaining_qty)
                else None
            )
        for order in self.pending_backorders:
            product_id = str(getattr(order, "requested_product_id", ""))
            if product_id in PRODUCT_SET and self.program_o_ledger.quantity(
                "rations_sb", product_id
            ) + 1e-9 >= float(order.remaining_qty):
                return order
        return None

    def _reserve_op9_order_stock(self, order: OrderRecord, quantity: float):
        qty = float(quantity)
        # Reserve the product partition before yielding the immediately
        # satisfiable aggregate Container.get event.  Otherwise another
        # same-time Op8 callback can mutate the metadata partition between the
        # physical and product reservations, defeating the frozen atomic move.
        if self.program_o_complete_substitution:
            slices = self.program_o_ledger.take_fifo("rations_sb", qty)
        else:
            slices = self.program_o_ledger.take_product(
                "rations_sb", str(getattr(order, "requested_product_id")), qty
            )
        inventory_after_reservation = {
            product_id: self.program_o_ledger.quantity("rations_sb", product_id)
            for product_id in PRODUCTS
        }
        yield from super()._reserve_op9_order_stock(order, quantity)
        self.program_o_ledger.put_slices("order_transit", slices, now=self.env.now)
        setattr(order, "_program_o_supply_tokens", slices)
        supplied: dict[str, float] = {}
        for item in slices:
            supplied[item.product_id] = supplied.get(item.product_id, 0.0) + float(
                item.quantity
            )
        setattr(order, "supplied_product_quantities", supplied)
        self.program_o_actual_loaded_departures += 1
        self.program_o_actual_payload += qty
        self.program_o_actual_downstream_vehicle_hours += 48.0
        self.program_o_order_route_events.append(
            {
                "time": float(self.env.now),
                "order_j": int(order.j),
                "event": "op9_reserved",
                "quantity": qty,
                "oat_is_none": order.OATj is None,
                "product_inventory_after": inventory_after_reservation,
            }
        )

    def _program_o_fixed_clock_window_is_open(self) -> bool:
        if self.program_o_decision_start is None:
            return False
        start = float(self.program_o_decision_start)
        stop = (
            start
            + float(self.program_o_decision_weeks) * float(HOURS_PER_WEEK)
            + float(self.program_o_clearance_hours)
        )
        return start <= float(self.env.now) < stop

    def _program_o_empty_downstream_mission(self, mission_id: int):
        """Occupy the real Op10/Op12 convoy servers for an empty daily mission."""
        started_at = float(self.env.now)
        event = {
            "mission_id": int(mission_id),
            "kind": "empty",
            "scheduled_at": started_at,
            "op10_started_at": None,
            "op10_completed_at": None,
            "op12_started_at": None,
            "op12_completed_at": None,
        }
        self.program_o_downstream_mission_events.append(event)
        with self.op10_convoy.request() as request:
            yield request
            event["op10_started_at"] = float(self.env.now)
            yield self.env.timeout(float(self.params["op10_pt"]))
            event["op10_completed_at"] = float(self.env.now)
        with self.op12_convoy.request() as request:
            yield request
            event["op12_started_at"] = float(self.env.now)
            yield self.env.timeout(float(self.params["op12_pt"]))
            event["op12_completed_at"] = float(self.env.now)

    def _dispatch_one_op9_order_if_ready(self, *, next_check_hours: float = 24.0):
        """Execute one scheduled downstream mission, loaded or empty, when enabled."""
        dispatched = yield from super()._dispatch_one_op9_order_if_ready(
            next_check_hours=next_check_hours
        )
        if (
            self.program_o_downstream_freight_physics_mode
            != "fixed_clock_physical_v1"
            or not self._program_o_fixed_clock_window_is_open()
        ):
            return dispatched

        self.program_o_scheduled_downstream_missions += 1
        mission_id = self.program_o_scheduled_downstream_missions
        route_hours = float(self.params["op10_pt"]) + float(self.params["op12_pt"])
        self.program_o_scheduled_downstream_vehicle_hours += route_hours
        self.program_o_scheduled_downstream_crew_hours += route_hours
        self.program_o_scheduled_payload_capacity += float(DEMAND["b"])
        if dispatched:
            self.program_o_downstream_mission_events.append(
                {
                    "mission_id": int(mission_id),
                    "kind": "loaded",
                    "scheduled_at": float(self.env.now),
                }
            )
        else:
            self.program_o_empty_downstream_missions += 1
            self.env.process(self._program_o_empty_downstream_mission(mission_id))
        return dispatched

    def _on_order_physical_delivery(self, order: OrderRecord, quantity: float) -> None:
        tokens = list(getattr(order, "_program_o_supply_tokens", []))
        if abs(sum(item.quantity for item in tokens) - float(quantity)) > 1e-8:
            raise AssertionError("order product tokens do not match delivery quantity")
        self.program_o_ledger.remove_tokens("order_transit", tokens)
        self.program_o_ledger.put_slices("delivered", tokens, now=self.env.now)
        setattr(order, "_program_o_supply_tokens", [])
        self.program_o_order_route_events.append(
            {
                "time": float(self.env.now),
                "order_j": int(order.j),
                "event": "post_op12_physical_delivery",
                "quantity": float(quantity),
                "oat_is_none_before_finalize": order.OATj is None,
            }
        )

    def _start_processes(self) -> None:
        if self._program_o_processes_started:
            return
        super()._start_processes()
        self.env.process(self._program_o_action_controller())
        self._program_o_processes_started = True

    def run_contract(self) -> "ProgramOFullDESSimulation":
        self._start_processes()
        self.env.run(until=self.program_o_warmup_event)
        if self.program_o_decision_start is None:
            raise RuntimeError("Program O warm-up did not establish both product lots")
        score_time = (
            float(self.program_o_decision_start)
            + float(self.program_o_decision_weeks) * float(HOURS_PER_WEEK)
            + float(self.program_o_clearance_hours)
        )
        self.horizon = score_time
        self.env.run(until=score_time)
        return self

    def aggregate_state_hash(self) -> str:
        orders = [
            {
                "j": int(order.j),
                "OPTj": float(order.OPTj),
                "Q": float(order.quantity),
                "OATj": None if order.OATj is None else float(order.OATj),
                "lost": bool(order.lost),
            }
            for order in self.orders
        ]
        payload = {
            "time": float(self.env.now),
            "inventory": self._inventory_detail(),
            "pending_batch": float(self._pending_batch),
            "orders": orders,
            "daily_production": list(self.daily_production),
            "delivery_events": list(self.delivery_events),
            "flow_ledger": self.flow_ledger(),
        }
        return _json_digest(payload)

    def product_conservation_ledger(self) -> dict[str, Any]:
        stock_nodes = [
            "pending_batch",
            "rations_al",
            "op8_transit",
            "rations_sb",
            "order_transit",
            "delivered",
            "scrap",
        ]
        if (
            self.program_o_ledger.quantity("rework_op6") > 1e-12
            or float(self.rework_op6.level) > 1e-12
            or any(
                event.get("event") == "r14_product_rework_started"
                for event in self.program_o_product_events
            )
        ):
            stock_nodes.insert(1, "rework_op6")
        per_product: dict[str, Any] = {}
        for product_id in PRODUCTS:
            nodes = {
                node: self.program_o_ledger.quantity(node, product_id)
                for node in stock_nodes
            }
            residual = float(self.program_o_ledger.produced[product_id]) - sum(
                nodes.values()
            )
            per_product[product_id] = {
                "produced": float(self.program_o_ledger.produced[product_id]),
                "nodes": nodes,
                "residual": residual,
            }
        partitions = {
            "pending_batch": self.program_o_ledger.quantity("pending_batch")
            - float(self._pending_batch),
            "rations_al": self.program_o_ledger.quantity("rations_al")
            - float(self.rations_al.level),
            "op8_transit": self.program_o_ledger.quantity("op8_transit")
            - max(
                0.0,
                float(self._in_transit)
                - self.program_o_ledger.quantity("order_transit"),
            ),
            "rations_sb": self.program_o_ledger.quantity("rations_sb")
            - float(self.rations_sb.level),
            "order_transit": self.program_o_ledger.quantity("order_transit")
            - sum(float(order.in_flight_qty) for order in self.orders),
        }
        if "rework_op6" in stock_nodes:
            partitions["rework_op6"] = self.program_o_ledger.quantity(
                "rework_op6"
            ) - float(self.rework_op6.level)
        return {
            "per_product": per_product,
            "partition_residuals": partitions,
            "max_abs_product_residual": max(
                abs(float(row["residual"])) for row in per_product.values()
            ),
            "max_abs_partition_residual": max(
                abs(value) for value in partitions.values()
            ),
            "aggregate_flow_ledger": self.flow_ledger(),
        }

    def program_o_resource_ledger(self) -> dict[str, Any]:
        treatment_days = (
            float(self.program_o_decision_weeks) * float(HOURS_PER_WEEK)
            + self.program_o_clearance_hours
        ) / 24.0
        charged_slots = treatment_days
        return {
            "committed_action_batch_slots": float(
                self.program_o_committed_action_slots
            ),
            "completed_action_batch_slots": float(
                self.program_o_completed_action_slots
            ),
            "gross_action_production_quantity": float(
                self.program_o_committed_action_slots * self.params["batch_size"]
            ),
            "actual_loaded_departures": float(self.program_o_actual_loaded_departures),
            "actual_payload": float(self.program_o_actual_payload),
            "actual_downstream_vehicle_hours": float(
                self.program_o_actual_downstream_vehicle_hours
            ),
            "charged_daily_dispatch_slots": float(charged_slots),
            "charged_downstream_vehicle_hours": float(charged_slots * 48.0),
            "downstream_freight_physics_mode": self.program_o_downstream_freight_physics_mode,
            "scheduled_downstream_missions": float(
                self.program_o_scheduled_downstream_missions
            ),
            "empty_downstream_missions": float(
                self.program_o_empty_downstream_missions
            ),
            "scheduled_downstream_vehicle_hours": float(
                self.program_o_scheduled_downstream_vehicle_hours
            ),
            "scheduled_downstream_crew_hours": float(
                self.program_o_scheduled_downstream_crew_hours
            ),
            "scheduled_payload_capacity": float(
                self.program_o_scheduled_payload_capacity
            ),
            "setup_hours": 0.0,
        }

    def product_outcome_panel(self) -> dict[str, Any]:
        metrics = compute_episode_metrics(
            self,
            treatment_start=float(self.program_o_decision_start or 0.0),
            ret_excel_contract_version="ret_excel_request_snapshot_v2",
        )
        orders = [
            order
            for order in self.orders
            if not order.metrics_excluded
            and float(order.OPTj) >= float(self.program_o_decision_start or 0.0)
        ]
        by_product: dict[str, Any] = {}
        for product_id in PRODUCTS:
            subset = [
                order
                for order in orders
                if getattr(order, "requested_product_id", None) == product_id
            ]
            demanded = sum(float(order.quantity) for order in subset)
            completed = sum(
                float(order.quantity) for order in subset if order.OATj is not None
            )
            unresolved = sum(
                float(order.remaining_qty) + float(order.in_flight_qty)
                for order in subset
                if order.OATj is None
            )
            ages = [
                max(0.0, float(self.env.now) - float(order.OPTj))
                for order in subset
                if order.OATj is None
            ]
            by_product[product_id] = {
                "orders": len(subset),
                "demanded_quantity": demanded,
                "completed_quantity": completed,
                "unresolved_quantity": unresolved,
                "lost_orders": sum(bool(order.lost) for order in subset),
                "fill": completed / demanded if demanded > 0.0 else 1.0,
                "max_backlog_age": max(ages) if ages else 0.0,
            }
        conservation = self.product_conservation_ledger()
        return {
            "metrics": metrics,
            "products": by_product,
            "worst_product_fill": min(row["fill"] for row in by_product.values()),
            "resources": self.program_o_resource_ledger(),
            "conservation": conservation,
            "tape_sha256": self.program_o_tape["sha256"],
            "prefix_state_hash": self.program_o_prefix_state_hash,
            "aggregate_state_hash": self.aggregate_state_hash(),
        }


def run_program_o_full_des_episode(
    *,
    seed: int,
    calendar: Sequence[int],
    scheduler: Mapping[str, Sequence[str]],
    regime_persistence: float,
    dominant_share: float,
    complete_substitution: bool = False,
    downstream_freight_physics_mode: str = "loaded_only",
    risks_enabled: bool = False,
    enabled_risks: set[str] | None = None,
    risk_frequency_multipliers_by_id: dict[str, float] | None = None,
    risk_impact_multipliers_by_id: dict[str, float] | None = None,
    risk_event_tape: Iterable[dict[str, Any]] | None = None,
    risk_rng_mode: str = "shared",
    initial_regime: str | None = None,
) -> tuple[ProgramOFullDESSimulation, dict[str, Any]]:
    sim = ProgramOFullDESSimulation(
        seed=int(seed),
        calendar=calendar,
        scheduler=scheduler,
        regime_persistence=float(regime_persistence),
        dominant_share=float(dominant_share),
        complete_substitution=bool(complete_substitution),
        downstream_freight_physics_mode=str(downstream_freight_physics_mode),
        risks_enabled=bool(risks_enabled),
        enabled_risks=enabled_risks,
        risk_frequency_multipliers_by_id=risk_frequency_multipliers_by_id,
        risk_impact_multipliers_by_id=risk_impact_multipliers_by_id,
        risk_event_tape=risk_event_tape,
        risk_rng_mode=str(risk_rng_mode),
        initial_regime=initial_regime,
    ).run_contract()
    return sim, sim.product_outcome_panel()
