"""Certified event transducer for Program O's risk-off full-DES contract.

One direct SimPy run extracts the action-independent Op1--Op13 physical
skeleton: treatment anchor, batch-arrival times, native demand quantities and
times, and fixed daily Op9 slots.  This module replays only the product-feasible
state transition over the complete open-loop frontier.  Promotion requires
direct-SimPy parity; the transducer alone is never full-DES evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from itertools import product
import json
import math
import re
from typing import Any, Mapping, Sequence

import numpy as np

from supply_chain.program_o_full_des import (
    PRODUCTS,
    ProgramOFullDESSimulation,
)
from supply_chain.ret_thesis import (
    compute_order_level_ret_excel_request_snapshot_ledger,
)

POLICY_SLOT_PATTERN = re.compile(r"^policy:w(?P<week>\d+):s(?P<position>\d+)$")

MATRIX_KEYS = (
    "ret_visible",
    "ret_full",
    "quantity_ret_full",
    "ration_ret_visible",
    "ret_visible_cvar10",
    "visible_rows",
    "omitted_rows",
    "omitted_quantity",
    "generated_orders",
    "lost_orders",
    "lost_quantity",
    "unresolved_orders",
    "unresolved_quantity",
    "remaining_quantity_P_C",
    "remaining_quantity_P_H",
    "max_backlog_age",
    "service_loss_auc",
    "fill_P_C",
    "fill_P_H",
    "worst_product_fill",
    "ending_inventory_P_C",
    "ending_inventory_P_H",
    "ending_inventory_total",
    "gross_policy_batch_slots",
    "gross_production_quantity",
    "charged_daily_dispatch_slots",
    "charged_downstream_vehicle_hours",
    "actual_loaded_departures",
    "actual_payload",
    "actual_downstream_vehicle_hours",
    "mass_residual",
    "partition_residual",
    "aggregate_ration_residual",
    "raw_material_residual",
)


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True)
class FullDESSkeleton:
    seed: int
    decision_weeks: int
    decision_start: float
    score_time: float
    batch_arrivals: tuple[tuple[float, int, int], ...]
    order_times: tuple[float, ...]
    order_quantities: tuple[float, ...]
    order_products: tuple[str, ...]
    release_slots: tuple[float, ...]
    opening_inventory: tuple[float, float]
    tape_sha256: str
    prefix_state_hash: str
    skeleton_sha256: str
    release_completion_slots: tuple[float, ...] | None = None
    release_available: tuple[bool, ...] | None = None
    risk_events: tuple[dict[str, Any], ...] = ()
    order_contingent: tuple[bool, ...] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "decision_weeks": self.decision_weeks,
            "decision_start": self.decision_start,
            "score_time": self.score_time,
            "batch_arrivals": [list(row) for row in self.batch_arrivals],
            "order_times": list(self.order_times),
            "order_quantities": list(self.order_quantities),
            "order_products": list(self.order_products),
            "release_slots": list(self.release_slots),
            "opening_inventory": list(self.opening_inventory),
            "tape_sha256": self.tape_sha256,
            "prefix_state_hash": self.prefix_state_hash,
            "skeleton_sha256": self.skeleton_sha256,
            "release_completion_slots": (
                None
                if self.release_completion_slots is None
                else list(self.release_completion_slots)
            ),
            "release_available": (
                None if self.release_available is None else list(self.release_available)
            ),
            "risk_events": list(self.risk_events),
            "order_contingent": (
                None if self.order_contingent is None else list(self.order_contingent)
            ),
        }


def full_action_calendars(weeks: int = 8) -> np.ndarray:
    if int(weeks) <= 0 or int(weeks) > 8:
        raise ValueError("weeks must be in 1..8")
    return np.asarray(tuple(product(range(4), repeat=int(weeks))), dtype=np.uint8)


def direct_full_des_vector(
    sim: ProgramOFullDESSimulation, panel: Mapping[str, Any]
) -> dict[str, float]:
    """Project one direct SimPy episode onto the transducer matrix schema."""
    metrics = panel["metrics"]
    products = panel["products"]
    resources = panel["resources"]
    conservation = panel["conservation"]
    visible = compute_order_level_ret_excel_request_snapshot_ledger(sim.orders)
    visible_quantity = sum(float(row["quantity"]) for row in visible["ret_rows"])
    ration_ret_visible = (
        sum(float(row["ret"]) * float(row["quantity"]) for row in visible["ret_rows"])
        / visible_quantity
        if visible_quantity > 0.0
        else 0.0
    )
    orders = [
        order
        for order in sim.orders
        if not order.metrics_excluded
        and float(order.OPTj) >= float(sim.program_o_decision_start or 0.0)
    ]
    omitted_quantity = sum(
        float(order.remaining_qty) + float(order.in_flight_qty)
        for order in orders
        if order.OATj is None
    )
    lost_quantity = sum(float(order.quantity) for order in orders if order.lost)
    output = {
        "ret_visible": float(metrics["ret_excel"]),
        "ret_full": float(metrics["ret_excel_full_ledger"]),
        "quantity_ret_full": float(metrics["ration_ret_excel"]),
        "ration_ret_visible": float(ration_ret_visible),
        "ret_visible_cvar10": float(metrics["ret_excel_cvar10"]),
        "visible_rows": float(metrics["ret_excel_visible_n"]),
        "omitted_rows": float(metrics["ret_excel_omitted_n"]),
        "omitted_quantity": float(omitted_quantity),
        "generated_orders": float(metrics["n_orders"]),
        "lost_orders": float(metrics["n_lost"]),
        "lost_quantity": float(lost_quantity),
        "unresolved_orders": float(metrics["n_orders"] - metrics["n_served"]),
        "unresolved_quantity": float(omitted_quantity),
        "remaining_quantity_P_C": float(products["P_C"]["unresolved_quantity"]),
        "remaining_quantity_P_H": float(products["P_H"]["unresolved_quantity"]),
        "max_backlog_age": float(metrics["backlog_age_max"]),
        "service_loss_auc": float(metrics["service_loss_auc_ration_hours"]),
        "fill_P_C": float(products["P_C"]["fill"]),
        "fill_P_H": float(products["P_H"]["fill"]),
        "worst_product_fill": float(panel["worst_product_fill"]),
        "ending_inventory_P_C": float(
            conservation["per_product"]["P_C"]["nodes"]["rations_sb"]
        ),
        "ending_inventory_P_H": float(
            conservation["per_product"]["P_H"]["nodes"]["rations_sb"]
        ),
        "ending_inventory_total": float(
            conservation["per_product"]["P_C"]["nodes"]["rations_sb"]
            + conservation["per_product"]["P_H"]["nodes"]["rations_sb"]
        ),
        "gross_policy_batch_slots": float(resources["committed_action_batch_slots"]),
        "gross_production_quantity": float(
            resources["gross_action_production_quantity"]
        ),
        "charged_daily_dispatch_slots": float(
            resources["charged_daily_dispatch_slots"]
        ),
        "charged_downstream_vehicle_hours": float(
            resources["charged_downstream_vehicle_hours"]
        ),
        "actual_loaded_departures": float(resources["actual_loaded_departures"]),
        "actual_payload": float(resources["actual_payload"]),
        "actual_downstream_vehicle_hours": float(
            resources["actual_downstream_vehicle_hours"]
        ),
        "mass_residual": float(conservation["max_abs_product_residual"]),
        "partition_residual": float(conservation["max_abs_partition_residual"]),
        "aggregate_ration_residual": abs(
            float(conservation["aggregate_flow_ledger"]["ration_residual"])
        ),
        "raw_material_residual": abs(
            float(conservation["aggregate_flow_ledger"]["raw_residual"])
        ),
    }
    if tuple(output) != MATRIX_KEYS:
        raise AssertionError("direct full-DES vector schema drift")
    return output


def direct_full_des_trace(sim: ProgramOFullDESSimulation) -> dict[str, Any]:
    """Return the promotion-relevant physical/order trace from direct SimPy."""
    start = float(sim.program_o_decision_start or 0.0)
    state_events: list[dict[str, Any]] = []
    for event in sim.program_o_product_events:
        if event.get("event") != "op8_arrived_sb":
            continue
        tokens = event.get("tokens", [])
        if len(tokens) != 1 or not POLICY_SLOT_PATTERN.match(
            str(tokens[0].get("lot_id", ""))
        ):
            continue
        state_events.append(
            {
                "time": float(event["time"]),
                "event": "batch",
                "product_id": str(tokens[0]["product_id"]),
                "quantity": float(event["quantity"]),
                "product_inventory_after": {
                    key: float(value)
                    for key, value in event["product_inventory_after"].items()
                },
            }
        )
    for event in sim.program_o_order_route_events:
        if event.get("event") != "op9_reserved":
            continue
        state_events.append(
            {
                "time": float(event["time"]),
                "event": "release",
                "order_j": int(event["order_j"]),
                "quantity": float(event["quantity"]),
                "product_inventory_after": {
                    key: float(value)
                    for key, value in event["product_inventory_after"].items()
                },
            }
        )
    state_events.sort(
        key=lambda row: (
            float(row["time"]),
            0 if row["event"] == "release" else 1,
            int(row.get("order_j", 0)),
        )
    )
    orders = []
    for order in sorted(
        (
            row
            for row in sim.orders
            if not row.metrics_excluded and float(row.OPTj) >= start
        ),
        key=lambda row: int(row.j),
    ):
        orders.append(
            {
                "j": int(order.j),
                "OPTj": float(order.OPTj),
                "OATj": None if order.OATj is None else float(order.OATj),
                "CTj": None if order.CTj is None else float(order.CTj),
                "ret_bt_at_request": int(getattr(order, "ret_bt_at_request", 0) or 0),
                "ret_ut_at_request": int(getattr(order, "ret_ut_at_request", 0) or 0),
                "requested_product_id": str(order.requested_product_id),
                "quantity": float(order.quantity),
                "lost": bool(order.lost),
            }
        )
    trace = {"state_events": state_events, "orders": orders}
    return {**trace, "sha256": _digest(trace)}


def normalize_scheduler(scheduler: Mapping[str, Sequence[str]]) -> np.ndarray:
    index = {product_id: idx for idx, product_id in enumerate(PRODUCTS)}
    keys = sorted(int(key) for key in scheduler)
    if keys != list(range(len(keys))):
        raise ValueError("scheduler actions must be consecutive from zero")
    return np.asarray(
        [
            [index[product_id] for product_id in scheduler[str(action)]]
            for action in keys
        ],
        dtype=np.uint8,
    )


def extract_full_des_skeleton(
    *,
    seed: int,
    scheduler: Mapping[str, Sequence[str]],
    regime_persistence: float,
    dominant_share: float,
    decision_weeks: int = 8,
    downstream_freight_physics_mode: str = "loaded_only",
    initial_regime: str | None = None,
) -> tuple[FullDESSkeleton, ProgramOFullDESSimulation]:
    """Run one direct calendar and extract only action-independent events."""
    sim = ProgramOFullDESSimulation(
        seed=int(seed),
        calendar=(2,) * int(decision_weeks),
        scheduler=scheduler,
        regime_persistence=float(regime_persistence),
        dominant_share=float(dominant_share),
        complete_substitution=False,
        downstream_freight_physics_mode=str(downstream_freight_physics_mode),
        initial_regime=initial_regime,
    ).run_contract()
    arrivals: dict[tuple[int, int], float] = {}
    for event in sim.program_o_product_events:
        if event["event"] != "op8_arrived_sb":
            continue
        tokens = event["tokens"]
        if len(tokens) != 1:
            raise AssertionError("full-DES policy batch arrival is not one product lot")
        match = POLICY_SLOT_PATTERN.match(str(tokens[0]["lot_id"]))
        if match is None:
            continue
        key = (int(match.group("week")), int(match.group("position")))
        if key in arrivals:
            raise AssertionError(f"duplicate policy slot arrival: {key}")
        arrivals[key] = float(event["time"])
    expected = {
        (week, position) for week in range(int(decision_weeks)) for position in range(3)
    }
    if set(arrivals) != expected:
        raise AssertionError(
            f"missing policy batch arrivals: {sorted(expected-set(arrivals))}"
        )

    orders = sorted(sim.orders, key=lambda order: int(order.j))
    expected_orders = int(decision_weeks) * 6
    if len(orders) != expected_orders:
        raise AssertionError(
            f"expected {expected_orders} orders, observed {len(orders)}"
        )
    start = float(sim.program_o_decision_start or 0.0)
    score = float(sim.env.now)
    first_release = math.floor(start / 24.0) * 24.0 + float(
        sim.op9_freight_offset_hours
    )
    if first_release <= start + 1e-12:
        first_release += 24.0
    release_slots: list[float] = []
    value = first_release
    while value <= score + 1e-12:
        release_slots.append(float(value))
        value += 24.0
    raw = {
        "seed": int(seed),
        "decision_weeks": int(decision_weeks),
        "decision_start": start,
        "score_time": score,
        "batch_arrivals": [
            [arrivals[(week, position)], week, position]
            for week in range(int(decision_weeks))
            for position in range(3)
        ],
        "order_times": [float(order.OPTj) for order in orders],
        "order_quantities": [float(order.quantity) for order in orders],
        "order_products": [str(order.requested_product_id) for order in orders],
        "release_slots": release_slots,
        "opening_inventory": [5000.0, 5000.0],
        "tape_sha256": sim.program_o_tape["sha256"],
        "prefix_state_hash": str(sim.program_o_prefix_state_hash),
    }
    skeleton = FullDESSkeleton(
        seed=int(seed),
        decision_weeks=int(decision_weeks),
        decision_start=start,
        score_time=score,
        batch_arrivals=tuple(
            (float(time), int(week), int(position))
            for time, week, position in raw["batch_arrivals"]
        ),
        order_times=tuple(raw["order_times"]),
        order_quantities=tuple(raw["order_quantities"]),
        order_products=tuple(raw["order_products"]),
        release_slots=tuple(release_slots),
        opening_inventory=(5000.0, 5000.0),
        tape_sha256=str(raw["tape_sha256"]),
        prefix_state_hash=str(raw["prefix_state_hash"]),
        skeleton_sha256=_digest(raw),
    )
    return skeleton, sim


def _tail_mean(values: np.ndarray, valid: np.ndarray, fraction: float) -> np.ndarray:
    """Row-wise lower-tail mean over valid order cells."""
    count = valid.sum(axis=1)
    output = np.ones(valid.shape[0], dtype=np.float64)
    for n_visible in np.unique(count):
        mask = count == n_visible
        if not np.any(mask):
            continue
        if int(n_visible) == 0:
            output[mask] = 0.0
            continue
        k = max(1, int(math.ceil(float(n_visible) * float(fraction))))
        panel = np.where(valid[mask], values[mask], np.inf)
        output[mask] = np.partition(panel, k - 1, axis=1)[:, :k].mean(axis=1)
    return output


def _risk_adjusted_order_values(
    *,
    skeleton: FullDESSkeleton,
    oat: np.ndarray,
    opt: np.ndarray,
    bt: np.ndarray,
    completed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce DES AP/RP risk branches for a policy-independent event tape."""
    n_calendar, n_order = oat.shape
    request_values = 1.0 - bt.astype(np.float64) / np.arange(
        1, n_order + 1
    )[None, :]
    active = np.zeros((n_calendar, n_order), dtype=bool)
    r24_indicator = np.zeros((n_calendar, n_order), dtype=bool)
    disruption_hours = np.zeros((n_calendar, n_order), dtype=np.float64)
    point_events: list[tuple[float, str, float]] = []
    origin = float(skeleton.decision_start)
    for event in skeleton.risk_events:
        risk_id = str(event["risk_id"])
        start = origin + float(event["start_time"])
        duration = max(0.0, float(event.get("duration", 0.0)))
        end = origin + float(event.get("end_time", event["start_time"] + duration))
        if duration <= 0.0:
            magnitude = max(0.0, float(event.get("magnitude", 0.0)))
            inside = completed & (opt[None, :] <= start + 1e-12) & (
                start <= oat + 1e-12
            )
            active |= inside
            if risk_id == "R24":
                r24_indicator |= inside
            if risk_id in {"R14", "R24"} and magnitude > 0.0:
                units = 1.0 if risk_id == "R14" else max(1.0, magnitude / 2600.0)
                point_events.append((start, risk_id, units))
            continue
        # A duration event enters sim.risk_events only when it ends.  An order
        # finalized before that callback cannot yet carry its indicator.
        overlap = np.maximum(
            0.0,
            np.minimum(oat, end) - np.maximum(opt[None, :], start),
        )
        eligible = completed & (end <= oat + 1e-12) & (overlap > 0.0)
        active |= eligible
        disruption_hours += np.where(eligible, overlap, 0.0)

    contingent = (
        np.zeros(n_order, dtype=bool)
        if skeleton.order_contingent is None
        else np.asarray(skeleton.order_contingent, dtype=bool)
    )
    point_events.sort(key=lambda row: (row[0], row[1]))
    # Quantity-risk ledgers are consumed in physical order-completion order.
    # A scalar loop over 48 orders is cheap; all other state remains vectorized.
    for calendar_index in range(n_calendar):
        completed_indices = np.flatnonzero(completed[calendar_index])
        completion_order = sorted(
            completed_indices.tolist(),
            key=lambda order_index: (
                float(oat[calendar_index, order_index]),
                int(order_index),
            ),
        )
        event_index = 0
        r14_available = 0.0
        r24_available = 0.0
        for order_index in completion_order:
            completion_time = float(oat[calendar_index, order_index])
            while (
                event_index < len(point_events)
                and point_events[event_index][0] <= completion_time + 1e-12
            ):
                _time, risk_id, units = point_events[event_index]
                if risk_id == "R14":
                    r14_available += units
                else:
                    r24_available += units
                event_index += 1
            if r14_available > 0.0:
                active[calendar_index, order_index] = True
                disruption_hours[calendar_index, order_index] += 72.0
            if r24_available > 0.0:
                active[calendar_index, order_index] = True
                r24_indicator[calendar_index, order_index] = True
                disruption_hours[calendar_index, order_index] += 1.0
                r24_available = max(0.0, r24_available - min(1.0, r24_available))
            if contingent[order_index] and not r24_indicator[calendar_index, order_index]:
                active[calendar_index, order_index] = True
                r24_indicator[calendar_index, order_index] = True
                disruption_hours[calendar_index, order_index] += min(
                    max(0.0, completion_time - float(opt[order_index])), 48.0
                )

    ct = oat - opt[None, :]
    risk_values = np.zeros((n_calendar, n_order), dtype=np.float64)
    on_time = active & completed & (ct <= 48.0 + 1e-12)
    late = active & completed & ~on_time
    risk_values[on_time] = np.minimum(disruption_hours[on_time], 48.0) / 48.0
    positive_recovery = late & (disruption_hours > 0.0)
    risk_values[positive_recovery] = 0.5 / disruption_hours[positive_recovery]
    request_values = np.where(active, risk_values, request_values)
    return request_values, active


def simulate_full_des_frontier(
    *,
    skeleton: FullDESSkeleton,
    scheduler: Mapping[str, Sequence[str]],
    calendars: np.ndarray | None = None,
    complete_substitution: bool = False,
    trace_out: dict[str, Any] | None = None,
    include_q_r1_metrics: bool = False,
) -> dict[str, np.ndarray]:
    """Vectorized replay of every calendar over one direct-DES skeleton."""
    calendars = (
        full_action_calendars(skeleton.decision_weeks)
        if calendars is None
        else np.asarray(calendars)
    )
    if calendars.ndim != 2 or calendars.shape[1] != skeleton.decision_weeks:
        raise ValueError(f"calendars must have shape (n, {skeleton.decision_weeks})")
    n_calendar = int(calendars.shape[0])
    if trace_out is not None and n_calendar != 1:
        raise ValueError("trace_out is available only for one calendar")
    n_order = len(skeleton.order_times)
    scheduler_array = normalize_scheduler(scheduler)
    if calendars.size and (
        int(calendars.min()) < 0 or int(calendars.max()) >= len(scheduler_array)
    ):
        raise ValueError("calendar contains an action absent from scheduler")
    product_index = {product_id: idx for idx, product_id in enumerate(PRODUCTS)}
    requested_product = np.asarray(
        [product_index[product_id] for product_id in skeleton.order_products],
        dtype=np.uint8,
    )
    opt = np.asarray(skeleton.order_times, dtype=np.float64)
    quantities = np.asarray(skeleton.order_quantities, dtype=np.float64)
    rows = np.arange(n_calendar)

    inventory = np.repeat(
        np.asarray(skeleton.opening_inventory, dtype=np.float64)[None, :],
        n_calendar,
        axis=0,
    )
    pending = np.zeros((n_calendar, n_order), dtype=bool)
    oat = np.full((n_calendar, n_order), np.inf, dtype=np.float64)
    released = np.zeros((n_calendar, n_order), dtype=bool)
    bt = np.zeros((n_calendar, n_order), dtype=np.uint8)
    created = np.zeros(n_order, dtype=bool)
    trace_events: list[dict[str, Any]] = []

    events: list[tuple[float, int, str, tuple[int, int] | int | None]] = []
    # Match the frozen SimPy registration semantics exactly.  The daily Op9
    # release wakes before an Op8 batch arrival at the same timestamp, so stock
    # arriving at that instant is first eligible at the next daily release.
    # This ordering is physical contract state, not an implementation detail:
    # reversing it changes OATj and request-time backlog snapshots.
    release_completion_slots = (
        tuple(float(time) + 48.0 for time in skeleton.release_slots)
        if skeleton.release_completion_slots is None
        else tuple(map(float, skeleton.release_completion_slots))
    )
    release_available = (
        tuple(True for _ in skeleton.release_slots)
        if skeleton.release_available is None
        else tuple(map(bool, skeleton.release_available))
    )
    if not (
        len(release_completion_slots)
        == len(release_available)
        == len(skeleton.release_slots)
    ):
        raise ValueError("release completion/availability vectors must match release_slots")
    for release_index, time in enumerate(skeleton.release_slots):
        events.append((float(time), 0, "release", int(release_index)))
    for time, week, position in skeleton.batch_arrivals:
        events.append((float(time), 1, "batch", (int(week), int(position))))
    for order_index, time in enumerate(skeleton.order_times):
        events.append((float(time), 2, "demand", int(order_index)))

    contingent_class = (
        np.ones(n_order, dtype=np.uint8)
        if skeleton.order_contingent is None
        else 1 - np.asarray(skeleton.order_contingent, dtype=np.uint8)
    )
    if contingent_class.shape != (n_order,):
        raise ValueError("order_contingent must match order count")
    priority_order = np.lexsort(
        (
            np.arange(n_order, dtype=np.int64),
            opt,
            quantities,
            contingent_class,
        )
    )
    for now, _priority, kind, payload in sorted(events):
        if kind == "batch":
            week, position = payload  # type: ignore[misc]
            labels = scheduler_array[calendars[:, week], position]
            inventory[rows, labels] += 5000.0
            if trace_out is not None:
                trace_events.append(
                    {
                        "time": float(now),
                        "event": "batch",
                        "product_id": PRODUCTS[int(labels[0])],
                        "quantity": 5000.0,
                        "product_inventory_after": {
                            PRODUCTS[idx]: float(inventory[0, idx]) for idx in range(2)
                        },
                    }
                )
            continue
        if kind == "demand":
            order_index = int(payload)  # type: ignore[arg-type]
            prior = np.arange(order_index)
            if len(prior):
                late = (opt[prior] + 48.0 <= float(now) + 1e-12)[None, :] & (
                    oat[:, prior] > float(now) + 1e-12
                )
                bt[:, order_index] = np.minimum(60, late.sum(axis=1)).astype(np.uint8)
            pending[:, order_index] = True
            created[order_index] = True
            continue

        release_index = int(payload)  # type: ignore[arg-type]
        if not release_available[release_index]:
            continue
        chosen = np.full(n_calendar, -1, dtype=np.int16)
        for order_index in priority_order:
            if not created[order_index]:
                continue
            if complete_substitution:
                enough = inventory.sum(axis=1) + 1e-9 >= quantities[order_index]
            else:
                enough = (
                    inventory[:, requested_product[order_index]] + 1e-9
                    >= quantities[order_index]
                )
            mask = (chosen < 0) & pending[:, order_index] & enough
            chosen[mask] = int(order_index)
        for order_index in priority_order:
            mask = chosen == int(order_index)
            if not np.any(mask):
                continue
            qty = float(quantities[order_index])
            if complete_substitution:
                take_c = np.minimum(inventory[mask, 0], qty)
                inventory[mask, 0] -= take_c
                inventory[mask, 1] -= qty - take_c
            else:
                inventory[mask, requested_product[order_index]] -= qty
            pending[mask, order_index] = False
            released[mask, order_index] = True
            oat[mask, order_index] = float(release_completion_slots[release_index])
            if trace_out is not None and bool(mask[0]):
                trace_events.append(
                    {
                        "time": float(now),
                        "event": "release",
                        "order_j": int(order_index + 1),
                        "quantity": qty,
                        "product_inventory_after": {
                            PRODUCTS[idx]: float(inventory[0, idx]) for idx in range(2)
                        },
                    }
                )

    completed = oat <= float(skeleton.score_time) + 1e-12
    ct = oat - opt[None, :]
    visible_values, risk_active = _risk_adjusted_order_values(
        skeleton=skeleton,
        oat=oat,
        opt=opt,
        bt=bt,
        completed=completed,
    )
    visible_count = completed.sum(axis=1)
    visible_sum = np.where(completed, visible_values, 0.0).sum(axis=1)
    ret_visible = np.divide(
        visible_sum,
        visible_count,
        out=np.ones(n_calendar, dtype=np.float64),
        where=visible_count > 0,
    )
    completed_quantity = np.where(completed, quantities[None, :], 0.0)
    visible_quantity_total = completed_quantity.sum(axis=1)
    ration_ret_visible = np.divide(
        (completed_quantity * visible_values).sum(axis=1),
        visible_quantity_total,
        out=np.zeros(n_calendar, dtype=np.float64),
        where=visible_quantity_total > 0.0,
    )

    unresolved = ~completed
    unresolved_quantity = np.where(unresolved, quantities[None, :], 0.0)
    unresolved_count = unresolved.sum(axis=1)
    # Exact no-risk branch of compute_order_level_ret_excel_formula.  A late
    # completed order increments the accumulated Bt before its own row.  A
    # pending overdue order does likewise; an in-flight order with no remaining
    # queue quantity does not.  Unfulfilled rows themselves score zero.
    order_backorder = (completed & (ct > 48.0 + 1e-12)) | (
        unresolved
        & pending
        & (float(skeleton.score_time) - opt[None, :] > 48.0 + 1e-12)
    )
    cumulative_backorders = np.minimum(
        60, np.cumsum(order_backorder.astype(np.int16), axis=1)
    )
    full_order_values = np.where(
        completed,
        1.0
        - cumulative_backorders / np.arange(1, n_order + 1, dtype=np.float64)[None, :],
        0.0,
    )
    full_order_values = np.where(
        completed & risk_active,
        visible_values,
        full_order_values,
    )
    ret_full = full_order_values.mean(axis=1)
    quantity_ret_full = (full_order_values * quantities[None, :]).sum(
        axis=1
    ) / quantities.sum()
    end = np.minimum(oat, float(skeleton.score_time))
    lateness = np.maximum(0.0, end - (opt[None, :] + 48.0))
    service_loss_auc = (lateness * quantities[None, :]).sum(axis=1)
    ages = np.where(
        unresolved,
        float(skeleton.score_time) - opt[None, :],
        0.0,
    )
    max_backlog_age = ages.max(axis=1)

    demand_by_product = np.asarray(
        [quantities[requested_product == idx].sum() for idx in range(2)],
        dtype=np.float64,
    )
    completed_by_product = np.stack(
        [
            np.where(
                completed[:, requested_product == idx],
                quantities[requested_product == idx][None, :],
                0.0,
            ).sum(axis=1)
            for idx in range(2)
        ],
        axis=1,
    )
    fill = np.divide(
        completed_by_product,
        demand_by_product[None, :],
        out=np.ones_like(completed_by_product),
        where=demand_by_product[None, :] > 0.0,
    )
    remaining_by_product = demand_by_product[None, :] - completed_by_product
    loaded = released.sum(axis=1).astype(np.float64)
    payload = (released * quantities[None, :]).sum(axis=1)
    gross_slots = float(skeleton.decision_weeks * 3)
    production = np.full(n_calendar, gross_slots * 5000.0)
    opening = float(sum(skeleton.opening_inventory))
    mass_residual = opening + production - inventory.sum(axis=1) - payload

    output: dict[str, np.ndarray] = {
        "ret_visible": ret_visible,
        "ret_full": ret_full,
        "quantity_ret_full": quantity_ret_full,
        "ration_ret_visible": ration_ret_visible,
        "ret_visible_cvar10": _tail_mean(visible_values, completed, 0.10),
        "visible_rows": visible_count.astype(np.float64),
        "omitted_rows": (n_order - visible_count).astype(np.float64),
        "omitted_quantity": unresolved_quantity.sum(axis=1),
        "generated_orders": np.full(n_calendar, float(n_order)),
        "lost_orders": np.zeros(n_calendar),
        "lost_quantity": np.zeros(n_calendar),
        "unresolved_orders": unresolved_count.astype(np.float64),
        "unresolved_quantity": unresolved_quantity.sum(axis=1),
        "remaining_quantity_P_C": remaining_by_product[:, 0],
        "remaining_quantity_P_H": remaining_by_product[:, 1],
        "max_backlog_age": max_backlog_age,
        "service_loss_auc": service_loss_auc,
        "fill_P_C": fill[:, 0],
        "fill_P_H": fill[:, 1],
        "worst_product_fill": fill.min(axis=1),
        "ending_inventory_P_C": inventory[:, 0],
        "ending_inventory_P_H": inventory[:, 1],
        "ending_inventory_total": inventory.sum(axis=1),
        "gross_policy_batch_slots": np.full(n_calendar, gross_slots),
        "gross_production_quantity": production,
        "charged_daily_dispatch_slots": np.full(
            n_calendar, float(len(skeleton.release_slots))
        ),
        "charged_downstream_vehicle_hours": np.full(
            n_calendar, float(len(skeleton.release_slots) * 48.0)
        ),
        "actual_loaded_departures": loaded,
        "actual_payload": payload,
        "actual_downstream_vehicle_hours": loaded * 48.0,
        "mass_residual": mass_residual,
        "partition_residual": np.zeros(n_calendar),
        "aggregate_ration_residual": np.zeros(n_calendar),
        "raw_material_residual": np.zeros(n_calendar),
    }
    if tuple(output) != MATRIX_KEYS:
        raise AssertionError("full-DES transducer matrix schema drift")
    if include_q_r1_metrics:
        early_mask = opt < float(skeleton.decision_start) + 2.0 * 168.0 - 1e-12
        early_generated = int(early_mask.sum())
        early_completed = completed[:, early_mask]
        early_values = visible_values[:, early_mask]
        early_visible_count = early_completed.sum(axis=1)
        early_visible_sum = np.where(early_completed, early_values, 0.0).sum(axis=1)
        early_visible = np.divide(
            early_visible_sum,
            early_visible_count,
            out=np.ones(n_calendar, dtype=np.float64),
            where=early_visible_count > 0,
        )
        early_complete = (
            early_visible_sum / float(early_generated)
            if early_generated
            else np.ones(n_calendar, dtype=np.float64)
        )
        early_demand_by_product = np.asarray(
            [
                quantities[early_mask & (requested_product == idx)].sum()
                for idx in range(2)
            ],
            dtype=np.float64,
        )
        early_completed_by_product = np.stack(
            [
                np.where(
                    completed[:, early_mask & (requested_product == idx)],
                    quantities[early_mask & (requested_product == idx)][None, :],
                    0.0,
                ).sum(axis=1)
                for idx in range(2)
            ],
            axis=1,
        )
        early_fill = np.divide(
            early_completed_by_product,
            early_demand_by_product[None, :],
            out=np.ones_like(early_completed_by_product),
            where=early_demand_by_product[None, :] > 0.0,
        )
        output.update(
            {
                "early_ret_visible": early_visible,
                "early_ret_complete_cohort": early_complete,
                "early_generated_orders": np.full(
                    n_calendar, float(early_generated)
                ),
                "early_visible_rows": early_visible_count.astype(np.float64),
                "early_unresolved_orders": (
                    early_generated - early_visible_count
                ).astype(np.float64),
                "early_fill_P_C": early_fill[:, 0],
                "early_fill_P_H": early_fill[:, 1],
                "early_worst_product_fill": early_fill.min(axis=1),
            }
        )
    if trace_out is not None:
        orders = []
        for order_index in range(n_order):
            oat_value = float(oat[0, order_index])
            orders.append(
                {
                    "j": int(order_index + 1),
                    "OPTj": float(opt[order_index]),
                    "OATj": None if not math.isfinite(oat_value) else oat_value,
                    "CTj": (
                        None
                        if not math.isfinite(oat_value)
                        else float(oat_value - opt[order_index])
                    ),
                    "ret_bt_at_request": int(bt[0, order_index]),
                    "ret_ut_at_request": 0,
                    "requested_product_id": str(skeleton.order_products[order_index]),
                    "quantity": float(quantities[order_index]),
                    "lost": False,
                }
            )
        trace = {"state_events": trace_events, "orders": orders}
        trace_out.update({**trace, "sha256": _digest(trace)})
    return output
