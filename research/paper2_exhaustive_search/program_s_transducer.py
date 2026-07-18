"""Risk-aware Program S skeleton extraction and exactness gates.

The vectorized Program O transducer is reused only after the direct SimPy
skeleton is proven action-independent.  Any missing slot, action-dependent
timing/quantity, or matrix mismatch blocks that mask before S1 screening.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence

import numpy as np

from supply_chain.program_o_full_des_transducer import (
    FullDESSkeleton,
    MATRIX_KEYS,
    direct_full_des_vector,
    full_action_calendars,
    simulate_full_des_frontier,
)
from supply_chain.program_s_risk_interaction import (
    ProgramSCell,
    ProgramSRiskAwareSimulation,
)


_POLICY_SLOT = re.compile(r"^policy:w(?P<week>\d+):s(?P<position>\d+)$")


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def extract_program_s_skeleton(sim: ProgramSRiskAwareSimulation) -> FullDESSkeleton:
    weeks = int(sim.program_o_decision_weeks)
    arrivals: dict[tuple[int, int], float] = {}
    for event in sim.program_o_product_events:
        if event.get("event") != "op8_arrived_sb":
            continue
        tokens = event.get("tokens", ())
        if len(tokens) != 1:
            raise AssertionError("risk-aware finished batch spans product targets")
        match = _POLICY_SLOT.match(str(tokens[0].get("lot_id", "")))
        if match is None:
            continue
        key = (int(match.group("week")), int(match.group("position")))
        if key in arrivals:
            raise AssertionError(f"duplicate policy slot arrival: {key}")
        arrivals[key] = float(event["time"])
    expected = {(week, slot) for week in range(weeks) for slot in range(3)}
    if set(arrivals) != expected:
        raise AssertionError(f"missing risk-aware policy slots: {sorted(expected - set(arrivals))}")

    start = float(sim.program_o_decision_start or 0.0)
    score = float(sim.env.now)
    orders = sorted(
        (
            order
            for order in sim.orders
            if not order.metrics_excluded and float(order.OPTj) >= start
        ),
        key=lambda order: int(order.j),
    )
    if len(orders) != weeks * 6:
        raise AssertionError(f"expected {weeks * 6} orders, observed {len(orders)}")
    first_release = math.floor(start / 24.0) * 24.0 + float(sim.op9_freight_offset_hours)
    if first_release <= start + 1e-12:
        first_release += 24.0
    release_slots: list[float] = []
    value = first_release
    while value <= score + 1e-12:
        release_slots.append(float(value))
        value += 24.0
    route_by_order: dict[int, dict[str, float]] = {}
    for row in sim.program_o_order_route_events:
        order_j = int(row.get("order_j", 0) or 0)
        if not order_j:
            continue
        route = route_by_order.setdefault(order_j, {})
        if row.get("event") == "op9_reserved":
            route["release"] = float(row["time"])
        elif row.get("event") == "post_op12_physical_delivery":
            route["completion"] = float(row["time"])
    loaded_completion_by_release = {
        float(route["release"]): float(route["completion"])
        for route in route_by_order.values()
        if "release" in route and "completion" in route
    }
    mission_completion: dict[float, float] = {}
    for mission in sim.program_o_downstream_mission_events:
        scheduled = float(mission["scheduled_at"])
        if mission["kind"] == "loaded":
            mission_completion[scheduled] = loaded_completion_by_release.get(
                scheduled, float("inf")
            )
        else:
            completed = mission.get("op12_completed_at")
            mission_completion[scheduled] = (
                float(completed) if completed is not None else float("inf")
            )
    if set(mission_completion) != set(release_slots):
        raise AssertionError("fixed-clock mission schedule is incomplete")
    relative_events = tuple(
        {
            "risk_id": str(event.risk_id),
            "start_time": float(event.start_time) - start,
            "end_time": float(event.end_time) - start,
            "duration": float(event.duration),
            "affected_ops": list(event.affected_ops),
            "magnitude": float(event.magnitude),
            "unit": str(event.unit),
        }
        for event in sim.risk_events
        if float(event.start_time) >= start - 1e-12
    )
    release_available = tuple(
        not any(
            9 in row.get("affected_ops", ())
            and float(row["start_time"]) <= slot - start < float(row["end_time"])
            for row in relative_events
        )
        for slot in release_slots
    )
    raw = {
        "seed": int(sim.seed),
        "decision_weeks": weeks,
        "decision_start": start,
        "score_time": score,
        "batch_arrivals": [
            [arrivals[(week, position)], week, position]
            for week in range(weeks)
            for position in range(3)
        ],
        "order_times": [float(order.OPTj) for order in orders],
        "order_quantities": [float(order.quantity) for order in orders],
        "order_products": [str(order.requested_product_id) for order in orders],
        "order_contingent": [bool(order.contingent) for order in orders],
        "release_slots": release_slots,
        "opening_inventory": [5000.0, 5000.0],
        "tape_sha256": sim.program_o_tape["sha256"],
        "prefix_state_hash": str(sim.program_o_prefix_state_hash),
        "risk_event_tape_sha256": sim.program_s_risk_tape_sha256,
        "release_completion_slots": [
            (
                mission_completion[slot]
                if math.isfinite(mission_completion[slot])
                else None
            )
            for slot in release_slots
        ],
        "release_available": list(release_available),
        "risk_events": list(relative_events),
    }
    # FullDESSkeleton's digest historically excludes risk identity.  Program S
    # binds it into skeleton_sha256 so a risk tape cannot be silently swapped.
    return FullDESSkeleton(
        seed=int(sim.seed),
        decision_weeks=weeks,
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
        release_completion_slots=tuple(
            float(mission_completion[slot]) for slot in release_slots
        ),
        release_available=release_available,
        risk_events=relative_events,
        order_contingent=tuple(bool(order.contingent) for order in orders),
    )


def run_program_s_direct(
    *,
    seed: int,
    calendar: Sequence[int],
    scheduler: Mapping[str, Sequence[str]],
    cell: ProgramSCell,
    risk_event_tape: Sequence[Mapping[str, Any]],
) -> ProgramSRiskAwareSimulation:
    return ProgramSRiskAwareSimulation(
        seed=int(seed),
        calendar=calendar,
        scheduler=scheduler,
        cell=cell,
        risk_event_tape=risk_event_tape,
        downstream_freight_physics_mode="fixed_clock_physical_v1",
    ).run_contract()


def exact_short_horizon_gate(
    *,
    seed: int,
    scheduler: Mapping[str, Sequence[str]],
    cell: ProgramSCell,
    risk_tapes_by_horizon: Mapping[int, Sequence[Mapping[str, Any]]],
    horizons: Sequence[int] = (1, 2, 3),
    tolerance: float = 1e-10,
) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for horizon in horizons:
        calendars = full_action_calendars(int(horizon))
        reference = run_program_s_direct(
            seed=seed,
            calendar=(2,) * int(horizon),
            scheduler=scheduler,
            cell=cell,
            risk_event_tape=risk_tapes_by_horizon[int(horizon)],
        )
        skeleton = extract_program_s_skeleton(reference)
        frontier = simulate_full_des_frontier(
            skeleton=skeleton, scheduler=scheduler, calendars=calendars
        )
        max_error = 0.0
        skeleton_hashes = set()
        for index, calendar in enumerate(calendars):
            direct_sim = run_program_s_direct(
                seed=seed,
                calendar=calendar.tolist(),
                scheduler=scheduler,
                cell=cell,
                risk_event_tape=risk_tapes_by_horizon[int(horizon)],
            )
            direct_skeleton = extract_program_s_skeleton(direct_sim)
            skeleton_hashes.add(direct_skeleton.skeleton_sha256)
            direct = direct_full_des_vector(direct_sim, direct_sim.product_outcome_panel())
            for key in MATRIX_KEYS:
                max_error = max(max_error, abs(float(direct[key]) - float(frontier[key][index])))
        summaries[str(horizon)] = {
            "calendars": int(len(calendars)),
            "unique_skeleton_hashes": len(skeleton_hashes),
            "max_matrix_abs_error": max_error,
            "pass": len(skeleton_hashes) == 1 and max_error <= float(tolerance),
        }
    return {
        "horizons": summaries,
        "tolerance": float(tolerance),
        "pass": all(row["pass"] for row in summaries.values()),
    }
