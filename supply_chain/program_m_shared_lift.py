"""Program M shared-lift reservation extension for the full MFSC DES.

The extension is deliberately narrow.  It reuses ``MFSCSimulation``'s split
CSSU topology, order queue, Op9 stock withdrawal, localized risk replay, and
order finalization.  A protected movement is eligible only while the reserved
destination's *local* Op10 or Op12 path is down.  It ignores that local route
flag for the two nominal 24-hour legs, but it does not ignore aggregate
operation outages or the destination-local Op11 node.

Every reservation commits one 2,600-ration / 48-vehicle-hour movement whether
it ultimately runs loaded or empty.  Consequently action calendars have the
same resource envelope before any outcomes are known.  With the extension
disabled, or with ``bypass_local_route=False`` (the null-physics cell), order
transitions delegate directly to the unmodified DES.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from typing import Any, Iterable, Mapping, Sequence

from .config import HOURS_PER_WEEK
from .supply_chain import MFSCSimulation, OrderRecord, RiskEvent


CONTRACT_ID = "program_m_shared_lift_reservation_v1"
DESTINATIONS = ("A", "B")
ACTION_NAMES = ("RESERVE_A", "RESERVE_B")
DECISION_WEEKS = 8
ACTIVATION_LEAD_HOURS = 24.0
SLOT_WINDOW_HOURS = 144.0
PAYLOAD_CAPACITY_RATIONS = 2_600.0
COMMITMENT_VEHICLE_HOURS = 48.0
LOCAL_BYPASS_OPS = frozenset({10, 12})


@dataclass
class ReservationRecord:
    """One irrevocable, destination-specific weekly commitment."""

    week: int
    action: str
    destination: str
    decision_time: float
    activation_time: float
    expiry_time: float
    status: str = "scheduled"
    used_at: float | None = None
    order_j: int | None = None
    payload_rations: float = 0.0


@dataclass(frozen=True)
class WarningRecord:
    """Deployable signal emitted at a weekly decision epoch."""

    week: int
    observed_at: float
    warning_A: int
    warning_B: int


def _event_value(event: RiskEvent | Mapping[str, Any], name: str) -> Any:
    return getattr(event, name) if isinstance(event, RiskEvent) else event.get(name)


def _uniform01(*parts: Any) -> float:
    digest = hashlib.sha256(":".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def _event_interval(event: RiskEvent | Mapping[str, Any]) -> tuple[float, float]:
    start = float(_event_value(event, "start_time") or 0.0)
    duration_end = start + float(_event_value(event, "duration") or 0.0)
    recorded_end = float(_event_value(event, "end_time") or start)
    return start, max(recorded_end, duration_end)


def materialize_warning_records(
    *,
    seed: int,
    decision_start_time: float,
    risk_events: Iterable[RiskEvent | Mapping[str, Any]],
    sensitivity: float,
    specificity: float,
) -> tuple[WarningRecord, ...]:
    """Create deterministic joint A/B warnings without exposing future truth.

    Each bit is an independent Bernoulli draw conditional on whether a local
    Op10/Op12 event for that destination overlaps the slot's activation
    window.  Thus ``00``, ``01``, ``10`` and ``11`` are all possible; the
    generator does not force a one-hot warning or balance locations.  Event
    starts remain properties of the exogenous tape and are never aligned to a
    policy's departure.
    """

    if not 0.0 <= float(sensitivity) <= 1.0:
        raise ValueError("sensitivity must be in [0, 1]")
    if not 0.0 <= float(specificity) <= 1.0:
        raise ValueError("specificity must be in [0, 1]")
    events = tuple(risk_events)
    records: list[WarningRecord] = []
    for week in range(DECISION_WEEKS):
        decision_time = float(decision_start_time) + week * HOURS_PER_WEEK
        window_start = decision_time + ACTIVATION_LEAD_HOURS
        window_end = window_start + SLOT_WINDOW_HOURS
        bits: dict[str, int] = {}
        for destination in DESTINATIONS:
            truth = any(
                _event_value(event, "affected_cssu") == destination
                and bool(
                    set(_event_value(event, "affected_ops") or ())
                    & LOCAL_BYPASS_OPS
                )
                and _event_interval(event)[0] < window_end
                and _event_interval(event)[1] > window_start
                for event in events
            )
            probability = float(sensitivity) if truth else 1.0 - float(specificity)
            draw = _uniform01(CONTRACT_ID, int(seed), week, destination, "warning")
            bits[destination] = int(draw < probability)
        records.append(
            WarningRecord(
                week=week,
                observed_at=decision_time,
                warning_A=bits["A"],
                warning_B=bits["B"],
            )
        )
    return tuple(records)


class ProgramMSharedLiftSimulation(MFSCSimulation):
    """Opt-in full-DES implementation of the frozen Program M actuator."""

    def __init__(
        self,
        *args: Any,
        program_m_enabled: bool = False,
        reservation_calendar: Sequence[str] | None = None,
        decision_start_time: float = 0.0,
        bypass_local_route: bool = True,
        warning_sensitivity: float = 0.85,
        warning_specificity: float = 0.90,
        **kwargs: Any,
    ) -> None:
        self.program_m_enabled = bool(program_m_enabled)
        self.program_m_decision_start_time = float(decision_start_time)
        self.program_m_bypass_local_route = bool(bypass_local_route)
        self.program_m_reservations: list[ReservationRecord] = []
        self.program_m_action_events: list[dict[str, Any]] = []
        self.program_m_movement_events: list[dict[str, Any]] = []
        self.program_m_risk_event_ledger: list[dict[str, Any]] = []
        self._program_m_recorded_empty_weeks: set[int] = set()

        if self.program_m_enabled:
            for key, required in (
                ("cssu_topology_mode", "split_v1"),
                ("order_fulfillment_mode", "op9_linked"),
            ):
                supplied = kwargs.get(key, required)
                if supplied != required:
                    raise ValueError(f"Program M requires {key}={required!r}")
                kwargs[key] = required
            # A protected mode is an additional mode, not additional stock or
            # baseline terrestrial capacity.  The normal tandem remains intact.
            kwargs.setdefault("downstream_transport_capacity_mode", "tandem_capacity_one")

        super().__init__(*args, **kwargs)

        if self.program_m_enabled:
            self._validate_program_m_risk_events()

        self.program_m_warning_records = materialize_warning_records(
            seed=int(self.seed or 0),
            decision_start_time=self.program_m_decision_start_time,
            risk_events=self.risk_event_tape or (),
            sensitivity=float(warning_sensitivity),
            specificity=float(warning_specificity),
        )
        if reservation_calendar is not None:
            if not self.program_m_enabled:
                raise ValueError("reservation_calendar requires program_m_enabled=True")
            if len(reservation_calendar) != DECISION_WEEKS:
                raise ValueError("reservation_calendar must contain exactly eight actions")
            for week, action in enumerate(reservation_calendar):
                self._book_reservation(week=week, action=str(action))

        if self.program_m_enabled:
            required_end = (
                self.program_m_decision_start_time + DECISION_WEEKS * HOURS_PER_WEEK
            )
            if float(self.horizon) + 1e-9 < required_end:
                raise ValueError(
                    "Program M horizon must include all eight activation windows; "
                    f"need horizon >= {required_end:g}"
                )

    def _book_reservation(self, *, week: int, action: str) -> ReservationRecord:
        if action not in ACTION_NAMES:
            raise ValueError(f"Unknown Program M action {action!r}")
        if not 0 <= int(week) < DECISION_WEEKS:
            raise ValueError("Program M week must be in [0, 7]")
        if any(record.week == int(week) for record in self.program_m_reservations):
            raise ValueError(f"Program M week {week} already has a reservation")
        decision_time = self.program_m_decision_start_time + int(week) * HOURS_PER_WEEK
        activation_time = decision_time + ACTIVATION_LEAD_HOURS
        record = ReservationRecord(
            week=int(week),
            action=action,
            destination=action.removeprefix("RESERVE_"),
            decision_time=float(decision_time),
            activation_time=float(activation_time),
            expiry_time=float(activation_time + SLOT_WINDOW_HOURS),
        )
        self.program_m_reservations.append(record)
        self.program_m_reservations.sort(key=lambda row: row.week)
        self.program_m_action_events.append(
            {**asdict(record), "event": "reservation_committed"}
        )
        return record

    def reserve_current_week(self, action: str) -> ReservationRecord:
        """Commit exactly at the current weekly decision epoch."""

        if not self.program_m_enabled:
            raise RuntimeError("Program M is disabled")
        relative = float(self.env.now) - self.program_m_decision_start_time
        week = int(round(relative / HOURS_PER_WEEK))
        expected = self.program_m_decision_start_time + week * HOURS_PER_WEEK
        if not math.isclose(float(self.env.now), expected, abs_tol=1e-9):
            raise RuntimeError("reservation must be committed at an exact weekly epoch")
        return self._book_reservation(week=week, action=action)

    def assert_complete_calendar(self) -> None:
        weeks = [record.week for record in self.program_m_reservations]
        if weeks != list(range(DECISION_WEEKS)):
            raise AssertionError("Program M requires exactly one reservation in every week")

    def _refresh_reservations(self) -> None:
        now = float(self.env.now)
        for record in self.program_m_reservations:
            if record.status == "scheduled" and now >= record.activation_time - 1e-9:
                record.status = "active"
                self.program_m_action_events.append(
                    {**asdict(record), "event": "reservation_activated"}
                )
            if record.status == "active" and now >= record.expiry_time - 1e-9:
                record.status = "expired_empty"
                self.program_m_action_events.append(
                    {**asdict(record), "event": "reservation_expired_empty"}
                )
                self._record_empty_movement(record)

    def _validate_program_m_risk_events(self) -> None:
        """Validate only the disclosed extension events, leaving native risks intact."""

        local_events = [
            event
            for event in self.risk_event_tape or ()
            if event.risk_id == "researcher_introduced_localized_access_disruption"
        ]
        starts_by_week: dict[int, int] = {}
        for event in local_events:
            if event.affected_cssu not in DESTINATIONS:
                raise ValueError("Program M local-access events require affected_cssu A/B")
            affected = set(map(int, event.affected_ops))
            if not affected or not affected <= LOCAL_BYPASS_OPS:
                raise ValueError("Program M local-access events may affect only Op10/Op12")
            if float(event.duration) <= 0.0:
                raise ValueError("Program M local-access events require positive duration")
            relative = float(event.start_time) - self.program_m_decision_start_time
            week = int(math.floor(relative / HOURS_PER_WEEK))
            starts_by_week[week] = starts_by_week.get(week, 0) + 1
        if any(count > 1 for count in starts_by_week.values()):
            raise ValueError("Program M permits at most one local-access event start per week")

    def _record_empty_movement(self, record: ReservationRecord) -> None:
        if record.week in self._program_m_recorded_empty_weeks:
            return
        self._program_m_recorded_empty_weeks.add(record.week)
        self.program_m_movement_events.append(
            {
                "week": record.week,
                "destination": record.destination,
                "departure_time": record.activation_time,
                "loaded": False,
                "order_j": None,
                "payload_rations": 0.0,
                "payload_capacity_rations": PAYLOAD_CAPACITY_RATIONS,
                "vehicle_hours": COMMITMENT_VEHICLE_HOURS,
            }
        )

    def _eligible_reservation(self, order: OrderRecord, qty: float) -> ReservationRecord | None:
        if not self.program_m_enabled or not self.program_m_bypass_local_route:
            return None
        destination = order.cssu_destination
        if destination not in DESTINATIONS or float(qty) > PAYLOAD_CAPACITY_RATIONS + 1e-9:
            return None
        # Calm-period movements remain transition-identical to the base DES.
        if not any(self._is_cssu_path_down(op, destination) for op in LOCAL_BYPASS_OPS):
            return None
        self._refresh_reservations()
        return next(
            (
                record
                for record in self.program_m_reservations
                if record.status == "active"
                and record.destination == destination
                and record.activation_time - 1e-9 <= float(self.env.now) < record.expiry_time
            ),
            None,
        )

    def _deliver_order_from_op9(self, order: OrderRecord, qty: float):
        reservation = self._eligible_reservation(order, qty)
        if reservation is None:
            yield from super()._deliver_order_from_op9(order, qty)
            return

        reservation.status = "used"
        reservation.used_at = float(self.env.now)
        reservation.order_j = int(order.j)
        reservation.payload_rations = float(qty)
        self.program_m_action_events.append(
            {**asdict(reservation), "event": "reservation_used"}
        )
        self.program_m_movement_events.append(
            {
                "week": reservation.week,
                "destination": reservation.destination,
                "departure_time": float(self.env.now),
                "loaded": True,
                "order_j": int(order.j),
                "payload_rations": float(qty),
                "payload_capacity_rations": PAYLOAD_CAPACITY_RATIONS,
                "vehicle_hours": COMMITMENT_VEHICLE_HOURS,
            }
        )
        yield from self._deliver_protected(order, float(qty))

    def _wait_protected_route_leg(self, order: OrderRecord, op_id: int):
        """Ignore only destination-local Op10/12 state, never aggregate state."""

        start = float(self.env.now)
        while self._is_down(op_id):
            yield self.env.timeout(1.0)
        end = float(self.env.now)
        if end > start:
            order.causal_wait_hours[f"op{op_id}_global_down"] = end - start

    def _deliver_protected(self, order: OrderRecord, qty: float):
        destination = str(order.cssu_destination)
        self._in_transit += qty
        self.cssu_in_transit[destination] += qty
        self.cssu_inbound_in_transit[destination] += qty

        yield from self._wait_protected_route_leg(order, 10)
        yield self.env.timeout(24.0)
        self.cssu_inbound_in_transit[destination] -= qty
        self.cssu_inventory[destination] += qty

        # Op11 is explicitly outside the bypass authority.
        yield from self._wait_order_for_cssu_operation(order, 11)
        self.cssu_inventory[destination] -= qty
        self.cssu_outbound_in_transit[destination] += qty

        yield from self._wait_protected_route_leg(order, 12)
        yield self.env.timeout(24.0)
        self._in_transit -= qty
        self.cssu_in_transit[destination] -= qty
        self.cssu_outbound_in_transit[destination] -= qty
        self.cssu_delivered[destination] += qty
        self.cssu_delivery_events.append((float(self.env.now), destination, qty))
        self.total_theatre_inflow += qty
        self.total_delivered += qty
        self.total_order_fulfilled += qty
        self.delivery_events.append((float(self.env.now), qty))
        order.in_flight_qty = max(0.0, float(order.in_flight_qty) - qty)
        if order.remaining_qty <= 1e-9 and order.in_flight_qty <= 1e-9:
            self._finalize_pending_backorder(order)

    def _risk_event_tape_event(self, event: RiskEvent):
        is_program_event = (
            event.affected_cssu in DESTINATIONS
            and bool(set(event.affected_ops) & LOCAL_BYPASS_OPS)
        )
        if is_program_event:
            self.program_m_risk_event_ledger.append(
                {
                    "event": "localized_access_started",
                    "risk_id": event.risk_id,
                    "time": float(self.env.now),
                    "destination": event.affected_cssu,
                    "affected_ops": list(event.affected_ops),
                    "duration": float(event.duration),
                }
            )
        yield from super()._risk_event_tape_event(event)
        if is_program_event:
            self.program_m_risk_event_ledger.append(
                {
                    "event": "localized_access_ended",
                    "risk_id": event.risk_id,
                    "time": float(self.env.now),
                    "destination": event.affected_cssu,
                    "affected_ops": list(event.affected_ops),
                }
            )

    def get_program_m_observation(self) -> dict[str, float]:
        """Return only frozen deployable current/past fields."""

        if not self.program_m_enabled:
            raise RuntimeError("Program M is disabled")
        self._refresh_reservations()
        now = float(self.env.now)
        elapsed = now - self.program_m_decision_start_time
        week = int(math.floor(elapsed / HOURS_PER_WEEK)) if elapsed >= 0 else -1
        warning = next(
            (row for row in self.program_m_warning_records if row.week == week), None
        )
        obs = self.get_cssu_observation()
        obs.update(
            {
                "program_m_week": float(week),
                "program_m_week_phase": float((elapsed % HOURS_PER_WEEK) / HOURS_PER_WEEK),
                "program_m_reservation_available": float(
                    any(row.status == "active" for row in self.program_m_reservations)
                ),
                "program_m_warning_A": float(warning.warning_A if warning else 0),
                "program_m_warning_B": float(warning.warning_B if warning else 0),
                "program_m_past_warning_A_count": float(
                    sum(row.warning_A for row in self.program_m_warning_records if row.observed_at < now)
                ),
                "program_m_past_warning_B_count": float(
                    sum(row.warning_B for row in self.program_m_warning_records if row.observed_at < now)
                ),
                "program_m_past_local_A_count": float(
                    sum(
                        row["event"] == "localized_access_ended" and row["destination"] == "A"
                        for row in self.program_m_risk_event_ledger
                    )
                ),
                "program_m_past_local_B_count": float(
                    sum(
                        row["event"] == "localized_access_ended" and row["destination"] == "B"
                        for row in self.program_m_risk_event_ledger
                    )
                ),
            }
        )
        assert_program_m_observation_whitelist(obs)
        return obs

    def program_m_ledger(self) -> dict[str, Any]:
        """Return complete action, movement, event and conserved-resource ledgers."""

        self._refresh_reservations()
        if float(self.env.now) >= (
            self.program_m_decision_start_time + DECISION_WEEKS * HOURS_PER_WEEK
        ):
            for record in self.program_m_reservations:
                if record.status == "active":
                    record.status = "expired_empty"
                    self._record_empty_movement(record)
        count = len(self.program_m_reservations)
        loaded = [row for row in self.program_m_movement_events if row["loaded"]]
        committed_vehicle_hours = count * COMMITMENT_VEHICLE_HOURS
        return {
            "contract_id": CONTRACT_ID,
            "actions": [asdict(row) for row in self.program_m_reservations],
            "action_events": list(self.program_m_action_events),
            "movements": list(self.program_m_movement_events),
            "risk_events": list(self.program_m_risk_event_ledger),
            "resources": {
                "reserved_slots": count,
                "total_committed_departures": count,
                "reserved_payload_capacity_rations": count * PAYLOAD_CAPACITY_RATIONS,
                "reserved_vehicle_hours": committed_vehicle_hours,
                "total_committed_vehicle_hours": committed_vehicle_hours,
                "loaded_departures": len(loaded),
                "empty_departures": count - len(loaded),
                "actual_payload_rations": sum(float(row["payload_rations"]) for row in loaded),
                "actual_loaded_vehicle_hours": len(loaded) * COMMITMENT_VEHICLE_HOURS,
            },
            "flow_ledger": self.flow_ledger(),
        }


_PROGRAM_M_EXTRA_OBSERVATION_FIELDS = frozenset(
    {
        "program_m_week",
        "program_m_week_phase",
        "program_m_reservation_available",
        "program_m_warning_A",
        "program_m_warning_B",
        "program_m_past_warning_A_count",
        "program_m_past_warning_B_count",
        "program_m_past_local_A_count",
        "program_m_past_local_B_count",
    }
)


def assert_program_m_observation_whitelist(observation: Mapping[str, Any]) -> None:
    """Fail closed against privileged tape, future, oracle or latent fields."""

    base_probe = MFSCSimulation(cssu_topology_mode="split_v1").get_cssu_observation()
    allowed = frozenset(base_probe) | _PROGRAM_M_EXTRA_OBSERVATION_FIELDS
    unexpected = set(observation) - allowed
    if unexpected:
        raise AssertionError(f"Program M observation contains non-whitelisted fields: {sorted(unexpected)}")
    forbidden_fragments = (
        "seed",
        "tape",
        "future",
        "oracle",
        "latent",
        "duration",
        "recovery_time",
        "next_event",
    )
    leaked = [key for key in observation if any(token in key.lower() for token in forbidden_fragments)]
    if leaked:
        raise AssertionError(f"Program M observation leaks privileged fields: {sorted(leaked)}")
