"""Prospective Program S risk x product x timing extension.

This module deliberately leaves the Program O/O-R contracts untouched.  Risk
events are materialized before policy evaluation and replayed relative to the
Program O decision start, so the neutral prefix remains risk-free and identical
across policies.  Operational alarms are a lossy projection of that tape; they
never expose risk IDs, generator parameters, seeds, or future demand labels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Iterable, Literal, Mapping, Sequence

from supply_chain.program_o_full_des import (
    PRODUCTS,
    ProgramOFullDESSimulation,
)


PROGRAM_S_STRATA = frozenset(
    {"THESIS_NATIVE_INDEPENDENT", "RESEARCHER_WARTIME_COUPLED"}
)
PROGRAM_S_MASKS: dict[str, tuple[str, ...]] = {
    "PRODUCTION_QUALITY_SURGE": ("R11", "R14", "R24"),
    "LOC_SURGE": ("R22", "R24"),
    "CROSS_ECHELON_SURGE": ("R21", "R23", "R24"),
}
PROGRAM_S_COUPLINGS = frozenset(
    {"independent", "disruption_leads_r24_72h", "coincident", "r24_leads_disruption_72h"}
)
CAPACITY_FAMILIES: dict[str, tuple[int, ...]] = {
    "production": (5, 6, 7),
    "loc": (4, 8, 10, 12),
    "cross_echelon": (3, 5, 6, 7, 9, 11),
}


def _digest(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ProgramSCell:
    stratum: Literal[
        "THESIS_NATIVE_INDEPENDENT", "RESEARCHER_WARTIME_COUPLED"
    ]
    mask: str
    coupling: str
    phi_by_risk: Mapping[str, float]
    psi_by_risk: Mapping[str, float]
    r14_probability_multiplier: float
    baseline_capacity_multiplier: float
    regime_persistence: float
    dominant_share: float
    alarm_lead_hours: float
    alarm_balanced_accuracy: float

    def __post_init__(self) -> None:
        if self.stratum not in PROGRAM_S_STRATA:
            raise ValueError(f"unknown Program S stratum: {self.stratum}")
        if self.mask not in PROGRAM_S_MASKS:
            raise ValueError(f"unknown Program S mask: {self.mask}")
        if self.coupling not in PROGRAM_S_COUPLINGS:
            raise ValueError(f"unknown Program S coupling: {self.coupling}")
        if self.stratum == "THESIS_NATIVE_INDEPENDENT" and self.coupling != "independent":
            raise ValueError("thesis-native cells require independent risk timing")
        if self.stratum == "RESEARCHER_WARTIME_COUPLED" and self.coupling == "independent":
            raise ValueError("wartime cells require an explicit R24 coupling")
        allowed = set(PROGRAM_S_MASKS[self.mask])
        for name, values in (
            ("phi_by_risk", self.phi_by_risk),
            ("psi_by_risk", self.psi_by_risk),
        ):
            foreign = set(map(str, values)) - allowed
            if foreign:
                raise ValueError(f"{name} contains risks outside the mask: {sorted(foreign)}")
            if any(float(value) <= 0.0 for value in values.values()):
                raise ValueError(f"{name} multipliers must be positive")
        if not 0.0 < float(self.r14_probability_multiplier):
            raise ValueError("r14_probability_multiplier must be positive")
        if self.mask != "PRODUCTION_QUALITY_SURGE" and abs(
            float(self.r14_probability_multiplier) - 1.0
        ) > 1e-12:
            raise ValueError("R14 probability can change only in the production mask")
        if float(self.baseline_capacity_multiplier) not in {0.9, 1.0, 1.1}:
            raise ValueError("baseline_capacity_multiplier must be 0.9, 1.0, or 1.1")
        if not 0.5 <= float(self.regime_persistence) <= 1.0:
            raise ValueError("regime_persistence must be in [0.5, 1]")
        if not 0.5 <= float(self.dominant_share) <= 1.0:
            raise ValueError("dominant_share must be in [0.5, 1]")
        if float(self.alarm_lead_hours) not in {0.0, 24.0, 72.0}:
            raise ValueError("alarm_lead_hours must be 0, 24, or 72")
        if float(self.alarm_balanced_accuracy) not in {0.50, 0.70, 0.85}:
            raise ValueError("alarm_balanced_accuracy must be 0.50, 0.70, or 0.85")

    @property
    def cell_id(self) -> str:
        return _digest(asdict(self))[:16]


@dataclass(frozen=True)
class OperationalAlarm:
    issued_at: float
    effective_window: tuple[float, float]
    expected_capacity_family: str
    expected_severity_bin: str
    expected_contingent_product_share: float
    confidence: float

    def __post_init__(self) -> None:
        start, end = map(float, self.effective_window)
        if float(self.issued_at) > start + 1e-12:
            raise ValueError("an alarm must be issued no later than its effective window")
        if end < start:
            raise ValueError("alarm effective_window is reversed")
        if self.expected_capacity_family not in CAPACITY_FAMILIES:
            raise ValueError("alarm exposes an unknown capacity family")
        if self.expected_severity_bin not in {"low", "moderate", "high"}:
            raise ValueError("alarm severity must be low, moderate, or high")
        if not 0.0 <= float(self.expected_contingent_product_share) <= 1.0:
            raise ValueError("expected contingent product share must be a probability")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("alarm confidence must be a probability")


def validate_program_s_risk_tape(
    events: Iterable[Mapping[str, Any]], *, mask: str
) -> tuple[dict[str, Any], ...]:
    """Fail closed on foreign risks, R3, malformed times, and duplicated native events."""
    if mask not in PROGRAM_S_MASKS:
        raise ValueError(f"unknown Program S mask: {mask}")
    allowed = set(PROGRAM_S_MASKS[mask])
    normalized: list[dict[str, Any]] = []
    identities: set[tuple[Any, ...]] = set()
    for raw in events:
        row = dict(raw)
        risk_id = str(row["risk_id"])
        if risk_id == "R3":
            raise ValueError("R3 is diagnostic-only and forbidden in Program S cells")
        if risk_id not in allowed:
            raise ValueError(f"risk {risk_id} is outside mask {mask}")
        start = float(row["start_time"])
        duration = max(0.0, float(row.get("duration", 0.0)))
        end = float(row.get("end_time", start + duration))
        if start < 0.0 or abs(end - (start + duration)) > 1e-9:
            raise ValueError("risk event has inconsistent relative timing")
        affected = tuple(int(value) for value in row.get("affected_ops", ()))
        identity = (risk_id, start, affected)
        if identity in identities:
            raise ValueError("duplicated native/cluster risk event")
        identities.add(identity)
        row.update(
            start_time=start,
            end_time=end,
            duration=duration,
            affected_ops=list(affected),
            magnitude=float(row.get("magnitude", 1.0)),
            unit=str(row.get("unit", "incidents")),
        )
        normalized.append(row)
    return tuple(
        sorted(
            normalized,
            key=lambda row: (
                float(row["start_time"]),
                str(row["risk_id"]),
                tuple(row["affected_ops"]),
            ),
        )
    )


class ProgramSRiskAwareSimulation(ProgramOFullDESSimulation):
    """Program O physics with a policy-independent, decision-relative risk tape."""

    def __init__(
        self,
        *,
        cell: ProgramSCell,
        risk_event_tape: Iterable[Mapping[str, Any]],
        operational_alarms: Sequence[OperationalAlarm] = (),
        **program_o_kwargs: Any,
    ) -> None:
        if "risks_enabled" in program_o_kwargs or "risk_event_tape" in program_o_kwargs:
            raise TypeError("Program S owns risks_enabled and risk_event_tape")
        super().__init__(
            regime_persistence=float(cell.regime_persistence),
            dominant_share=float(cell.dominant_share),
            risks_enabled=False,
            **program_o_kwargs,
        )
        self.program_s_cell = cell
        self.program_s_relative_risk_tape = validate_program_s_risk_tape(
            risk_event_tape, mask=cell.mask
        )
        self.program_s_risk_tape_sha256 = _digest(self.program_s_relative_risk_tape)
        self.program_s_alarms = tuple(sorted(operational_alarms, key=lambda row: row.issued_at))
        self.program_s_alarm_sha256 = _digest([asdict(row) for row in self.program_s_alarms])
        self.program_s_replay_started = False

    def _assembly_output_capacity(self, nominal_capacity: float) -> float:
        scaled = float(nominal_capacity) * float(
            self.program_s_cell.baseline_capacity_multiplier
        )
        return super()._assembly_output_capacity(scaled)

    def _program_s_risk_replay(self):
        yield self.program_o_warmup_event
        origin = float(self.program_o_decision_start or self.env.now)
        self.program_s_replay_started = True
        for row in self.program_s_relative_risk_tape:
            target = origin + float(row["start_time"])
            if target > float(self.env.now):
                yield self.env.timeout(target - float(self.env.now))
            absolute = dict(row)
            absolute["start_time"] = target
            absolute["end_time"] = target + float(row["duration"])
            event = self._normalize_risk_event_tape([absolute])[0]
            self.env.process(self._risk_event_tape_event(event))

    def _start_processes(self) -> None:
        super()._start_processes()
        self.env.process(self._program_s_risk_replay())

    def current_operational_alarm(self) -> OperationalAlarm | None:
        if self.program_o_decision_start is None:
            return None
        relative_now = float(self.env.now) - float(self.program_o_decision_start)
        visible = [row for row in self.program_s_alarms if float(row.issued_at) <= relative_now]
        return visible[-1] if visible else None

    def program_s_observation(self) -> dict[str, Any]:
        """Return only deployable state and the aggregated alarm projection."""
        now = float(self.env.now)
        backlog_quantity = {product_id: 0.0 for product_id in PRODUCTS}
        backlog_age = {product_id: 0.0 for product_id in PRODUCTS}
        for order in self.pending_backorders:
            product_id = str(getattr(order, "requested_product_id", ""))
            if product_id in backlog_quantity and not bool(order.lost):
                backlog_quantity[product_id] += float(order.remaining_qty)
                backlog_age[product_id] = max(backlog_age[product_id], now - float(order.OPTj))
        alarm = self.current_operational_alarm()
        payload = {
            "time_since_decision_start": (
                None
                if self.program_o_decision_start is None
                else now - float(self.program_o_decision_start)
            ),
            "inventory": {
                product_id: self.program_o_ledger.quantity("rations_sb", product_id)
                for product_id in PRODUCTS
            },
            "wip": {
                product_id: sum(
                    self.program_o_ledger.quantity(node, product_id)
                    for node in ("pending_batch", "rations_al", "op8_transit")
                )
                for product_id in PRODUCTS
            },
            "pipeline": {
                product_id: sum(
                    float(target.remaining)
                    for target in tuple(self.program_o_target_queue)
                    if target.product_id == product_id
                )
                + (
                    float(self.program_o_active_target.remaining)
                    if self.program_o_active_target is not None
                    and self.program_o_active_target.product_id == product_id
                    else 0.0
                )
                for product_id in PRODUCTS
            },
            "backlog_quantity": backlog_quantity,
            "max_backlog_age": backlog_age,
            "available_capacity": {
                family: sum(not self._is_down(op_id) for op_id in operations) / len(operations)
                for family, operations in CAPACITY_FAMILIES.items()
            },
            "realized_downtime_by_echelon": {
                family: sum(
                    max(
                        0.0,
                        min(now, float(event.end_time)) - float(event.start_time),
                    )
                    for event in self.risk_events
                    if any(op_id in operations for op_id in event.affected_ops)
                )
                for family, operations in CAPACITY_FAMILIES.items()
            },
            "alarm": None if alarm is None else asdict(alarm),
        }
        # The whitelist is intentionally explicit: no cell fields, risk IDs,
        # seed, tape hash, future labels, or oracle calendar are returned.
        payload["observation_sha256"] = _digest(payload)
        return payload
