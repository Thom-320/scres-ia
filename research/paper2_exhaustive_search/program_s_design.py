"""Frozen Program S tape/alarm design helpers.

The established wartime tape generator supplies common-random-number base
streams.  This adapter translates the Program S cell vocabulary and performs a
second fail-closed validation before the tape reaches SimPy.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np

from research.paper2_exhaustive_search.war_stress_risk_tapes import build_risk_tape
from supply_chain.program_s_risk_interaction import (
    CAPACITY_FAMILIES,
    OperationalAlarm,
    ProgramSCell,
    validate_program_s_risk_tape,
)


_MASK_TRANSLATION = {
    "PRODUCTION_QUALITY_SURGE": "PRODUCTION_QUALITY_SURGE",
    "LOC_SURGE": "LOC_SURGE",
    "CROSS_ECHELON_SURGE": "THEATER_CAPACITY_SURGE",
}
_COUPLING_TRANSLATION = {
    "independent": "independent",
    "disruption_leads_r24_72h": "disruption_leads_surge_72h",
    "coincident": "coincident",
    "r24_leads_disruption_72h": "surge_leads_disruption_72h",
}


def program_s_tape_configuration(cell: ProgramSCell) -> dict[str, Any]:
    multipliers: dict[str, float] = {}
    for risk_id, value in cell.phi_by_risk.items():
        multipliers[f"phi_{risk_id}"] = float(value)
    for risk_id, value in cell.psi_by_risk.items():
        multipliers[f"psi_{risk_id}"] = float(value)
    if "R14" in cell.phi_by_risk or cell.mask == "PRODUCTION_QUALITY_SURGE":
        # In the source generator phi_R14 is the binomial defect-probability
        # multiplier, not an event-frequency or generic impact multiplier.
        multipliers["phi_R14"] = float(cell.r14_probability_multiplier)
    return {
        "config_id": f"program_s_{cell.cell_id}",
        "mask": _MASK_TRANSLATION[cell.mask],
        "coupling": _COUPLING_TRANSLATION[cell.coupling],
        "physical_multipliers": multipliers,
    }


def build_program_s_risk_tape(
    cell: ProgramSCell,
    *,
    tape_id: int,
    horizon_hours: float,
) -> dict[str, Any]:
    built = build_risk_tape(
        program_s_tape_configuration(cell),
        tape_id=int(tape_id),
        horizon_hours=float(horizon_hours),
        start_hour=0.0,
    )
    events = validate_program_s_risk_tape(built.events, mask=cell.mask)
    return {
        "cell_id": cell.cell_id,
        "tape_id": int(tape_id),
        "events": events,
        "base_stream_sha256": built.base_stream_sha256,
        "event_tape_sha256": built.event_tape_sha256,
        "r3_event_count": int(built.r3_event_count),
    }


def _alarm_seed(tape_id: int, cell_id: str) -> int:
    digest = hashlib.sha256(f"program_s_alarm_v1:{tape_id}:{cell_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _family_for_ops(affected_ops: Sequence[int]) -> str:
    scores = {
        family: len(set(map(int, affected_ops)).intersection(operations))
        for family, operations in CAPACITY_FAMILIES.items()
    }
    return max(scores, key=lambda family: (scores[family], family))


def _severity(duration: float) -> str:
    if float(duration) >= 72.0:
        return "high"
    if float(duration) >= 12.0:
        return "moderate"
    return "low"


def build_operational_alarms(
    cell: ProgramSCell,
    *,
    tape_id: int,
    events: Sequence[Mapping[str, Any]],
    horizon_hours: float,
    contingent_product_share_by_week: Sequence[float] | None = None,
) -> tuple[OperationalAlarm, ...]:
    """Legacy v1 alarm generator, prohibited for amended Program S-NATIVE.

    The v1.1 amendment separates anticipatory information into Program S-P.
    This function is retained only so the frozen S0 implementation remains
    importable; calling it for a thesis-native cell now fails closed.
    """
    if cell.stratum == "THESIS_NATIVE_INDEPENDENT":
        raise RuntimeError(
            "anticipatory alarms are forbidden in amended Program S-NATIVE; "
            "use the separately frozen Program S-P generator"
        )
    accuracy = float(cell.alarm_balanced_accuracy)
    lead = float(cell.alarm_lead_hours)
    rng = np.random.default_rng(_alarm_seed(int(tape_id), cell.cell_id))
    shares = tuple(contingent_product_share_by_week or ())
    alarms: list[OperationalAlarm] = []

    # Aggregate same-family events with the same onset (notably simultaneous
    # R21 operation outages) into one operational warning.
    event_groups: dict[tuple[float, str], list[Mapping[str, Any]]] = {}
    for event in events:
        if str(event["risk_id"]) == "R24":
            continue
        family = _family_for_ops(event.get("affected_ops", ()))
        event_groups.setdefault((float(event["start_time"]), family), []).append(event)

    occupied: set[tuple[int, str]] = set()
    for (start, family), rows in sorted(event_groups.items()):
        week = max(0, int(start // 168.0))
        occupied.add((week, family))
        if rng.random() > accuracy:
            continue
        end = max(float(row.get("end_time", start)) for row in rows)
        duration = max(float(row.get("duration", 0.0)) for row in rows)
        share = float(shares[week]) if week < len(shares) else 0.5
        alarms.append(
            OperationalAlarm(
                issued_at=max(0.0, start - lead),
                effective_window=(start, end),
                expected_capacity_family=family,
                expected_severity_bin=_severity(duration),
                expected_contingent_product_share=share,
                confidence=accuracy,
            )
        )

    # Matched no-event opportunities implement specificity = accuracy.
    weeks = int(math.ceil(float(horizon_hours) / 168.0))
    for week in range(weeks):
        for family in CAPACITY_FAMILIES:
            if (week, family) in occupied or rng.random() <= accuracy:
                continue
            start = week * 168.0 + 84.0
            share = float(shares[week]) if week < len(shares) else 0.5
            alarms.append(
                OperationalAlarm(
                    issued_at=max(0.0, start - lead),
                    effective_window=(start, start + 24.0),
                    expected_capacity_family=family,
                    expected_severity_bin="moderate",
                    expected_contingent_product_share=share,
                    confidence=max(0.0, 1.0 - accuracy),
                )
            )
    return tuple(
        sorted(
            alarms,
            key=lambda alarm: (
                alarm.issued_at,
                alarm.expected_capacity_family,
                alarm.effective_window,
            ),
        )
    )
