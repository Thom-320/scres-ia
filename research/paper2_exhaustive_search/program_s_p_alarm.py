"""Non-clairvoyant, risk-family-specific alarm generator for Program S-P.

This annex is deliberately disconnected from Program S-NATIVE.  Event tapes
are used only to simulate an imperfect detector.  The emitted alarm contains a
jittered onset bin and a confused severity class; exact event start, end,
duration, risk ID, seed, and generator parameters are never exposed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np


RISK_TO_FAMILY = {
    "R24": "planned_contingent_demand",
    "R21": "weather_natural_disaster",
    "R22": "loc_threat",
    "R11": "condition_monitoring",
    "R14": "condition_monitoring",
}
SEVERITIES = ("low", "moderate", "high")


@dataclass(frozen=True)
class AlarmFamilySpec:
    lead_hours: float
    detection_probability: float
    specificity: float
    onset_bin_hours: float

    def __post_init__(self) -> None:
        if self.lead_hours < 0 or self.onset_bin_hours <= 0:
            raise ValueError("alarm time parameters must be non-negative")
        if not 0.0 <= self.detection_probability <= 1.0:
            raise ValueError("detection_probability must be in [0,1]")
        if not 0.0 <= self.specificity <= 1.0:
            raise ValueError("specificity must be in [0,1]")


@dataclass(frozen=True)
class ProgramSPAlarm:
    issued_at: float
    predicted_onset_window: tuple[float, float]
    expected_capacity_family: str
    expected_severity_bin: str
    expected_contingent_product_share_bin: str
    confidence_bin: str


def _seed(tape_id: int, model_id: str) -> int:
    digest = hashlib.sha256(f"program_s_p_alarm_v1:{tape_id}:{model_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _severity_from_duration(duration: float) -> int:
    return 2 if duration >= 72 else 1 if duration >= 12 else 0


def _confused_severity(true_index: int, rng: np.random.Generator) -> str:
    # Frozen adjacent-category confusion; even detected events do not disclose
    # realized duration or its exact bin deterministically.
    probabilities = {
        0: (0.70, 0.25, 0.05),
        1: (0.15, 0.70, 0.15),
        2: (0.05, 0.25, 0.70),
    }[int(true_index)]
    return SEVERITIES[int(rng.choice(3, p=probabilities))]


def _binned_window(start: float, width: float, rng: np.random.Generator) -> tuple[float, float]:
    # Jitter by a full bin before rounding.  The true onset is therefore not
    # recoverable from the emitted endpoints.
    jittered = max(0.0, float(start) + float(rng.uniform(-width, width)))
    lower = math.floor(jittered / width) * width
    if abs(lower - float(start)) <= 1e-12:
        lower = max(0.0, lower - width) if rng.random() < 0.5 else lower + width
    return (float(lower), float(lower + width))


def _confidence_bin(value: float) -> str:
    return "high" if value >= 0.80 else "moderate" if value >= 0.65 else "low"


def _forecast_window(
    start: float, spec: AlarmFamilySpec, rng: np.random.Generator
) -> tuple[float, tuple[float, float]]:
    issued_at = max(
        0.0,
        float(start)
        - spec.lead_hours
        + float(rng.uniform(-0.5 * spec.onset_bin_hours, 0.5 * spec.onset_bin_hours)),
    )
    lower, upper = _binned_window(start, spec.onset_bin_hours, rng)
    if lower < issued_at - 1e-12:
        lower = math.ceil(issued_at / spec.onset_bin_hours) * spec.onset_bin_hours
        if abs(lower - float(start)) <= 1e-12:
            lower += spec.onset_bin_hours
        upper = lower + spec.onset_bin_hours
    return issued_at, (float(lower), float(upper))


def build_risk_specific_alarms(
    *,
    tape_id: int,
    model_id: str,
    events: Sequence[Mapping[str, Any]],
    specs: Mapping[str, AlarmFamilySpec],
    horizon_hours: float,
    contingent_product_share_by_week: Sequence[float] = (),
) -> tuple[ProgramSPAlarm, ...]:
    rng = np.random.default_rng(_seed(int(tape_id), str(model_id)))
    alarms: list[ProgramSPAlarm] = []
    occupied: set[tuple[int, str]] = set()

    for event in sorted(events, key=lambda row: (float(row["start_time"]), str(row["risk_id"]))):
        risk_id = str(event["risk_id"])
        family = RISK_TO_FAMILY.get(risk_id)
        if family is None or family not in specs:
            continue
        spec = specs[family]
        start = float(event["start_time"])
        week = max(0, int(start // 168.0))
        occupied.add((week, family))
        if rng.random() > spec.detection_probability:
            continue
        share = (
            float(contingent_product_share_by_week[week])
            if week < len(contingent_product_share_by_week)
            else 0.5
        )
        issued_at, predicted_window = _forecast_window(start, spec, rng)
        alarms.append(
            ProgramSPAlarm(
                issued_at=issued_at,
                predicted_onset_window=predicted_window,
                expected_capacity_family=family,
                expected_severity_bin=_confused_severity(
                    _severity_from_duration(float(event.get("duration", 0.0))), rng
                ),
                expected_contingent_product_share_bin=(
                    "C_high" if share >= 0.67 else "H_high" if share <= 0.33 else "balanced"
                ),
                confidence_bin=_confidence_bin(spec.detection_probability),
            )
        )

    weeks = int(math.ceil(float(horizon_hours) / 168.0))
    for family, spec in sorted(specs.items()):
        for week in range(weeks):
            if (week, family) in occupied or rng.random() <= spec.specificity:
                continue
            nominal = week * 168.0 + float(rng.uniform(0.0, 168.0))
            issued_at, predicted_window = _forecast_window(nominal, spec, rng)
            alarms.append(
                ProgramSPAlarm(
                    issued_at=issued_at,
                    predicted_onset_window=predicted_window,
                    expected_capacity_family=family,
                    expected_severity_bin=SEVERITIES[int(rng.integers(0, 3))],
                    expected_contingent_product_share_bin="balanced",
                    confidence_bin=_confidence_bin(1.0 - spec.specificity),
                )
            )

    return tuple(
        sorted(
            alarms,
            key=lambda alarm: (
                alarm.issued_at,
                alarm.expected_capacity_family,
                alarm.predicted_onset_window,
            ),
        )
    )


def alarm_digest(alarms: Sequence[ProgramSPAlarm]) -> str:
    payload = [asdict(alarm) for alarm in alarms]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
