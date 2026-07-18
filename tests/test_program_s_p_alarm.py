from __future__ import annotations

from dataclasses import asdict

from research.paper2_exhaustive_search.program_s_p_alarm import (
    AlarmFamilySpec,
    alarm_digest,
    build_risk_specific_alarms,
)


def specs():
    return {
        "planned_contingent_demand": AlarmFamilySpec(72, 0.85, 0.85, 24),
        "weather_natural_disaster": AlarmFamilySpec(24, 0.70, 0.80, 24),
        "loc_threat": AlarmFamilySpec(0, 0.60, 0.70, 24),
        "condition_monitoring": AlarmFamilySpec(0, 0.65, 0.80, 12),
    }


def events():
    return [
        {"risk_id": "R21", "start_time": 96.0, "duration": 120.0},
        {"risk_id": "R22", "start_time": 240.0, "duration": 24.0},
        {"risk_id": "R24", "start_time": 360.0, "duration": 0.0},
        {"risk_id": "R14", "start_time": 500.0, "duration": 0.0},
    ]


def test_alarm_annex_is_reproducible_and_exposes_no_exact_event_fields() -> None:
    first = build_risk_specific_alarms(
        tape_id=7430001,
        model_id="burned_test",
        events=events(),
        specs=specs(),
        horizon_hours=8 * 168,
    )
    second = build_risk_specific_alarms(
        tape_id=7430001,
        model_id="burned_test",
        events=events(),
        specs=specs(),
        horizon_hours=8 * 168,
    )
    assert first == second
    assert alarm_digest(first) == alarm_digest(second)
    forbidden = {"risk_id", "start_time", "end_time", "duration", "seed", "tape_id"}
    assert all(not forbidden.intersection(asdict(alarm)) for alarm in first)
    assert all(alarm.issued_at <= alarm.predicted_onset_window[0] for alarm in first)


def test_alarm_annex_does_not_emit_exact_duration_or_exact_onset_pair() -> None:
    alarms = build_risk_specific_alarms(
        tape_id=7430002,
        model_id="burned_test",
        events=events(),
        specs=specs(),
        horizon_hours=8 * 168,
    )
    event_pairs = {(float(row["start_time"]), float(row["start_time"]) + float(row["duration"])) for row in events()}
    assert all(alarm.predicted_onset_window not in event_pairs for alarm in alarms)
    assert all(alarm.expected_severity_bin in {"low", "moderate", "high"} for alarm in alarms)
