from __future__ import annotations

from scripts import run_garrido_v0_recovery_gate_v2 as gate
from supply_chain.garrido_v0_recovery import RECOVERY_WINDOW_HOURS, risk_event_rows


def test_event_clock_precedes_the_procurement_cycle_by_one_hour() -> None:
    assert gate.EVENT_ONSET_HOURS == 4031.0
    assert gate.EVENT_ONSET_HOURS < 4032.0 <= gate.EVENT_ONSET_HOURS + 1.0
    assert gate.HORIZON_HOURS >= gate.EVENT_ONSET_HOURS + RECOVERY_WINDOW_HOURS


def test_v2_does_not_change_the_frozen_risk_magnitudes() -> None:
    assert risk_event_rows("R12", onset_hours=gate.EVENT_ONSET_HOURS)[0]["duration"] == 672.0
    assert risk_event_rows("R13", onset_hours=gate.EVENT_ONSET_HOURS)[0]["duration"] == 120.0
    assert risk_event_rows("R24", onset_hours=gate.EVENT_ONSET_HOURS)[0]["magnitude"] == 2495.0


def test_sentinel_postures_are_unchanged_from_v1() -> None:
    from scripts.run_garrido_v0_recovery_gate_v1 import SENTINEL_POSTURES as v1

    assert gate.SENTINEL_POSTURES == v1
