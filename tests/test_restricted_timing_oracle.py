from __future__ import annotations

import csv
from types import SimpleNamespace

from research.paper2_exhaustive_search.restricted_timing_oracle import (
    ObservableEwmaTrigger,
    Posture,
    ScheduleSpec,
    paired_promotion_summary,
    periodic_binary_calendars,
    placebo_signal_series,
    posture_from_label,
    privileged_windows,
    safe_against,
    schedule_is_high,
    select_frozen_postures,
    evaluate_schedule,
)


def _event(risk, start, end):
    return SimpleNamespace(risk_id=risk, start_time=float(start), end_time=float(end))


def test_complete_periodic_frontier_has_256_unique_calendars() -> None:
    calendars = periodic_binary_calendars()
    assert len(calendars) == 256
    assert len({tuple(spec.payload) for spec in calendars}) == 256
    assert all(len(spec.payload) == 8 for spec in calendars)


def test_privileged_windows_ignore_r3_and_union_overlaps() -> None:
    windows = privileged_windows(
        [_event("R22", 200, 240), _event("R24", 250, 250), _event("R3", 100, 800)],
        entry_offset_hours=-72,
        exit_offset_hours=72,
    )
    assert windows == [(128.0, 322.0)]


def test_daily_and_weekly_privileged_schedules_are_distinct() -> None:
    events = [_event("R22", 250, 274)]
    daily = ScheduleSpec("restricted_privileged", "d", -24.0)
    weekly = ScheduleSpec("weekly_privileged", "w", -24.0)
    assert not schedule_is_high(daily, now=200, treatment_start=0, risk_events=events)
    assert schedule_is_high(daily, now=240, treatment_start=0, risk_events=events)
    assert schedule_is_high(weekly, now=168, treatment_start=0, risk_events=events)


def test_observable_trigger_has_hysteresis_and_uses_only_supplied_signal() -> None:
    trigger = ObservableEwmaTrigger(decay=0.0, enter=0.5, exit=0.1)
    assert trigger.decide(
        now=0.0, observed_signal=1.0, backlog_age_hours=0.0, inventory_shortfall_fraction=0.0
    )
    assert trigger.decide(
        now=24.0, observed_signal=0.0, backlog_age_hours=0.0, inventory_shortfall_fraction=0.0
    )
    assert not trigger.decide(
        now=72.0, observed_signal=0.0, backlog_age_hours=0.0, inventory_shortfall_fraction=0.0
    )


def test_placebos_are_deterministic_and_stale_is_exactly_seven_days() -> None:
    signal = [float(index) for index in range(12)]
    assert placebo_signal_series(signal, family="stale_168h", seed=1) == [0.0] * 7 + signal[:5]
    one = placebo_signal_series(signal, family="shuffled_within_tape", seed=9)
    two = placebo_signal_series(signal, family="shuffled_within_tape", seed=9)
    assert one == two
    assert sorted(one) == signal


def test_posture_parser_matches_risk_runner_labels() -> None:
    posture = posture_from_label("f0.375_S2")
    assert posture.buffer_fraction == 0.375
    assert posture.shifts == 2
    assert posture.nominal_resource == 0.4375


def test_first_passing_regime_selection_is_frozen(tmp_path) -> None:
    path = tmp_path / "rows.csv"
    fields = [
        "profile", "candidate", "candidate_resource_nominal", "ret_excel",
        "ration_ret_excel", "ret_excel_cvar10", "lost_orders", "backorder_qty_final",
        "backlog_age_max", "service_loss_auc_ration_hours", "resource",
    ]
    rows = []
    for profile in ("R2_current", "R2_OAT_R24_increased", "R2_OAT_R22_increased", "Cf19"):
        for candidate, ret in (("f0_S1", 0.50), ("f0.5_S1", 0.50)):
            if profile == "R2_OAT_R24_increased" and candidate == "f0.5_S1":
                ret = 0.53
            rows.append({
                "profile": profile,
                "candidate": candidate,
                "candidate_resource_nominal": 0.0 if candidate == "f0_S1" else 0.25,
                "ret_excel": ret,
                "ration_ret_excel": 0.5,
                "ret_excel_cvar10": 0.5,
                "lost_orders": 0,
                "backorder_qty_final": 0,
                "backlog_age_max": 0,
                "service_loss_auc_ration_hours": 0,
                "resource": 0,
            })
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    selection = select_frozen_postures(path)
    assert selection is not None
    low, high, profile = selection
    assert low.label == "f0_S1"
    assert high.label == "f0.5_S1"
    assert profile == "R2_OAT_R24_increased"


def _panel(ret: float, *, lost: float = 0.0, resources: float = 1.0):
    return {
        "ret_excel": ret,
        "ret_excel_full_ledger": ret,
        "ration_ret_excel": ret,
        "ret_excel_cvar10": ret,
        "worst_node_or_product_fill": ret,
        "lost_orders": lost,
        "ret_excel_omitted_n": 0.0,
        "backorder_qty_final": 0.0,
        "backlog_age_max": 0.0,
        "shift_hours": resources,
        "surge_hours": resources,
        "buffer_target_unit_hours": resources,
        "op8_convoy_vehicle_hours": resources,
        "action_trajectory": [
            {"decision": "INTERVENE", "decision_time": resources}
        ],
    }


def test_safe_oracle_rejects_shed_to_win() -> None:
    comparator = _panel(0.5)
    shed = _panel(0.8, lost=1.0)
    assert not safe_against(shed, comparator)


def test_promotion_gate_requires_all_tapes_safe() -> None:
    comparator = [_panel(0.50, resources=float(index)) for index in range(48)]
    candidate = [_panel(0.52, resources=float(index)) for index in range(48)]
    result = paired_promotion_summary(candidate, comparator, n_bootstrap=1000)
    assert result["promotion_pass"]
    candidate[-1] = _panel(0.52, lost=1.0, resources=47.0)
    result = paired_promotion_summary(candidate, comparator, n_bootstrap=1000)
    assert not result["promotion_pass"]


def test_short_schedule_runs_canonical_and_secondary_panels() -> None:
    low = Posture("f0_S1", 0.0, 1, 0.0)
    row = evaluate_schedule(
        seed=42,
        risk_overrides={"R21": "current", "R22": "current", "R23": "current", "R24": "current"},
        low=low,
        high=low,
        spec=ScheduleSpec("constant", "low", False),
        max_daily_steps=2,
    )
    assert row["ret_excel_contract_version"] == "ret_excel_request_snapshot_v2"
    assert row["temporal_panel_version"] == "risk_cluster_daily_v1"
    assert row["action_trajectory_daily_length"] == 2
    assert len(row["action_trajectory_sha256"]) == 64
