from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from research.paper2_exhaustive_search.war_stress_risk_tapes import build_risk_tape
from research.paper2_exhaustive_search.restricted_timing_oracle import (
    ScheduleSpec,
    evaluate_schedule,
    posture_from_label,
)
from scripts.build_war_stress_exact_reduction_certificate import (
    build_certificate as build_reduction_certificate,
)
from scripts.build_war_stress_policy_manifest import build_manifest
from supply_chain.event_triggered_env import make_event_triggered_track_a_env


ROOT = Path(__file__).resolve().parents[1]
GSA_MANIFEST = (
    ROOT
    / "research/paper2_exhaustive_search/war_stress_gsa_overlay_manifest_20260716.json"
)


def _configuration(config_id: str) -> dict:
    payload = json.loads(GSA_MANIFEST.read_text())
    rows = [*payload["morris"]["rows"], *payload["qmc_pool"]["rows"]]
    return next(row for row in rows if row["config_id"] == config_id)


def test_policy_manifest_exact_counts_and_hash_are_reproducible() -> None:
    left = build_manifest()
    right = build_manifest()
    assert left == right
    assert left["counts_by_family"] == {
        "constant": 18,
        "open_loop_8week_periodic": 39_168,
        "restricted_privileged": 5_508,
        "weekly_privileged": 5_508,
    }
    assert left["total_policy_templates"] == 50_202


def test_exact_reduction_removes_only_all_low_calendar_duplicates() -> None:
    certificate = build_reduction_certificate()
    assert certificate["full_policy_count"] == 50_202
    assert certificate["reduced_policy_count"] == 50_049
    assert certificate["removed_duplicate_count"] == 153
    assert certificate["vector_dominance_used"] is False
    assert certificate["branch_and_bound_used"] is False
    assert all(
        row["removed_policy_id"].endswith("::000")
        and row["retained_policy_id"].startswith("constant::")
        for row in certificate["duplicate_mappings"]
    )


def test_all_low_calendar_is_bit_identical_to_constant_low_in_des() -> None:
    low = posture_from_label("f0_S1")
    high = posture_from_label("f0.125_S2")
    common = {
        "seed": 94_700_002,
        "risk_overrides": {},
        "max_daily_steps": 14,
        "enabled_risks": (),
    }
    constant = evaluate_schedule(
        **common,
        low=low,
        high=low,
        spec=ScheduleSpec("constant", "constant-low", False),
    )
    calendar = evaluate_schedule(
        **common,
        low=low,
        high=high,
        spec=ScheduleSpec("open_loop_8week_periodic", "all-low", (0,) * 8),
    )
    assert constant == calendar


def test_policy_independent_tapes_share_base_stream_but_change_with_configuration() -> None:
    first = _configuration("morris::LOC_SURGE::independent::00::00")
    second = _configuration("morris::LOC_SURGE::independent::00::01")
    tape_a = build_risk_tape(first, tape_id=94_700_001, horizon_hours=8_736)
    tape_a_repeat = build_risk_tape(first, tape_id=94_700_001, horizon_hours=8_736)
    tape_b = build_risk_tape(second, tape_id=94_700_001, horizon_hours=8_736)
    assert tape_a == tape_a_repeat
    assert tape_a.base_stream_sha256 == tape_b.base_stream_sha256
    assert tape_a.event_tape_sha256 != tape_b.event_tape_sha256
    assert tape_a.r3_event_count == tape_b.r3_event_count == 0


def test_coupled_tape_uses_exact_frozen_offset() -> None:
    config = _configuration("morris::LOC_SURGE::disruption_leads_surge_72h::00::00")
    tape = build_risk_tape(config, tape_id=94_700_001, horizon_hours=8_736)
    r24 = [row["start_time"] for row in tape.events if row["risk_id"] == "R24"]
    r22 = [row["start_time"] for row in tape.events if row["risk_id"] == "R22"]
    expected = [start - 72.0 for start in r24 if start - 72.0 >= 0.0]
    assert r22 == expected


def test_scientific_executor_fails_closed_before_creating_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "forbidden"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_war_stress_gsa_executor.py",
            "--mode",
            "scientific",
            "--run-dir",
            str(run_dir),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "not authorized" in completed.stderr
    assert not run_dir.exists()


def test_r24_replay_records_generated_admitted_and_clipped_quantity() -> None:
    env = make_event_triggered_track_a_env(
        init_frac=0.0,
        init_shifts=1,
        max_steps=1,
        enabled_risks=("R24",),
        stochastic_pt=False,
        priming_enabled=False,
        risk_event_tape=[
            {
                "risk_id": "R24",
                "start_time": 950.0,
                "end_time": 950.0,
                "duration": 0.0,
                "affected_ops": [13],
                "magnitude": 20_000.0,
                "unit": "rations",
            }
        ],
    )
    try:
        env.reset(seed=94_700_001)
        env.step([-1.0, 0.0, 0.0])
        sim = env.unwrapped.sim
        assert sim.r24_generated_surge_quantity == 20_000.0
        assert sim.r24_admitted_surge_quantity == 13_000.0
        assert sim.r24_clipped_surge_quantity == 7_000.0
        assert sim.r24_cap_hit_count == 1
    finally:
        env.close()
