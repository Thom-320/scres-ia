from __future__ import annotations

import json
from pathlib import Path

import pytest

from supply_chain.external_env_interface import make_shift_control_env
from scripts.benchmark_control_reward import (
    build_comparison_rows,
    build_parser,
    pick_survivors,
    run_benchmark,
    static_policy_action,
)


def test_control_v1_step_exposes_reward_components_and_corrected_ret() -> None:
    env = make_shift_control_env(
        reward_mode="control_v1",
        step_size_hours=24,
        max_steps=2,
        w_bo=2.0,
        w_cost=0.06,
        w_disr=0.1,
    )
    env.reset(seed=7)
    _, reward, _, _, info = env.step(static_policy_action("static_s2"))

    components = info["control_components"]
    expected_reward = -(
        components["weighted_service_loss"]
        + components["weighted_shift_cost"]
        + components["weighted_disruption"]
    )
    assert reward == pytest.approx(expected_reward)
    assert info["reward_mode"] == "control_v1"
    assert "ret_thesis_corrected" in info
    assert info["ret_thesis_corrected"]["correction_mode"] == "autotomy_equals_recovery"
    assert info["shift_cost_step"] == pytest.approx(1.0)
    assert info["service_loss_step"] == pytest.approx(components["service_loss_step"])


def test_static_policy_actions_cover_all_shift_modes() -> None:
    env = make_shift_control_env(
        reward_mode="control_v1",
        step_size_hours=24,
        max_steps=1,
        risk_level="increased",
    )
    env.reset(seed=7)
    _, _, _, _, s1_info = env.step(static_policy_action("static_s1"))
    assert s1_info["shifts_active"] == 1

    env.reset(seed=7)
    _, _, _, _, s2_info = env.step(static_policy_action("static_s2"))
    assert s2_info["shifts_active"] == 2

    env.reset(seed=7)
    _, _, _, _, s3_info = env.step(static_policy_action("static_s3"))
    assert s3_info["shifts_active"] == 3


def test_pick_survivors_prefers_non_s1_fixed_baselines() -> None:
    args = build_parser().parse_args([])
    policy_rows = [
        {
            "phase": "static_screen",
            "policy": "static_s1",
            "w_bo": 1.0,
            "w_cost": 0.02,
            "w_disr": 0.0,
            "reward_total_mean": 10.0,
            "fill_rate_mean": 0.60,
        },
        {
            "phase": "static_screen",
            "policy": "static_s2",
            "w_bo": 1.0,
            "w_cost": 0.02,
            "w_disr": 0.0,
            "reward_total_mean": 12.5,
            "fill_rate_mean": 0.75,
        },
        {
            "phase": "static_screen",
            "policy": "static_s3",
            "w_bo": 1.0,
            "w_cost": 0.02,
            "w_disr": 0.0,
            "reward_total_mean": 11.0,
            "fill_rate_mean": 0.72,
        },
        {
            "phase": "static_screen",
            "policy": "static_s1",
            "w_bo": 1.0,
            "w_cost": 0.10,
            "w_disr": 0.0,
            "reward_total_mean": 10.0,
            "fill_rate_mean": 0.60,
        },
        {
            "phase": "static_screen",
            "policy": "static_s2",
            "w_bo": 1.0,
            "w_cost": 0.10,
            "w_disr": 0.0,
            "reward_total_mean": 9.0,
            "fill_rate_mean": 0.75,
        },
        {
            "phase": "static_screen",
            "policy": "static_s3",
            "w_bo": 1.0,
            "w_cost": 0.10,
            "w_disr": 0.0,
            "reward_total_mean": 8.0,
            "fill_rate_mean": 0.78,
        },
    ]

    survivors = pick_survivors(policy_rows, args)
    assert len(survivors) == 1
    assert survivors[0]["best_static_policy"] == "static_s2"
    assert survivors[0]["static_reward_gap_best_minus_s1"] == pytest.approx(2.5)


def test_build_comparison_rows_marks_collapse_and_reward_wins() -> None:
    survivors = [
        {
            "w_bo": 2.0,
            "w_cost": 0.02,
            "w_disr": 0.0,
            "best_static_policy": "static_s2",
            "static_reward_gap_best_minus_s1": 5.0,
        }
    ]
    policy_rows = [
        {
            "phase": "static_screen",
            "policy": "static_s2",
            "w_bo": 2.0,
            "w_cost": 0.02,
            "w_disr": 0.0,
            "reward_total_mean": 15.0,
            "fill_rate_mean": 0.82,
            "backorder_rate_mean": 0.18,
            "ret_thesis_corrected_total_mean": 240.0,
        },
        {
            "phase": "ppo_eval",
            "policy": "ppo",
            "w_bo": 2.0,
            "w_cost": 0.02,
            "w_disr": 0.0,
            "reward_total_mean": 16.0,
            "fill_rate_mean": 0.82,
            "backorder_rate_mean": 0.18,
            "ret_thesis_corrected_total_mean": 241.0,
            "pct_steps_S1_mean": 95.0,
            "pct_steps_S2_mean": 5.0,
            "pct_steps_S3_mean": 0.0,
        },
    ]

    comparison_rows = build_comparison_rows(policy_rows, survivors)
    assert len(comparison_rows) == 1
    assert comparison_rows[0]["ppo_beats_static_s2"] is True
    assert comparison_rows[0]["ppo_beats_best_static"] is True
    assert comparison_rows[0]["collapsed_to_S1"] is True
    assert comparison_rows[0]["collapsed_to_S2"] is False


def test_run_benchmark_smoke_writes_expected_artifacts(tmp_path: Path) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--seeds",
            "1",
            "--train-timesteps",
            "32",
            "--eval-episodes",
            "1",
            "--step-size-hours",
            "24",
            "--max-steps",
            "4",
            "--w-bo",
            "1.0",
            "--w-cost",
            "0.02",
            "--w-disr",
            "0.0",
            "--output-dir",
            str(tmp_path),
        ]
    )
    summary = run_benchmark(args)

    episode_csv = tmp_path / "episode_metrics.csv"
    policy_csv = tmp_path / "policy_summary.csv"
    comparison_csv = tmp_path / "comparison_table.csv"
    summary_json = tmp_path / "summary.json"

    assert episode_csv.exists()
    assert policy_csv.exists()
    assert comparison_csv.exists()
    assert summary_json.exists()
    assert "static_s3" in summary["policies"]

    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    assert payload["config"]["train_timesteps"] == 32
    assert payload["config"]["w_disr"] == [0.0]
