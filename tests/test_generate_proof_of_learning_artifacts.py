from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.generate_proof_of_learning_artifacts import (
    build_cross_scenario_rows,
    choose_timeline_risk_level,
    generate_proof_of_learning_artifacts,
)


def _write_csv(
    path: Path, fieldnames: list[str], rows: list[dict[str, object]]
) -> None:
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_minimal_run_dir(run_dir: Path) -> None:
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "config": {
                    "algo": "ppo",
                    "risk_level": "increased",
                    "eval_risk_levels": ["current", "increased", "severe"],
                },
                "weight_combinations": [{"w_bo": 4.0, "w_cost": 0.02, "w_disr": 0.0}],
                "survivors": [{"w_bo": 4.0, "w_cost": 0.02, "w_disr": 0.0}],
            }
        ),
        encoding="utf-8",
    )
    policy_rows = [
        {
            "phase": "static_screen",
            "policy": "static_s1",
            "w_bo": "4.0",
            "w_cost": "0.02",
            "w_disr": "0.0",
            "fill_rate_mean": "0.79",
            "backorder_rate_mean": "0.21",
            "order_level_ret_mean_mean": "0.75",
            "pct_steps_S1_mean": "100.0",
            "pct_steps_S2_mean": "0.0",
            "pct_steps_S3_mean": "0.0",
        },
        {
            "phase": "static_screen",
            "policy": "static_s2",
            "w_bo": "4.0",
            "w_cost": "0.02",
            "w_disr": "0.0",
            "fill_rate_mean": "0.84",
            "backorder_rate_mean": "0.16",
            "order_level_ret_mean_mean": "0.80",
            "pct_steps_S1_mean": "0.0",
            "pct_steps_S2_mean": "100.0",
            "pct_steps_S3_mean": "0.0",
        },
        {
            "phase": "static_screen",
            "policy": "static_s3",
            "w_bo": "4.0",
            "w_cost": "0.02",
            "w_disr": "0.0",
            "fill_rate_mean": "0.82",
            "backorder_rate_mean": "0.18",
            "order_level_ret_mean_mean": "0.79",
            "pct_steps_S1_mean": "0.0",
            "pct_steps_S2_mean": "0.0",
            "pct_steps_S3_mean": "100.0",
        },
        {
            "phase": "ppo_eval",
            "policy": "ppo",
            "w_bo": "4.0",
            "w_cost": "0.02",
            "w_disr": "0.0",
            "fill_rate_mean": "0.845",
            "backorder_rate_mean": "0.155",
            "order_level_ret_mean_mean": "0.81",
            "pct_steps_S1_mean": "5.0",
            "pct_steps_S2_mean": "65.0",
            "pct_steps_S3_mean": "30.0",
        },
        {
            "phase": "cross_eval_current",
            "policy": "static_s2",
            "w_bo": "4.0",
            "w_cost": "0.02",
            "w_disr": "0.0",
            "fill_rate_mean": "0.88",
            "backorder_rate_mean": "0.12",
            "order_level_ret_mean_mean": "0.84",
            "pct_steps_S1_mean": "0.0",
            "pct_steps_S2_mean": "100.0",
            "pct_steps_S3_mean": "0.0",
        },
        {
            "phase": "cross_eval_current",
            "policy": "ppo",
            "w_bo": "4.0",
            "w_cost": "0.02",
            "w_disr": "0.0",
            "fill_rate_mean": "0.879",
            "backorder_rate_mean": "0.121",
            "order_level_ret_mean_mean": "0.845",
            "pct_steps_S1_mean": "10.0",
            "pct_steps_S2_mean": "60.0",
            "pct_steps_S3_mean": "30.0",
        },
        {
            "phase": "cross_eval_severe",
            "policy": "static_s2",
            "w_bo": "4.0",
            "w_cost": "0.02",
            "w_disr": "0.0",
            "fill_rate_mean": "0.76",
            "backorder_rate_mean": "0.24",
            "order_level_ret_mean_mean": "0.73",
            "pct_steps_S1_mean": "0.0",
            "pct_steps_S2_mean": "100.0",
            "pct_steps_S3_mean": "0.0",
        },
        {
            "phase": "cross_eval_severe",
            "policy": "static_s3",
            "w_bo": "4.0",
            "w_cost": "0.02",
            "w_disr": "0.0",
            "fill_rate_mean": "0.77",
            "backorder_rate_mean": "0.23",
            "order_level_ret_mean_mean": "0.74",
            "pct_steps_S1_mean": "0.0",
            "pct_steps_S2_mean": "0.0",
            "pct_steps_S3_mean": "100.0",
        },
        {
            "phase": "cross_eval_severe",
            "policy": "ppo",
            "w_bo": "4.0",
            "w_cost": "0.02",
            "w_disr": "0.0",
            "fill_rate_mean": "0.79",
            "backorder_rate_mean": "0.21",
            "order_level_ret_mean_mean": "0.76",
            "pct_steps_S1_mean": "12.0",
            "pct_steps_S2_mean": "58.0",
            "pct_steps_S3_mean": "30.0",
        },
    ]
    _write_csv(run_dir / "policy_summary.csv", list(policy_rows[0].keys()), policy_rows)

    training_rows = [
        {
            "seed": "11",
            "algo": "ppo",
            "reward_mode": "ReT_seq_v1",
            "reward_family": "resilience_index",
            "w_bo": "4.0",
            "w_cost": "0.02",
            "w_disr": "0.0",
            "episode_index": "1",
            "timesteps": "32",
            "progress_fraction": "0.2",
            "episode_reward": "0.45",
            "episode_length": "4",
            "time_elapsed_seconds": "0.01",
        },
        {
            "seed": "11",
            "algo": "ppo",
            "reward_mode": "ReT_seq_v1",
            "reward_family": "resilience_index",
            "w_bo": "4.0",
            "w_cost": "0.02",
            "w_disr": "0.0",
            "episode_index": "2",
            "timesteps": "64",
            "progress_fraction": "0.4",
            "episode_reward": "0.55",
            "episode_length": "4",
            "time_elapsed_seconds": "0.02",
        },
        {
            "seed": "22",
            "algo": "ppo",
            "reward_mode": "ReT_seq_v1",
            "reward_family": "resilience_index",
            "w_bo": "4.0",
            "w_cost": "0.02",
            "w_disr": "0.0",
            "episode_index": "1",
            "timesteps": "32",
            "progress_fraction": "0.2",
            "episode_reward": "0.41",
            "episode_length": "4",
            "time_elapsed_seconds": "0.01",
        },
        {
            "seed": "22",
            "algo": "ppo",
            "reward_mode": "ReT_seq_v1",
            "reward_family": "resilience_index",
            "w_bo": "4.0",
            "w_cost": "0.02",
            "w_disr": "0.0",
            "episode_index": "2",
            "timesteps": "64",
            "progress_fraction": "0.4",
            "episode_reward": "0.58",
            "episode_length": "4",
            "time_elapsed_seconds": "0.02",
        },
    ]
    _write_csv(
        run_dir / "training_trace.csv", list(training_rows[0].keys()), training_rows
    )

    proof_rows = [
        {
            "phase": "proof_increased",
            "risk_level": "increased",
            "policy": "ppo",
            "algo": "ppo",
            "reward_mode": "ReT_seq_v1",
            "reward_family": "resilience_index",
            "frame_stack": "1",
            "observation_version": "v1",
            "seed": "11",
            "episode": "1",
            "eval_seed": "80011",
            "w_bo": "4.0",
            "w_cost": "0.02",
            "w_disr": "0.0",
            "step": "0",
            "shifts_active": "2",
            "fill_rate": "0.92",
            "backorder_rate": "0.08",
            "disruption_fraction": "0.10",
            "reward": "0.12",
            "service_loss": "0.08",
        },
        {
            "phase": "proof_increased",
            "risk_level": "increased",
            "policy": "ppo",
            "algo": "ppo",
            "reward_mode": "ReT_seq_v1",
            "reward_family": "resilience_index",
            "frame_stack": "1",
            "observation_version": "v1",
            "seed": "11",
            "episode": "1",
            "eval_seed": "80011",
            "w_bo": "4.0",
            "w_cost": "0.02",
            "w_disr": "0.0",
            "step": "1",
            "shifts_active": "3",
            "fill_rate": "0.90",
            "backorder_rate": "0.10",
            "disruption_fraction": "0.35",
            "reward": "0.09",
            "service_loss": "0.10",
        },
        {
            "phase": "proof_increased",
            "risk_level": "increased",
            "policy": "static_s2",
            "algo": "ppo",
            "reward_mode": "ReT_seq_v1",
            "reward_family": "resilience_index",
            "frame_stack": "1",
            "observation_version": "v1",
            "seed": "11",
            "episode": "1",
            "eval_seed": "80011",
            "w_bo": "4.0",
            "w_cost": "0.02",
            "w_disr": "0.0",
            "step": "0",
            "shifts_active": "2",
            "fill_rate": "0.92",
            "backorder_rate": "0.08",
            "disruption_fraction": "0.10",
            "reward": "0.10",
            "service_loss": "0.08",
        },
        {
            "phase": "proof_severe",
            "risk_level": "severe",
            "policy": "ppo",
            "algo": "ppo",
            "reward_mode": "ReT_seq_v1",
            "reward_family": "resilience_index",
            "frame_stack": "1",
            "observation_version": "v1",
            "seed": "11",
            "episode": "1",
            "eval_seed": "80011",
            "w_bo": "4.0",
            "w_cost": "0.02",
            "w_disr": "0.0",
            "step": "0",
            "shifts_active": "2",
            "fill_rate": "0.85",
            "backorder_rate": "0.15",
            "disruption_fraction": "0.25",
            "reward": "0.07",
            "service_loss": "0.15",
        },
        {
            "phase": "proof_severe",
            "risk_level": "severe",
            "policy": "ppo",
            "algo": "ppo",
            "reward_mode": "ReT_seq_v1",
            "reward_family": "resilience_index",
            "frame_stack": "1",
            "observation_version": "v1",
            "seed": "11",
            "episode": "1",
            "eval_seed": "80011",
            "w_bo": "4.0",
            "w_cost": "0.02",
            "w_disr": "0.0",
            "step": "1",
            "shifts_active": "3",
            "fill_rate": "0.80",
            "backorder_rate": "0.20",
            "disruption_fraction": "0.60",
            "reward": "0.03",
            "service_loss": "0.20",
        },
        {
            "phase": "proof_severe",
            "risk_level": "severe",
            "policy": "static_s2",
            "algo": "ppo",
            "reward_mode": "ReT_seq_v1",
            "reward_family": "resilience_index",
            "frame_stack": "1",
            "observation_version": "v1",
            "seed": "11",
            "episode": "1",
            "eval_seed": "80011",
            "w_bo": "4.0",
            "w_cost": "0.02",
            "w_disr": "0.0",
            "step": "0",
            "shifts_active": "2",
            "fill_rate": "0.84",
            "backorder_rate": "0.16",
            "disruption_fraction": "0.25",
            "reward": "0.05",
            "service_loss": "0.16",
        },
    ]
    _write_csv(
        run_dir / "proof_trajectories.csv", list(proof_rows[0].keys()), proof_rows
    )


def test_build_cross_scenario_rows_uses_training_and_cross_eval_phases(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "paper_ret_seq_k020_500k"
    _build_minimal_run_dir(run_dir)

    rows = build_cross_scenario_rows(run_dir)

    assert [row["risk_level"] for row in rows] == ["current", "increased", "severe"]
    increased_row = next(row for row in rows if row["risk_level"] == "increased")
    severe_row = next(row for row in rows if row["risk_level"] == "severe")

    assert increased_row["learned_phase"] == "ppo_eval"
    assert increased_row["static_phase"] == "static_screen"
    assert increased_row["delta_order_level_ret_mean_vs_static_s2"] == pytest.approx(
        0.01
    )
    assert severe_row["best_static_policy"] == "static_s3"


def test_choose_timeline_risk_level_prefers_severe() -> None:
    proof_rows = [
        {"risk_level": "increased", "policy": "ppo", "disruption_fraction": "0.2"},
        {"risk_level": "severe", "policy": "ppo", "disruption_fraction": "0.1"},
    ]

    assert choose_timeline_risk_level(proof_rows) == "severe"


def test_generate_proof_of_learning_artifacts_writes_expected_bundle(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "paper_ret_seq_k020_500k"
    _build_minimal_run_dir(run_dir)

    manifest = generate_proof_of_learning_artifacts(run_dir)
    output_dir = Path(manifest["output_dir"])

    assert output_dir.exists()
    assert (output_dir / "learning_curve.png").exists()
    assert (output_dir / "shift_vs_disruption_timeline.png").exists()
    assert (output_dir / "cross_scenario_comparison.csv").exists()
    assert (output_dir / "cross_scenario_comparison.md").exists()
    assert (output_dir / "manifest.json").exists()
    assert manifest["selected_timeline_risk_level"] == "severe"
