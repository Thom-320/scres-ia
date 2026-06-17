from __future__ import annotations

import csv
from pathlib import Path

import gymnasium as gym
import torch
import pytest

import scripts.run_thesis_decision_ppo_smoke as thesis_smoke
import scripts.run_track_a_exhaustion_sweep as track_a_sweep


def test_thesis_smoke_forwards_reward_shaping_knobs() -> None:
    args = thesis_smoke.build_parser().parse_args(
        [
            "--reward-mode",
            "ReT_ladder_v1",
            "--w-bo",
            "5.0",
            "--w-cost",
            "0.03",
            "--ret-ladder-w-sc",
            "0.80",
            "--ret-ladder-w-rc",
            "0.20",
            "--ret-ladder-w-ef",
            "0.00",
            "--ret-ladder-gate-beta",
            "20.0",
        ]
    )

    kwargs = thesis_smoke.env_kwargs(args)

    assert kwargs["reward_mode"] == "ReT_ladder_v1"
    assert kwargs["w_bo"] == pytest.approx(5.0)
    assert kwargs["w_cost"] == pytest.approx(0.03)
    assert kwargs["ret_ladder_w_sc"] == pytest.approx(0.80)
    assert kwargs["ret_ladder_w_rc"] == pytest.approx(0.20)
    assert kwargs["ret_ladder_w_ef"] == pytest.approx(0.00)
    assert kwargs["ret_ladder_gate_beta"] == pytest.approx(20.0)


def test_thesis_smoke_forwards_ret_tail_knobs() -> None:
    args = thesis_smoke.build_parser().parse_args(
        [
            "--reward-mode",
            "ReT_tail_v1",
            "--ret-tail-w-sc",
            "0.25",
            "--ret-tail-w-rc",
            "0.60",
            "--ret-tail-w-ce",
            "0.15",
            "--ret-tail-cap-kappa",
            "0.35",
            "--ret-tail-inv-kappa",
            "0.75",
            "--ret-tail-boost",
            "3.0",
        ]
    )

    kwargs = thesis_smoke.env_kwargs(args)

    assert kwargs["reward_mode"] == "ReT_tail_v1"
    assert kwargs["ret_tail_w_sc"] == pytest.approx(0.25)
    assert kwargs["ret_tail_w_rc"] == pytest.approx(0.60)
    assert kwargs["ret_tail_w_ce"] == pytest.approx(0.15)
    assert kwargs["ret_tail_cap_kappa"] == pytest.approx(0.35)
    assert kwargs["ret_tail_inv_kappa"] == pytest.approx(0.75)
    assert kwargs["ret_tail_boost"] == pytest.approx(3.0)


def test_track_a_sweep_command_uses_faithful_fixes_and_profile_args(
    tmp_path: Path,
) -> None:
    args = track_a_sweep.build_parser().parse_args(
        [
            "--output-root",
            str(tmp_path),
            "--train-timesteps",
            "64",
            "--eval-episodes",
            "1",
            "--max-steps",
            "2",
            "--n-steps",
            "32",
            "--batch-size",
            "32",
            "--n-epochs",
            "1",
        ]
    )

    command = track_a_sweep.build_command(
        args=args,
        run_root=tmp_path / "runs",
        label="probe",
        algo="ppo_mlp",
        action_space_mode="continuous_it_s",
        reward_profile="control_steep",
        risk_level="severe_extended",
        pt_profile="stoch_pt_mean_hi",
    )

    assert "--risk-occurrence-mode" in command
    assert command[command.index("--risk-occurrence-mode") + 1] == "thesis_periodic"
    assert "--raw-material-flow-mode" in command
    assert (
        command[command.index("--raw-material-flow-mode") + 1]
        == "kit_equivalent_order_up_to"
    )
    assert "--action-space-mode" in command
    assert command[command.index("--action-space-mode") + 1] == "continuous_it_s"
    assert "--reward-mode" in command
    assert command[command.index("--reward-mode") + 1] == "control_v1"
    assert "--w-bo" in command
    assert command[command.index("--w-bo") + 1] == "5.0"
    assert "--stochastic-pt" in command
    assert "--stochastic-pt-mean-preserving" in command
    assert "--stochastic-pt-spread" in command
    assert command[command.index("--stochastic-pt-spread") + 1] == "2.0"


def test_track_a_sweep_accepts_ret_tail_reward_profile(tmp_path: Path) -> None:
    args = track_a_sweep.build_parser().parse_args(
        [
            "--output-root",
            str(tmp_path),
        ]
    )

    command = track_a_sweep.build_command(
        args=args,
        run_root=tmp_path / "runs",
        label="probe",
        algo="ppo_mlp",
        action_space_mode="thesis_factorized",
        reward_profile="ret_tail",
        risk_level="increased",
        pt_profile="stoch_pt_hist",
    )

    assert "--reward-mode" in command
    assert command[command.index("--reward-mode") + 1] == "ReT_tail_v1"


def test_track_a_sweep_can_use_cf_risk_profile_panel(tmp_path: Path) -> None:
    args = track_a_sweep.build_parser().parse_args(
        [
            "--output-root",
            str(tmp_path),
            "--use-cf-risk-profile",
            "--panel-cfis",
            "31-32",
        ]
    )

    command = track_a_sweep.build_command(
        args=args,
        run_root=tmp_path / "runs",
        label="probe",
        algo="ppo_mlp",
        action_space_mode="thesis_factorized",
        reward_profile="ret_ladder",
        risk_level="war_stress_v1",
        pt_profile="stoch_pt_hist",
    )

    assert "--train-cfis" in command
    assert command[command.index("--train-cfis") + 1] == "31-32"
    assert "--garrido-cfis" in command
    assert command[command.index("--garrido-cfis") + 1] == "31-32"
    assert "--train-risk-profile" in command
    assert command[command.index("--train-risk-profile") + 1] == "war_stress_v1"
    assert "--eval-risk-profile" in command
    assert command[command.index("--eval-risk-profile") + 1] == "war_stress_v1"


def test_track_a_sweep_command_forwards_algo_device_and_history(tmp_path: Path) -> None:
    args = track_a_sweep.build_parser().parse_args(
        [
            "--output-root",
            str(tmp_path),
            "--algos",
            "dmlpa_ppo",
            "--device",
            "auto",
            "--history-window",
            "30",
        ]
    )

    command = track_a_sweep.build_command(
        args=args,
        run_root=tmp_path / "runs",
        label="probe",
        algo="dmlpa_ppo",
        action_space_mode="continuous_it_s",
        reward_profile="ret_ladder_steep",
        risk_level="war_stress_v1",
        pt_profile="stoch_pt_hist",
    )

    assert "--algo" in command
    assert command[command.index("--algo") + 1] == "dmlpa_ppo"
    assert "--device" in command
    assert command[command.index("--device") + 1] == "auto"
    assert "--history-window" in command
    assert command[command.index("--history-window") + 1] == "30"


def test_track_a_sweep_command_can_forward_reward_normalization(tmp_path: Path) -> None:
    args = track_a_sweep.build_parser().parse_args(
        [
            "--output-root",
            str(tmp_path),
            "--norm-reward",
        ]
    )

    command = track_a_sweep.build_command(
        args=args,
        run_root=tmp_path / "runs",
        label="probe",
        algo="recurrent_ppo",
        action_space_mode="thesis_factorized",
        reward_profile="ret_ladder_steep",
        risk_level="severe_training",
        pt_profile="stoch_pt_hist",
    )

    assert "--norm-reward" in command
    assert "--no-norm-reward" not in command


def test_track_a_sweep_forwards_parallel_envs_and_eval_seed_base(
    tmp_path: Path,
) -> None:
    args = track_a_sweep.build_parser().parse_args(
        [
            "--output-root",
            str(tmp_path),
            "--n-envs",
            "8",
            "--eval-seed-base",
            "990000",
        ]
    )

    command = track_a_sweep.build_command(
        args=args,
        run_root=tmp_path / "runs",
        label="probe",
        algo="ppo_mlp",
        action_space_mode="thesis_factorized",
        reward_profile="ret_ladder_steep",
        risk_level="increased",
        pt_profile="stoch_pt_hist",
    )

    assert "--n-envs" in command
    assert command[command.index("--n-envs") + 1] == "8"
    assert "--eval-seed-base" in command
    assert command[command.index("--eval-seed-base") + 1] == "990000"


def test_dmlpa_extractor_uses_history_window() -> None:
    observation_space = gym.spaces.Box(-10.0, 10.0, shape=(90,), dtype=float)
    extractor = thesis_smoke.DMLPAPositionalExtractor(
        observation_space,
        history_window=30,
        features_dim=120,
    )

    features = extractor(torch.zeros((2, 90), dtype=torch.float32))

    assert extractor.obs_dimension == 3
    assert tuple(features.shape) == (2, 120)


def test_dmlpa_extractor_rejects_non_divisible_observation_dim() -> None:
    observation_space = gym.spaces.Box(-10.0, 10.0, shape=(91,), dtype=float)

    with pytest.raises(ValueError, match="divisible"):
        thesis_smoke.DMLPAPositionalExtractor(
            observation_space,
            history_window=30,
            features_dim=120,
        )


def test_track_a_sweep_policy_summary_selects_best_static(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rows = [
        {
            "policy": "ppo_mlp",
            "fill_rate_order_level_mean": "0.60",
            "order_level_ret_mean": "0.20",
            "reward_total_mean": "12.0",
        },
        {
            "policy": "static_grid_I168_S2",
            "fill_rate_order_level_mean": "0.55",
            "order_level_ret_mean": "0.25",
            "reward_total_mean": "11.0",
        },
        {
            "policy": "static_grid_I504_S2",
            "fill_rate_order_level_mean": "0.65",
            "order_level_ret_mean": "0.30",
            "reward_total_mean": "10.0",
        },
    ]
    with (run_dir / "policy_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = track_a_sweep.read_policy_summary(run_dir)

    assert summary["best_static_policy"] == "static_grid_I504_S2"
    assert summary["ppo_fill"] == pytest.approx(0.60)
    assert summary["best_static_fill"] == pytest.approx(0.65)
    assert summary["delta_fill"] == pytest.approx(-0.05)
    assert summary["delta_ret"] == pytest.approx(-0.10)


def test_track_a_sweep_policy_summary_accepts_non_mlp_algo(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rows = [
        {
            "policy": "dmlpa_ppo",
            "fill_rate_order_level_mean": "0.71",
            "order_level_ret_mean": "0.62",
            "ret_mean_all_orders_zero_unfulfilled_mean": "0.60",
            "flow_fill_rate_mean": "0.59",
            "stockout_week_pct_mean": "30.0",
            "reward_total_mean": "12.0",
        },
        {
            "policy": "static_grid_I504_S2",
            "fill_rate_order_level_mean": "0.70",
            "order_level_ret_mean": "0.61",
            "ret_mean_all_orders_zero_unfulfilled_mean": "0.58",
            "flow_fill_rate_mean": "0.57",
            "stockout_week_pct_mean": "35.0",
            "reward_total_mean": "10.0",
        },
    ]
    with (run_dir / "policy_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = track_a_sweep.read_policy_summary(run_dir, algo="dmlpa_ppo")

    assert summary["ppo_fill"] == pytest.approx(0.71)
    assert summary["best_static_policy"] == "static_grid_I504_S2"
    assert summary["delta_fill"] == pytest.approx(0.01)
    assert summary["delta_ret_all_orders"] == pytest.approx(0.02)
