from __future__ import annotations

import csv
from pathlib import Path

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
