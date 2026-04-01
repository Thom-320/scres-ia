from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.run_track_b_benchmark as track_b_benchmark


def _write_csv(path: Path, header: str) -> None:
    path.write_text(f"{header}\n", encoding="utf-8")


def test_build_benchmark_cli_args_freezes_track_b_backbone(tmp_path: Path) -> None:
    args = track_b_benchmark.build_parser().parse_args(
        [
            "--label",
            "track_b_long",
            "--output-root",
            str(tmp_path),
        ]
    )

    cli_args = track_b_benchmark.build_benchmark_cli_args(
        args, track_b_benchmark.resolve_run_dir(args)
    )
    command = track_b_benchmark.build_benchmark_command(cli_args)

    assert "--risk-level adaptive_benchmark_v2" in command
    assert "--step-size-hours 168.0" in command
    assert "--max-steps 260" in command
    assert "--reward-mode ReT_seq_v1" in command
    assert "--ret-seq-kappa 0.2" in command


def test_run_launcher_writes_auditable_trail_on_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _fake_invoke(benchmark_args) -> dict[str, object]:
        run_dir = Path(benchmark_args.output_dir)
        _write_csv(run_dir / "policy_summary.csv", "policy,fill_rate_mean")
        _write_csv(run_dir / "comparison_table.csv", "policy,fill_rate_mean")
        summary = {
            "config": {
                "reward_mode": benchmark_args.reward_mode,
                "observation_version": "v7",
                "action_contract": "track_b_v1",
                "risk_level": "adaptive_benchmark_v2",
            },
            "backbone": {
                "env_variant": "track_b_adaptive_control",
                "observation_version": "v7",
                "action_contract": "track_b_v1",
                "risk_level": "adaptive_benchmark_v2",
                "year_basis": "thesis",
            },
            "reward_contract": {
                "reward_mode": benchmark_args.reward_mode,
                "reward_family": "resilience_index",
                "cross_mode_reward_comparison_allowed": False,
            },
            "metric_contract": {
                "fill_rate_primary": "terminal_order_level",
            },
            "artifacts": {
                "summary_json": str((run_dir / "summary.json").resolve()),
            },
        }
        (run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        return summary

    monkeypatch.setattr(track_b_benchmark, "invoke_benchmark", _fake_invoke)

    args = track_b_benchmark.build_parser().parse_args(
        [
            "--label",
            "track_b_seq_500k",
            "--output-root",
            str(tmp_path),
            "--heartbeat-interval-seconds",
            "0.01",
        ]
    )
    exit_code = track_b_benchmark.run_launcher(args)
    run_dir = tmp_path / "track_b_seq_500k"

    assert exit_code == 0
    assert (run_dir / "command.txt").exists()
    assert (run_dir / "pid.txt").exists()
    assert (run_dir / "status.json").exists()
    assert (run_dir / "stdout.log").exists()
    assert (run_dir / "heartbeat.json").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "manifest.json").exists()
    assert not (run_dir / "FAILED.json").exists()

    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    heartbeat = json.loads((run_dir / "heartbeat.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))

    assert status["state"] == "completed"
    assert heartbeat["state"] == "completed"
    assert summary["launcher_metadata"]["label"] == "track_b_seq_500k"
    assert manifest["status"] == "completed"
    assert manifest["backbone"]["observation_version"] == "v7"
    assert manifest["frozen_backbone"]["action_contract"] == "track_b_v1"


def test_run_launcher_marks_invalid_when_outputs_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _fake_invoke(benchmark_args) -> dict[str, object]:
        run_dir = Path(benchmark_args.output_dir)
        summary = {
            "config": {"reward_mode": "ReT_seq_v1"},
            "backbone": {"observation_version": "v7"},
            "artifacts": {
                "summary_json": str((run_dir / "summary.json").resolve()),
            },
        }
        (run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        return summary

    monkeypatch.setattr(track_b_benchmark, "invoke_benchmark", _fake_invoke)

    args = track_b_benchmark.build_parser().parse_args(
        [
            "--label",
            "track_b_invalid",
            "--output-root",
            str(tmp_path),
            "--heartbeat-interval-seconds",
            "0.01",
        ]
    )
    exit_code = track_b_benchmark.run_launcher(args)
    run_dir = tmp_path / "track_b_invalid"

    assert exit_code == 1
    failed = json.loads((run_dir / "FAILED.json").read_text(encoding="utf-8"))
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))

    assert status["state"] == "invalid"
    assert "policy_summary.csv" in failed["missing_outputs"]
    assert "comparison_table.csv" in failed["missing_outputs"]
