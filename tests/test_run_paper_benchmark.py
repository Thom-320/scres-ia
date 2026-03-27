from __future__ import annotations

import json
from pathlib import Path

import scripts.run_paper_benchmark as paper_benchmark


def _write_csv(path: Path, header: str) -> None:
    path.write_text(f"{header}\n", encoding="utf-8")


def test_build_benchmark_cli_args_freezes_paper_backbone(tmp_path: Path) -> None:
    args = paper_benchmark.build_parser().parse_args(
        [
            "--label",
            "ret_kappa_010",
            "--reward-mode",
            "ReT_seq_v1",
            "--kappa",
            "0.10",
            "--output-root",
            str(tmp_path),
        ]
    )

    cli_args = paper_benchmark.build_benchmark_cli_args(
        args, paper_benchmark.resolve_run_dir(args)
    )
    command = paper_benchmark.build_benchmark_command(cli_args)

    assert "--observation-version v1" in command
    assert "--frame-stack 1" in command
    assert "--year-basis thesis" in command
    assert "--risk-level increased" in command
    assert "--stochastic-pt" in command
    assert "--reward-mode ReT_seq_v1" in command
    assert "--ret-seq-kappa 0.1" in command


def test_run_paper_benchmark_defaults_to_ret_seq_v1() -> None:
    args = paper_benchmark.build_parser().parse_args(["--label", "default_run"])
    assert args.reward_mode == "ReT_seq_v1"
    assert args.kappa == 0.20


def test_run_launcher_writes_auditable_trail_on_success(
    tmp_path: Path, monkeypatch
) -> None:
    def _fake_invoke(benchmark_args, *, benchmark_command: str) -> dict[str, object]:
        run_dir = Path(benchmark_args.output_dir)
        _write_csv(run_dir / "policy_summary.csv", "policy,fill_rate_mean")
        _write_csv(run_dir / "comparison_table.csv", "policy,fill_rate_mean")
        summary = {
            "config": {
                "algo": "ppo",
                "frame_stack": 1,
                "observation_version": "v1",
                "reward_mode": benchmark_args.reward_mode,
                "risk_level": "increased",
                "stochastic_pt": True,
                "year_basis": "thesis",
            },
            "backbone": {
                "env_variant": "shift_control",
                "observation_version": "v1",
                "frame_stack": 1,
                "year_basis": "thesis",
                "risk_level": "increased",
                "stochastic_pt": True,
                "step_size_hours": 168.0,
                "max_steps": 260,
                "reward_mode": benchmark_args.reward_mode,
            },
            "reward_contract": {
                "reward_mode": benchmark_args.reward_mode,
                "reward_family": "operational_penalty",
                "cross_mode_reward_comparison_allowed": False,
            },
            "artifacts": {
                "summary_json": str((run_dir / "summary.json").resolve()),
            },
        }
        (run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        return summary

    monkeypatch.setattr(paper_benchmark, "invoke_benchmark", _fake_invoke)

    args = paper_benchmark.build_parser().parse_args(
        [
            "--label",
            "control_v1_500k",
            "--output-root",
            str(tmp_path),
            "--heartbeat-interval-seconds",
            "0.01",
        ]
    )
    exit_code = paper_benchmark.run_launcher(args)
    run_dir = tmp_path / "control_v1_500k"

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
    stdout_log = (run_dir / "stdout.log").read_text(encoding="utf-8")

    assert status["state"] == "completed"
    assert heartbeat["state"] == "completed"
    assert summary["metric_contract"]["fill_rate_primary"] == "terminal_order_level"
    assert summary["launcher_metadata"]["label"] == "control_v1_500k"
    assert manifest["status"] == "completed"
    assert manifest["backbone"]["observation_version"] == "v1"
    assert (
        manifest["frozen_backbone"]["benchmark_protocol"]
        == "reward_benchmark_corrected"
    )
    assert manifest["reward_contract"]["cross_mode_reward_comparison_allowed"] is False
    assert "starting paper benchmark" in stdout_log


def test_run_launcher_marks_run_invalid_when_required_outputs_are_missing(
    tmp_path: Path, monkeypatch
) -> None:
    def _fake_invoke_missing(
        benchmark_args, *, benchmark_command: str
    ) -> dict[str, object]:
        run_dir = Path(benchmark_args.output_dir)
        summary = {
            "config": {"algo": "ppo"},
            "backbone": {"observation_version": "v1"},
            "artifacts": {
                "summary_json": str((run_dir / "summary.json").resolve()),
            },
        }
        (run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        return summary

    monkeypatch.setattr(paper_benchmark, "invoke_benchmark", _fake_invoke_missing)

    args = paper_benchmark.build_parser().parse_args(
        [
            "--label",
            "invalid_run",
            "--output-root",
            str(tmp_path),
            "--heartbeat-interval-seconds",
            "0.01",
        ]
    )
    exit_code = paper_benchmark.run_launcher(args)
    run_dir = tmp_path / "invalid_run"

    assert exit_code == 1
    assert (run_dir / "FAILED.json").exists()

    failed = json.loads((run_dir / "FAILED.json").read_text(encoding="utf-8"))
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))

    assert failed["state"] == "invalid"
    assert "policy_summary.csv" in failed["missing_outputs"]
    assert "comparison_table.csv" in failed["missing_outputs"]
    assert status["state"] == "invalid"
    assert manifest["status"] == "invalid"
