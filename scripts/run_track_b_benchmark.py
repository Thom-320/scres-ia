#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import threading
import traceback
from typing import Any, TextIO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_track_b_smoke import (  # noqa: E402
    build_parser as build_track_b_parser,
    run_smoke,
)

DEFAULT_SEEDS = (11, 22, 33, 44, 55)
DEFAULT_TIMESTEPS = 500_000
DEFAULT_EVAL_EPISODES = 20
DEFAULT_STEP_SIZE_HOURS = 168.0
DEFAULT_MAX_STEPS = 260
DEFAULT_OUTPUT_ROOT = Path("outputs/track_b_benchmarks")
REQUIRED_OUTPUTS = ("summary.json", "policy_summary.csv", "comparison_table.csv")
FROZEN_BACKBONE = {
    "code_ref": "HEAD",
    "benchmark_protocol": "track_b_minimal_v1",
    "env_variant": "track_b_adaptive_control",
    "observation_version": "v7",
    "action_contract": "track_b_v1",
    "year_basis": "thesis",
    "risk_level": "adaptive_benchmark_v2",
    "stochastic_pt": True,
    "step_size_hours": DEFAULT_STEP_SIZE_HOURS,
    "max_steps": DEFAULT_MAX_STEPS,
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class TeeStream:
    def __init__(self, *streams: TextIO) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


@dataclass(frozen=True)
class RunArtifacts:
    run_dir: Path
    command_txt: Path
    pid_txt: Path
    status_json: Path
    stdout_log: Path
    heartbeat_json: Path
    manifest_json: Path
    failed_json: Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen Track B benchmark backbone with an auditable "
            "launcher and static-vs-PPO comparison bundle."
        )
    )
    parser.add_argument("--label", required=True, help="Run directory label.")
    parser.add_argument(
        "--reward-mode",
        choices=["ReT_seq_v1", "ReT_unified_v1", "control_v1"],
        default="ReT_seq_v1",
        help="Track B training reward.",
    )
    parser.add_argument(
        "--ret-seq-kappa",
        type=float,
        default=0.20,
        help="ReT_seq_v1 kappa. Ignored by other reward modes.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--train-timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--eval-episodes", type=int, default=DEFAULT_EVAL_EPISODES)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for Track B auditable run folders.",
    )
    parser.add_argument(
        "--heartbeat-interval-seconds",
        type=float,
        default=30.0,
        help="Heartbeat cadence for long-running audits.",
    )
    return parser


def resolve_run_dir(args: argparse.Namespace) -> Path:
    return args.output_root / args.label


def resolve_artifacts(run_dir: Path) -> RunArtifacts:
    return RunArtifacts(
        run_dir=run_dir,
        command_txt=run_dir / "command.txt",
        pid_txt=run_dir / "pid.txt",
        status_json=run_dir / "status.json",
        stdout_log=run_dir / "stdout.log",
        heartbeat_json=run_dir / "heartbeat.json",
        manifest_json=run_dir / "manifest.json",
        failed_json=run_dir / "FAILED.json",
    )


def build_launcher_command(args: argparse.Namespace) -> str:
    return " ".join(
        [
            "python",
            "scripts/run_track_b_benchmark.py",
            "--label",
            str(args.label),
            "--reward-mode",
            str(args.reward_mode),
            "--ret-seq-kappa",
            str(args.ret_seq_kappa),
            "--train-timesteps",
            str(args.train_timesteps),
            "--eval-episodes",
            str(args.eval_episodes),
            "--output-root",
            str(args.output_root),
            "--heartbeat-interval-seconds",
            str(args.heartbeat_interval_seconds),
            "--seeds",
            *[str(seed) for seed in args.seeds],
        ]
    )


def build_benchmark_cli_args(args: argparse.Namespace, run_dir: Path) -> list[str]:
    return [
        "--output-dir",
        str(run_dir),
        "--reward-mode",
        str(args.reward_mode),
        "--ret-seq-kappa",
        str(args.ret_seq_kappa),
        "--risk-level",
        str(FROZEN_BACKBONE["risk_level"]),
        "--step-size-hours",
        str(FROZEN_BACKBONE["step_size_hours"]),
        "--max-steps",
        str(FROZEN_BACKBONE["max_steps"]),
        "--train-timesteps",
        str(args.train_timesteps),
        "--eval-episodes",
        str(args.eval_episodes),
        "--seeds",
        *[str(seed) for seed in args.seeds],
    ]


def build_benchmark_command(cli_args: list[str]) -> str:
    return "python scripts/run_track_b_smoke.py " + " ".join(cli_args)


def validate_required_outputs(run_dir: Path) -> list[str]:
    return [name for name in REQUIRED_OUTPUTS if not (run_dir / name).exists()]


def build_status_payload(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    benchmark_command: str,
    state: str,
    started_at_utc: str,
    finished_at_utc: str | None = None,
    invalid_reason: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "label": args.label,
        "state": state,
        "pid": os.getpid(),
        "run_dir": str(run_dir.resolve()),
        "started_at_utc": started_at_utc,
        "updated_at_utc": utc_now_iso(),
        "finished_at_utc": finished_at_utc,
        "launcher_command": build_launcher_command(args),
        "benchmark_command": benchmark_command,
        "required_outputs": list(REQUIRED_OUTPUTS),
        "frozen_backbone": dict(FROZEN_BACKBONE),
    }
    if invalid_reason is not None:
        payload["invalid_reason"] = invalid_reason
    return payload


def write_manifest(
    *,
    args: argparse.Namespace,
    artifacts: RunArtifacts,
    benchmark_command: str,
    summary: dict[str, Any] | None,
    state: str,
) -> None:
    files = {
        "command.txt": str(artifacts.command_txt.resolve()),
        "pid.txt": str(artifacts.pid_txt.resolve()),
        "status.json": str(artifacts.status_json.resolve()),
        "stdout.log": str(artifacts.stdout_log.resolve()),
        "heartbeat.json": str(artifacts.heartbeat_json.resolve()),
    }
    for filename in REQUIRED_OUTPUTS:
        path = artifacts.run_dir / filename
        if path.exists():
            files[filename] = str(path.resolve())
    if artifacts.failed_json.exists():
        files["FAILED.json"] = str(artifacts.failed_json.resolve())

    manifest = {
        "artifact_type": "track_b_benchmark_run",
        "status": state,
        "generated_at_utc": utc_now_iso(),
        "label": args.label,
        "run_directory": str(artifacts.run_dir.resolve()),
        "launcher_pid": os.getpid(),
        "launcher_command": build_launcher_command(args),
        "benchmark_command": benchmark_command,
        "backbone": (summary or {}).get("backbone", dict(FROZEN_BACKBONE)),
        "frozen_backbone": dict(FROZEN_BACKBONE),
        "metric_contract": (summary or {}).get("metric_contract", {}),
        "reward_contract": (summary or {}).get("reward_contract", {}),
        "config": (summary or {}).get("config", {}),
        "files": files,
    }
    write_json(artifacts.manifest_json, manifest)


def augment_summary(
    *,
    summary_path: Path,
    args: argparse.Namespace,
    benchmark_command: str,
    artifacts: RunArtifacts,
) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["launcher_metadata"] = {
        "label": args.label,
        "launcher_pid": os.getpid(),
        "launcher_command": build_launcher_command(args),
        "benchmark_command": benchmark_command,
        "status_json": str(artifacts.status_json.resolve()),
        "heartbeat_json": str(artifacts.heartbeat_json.resolve()),
        "stdout_log": str(artifacts.stdout_log.resolve()),
        "manifest_json": str(artifacts.manifest_json.resolve()),
        "command_txt": str(artifacts.command_txt.resolve()),
        "pid_txt": str(artifacts.pid_txt.resolve()),
    }
    summary.setdefault("artifacts", {})
    summary["artifacts"]["manifest_json"] = str(artifacts.manifest_json.resolve())
    write_json(summary_path, summary)
    return summary


def invoke_benchmark(benchmark_args: argparse.Namespace) -> dict[str, Any]:
    return run_smoke(benchmark_args)


def run_with_heartbeat(
    *,
    args: argparse.Namespace,
    artifacts: RunArtifacts,
    benchmark_args: argparse.Namespace,
) -> dict[str, Any]:
    stop_event = threading.Event()
    heartbeat_lock = threading.Lock()

    def _write_heartbeat(state: str) -> None:
        payload = {
            "label": args.label,
            "state": state,
            "pid": os.getpid(),
            "last_activity_utc": utc_now_iso(),
            "required_outputs": list(REQUIRED_OUTPUTS),
        }
        with heartbeat_lock:
            write_json(artifacts.heartbeat_json, payload)

    def _loop() -> None:
        while not stop_event.wait(args.heartbeat_interval_seconds):
            _write_heartbeat("running")

    thread = threading.Thread(target=_loop, daemon=True)
    _write_heartbeat("running")
    thread.start()
    try:
        return invoke_benchmark(benchmark_args)
    finally:
        stop_event.set()
        thread.join(timeout=max(1.0, args.heartbeat_interval_seconds))


def run_launcher(args: argparse.Namespace) -> int:
    run_dir = resolve_run_dir(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts = resolve_artifacts(run_dir)
    benchmark_cli_args = build_benchmark_cli_args(args, run_dir)
    benchmark_command = build_benchmark_command(benchmark_cli_args)
    benchmark_args = build_track_b_parser().parse_args(benchmark_cli_args)
    benchmark_args.invocation = benchmark_command
    started_at_utc = utc_now_iso()

    artifacts.command_txt.write_text(f"{benchmark_command}\n", encoding="utf-8")
    artifacts.pid_txt.write_text(f"{os.getpid()}\n", encoding="utf-8")
    artifacts.stdout_log.touch()
    write_json(
        artifacts.status_json,
        build_status_payload(
            args=args,
            run_dir=run_dir,
            benchmark_command=benchmark_command,
            state="running",
            started_at_utc=started_at_utc,
        ),
    )

    with artifacts.stdout_log.open("a", encoding="utf-8") as log_file:
        tee_stdout = TeeStream(sys.stdout, log_file)
        tee_stderr = TeeStream(sys.stderr, log_file)
        with contextlib.redirect_stdout(tee_stdout), contextlib.redirect_stderr(
            tee_stderr
        ):
            print(f"[{utc_now_iso()}] starting Track B benchmark: {args.label}")
            print(f"[{utc_now_iso()}] benchmark command: {benchmark_command}")
            try:
                run_with_heartbeat(
                    args=args,
                    artifacts=artifacts,
                    benchmark_args=benchmark_args,
                )
                missing_outputs = validate_required_outputs(run_dir)
                if missing_outputs:
                    reason = (
                        "Run completed without the minimum valid artifacts: "
                        + ", ".join(missing_outputs)
                    )
                    write_json(
                        artifacts.failed_json,
                        {
                            "label": args.label,
                            "state": "invalid",
                            "failed_at_utc": utc_now_iso(),
                            "reason": reason,
                            "missing_outputs": missing_outputs,
                            "benchmark_command": benchmark_command,
                            "run_directory": str(run_dir.resolve()),
                        },
                    )
                    write_json(
                        artifacts.status_json,
                        build_status_payload(
                            args=args,
                            run_dir=run_dir,
                            benchmark_command=benchmark_command,
                            state="invalid",
                            started_at_utc=started_at_utc,
                            finished_at_utc=utc_now_iso(),
                            invalid_reason=reason,
                        ),
                    )
                    write_manifest(
                        args=args,
                        artifacts=artifacts,
                        benchmark_command=benchmark_command,
                        summary=None,
                        state="invalid",
                    )
                    print(f"[{utc_now_iso()}] invalid run: {reason}")
                    return 1

                summary = augment_summary(
                    summary_path=run_dir / "summary.json",
                    args=args,
                    benchmark_command=benchmark_command,
                    artifacts=artifacts,
                )
                write_manifest(
                    args=args,
                    artifacts=artifacts,
                    benchmark_command=benchmark_command,
                    summary=summary,
                    state="completed",
                )
                write_json(
                    artifacts.status_json,
                    build_status_payload(
                        args=args,
                        run_dir=run_dir,
                        benchmark_command=benchmark_command,
                        state="completed",
                        started_at_utc=started_at_utc,
                        finished_at_utc=utc_now_iso(),
                    ),
                )
                write_json(
                    artifacts.heartbeat_json,
                    {
                        "label": args.label,
                        "state": "completed",
                        "pid": os.getpid(),
                        "last_activity_utc": utc_now_iso(),
                        "required_outputs": list(REQUIRED_OUTPUTS),
                    },
                )
                print(
                    f"[{utc_now_iso()}] completed Track B benchmark: {run_dir.resolve()}"
                )
                return 0
            except Exception as exc:  # pragma: no cover
                write_json(
                    artifacts.failed_json,
                    {
                        "label": args.label,
                        "state": "failed",
                        "failed_at_utc": utc_now_iso(),
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "traceback": traceback.format_exc(),
                        "benchmark_command": benchmark_command,
                        "run_directory": str(run_dir.resolve()),
                    },
                )
                write_json(
                    artifacts.status_json,
                    build_status_payload(
                        args=args,
                        run_dir=run_dir,
                        benchmark_command=benchmark_command,
                        state="failed",
                        started_at_utc=started_at_utc,
                        finished_at_utc=utc_now_iso(),
                        invalid_reason=str(exc),
                    ),
                )
                write_manifest(
                    args=args,
                    artifacts=artifacts,
                    benchmark_command=benchmark_command,
                    summary=None,
                    state="failed",
                )
                write_json(
                    artifacts.heartbeat_json,
                    {
                        "label": args.label,
                        "state": "failed",
                        "pid": os.getpid(),
                        "last_activity_utc": utc_now_iso(),
                        "required_outputs": list(REQUIRED_OUTPUTS),
                    },
                )
                print(f"[{utc_now_iso()}] failed Track B benchmark: {exc}")
                return 1


def main() -> None:
    raise SystemExit(run_launcher(build_parser().parse_args()))


if __name__ == "__main__":
    main()
