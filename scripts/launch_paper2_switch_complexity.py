#!/usr/bin/env python3
"""Launch the switch-complexity screen or deep verifier after its watcher."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
RESULT_ROOT = (
    ROOT / "results" / "paper2_bound_harness" / "switch_complexity_screen"
)
SEARCH = ROOT / "scripts" / "search_paper2_bottleneck_switch_complexity.py"
VERIFIER = ROOT / "scripts" / "verify_paper2_bottleneck_switch_complexity.py"
WATCHER = ROOT / "scripts" / "watch_paper2_switch_complexity.py"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def exclusive_json(path: Path, payload: dict[str, Any]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--watch-interval-seconds", type=float, default=10.0)
    parser.add_argument("--deep-verify-result", type=Path)
    args = parser.parse_args()
    if args.workers < 1 or args.watch_interval_seconds <= 0:
        parser.error("workers and watcher interval must be positive")
    run_dir = args.run_dir.resolve(strict=False)
    try:
        run_dir.relative_to(RESULT_ROOT.resolve())
    except ValueError as exc:
        parser.error(f"run directory must be under {RESULT_ROOT}: {exc}")
    if run_dir.exists():
        parser.error("run directory already exists; runs are non-overwriting")
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, check=True,
        text=True, stdout=subprocess.PIPE,
    ).stdout.splitlines()
    if status:
        parser.error("detached launch requires a clean immutable worktree")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        text=True, stdout=subprocess.PIPE,
    ).stdout.strip()
    run_dir.mkdir(parents=True, mode=0o700)
    os.chmod(run_dir, 0o700)
    progress = run_dir / "progress.json"
    verified_result: Path | None = None
    if args.deep_verify_result is None:
        operation = "calibration_screen"
        output = run_dir / "result.json"
        command = [
            sys.executable, str(SEARCH), "--output", str(output),
            "--progress", str(progress), "--workers", str(args.workers),
        ]
    else:
        operation = "deep_verification"
        output = run_dir / "verification.json"
        verified_result = args.deep_verify_result.resolve(strict=True)
        command = [
            sys.executable, str(VERIFIER), "--result", str(verified_result),
            "--output", str(output), "--progress", str(progress),
            "--workers", str(args.workers), "--deep",
        ]
    manifest = {
        "schema_version": "paper2_switch_complexity_detached_launch_v1",
        "created_at_utc": utc_now(),
        "git_head": head,
        "launch_git_status_porcelain": status,
        "cwd": str(ROOT),
        "operation": operation,
        "command": command,
        "output": str(output),
        "progress": str(progress),
        "workers": args.workers,
        "watch_interval_seconds": args.watch_interval_seconds,
        "launcher_sha256": file_sha256(Path(__file__).resolve()),
        "search_sha256": file_sha256(SEARCH),
        "verifier_sha256": file_sha256(VERIFIER),
        "watcher_sha256": file_sha256(WATCHER),
        "verified_result": str(verified_result) if verified_result else None,
        "verified_result_sha256": (
            file_sha256(verified_result) if verified_result else None
        ),
    }
    exclusive_json(run_dir / "launch_manifest.json", manifest)
    watcher_stdout = (run_dir / "watcher_stdout.log").open("ab", buffering=0)
    watcher_stderr = (run_dir / "watcher_stderr.log").open("ab", buffering=0)
    watcher = subprocess.Popen(
        [
            sys.executable, str(WATCHER), "--run-dir", str(run_dir),
            "--interval-seconds", str(args.watch_interval_seconds),
        ],
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        stdout=watcher_stdout,
        stderr=watcher_stderr,
        start_new_session=True,
        close_fds=True,
    )
    latest_path = run_dir / "watcher_latest.json"
    deadline = time.monotonic() + 15.0
    latest = None
    while time.monotonic() < deadline:
        latest = load_json(latest_path)
        if (
            latest
            and latest.get("state") == "watching_prestart"
            and latest.get("watcher_pid") == watcher.pid
        ):
            break
        time.sleep(0.1)
    else:
        watcher.terminate()
        parser.error("watcher did not attest prestart liveness")

    stdout = (run_dir / "stdout.log").open("ab", buffering=0)
    stderr = (run_dir / "stderr.log").open("ab", buffering=0)
    scientific = subprocess.Popen(
        command,
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        stdout=stdout,
        stderr=stderr,
        start_new_session=True,
        close_fds=True,
    )
    pid_payload = {
        "schema_version": "paper2_switch_complexity_pid_v1",
        "launched_at_utc": utc_now(),
        "scientific_pid": scientific.pid,
        "watcher_pid": watcher.pid,
        "watcher_prestart_observed_at_utc": latest["observed_at_utc"],
        "command": command,
        "output": str(output),
        "progress": str(progress),
        "git_head": head,
    }
    exclusive_json(run_dir / "pid.json", pid_payload)
    receipt = {
        **manifest,
        "scientific_pid": scientific.pid,
        "watcher_pid": watcher.pid,
        "watcher_started_before_scientific_process": True,
        "pid_record_sha256": file_sha256(run_dir / "pid.json"),
    }
    exclusive_json(run_dir / "launch_receipt.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
