#!/usr/bin/env python3
"""Launch the frozen reference-calendar screen detached with a prior watcher."""
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
SCREEN = ROOT / "scripts" / "search_paper2_bottleneck_reference_calendar.py"
VERIFIER = ROOT / "scripts" / "verify_paper2_bottleneck_reference_calendar.py"
WATCHER = ROOT / "scripts" / "watch_paper2_reference_screen.py"


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
    parser.add_argument(
        "--deep-verify-result",
        type=Path,
        help="launch the independent deep verifier instead of the screen",
    )
    args = parser.parse_args()
    if args.workers < 1 or args.watch_interval_seconds <= 0:
        parser.error("workers and watcher interval must be positive")
    run_dir = args.run_dir.resolve(strict=False)
    allowed = (
        ROOT / "results" / "paper2_bound_harness" / "reference_calendar_screen"
    ).resolve()
    try:
        run_dir.relative_to(allowed)
    except ValueError as exc:
        parser.error(f"run directory must be under {allowed}: {exc}")
    if run_dir.exists():
        parser.error("run directory already exists; evidence runs are non-overwriting")
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
    output = run_dir / "result.json"
    progress = run_dir / "progress.json"
    if args.deep_verify_result is None:
        operation = "screen"
        command = [
            sys.executable,
            str(SCREEN),
            "--output",
            str(output),
            "--progress",
            str(progress),
            "--workers",
            str(args.workers),
        ]
        verified_result = None
    else:
        operation = "deep_verification"
        verified_result = args.deep_verify_result.resolve(strict=True)
        command = [
            sys.executable,
            str(VERIFIER),
            "--result",
            str(verified_result),
            "--output",
            str(output),
            "--progress",
            str(progress),
            "--workers",
            str(args.workers),
            "--deep",
        ]
    manifest = {
        "schema_version": "paper2_reference_screen_detached_launch_v1",
        "created_at_utc": utc_now(),
        "git_head": head,
        "launch_git_status_porcelain": status,
        "cwd": str(ROOT),
        "command": command,
        "launcher_sha256": file_sha256(Path(__file__).resolve()),
        "watcher_sha256": file_sha256(WATCHER),
        "screen_sha256": file_sha256(SCREEN),
        "verifier_sha256": file_sha256(VERIFIER),
        "operation": operation,
        "verified_result": str(verified_result) if verified_result else None,
        "verified_result_sha256": (
            file_sha256(verified_result) if verified_result else None
        ),
        "workers": args.workers,
        "watch_interval_seconds": args.watch_interval_seconds,
    }
    exclusive_json(run_dir / "launch_manifest.json", manifest)
    watcher_stdout = (run_dir / "watcher_stdout.log").open("ab", buffering=0)
    watcher_stderr = (run_dir / "watcher_stderr.log").open("ab", buffering=0)
    watcher = subprocess.Popen(
        [
            sys.executable,
            str(WATCHER),
            "--run-dir",
            str(run_dir),
            "--interval-seconds",
            str(args.watch_interval_seconds),
        ],
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        stdout=watcher_stdout,
        stderr=watcher_stderr,
        start_new_session=True,
        close_fds=True,
    )
    deadline = time.monotonic() + 15.0
    latest_path = run_dir / "watcher_latest.json"
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
        parser.error("independent watcher did not attest prestart liveness")

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
        "schema_version": "paper2_reference_screen_pid_v1",
        "launched_at_utc": utc_now(),
        "scientific_pid": scientific.pid,
        "watcher_pid": watcher.pid,
        "watcher_prestart_observed_at_utc": latest["observed_at_utc"],
        "watcher_prestart_sha256": sha256(
            json.dumps(latest, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "command": command,
        "git_head": head,
    }
    exclusive_json(run_dir / "pid.json", pid_payload)
    launch_receipt = {
        **manifest,
        "scientific_pid": scientific.pid,
        "watcher_pid": watcher.pid,
        "watcher_started_before_scientific_process": True,
        "pid_record_sha256": file_sha256(run_dir / "pid.json"),
    }
    exclusive_json(run_dir / "launch_receipt.json", launch_receipt)
    print(json.dumps(launch_receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
