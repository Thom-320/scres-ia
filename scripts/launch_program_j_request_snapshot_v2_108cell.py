#!/usr/bin/env python3
"""Launch the frozen Program-J v2 108-cell screen after a prestart watcher."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

from produce_program_j_request_snapshot_v2_108cell_frontier import (
    ALL_SEQUENCES,
    all_cells,
    json_sha256,
    run_contract,
)


ROOT = Path(__file__).resolve().parent.parent
RUN_ROOT = ROOT / "outputs/program_j_request_snapshot_v2_108cell_vps_runs"
PRODUCER = ROOT / "scripts/produce_program_j_request_snapshot_v2_108cell_frontier.py"
WATCHER = ROOT / "scripts/watch_program_j_request_snapshot_v2_108cell.py"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def checked_output(command: list[str]) -> str:
    return subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed-start", type=int, default=1_200_001)
    parser.add_argument("--tapes-per-cell", type=int, default=4)
    parser.add_argument("--watch-interval-seconds", type=float, default=10.0)
    args = parser.parse_args()
    if not 1 <= args.workers <= 6:
        parser.error("workers must be between 1 and 6")
    if args.tapes_per_cell != 4 or args.seed_start != 1_200_001:
        parser.error("full screen seed contract is frozen at four tapes from 1200001")
    run_dir = args.run_dir.resolve(strict=False)
    try:
        run_dir.relative_to(RUN_ROOT.resolve())
    except ValueError as exc:
        parser.error(f"run directory must be under {RUN_ROOT}: {exc}")
    if run_dir.exists():
        parser.error("run directory already exists; runs are non-overwriting")
    status = checked_output(["git", "status", "--porcelain"]).splitlines()
    if status:
        parser.error("detached launch requires a clean immutable worktree")
    head = checked_output(["git", "rev-parse", "HEAD"])
    run_dir.mkdir(parents=True, mode=0o700)
    os.chmod(run_dir, 0o700)

    progress = run_dir / "progress.json"
    result = run_dir / "verdict.json"
    frozen_contract_path = run_dir / "frozen_run_contract.json"
    scientific_work = run_dir / "scientific_work"
    contract = run_contract(
        cells=all_cells(),
        sequences=ALL_SEQUENCES,
        seed_start=args.seed_start,
        tapes_per_cell=args.tapes_per_cell,
        smoke=False,
    )
    frozen = dict(contract)
    frozen["content_sha256"] = json_sha256(frozen)
    exclusive_json(frozen_contract_path, frozen)
    command = [
        sys.executable,
        str(PRODUCER),
        "--workers",
        str(args.workers),
        "--seed-start",
        str(args.seed_start),
        "--tapes-per-cell",
        str(args.tapes_per_cell),
        "--work-root",
        str(scientific_work),
        "--progress",
        str(progress),
        "--run-contract",
        str(frozen_contract_path),
        "--use-precreated-run-contract",
        "--verdict",
        str(result),
    ]
    seeds = list(range(args.seed_start, args.seed_start + args.tapes_per_cell))
    seed_manifest = {
        "schema_version": "program_j_request_snapshot_v2_seed_manifest_v1",
        "seeds": seeds,
        "use": "BURNED_DEVELOPMENT_SCREEN_ONLY",
        "common_random_numbers": "Same seed set in all 108 cells and all policies.",
        "virgin": False,
    }
    environment = {
        "schema_version": "program_j_request_snapshot_v2_environment_v1",
        "captured_at_utc": utc_now(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "pip_freeze": checked_output([sys.executable, "-m", "pip", "freeze"]).splitlines(),
    }
    exclusive_json(run_dir / "seed_manifest.json", seed_manifest)
    exclusive_json(run_dir / "environment_snapshot.json", environment)
    launch_manifest = {
        "schema_version": "program_j_request_snapshot_v2_108cell_launch_manifest_v1",
        "created_at_utc": utc_now(),
        "git_head": head,
        "git_status_porcelain": status,
        "cwd": str(ROOT),
        "command": command,
        "workers": args.workers,
        "progress": str(progress),
        "result": str(result),
        "run_contract": str(frozen_contract_path),
        "run_contract_file_sha256": file_sha256(frozen_contract_path),
        "run_contract_content_sha256": frozen["content_sha256"],
        "producer_sha256": file_sha256(PRODUCER),
        "launcher_sha256": file_sha256(Path(__file__).resolve()),
        "watcher_sha256": file_sha256(WATCHER),
        "seed_manifest_sha256": file_sha256(run_dir / "seed_manifest.json"),
        "environment_snapshot_sha256": file_sha256(
            run_dir / "environment_snapshot.json"
        ),
        "watch_interval_seconds": args.watch_interval_seconds,
    }
    exclusive_json(run_dir / "launch_manifest.json", launch_manifest)

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
    session_id = os.getsid(scientific.pid)
    process_group_id = os.getpgid(scientific.pid)
    if session_id != scientific.pid or process_group_id != scientific.pid:
        os.killpg(process_group_id, 15)
        parser.error("scientific process did not receive an isolated session/group")
    pid_record = {
        "schema_version": "program_j_request_snapshot_v2_108cell_pid_v1",
        "launched_at_utc": utc_now(),
        "scientific_pid": scientific.pid,
        "scientific_process_group_id": process_group_id,
        "scientific_session_id": session_id,
        "watcher_pid": watcher.pid,
        "watcher_prestart_observed_at_utc": latest["observed_at_utc"],
        "command": command,
        "progress": str(progress),
        "result": str(result),
        "git_head": head,
    }
    exclusive_json(run_dir / "pid.json", pid_record)
    receipt = {
        **launch_manifest,
        "scientific_pid": scientific.pid,
        "scientific_process_group_id": process_group_id,
        "scientific_session_id": session_id,
        "watcher_pid": watcher.pid,
        "watcher_started_before_scientific_process": True,
        "watcher_scope": "ENTIRE_SCIENTIFIC_SESSION_INCLUDING_REPARENTED_WORKERS",
        "pid_record_sha256": file_sha256(run_dir / "pid.json"),
    }
    exclusive_json(run_dir / "launch_receipt.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
