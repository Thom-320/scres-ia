#!/usr/bin/env python3
"""Launch watcher-first Program O execution under whole-session custody."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parent.parent


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("development", "validation"), required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--validation-freeze", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--watch-interval-seconds", type=float, default=10.0)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    contract = args.contract.resolve()
    validation_freeze = args.validation_freeze.resolve()
    failures = []
    current_commit = git_commit()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip()
    if current_commit != str(args.expected_commit):
        failures.append("HEAD does not equal expected immutable commit")
    if dirty:
        failures.append("worktree is dirty")
    if not contract.is_file():
        failures.append("contract is absent")
    if args.stage == "validation" and not validation_freeze.is_file():
        failures.append("validation freeze is absent")
    if run_dir.exists():
        failures.append("run identity already exists")
    preflight = {
        "checked_at_utc": now_utc(),
        "passed": not failures,
        "failures": failures,
        "run_dir": str(run_dir),
        "stage": str(args.stage),
        "workers": int(args.workers),
        "expected_commit": str(args.expected_commit),
        "current_commit": current_commit,
        "contract": str(contract),
        "contract_sha256": sha256(contract) if contract.is_file() else None,
        "validation_freeze": str(validation_freeze),
    }
    if args.preflight_only:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return 0 if not failures else 1
    if failures:
        raise RuntimeError("launch preflight failed: " + "; ".join(failures))

    custody = run_dir / "custody"
    custody.mkdir(parents=True)
    write_json_atomic(custody / "launch_manifest.json", preflight)
    watcher_command = [
        sys.executable,
        str(ROOT / "scripts/watch_program_o_full_des_hpi.py"),
        "--run-dir",
        str(run_dir),
        "--interval-seconds",
        str(args.watch_interval_seconds),
    ]
    watcher_stdout = (custody / "watcher.stdout.log").open("ab")
    watcher_stderr = (custody / "watcher.stderr.log").open("ab")
    watcher = subprocess.Popen(
        watcher_command,
        cwd=ROOT,
        stdout=watcher_stdout,
        stderr=watcher_stderr,
        start_new_session=True,
    )
    watcher_stdout.close()
    watcher_stderr.close()
    ready_path = custody / "watcher_ready.json"
    watcher_state_path = custody / "watcher_state.json"
    deadline = time.time() + 30.0
    prestart_confirmed = False
    while time.time() < deadline:
        if watcher.poll() is not None:
            raise RuntimeError("watcher exited before declaring readiness")
        if ready_path.is_file() and watcher_state_path.is_file():
            try:
                watcher_state = json.loads(watcher_state_path.read_text())
            except json.JSONDecodeError:
                watcher_state = {}
            if watcher_state.get("status") == "AWAITING_PRODUCER_CONTROL":
                prestart_confirmed = True
                break
        time.sleep(0.1)
    if not prestart_confirmed:
        watcher.terminate()
        raise TimeoutError("watcher did not record fail-closed prestart custody")

    runner_command = [
        sys.executable,
        str(ROOT / "scripts/run_program_o_full_des_hpi_custodied.py"),
        "--run-dir",
        str(run_dir),
        "--stage",
        str(args.stage),
        "--workers",
        str(args.workers),
        "--contract",
        str(contract),
        "--validation-freeze",
        str(validation_freeze),
    ]
    producer_stdout = (custody / "producer.stdout.log").open("ab")
    producer_stderr = (custody / "producer.stderr.log").open("ab")
    producer = subprocess.Popen(
        runner_command,
        cwd=ROOT,
        stdout=producer_stdout,
        stderr=producer_stderr,
        start_new_session=True,
    )
    producer_stdout.close()
    producer_stderr.close()
    control = {
        "schema_version": "program_o_full_des_producer_control_v1",
        "launched_at_utc": now_utc(),
        "stage": str(args.stage),
        "producer_pid": producer.pid,
        "producer_pgid": os.getpgid(producer.pid),
        "producer_sid": os.getsid(producer.pid),
        "watcher_pid": watcher.pid,
        "watcher_pgid": os.getpgid(watcher.pid),
        "runner_command": runner_command,
        "watcher_command": watcher_command,
        "scientific_commit": current_commit,
        "contract_sha256": sha256(contract),
    }
    write_json_atomic(custody / "producer_control.json", control)
    print(json.dumps(control, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
