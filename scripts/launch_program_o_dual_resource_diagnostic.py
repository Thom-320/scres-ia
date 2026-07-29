#!/usr/bin/env python3
"""Launch the dual-resource diagnostic with watcher-first session custody."""

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
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--parent-run", type=Path, required=True)
    parser.add_argument("--execution-freeze", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--watch-interval-seconds", type=float, default=5.0)
    args = parser.parse_args()

    run_dir = Path(os.path.abspath(str(args.run_dir)))
    contract = args.contract.resolve()
    source_result = args.source_result.resolve()
    parent_run = args.parent_run.resolve()
    parent_result = parent_run / "result.json"
    freeze_path = args.execution_freeze.resolve()
    freeze = json.loads(freeze_path.read_text())
    current_head = git_commit()
    failures = []
    if current_head != str(args.expected_head):
        failures.append("HEAD does not equal expected immutable execution commit")
    if subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip():
        failures.append("worktree is dirty")
    if run_dir.exists():
        failures.append("run identity already exists")
    if sha256(contract) != str(freeze["contract_sha256"]):
        failures.append("contract hash differs from execution freeze")
    if sha256(source_result) != str(freeze["source_result_sha256"]):
        failures.append("stopped state-rich result hash differs from freeze")
    if not parent_result.is_file() or sha256(parent_result) != str(
        freeze["parent_run"]["result_sha256"]
    ):
        failures.append("parent label-only result missing or hash mismatch")
    for relative, expected in freeze["source_sha256"].items():
        if sha256(ROOT / relative) != str(expected):
            failures.append(f"source hash differs from freeze: {relative}")
    source_payload = json.loads(source_result.read_text())
    if source_payload.get("seeds") != list(range(7420001, 7420049)):
        failures.append("source result is not the burned-fit seed block")
    if source_payload.get("validation_seed_accessed") is not False:
        failures.append("source result does not fail closed on validation access")

    manifest = {
        "schema_version": "program_o_dual_resource_diagnostic_launch_manifest_v1",
        "checked_at_utc": now_utc(),
        "passed": not failures,
        "failures": failures,
        "stage": "diagnostic",
        "run_id": str(args.run_id),
        "run_dir": str(run_dir),
        "expected_head": str(args.expected_head),
        "current_head": current_head,
        "contract": str(contract),
        "contract_sha256": sha256(contract),
        "source_result": str(source_result),
        "source_result_sha256": sha256(source_result),
        "parent_run": str(parent_run),
        "parent_result_sha256": (
            sha256(parent_result) if parent_result.is_file() else None
        ),
        "execution_freeze": str(freeze_path),
        "execution_freeze_sha256": sha256(freeze_path),
        "validation_seed_access_authorized": False,
    }
    if failures:
        raise RuntimeError("launch preflight failed: " + "; ".join(failures))

    custody = run_dir / "custody"
    custody.mkdir(parents=True)
    write_json_atomic(custody / "launch_manifest.json", manifest)
    watcher_command = [
        sys.executable,
        str(ROOT / "scripts/watch_program_o_full_des_hpi.py"),
        "--run-dir",
        str(run_dir),
        "--interval-seconds",
        str(args.watch_interval_seconds),
    ]
    with (custody / "watcher.stdout.log").open("ab") as stdout, (
        custody / "watcher.stderr.log"
    ).open("ab") as stderr:
        watcher = subprocess.Popen(
            watcher_command,
            cwd=ROOT,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
    deadline = time.time() + 30.0
    while time.time() < deadline:
        state_path = custody / "watcher_state.json"
        if watcher.poll() is not None:
            raise RuntimeError("watcher exited before producer launch")
        if state_path.is_file():
            try:
                state = json.loads(state_path.read_text())
            except json.JSONDecodeError:
                state = {}
            if state.get("status") == "AWAITING_PRODUCER_CONTROL":
                break
        time.sleep(0.1)
    else:
        watcher.terminate()
        raise TimeoutError("watcher did not establish prestart custody")

    runner_command = [
        sys.executable,
        str(ROOT / "scripts/run_program_o_dual_resource_diagnostic_custodied.py"),
        "--run-dir",
        str(run_dir),
        "--run-id",
        str(args.run_id),
        "--contract",
        str(contract),
        "--source-result",
        str(source_result),
        "--parent-run",
        str(parent_run),
        "--execution-freeze",
        str(freeze_path),
    ]
    with (custody / "producer.stdout.log").open("ab") as stdout, (
        custody / "producer.stderr.log"
    ).open("ab") as stderr:
        producer = subprocess.Popen(
            runner_command,
            cwd=ROOT,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
    control = {
        "schema_version": "program_o_dual_resource_diagnostic_control_v1",
        "launched_at_utc": now_utc(),
        "stage": "diagnostic",
        "run_id": str(args.run_id),
        "producer_pid": producer.pid,
        "producer_pgid": os.getpgid(producer.pid),
        "producer_sid": os.getsid(producer.pid),
        "watcher_pid": watcher.pid,
        "watcher_pgid": os.getpgid(watcher.pid),
        "runner_command": runner_command,
        "watcher_command": watcher_command,
        "execution_commit": current_head,
        "scientific_commit": str(freeze["scientific_commit"]),
        "contract_sha256": sha256(contract),
        "execution_freeze_sha256": sha256(freeze_path),
    }
    if not (
        int(control["producer_pid"])
        == int(control["producer_pgid"])
        == int(control["producer_sid"])
    ):
        producer.terminate()
        watcher.terminate()
        raise RuntimeError("producer is not isolated as PID=PGID=SID")
    write_json_atomic(custody / "producer_control.json", control)
    print(json.dumps(control, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
