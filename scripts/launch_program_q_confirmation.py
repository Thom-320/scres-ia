#!/usr/bin/env python3
"""Watcher-first launcher for the one-shot Program Q confirmation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.evaluate_program_q_replication import (  # noqa: E402
    CONTRACT,
    verify_authorization,
    verify_model_hashes,
)
from scripts.audit_program_q_seed_custody import scan  # noqa: E402
from supply_chain.program_o_eval_custody import sha256  # noqa: E402

PREOPEN_AUDIT = (
    ROOT
    / "research/paper2_exhaustive_search/program_q_confirmation_preopen_audit_v1.json"
)

def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


LIVE_ATTEMPT_FILES = (
    "producer_control.json",
    "producer_exit.json",
    "watcher_ready.json",
    "watcher_state.json",
)


def archive_previous_attempt(custody: Path, names: Iterable[str] = LIVE_ATTEMPT_FILES) -> int:
    attempts = custody / "attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    existing = [
        int(path.name.split("_")[-1])
        for path in attempts.glob("attempt_*")
        if path.is_dir() and path.name.split("_")[-1].isdigit()
    ]
    attempt_id = max(existing, default=0) + 1
    destination = attempts / f"attempt_{attempt_id}"
    destination.mkdir()
    moved = 0
    for name in names:
        source = custody / name
        if source.exists():
            os.replace(source, destination / name)
            moved += 1
    if moved == 0:
        destination.rmdir()
        raise RuntimeError("resume requested without prior live custody state")
    return attempt_id


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    failures: list[str] = []
    current_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    if run_dir.exists() and not args.resume:
        failures.append("run identity already exists")
    if args.resume and not run_dir.exists():
        failures.append("resume requested but run identity does not exist")
    if args.resume and (run_dir / "artifacts/confirmation/adjudication.json").is_file():
        failures.append("completed Program Q run cannot be resumed")
    if args.resume:
        for filename, field in (
            ("producer_control.json", "producer_pid"),
            ("watcher_ready.json", "watcher_pid"),
        ):
            state_path = run_dir / "custody" / filename
            if state_path.is_file():
                try:
                    pid = json.loads(state_path.read_text()).get(field)
                except json.JSONDecodeError:
                    failures.append(f"malformed prior custody file: {filename}")
                else:
                    if pid_alive(pid):
                        failures.append(f"prior {field} is still alive")
    if current_commit != args.expected_commit:
        failures.append("HEAD does not equal expected immutable commit")
    tracked_dirty = (
        subprocess.run(["git", "diff", "--quiet"], cwd=ROOT, check=False).returncode != 0
        or subprocess.run(
            ["git", "diff", "--cached", "--quiet"], cwd=ROOT, check=False
        ).returncode
        != 0
    )
    if tracked_dirty:
        failures.append("tracked worktree differs from immutable commit")
    try:
        authorization = verify_authorization(args.authorization)
        if authorization.get("preopen_audit_sha256") != sha256(PREOPEN_AUDIT):
            raise RuntimeError("Program Q authorization preopen audit hash mismatch")
    except (OSError, RuntimeError, json.JSONDecodeError) as error:
        failures.append(str(error))
    seed_scan = scan(ROOT)
    if seed_scan.get("status") != "PROGRAM_Q_SEEDS_VIRGIN":
        failures.append("live Program Q reserved-seed scan failed")
    try:
        model_hashes = verify_model_hashes(args.models)
    except (OSError, RuntimeError) as error:
        failures.append(str(error))
        model_hashes = {}
    existing_shards = (
        len(list((run_dir / "artifacts/confirmation/shards").rglob("*.npz")))
        if run_dir.exists()
        else 0
    )
    preflight = {
        "checked_at": now_utc(),
        "passed": not failures,
        "failures": failures,
        "current_commit": current_commit,
        "expected_commit": args.expected_commit,
        "contract_sha256": sha256(CONTRACT),
        "authorization_sha256": (
            sha256(args.authorization) if args.authorization.is_file() else None
        ),
        "model_hashes": model_hashes,
        "workers": args.workers,
        "live_seed_scan_status": seed_scan.get("status"),
        "resume": bool(args.resume),
        "existing_completed_shards": existing_shards,
        "scientific_seeds_opened": existing_shards > 0,
    }
    if args.preflight_only:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return 0 if not failures else 1
    if failures:
        raise RuntimeError("Program Q launch preflight failed: " + "; ".join(failures))
    custody = run_dir / "custody"
    custody.mkdir(parents=True, exist_ok=bool(args.resume))
    attempt_id = archive_previous_attempt(custody) if args.resume else 0
    manifest_name = (
        f"resume_manifest_{attempt_id}.json" if args.resume else "launch_manifest.json"
    )
    write_json_atomic(custody / manifest_name, preflight)
    watcher_command = [
        sys.executable,
        str(ROOT / "scripts/watch_program_o_full_des_hpi.py"),
        "--run-dir",
        str(run_dir),
        "--interval-seconds",
        "10",
    ]
    watcher = subprocess.Popen(
        watcher_command,
        cwd=ROOT,
        stdout=(custody / "watcher.stdout.log").open("ab"),
        stderr=(custody / "watcher.stderr.log").open("ab"),
        start_new_session=True,
    )
    deadline = time.time() + 30
    while time.time() < deadline:
        state_path = custody / "watcher_state.json"
        if watcher.poll() is not None:
            raise RuntimeError("Program Q watcher exited before producer start")
        if state_path.is_file():
            state = json.loads(state_path.read_text())
            if state.get("status") == "AWAITING_PRODUCER_CONTROL":
                break
        time.sleep(0.1)
    else:
        watcher.terminate()
        raise RuntimeError("Program Q watcher did not establish prestart custody")
    producer_command = [
        sys.executable,
        str(ROOT / "scripts/run_program_q_confirmation.py"),
        "--run-dir",
        str(run_dir),
        "--models",
        str(args.models.resolve()),
        "--authorization",
        str(args.authorization.resolve()),
        "--workers",
        str(args.workers),
        "--watcher-ready-sha256",
        sha256(custody / "watcher_ready.json"),
        "--attempt-id",
        str(attempt_id),
    ]
    producer = subprocess.Popen(
        producer_command,
        cwd=ROOT,
        stdout=(custody / "producer.stdout.log").open("ab"),
        stderr=(custody / "producer.stderr.log").open("ab"),
        start_new_session=True,
    )
    control = {
        "schema_version": "program_q_producer_control_v1",
        "launched_at": now_utc(),
        "stage": "confirmation",
        "producer_pid": producer.pid,
        "producer_pgid": os.getpgid(producer.pid),
        "producer_sid": os.getsid(producer.pid),
        "watcher_pid": watcher.pid,
        "runner_command": producer_command,
        "watcher_command": watcher_command,
        "scientific_commit": current_commit,
        "contract_sha256": sha256(CONTRACT),
        "authorization_sha256": sha256(args.authorization),
        "result_relative_path": "artifacts/confirmation/evaluation/result.json",
        "attempt_id": attempt_id,
    }
    write_json_atomic(custody / "producer_control.json", control)
    print(json.dumps(control, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
