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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.evaluate_program_q_replication import (  # noqa: E402
    CONTRACT,
    verify_authorization,
    verify_model_hashes,
)
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    failures: list[str] = []
    current_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    if run_dir.exists():
        failures.append("run identity already exists")
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
    try:
        model_hashes = verify_model_hashes(args.models)
    except (OSError, RuntimeError) as error:
        failures.append(str(error))
        model_hashes = {}
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
        "scientific_seeds_opened": False,
    }
    if args.preflight_only:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return 0 if not failures else 1
    if failures:
        raise RuntimeError("Program Q launch preflight failed: " + "; ".join(failures))
    custody = run_dir / "custody"
    custody.mkdir(parents=True)
    write_json_atomic(custody / "launch_manifest.json", preflight)
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
        "--bootstrap",
        str(args.bootstrap),
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
    }
    write_json_atomic(custody / "producer_control.json", control)
    print(json.dumps(control, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
