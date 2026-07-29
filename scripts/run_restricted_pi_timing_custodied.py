#!/usr/bin/env python3
"""Custody wrapper for the restricted timing-ceiling executor."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any


def now() -> str:
    return datetime.now(UTC).isoformat()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--risk-result", type=Path, required=True)
    parser.add_argument("--risk-raw-rows", type=Path, required=True)
    parser.add_argument("--seed-claim", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    custody = run_dir / "custody"
    artifacts = run_dir / "artifacts" / "development"
    if not (custody / "watcher_ready.json").is_file():
        raise RuntimeError("watcher must be ready before producer starts")
    if not args.risk_result.is_file() or not args.risk_raw_rows.is_file():
        raise RuntimeError("complete retrieved risk artifacts are required")
    claim = json.loads(args.seed_claim.read_text())
    if claim.get("status") != "CLAIMED_FOR_RESTRICTED_TIMING_DEVELOPMENT":
        raise RuntimeError("exclusive timing seed claim is not active")
    if claim.get("seed_start") != 7460001 or claim.get("seed_end") != 7460048:
        raise RuntimeError("timing seed claim does not match the frozen block")
    artifacts.mkdir(parents=True, exist_ok=True)

    command = [
        str(args.python),
        "scripts/run_restricted_pi_timing_ceiling.py",
        "--risk-result", str(args.risk_result.resolve()),
        "--risk-raw-rows", str(args.risk_raw_rows.resolve()),
        "--output", str(artifacts),
        "--workers", str(max(1, int(args.workers))),
    ]
    control = {
        "schema_version": "restricted_pi_timing_producer_control_v1",
        "started_at_utc": now(),
        "stage": "development",
        "source_commit": args.source_commit,
        "producer_pid": os.getpid(),
        "producer_pgid": os.getpgid(0),
        "producer_sid": os.getsid(0),
        "command": command,
        "cwd": str(Path.cwd()),
        "risk_result_sha256": sha256(args.risk_result),
        "risk_raw_rows_sha256": sha256(args.risk_raw_rows),
        "seed_claim_sha256": sha256(args.seed_claim),
    }
    write_json_atomic(custody / "producer_control.json", control)
    write_json_atomic(custody / "seed_claim.json", claim)
    write_json_atomic(custody / "environment_manifest.json", {
        "captured_at_utc": now(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "python_executable": str(args.python),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
    })
    started = now()
    completed = subprocess.run(command, cwd=Path.cwd(), check=False)
    write_json_atomic(custody / "producer_exit.json", {
        "schema_version": "restricted_pi_timing_producer_exit_v1",
        "started_at_utc": started,
        "finished_at_utc": now(),
        "returncode": int(completed.returncode),
        "result_exists": (artifacts / "result.json").is_file(),
    })
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
