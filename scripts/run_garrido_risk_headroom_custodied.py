#!/usr/bin/env python3
"""Run the frozen Garrido risk screen inside a detached, watched POSIX session."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any


def now() -> str:
    return datetime.now(UTC).isoformat()


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    custody = run_dir / "custody"
    artifacts = run_dir / "artifacts" / "development"
    if not (custody / "watcher_ready.json").is_file():
        raise RuntimeError("watcher must be ready before producer starts")
    artifacts.mkdir(parents=True, exist_ok=True)

    command = [
        str(args.python),
        "scripts/run_garrido_risk_headroom_sensitivity.py",
        "--output",
        str(artifacts),
        "--workers",
        str(max(1, int(args.workers))),
    ]
    control = {
        "schema_version": "garrido_risk_headroom_producer_control_v1",
        "started_at_utc": now(),
        "stage": "development",
        "source_commit": args.source_commit,
        "producer_pid": os.getpid(),
        "producer_pgid": os.getpgid(0),
        "producer_sid": os.getsid(0),
        "command": command,
        "cwd": str(Path.cwd()),
    }
    write_json_atomic(custody / "producer_control.json", control)
    write_json_atomic(
        custody / "environment_manifest.json",
        {
            "captured_at_utc": now(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "python_executable": str(args.python),
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        },
    )

    started = now()
    completed = subprocess.run(command, cwd=Path.cwd(), check=False)
    result_path = artifacts / "result.json"
    exit_state = {
        "schema_version": "garrido_risk_headroom_producer_exit_v1",
        "started_at_utc": started,
        "finished_at_utc": now(),
        "returncode": int(completed.returncode),
        "result_exists": result_path.is_file(),
    }
    write_json_atomic(custody / "producer_exit.json", exit_state)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
