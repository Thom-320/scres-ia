#!/usr/bin/env python3
"""Session-wide watcher for future custodied wartime GSA execution."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _process_table() -> list[dict[str, int]]:
    session_field = "sess=" if sys.platform == "darwin" else "sid="
    completed = subprocess.run(
        ["ps", "-axo", f"pid=,pgid=,{session_field}"],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = []
    for line in completed.stdout.splitlines():
        values = line.split()
        if len(values) == 3:
            rows.append({"pid": int(values[0]), "pgid": int(values[1]), "sid": int(values[2])})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument("--producer-timeout-seconds", type=float, default=60.0)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    custody = run_dir / "custody"
    custody.mkdir(parents=True, exist_ok=True)
    ready = custody / "watcher_ready.json"
    control_path = custody / "producer_control.json"
    _write_json_atomic(
        ready,
        {
            "schema_version": "war_stress_gsa_watcher_ready_v1",
            "ready_at_utc": _now(),
            "watcher_pid": os.getpid(),
            "watcher_pgid": os.getpgid(0),
            "watcher_sid": os.getsid(0),
        },
    )

    deadline = time.monotonic() + float(args.producer_timeout_seconds)
    while not control_path.is_file() and time.monotonic() < deadline:
        _write_json_atomic(
            custody / "heartbeat.json",
            {"at_utc": _now(), "state": "AWAITING_PRODUCER_CONTROL"},
        )
        time.sleep(min(float(args.poll_seconds), 1.0))
    if not control_path.is_file():
        _write_json_atomic(
            custody / "watcher_terminal.json",
            {"at_utc": _now(), "status": "WATCHER_TIMEOUT_NO_PRODUCER"},
        )
        return 2

    control = json.loads(control_path.read_text())
    producer_pid = int(control["pid"])
    producer_pgid = int(control["pgid"])
    producer_sid = int(control["sid"])
    while True:
        table = _process_table()
        members = [
            row
            for row in table
            if row["pgid"] == producer_pgid or row["sid"] == producer_sid
        ]
        _write_json_atomic(
            custody / "heartbeat.json",
            {
                "at_utc": _now(),
                "state": "RUNNING" if members else "DRAINED",
                "producer_pid": producer_pid,
                "producer_pgid": producer_pgid,
                "producer_sid": producer_sid,
                "session_members": members,
            },
        )
        if not members:
            break
        time.sleep(max(0.1, float(args.poll_seconds)))

    exit_path = custody / "producer_exit.json"
    _write_json_atomic(
        custody / "watcher_terminal.json",
        {
            "at_utc": _now(),
            "status": (
                "TERMINAL_SESSION_EMPTY_WITH_EXIT_RECEIPT"
                if exit_path.is_file()
                else "TERMINAL_SESSION_EMPTY_MISSING_EXIT_RECEIPT"
            ),
            "producer_exit_exists": exit_path.is_file(),
            "session_empty": True,
        },
    )
    return 0 if exit_path.is_file() else 3


if __name__ == "__main__":
    raise SystemExit(main())
