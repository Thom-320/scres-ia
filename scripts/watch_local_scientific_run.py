#!/usr/bin/env python3
"""Independently watch a local scientific process and its progress artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _write_latest(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=float, default=5.0)
    args = parser.parse_args()
    if args.interval_seconds <= 0:
        parser.error("interval must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    latest_path = args.output_dir / "watcher_latest.json"
    log_path = args.output_dir / "watcher.jsonl"
    watcher_started = datetime.now(UTC).isoformat()

    while True:
        alive = _pid_alive(args.pid)
        result_exists = args.result.exists()
        payload = {
            "schema_version": "local_scientific_watcher_v1",
            "observed_at_utc": datetime.now(UTC).isoformat(),
            "watcher_pid": os.getpid(),
            "watcher_started_at_utc": watcher_started,
            "watcher_started_before_scientific_process": False,
            "scientific_pid": args.pid,
            "scientific_pid_alive": alive,
            "progress_path": str(args.progress.resolve()),
            "progress_sha256": _sha256(args.progress),
            "progress": _load_json(args.progress),
            "result_path": str(args.result.resolve()),
            "result_exists": result_exists,
            "result_sha256": _sha256(args.result),
            "state": "running" if alive else (
                "completed_unverified" if result_exists else "failed_or_incomplete"
            ),
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
        _write_latest(latest_path, payload)
        if not alive:
            break
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
