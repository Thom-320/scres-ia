#!/usr/bin/env python3
"""Prestart and whole-session watcher for the Program-J v2 108-cell screen."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def session_processes(session_id: int) -> list[dict[str, Any]]:
    if session_id <= 0:
        return []
    completed = subprocess.run(
        ["ps", "-eo", "pid=,ppid=,pgid=,sid=,%cpu=,rss=,stat=,command="],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    rows: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        fields = line.strip().split(None, 7)
        if len(fields) < 8:
            continue
        try:
            row = {
                "pid": int(fields[0]),
                "ppid": int(fields[1]),
                "process_group_id": int(fields[2]),
                "session_id": int(fields[3]),
                "cpu_percent": float(fields[4]),
                "rss_bytes": int(fields[5]) * 1024,
                "state": fields[6],
                "command": fields[7],
            }
        except ValueError:
            continue
        if row["session_id"] == session_id:
            rows.append(row)
    return rows


def memory_available_bytes() -> int | None:
    path = Path("/proc/meminfo")
    if not path.is_file():
        return None
    for line in path.read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return None


def snapshot(
    run_dir: Path,
    *,
    watcher_started_at_utc: str,
    peak_rss_bytes: int,
    minimum_memory_available_bytes: int | None,
) -> dict[str, Any]:
    pid_record = load_json(run_dir / "pid.json")
    progress_path = run_dir / "progress.json"
    progress = load_json(progress_path)
    result_path = Path(
        str(pid_record.get("result", run_dir / "verdict.json"))
    ) if pid_record else run_dir / "verdict.json"
    session_id = int(pid_record.get("scientific_session_id", -1)) if pid_record else -1
    members = session_processes(session_id)
    session_rss = sum(int(row["rss_bytes"]) for row in members)
    peak_rss = max(peak_rss_bytes, session_rss)
    memory_available = memory_available_bytes()
    if memory_available is None:
        minimum_memory = minimum_memory_available_bytes
    elif minimum_memory_available_bytes is None:
        minimum_memory = memory_available
    else:
        minimum_memory = min(memory_available, minimum_memory_available_bytes)
    progress_age = (
        max(0.0, time.time() - progress_path.stat().st_mtime)
        if progress_path.is_file()
        else None
    )
    disk = shutil.disk_usage(run_dir)
    result_hash = file_sha256(result_path)
    payload: dict[str, Any] = {
        "schema_version": "program_j_request_snapshot_v2_108cell_watcher_v1",
        "hostname": socket.gethostname(),
        "watcher_pid": os.getpid(),
        "watcher_started_at_utc": watcher_started_at_utc,
        "observed_at_utc": utc_now(),
        "watcher_scope": "ENTIRE_SCIENTIFIC_SESSION_INCLUDING_REPARENTED_WORKERS",
        "scientific_pid": pid_record.get("scientific_pid") if pid_record else None,
        "scientific_process_group_id": (
            pid_record.get("scientific_process_group_id") if pid_record else None
        ),
        "scientific_session_id": session_id if session_id > 0 else None,
        "scientific_session_alive": bool(members),
        "scientific_session_process_count": len(members),
        "scientific_session_processes": members,
        "scientific_session_cpu_percent": sum(
            float(row["cpu_percent"]) for row in members
        ),
        "scientific_session_rss_bytes": session_rss,
        "peak_scientific_session_rss_bytes": peak_rss,
        "memory_available_bytes": memory_available,
        "minimum_memory_available_bytes": minimum_memory,
        "progress": progress,
        "progress_sha256": file_sha256(progress_path),
        "progress_age_seconds": progress_age,
        "result": str(result_path),
        "result_exists": result_path.is_file() and result_path.stat().st_size > 0,
        "result_sha256": result_hash,
        "stdout_bytes": (run_dir / "stdout.log").stat().st_size
        if (run_dir / "stdout.log").is_file()
        else 0,
        "stderr_bytes": (run_dir / "stderr.log").stat().st_size
        if (run_dir / "stderr.log").is_file()
        else 0,
        "disk_free_bytes": disk.free,
        "load_average": list(os.getloadavg()),
    }
    if pid_record is None:
        payload["state"] = "watching_prestart"
    elif members and progress is None:
        payload["state"] = "running_alive_awaiting_first_progress"
    elif members:
        payload["state"] = (
            "running_fresh"
            if progress_age is not None and progress_age <= 600.0
            else "running_progress_stale"
        )
    else:
        complete = bool(
            progress
            and progress.get("state") == "complete"
            and progress.get("verdict_sha256") == result_hash
            and result_hash is not None
        )
        payload["state"] = "completed_verified" if complete else "failed_or_incomplete"
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=float, default=10.0)
    parser.add_argument("--prestart-timeout-seconds", type=float, default=60.0)
    args = parser.parse_args()
    if args.interval_seconds <= 0 or args.prestart_timeout_seconds <= 0:
        parser.error("watcher intervals must be positive")
    run_dir = args.run_dir.resolve(strict=True)
    started_at_utc = utc_now()
    started_monotonic = time.monotonic()
    peak_rss = 0
    minimum_memory: int | None = None
    while True:
        payload = snapshot(
            run_dir,
            watcher_started_at_utc=started_at_utc,
            peak_rss_bytes=peak_rss,
            minimum_memory_available_bytes=minimum_memory,
        )
        peak_rss = int(payload["peak_scientific_session_rss_bytes"])
        minimum_memory = payload["minimum_memory_available_bytes"]
        if (
            payload["state"] == "watching_prestart"
            and time.monotonic() - started_monotonic > args.prestart_timeout_seconds
        ):
            payload["state"] = "failed_pid_not_received"
        atomic_json(run_dir / "watcher_latest.json", payload)
        append_jsonl(run_dir / "watcher.jsonl", payload)
        if payload["state"] in {
            "completed_verified",
            "failed_or_incomplete",
            "failed_pid_not_received",
        }:
            return 0 if payload["state"] == "completed_verified" else 1
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
