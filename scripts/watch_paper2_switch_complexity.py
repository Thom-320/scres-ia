#!/usr/bin/env python3
"""Detached watcher for the Paper-2 switch-complexity calibration gate."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
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
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
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


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def process_command(pid: int) -> str | None:
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "command="],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.stdout.strip() or None


def process_tree(root_pid: int) -> list[dict[str, Any]]:
    if root_pid <= 0:
        return []
    result = subprocess.run(
        ["ps", "-axo", "pid=,ppid=,%cpu=,rss=,command="],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    rows: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        fields = line.strip().split(None, 4)
        if len(fields) < 5:
            continue
        try:
            rows.append({
                "pid": int(fields[0]),
                "ppid": int(fields[1]),
                "cpu_percent": float(fields[2]),
                "rss_bytes": int(fields[3]) * 1024,
                "command": fields[4],
            })
        except ValueError:
            continue
    selected = {root_pid}
    while True:
        descendants = {
            int(row["pid"]) for row in rows if int(row["ppid"]) in selected
        }
        updated = selected | descendants
        if updated == selected:
            break
        selected = updated
    return [row for row in rows if int(row["pid"]) in selected]


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def snapshot(run_dir: Path, *, watcher_started: str) -> dict[str, Any]:
    pid_payload = load_json(run_dir / "pid.json")
    progress_path = run_dir / "progress.json"
    progress = load_json(progress_path)
    output_path = Path(
        str(pid_payload.get("output", run_dir / "result.json"))
    ) if pid_payload else run_dir / "result.json"
    pid = int(pid_payload.get("scientific_pid", -1)) if pid_payload else -1
    alive = pid_alive(pid)
    tree = process_tree(pid) if alive else []
    progress_age = (
        max(0.0, time.time() - progress_path.stat().st_mtime)
        if progress_path.exists()
        else None
    )
    usage = shutil.disk_usage(run_dir)
    payload: dict[str, Any] = {
        "schema_version": "paper2_switch_complexity_watcher_v1",
        "watcher_pid": os.getpid(),
        "watcher_started_at_utc": watcher_started,
        "observed_at_utc": utc_now(),
        "scientific_pid": pid if pid > 0 else None,
        "scientific_pid_alive": alive,
        "scientific_command": process_command(pid) if alive else None,
        "scientific_process_tree": tree,
        "scientific_process_tree_cpu_percent": sum(
            float(row["cpu_percent"]) for row in tree
        ),
        "scientific_process_tree_rss_bytes": sum(
            int(row["rss_bytes"]) for row in tree
        ),
        "progress": progress,
        "progress_sha256": file_sha256(progress_path),
        "progress_age_seconds": progress_age,
        "output": str(output_path),
        "result_exists": output_path.is_file() and output_path.stat().st_size > 0,
        "result_sha256": file_sha256(output_path),
        "stdout_bytes": (run_dir / "stdout.log").stat().st_size
        if (run_dir / "stdout.log").exists() else 0,
        "stderr_bytes": (run_dir / "stderr.log").stat().st_size
        if (run_dir / "stderr.log").exists() else 0,
        "disk_free_bytes": usage.free,
        "load_average": list(os.getloadavg()),
    }
    if pid_payload is None:
        payload["state"] = "watching_prestart"
    elif alive and progress is None:
        payload["state"] = "running_alive_awaiting_first_progress"
    elif alive:
        payload["state"] = (
            "running_fresh"
            if progress_age is not None and progress_age <= 600.0
            else "running_progress_stale"
        )
    else:
        complete = bool(
            progress
            and progress.get("stage") == "complete"
            and progress.get("output_sha256") == payload["result_sha256"]
        )
        payload["state"] = "completed_unverified" if complete else "failed_or_incomplete"
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=float, default=10.0)
    parser.add_argument("--prestart-timeout-seconds", type=float, default=60.0)
    args = parser.parse_args()
    if args.interval_seconds <= 0 or args.prestart_timeout_seconds <= 0:
        parser.error("watcher intervals must be positive")
    run_dir = args.run_dir.resolve()
    started_monotonic = time.monotonic()
    started_utc = utc_now()
    while True:
        payload = snapshot(run_dir, watcher_started=started_utc)
        if (
            payload["state"] == "watching_prestart"
            and time.monotonic() - started_monotonic > args.prestart_timeout_seconds
        ):
            payload["state"] = "failed_pid_not_received"
        atomic_json(run_dir / "watcher_latest.json", payload)
        append_jsonl(run_dir / "watcher.jsonl", payload)
        if payload["state"] in {
            "completed_unverified", "failed_or_incomplete", "failed_pid_not_received"
        }:
            return 0 if payload["state"] == "completed_unverified" else 1
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
