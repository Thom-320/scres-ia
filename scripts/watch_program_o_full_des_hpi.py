#!/usr/bin/env python3
"""Whole-session watcher for a detached Program O full-DES H_PI run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time
from typing import Any


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def append_journal(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def process_group_alive(pgid: int | None) -> bool:
    if not pgid or pgid <= 0:
        return False
    try:
        os.killpg(int(pgid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def process_table() -> list[dict[str, Any]]:
    has_sid = platform.system() != "Darwin"
    fields = (
        "pid=,ppid=,pgid=,sid=,stat=,command="
        if has_sid
        else "pid=,ppid=,pgid=,stat=,command="
    )
    command = ["ps", "-eo", fields]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        return []
    members = []
    for line in completed.stdout.splitlines():
        parts = line.strip().split(None, 5 if has_sid else 4)
        minimum = 5 if has_sid else 4
        if len(parts) < minimum:
            continue
        try:
            pid, ppid, row_pgid = map(int, parts[:3])
            sid = int(parts[3]) if has_sid else None
        except ValueError:
            continue
        members.append(
            {
                "pid": pid,
                "ppid": ppid,
                "pgid": row_pgid,
                "sid": sid,
                "stat": parts[4] if has_sid else parts[3],
                "command": (
                    parts[5]
                    if has_sid and len(parts) > 5
                    else parts[4] if not has_sid and len(parts) > 4 else ""
                ),
            }
        )
    return members


def process_group_members(pgid: int | None) -> list[dict[str, Any]]:
    if not pgid:
        return []
    return [row for row in process_table() if row["pgid"] == int(pgid)]


def process_session_members(
    sid: int | None, *, fallback_pgid: int | None
) -> list[dict[str, Any]]:
    rows = process_table()
    if sid and platform.system() != "Darwin":
        return [row for row in rows if row["sid"] == int(sid)]
    return [row for row in rows if fallback_pgid and row["pgid"] == int(fallback_pgid)]


def snapshot(run_dir: Path, *, stale_seconds: float) -> dict[str, Any]:
    custody = run_dir / "custody"
    control = load_json(custody / "producer_control.json")
    exit_state = load_json(custody / "producer_exit.json")
    stage = str(control.get("stage")) if control else None
    progress_path = run_dir / "artifacts" / stage / "progress.json" if stage else None
    progress = load_json(progress_path) if progress_path else None
    pgid = int(control["producer_pgid"]) if control else None
    sid = int(control["producer_sid"]) if control else None
    group_alive = process_group_alive(pgid)
    group_members = process_group_members(pgid)
    session_members = process_session_members(sid, fallback_pgid=pgid)
    custody_scope_alive = bool(session_members) or group_alive
    progress_age = None
    if progress_path and progress_path.is_file():
        progress_age = max(0.0, time.time() - progress_path.stat().st_mtime)
    if control is None:
        status = "AWAITING_PRODUCER_CONTROL"
    elif (
        custody_scope_alive
        and progress_age is not None
        and progress_age > float(stale_seconds)
    ):
        status = "RUNNING_STALE_PROGRESS"
    elif custody_scope_alive:
        status = "RUNNING"
    elif exit_state is None:
        status = "INVALID_GROUP_EMPTY_WITHOUT_EXIT_MANIFEST"
    elif int(exit_state.get("returncode", 1)) == 0:
        status = "COMPLETE_PENDING_RETRIEVAL"
    else:
        status = "FAILED_PRODUCER_EXIT"
    result_path = run_dir / "artifacts" / str(stage) / "result.json" if stage else None
    state = {
        "observed_at_utc": now_utc(),
        "status": status,
        "run_dir": str(run_dir),
        "producer_pid": control.get("producer_pid") if control else None,
        "producer_pgid": pgid,
        "producer_sid": sid,
        "group_alive": group_alive,
        "group_member_count": len(group_members),
        "group_members": group_members,
        "session_member_count": len(session_members),
        "session_members": session_members,
        "custody_scope_alive": custody_scope_alive,
        "progress_age_seconds": progress_age,
        "progress": progress,
        "producer_exit": exit_state,
        "result_exists": bool(result_path and result_path.is_file()),
        "result_sha256": (
            sha256(result_path) if result_path and result_path.is_file() else None
        ),
    }
    return state


def write_remote_checksums(run_dir: Path) -> Path:
    destination = run_dir / "custody" / "remote_files.sha256"
    excluded = {
        destination,
        run_dir / "custody" / "watcher_state.json",
        run_dir / "custody" / "watcher_state.jsonl",
        run_dir / "custody" / "watcher.stdout.log",
        run_dir / "custody" / "watcher.stderr.log",
    }
    lines = []
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path not in excluded and not path.name.endswith(".tmp"):
            lines.append(f"{sha256(path)}  {path.relative_to(run_dir)}")
    destination.write_text("\n".join(lines) + "\n")
    return destination


def monitor(
    run_dir: Path,
    *,
    interval_seconds: float,
    stale_seconds: float,
    once: bool,
) -> int:
    custody = run_dir / "custody"
    custody.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        custody / "watcher_ready.json",
        {
            "ready_at_utc": now_utc(),
            "watcher_pid": os.getpid(),
            "run_dir": str(run_dir),
        },
    )
    terminal = {
        "COMPLETE_PENDING_RETRIEVAL",
        "FAILED_PRODUCER_EXIT",
        "INVALID_GROUP_EMPTY_WITHOUT_EXIT_MANIFEST",
    }
    while True:
        state = snapshot(run_dir, stale_seconds=stale_seconds)
        write_json_atomic(custody / "watcher_state.json", state)
        append_journal(custody / "watcher_state.jsonl", state)
        if state["status"] in terminal:
            checksums = write_remote_checksums(run_dir)
            final = {
                **state,
                "remote_checksums": str(checksums),
                "remote_checksums_sha256": sha256(checksums),
                "watcher_terminal_at_utc": now_utc(),
            }
            write_json_atomic(custody / "watcher_state.json", final)
            append_journal(custody / "watcher_state.jsonl", final)
            return 0 if state["status"] == "COMPLETE_PENDING_RETRIEVAL" else 1
        if once:
            return 0
        time.sleep(float(interval_seconds))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=float, default=10.0)
    parser.add_argument("--stale-seconds", type=float, default=600.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    return monitor(
        args.run_dir.resolve(),
        interval_seconds=float(args.interval_seconds),
        stale_seconds=float(args.stale_seconds),
        once=bool(args.once),
    )


if __name__ == "__main__":
    raise SystemExit(main())
