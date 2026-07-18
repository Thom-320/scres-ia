#!/usr/bin/env python3
"""Independent watcher for the Program S S1 producer process group."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def process_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def available_memory_kib() -> int | None:
    path = Path("/proc/meminfo")
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1])
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=15.0)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    ready = args.output_root / "watcher_ready.json"
    state = args.output_root / "watcher_state.json"
    log = args.output_root / "watcher_state.jsonl"
    if ready.exists() or state.exists() or log.exists():
        raise FileExistsError("refusing to overwrite Program S watcher identity")
    atomic_json(ready, {"status": "READY", "watcher_pid": os.getpid(), "time": utc_now()})
    while True:
        control_path = args.output_root / "producer_control.json"
        control = json.loads(control_path.read_text()) if control_path.exists() else {}
        pid = control.get("producer_pid")
        alive = process_alive(pid)
        shards = len(list((args.output_root / "matrices").glob("*.npz")))
        stderr = args.output_root / "producer.stderr.log"
        exit_path = args.output_root / "producer_exit.json"
        payload = {
            "time": utc_now(),
            "watcher_pid": os.getpid(),
            "producer_pid": pid,
            "producer_pgid": control.get("producer_pgid"),
            "producer_alive": alive,
            "shards": shards,
            "expected_shards": 5_760,
            "stderr_bytes": stderr.stat().st_size if stderr.exists() else 0,
            "mem_available_kib": available_memory_kib(),
            "terminal_receipt_present": exit_path.exists(),
        }
        atomic_json(state, payload)
        with log.open("a") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if exit_path.exists() and not alive:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
