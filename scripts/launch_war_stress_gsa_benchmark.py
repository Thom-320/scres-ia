#!/usr/bin/env python3
"""Launch the non-scientific GSA/DES benchmark with watcher-first custody."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config-id", default="morris::LOC_SURGE::coincident::00::00")
    parser.add_argument("--max-daily-steps", type=int, default=14)
    parser.add_argument("--projected-workers", type=int, default=32)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    watcher = subprocess.Popen(
        [
            sys.executable,
            "scripts/watch_war_stress_gsa_session.py",
            "--run-dir",
            str(run_dir),
            "--poll-seconds",
            "0.2",
        ]
    )
    ready = run_dir / "custody" / "watcher_ready.json"
    deadline = time.monotonic() + 20.0
    while not ready.is_file() and time.monotonic() < deadline:
        time.sleep(0.05)
    if not ready.is_file():
        watcher.terminate()
        raise RuntimeError("watcher did not become ready before producer launch")

    producer = subprocess.Popen(
        [
            sys.executable,
            "scripts/run_war_stress_gsa_executor.py",
            "--mode",
            "benchmark",
            "--run-dir",
            str(run_dir),
            "--config-id",
            args.config_id,
            "--max-daily-steps",
            str(args.max_daily_steps),
            "--projected-workers",
            str(args.projected_workers),
        ],
        start_new_session=True,
    )
    producer_code = producer.wait()
    watcher_code = watcher.wait(timeout=30.0)
    summary = {
        "run_dir": str(run_dir),
        "producer_returncode": producer_code,
        "watcher_returncode": watcher_code,
        "watcher_ready_before_producer": True,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return producer_code or watcher_code


if __name__ == "__main__":
    raise SystemExit(main())
