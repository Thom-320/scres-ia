#!/usr/bin/env python3
"""Poll two Kaggle kernels and fetch outputs on terminal status.

This intentionally avoids clever shell state so a watcher process can be
checked with ps and a plain log file.
"""

from __future__ import annotations

import argparse
import subprocess
import time
from datetime import datetime
from pathlib import Path


TERMINAL_MARKERS = ("COMPLETE", "ERROR", "CANCELLED", "FAILED")


def stamp() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S%z")


def run_text(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    text = (proc.stdout or "") + (proc.stderr or "")
    return text.strip()


def append(log: Path, text: str) -> None:
    with log.open("a", encoding="utf-8") as fh:
        fh.write(f"[{stamp()}] {text}\n")
        fh.flush()


def fetch(slug: str, out_dir: Path, log: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    text = run_text(["kaggle", "kernels", "output", slug, "-p", str(out_dir)])
    append(log, f"FETCH {slug} -> {out_dir}: {text}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--track-a-slug", required=True)
    parser.add_argument("--track-b-slug", required=True)
    parser.add_argument("--track-a-output", required=True)
    parser.add_argument("--track-b-output", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--interval-sec", type=int, default=180)
    args = parser.parse_args()

    log = Path(args.log)
    log.parent.mkdir(parents=True, exist_ok=True)
    done = {"A": False, "B": False}
    append(log, "watcher start")

    while not all(done.values()):
        for key, slug, out_dir in (
            ("A", args.track_a_slug, Path(args.track_a_output)),
            ("B", args.track_b_slug, Path(args.track_b_output)),
        ):
            if done[key]:
                continue
            status = run_text(["kaggle", "kernels", "status", slug])
            append(log, f"TRACK_{key} {status}")
            if any(marker in status for marker in TERMINAL_MARKERS):
                fetch(slug, out_dir, log)
                append(log, f"TRACK_{key} terminal")
                done[key] = True
        if not all(done.values()):
            time.sleep(args.interval_sec)

    append(log, "watcher done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
