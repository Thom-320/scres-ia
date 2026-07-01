#!/usr/bin/env python3
"""Poll one Kaggle kernel and fetch output on terminal status.

This is intentionally boring and inspectable: a single loop, timestamped log,
direct `kaggle kernels status`, and one fetch when the kernel completes or
errors.  Use it for long confirmatory runs so stale watcher state cannot be
mistaken for a live Kaggle job.
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
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as fh:
        fh.write(f"[{stamp()}] {text}\n")
        fh.flush()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slug", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--interval-sec", type=int, default=180)
    parser.add_argument("--max-polls", type=int, default=0, help="0 means forever")
    args = parser.parse_args()

    slug = args.slug
    out_dir = Path(args.output_dir)
    log = Path(args.log)
    append(log, f"watcher start slug={slug} output={out_dir}")

    polls = 0
    while True:
        polls += 1
        status = run_text(["kaggle", "kernels", "status", slug])
        append(log, f"STATUS {status}")
        if any(marker in status for marker in TERMINAL_MARKERS):
            out_dir.mkdir(parents=True, exist_ok=True)
            fetch = run_text(["kaggle", "kernels", "output", slug, "-p", str(out_dir)])
            append(log, f"FETCH {fetch}")
            append(log, "watcher done")
            return 0
        if args.max_polls and polls >= args.max_polls:
            append(log, "watcher max_polls reached without terminal status")
            return 2
        time.sleep(args.interval_sec)


if __name__ == "__main__":
    raise SystemExit(main())
