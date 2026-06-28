#!/usr/bin/env python3
"""Poll Kaggle kernels and download outputs when they finish."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import time


TERMINAL = (
    "KernelWorkerStatus.COMPLETE",
    "KernelWorkerStatus.ERROR",
    "KernelWorkerStatus.CANCELLED",
)


def run(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return proc.stdout.strip()


def slug(kernel: str) -> str:
    return kernel.split("/", 1)[-1].replace("/", "_")


def watch_kernel(kernel: str, output_root: Path, interval_s: int) -> None:
    name = slug(kernel)
    log = output_root / "watchers" / f"{name}.watch.log"
    out_dir = output_root / f"{name}_auto"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(log.read_text() if log.exists() else "")
    with log.open("a") as fh:
        fh.write(f"[{datetime.now().isoformat()}] watcher started for {kernel}\n")
        fh.flush()
        while True:
            status = run(["kaggle", "kernels", "status", kernel])
            fh.write(f"[{datetime.now().isoformat()}] {status}\n")
            fh.flush()
            if any(term in status for term in TERMINAL):
                out_dir.mkdir(parents=True, exist_ok=True)
                output = run(["kaggle", "kernels", "output", kernel, "-p", str(out_dir)])
                fh.write(output + "\n")
                fh.write(f"[{datetime.now().isoformat()}] downloaded to {out_dir}\n")
                fh.flush()
                return
            time.sleep(interval_s)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("kernels", nargs="+")
    parser.add_argument("--output-root", default="outputs/kaggle")
    parser.add_argument("--interval-s", type=int, default=300)
    args = parser.parse_args()
    for kernel in args.kernels:
        watch_kernel(kernel, Path(args.output_root), args.interval_s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
