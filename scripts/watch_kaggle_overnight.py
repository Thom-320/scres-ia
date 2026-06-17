#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import time


DEFAULT_KERNELS = [
    "thomaschisica/scresia-tail-resilience-ppo-vs-static",
    "thomaschisica/scresia-track-a-history-screen",
]
TERMINAL_STATUSES = {
    "KernelWorkerStatus.COMPLETE",
    "KernelWorkerStatus.ERROR",
    "KernelWorkerStatus.CANCELED",
}


def utc_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def run(cmd: list[str], *, log_path: Path) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n$ {' '.join(cmd)}\n")
        handle.write(proc.stdout)
        handle.write(f"\n[returncode={proc.returncode}]\n")
    return proc


def parse_status(output: str) -> str:
    marker = ' has status "'
    if marker not in output:
        return "UNKNOWN"
    return output.split(marker, 1)[1].split('"', 1)[0]


def kernel_slug(ref: str) -> str:
    return ref.rsplit("/", 1)[-1].replace("_", "-")


def append_status_csv(path: Path, row: dict[str, str]) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["time_utc", "kernel", "status"])
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description="Watch Kaggle kernels overnight.")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/kaggle_overnight_watch"))
    parser.add_argument("--interval-seconds", type=int, default=600)
    parser.add_argument("--max-hours", type=float, default=12.0)
    parser.add_argument("--kernels", nargs="+", default=DEFAULT_KERNELS)
    args = parser.parse_args()

    run_dir = args.output_root / f"watch_{utc_slug()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "watch.log"
    status_csv = run_dir / "status.csv"
    terminal: dict[str, str] = {}
    downloaded: set[str] = set()

    deadline = time.time() + args.max_hours * 3600
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"started_at_utc={datetime.now(timezone.utc).isoformat()}\n")
        handle.write(f"kernels={args.kernels}\n")

    while time.time() < deadline:
        for ref in args.kernels:
            if ref in terminal:
                continue
            proc = run(["kaggle", "kernels", "status", ref], log_path=log_path)
            status = parse_status(proc.stdout)
            append_status_csv(
                status_csv,
                {
                    "time_utc": datetime.now(timezone.utc).isoformat(),
                    "kernel": ref,
                    "status": status,
                },
            )
            if status in TERMINAL_STATUSES:
                terminal[ref] = status
                if ref not in downloaded:
                    out_dir = run_dir / "outputs" / kernel_slug(ref)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    run(["kaggle", "kernels", "output", ref, "-p", str(out_dir)], log_path=log_path)
                    downloaded.add(ref)
        if len(terminal) == len(args.kernels):
            break
        time.sleep(args.interval_seconds)

    summary = run_dir / "SUMMARY.md"
    lines = [
        "# Kaggle Overnight Watch",
        "",
        f"- started/updated UTC: {datetime.now(timezone.utc).isoformat()}",
        f"- run_dir: `{run_dir}`",
        "",
        "| kernel | final_status | output_dir |",
        "|---|---:|---|",
    ]
    for ref in args.kernels:
        out_dir = run_dir / "outputs" / kernel_slug(ref)
        lines.append(f"| `{ref}` | `{terminal.get(ref, 'not_terminal')}` | `{out_dir}` |")
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
