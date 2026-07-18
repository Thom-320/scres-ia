#!/usr/bin/env python3
"""Watcher-first detached launcher for the frozen Program S S1 run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    actual_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    if actual_commit != args.expected_commit:
        raise RuntimeError(f"commit mismatch: {actual_commit} != {args.expected_commit}")
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite run root {args.output_root}")
    args.output_root.mkdir(parents=True)
    manifest = {
        "schema_version": "program_s_s1_launch_manifest_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_commit": actual_commit,
        "expected_shards": 5_760,
        "workers": 2,
        "seed_range": [7_510_001, 7_510_012],
        "contract_sha256": sha256(ROOT / "contracts/program_s_product_mix_risk_interaction_gsa_v1.json"),
        "amendment_sha256": sha256(ROOT / "contracts/program_s_product_mix_risk_interaction_gsa_v1_1_amendment.json"),
        "design_sha256": sha256(ROOT / "research/paper2_exhaustive_search/program_s_native_morris_design_v1_1.json"),
        "preopen_audit_sha256": sha256(ROOT / "research/paper2_exhaustive_search/program_s_s1_preopen_audit_v1_2.json"),
    }
    (args.output_root / "launch_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    watcher_stdout = (args.output_root / "watcher.stdout.log").open("x")
    watcher_stderr = (args.output_root / "watcher.stderr.log").open("x")
    watcher = subprocess.Popen(
        [sys.executable, str(ROOT / "scripts/watch_program_s_s1_native.py"),
         "--output-root", str(args.output_root)],
        cwd=ROOT,
        stdout=watcher_stdout,
        stderr=watcher_stderr,
        start_new_session=True,
    )
    deadline = time.time() + 30
    while time.time() < deadline and not (args.output_root / "watcher_ready.json").exists():
        if watcher.poll() is not None:
            raise RuntimeError("Program S watcher exited before READY")
        time.sleep(0.1)
    if not (args.output_root / "watcher_ready.json").exists():
        watcher.terminate()
        raise TimeoutError("Program S watcher did not become READY")
    producer_stdout = (args.output_root / "producer.stdout.log").open("x")
    producer_stderr = (args.output_root / "producer.stderr.log").open("x")
    producer = subprocess.Popen(
        [sys.executable, str(ROOT / "scripts/run_program_s_s1_native.py"),
         "--output-root", str(args.output_root), "--workers", "2"],
        cwd=ROOT,
        stdout=producer_stdout,
        stderr=producer_stderr,
        start_new_session=True,
    )
    control = {
        "producer_pid": producer.pid,
        "producer_pgid": os.getpgid(producer.pid),
        "watcher_pid": watcher.pid,
        "watcher_pgid": os.getpgid(watcher.pid),
        "source_commit": actual_commit,
    }
    (args.output_root / "producer_control.json").write_text(
        json.dumps(control, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(control, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
