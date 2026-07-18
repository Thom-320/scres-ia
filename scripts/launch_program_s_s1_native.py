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
SCIENTIFIC_PATHS = (
    "contracts/program_s_product_mix_risk_interaction_gsa_v1.json",
    "contracts/program_s_product_mix_risk_interaction_gsa_v1_1_amendment.json",
    "research/paper2_exhaustive_search/program_s_native_morris_design_v1_1.json",
    "research/paper2_exhaustive_search/program_s_s1_preopen_audit_v1_2.json",
    "scripts/run_program_s_s1_shard.py",
    "scripts/summarize_program_s_s1_point.py",
    "supply_chain/program_s_risk_interaction.py",
    "supply_chain/program_o_full_des.py",
    "supply_chain/program_o_full_des_transducer.py",
    "supply_chain/program_o_state_rich.py",
    "supply_chain/supply_chain.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def process_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False


def next_resume_custody(output_root: Path) -> Path:
    attempts = output_root / "resume_attempts"
    for index in range(1, 1000):
        candidate = attempts / f"attempt-{index:03d}"
        if not candidate.exists():
            return candidate
    raise RuntimeError("Program S resume attempt namespace exhausted")


def assert_scientific_tree_unchanged(original_commit: str, current_commit: str) -> None:
    completed = subprocess.run(
        ["git", "diff", "--quiet", original_commit, current_commit, "--", *SCIENTIFIC_PATHS],
        cwd=ROOT,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "scientific Program S files drifted since the original launch commit"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    actual_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    if actual_commit != args.expected_commit:
        raise RuntimeError(f"commit mismatch: {actual_commit} != {args.expected_commit}")
    if args.resume:
        if not args.output_root.is_dir():
            raise FileNotFoundError("resume requires the original run root")
        original_manifest_path = args.output_root / "launch_manifest.json"
        original_manifest = json.loads(original_manifest_path.read_text())
        scientific_commit = str(original_manifest["source_commit"])
        assert_scientific_tree_unchanged(scientific_commit, actual_commit)
        controls = [args.output_root / "producer_control.json"] + sorted(
            (args.output_root / "resume_attempts").glob("attempt-*/producer_control.json")
        )
        for control_path in controls:
            if control_path.exists() and process_alive(
                json.loads(control_path.read_text()).get("producer_pid")
            ):
                raise RuntimeError(f"producer is still alive: {control_path}")
        receipts = [args.output_root / "producer_exit.json"] + sorted(
            (args.output_root / "resume_attempts").glob("attempt-*/producer_exit.json")
        )
        for receipt_path in receipts:
            if receipt_path.exists() and json.loads(receipt_path.read_text()).get("status") == "COMPLETE":
                raise RuntimeError(f"completed Program S run cannot be resumed: {receipt_path}")
        if not any((args.output_root / "matrices").glob("*.npz")):
            raise RuntimeError("resume requires preserved completed matrices")
        custody_dir = next_resume_custody(args.output_root)
        custody_dir.mkdir(parents=True)
    else:
        if args.output_root.exists():
            raise FileExistsError(f"refusing to overwrite run root {args.output_root}")
        args.output_root.mkdir(parents=True)
        custody_dir = args.output_root
        scientific_commit = actual_commit
    manifest = {
        "schema_version": "program_s_s1_launch_manifest_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_commit": scientific_commit,
        "recovery_harness_commit": actual_commit if args.resume else None,
        "resume": args.resume,
        "expected_shards": 5_760,
        "workers": 2,
        "seed_range": [7_510_001, 7_510_012],
        "contract_sha256": sha256(ROOT / "contracts/program_s_product_mix_risk_interaction_gsa_v1.json"),
        "amendment_sha256": sha256(ROOT / "contracts/program_s_product_mix_risk_interaction_gsa_v1_1_amendment.json"),
        "design_sha256": sha256(ROOT / "research/paper2_exhaustive_search/program_s_native_morris_design_v1_1.json"),
        "preopen_audit_sha256": sha256(ROOT / "research/paper2_exhaustive_search/program_s_s1_preopen_audit_v1_2.json"),
    }
    (custody_dir / "launch_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    watcher_stdout = (custody_dir / "watcher.stdout.log").open("x")
    watcher_stderr = (custody_dir / "watcher.stderr.log").open("x")
    watcher = subprocess.Popen(
        [sys.executable, str(ROOT / "scripts/watch_program_s_s1_native.py"),
         "--output-root", str(args.output_root),
         "--custody-dir", str(custody_dir)],
        cwd=ROOT,
        stdout=watcher_stdout,
        stderr=watcher_stderr,
        start_new_session=True,
    )
    deadline = time.time() + 30
    while time.time() < deadline and not (custody_dir / "watcher_ready.json").exists():
        if watcher.poll() is not None:
            raise RuntimeError("Program S watcher exited before READY")
        time.sleep(0.1)
    if not (custody_dir / "watcher_ready.json").exists():
        watcher.terminate()
        raise TimeoutError("Program S watcher did not become READY")
    producer_stdout = (custody_dir / "producer.stdout.log").open("x")
    producer_stderr = (custody_dir / "producer.stderr.log").open("x")
    producer_command = [
        sys.executable, str(ROOT / "scripts/run_program_s_s1_native.py"),
        "--output-root", str(args.output_root), "--workers", "2",
        "--custody-dir", str(custody_dir),
    ]
    if args.resume:
        producer_command.append("--resume")
    producer = subprocess.Popen(
        producer_command,
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
        "source_commit": scientific_commit,
        "recovery_harness_commit": actual_commit if args.resume else None,
        "custody_dir": str(custody_dir),
    }
    (custody_dir / "producer_control.json").write_text(
        json.dumps(control, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(control, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
