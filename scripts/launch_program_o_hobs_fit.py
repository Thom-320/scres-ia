#!/usr/bin/env python3
"""Launch Program O H_obs fit with watcher-first whole-session custody."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.program_o_full_des_guard import create_seed_claim  # noqa: E402
from scripts.program_o_hobs_guard import verify_hobs_fit_freeze  # noqa: E402


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--execution-freeze", type=Path, required=True)
    parser.add_argument("--seed-claim-root", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--watch-interval-seconds", type=float, default=10.0)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    run_dir = Path(os.path.abspath(str(args.run_dir)))
    contract = args.contract.resolve()
    freeze = args.execution_freeze.resolve()
    seed_range = list(map(int, json.loads(contract.read_text())["tape_blocks"]["fit"]["range"]))
    current_commit = git_commit()
    failures = []
    if current_commit != str(args.expected_commit):
        failures.append("HEAD does not equal expected immutable commit")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        failures.append("worktree is dirty")
    if run_dir.exists():
        failures.append("run identity already exists")
    authorization = None
    if not failures:
        try:
            authorization = verify_hobs_fit_freeze(
                freeze_path=freeze,
                contract_path=contract,
                run_id=str(args.run_id),
                run_dir=run_dir,
                seed_range=seed_range,
                expected_commit=str(args.expected_commit),
            )
        except RuntimeError as exc:
            failures.append(str(exc))
    preflight = {
        "schema_version": "program_o_hobs_fit_launch_manifest_v1",
        "checked_at_utc": now_utc(),
        "passed": not failures,
        "failures": failures,
        "stage": "fit",
        "run_id": str(args.run_id),
        "run_dir": str(run_dir),
        "workers": int(args.workers),
        "expected_commit": str(args.expected_commit),
        "current_commit": current_commit,
        "contract": str(contract),
        "contract_sha256": sha256(contract),
        "execution_freeze": str(freeze),
        "seed_range": seed_range,
        "authorization": authorization,
    }
    if args.preflight_only:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return 0 if not failures else 1
    if failures:
        raise RuntimeError("launch preflight failed: " + "; ".join(failures))

    custody = run_dir / "custody"
    custody.mkdir(parents=True)
    write_json_atomic(custody / "launch_manifest.json", preflight)
    watcher_command = [
        sys.executable,
        str(ROOT / "scripts/watch_program_o_full_des_hpi.py"),
        "--run-dir",
        str(run_dir),
        "--interval-seconds",
        str(args.watch_interval_seconds),
    ]
    with (custody / "watcher.stdout.log").open("ab") as stdout, (
        custody / "watcher.stderr.log"
    ).open("ab") as stderr:
        watcher = subprocess.Popen(
            watcher_command,
            cwd=ROOT,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
    deadline = time.time() + 30.0
    while time.time() < deadline:
        state_path = custody / "watcher_state.json"
        if watcher.poll() is not None:
            raise RuntimeError("watcher exited before producer launch")
        if state_path.is_file():
            try:
                state = json.loads(state_path.read_text())
            except json.JSONDecodeError:
                state = {}
            if state.get("status") == "AWAITING_PRODUCER_CONTROL":
                break
        time.sleep(0.1)
    else:
        watcher.terminate()
        raise TimeoutError("watcher did not establish prestart custody")

    if authorization is None:
        raise AssertionError("authorization missing after preflight")
    try:
        seed_claim = create_seed_claim(
            claim_root=args.seed_claim_root.resolve(),
            authorization=authorization,
            contract_sha256=sha256(contract),
        )
    except BaseException:
        watcher.terminate()
        watcher.wait(timeout=10)
        raise
    shutil.copyfile(seed_claim, custody / "seed_claim.json")
    write_json_atomic(
        custody / "seed_claim_reference.json",
        {"path": str(seed_claim), "sha256": sha256(seed_claim)},
    )
    runner_command = [
        sys.executable,
        str(ROOT / "scripts/run_program_o_hobs_fit_custodied.py"),
        "--run-dir",
        str(run_dir),
        "--run-id",
        str(args.run_id),
        "--workers",
        str(args.workers),
        "--contract",
        str(contract),
        "--execution-freeze",
        str(freeze),
        "--seed-claim",
        str(seed_claim),
    ]
    with (custody / "producer.stdout.log").open("ab") as stdout, (
        custody / "producer.stderr.log"
    ).open("ab") as stderr:
        producer = subprocess.Popen(
            runner_command,
            cwd=ROOT,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
    control = {
        "schema_version": "program_o_hobs_fit_producer_control_v1",
        "launched_at_utc": now_utc(),
        "stage": "fit",
        "run_id": str(args.run_id),
        "producer_pid": producer.pid,
        "producer_pgid": os.getpgid(producer.pid),
        "producer_sid": os.getsid(producer.pid),
        "watcher_pid": watcher.pid,
        "watcher_pgid": os.getpgid(watcher.pid),
        "runner_command": runner_command,
        "watcher_command": watcher_command,
        "scientific_commit": current_commit,
        "contract_sha256": sha256(contract),
        "execution_freeze_sha256": sha256(freeze),
        "seed_claim": str(seed_claim),
        "seed_claim_sha256": sha256(seed_claim),
    }
    if not (
        int(control["producer_pid"])
        == int(control["producer_pgid"])
        == int(control["producer_sid"])
    ):
        producer.terminate()
        raise RuntimeError("producer is not isolated as PID=PGID=SID")
    write_json_atomic(custody / "producer_control.json", control)
    print(json.dumps(control, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
