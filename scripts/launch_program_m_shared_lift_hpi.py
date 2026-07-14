#!/usr/bin/env python3
"""Launch Program M's frozen H_PI screen after whole-session pre-attestation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parent.parent
RUN_ROOT = ROOT / "outputs/program_m_shared_lift_hpi_vps_runs"
PRODUCER = ROOT / "scripts/screen_program_m_shared_lift_hpi.py"
WATCHER = ROOT / "scripts/watch_program_m_shared_lift_hpi.py"
CONTRACT = ROOT / "contracts/program_m_shared_lift_reservation_v1.json"
SCREEN_SEED_START = 7_300_001
SCREEN_SEED_END = 7_300_024
MAX_WORKERS = 6
SOURCE_PATHS = (
    "contracts/program_m_shared_lift_reservation_v1.json",
    "scripts/launch_program_m_shared_lift_hpi.py",
    "scripts/screen_program_m_shared_lift_hpi.py",
    "scripts/watch_program_m_shared_lift_hpi.py",
    "supply_chain/program_m_shared_lift.py",
    "supply_chain/episode_metrics.py",
    "supply_chain/ret_thesis.py",
    "supply_chain/supply_chain.py",
    "supply_chain/data/garrido_proxy_v1_freeze_2026-07-10.json",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exclusive_json(path: Path, payload: dict[str, Any]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def checked_output(command: Sequence[str]) -> str:
    return subprocess.run(
        list(command),
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def screen_seed_manifest() -> dict[str, Any]:
    """Describe, but never materialize or read, the producer-owned seed block."""

    return {
        "schema_version": "program_m_shared_lift_hpi_seed_manifest_v1",
        "seed_owner": "scripts/screen_program_m_shared_lift_hpi.py:SCREEN_SEEDS",
        "seed_start": SCREEN_SEED_START,
        "seed_end": SCREEN_SEED_END,
        "count": SCREEN_SEED_END - SCREEN_SEED_START + 1,
        "use": "BURNED_DEVELOPMENT_SCREEN_ONLY",
        "virgin": False,
        "locked": False,
        "launcher_override_permitted": False,
        "common_random_numbers": (
            "The producer reuses each frozen seed across all 19 cells and all "
            "256 calendars."
        ),
    }


def scientific_command(*, workers: int, scientific_dir: Path) -> list[str]:
    if not 1 <= int(workers) <= MAX_WORKERS:
        raise ValueError(f"workers must be between 1 and {MAX_WORKERS}")
    return [
        sys.executable,
        str(PRODUCER),
        "--run-dir",
        str(scientific_dir),
        "--workers",
        str(int(workers)),
    ]


def repository_attestation() -> tuple[str, list[str]]:
    status = checked_output(
        ["git", "status", "--porcelain", "--untracked-files=all"]
    ).splitlines()
    if status:
        raise RuntimeError("launch requires a clean immutable git worktree")
    head = checked_output(["git", "rev-parse", "--verify", "HEAD"])
    for relative in SOURCE_PATHS:
        completed = subprocess.run(
            ["git", "cat-file", "-e", f"{head}:{relative}"],
            cwd=ROOT,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"scientific source is not committed at {head}: {relative}")
    return head, status


def write_prestart_manifests(
    *,
    run_dir: Path,
    head: str,
    status: list[str],
    command: list[str],
    workers: int,
    watch_interval_seconds: float,
    progress_stale_seconds: float,
) -> dict[str, Any]:
    contract_payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    source_manifest = {
        "schema_version": "program_m_shared_lift_hpi_source_manifest_v1",
        "git_head": head,
        "git_status_porcelain": status,
        "files": {
            relative: file_sha256(ROOT / relative) for relative in SOURCE_PATHS
        },
    }
    contract_manifest = {
        "schema_version": "program_m_shared_lift_hpi_contract_manifest_v1",
        "contract_path": str(CONTRACT),
        "contract_id": contract_payload.get("contract_id"),
        "contract_status": contract_payload.get("status"),
        "contract_file_sha256": file_sha256(CONTRACT),
        "governing_metric": contract_payload.get("governing_metric"),
        "paper2_learner_authorized": contract_payload.get(
            "paper2_learner_authorized"
        ),
        "paper3_authorized": contract_payload.get("paper3_authorized"),
        "virgin_tapes_authorized": contract_payload.get("virgin_tapes_authorized"),
    }
    environment_manifest = {
        "schema_version": "program_m_shared_lift_hpi_environment_manifest_v1",
        "captured_at_utc": utc_now(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "pip_freeze": checked_output([sys.executable, "-m", "pip", "freeze"]).splitlines(),
    }
    command_manifest = {
        "schema_version": "program_m_shared_lift_hpi_command_manifest_v1",
        "created_at_utc": utc_now(),
        "cwd": str(ROOT),
        "command": command,
        "workers": int(workers),
        "maximum_workers": MAX_WORKERS,
        "scientific_output": str(run_dir / "scientific_output"),
        "progress": str(run_dir / "scientific_output/progress.json"),
        "result": str(run_dir / "scientific_output/result.json"),
        "watch_interval_seconds": float(watch_interval_seconds),
        "progress_stale_seconds": float(progress_stale_seconds),
    }
    manifests = {
        "source_manifest.json": source_manifest,
        "contract_manifest.json": contract_manifest,
        "environment_manifest.json": environment_manifest,
        "seed_manifest.json": screen_seed_manifest(),
        "command_manifest.json": command_manifest,
    }
    for name, payload in manifests.items():
        exclusive_json(run_dir / name, payload)
    checksums = {
        "schema_version": "program_m_shared_lift_hpi_prestart_checksums_v1",
        "created_at_utc": utc_now(),
        "git_head": head,
        "files": {
            name: file_sha256(run_dir / name) for name in sorted(manifests)
        },
    }
    exclusive_json(run_dir / "prestart_checksums.json", checksums)
    return checksums


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--watch-interval-seconds", type=float, default=10.0)
    parser.add_argument("--progress-stale-seconds", type=float, default=600.0)
    args = parser.parse_args()
    if not 1 <= args.workers <= MAX_WORKERS:
        parser.error(f"workers must be between 1 and {MAX_WORKERS}")
    if min(args.watch_interval_seconds, args.progress_stale_seconds) <= 0:
        parser.error("watch interval and progress stale threshold must be positive")

    run_dir = args.run_dir.resolve(strict=False)
    try:
        run_dir.relative_to(RUN_ROOT.resolve())
    except ValueError as exc:
        parser.error(f"run directory must be under {RUN_ROOT}: {exc}")
    if run_dir.exists():
        parser.error("run directory already exists; runs are non-overwriting")
    try:
        head, status = repository_attestation()
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))

    run_dir.mkdir(parents=True, mode=0o700)
    os.chmod(run_dir, 0o700)
    scientific_dir = run_dir / "scientific_output"
    command = scientific_command(workers=args.workers, scientific_dir=scientific_dir)
    checksums = write_prestart_manifests(
        run_dir=run_dir,
        head=head,
        status=status,
        command=command,
        workers=args.workers,
        watch_interval_seconds=args.watch_interval_seconds,
        progress_stale_seconds=args.progress_stale_seconds,
    )

    watcher_stdout = (run_dir / "watcher_stdout.log").open("ab", buffering=0)
    watcher_stderr = (run_dir / "watcher_stderr.log").open("ab", buffering=0)
    watcher = subprocess.Popen(
        [
            sys.executable,
            str(WATCHER),
            "--run-dir",
            str(run_dir),
            "--interval-seconds",
            str(args.watch_interval_seconds),
            "--progress-stale-seconds",
            str(args.progress_stale_seconds),
        ],
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        stdout=watcher_stdout,
        stderr=watcher_stderr,
        start_new_session=True,
        close_fds=True,
    )
    watcher_stdout.close()
    watcher_stderr.close()
    latest_path = run_dir / "watcher_latest.json"
    deadline = time.monotonic() + 15.0
    latest = None
    while time.monotonic() < deadline:
        latest = load_json(latest_path)
        if (
            latest
            and latest.get("state") == "watching_prestart"
            and latest.get("watcher_pid") == watcher.pid
        ):
            break
        time.sleep(0.1)
    else:
        watcher.terminate()
        parser.error("watcher did not attest prestart liveness")

    stdout = (run_dir / "stdout.log").open("ab", buffering=0)
    stderr = (run_dir / "stderr.log").open("ab", buffering=0)
    scientific = subprocess.Popen(
        command,
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        stdout=stdout,
        stderr=stderr,
        start_new_session=True,
        close_fds=True,
    )
    stdout.close()
    stderr.close()
    session_id = os.getsid(scientific.pid)
    process_group_id = os.getpgid(scientific.pid)
    if session_id != scientific.pid or process_group_id != scientific.pid:
        os.killpg(process_group_id, 15)
        watcher.terminate()
        parser.error("scientific process did not receive an isolated session/group")

    pid_record = {
        "schema_version": "program_m_shared_lift_hpi_pid_v1",
        "launched_at_utc": utc_now(),
        "scientific_pid": scientific.pid,
        "scientific_process_group_id": process_group_id,
        "scientific_session_id": session_id,
        "watcher_pid": watcher.pid,
        "watcher_prestart_observed_at_utc": latest["observed_at_utc"],
        "command": command,
        "progress": str(scientific_dir / "progress.json"),
        "result": str(scientific_dir / "result.json"),
        "git_head": head,
    }
    exclusive_json(run_dir / "pid.json", pid_record)
    receipt = {
        "schema_version": "program_m_shared_lift_hpi_launch_receipt_v1",
        "launched_at_utc": utc_now(),
        "git_head": head,
        "scientific_pid": scientific.pid,
        "scientific_process_group_id": process_group_id,
        "scientific_session_id": session_id,
        "watcher_pid": watcher.pid,
        "watcher_started_before_scientific_process": True,
        "watcher_prestart_observed_at_utc": latest["observed_at_utc"],
        "watcher_scope": "ENTIRE_SCIENTIFIC_SESSION_INCLUDING_REPARENTED_WORKERS",
        "pid_record_sha256": file_sha256(run_dir / "pid.json"),
        "prestart_checksums_sha256": file_sha256(
            run_dir / "prestart_checksums.json"
        ),
        "prestart_manifest_checksums": checksums["files"],
    }
    exclusive_json(run_dir / "launch_receipt.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
