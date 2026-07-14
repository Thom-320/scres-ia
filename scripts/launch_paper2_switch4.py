#!/usr/bin/env python3
"""Launch the frozen <=4-switch preflight, producer, or verifier after its watcher."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.search_paper2_bottleneck_switch4 import (
    DEPENDENCIES as PREFLIGHT_DEPENDENCIES,
)


ROOT = Path(__file__).resolve().parent.parent
RESULT_ROOT = (
    ROOT / "results" / "paper2_bound_harness" / "switch_complexity_screen_v2"
)
CONTRACT = ROOT / "contracts" / "paper2_bottleneck_switch_complexity_screen_v2.json"
SEARCH = ROOT / "scripts" / "search_paper2_bottleneck_switch4.py"
VERIFIER = ROOT / "scripts" / "verify_paper2_bottleneck_switch4.py"
WATCHER = ROOT / "scripts" / "watch_paper2_switch4.py"


def git_blob_sha256(commit: str, relative: str) -> str | None:
    result = subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return sha256(result.stdout).hexdigest() if result.returncode == 0 else None


def spawn_worker_pids(row: dict[str, Any]) -> set[int]:
    workers: set[int] = set()
    root_pid = row.get("scientific_pid")
    tree = row.get("scientific_process_tree")
    if not isinstance(tree, list):
        return workers
    for process in tree:
        if not isinstance(process, dict):
            continue
        command = str(process.get("command", ""))
        pid = process.get("pid")
        if (
            isinstance(pid, int)
            and pid != root_pid
            and "multiprocessing.spawn" in command
            and "spawn_main" in command
            and "--multiprocessing-fork" in command
        ):
            workers.add(pid)
    return workers


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(value: Any) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def exclusive_json(path: Path, payload: dict[str, Any]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return rows
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            return []
        if not isinstance(value, dict):
            return []
        rows.append(value)
    return rows


def validate_preflight_evidence(
    result_path: Path,
    watcher_path: Path,
    receipt_path: Path,
    pid_path: Path,
    watcher_log_path: Path,
    *,
    expected_head: str,
) -> dict[str, Any]:
    result = load_json(result_path)
    watcher = load_json(watcher_path)
    receipt = load_json(receipt_path)
    pid_record = load_json(pid_path)
    watcher_rows = load_jsonl(watcher_log_path)
    if any(value is None for value in (result, watcher, receipt, pid_record)):
        raise ValueError("preflight custody JSON artifacts must be objects")
    assert result is not None
    assert watcher is not None
    assert receipt is not None
    assert pid_record is not None
    if not watcher_rows:
        raise ValueError("preflight watcher log must contain valid JSONL records")
    content = dict(result)
    recorded_content_hash = content.pop("content_sha256", None)
    contract = json.loads(CONTRACT.read_text())
    gate = contract["vps_preflight"]
    expected_hostname = contract["execution_discipline"]["expected_hostname"]
    current_host = socket.gethostname()
    result_sha = file_sha256(result_path)
    prestart_rows = [
        row
        for row in watcher_rows
        if row.get("state") == "watching_prestart"
        and row.get("scientific_pid") is None
        and row.get("watcher_pid") == receipt.get("watcher_pid")
    ]
    live_rows = [
        row
        for row in watcher_rows
        if row.get("scientific_pid_alive") is True
        and row.get("scientific_pid") == receipt.get("scientific_pid")
        and row.get("watcher_pid") == receipt.get("watcher_pid")
        and len(spawn_worker_pids(row)) >= 6
        and int(row.get("scientific_process_tree_rss_bytes", 0)) > 0
    ]
    observed_peak_rss = max(
        (
            int(row.get("scientific_process_tree_rss_bytes", 0))
            for row in watcher_rows
        ),
        default=0,
    )
    observed_memory = [
        int(row["memory_available_bytes"])
        for row in watcher_rows
        if isinstance(row.get("memory_available_bytes"), int)
    ]
    prestart_before_science = False
    if prestart_rows and isinstance(pid_record.get("launched_at_utc"), str):
        try:
            prestart_before_science = datetime.fromisoformat(
                str(prestart_rows[0]["observed_at_utc"])
            ) <= datetime.fromisoformat(str(pid_record["launched_at_utc"]))
        except (KeyError, TypeError, ValueError):
            prestart_before_science = False
    expected_dependencies = {
        str(path.relative_to(ROOT)): file_sha256(path)
        for path in PREFLIGHT_DEPENDENCIES
    }
    dependency_rows = result.get("dependency_sha256")
    dependency_hashes_current = dependency_rows == expected_dependencies
    dependency_blobs_match_commit = dependency_hashes_current and all(
        git_blob_sha256(expected_head, relative) == digest
        for relative, digest in expected_dependencies.items()
    )
    forbidden = (
        "h_pi_computed",
        "h_obs_computed",
        "w24_authorized",
        "learner_authorized",
        "paper2_authorized",
        "paper3_authorized",
    )
    checks = {
        "content_hash": recorded_content_hash == json_sha256(content),
        "schema": result.get("schema_version")
        == "paper2_bottleneck_switch4_preflight_v1",
        "contract_id": result.get("contract_id")
        == "paper2_bottleneck_switch_complexity_screen_v2",
        "contract_hash": result.get("contract_sha256") == file_sha256(CONTRACT),
        "preflight_only": result.get("preflight_only") is True,
        "launch_clean": result.get("launch_git_status_porcelain") == [],
        "same_commit": result.get("git_head") == expected_head,
        "dependency_hashes_current": dependency_hashes_current,
        "dependency_blobs_match_commit": dependency_blobs_match_commit,
        "designated_vps_runtime": current_host == expected_hostname,
        "same_host_result": result.get("environment", {}).get("hostname")
        == expected_hostname,
        "same_host_watcher": watcher.get("hostname") == expected_hostname,
        "same_host_receipt": receipt.get("hostname") == expected_hostname,
        "candidate_count": result.get("candidate_count") == 89_131,
        "scores_evaluated": result.get("scores_evaluated") == 534_786,
        "calibration_seed_block": result.get("seed_start") == 1_100_001
        and result.get("seed_end") == 1_100_006
        and result.get("n_tapes") == 6
        and isinstance(result.get("tapes"), list)
        and len(result["tapes"]) == 6
        and [row.get("seed") for row in result["tapes"]]
        == list(range(1_100_001, 1_100_007))
        and all(
            isinstance(row.get("scores_float_hex_sha256"), str)
            and len(row["scores_float_hex_sha256"]) == 64
            and isinstance(row.get("tape_sha256"), str)
            and len(row["tape_sha256"]) == 64
            for row in result["tapes"]
        ),
        "locked_closed": result.get("locked_tapes_accessed") is False,
        "virgin_closed": result.get("virgin_tapes_accessed") is False,
        "claim_flags_closed": all(result.get(field) is False for field in forbidden),
        "watcher_terminal": watcher.get("state")
        == gate["watcher_terminal_state"],
        "watcher_log_terminal_matches": watcher_rows[-1] == watcher,
        "watcher_pid_bound": watcher.get("watcher_pid")
        == receipt.get("watcher_pid") == pid_record.get("watcher_pid"),
        "scientific_pid_bound": watcher.get("scientific_pid")
        == receipt.get("scientific_pid") == pid_record.get("scientific_pid"),
        "receipt_schema": receipt.get("schema_version")
        == "paper2_switch4_detached_launch_v1",
        "receipt_operation": receipt.get("operation")
        == "vps_six_tape_memory_preflight",
        "receipt_commit": receipt.get("git_head") == expected_head,
        "receipt_contract": receipt.get("contract_sha256")
        == file_sha256(CONTRACT),
        "receipt_script_hashes": receipt.get("launcher_sha256")
        == file_sha256(Path(__file__).resolve())
        and receipt.get("search_sha256") == file_sha256(SEARCH)
        and receipt.get("verifier_sha256") == file_sha256(VERIFIER)
        and receipt.get("watcher_sha256") == file_sha256(WATCHER),
        "receipt_prestart_claim": receipt.get(
            "watcher_started_before_scientific_process"
        )
        is True,
        "receipt_pid_hash": receipt.get("pid_record_sha256")
        == file_sha256(pid_path),
        "receipt_output": Path(str(receipt.get("output", ""))).resolve()
        == result_path.resolve(),
        "receipt_command_bound": receipt.get("command")
        == pid_record.get("command") == result.get("command"),
        "pid_commit": pid_record.get("git_head") == expected_head,
        "pid_output": Path(str(pid_record.get("output", ""))).resolve()
        == result_path.resolve(),
        "prestart_recorded": bool(prestart_rows),
        "prestart_before_science": prestart_before_science,
        "six_worker_live_sample": bool(live_rows),
        "watcher_result_exists": watcher.get("result_exists") is True,
        "watcher_process_closed": watcher.get("scientific_pid_alive") is False,
        "watcher_output_path": Path(str(watcher.get("output", ""))).resolve()
        == result_path.resolve(),
        "watcher_result_hash": watcher.get("result_sha256") == result_sha,
        "watcher_progress_complete": isinstance(watcher.get("progress"), dict)
        and watcher["progress"].get("stage") == "complete"
        and watcher["progress"].get("completed") == 6
        and watcher["progress"].get("total") == 6
        and watcher["progress"].get("output_sha256") == result_sha,
        "stderr_zero": watcher.get("stderr_bytes") == gate["stderr_bytes"],
        "peak_rss_observed_not_claimed": observed_peak_rss > 0
        and watcher.get("peak_scientific_process_tree_rss_bytes")
        == observed_peak_rss,
        "rss_below_gate": isinstance(
            watcher.get("peak_scientific_process_tree_rss_bytes"), int
        )
        and watcher["peak_scientific_process_tree_rss_bytes"]
        <= gate["maximum_process_tree_rss_bytes"],
        "memory_available_above_gate": isinstance(
            watcher.get("minimum_memory_available_bytes"), int
        )
        and watcher["minimum_memory_available_bytes"]
        >= gate["minimum_observed_memory_available_bytes"],
        "minimum_memory_observed_not_claimed": bool(observed_memory)
        and watcher.get("minimum_memory_available_bytes") == min(observed_memory),
    }
    failed = sorted(key for key, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"preflight evidence failed closed: {failed}")
    return {
        "result_path": str(result_path),
        "result_sha256": result_sha,
        "watcher_path": str(watcher_path),
        "watcher_sha256": file_sha256(watcher_path),
        "launch_receipt_path": str(receipt_path),
        "launch_receipt_sha256": file_sha256(receipt_path),
        "pid_path": str(pid_path),
        "pid_sha256": file_sha256(pid_path),
        "watcher_log_path": str(watcher_log_path),
        "watcher_log_sha256": file_sha256(watcher_log_path),
        "git_head": expected_head,
        "hostname": current_host,
        "peak_scientific_process_tree_rss_bytes": watcher[
            "peak_scientific_process_tree_rss_bytes"
        ],
        "minimum_memory_available_bytes": watcher[
            "minimum_memory_available_bytes"
        ],
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--watch-interval-seconds", type=float, default=10.0)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--deep-verify-result", type=Path)
    parser.add_argument("--trusted-preflight-result", type=Path)
    parser.add_argument("--trusted-preflight-watcher", type=Path)
    parser.add_argument("--trusted-preflight-receipt", type=Path)
    parser.add_argument("--trusted-preflight-pid", type=Path)
    parser.add_argument("--trusted-preflight-watcher-log", type=Path)
    args = parser.parse_args()
    if args.workers < 1 or args.watch_interval_seconds <= 0:
        parser.error("workers and watcher interval must be positive")
    if args.preflight and args.workers != 6:
        parser.error("the frozen preflight requires --workers 6")
    if not args.preflight and args.deep_verify_result is None and args.workers != 6:
        parser.error("the frozen producer requires --workers 6")
    run_dir = args.run_dir.resolve(strict=False)
    try:
        run_dir.relative_to(RESULT_ROOT.resolve())
    except ValueError as exc:
        parser.error(f"run directory must be under {RESULT_ROOT}: {exc}")
    if run_dir.exists():
        parser.error("run directory already exists; runs are non-overwriting")
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.splitlines()
    if status:
        parser.error("detached launch requires a clean immutable worktree")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    expected_hostname = json.loads(CONTRACT.read_text())[
        "execution_discipline"
    ]["expected_hostname"]
    if args.deep_verify_result is None and socket.gethostname() != expected_hostname:
        parser.error(
            f"preflight and producer require hostname {expected_hostname}"
        )

    preflight_evidence: dict[str, Any] | None = None
    trusted_paths = (
        args.trusted_preflight_result,
        args.trusted_preflight_watcher,
        args.trusted_preflight_receipt,
        args.trusted_preflight_pid,
        args.trusted_preflight_watcher_log,
    )
    if not args.preflight and args.deep_verify_result is None:
        if any(path is None for path in trusted_paths):
            parser.error("producer requires the five trusted preflight custody paths")
        try:
            preflight_evidence = validate_preflight_evidence(
                args.trusted_preflight_result.resolve(strict=True),
                args.trusted_preflight_watcher.resolve(strict=True),
                args.trusted_preflight_receipt.resolve(strict=True),
                args.trusted_preflight_pid.resolve(strict=True),
                args.trusted_preflight_watcher_log.resolve(strict=True),
                expected_head=head,
            )
        except ValueError as exc:
            parser.error(str(exc))
    elif any(path is not None for path in trusted_paths):
        parser.error("trusted preflight paths apply only to the producer")

    run_dir.mkdir(parents=True, mode=0o700)
    os.chmod(run_dir, 0o700)
    progress = run_dir / "progress.json"
    verified_result: Path | None = None
    if args.preflight:
        operation = "vps_six_tape_memory_preflight"
        output = run_dir / "preflight.json"
        command = [
            sys.executable,
            str(SEARCH),
            "--output",
            str(output),
            "--progress",
            str(progress),
            "--workers",
            "6",
            "--preflight-only",
        ]
    elif args.deep_verify_result is not None:
        operation = "deep_verification"
        output = run_dir / "verification.json"
        verified_result = args.deep_verify_result.resolve(strict=True)
        command = [
            sys.executable,
            str(VERIFIER),
            "--result",
            str(verified_result),
            "--output",
            str(output),
            "--progress",
            str(progress),
            "--workers",
            str(args.workers),
            "--deep",
        ]
    else:
        operation = "calibration_switch4_screen"
        output = run_dir / "result.json"
        command = [
            sys.executable,
            str(SEARCH),
            "--output",
            str(output),
            "--progress",
            str(progress),
            "--workers",
            "6",
        ]
    manifest = {
        "schema_version": "paper2_switch4_detached_launch_v1",
        "created_at_utc": utc_now(),
        "git_head": head,
        "launch_git_status_porcelain": status,
        "cwd": str(ROOT),
        "hostname": socket.gethostname(),
        "operation": operation,
        "command": command,
        "output": str(output),
        "progress": str(progress),
        "workers": args.workers,
        "watch_interval_seconds": args.watch_interval_seconds,
        "contract_sha256": file_sha256(CONTRACT),
        "launcher_sha256": file_sha256(Path(__file__).resolve()),
        "search_sha256": file_sha256(SEARCH),
        "verifier_sha256": file_sha256(VERIFIER),
        "watcher_sha256": file_sha256(WATCHER),
        "verified_result": str(verified_result) if verified_result else None,
        "verified_result_sha256": (
            file_sha256(verified_result) if verified_result else None
        ),
        "trusted_preflight_evidence": preflight_evidence,
    }
    exclusive_json(run_dir / "launch_manifest.json", manifest)
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
        ],
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        stdout=watcher_stdout,
        stderr=watcher_stderr,
        start_new_session=True,
        close_fds=True,
    )
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
    pid_payload = {
        "schema_version": "paper2_switch4_pid_v1",
        "launched_at_utc": utc_now(),
        "scientific_pid": scientific.pid,
        "watcher_pid": watcher.pid,
        "watcher_prestart_observed_at_utc": latest["observed_at_utc"],
        "command": command,
        "output": str(output),
        "progress": str(progress),
        "git_head": head,
    }
    exclusive_json(run_dir / "pid.json", pid_payload)
    receipt = {
        **manifest,
        "scientific_pid": scientific.pid,
        "watcher_pid": watcher.pid,
        "watcher_started_before_scientific_process": True,
        "pid_record_sha256": file_sha256(run_dir / "pid.json"),
    }
    exclusive_json(run_dir / "launch_receipt.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
