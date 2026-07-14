#!/usr/bin/env python3
"""Validate completed <=4-switch producer content and detached custody."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.launch_paper2_switch4 import (
    git_blob_sha256,
    json_sha256,
    load_json,
    load_jsonl,
    file_sha256,
    spawn_worker_pids,
)
from scripts.verify_paper2_bottleneck_switch4 import validate_payload


SCHEMA = "paper2_switch4_producer_custody_validation_v1"
EXPECTED_HOSTNAME = "vps-f733423b"
EXPECTED_OPERATION = "calibration_switch4_screen"
EXPECTED_RESULT_SCHEMA = "paper2_bottleneck_switch4_screen_v1"
EXPECTED_PYTHON = "/home/ubuntu/scres-ia/.venv/bin/python"
VALIDATOR_RELATIVE_PATH = "scripts/validate_paper2_switch4_producer_custody.py"
EXPECTED_RUN_FILES = {
    "result": "result.json",
    "progress": "progress.json",
    "watcher": "watcher_latest.json",
    "manifest": "launch_manifest.json",
    "receipt": "launch_receipt.json",
    "pid": "pid.json",
    "watcher_log": "watcher.jsonl",
    "stderr": "stderr.log",
    "stdout": "stdout.log",
}
PREFLIGHT_FILES = {
    "result_sha256": "preflight.json",
    "watcher_sha256": "watcher_latest.json",
    "launch_receipt_sha256": "launch_receipt.json",
    "pid_sha256": "pid.json",
    "watcher_log_sha256": "watcher.jsonl",
}
ANCHOR_PREFLIGHT_FILES = {
    "preflight.json",
    "progress.json",
    "watcher_latest.json",
    "launch_manifest.json",
    "launch_receipt.json",
    "pid.json",
    "watcher.jsonl",
    "stdout.log",
    "stderr.log",
}
SCRIPT_BLOBS = {
    "launcher_sha256": "scripts/launch_paper2_switch4.py",
    "search_sha256": "scripts/search_paper2_bottleneck_switch4.py",
    "verifier_sha256": "scripts/verify_paper2_bottleneck_switch4.py",
    "watcher_sha256": "scripts/watch_paper2_switch4.py",
}
PREFLIGHT_DEPENDENCY_BLOBS = {
    "contracts/paper2_bottleneck_primary_bound_v2.json",
    "contracts/paper2_bottleneck_switch_complexity_screen_v2.json",
    "scripts/run_paper2_bottleneck_full_frontier.py",
    "scripts/search_paper2_bottleneck_switch4.py",
    "scripts/search_paper2_bottleneck_switch_complexity.py",
    "supply_chain/episode_metrics.py",
    "supply_chain/paper2_bottleneck.py",
    "supply_chain/program_f.py",
    "supply_chain/ret_thesis.py",
    "supply_chain/supply_chain.py",
}
VALIDATION_RUNTIME_PATHS = {
    "scripts/launch_paper2_switch4.py",
    "scripts/verify_paper2_bottleneck_switch4.py",
}
DYNAMIC_RECEIPT_FIELDS = {
    "pid_record_sha256",
    "scientific_pid",
    "watcher_pid",
    "watcher_started_before_scientific_process",
}


def _remote_tail_matches(value: Any, run_id: str, filename: str) -> bool:
    path = Path(str(value))
    return path.name == filename and path.parent.name == run_id


def _prestart_before_launch(
    rows: list[dict[str, Any]], receipt: dict[str, Any], pid: dict[str, Any]
) -> bool:
    eligible = [
        row
        for row in rows
        if row.get("state") == "watching_prestart"
        and row.get("scientific_pid") is None
        and row.get("watcher_pid") == receipt.get("watcher_pid")
    ]
    if not eligible:
        return False
    try:
        return datetime.fromisoformat(str(eligible[0]["observed_at_utc"])) <= (
            datetime.fromisoformat(str(pid["launched_at_utc"]))
        )
    except (KeyError, TypeError, ValueError):
        return False


def _command_is_frozen_producer(command: Any, run_id: str) -> bool:
    if not isinstance(command, list) or len(command) != 8:
        return False
    return (
        command[0] == EXPECTED_PYTHON
        and str(command[1]).endswith("/scripts/search_paper2_bottleneck_switch4.py")
        and command[2] == "--output"
        and _remote_tail_matches(command[3], run_id, "result.json")
        and command[4] == "--progress"
        and _remote_tail_matches(command[5], run_id, "progress.json")
        and command[6:] == ["--workers", "6"]
    )


def _parse_utc(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo is not None else None


def _watcher_log_is_bound(
    rows: list[dict[str, Any]],
    watcher: dict[str, Any],
    receipt: dict[str, Any],
    pid: dict[str, Any],
    *,
    expected_output: str,
) -> bool:
    if len(rows) < 3 or rows[-1] != watcher:
        return False
    timestamps = [_parse_utc(row.get("observed_at_utc")) for row in rows]
    if any(value is None for value in timestamps):
        return False
    ordered = [value for value in timestamps if value is not None]
    if ordered != sorted(ordered):
        return False
    created = _parse_utc(receipt.get("created_at_utc"))
    launched = _parse_utc(pid.get("launched_at_utc"))
    started_values = {row.get("watcher_started_at_utc") for row in rows}
    watcher_started = (
        _parse_utc(next(iter(started_values))) if len(started_values) == 1 else None
    )
    if (
        created is None
        or launched is None
        or watcher_started is None
        or not (created <= watcher_started <= ordered[0] <= launched <= ordered[-1])
    ):
        return False
    command = receipt.get("command")
    if not isinstance(command, list):
        return False
    rendered_command = " ".join(str(value) for value in command)
    scientific_pid = receipt.get("scientific_pid")
    watcher_pid = receipt.get("watcher_pid")
    if not (
        rows[0].get("state") == "watching_prestart"
        and rows[0].get("scientific_pid") is None
    ):
        return False
    for row in rows:
        if (
            row.get("schema_version") != "paper2_switch4_watcher_v1"
            or row.get("hostname") != EXPECTED_HOSTNAME
            or row.get("watcher_pid") != watcher_pid
        ):
            return False
        row_pid = row.get("scientific_pid")
        if row_pid not in (None, scientific_pid):
            return False
        if row.get("scientific_pid_alive") is True:
            if (
                _parse_utc(row.get("observed_at_utc")) < launched
                or row_pid != scientific_pid
                or row.get("scientific_command") != rendered_command
                or row.get("output") != expected_output
            ):
                return False
    live_indices = [
        index for index, row in enumerate(rows) if row.get("scientific_pid_alive") is True
    ]
    if not live_indices or max(live_indices) >= len(rows) - 1:
        return False
    return True


def _live_row_has_exact_workers(
    row: dict[str, Any], receipt: dict[str, Any]
) -> bool:
    tree = row.get("scientific_process_tree")
    command = receipt.get("command")
    if not isinstance(tree, list) or not isinstance(command, list):
        return False
    scientific_pid = receipt.get("scientific_pid")
    rendered_command = " ".join(str(value) for value in command)
    roots = [item for item in tree if item.get("pid") == scientific_pid]
    workers = [
        item
        for item in tree
        if item.get("ppid") == scientific_pid
        and str(item.get("command", "")).startswith(EXPECTED_PYTHON + " -c ")
        and "multiprocessing.spawn" in str(item.get("command", ""))
        and "--multiprocessing-fork" in str(item.get("command", ""))
    ]
    rss_values = [item.get("rss_bytes") for item in tree]
    return (
        len(roots) == 1
        and roots[0].get("ppid") == 1
        and roots[0].get("command") == rendered_command
        and len(workers) == 6
        and all(isinstance(value, int) and value >= 0 for value in rss_values)
        and sum(rss_values) == row.get("scientific_process_tree_rss_bytes")
    )


def _validator_provenance() -> dict[str, Any]:
    root = Path(__file__).resolve().parent.parent
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
        current_sha = file_sha256(Path(__file__).resolve())
        blob_sha = git_blob_sha256(head, VALIDATOR_RELATIVE_PATH)
        path_status = subprocess.check_output(
            ["git", "status", "--porcelain", "--", VALIDATOR_RELATIVE_PATH],
            cwd=root,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError, ValueError):
        return {
            "passed": False,
            "git_head": None,
            "validator_sha256": None,
            "validator_blob_sha256": None,
            "validator_path_git_status": None,
        }
    return {
        "passed": current_sha == blob_sha and path_status == "",
        "git_head": head,
        "validator_sha256": current_sha,
        "validator_blob_sha256": blob_sha,
        "validator_path_git_status": path_status,
    }


def _validation_runtime_provenance(expected_head: str) -> dict[str, Any]:
    root = Path(__file__).resolve().parent.parent
    sources: dict[str, dict[str, Any]] = {}
    for relative in sorted(VALIDATION_RUNTIME_PATHS):
        try:
            current_sha = file_sha256(root / relative)
            blob_sha = git_blob_sha256(expected_head, relative)
            status = subprocess.check_output(
                ["git", "status", "--porcelain", "--", relative],
                cwd=root,
                text=True,
            ).strip()
        except (OSError, subprocess.CalledProcessError, ValueError):
            sources[relative] = {"passed": False}
            continue
        sources[relative] = {
            "passed": current_sha == blob_sha and status == "",
            "current_sha256": current_sha,
            "expected_blob_sha256": blob_sha,
            "git_status": status,
        }
    return {
        "passed": len(sources) == len(VALIDATION_RUNTIME_PATHS)
        and all(value.get("passed") is True for value in sources.values()),
        "expected_head": expected_head,
        "sources": sources,
    }


def _tracked_file_provenance(path: Path) -> dict[str, Any]:
    root = Path(__file__).resolve().parent.parent
    try:
        relative = str(path.resolve(strict=True).relative_to(root))
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
        current_sha = file_sha256(path)
        blob_sha = git_blob_sha256(head, relative)
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--", relative],
            cwd=root,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError, ValueError):
        return {"passed": False}
    return {
        "passed": current_sha == blob_sha and status == "",
        "git_head": head,
        "relative_path": relative,
        "current_sha256": current_sha,
        "blob_sha256": blob_sha,
        "git_status": status,
    }


def _prefix_sha256(path: Path, size: int) -> str | None:
    if size <= 0 or path.stat().st_size < size:
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        remaining = size
        while remaining:
            block = handle.read(min(1_048_576, remaining))
            if not block:
                return None
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


def _validate_precompletion_anchor(
    anchor_path: Path,
    run_dir: Path,
    preflight_dir: Path,
    receipt: dict[str, Any],
    pid: dict[str, Any],
    *,
    expected_head: str,
) -> dict[str, Any]:
    anchor = load_json(anchor_path)
    if anchor is None:
        return {"passed": False, "checks": {"anchor_parse": False}}
    run_id = Path(str(receipt.get("output"))).parent.name
    scientific = anchor.get("processes", {}).get("scientific", {})
    watcher_process = anchor.get("processes", {}).get("watcher", {})
    prefix = anchor.get("watcher_log_prefix", {})
    prefix_size = prefix.get("bytes")
    observed = _parse_utc(anchor.get("observed_at_utc"))
    launched = _parse_utc(pid.get("launched_at_utc"))
    anchored_files = anchor.get("immutable_run_files")
    anchored_preflight = anchor.get("preflight_files")
    exact_preflight = (
        isinstance(anchored_preflight, dict)
        and set(anchored_preflight) == ANCHOR_PREFLIGHT_FILES
        and all(
            anchored_preflight.get(filename) == file_sha256(preflight_dir / filename)
            for filename in ANCHOR_PREFLIGHT_FILES
        )
    )
    immutable_names = {"launch_manifest.json", "launch_receipt.json", "pid.json"}
    exact_immutable = (
        isinstance(anchored_files, dict)
        and immutable_names.issubset(anchored_files)
        and all(
            anchored_files[filename].get("sha256") == file_sha256(run_dir / filename)
            and anchored_files[filename].get("bytes")
            == (run_dir / filename).stat().st_size
            for filename in immutable_names
        )
    )
    rendered_command = " ".join(str(value) for value in receipt.get("command", []))
    checks = {
        "anchor_parse": True,
        "schema": anchor.get("schema_version")
        == "paper2_switch4_precompletion_anchor_v1",
        "claim_limit": anchor.get("claim_limit")
        == "Pre-completion custody anchor only; no scientific result or policy ranking exists at this observation.",
        "host_and_commit": anchor.get("hostname") == EXPECTED_HOSTNAME
        and anchor.get("source_git_head") == expected_head
        and anchor.get("source_git_status_porcelain") == [],
        "run_id": anchor.get("run_id") == run_id,
        "observed_after_launch": observed is not None
        and launched is not None
        and observed >= launched,
        "no_result_at_anchor": anchor.get("result_bytes") == 0
        and anchor.get("progress_bytes") == 0,
        "process_identity": scientific.get("pid") == receipt.get("scientific_pid")
        == pid.get("scientific_pid")
        and scientific.get("ppid") == 1
        and isinstance(scientific.get("start_ticks"), int)
        and scientific.get("start_ticks") > 0
        and scientific.get("cmdline") == rendered_command
        and watcher_process.get("pid") == receipt.get("watcher_pid")
        == pid.get("watcher_pid")
        and watcher_process.get("ppid") == 1
        and isinstance(watcher_process.get("start_ticks"), int)
        and watcher_process.get("start_ticks") > 0
        and str(watcher_process.get("cmdline", "")).startswith(
            EXPECTED_PYTHON + " "
        )
        and f"--run-dir {receipt.get('output', '').rsplit('/', 1)[0]}"
        in str(watcher_process.get("cmdline", "")),
        "boot_identity": isinstance(anchor.get("boot_id"), str)
        and len(anchor["boot_id"]) == 36,
        "immutable_launch_files": exact_immutable,
        "preflight_files": exact_preflight,
        "watcher_prefix": isinstance(prefix_size, int)
        and _prefix_sha256(run_dir / "watcher.jsonl", prefix_size)
        == prefix.get("sha256"),
    }
    provenance = _tracked_file_provenance(anchor_path)
    checks["anchor_provenance"] = provenance.get("passed") is True
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "anchor_sha256": file_sha256(anchor_path),
        "provenance": provenance,
    }


def _manifest_is_receipt_plan(
    manifest: dict[str, Any], receipt: dict[str, Any]
) -> bool:
    return (
        set(receipt) - set(manifest) == DYNAMIC_RECEIPT_FIELDS
        and set(manifest).isdisjoint(DYNAMIC_RECEIPT_FIELDS)
        and all(receipt.get(key) == value for key, value in manifest.items())
    )


def _remote_paths_consistent(
    receipt: dict[str, Any],
    pid: dict[str, Any],
    watcher: dict[str, Any],
    run_id: str,
    *,
    output_filename: str = "result.json",
) -> bool:
    try:
        root = Path(str(receipt["cwd"]))
        command = receipt["command"]
        remote_run = (
            root
            / "results"
            / "paper2_bound_harness"
            / "switch_complexity_screen_v2"
            / run_id
        )
        return (
            isinstance(command, list)
            and Path(str(command[1]))
            == root / "scripts" / "search_paper2_bottleneck_switch4.py"
            and Path(str(command[3])) == remote_run / output_filename
            and Path(str(command[5])) == remote_run / "progress.json"
            and Path(str(receipt["output"])) == remote_run / output_filename
            and Path(str(receipt["progress"])) == remote_run / "progress.json"
            and pid.get("output") == receipt.get("output") == watcher.get("output")
            and pid.get("progress") == receipt.get("progress")
            and command[3] == receipt.get("output")
            and command[5] == receipt.get("progress")
        )
    except (IndexError, KeyError, TypeError):
        return False


def _validate_relocated_preflight(
    preflight_dir: Path, *, expected_head: str
) -> dict[str, Any]:
    result_path = preflight_dir / "preflight.json"
    progress_path = preflight_dir / "progress.json"
    watcher_path = preflight_dir / "watcher_latest.json"
    manifest_path = preflight_dir / "launch_manifest.json"
    receipt_path = preflight_dir / "launch_receipt.json"
    pid_path = preflight_dir / "pid.json"
    watcher_log_path = preflight_dir / "watcher.jsonl"
    stderr_path = preflight_dir / "stderr.log"
    stdout_path = preflight_dir / "stdout.log"
    result = load_json(result_path)
    progress_artifact = load_json(progress_path)
    watcher = load_json(watcher_path)
    manifest = load_json(manifest_path)
    receipt = load_json(receipt_path)
    pid = load_json(pid_path)
    rows = load_jsonl(watcher_log_path)
    if (
        any(
            value is None
            for value in (result, progress_artifact, watcher, manifest, receipt, pid)
        )
        or not rows
    ):
        return {"passed": False, "checks": {"artifacts_parse": False}}
    assert result is not None
    assert progress_artifact is not None
    assert watcher is not None
    assert manifest is not None
    assert receipt is not None
    assert pid is not None
    run_id = Path(str(receipt.get("output"))).parent.name
    local_identity_bound = preflight_dir.name == run_id or (
        preflight_dir.name == "retrieved" and preflight_dir.parent.name == run_id
    )
    content = dict(result)
    recorded_content_sha = content.pop("content_sha256", None)
    result_sha = file_sha256(result_path)
    command = receipt.get("command")
    dependencies = result.get("dependency_sha256")
    try:
        dependency_blobs = (
            isinstance(dependencies, dict)
            and set(dependencies) == PREFLIGHT_DEPENDENCY_BLOBS
            and all(
                git_blob_sha256(expected_head, str(relative)) == digest
                for relative, digest in dependencies.items()
            )
        )
    except (OSError, subprocess.CalledProcessError, ValueError):
        dependency_blobs = False
    live_rows = [
        row
        for row in rows
        if row.get("scientific_pid_alive") is True
        and row.get("scientific_pid") == receipt.get("scientific_pid")
        and row.get("watcher_pid") == receipt.get("watcher_pid")
        and len(spawn_worker_pids(row)) == 6
        and _live_row_has_exact_workers(row, receipt)
        and int(row.get("scientific_process_tree_rss_bytes", 0)) > 0
    ]
    observed_peak = max(
        (int(row.get("scientific_process_tree_rss_bytes", 0)) for row in rows),
        default=0,
    )
    observed_memory = [
        int(row["memory_available_bytes"])
        for row in rows
        if isinstance(row.get("memory_available_bytes"), int)
    ]
    progress = watcher.get("progress")
    claims = (
        "h_pi_computed",
        "h_obs_computed",
        "w24_authorized",
        "learner_authorized",
        "paper2_authorized",
        "paper3_authorized",
    )
    command_shape = (
        isinstance(command, list)
        and len(command) == 9
        and command[0] == EXPECTED_PYTHON
        and str(command[1]).endswith("/scripts/search_paper2_bottleneck_switch4.py")
        and command[2] == "--output"
        and command[4] == "--progress"
        and command[6:] == ["--workers", "6", "--preflight-only"]
    )
    paths_bound = _remote_paths_consistent(
        receipt, pid, watcher, run_id, output_filename="preflight.json"
    )
    expected_output = str(receipt.get("output"))
    watcher_log_bound = _watcher_log_is_bound(
        rows, watcher, receipt, pid, expected_output=expected_output
    )
    tapes = result.get("tapes")
    exact_tape_manifest = (
        isinstance(tapes, list)
        and len(tapes) == 6
        and [row.get("index") for row in tapes] == list(range(6))
        and [row.get("seed") for row in tapes] == list(range(1_100_001, 1_100_007))
        and all(
            row.get("split") == "calibration"
            and row.get("context")
            in {"equipment_pressure", "interdiction_campaign", "mission_surge"}
            and isinstance(row.get("tape_sha256"), str)
            and len(row["tape_sha256"]) == 64
            and isinstance(row.get("scores_float_hex_sha256"), str)
            and len(row["scores_float_hex_sha256"]) == 64
            and isinstance(row.get("exogenous_hashes"), list)
            and len(row["exogenous_hashes"]) == 2
            and all(
                isinstance(value, str) and len(value) == 64
                for value in row["exogenous_hashes"]
            )
            for row in tapes
        )
    )
    checks = {
        "artifacts_parse": True,
        "content_hash": recorded_content_sha == json_sha256(content),
        "schema": result.get("schema_version")
        == "paper2_bottleneck_switch4_preflight_v1",
        "contract": result.get("contract_id")
        == "paper2_bottleneck_switch_complexity_screen_v2"
        and result.get("contract_sha256")
        == receipt.get("contract_sha256")
        == git_blob_sha256(
            expected_head,
            "contracts/paper2_bottleneck_switch_complexity_screen_v2.json",
        ),
        "same_commit": result.get("git_head")
        == receipt.get("git_head")
        == pid.get("git_head")
        == expected_head,
        "launch_clean": result.get("launch_git_status_porcelain") == []
        and receipt.get("launch_git_status_porcelain") == [],
        "same_host": result.get("environment", {}).get("hostname")
        == watcher.get("hostname")
        == receipt.get("hostname")
        == EXPECTED_HOSTNAME,
        "runtime": command_shape
        and result.get("environment", {}).get("python_executable") == EXPECTED_PYTHON,
        "dependency_blobs": dependency_blobs,
        "preflight_scope": result.get("preflight_only") is True
        and result.get("candidate_count") == 89_131
        and result.get("scores_evaluated") == 534_786
        and result.get("seed_start") == 1_100_001
        and result.get("seed_end") == 1_100_006
        and result.get("n_tapes") == 6,
        "exact_tape_manifest": exact_tape_manifest,
        "claims_closed": result.get("locked_tapes_accessed") is False
        and result.get("virgin_tapes_accessed") is False
        and all(result.get(field) is False for field in claims),
        "receipt": receipt.get("schema_version")
        == "paper2_switch4_detached_launch_v1"
        and receipt.get("operation") == "vps_six_tape_memory_preflight"
        and receipt.get("workers") == 6
        and receipt.get("watcher_started_before_scientific_process") is True,
        "launch_manifest": _manifest_is_receipt_plan(manifest, receipt),
        "script_blobs": all(
            receipt.get(key) == git_blob_sha256(expected_head, relative)
            for key, relative in SCRIPT_BLOBS.items()
        ),
        "command_bound": command == pid.get("command") == result.get("command"),
        "paths_bound": paths_bound,
        "local_identity_bound": local_identity_bound,
        "watcher_log_bound": watcher_log_bound,
        "pid_hash": receipt.get("pid_record_sha256") == file_sha256(pid_path),
        "pids_bound": watcher.get("watcher_pid")
        == receipt.get("watcher_pid")
        == pid.get("watcher_pid")
        and watcher.get("scientific_pid")
        == receipt.get("scientific_pid")
        == pid.get("scientific_pid"),
        "prestart": _prestart_before_launch(rows, receipt, pid),
        "six_workers": bool(live_rows),
        "terminal": rows[-1] == watcher
        and watcher.get("state") == "completed_unverified"
        and watcher.get("scientific_pid_alive") is False,
        "result_hash": watcher.get("result_exists") is True
        and watcher.get("result_sha256") == result_sha,
        "progress": isinstance(progress, dict)
        and progress_artifact == progress
        and watcher.get("progress_sha256") == file_sha256(progress_path)
        and progress.get("stage") == "complete"
        and progress.get("completed") == 6
        and progress.get("total") == 6
        and progress.get("output_sha256") == result_sha,
        "stderr": watcher.get("stderr_bytes") == 0
        and stderr_path.is_file()
        and stderr_path.stat().st_size == 0,
        "stdout": stdout_path.is_file()
        and watcher.get("stdout_bytes") == stdout_path.stat().st_size,
        "resource_observations": observed_peak > 0
        and watcher.get("peak_scientific_process_tree_rss_bytes") == observed_peak
        and bool(observed_memory)
        and watcher.get("minimum_memory_available_bytes") == min(observed_memory),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "result_sha256": result_sha,
        "watcher_sha256": file_sha256(watcher_path),
        "progress_sha256": file_sha256(progress_path),
        "launch_manifest_sha256": file_sha256(manifest_path),
        "launch_receipt_sha256": file_sha256(receipt_path),
        "pid_sha256": file_sha256(pid_path),
        "watcher_log_sha256": file_sha256(watcher_log_path),
        "stdout_sha256": file_sha256(stdout_path),
    }


def validate_producer_custody(
    run_dir: Path,
    preflight_dir: Path,
    anchor_path: Path,
    *,
    expected_head: str,
) -> dict[str, Any]:
    run_dir = run_dir.resolve(strict=True)
    preflight_dir = preflight_dir.resolve(strict=True)
    anchor_path = anchor_path.resolve(strict=True)
    paths = {key: run_dir / name for key, name in EXPECTED_RUN_FILES.items()}
    result = load_json(paths["result"])
    progress_artifact = load_json(paths["progress"])
    watcher = load_json(paths["watcher"])
    manifest = load_json(paths["manifest"])
    receipt = load_json(paths["receipt"])
    pid = load_json(paths["pid"])
    rows = load_jsonl(paths["watcher_log"])
    if any(
        value is None
        for value in (result, progress_artifact, watcher, manifest, receipt, pid)
    ):
        raise ValueError("producer custody JSON artifacts must be objects")
    if not rows:
        raise ValueError("producer watcher log must contain valid JSONL records")
    assert result is not None
    assert progress_artifact is not None
    assert watcher is not None
    assert manifest is not None
    assert receipt is not None
    assert pid is not None

    run_id = Path(str(receipt.get("output"))).parent.name
    local_identity_bound = run_dir.name == run_id or (
        run_dir.name == "retrieved" and run_dir.parent.name == run_id
    )
    result_sha = file_sha256(paths["result"])
    progress = watcher.get("progress")
    command = receipt.get("command")
    content = dict(result)
    recorded_content_sha = content.pop("content_sha256", None)
    science_failures = validate_payload(result)
    live_rows = [
        row
        for row in rows
        if row.get("scientific_pid_alive") is True
        and row.get("scientific_pid") == receipt.get("scientific_pid")
        and row.get("watcher_pid") == receipt.get("watcher_pid")
        and len(spawn_worker_pids(row)) == 6
        and _live_row_has_exact_workers(row, receipt)
        and int(row.get("scientific_process_tree_rss_bytes", 0)) > 0
    ]
    observed_peak_rss = max(
        (int(row.get("scientific_process_tree_rss_bytes", 0)) for row in rows),
        default=0,
    )
    observed_memory = [
        int(row["memory_available_bytes"])
        for row in rows
        if isinstance(row.get("memory_available_bytes"), int)
    ]
    trusted_preflight = receipt.get("trusted_preflight_evidence")
    trusted_checks = (
        trusted_preflight.get("checks")
        if isinstance(trusted_preflight, dict)
        else None
    )
    preflight_validation = _validate_relocated_preflight(
        preflight_dir, expected_head=expected_head
    )
    actual_preflight_hashes = {
        key: file_sha256(preflight_dir / filename)
        for key, filename in PREFLIGHT_FILES.items()
    }
    receipt_blob_hashes = {
        key: git_blob_sha256(expected_head, relative)
        for key, relative in SCRIPT_BLOBS.items()
    }
    claim_fields = (
        "h_pi_computed",
        "h_obs_computed",
        "w24_authorized",
        "learner_authorized",
        "paper2_authorized",
        "paper3_authorized",
    )
    validator_provenance = _validator_provenance()
    validation_runtime_provenance = _validation_runtime_provenance(expected_head)
    anchor_validation = _validate_precompletion_anchor(
        anchor_path,
        run_dir,
        preflight_dir,
        receipt,
        pid,
        expected_head=expected_head,
    )
    expected_output = str(receipt.get("output"))
    checks = {
        "science_payload_valid": not science_failures,
        "result_content_hash": recorded_content_sha == json_sha256(content),
        "result_schema": result.get("schema_version") == EXPECTED_RESULT_SCHEMA,
        "same_commit": result.get("git_head")
        == receipt.get("git_head")
        == pid.get("git_head")
        == expected_head,
        "launch_clean": result.get("launch_git_status_porcelain") == []
        and receipt.get("launch_git_status_porcelain") == [],
        "same_host": result.get("environment", {}).get("hostname")
        == watcher.get("hostname")
        == receipt.get("hostname")
        == EXPECTED_HOSTNAME,
        "runtime_bound": result.get("environment", {}).get("python_executable")
        == EXPECTED_PYTHON,
        "receipt_schema": receipt.get("schema_version")
        == "paper2_switch4_detached_launch_v1",
        "receipt_operation": receipt.get("operation") == EXPECTED_OPERATION,
        "receipt_contract": receipt.get("contract_sha256")
        == result.get("contract_sha256")
        == git_blob_sha256(
            expected_head,
            "contracts/paper2_bottleneck_switch_complexity_screen_v2.json",
        ),
        "receipt_script_blobs": all(
            receipt.get(key) == digest for key, digest in receipt_blob_hashes.items()
        ),
        "receipt_mode": receipt.get("workers") == 6
        and receipt.get("verified_result") is None
        and receipt.get("verified_result_sha256") is None,
        "launch_manifest": _manifest_is_receipt_plan(manifest, receipt),
        "command_bound": command == pid.get("command") == result.get("command"),
        "command_is_frozen_producer": _command_is_frozen_producer(command, run_id),
        "remote_paths_consistent": _remote_paths_consistent(
            receipt, pid, watcher, run_id
        ),
        "local_identity_bound": local_identity_bound,
        "output_paths_bound": _remote_tail_matches(
            receipt.get("output"), run_id, "result.json"
        )
        and _remote_tail_matches(pid.get("output"), run_id, "result.json")
        and _remote_tail_matches(watcher.get("output"), run_id, "result.json"),
        "progress_paths_bound": _remote_tail_matches(
            receipt.get("progress"), run_id, "progress.json"
        )
        and _remote_tail_matches(pid.get("progress"), run_id, "progress.json"),
        "watcher_prestart_claim": receipt.get(
            "watcher_started_before_scientific_process"
        )
        is True,
        "prestart_before_science": _prestart_before_launch(rows, receipt, pid),
        "pid_hash_bound": receipt.get("pid_record_sha256")
        == file_sha256(paths["pid"]),
        "watcher_pid_bound": watcher.get("watcher_pid")
        == receipt.get("watcher_pid")
        == pid.get("watcher_pid"),
        "scientific_pid_bound": watcher.get("scientific_pid")
        == receipt.get("scientific_pid")
        == pid.get("scientific_pid"),
        "six_worker_live_sample": bool(live_rows),
        "watcher_log_bound": _watcher_log_is_bound(
            rows, watcher, receipt, pid, expected_output=expected_output
        ),
        "watcher_log_terminal_matches": rows[-1] == watcher,
        "watcher_terminal": watcher.get("state") == "completed_unverified",
        "scientific_process_closed": watcher.get("scientific_pid_alive") is False,
        "result_exists": watcher.get("result_exists") is True,
        "result_hash_bound": watcher.get("result_sha256") == result_sha,
        "progress_complete": isinstance(progress, dict)
        and progress_artifact == progress
        and watcher.get("progress_sha256") == file_sha256(paths["progress"])
        and progress.get("stage") == "complete"
        and progress.get("completed") == 120
        and progress.get("total") == 120
        and progress.get("output_sha256") == result_sha,
        "stderr_zero": watcher.get("stderr_bytes") == 0
        and paths["stderr"].is_file()
        and paths["stderr"].stat().st_size == 0,
        "stdout_bound": paths["stdout"].is_file()
        and watcher.get("stdout_bytes") == paths["stdout"].stat().st_size,
        "peak_rss_observed": observed_peak_rss > 0
        and watcher.get("peak_scientific_process_tree_rss_bytes")
        == observed_peak_rss,
        "minimum_memory_observed": bool(observed_memory)
        and watcher.get("minimum_memory_available_bytes") == min(observed_memory),
        "trusted_preflight_checks": isinstance(trusted_checks, dict)
        and bool(trusted_checks)
        and all(trusted_checks.values()),
        "trusted_preflight_head": isinstance(trusted_preflight, dict)
        and trusted_preflight.get("git_head") == expected_head
        and trusted_preflight.get("hostname") == EXPECTED_HOSTNAME,
        "trusted_preflight_hashes": isinstance(trusted_preflight, dict)
        and all(
            trusted_preflight.get(key) == digest
            for key, digest in actual_preflight_hashes.items()
        ),
        "preflight_custody_valid": preflight_validation.get("passed") is True,
        "validator_provenance": validator_provenance.get("passed") is True,
        "validation_runtime_provenance": validation_runtime_provenance.get("passed")
        is True,
        "precompletion_anchor": anchor_validation.get("passed") is True,
        "claim_flags_closed": all(result.get(field) is False for field in claim_fields),
        "calibration_only": result.get("locked_tapes_accessed") is False
        and result.get("virgin_tapes_accessed") is False,
    }
    failed = sorted(key for key, passed in checks.items() if not passed)
    return {
        "schema_version": SCHEMA,
        "run_id": run_id,
        "expected_head": expected_head,
        "result_path": str(paths["result"]),
        "result_sha256": result_sha,
        "watcher_path": str(paths["watcher"]),
        "watcher_sha256": file_sha256(paths["watcher"]),
        "progress_path": str(paths["progress"]),
        "progress_sha256": file_sha256(paths["progress"]),
        "launch_manifest_path": str(paths["manifest"]),
        "launch_manifest_sha256": file_sha256(paths["manifest"]),
        "launch_receipt_path": str(paths["receipt"]),
        "launch_receipt_sha256": file_sha256(paths["receipt"]),
        "pid_path": str(paths["pid"]),
        "pid_sha256": file_sha256(paths["pid"]),
        "watcher_log_path": str(paths["watcher_log"]),
        "watcher_log_sha256": file_sha256(paths["watcher_log"]),
        "stdout_sha256": file_sha256(paths["stdout"]),
        "preflight_dir": str(preflight_dir),
        "precompletion_anchor_path": str(anchor_path),
        "precompletion_anchor_sha256": file_sha256(anchor_path),
        "observed_peak_rss_bytes": observed_peak_rss,
        "observed_minimum_memory_available_bytes": min(observed_memory)
        if observed_memory
        else None,
        "science_payload_failures": science_failures,
        "preflight_validation": preflight_validation,
        "validator_provenance": validator_provenance,
        "validation_runtime_provenance": validation_runtime_provenance,
        "precompletion_anchor_validation": anchor_validation,
        "checks": checks,
        "failed_checks": failed,
        "passed": not failed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--preflight-dir", type=Path, required=True)
    parser.add_argument("--precompletion-anchor", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    audit = validate_producer_custody(
        args.run_dir,
        args.preflight_dir,
        args.precompletion_anchor,
        expected_head=args.expected_head,
    )
    rendered = json.dumps(audit, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        output = args.output.resolve(strict=False)
        if output.exists():
            parser.error("output already exists; custody audits are non-overwriting")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered)
    print(rendered, end="")
    return 0 if audit["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
