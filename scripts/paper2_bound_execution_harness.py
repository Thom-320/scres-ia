#!/usr/bin/env python3
"""Fail-closed local/VPS execution wrapper for the frozen Paper-2 M/T/R bound.

The wrapper is deliberately separate from the scientific runner and contract.
It prepares an immutable execution envelope, runs smoke tapes independently or
one exact scientific phase process over its frozen tape block, maintains
heartbeat/per-seed status, and verifies returned checksums.  A launch,
or even a completed remote process, is always labelled NOT_EVIDENCE; scientific
use still requires retrieval, checksum verification, and an independent audit.

The frozen ``reduced_w12``, ``reduced_w16`` and ``w24_audit`` modes produce
pre-gate evidence envelopes without authorizing the full bound. A later full
``scientific`` run uses a separate control authorization and explicit sealing.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parent.parent
# Direct ``python scripts/...`` execution otherwise exposes only ``scripts/``
# on sys.path, while the harness imports sibling modules through the repository
# namespace.  Pin the tracked repository root explicitly for local and VPS CLI
# entrypoints; module imports used by tests already have the same path.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
HARNESS_PATH = Path(__file__).resolve()
DEFAULT_CONTRACT = ROOT / "contracts" / "paper2_bottleneck_primary_bound_v2.json"
DEFAULT_RUNNER = ROOT / "scripts" / "run_paper2_bottleneck_full_frontier.py"
DEFAULT_SMOKE_RUNNER = ROOT / "scripts" / "run_paper2_bottleneck_exact_transducer.py"
FULL_HORIZON_CONTRACT = (
    ROOT / "contracts" / "paper2_bottleneck_full_horizon_bound_v1.json"
)
DEFAULT_HOST = "ovh-agent-lab"
DEFAULT_REMOTE_ROOT = "~/paper2-bound-runs"
SCHEMA = "paper2_bound_execution_harness_v1"
AUTH_SCHEMA = "paper2_bound_execution_authorization_v4"
CONTEXTS = ("equipment_pressure", "interdiction_campaign", "mission_surge")
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
SAFE_REMOTE_RE = re.compile(r"^[A-Za-z0-9_./~+-]+$")
URL_USERINFO_RE = re.compile(r"([A-Za-z][A-Za-z0-9+.-]*://)([^/@\s]+)@")
SOURCE_PREFIXES = (
    "supply_chain/",
    "scripts/",
    "contracts/",
    "requirements",
    "pyproject.toml",
    "setup.cfg",
)
FROZEN_EVIDENCE_MODES = ("reduced_w12", "reduced_w16", "w24_audit")
EXECUTABLE_EVIDENCE_MODES = ("scientific",) + FROZEN_EVIDENCE_MODES
REDUCED_MODE_ROLE = {
    "reduced_w12": "w12_five_tape",
    "reduced_w16": "w16_hard_tape",
}


class HarnessError(RuntimeError):
    """Fail-closed harness error."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return sha256_bytes(payload)


def validate_scientific_environment_payload(
    payload: Any,
    *,
    local_reference: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a preflight environment without accepting package drift.

    A VPS may legitimately have a different SOABI from the preparing Mac.  All
    scientific package versions, Python identity, cache tag and tracked
    requirements hashes must nevertheless match.  The full supplied payload,
    including SOABI, is hashed and later reproduced by the executing runner.
    """

    if not isinstance(payload, dict):
        raise HarnessError("scientific environment snapshot must be a JSON object")
    expected_keys = {
        "python_implementation",
        "python_version",
        "python_cache_tag",
        "python_soabi",
        "packages",
        "requirements_sha256",
        "simpy_source_sha256",
        "environment_sha256",
    }
    if set(payload) != expected_keys:
        raise HarnessError("scientific environment snapshot schema mismatch")
    unhashed = {key: payload[key] for key in expected_keys - {"environment_sha256"}}
    if payload["environment_sha256"] != canonical_json_sha256(unhashed):
        raise HarnessError("scientific environment snapshot digest mismatch")
    if local_reference is not None:
        for key in (
            "python_implementation",
            "python_version",
            "python_cache_tag",
            "packages",
            "requirements_sha256",
            "simpy_source_sha256",
        ):
            if payload[key] != local_reference[key]:
                raise HarnessError(
                    f"scientific execution environment differs from preparation: {key}"
                )
    return dict(payload)


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise HarnessError(f"cannot read valid JSON {path}: {exc}") from exc


def run_capture(
    argv: Sequence[str],
    *,
    cwd: Path | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        list(argv),
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and result.returncode != 0:
        raise HarnessError(
            f"command failed ({result.returncode}): {shlex.join(argv)}: "
            f"{result.stderr.strip()}"
        )
    return result


def relative_to_root(path: Path, root: Path = ROOT) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError as exc:
        raise HarnessError(f"input must be inside repository root: {path}") from exc


def parse_seed_spec(value: str) -> tuple[int, str]:
    if ":" in value:
        seed_text, context = value.split(":", 1)
        if context not in CONTEXTS:
            raise argparse.ArgumentTypeError(f"unknown context {context!r}")
        seed = int(seed_text)
        return seed, context
    seed = int(value)
    start = 1_110_001 if seed >= 1_110_001 else 1_100_001
    return seed, CONTEXTS[(seed - start) % len(CONTEXTS)]


def sanitize_dependency_snapshot(text: str) -> tuple[str, int]:
    redactions = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal redactions
        redactions += 1
        return f"{match.group(1)}<redacted>@"

    return URL_USERINFO_RE.sub(replace, text), redactions


def machine_snapshot() -> dict[str, Any]:
    memory_bytes = None
    try:
        memory_bytes = int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        pass
    disk = shutil.disk_usage(Path.cwd())
    return {
        "captured_at_utc": utc_now(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable_basename": Path(sys.executable).name,
        "cpu_count": os.cpu_count(),
        "physical_memory_bytes": memory_bytes,
        "disk_total_bytes": disk.total,
        "disk_free_bytes": disk.free,
        "hostname_omitted_to_avoid_environment_disclosure": True,
    }


def dependency_snapshot(python: str = sys.executable) -> tuple[str, int]:
    result = run_capture([python, "-m", "pip", "freeze", "--all"], check=True)
    return sanitize_dependency_snapshot(result.stdout)


def git_snapshot(repo_root: Path, critical_paths: Sequence[Path]) -> dict[str, Any]:
    commit = run_capture(["git", "rev-parse", "HEAD"], cwd=repo_root).stdout.strip()
    branch = run_capture(
        ["git", "branch", "--show-current"], cwd=repo_root
    ).stdout.strip()
    status_lines = run_capture(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"], cwd=repo_root
    ).stdout.splitlines()
    source_status = []
    for line in status_lines:
        candidate = line[3:] if len(line) > 3 else line
        if " -> " in candidate:
            candidate = candidate.split(" -> ", 1)[1]
        if any(candidate == prefix.rstrip("/") or candidate.startswith(prefix) for prefix in SOURCE_PREFIXES):
            source_status.append(line)

    tracked: dict[str, bool] = {}
    head_hashes: dict[str, str | None] = {}
    working_hashes: dict[str, str] = {}
    for path in critical_paths:
        rel = relative_to_root(path, repo_root)
        check = run_capture(
            ["git", "ls-files", "--error-unmatch", "--", rel],
            cwd=repo_root,
            check=False,
        )
        tracked[rel] = check.returncode == 0
        working_hashes[rel] = sha256_file(path)
        shown = run_capture(
            ["git", "show", f"{commit}:{rel}"], cwd=repo_root, check=False
        )
        head_hashes[rel] = sha256_bytes(shown.stdout.encode()) if shown.returncode == 0 else None

    inputs_match_head = all(
        tracked[rel] and head_hashes[rel] == working_hashes[rel]
        for rel in tracked
    )
    return {
        "commit": commit,
        "branch": branch,
        "critical_paths_tracked": tracked,
        "critical_path_working_sha256": working_hashes,
        "critical_path_head_sha256": head_hashes,
        "critical_inputs_match_head": inputs_match_head,
        "source_tree_status": source_status,
        "source_tree_clean": not source_status,
        "scientific_source_immutable": inputs_match_head and not source_status,
    }


def _validate_run_id(run_id: str) -> None:
    if not RUN_ID_RE.fullmatch(run_id):
        raise HarnessError(f"unsafe run id: {run_id!r}")


def _validate_remote_value(value: str, label: str) -> None:
    if not SAFE_REMOTE_RE.fullmatch(value):
        raise HarnessError(f"unsafe {label}: {value!r}")


def _remote_shell_path(value: str) -> str:
    """Quote a remote path while preserving an intentional leading ``~/``."""
    _validate_remote_value(value, "remote path")
    if value == "~":
        return '"$HOME"'
    if value.startswith("~/"):
        return '"$HOME"/' + shlex.quote(value[2:])
    return shlex.quote(value)


def _manifest_paths(run_dir: Path) -> dict[str, Path]:
    return {
        "run": run_dir / "run_manifest.json",
        "seeds": run_dir / "seed_manifest.json",
        "commands": run_dir / "command_manifest.json",
        "machine": run_dir / "environment" / "machine.json",
        "dependencies": run_dir / "environment" / "pip_freeze.txt",
        "status": run_dir / "status" / "run_status.json",
        "heartbeat": run_dir / "status" / "heartbeat.json",
        "checksums": run_dir / "artifact_checksums.json",
    }


def _assert_empty_or_missing(run_dir: Path) -> None:
    if run_dir.exists() and any(run_dir.iterdir()):
        raise HarnessError(f"run directory is not empty: {run_dir}")


def _seed_rows(seed_specs: Sequence[tuple[int, str]], split: str, weeks: int) -> list[dict[str, Any]]:
    if len({seed for seed, _ in seed_specs}) != len(seed_specs):
        raise HarnessError("seed manifest contains duplicate seeds")
    return [
        {
            "seed": int(seed),
            "context": context,
            "split": split,
            "weeks": int(weeks),
            "opened_status": "BURNED_DEVELOPMENT_OR_CORRECTIVE",
        }
        for seed, context in seed_specs
    ]


def _command_rows(
    seeds: Sequence[dict[str, Any]],
    *,
    runner_rel: str,
    runner_workers: int,
) -> list[dict[str, Any]]:
    rows = []
    for row in seeds:
        seed_id = f"seed_{row['seed']}_{row['context']}"
        output_rel = f"artifacts/{seed_id}/result.json"
        argv = [
            "{python}",
            f"{{repo_root}}/{runner_rel}",
            "--weeks",
            str(row["weeks"]),
            "--seed",
            f"{row['seed']}:{row['context']}",
            "--split",
            row["split"],
            "--workers",
            str(runner_workers),
            "--max-calendars",
            "13",
            "--non-scientific-smoke",
            "--output",
            f"{{run_dir}}/{output_rel}",
        ]
        rows.append(
            {
                "seed_id": seed_id,
                "seed": row["seed"],
                "context": row["context"],
                "argv_template": argv,
                "argv_template_sha256": canonical_json_sha256(argv),
                "output_relative": output_rel,
                "stdout_relative": f"logs/{seed_id}.stdout.log",
                "stderr_relative": f"logs/{seed_id}.stderr.log",
            }
        )
    return rows


def _frozen_evidence_profile(
    contract: dict[str, Any], mode: str
) -> dict[str, Any]:
    """Read one immutable pre-gate execution profile from the primary contract."""
    if mode not in FROZEN_EVIDENCE_MODES:
        raise HarnessError(f"unknown frozen evidence mode: {mode}")
    role = REDUCED_MODE_ROLE.get(mode, "w24_profile_state_audit")
    profile = contract.get("reduced_horizon_certification", {}).get(role)
    if not isinstance(profile, dict):
        raise HarnessError(f"contract lacks frozen profile {role}")
    try:
        weeks = int(profile["weeks"])
        split = str(profile["split"])
        seeds = [(int(seed), str(context)) for seed, context in profile["seed_context"]]
        output_path = str(profile["output_path"])
    except (KeyError, TypeError, ValueError) as exc:
        raise HarnessError(f"malformed frozen profile {role}") from exc
    if not split or not seeds or any(context not in CONTEXTS for _, context in seeds):
        raise HarnessError(f"invalid seed/context/split in frozen profile {role}")
    output = Path(output_path)
    if output.is_absolute() or ".." in output.parts or output.parts[:2] != (
        "results",
        "paper2_bottleneck",
    ):
        raise HarnessError(f"invalid contractual output path for {role}")
    expected_weeks = 12 if mode == "reduced_w12" else 16 if mode == "reduced_w16" else 24
    expected_count = 5 if mode == "reduced_w12" else 1
    if weeks != expected_weeks or len(seeds) != expected_count:
        raise HarnessError(f"frozen profile scope mismatch for {role}")
    return {
        "mode": mode,
        "role": role,
        "weeks": weeks,
        "split": split,
        "seeds": seeds,
        "output_path": output_path,
        "runner": str(
            (DEFAULT_SMOKE_RUNNER if mode.startswith("reduced_") else DEFAULT_RUNNER)
            .relative_to(ROOT)
        ),
        "result_contract": str(
            (FULL_HORIZON_CONTRACT if mode.startswith("reduced_") else DEFAULT_CONTRACT)
            .relative_to(ROOT)
        ),
    }


def _frozen_evidence_command_row(
    seeds: Sequence[dict[str, Any]],
    *,
    profile: dict[str, Any],
    runner_rel: str,
    runner_workers: int,
) -> list[dict[str, Any]]:
    """Build one contract-bound command for W12, W16, or the W24 audit."""
    role = str(profile["role"])
    output_rel = str(profile["output_path"])
    progress_rel = f"status/{role}.progress.json"
    checkpoint_rel = f"artifacts/{role}/checkpoints"
    argv = ["{python}", f"{{repo_root}}/{runner_rel}"]
    if profile["mode"].startswith("reduced_"):
        argv.extend(["--weeks", str(profile["weeks"])])
        for row in seeds:
            argv.extend(["--seed", f"{row['seed']}:{row['context']}"])
        argv.extend(
            [
                "--split",
                str(profile["split"]),
                "--workers",
                str(runner_workers),
                "--progress",
                f"{{run_dir}}/{progress_rel}",
                "--output",
                f"{{run_dir}}/{output_rel}",
            ]
        )
        runner_mode = "reduced_exact_transducer_certification"
    else:
        argv.extend(
            [
                "--weeks",
                "24",
                "--write-w24-audit",
                f"{{run_dir}}/{output_rel}",
                "--build-workers",
                str(runner_workers),
                "--checkpoint-dir",
                f"{{run_dir}}/{checkpoint_rel}",
                "--progress",
                f"{{run_dir}}/{progress_rel}",
            ]
        )
        runner_mode = "w24_profile_state_audit"
    return [
        {
            "job_id": role,
            "runner_mode": runner_mode,
            "evidence_role": role,
            "covered_seed_ids": [
                f"seed_{row['seed']}_{row['context']}" for row in seeds
            ],
            "argv_template": argv,
            "argv_template_sha256": canonical_json_sha256(argv),
            "output_relative": output_rel,
            "checkpoint_relative": (
                checkpoint_rel if profile["mode"] == "w24_audit" else None
            ),
            "progress_relative": progress_rel,
            "stdout_relative": f"logs/{role}.stdout.log",
            "stderr_relative": f"logs/{role}.stderr.log",
        }
    ]


def _frontier_command_row(
    seeds: Sequence[dict[str, Any]],
    *,
    runner_rel: str,
    phase: str,
    weeks: int,
    calibration_result_rel: str | None,
    batch_size: int,
    max_contenders: int,
    build_workers: int,
) -> list[dict[str, Any]]:
    """Build one exact-frontier phase command covering the frozen seed block."""
    job_id = f"frontier_{phase}"
    output_rel = f"artifacts/{job_id}/result.json"
    runner_manifest_rel = f"artifacts/{job_id}/runner_manifest.json"
    checkpoint_rel = f"artifacts/{job_id}/checkpoints"
    progress_rel = f"status/{job_id}.progress.json"
    argv = [
        "{python}",
        f"{{repo_root}}/{runner_rel}",
        "--phase",
        phase,
        "--weeks",
        str(weeks),
        "--batch-size",
        str(batch_size),
        "--max-contenders",
        str(max_contenders),
        "--authorization",
        "{run_dir}/authorization.json",
        "--build-workers",
        str(build_workers),
        "--checkpoint-dir",
        f"{{run_dir}}/{checkpoint_rel}",
        "--output",
        f"{{run_dir}}/{output_rel}",
        "--manifest",
        f"{{run_dir}}/{runner_manifest_rel}",
        "--progress",
        f"{{run_dir}}/{progress_rel}",
    ]
    if phase == "locked":
        if calibration_result_rel is None:
            raise HarnessError("locked exact-frontier phase requires a calibration result")
        argv.extend(
            ["--calibration-result", f"{{repo_root}}/{calibration_result_rel}"]
        )
    return [
        {
            "job_id": job_id,
            "runner_mode": "exact_primary_frontier_phase",
            "phase": phase,
            "covered_seed_ids": [
                f"seed_{row['seed']}_{row['context']}" for row in seeds
            ],
            "argv_template": argv,
            "argv_template_sha256": canonical_json_sha256(argv),
            "output_relative": output_rel,
            "runner_manifest_relative": runner_manifest_rel,
            "checkpoint_relative": checkpoint_rel,
            "progress_relative": progress_rel,
            "stdout_relative": f"logs/{job_id}.stdout.log",
            "stderr_relative": f"logs/{job_id}.stderr.log",
        }
    ]


def _frozen_phase_seed_specs(phase: str) -> list[tuple[int, str]]:
    if phase == "calibration":
        start, end, offset = 1_100_001, 1_100_060, 1_100_001
    elif phase == "locked":
        start, end, offset = 1_110_002, 1_110_120, 1_110_001
    else:
        raise HarnessError("scientific exact-frontier phase must be calibration or locked")
    return [
        (seed, CONTEXTS[(seed - offset) % len(CONTEXTS)])
        for seed in range(start, end + 1)
    ]


def _validate_scientific_seed_scope(seeds: Sequence[dict[str, Any]], weeks: int) -> None:
    if weeks != 24:
        raise HarnessError("scientific bound execution requires exactly 24 weeks")
    for row in seeds:
        seed = int(row["seed"])
        calibration = 1_100_001 <= seed <= 1_100_060
        locked = 1_110_002 <= seed <= 1_110_120
        if not (calibration or locked):
            raise HarnessError(
                f"scientific seed {seed} is outside frozen calibration/locked blocks "
                "or is the excluded algorithm-development seed"
            )


def _tracked_archive_artifact(
    repo_root: Path,
    relative: Any,
    expected_sha256: Any,
) -> tuple[Path, dict[str, Any]]:
    """Load an artifact only if its exact bytes exist in the transported commit."""
    path = (repo_root / str(relative)).resolve()
    try:
        rel = str(path.relative_to(repo_root.resolve()))
    except ValueError as exc:
        raise HarnessError("authorization artifact escapes repository root") from exc
    if not path.is_file():
        raise HarnessError(f"authorization artifact missing: {rel}")
    if sha256_file(path) != expected_sha256:
        raise HarnessError(f"authorization artifact hash mismatch: {rel}")
    tracked = run_capture(
        ["git", "ls-files", "--error-unmatch", "--", rel],
        cwd=repo_root,
        check=False,
    )
    if tracked.returncode != 0:
        raise HarnessError(f"authorization artifact is not tracked: {rel}")
    head = run_capture(["git", "show", f"HEAD:{rel}"], cwd=repo_root, check=False)
    if head.returncode != 0 or sha256_bytes(head.stdout.encode()) != expected_sha256:
        raise HarnessError(f"authorization artifact differs from HEAD: {rel}")
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise HarnessError(f"authorization artifact is not a JSON object: {rel}")
    return path, payload


def _validate_authorization(
    authorization: dict[str, Any],
    *,
    run_manifest: dict[str, Any],
    seed_manifest_sha256: str,
    command_manifest_sha256: str,
    repo_root: Path = ROOT,
) -> None:
    required_true = (
        "execution_authorized",
        "primary_bound_batch_authorized",
        "reduced_horizon_key_v3_certified",
        "full_horizon_primary_acceleration_authorized",
        "primary_frontier_exactness_required",
        "original_runner_replay_required",
        "resource_semantics_frozen",
    )
    failures = []
    if authorization.get("schema_version") != AUTH_SCHEMA:
        failures.append("authorization schema mismatch")
    scope = authorization.get("authorization_scope")
    if scope not in {"primary_bound_only", "full_guardrail_frontier"}:
        failures.append("authorization_scope must be primary_bound_only or full_guardrail_frontier")
    for key in required_true:
        if authorization.get(key) is not True:
            failures.append(f"{key} is not true")
    if authorization.get("key_schema_version") != "paper2_bottleneck_semantic_markov_key_v3":
        failures.append("authorization does not require semantic Markov key v3")
    if scope == "full_guardrail_frontier" and authorization.get("full_guardrail_label_certified") is not True:
        failures.append("full_guardrail_frontier requires full_guardrail_label_certified")
    if authorization.get("material_hpi_promotion_authorized") is not False:
        failures.append("material_hpi_promotion_authorized must be false")
    if authorization.get("learner_authorized") is not False:
        failures.append("learner_authorized must be false")
    if authorization.get("paper3_authorized") is not False:
        failures.append("paper3_authorized must be false")
    expected = {
        "git_commit": run_manifest["git"]["commit"],
        "contract_sha256": run_manifest["inputs"]["contract_sha256"],
        "runner_sha256": run_manifest["inputs"]["runner_sha256"],
        "harness_sha256": run_manifest["inputs"]["harness_sha256"],
        "seed_manifest_sha256": seed_manifest_sha256,
        "command_manifest_sha256": command_manifest_sha256,
        "calibration_result_sha256": run_manifest["inputs"].get(
            "calibration_result_sha256"
        ),
        "environment_sha256": run_manifest["inputs"].get(
            "environment_sha256"
        ),
    }
    for key, value in expected.items():
        if authorization.get(key) != value:
            failures.append(f"authorization {key} mismatch")
    if not str(authorization.get("authorized_by", "")).strip():
        failures.append("authorized_by is empty")
    certifications = authorization.get("reduced_horizon_certification_artifacts", [])
    roles = {row.get("role") for row in certifications if isinstance(row, dict)}
    if roles != {"w12_five_tape", "w16_hard_tape"}:
        failures.append("reduced-horizon certification roles must be exactly W12-five-tape and W16-hard-tape")
    for row in certifications if isinstance(certifications, list) else []:
        try:
            _path, payload = _tracked_archive_artifact(
                repo_root, row["path"], row.get("sha256")
            )
            from scripts.run_paper2_bottleneck_exact_transducer import (
                validate_reduced_certification_payload,
            )
            failures.extend(
                validate_reduced_certification_payload(
                    payload,
                    str(row.get("role")),
                    expected_environment_sha256=run_manifest["inputs"].get(
                        "environment_sha256"
                    ),
                )
            )
            if row.get("source_git_commit") != payload.get("provenance", {}).get(
                "git_commit"
            ):
                failures.append(
                    f"authorization does not pin reduced source commit: {row.get('role')}"
                )
        except (KeyError, HarnessError, TypeError) as exc:
            failures.append(f"invalid certification artifact {row!r}: {exc}")
    audit_row = authorization.get("w24_profile_state_audit")
    if not isinstance(audit_row, dict):
        failures.append("tracked W24 profile/state audit is missing")
    else:
        try:
            _path, audit = _tracked_archive_artifact(
                repo_root, audit_row["path"], audit_row.get("sha256")
            )
            from scripts.run_paper2_bottleneck_full_frontier import (
                validate_w24_profile_state_audit_payload,
            )
            failures.extend(
                validate_w24_profile_state_audit_payload(
                    audit,
                    expected_environment_sha256=run_manifest["inputs"].get(
                        "environment_sha256"
                    ),
                )
            )
            if audit_row.get("source_git_commit") != audit.get("git_head"):
                failures.append("W24 audit source commit is not pinned")
        except (KeyError, HarnessError, TypeError) as exc:
            failures.append(f"invalid W24 profile/state audit: {exc}")
    if failures:
        raise HarnessError("scientific authorization failed closed: " + "; ".join(failures))


def _authorization_template(
    run_manifest: dict[str, Any],
    *,
    seed_manifest_sha256: str,
    command_manifest_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": AUTH_SCHEMA,
        "authorization_scope": "primary_bound_only",
        "execution_authorized": False,
        "primary_bound_batch_authorized": False,
        "key_schema_version": "paper2_bottleneck_semantic_markov_key_v3",
        "reduced_horizon_key_v3_certified": False,
        "full_horizon_primary_acceleration_authorized": False,
        "reduced_horizon_certification_artifacts": [
            {
                "role": "w12_five_tape",
                "path": "results/paper2_bottleneck/exact_transducer_certification_w12.json",
                "sha256": "FILL_AFTER_VERIFICATION",
                "source_git_commit": "FILL_FROM_CERTIFICATE_PROVENANCE",
            },
            {
                "role": "w16_hard_tape",
                "path": "results/paper2_bottleneck/exact_transducer_certification_w16_hard.json",
                "sha256": "FILL_AFTER_VERIFICATION",
                "source_git_commit": "FILL_FROM_CERTIFICATE_PROVENANCE",
            },
        ],
        "w24_profile_state_audit": {
            "path": "results/paper2_bottleneck/w24_profile_state_audit.json",
            "sha256": "FILL_AFTER_TRACKED_AUDIT_VERIFICATION",
            "source_git_commit": "FILL_FROM_AUDIT_GIT_HEAD",
        },
        "primary_frontier_exactness_required": True,
        "original_runner_replay_required": True,
        "resource_semantics_frozen": False,
        "full_guardrail_label_certified": False,
        "material_hpi_promotion_authorized": False,
        "learner_authorized": False,
        "paper3_authorized": False,
        "git_commit": run_manifest["git"]["commit"],
        "contract_sha256": run_manifest["inputs"]["contract_sha256"],
        "runner_sha256": run_manifest["inputs"]["runner_sha256"],
        "harness_sha256": run_manifest["inputs"]["harness_sha256"],
        "seed_manifest_sha256": seed_manifest_sha256,
        "command_manifest_sha256": command_manifest_sha256,
        "calibration_result_sha256": run_manifest["inputs"].get(
            "calibration_result_sha256"
        ),
        "environment_sha256": run_manifest["inputs"]["environment_sha256"],
        "authorized_by": "",
        "authorized_at_utc": "",
        "scope_note": (
            "primary_bound_only may close the family only when H_PI UCB95 < 0.01; "
            "material H_PI is diagnostic and cannot promote learning or Paper 2"
        ),
    }


def prepare_run(
    *,
    run_dir: Path,
    run_id: str,
    mode: str,
    contract_path: Path,
    runner_path: Path,
    seeds: Sequence[tuple[int, str]],
    split: str,
    weeks: int,
    runner_workers: int,
    heartbeat_interval: float,
    authorization_path: Path | None = None,
    repo_root: Path = ROOT,
    phase: str | None = None,
    calibration_result_path: Path | None = None,
    batch_size: int = 65_536,
    max_contenders: int = 100_000,
    scientific_environment_path: Path | None = None,
) -> dict[str, Any]:
    _validate_run_id(run_id)
    if mode not in {"dry-run", "smoke", "scientific", *FROZEN_EVIDENCE_MODES}:
        raise HarnessError(f"unknown mode: {mode}")
    if mode == "smoke" and (weeks > 4 or len(seeds) != 1):
        raise HarnessError("smoke mode is limited to one burned tape and at most W4")
    if runner_workers < 1:
        raise HarnessError("runner workers must be positive")
    if heartbeat_interval <= 0:
        raise HarnessError("heartbeat interval must be positive")
    if batch_size < 1 or max_contenders < 1:
        raise HarnessError("batch size and max contenders must be positive")
    for required in (contract_path, runner_path, HARNESS_PATH):
        if not required.is_file():
            raise HarnessError(f"required input missing: {required}")

    _assert_empty_or_missing(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    paths = _manifest_paths(run_dir)
    contract = load_json(contract_path)
    frozen_profile: dict[str, Any] | None = None
    if mode in EXECUTABLE_EVIDENCE_MODES:
        if contract.get("contract_id") != "paper2_bottleneck_primary_bound_v2":
            raise HarnessError("evidence execution requires paper2_bottleneck_primary_bound_v2")
        if contract.get("acceleration_proof", {}).get("required_key_schema") != "paper2_bottleneck_semantic_markov_key_v3":
            raise HarnessError("primary-bound contract does not require semantic Markov key v3")
        if contract.get("decision_rules", {}).get("learner_authorized") is not False:
            raise HarnessError("primary-bound contract must keep learner authorization false")
        if contract_path.resolve() != DEFAULT_CONTRACT.resolve():
            raise HarnessError("evidence execution requires the canonical primary-bound contract")
    if mode in FROZEN_EVIDENCE_MODES:
        frozen_profile = _frozen_evidence_profile(contract, mode)
        expected_runner = repo_root / frozen_profile["runner"]
        if runner_path.resolve() != expected_runner.resolve():
            raise HarnessError(f"{mode} requires its frozen runner")
        if (
            list(seeds) != frozen_profile["seeds"]
            or split != frozen_profile["split"]
            or weeks != frozen_profile["weeks"]
        ):
            raise HarnessError(f"{mode} seed/context/split/weeks differ from contract")
        if phase is not None or calibration_result_path is not None:
            raise HarnessError(f"{mode} cannot consume a frontier phase or calibration result")
        if authorization_path is not None:
            raise HarnessError(f"{mode} does not consume frontier authorization")
    if mode == "scientific":
        if runner_path.resolve() != DEFAULT_RUNNER.resolve():
            raise HarnessError(
                "scientific primary-bound mode requires the exact full-frontier runner"
            )
        if phase not in {"calibration", "locked"}:
            raise HarnessError(
                "scientific primary-bound mode requires --phase calibration or locked"
            )
        if split != phase:
            raise HarnessError("scientific split must equal the frozen frontier phase")
        expected_seed_specs = _frozen_phase_seed_specs(phase)
        if list(seeds) != expected_seed_specs:
            raise HarnessError(
                f"scientific {phase} phase requires its complete frozen seed block in order"
            )
        if phase == "locked" and (
            calibration_result_path is None or not calibration_result_path.is_file()
        ):
            raise HarnessError(
                "locked exact-frontier mode requires the certified calibration result"
            )
        if phase == "calibration" and calibration_result_path is not None:
            raise HarnessError("calibration phase must not consume a prior calibration result")
    seed_rows = _seed_rows(seeds, split, weeks)
    if mode == "scientific":
        _validate_scientific_seed_scope(seed_rows, weeks)

    runner_rel = relative_to_root(runner_path, repo_root)
    contract_rel = relative_to_root(contract_path, repo_root)
    harness_rel = relative_to_root(HARNESS_PATH, repo_root)
    calibration_result_rel = (
        relative_to_root(calibration_result_path, repo_root)
        if calibration_result_path is not None
        else None
    )
    if mode == "scientific":
        assert phase is not None
        commands = _frontier_command_row(
            seed_rows,
            runner_rel=runner_rel,
            phase=phase,
            weeks=weeks,
            calibration_result_rel=calibration_result_rel,
            batch_size=batch_size,
            max_contenders=max_contenders,
            build_workers=runner_workers,
        )
        runner_mode = "exact_primary_frontier_phase"
    elif mode in FROZEN_EVIDENCE_MODES:
        assert frozen_profile is not None
        commands = _frozen_evidence_command_row(
            seed_rows,
            profile=frozen_profile,
            runner_rel=runner_rel,
            runner_workers=runner_workers,
        )
        runner_mode = commands[0]["runner_mode"]
    else:
        commands = _command_rows(
            seed_rows, runner_rel=runner_rel, runner_workers=runner_workers
        )
        runner_mode = "per_seed_transducer"
    seed_manifest = {
        "schema_version": SCHEMA,
        "run_id": run_id,
        "mode": mode,
        "seed_count": len(seed_rows),
        "seeds": seed_rows,
    }
    command_manifest = {
        "schema_version": SCHEMA,
        "run_id": run_id,
        "mode": mode,
        "shell_execution": False,
        "runner_mode": runner_mode,
        "commands": commands,
    }
    atomic_write_json(paths["seeds"], seed_manifest)
    atomic_write_json(paths["commands"], command_manifest)
    seed_hash = sha256_file(paths["seeds"])
    command_hash = sha256_file(paths["commands"])

    deps, redactions = dependency_snapshot(sys.executable)
    paths["dependencies"].parent.mkdir(parents=True, exist_ok=True)
    paths["dependencies"].write_text(deps)
    atomic_write_json(paths["machine"], machine_snapshot())
    critical_paths = [contract_path, runner_path, HARNESS_PATH]
    result_contract_path: Path | None = None
    if frozen_profile is not None:
        result_contract_path = repo_root / frozen_profile["result_contract"]
        critical_paths.append(result_contract_path)
    if calibration_result_path is not None:
        critical_paths.append(calibration_result_path)
    git = git_snapshot(repo_root, critical_paths)
    from scripts.run_paper2_bottleneck_exact_transducer import (
        certification_environment,
    )
    local_scientific_environment = certification_environment()
    if scientific_environment_path is not None:
        if mode not in EXECUTABLE_EVIDENCE_MODES:
            raise HarnessError(
                "an external scientific environment is only valid for evidence modes"
            )
        if not scientific_environment_path.is_file():
            raise HarnessError("scientific environment snapshot is missing")
        scientific_environment = validate_scientific_environment_payload(
            load_json(scientific_environment_path),
            local_reference=local_scientific_environment,
        )
        scientific_environment_source = "provided_remote_preflight"
        scientific_environment_source_sha256 = sha256_file(
            scientific_environment_path
        )
    else:
        scientific_environment = local_scientific_environment
        scientific_environment_source = "preparation_runtime"
        scientific_environment_source_sha256 = None
    manifest = {
        "schema_version": SCHEMA,
        "run_id": run_id,
        "created_at_utc": utc_now(),
        "mode": mode,
        "scientific_run": mode in EXECUTABLE_EVIDENCE_MODES,
        "evidence": False,
        "evidence_status": "PREPARED_NOT_SUBMITTED_NOT_EVIDENCE",
        "contract_id": contract.get("contract_id"),
        "contract_status": contract.get("status"),
        "git": git,
        "inputs": {
            "contract_relative": contract_rel,
            "contract_sha256": sha256_file(contract_path),
            "runner_relative": runner_rel,
            "runner_sha256": sha256_file(runner_path),
            "harness_relative": harness_rel,
            "harness_sha256": sha256_file(HARNESS_PATH),
            "seed_manifest_sha256": seed_hash,
            "command_manifest_sha256": command_hash,
            "dependency_snapshot_sha256": sha256_file(paths["dependencies"]),
            "machine_snapshot_sha256": sha256_file(paths["machine"]),
            "environment": scientific_environment,
            "environment_sha256": scientific_environment["environment_sha256"],
            "environment_source": scientific_environment_source,
            "environment_source_sha256": scientific_environment_source_sha256,
            "dependency_snapshot_redactions": redactions,
            "calibration_result_relative": calibration_result_rel,
            "calibration_result_sha256": (
                sha256_file(calibration_result_path)
                if calibration_result_path is not None
                else None
            ),
            "result_contract_relative": (
                relative_to_root(result_contract_path, repo_root)
                if result_contract_path is not None
                else None
            ),
            "result_contract_sha256": (
                sha256_file(result_contract_path)
                if result_contract_path is not None
                else None
            ),
        },
        "execution": {
            "weeks": weeks,
            "split": split,
            "phase": phase,
            "runner_mode": runner_mode,
            "runner_workers_per_seed": runner_workers,
            "batch_size": batch_size,
            "max_contenders": max_contenders,
            "heartbeat_interval_seconds": heartbeat_interval,
            "one_process_per_seed": runner_mode == "per_seed_transducer",
            "one_process_per_frozen_phase": runner_mode == "exact_primary_frontier_phase",
            "one_process_per_frozen_evidence_profile": mode
            in FROZEN_EVIDENCE_MODES,
            "frozen_evidence_profile": frozen_profile,
            "full_batch_launched": False,
            "authorization_scope": (
                "UNSEALED_PRIMARY_BOUND_ONLY"
                if mode == "scientific"
                else "PRE_GATE_CERTIFICATION_ONLY"
                if mode in FROZEN_EVIDENCE_MODES
                else "NONSCIENTIFIC"
            ),
            "primary_frontier_must_be_exact": mode == "scientific",
            "original_runner_replay_required": mode == "scientific",
            "resource_semantics_required": mode == "scientific",
            "material_hpi_promotion": "PROHIBITED",
        },
        "remote": {
            "default_ssh_alias": DEFAULT_HOST,
            "submission_is_evidence": False,
            "completed_remote_run_before_retrieval_is_evidence": False,
        },
        "required_chain_of_custody": [
            "completed",
            "retrieved",
            "artifact checksums verified",
            "independent scientific audit",
        ],
    }
    sealed = mode != "scientific"
    if mode in EXECUTABLE_EVIDENCE_MODES:
        if not git["scientific_source_immutable"]:
            raise HarnessError(
                "evidence preparation failed closed: critical source is not tracked, "
                "clean, and identical to HEAD"
            )
    if mode == "scientific":
        template = _authorization_template(
            manifest,
            seed_manifest_sha256=seed_hash,
            command_manifest_sha256=command_hash,
        )
        atomic_write_json(run_dir / "authorization_required_template.json", template)
        if authorization_path is not None:
            authorization = load_json(authorization_path)
            _validate_authorization(
                authorization,
                run_manifest=manifest,
                seed_manifest_sha256=seed_hash,
                command_manifest_sha256=command_hash,
                repo_root=repo_root,
            )
            auth_copy = run_dir / "authorization.json"
            auth_copy.write_bytes(authorization_path.read_bytes())
            manifest["inputs"]["authorization_sha256"] = sha256_file(auth_copy)
            manifest["execution"]["authorization_scope"] = authorization["authorization_scope"]
            sealed = True

    manifest["execution"]["sealed_for_execution"] = sealed
    if mode == "scientific" and not sealed:
        manifest["evidence_status"] = "AWAITING_AUTHORIZATION_NOT_EXECUTABLE_NOT_EVIDENCE"

    atomic_write_json(paths["run"], manifest)
    initial_status = {
        "schema_version": SCHEMA,
        "run_id": run_id,
        "state": "prepared" if sealed else "awaiting_authorization",
        "updated_at_utc": utc_now(),
        "evidence": False,
        "evidence_status": manifest["evidence_status"],
        "submitted": False,
        "sealed_for_execution": sealed,
        "completed_seed_count": 0,
        "seed_count": len(seed_rows),
    }
    atomic_write_json(paths["status"], initial_status)
    atomic_write_json(
        paths["heartbeat"],
        {
            "schema_version": SCHEMA,
            "run_id": run_id,
            "state": "prepared" if sealed else "awaiting_authorization",
            "last_activity_utc": utc_now(),
            "evidence": False,
        },
    )
    return manifest


def _validate_prepared_inputs(
    run_dir: Path,
    repo_root: Path,
    *,
    allow_unsealed_scientific: bool = False,
    allow_preflight_platform_environment: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    paths = _manifest_paths(run_dir)
    run_manifest = load_json(paths["run"])
    seed_manifest = load_json(paths["seeds"])
    command_manifest = load_json(paths["commands"])
    if run_manifest.get("schema_version") != SCHEMA:
        raise HarnessError("run manifest schema mismatch")
    if run_manifest.get("run_id") != seed_manifest.get("run_id") or run_manifest.get("run_id") != command_manifest.get("run_id"):
        raise HarnessError("run/seed/command run ids differ")
    expected = run_manifest["inputs"]
    from scripts.run_paper2_bottleneck_exact_transducer import (
        certification_environment,
    )
    live_environment = certification_environment()
    if allow_preflight_platform_environment:
        if expected.get("environment_source") == "provided_remote_preflight":
            validate_scientific_environment_payload(
                expected.get("environment"),
                local_reference=live_environment,
            )
        elif expected.get("environment") != live_environment:
            raise HarnessError(
                "immutable environment identity changed since preparation"
            )
    else:
        if expected.get("environment") != live_environment:
            raise HarnessError("immutable environment identity changed since preparation")
        if expected.get("environment_sha256") != live_environment.get(
            "environment_sha256"
        ):
            raise HarnessError("environment digest changed since preparation")
    checks = {
        "contract_sha256": sha256_file(repo_root / expected["contract_relative"]),
        "runner_sha256": sha256_file(repo_root / expected["runner_relative"]),
        "harness_sha256": sha256_file(repo_root / expected["harness_relative"]),
        "seed_manifest_sha256": sha256_file(paths["seeds"]),
        "command_manifest_sha256": sha256_file(paths["commands"]),
    }
    result_contract_relative = expected.get("result_contract_relative")
    if result_contract_relative is not None:
        checks["result_contract_sha256"] = sha256_file(
            repo_root / result_contract_relative
        )
    for stem in ("calibration_result",):
        relative = expected.get(f"{stem}_relative")
        recorded_hash = expected.get(f"{stem}_sha256")
        if relative is None:
            if recorded_hash is not None:
                raise HarnessError(f"{stem} has a hash but no immutable path")
            continue
        checks[f"{stem}_sha256"] = sha256_file(repo_root / relative)
    for key, actual in checks.items():
        if actual != expected[key]:
            raise HarnessError(f"immutable input check failed: {key}")
    mode = run_manifest.get("mode")
    if mode in EXECUTABLE_EVIDENCE_MODES:
        critical_paths = [
            repo_root / expected["contract_relative"],
            repo_root / expected["runner_relative"],
            repo_root / expected["harness_relative"],
        ]
        if result_contract_relative is not None:
            critical_paths.append(repo_root / result_contract_relative)
        calibration_relative = expected.get("calibration_result_relative")
        if calibration_relative is not None:
            critical_paths.append(repo_root / calibration_relative)
        live_git = git_snapshot(repo_root, critical_paths)
        if live_git["commit"] != run_manifest.get("git", {}).get("commit"):
            raise HarnessError("prepared source commit differs from live HEAD")
        if live_git["scientific_source_immutable"] is not True:
            raise HarnessError("evidence execution requires clean tracked HEAD inputs")
    if mode in FROZEN_EVIDENCE_MODES:
        contract = load_json(repo_root / expected["contract_relative"])
        profile = _frozen_evidence_profile(contract, mode)
        if canonical_json_sha256(
            run_manifest.get("execution", {}).get("frozen_evidence_profile")
        ) != canonical_json_sha256(profile):
            raise HarnessError("frozen evidence profile differs from contract")
        expected_seed_rows = _seed_rows(
            profile["seeds"], profile["split"], profile["weeks"]
        )
        if seed_manifest.get("seeds") != expected_seed_rows:
            raise HarnessError("frozen evidence seed manifest differs from contract")
        expected_commands = _frozen_evidence_command_row(
            expected_seed_rows,
            profile=profile,
            runner_rel=expected["runner_relative"],
            runner_workers=int(
                run_manifest["execution"]["runner_workers_per_seed"]
            ),
        )
        if command_manifest.get("commands") != expected_commands:
            raise HarnessError("frozen evidence command manifest differs from contract")
        if command_manifest.get("runner_mode") != expected_commands[0]["runner_mode"]:
            raise HarnessError("frozen evidence runner mode mismatch")
    if run_manifest["mode"] == "scientific":
        if run_manifest.get("execution", {}).get("sealed_for_execution") is not True:
            if allow_unsealed_scientific:
                return run_manifest, seed_manifest, command_manifest
            raise HarnessError("scientific manifest is unsealed and cannot execute")
        authorization = load_json(run_dir / "authorization.json")
        _validate_authorization(
            authorization,
            run_manifest=run_manifest,
            seed_manifest_sha256=checks["seed_manifest_sha256"],
            command_manifest_sha256=checks["command_manifest_sha256"],
            repo_root=repo_root,
        )
    return run_manifest, seed_manifest, command_manifest


def seal_run(
    *,
    run_dir: Path,
    authorization_path: Path,
    repo_root: Path = ROOT,
) -> dict[str, Any]:
    """Seal a previously prepared scientific run without regenerating manifests."""
    run_manifest, _seed_manifest, _command_manifest = _validate_prepared_inputs(
        run_dir,
        repo_root,
        allow_unsealed_scientific=True,
        allow_preflight_platform_environment=True,
    )
    if run_manifest.get("mode") != "scientific":
        raise HarnessError("seal is only valid for a scientific frontier run")
    if run_manifest.get("execution", {}).get("sealed_for_execution") is True:
        raise HarnessError("scientific run is already sealed")
    paths = _manifest_paths(run_dir)
    authorization = load_json(authorization_path)
    _validate_authorization(
        authorization,
        run_manifest=run_manifest,
        seed_manifest_sha256=sha256_file(paths["seeds"]),
        command_manifest_sha256=sha256_file(paths["commands"]),
        repo_root=repo_root,
    )
    auth_copy = run_dir / "authorization.json"
    if auth_copy.exists():
        raise HarnessError("authorization copy already exists")
    auth_copy.write_bytes(authorization_path.read_bytes())
    run_manifest["inputs"]["authorization_sha256"] = sha256_file(auth_copy)
    run_manifest["execution"]["authorization_scope"] = authorization[
        "authorization_scope"
    ]
    run_manifest["execution"]["sealed_for_execution"] = True
    run_manifest["evidence_status"] = "PREPARED_SEALED_NOT_SUBMITTED_NOT_EVIDENCE"
    atomic_write_json(paths["run"], run_manifest)
    status = load_json(paths["status"])
    status.update(
        {
            "state": "prepared",
            "updated_at_utc": utc_now(),
            "evidence_status": run_manifest["evidence_status"],
            "sealed_for_execution": True,
        }
    )
    atomic_write_json(paths["status"], status)
    heartbeat = load_json(paths["heartbeat"])
    heartbeat.update({"state": "prepared", "last_activity_utc": utc_now()})
    atomic_write_json(paths["heartbeat"], heartbeat)
    return run_manifest


def _materialize_argv(template: Sequence[str], *, repo_root: Path, run_dir: Path) -> list[str]:
    replacements = {
        "{python}": sys.executable,
        "{repo_root}": str(repo_root.resolve()),
        "{run_dir}": str(run_dir.resolve()),
    }
    output = []
    for item in template:
        value = item
        for token, replacement in replacements.items():
            value = value.replace(token, replacement)
        output.append(value)
    return output


def _any_true_key(value: Any, forbidden_keys: set[str]) -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key in forbidden_keys and child is True:
                found.append(key)
            found.extend(_any_true_key(child, forbidden_keys))
    elif isinstance(value, list):
        for child in value:
            found.extend(_any_true_key(child, forbidden_keys))
    return found


def _validate_frozen_evidence_result(
    payload: dict[str, Any],
    run_manifest: dict[str, Any],
) -> list[str]:
    """Validate W12/W16/W24 output before the harness marks the job complete."""
    failures: list[str] = []
    mode = str(run_manifest.get("mode"))
    inputs = run_manifest["inputs"]
    environment_sha256 = inputs.get("environment_sha256")
    if mode in REDUCED_MODE_ROLE:
        from scripts.run_paper2_bottleneck_exact_transducer import (
            validate_reduced_certification_payload,
        )

        failures.extend(
            validate_reduced_certification_payload(
                payload,
                REDUCED_MODE_ROLE[mode],
                expected_environment_sha256=environment_sha256,
            )
        )
        provenance = payload.get("provenance", {})
        if provenance.get("git_commit") != run_manifest.get("git", {}).get("commit"):
            failures.append("reduced certificate source commit differs from harness")
        if provenance.get("producer_sha256") != inputs.get("runner_sha256"):
            failures.append("reduced certificate producer differs from harness runner")
        if payload.get("contract_sha256") != inputs.get("result_contract_sha256"):
            failures.append("reduced certificate result-contract hash mismatch")
        if payload.get("scientific_run") is not True:
            failures.append("reduced certificate was not a clean scientific run")
    elif mode == "w24_audit":
        from scripts.run_paper2_bottleneck_full_frontier import (
            validate_w24_profile_state_audit_payload,
        )

        failures.extend(
            validate_w24_profile_state_audit_payload(
                payload,
                expected_environment_sha256=environment_sha256,
            )
        )
        if payload.get("git_head") != run_manifest.get("git", {}).get("commit"):
            failures.append("W24 audit source commit differs from harness")
        if payload.get("generated_by_frontier_runner_sha256") != inputs.get(
            "runner_sha256"
        ):
            failures.append("W24 audit producer differs from harness runner")
        if payload.get("primary_contract_sha256") != inputs.get(
            "result_contract_sha256"
        ):
            failures.append("W24 audit result-contract hash mismatch")
    else:
        failures.append(f"unsupported frozen evidence result mode: {mode}")
    return failures


def _validate_scientific_result(
    payload: dict[str, Any],
    run_manifest: dict[str, Any],
    runner_manifest: dict[str, Any] | None = None,
) -> list[str]:
    """Validate execution assurances without interpreting H_PI as promotion."""
    failures = []
    expected = run_manifest["inputs"]
    assurance = payload.get("execution_assurance")
    if isinstance(assurance, dict):
        if assurance.get("key_schema_version") != "paper2_bottleneck_semantic_markov_key_v3":
            failures.append("result key schema is not v3")
        if assurance.get("primary_frontier_exact") is not True:
            failures.append("primary frontier is not declared exact")
        if assurance.get("numeric_approximation_gap") != 0:
            failures.append("numeric approximation gap is not exactly zero")
        if assurance.get("original_runner_replay_passed") is not True:
            failures.append("original-runner replay did not pass")
        if assurance.get("resource_semantics_passed") is not True:
            failures.append("resource semantics did not pass")
        if assurance.get("contract_sha256") != expected["contract_sha256"]:
            failures.append("result contract hash mismatch")
        if assurance.get("runner_sha256") != expected["runner_sha256"]:
            failures.append("result runner hash mismatch")
    elif payload.get("schema_version") == "paper2_bottleneck_full_frontier_v2":
        if runner_manifest is None:
            failures.append("native frontier result lacks its runner manifest")
        phase = run_manifest["execution"].get("phase")
        if payload.get("phase") != phase:
            failures.append("frontier result phase mismatch")
        if payload.get("weeks") != 24:
            failures.append("frontier result is not the frozen 24-week horizon")
        if payload.get("primary_contract_sha256") != expected["contract_sha256"]:
            failures.append("frontier result primary-contract hash mismatch")
        if payload.get("phase_execution_complete") is not True:
            failures.append("frontier phase did not complete")
        if payload.get("exact_maximum_certified") is not True:
            failures.append("primary frontier is not certified exact")
        if payload.get("selected_replay_complete") is not True:
            failures.append("original-runner winner replay did not pass")
        replay_audit = payload.get("selected_replay_audit", {})
        if replay_audit.get("passed") is not True:
            failures.append("selected replay CRN/resource audit did not pass")
        resolved = payload.get("resolved_frontier", {})
        if resolved.get("exact_maximum_certified") is not True:
            failures.append("resolved primary frontier failed closed")
        calendar_index = payload.get("calendar_index", {})
        screening = payload.get("screening", {})
        expected_count = int(calendar_index.get("calendar_count", -1))
        if expected_count != 11_184_811:
            failures.append("calendar frontier is not the frozen 11,184,811 schedules")
        if screening.get("pass1_count") != expected_count or screening.get("pass2_count") != expected_count:
            failures.append("calendar frontier enumeration is incomplete")
        if screening.get("passes_identical") is not True:
            failures.append("calendar frontier passes have different stream digests")
        overflow = screening.get("contender_overflow", {})
        if overflow.get("aggregate") or overflow.get("per_tape"):
            failures.append("calendar frontier contender resolution overflowed")
        authorization = payload.get("acceleration_authorization", {})
        if authorization.get("key_schema_version") != "paper2_bottleneck_semantic_markov_key_v3":
            failures.append("frontier acceleration key schema is not v3")
        if authorization.get("authorized") is not True:
            failures.append("frontier acceleration authorization is not authorized")
        if authorization.get("sha256") != expected.get("authorization_sha256"):
            failures.append("frontier acceleration authorization hash mismatch")
        if authorization.get("environment_sha256") != expected.get(
            "environment_sha256"
        ):
            failures.append("frontier authorization environment digest mismatch")
        checkpoints = payload.get("build", {}).get("checkpoints", [])
        if not checkpoints or any(
            row.get("environment", {}).get("environment_sha256")
            != expected.get("environment_sha256")
            for row in checkpoints
        ):
            failures.append("frontier checkpoint environment digest mismatch")
        certificate_coverage = payload.get("build", {}).get(
            "collision_certificate_coverage", {}
        )
        coverage_body = (
            dict(certificate_coverage)
            if isinstance(certificate_coverage, dict)
            else {}
        )
        coverage_digest = coverage_body.pop("coverage_sha256", None)
        coverage_rows = certificate_coverage.get("rows", []) if isinstance(
            certificate_coverage, dict
        ) else []
        expected_certificate_count = 60 if phase == "calibration" else 119
        certificate_identities = {
            (row.get("seed"), row.get("tape_sha256"))
            for row in coverage_rows
            if isinstance(row, dict)
        }
        if not (
            coverage_digest == canonical_json_sha256(coverage_body)
            and certificate_coverage.get("schema_version")
            == "paper2_collision_certificate_coverage_v1"
            and certificate_coverage.get("passed") is True
            and certificate_coverage.get("required_count")
            == expected_certificate_count
            and certificate_coverage.get("complete_count")
            == expected_certificate_count
            and certificate_coverage.get("unique_identity_count")
            == expected_certificate_count
            and len(coverage_rows) == expected_certificate_count
            and len(certificate_identities) == expected_certificate_count
            and certificate_coverage.get("rows_sha256")
            == canonical_json_sha256(coverage_rows)
            and all(
                row.get("complete") is True
                and isinstance(row.get("certificate_sha256"), str)
                and len(row["certificate_sha256"]) == 64
                for row in coverage_rows
                if isinstance(row, dict)
            )
            and not certificate_coverage.get("failures")
        ):
            failures.append(
                "frontier does not carry complete unique per-tape collision certificates"
            )
        replays = payload.get("selected_replays", [])
        if not replays:
            failures.append("frontier has no unaccelerated winner replays")
        for row in replays:
            guardrails = row.get("guardrails", {})
            token_values = [guardrails.get(f"token_hours_{letter}") for letter in "mtr"]
            if any(not isinstance(value, (int, float)) for value in token_values):
                failures.append("winner replay lacks response-team resource ledger")
                break
            total = guardrails.get("total_token_hours")
            if total != 4032 or sum(map(float, token_values)) != 4032:
                failures.append("winner replay violates the frozen 4032 team-hour envelope")
                break
            if guardrails.get("mass_residual") != 0:
                failures.append("winner replay has nonzero mass residual")
                break
            required_reserve = {
                "reserve_inventory_initial": 10_000.0,
                "reserve_capacity": 10_000.0,
                "reserve_target_terminal": 10_000.0,
                "reserve_replenishment_lead_time": 168.0,
                "reserve_issue_delay": 24.0,
                "reserve_stock_balance_residual": 0.0,
            }
            if any(guardrails.get(key) != value for key, value in required_reserve.items()):
                failures.append("winner replay reserve ledger/configuration mismatch")
                break
            if not all(
                isinstance(guardrails.get(key), str)
                and len(guardrails[key]) == 64
                for key in (
                    "consumed_base_threat_sha256",
                    "realized_demand_sha256",
                )
            ):
                failures.append("winner replay lacks CRN hashes")
                break
        if runner_manifest is not None:
            input_artifacts = runner_manifest.get("input_artifacts", {})
            if input_artifacts.get("primary_contract", {}).get("sha256") != expected["contract_sha256"]:
                failures.append("runner manifest contract hash mismatch")
            if input_artifacts.get("authorization", {}).get("sha256") != expected.get("authorization_sha256"):
                failures.append("runner manifest authorization hash mismatch")
            code_hashes = runner_manifest.get("code_sha256", {})
            if code_hashes.get(expected["runner_relative"]) != expected["runner_sha256"]:
                failures.append("runner manifest code hash mismatch")
            if runner_manifest.get("key_schema_version") != "paper2_bottleneck_semantic_markov_key_v3":
                failures.append("runner manifest key schema is not v3")
            if runner_manifest.get("exact_maximum_certified") is not True:
                failures.append("runner manifest does not certify the exact maximum")
            if runner_manifest.get("environment_sha256") != expected.get(
                "environment_sha256"
            ):
                failures.append("runner manifest environment digest mismatch")
            if runner_manifest.get("collision_certificate_coverage_sha256") != (
                certificate_coverage.get("coverage_sha256")
            ):
                failures.append(
                    "runner manifest collision-certificate coverage hash mismatch"
                )
    else:
        failures.append("scientific result lacks a recognized execution assurance")
    scope = run_manifest["execution"].get("authorization_scope")
    if scope == "full_guardrail_frontier":
        if not isinstance(assurance, dict) or assurance.get("full_guardrail_frontier_exact") is not True:
            failures.append("full-guardrail scope lacks an exact guardrail frontier")
        if not isinstance(assurance, dict) or assurance.get("all_mandatory_guardrails_audited") is not True:
            failures.append("full-guardrail scope lacks all mandatory guardrails")
    forbidden = _any_true_key(
        payload,
        {"promotion_authorized", "learner_authorized", "paper3_authorized"},
    )
    if forbidden:
        failures.append(
            "primary-bound result attempted prohibited authorization: "
            + ", ".join(sorted(set(forbidden)))
        )
    return failures


@dataclass
class LiveState:
    active_seed_id: str | None = None
    active_pid: int | None = None
    completed_seed_count: int = 0
    failed_seed_count: int = 0


def _frontier_seed_status_update(
    run_dir: Path,
    run_id: str,
    seed_manifest: dict[str, Any],
    *,
    state: str,
    progress: dict[str, Any] | None = None,
    output_sha256: str | None = None,
    returncode: int | None = None,
) -> None:
    """Mirror phase-runner progress into one machine-readable row per tape."""
    completed_builds = 0
    stage = None
    if isinstance(progress, dict):
        stage = progress.get("stage")
        if stage in {"build_transducers", "reduced_certification"}:
            completed_builds = int(progress.get("completed", 0))
    for index, row in enumerate(seed_manifest["seeds"]):
        seed_id = f"seed_{row['seed']}_{row['context']}"
        tape_state = state
        if state == "running":
            if stage == "build_transducers":
                tape_state = "transducer_built" if index < completed_builds else "pending_transducer"
            elif stage == "reduced_certification":
                if index < completed_builds:
                    tape_state = "certified"
                elif row["seed"] == progress.get("active_seed"):
                    tape_state = "certifying"
                else:
                    tape_state = "pending_certification"
            elif stage == "calendar_screen":
                tape_state = "exact_frontier_screening"
            else:
                tape_state = "phase_started"
        payload = {
            "schema_version": SCHEMA,
            "run_id": run_id,
            "seed_id": seed_id,
            "seed": row["seed"],
            "context": row["context"],
            "phase": row["split"],
            "state": tape_state,
            "updated_at_utc": utc_now(),
            "runner_progress": progress,
            "output_sha256": output_sha256,
            "returncode": returncode,
            "evidence": False,
        }
        atomic_write_json(run_dir / "status" / "seeds" / f"{seed_id}.json", payload)


def _write_heartbeat(run_dir: Path, run_id: str, state: str, live: LiveState) -> None:
    atomic_write_json(
        _manifest_paths(run_dir)["heartbeat"],
        {
            "schema_version": SCHEMA,
            "run_id": run_id,
            "state": state,
            "last_activity_utc": utc_now(),
            "active_seed_id": live.active_seed_id,
            "active_pid": live.active_pid,
            "completed_seed_count": live.completed_seed_count,
            "failed_seed_count": live.failed_seed_count,
            "evidence": False,
        },
    )


def _checksum_records(run_dir: Path, relative_paths: Iterable[str]) -> list[dict[str, Any]]:
    records = []
    for relative in sorted(set(relative_paths)):
        path = run_dir / relative
        if not path.is_file():
            raise HarnessError(f"expected artifact missing: {relative}")
        records.append(
            {"path": relative, "sha256": sha256_file(path), "bytes": path.stat().st_size}
        )
    return records


def _confined_checksum_path(run_dir: Path, relative: Any) -> tuple[str, Path]:
    """Resolve one manifest path without permitting aliases or traversal."""
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise HarnessError("checksum record path is not a canonical relative path")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise HarnessError(f"checksum record path escapes run directory: {relative!r}")
    canonical = pure.as_posix()
    if canonical != relative:
        raise HarnessError(f"checksum record path is not canonical: {relative!r}")
    root = run_dir.resolve()
    candidate = (root / Path(*pure.parts)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise HarnessError(
            f"checksum record path escapes run directory: {relative!r}"
        ) from exc
    return canonical, candidate


def execute_run(*, run_dir: Path, repo_root: Path = ROOT, location: str = "local") -> int:
    run_manifest, seed_manifest, command_manifest = _validate_prepared_inputs(run_dir, repo_root)
    if run_manifest["mode"] == "dry-run":
        raise HarnessError("dry-run manifest cannot execute")
    if run_manifest["mode"] == "smoke":
        if run_manifest["execution"]["weeks"] > 4 or len(seed_manifest["seeds"]) != 1:
            raise HarnessError("smoke execution exceeded one-tape W4 ceiling")
    run_id = run_manifest["run_id"]
    paths = _manifest_paths(run_dir)
    live = LiveState()
    stop = threading.Event()
    interval = float(run_manifest["execution"]["heartbeat_interval_seconds"])
    active_command: list[dict[str, Any] | None] = [None]

    def heartbeat_loop() -> None:
        while not stop.wait(interval):
            command = active_command[0]
            if command and command.get("progress_relative"):
                progress_path = run_dir / command["progress_relative"]
                progress = None
                if progress_path.is_file():
                    try:
                        candidate = load_json(progress_path)
                        progress = candidate if isinstance(candidate, dict) else None
                    except HarnessError:
                        progress = None
                _frontier_seed_status_update(
                    run_dir,
                    run_id,
                    seed_manifest,
                    state="running",
                    progress=progress,
                )
            _write_heartbeat(run_dir, run_id, "running", live)

    atomic_write_json(
        paths["status"],
        {
            "schema_version": SCHEMA,
            "run_id": run_id,
            "state": "running",
            "started_at_utc": utc_now(),
            "location": location,
            "evidence": False,
            "evidence_status": "RUNNING_NOT_EVIDENCE",
            "submitted": location == "vps",
            "seed_count": seed_manifest["seed_count"],
            "completed_seed_count": 0,
        },
    )
    runtime_dir = run_dir / "runtime_environment"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    deps, redactions = dependency_snapshot(sys.executable)
    (runtime_dir / "pip_freeze.txt").write_text(deps)
    runtime_machine = machine_snapshot()
    runtime_machine["dependency_snapshot_redactions"] = redactions
    atomic_write_json(runtime_dir / "machine.json", runtime_machine)
    _write_heartbeat(run_dir, run_id, "running", live)
    thread = threading.Thread(target=heartbeat_loop, daemon=True)
    thread.start()
    checksum_paths: list[str] = [
        "run_manifest.json",
        "seed_manifest.json",
        "command_manifest.json",
        "environment/pip_freeze.txt",
        "environment/machine.json",
        "runtime_environment/pip_freeze.txt",
        "runtime_environment/machine.json",
    ]
    if (run_dir / "authorization.json").is_file():
        checksum_paths.append("authorization.json")
    failed = False
    try:
        for index, command in enumerate(command_manifest["commands"]):
            grouped_job = bool(command.get("job_id"))
            item_id = command.get("job_id") or command["seed_id"]
            status_group = "jobs" if grouped_job else "seeds"
            status_path = run_dir / "status" / status_group / f"{item_id}.json"
            output_path = run_dir / command["output_relative"]
            stdout_path = run_dir / command["stdout_relative"]
            stderr_path = run_dir / command["stderr_relative"]
            for path in (output_path.parent, stdout_path.parent, status_path.parent):
                path.mkdir(parents=True, exist_ok=True)
            argv = _materialize_argv(
                command["argv_template"], repo_root=repo_root, run_dir=run_dir
            )
            command_sha = canonical_json_sha256(argv)
            started = utc_now()
            atomic_write_json(
                status_path,
                {
                    "schema_version": SCHEMA,
                    "run_id": run_id,
                    "item_id": item_id,
                    "seed": command.get("seed"),
                    "context": command.get("context"),
                    "phase": command.get("phase"),
                    "state": "starting",
                    "started_at_utc": started,
                    "command_sha256": command_sha,
                    "evidence": False,
                },
            )
            with stdout_path.open("w") as stdout_handle, stderr_path.open("w") as stderr_handle:
                process = subprocess.Popen(
                    argv,
                    cwd=repo_root,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    text=True,
                    shell=False,
                )
                active_command[0] = command
                live.active_seed_id = item_id
                live.active_pid = process.pid
                if grouped_job:
                    _frontier_seed_status_update(
                        run_dir,
                        run_id,
                        seed_manifest,
                        state="running",
                    )
                atomic_write_json(
                    status_path,
                    {
                        "schema_version": SCHEMA,
                        "run_id": run_id,
                        "item_id": item_id,
                        "seed": command.get("seed"),
                        "context": command.get("context"),
                        "phase": command.get("phase"),
                        "state": "running",
                        "started_at_utc": started,
                        "pid": process.pid,
                        "command_sha256": command_sha,
                        "evidence": False,
                    },
                )
                returncode = process.wait()
            output_valid = False
            output_hash = None
            output_error = None
            if output_path.is_file():
                try:
                    parsed = load_json(output_path)
                    output_valid = isinstance(parsed, dict)
                    if output_valid and run_manifest["mode"] == "scientific":
                        native_manifest = None
                        runner_manifest_relative = command.get("runner_manifest_relative")
                        if runner_manifest_relative:
                            native_manifest_path = run_dir / runner_manifest_relative
                            if native_manifest_path.is_file():
                                native_manifest = load_json(native_manifest_path)
                        assurance_failures = _validate_scientific_result(
                            parsed, run_manifest, native_manifest
                        )
                        if assurance_failures:
                            output_valid = False
                            output_error = "; ".join(assurance_failures)
                    elif output_valid and run_manifest["mode"] in FROZEN_EVIDENCE_MODES:
                        evidence_failures = _validate_frozen_evidence_result(
                            parsed, run_manifest
                        )
                        if evidence_failures:
                            output_valid = False
                            output_error = "; ".join(evidence_failures)
                    output_hash = sha256_file(output_path)
                except HarnessError as exc:
                    output_error = str(exc)
            state = "completed" if returncode == 0 and output_valid else "failed"
            if state == "completed":
                live.completed_seed_count += (
                    seed_manifest["seed_count"] if grouped_job else 1
                )
            else:
                live.failed_seed_count += (
                    seed_manifest["seed_count"] if grouped_job else 1
                )
                failed = True
            active_command[0] = None
            live.active_seed_id = None
            live.active_pid = None
            if grouped_job:
                final_progress = None
                progress_relative = command.get("progress_relative")
                if progress_relative and (run_dir / progress_relative).is_file():
                    try:
                        candidate = load_json(run_dir / progress_relative)
                        final_progress = candidate if isinstance(candidate, dict) else None
                    except HarnessError:
                        final_progress = None
                _frontier_seed_status_update(
                    run_dir,
                    run_id,
                    seed_manifest,
                    state=state,
                    progress=final_progress,
                    output_sha256=output_hash,
                    returncode=returncode,
                )
            atomic_write_json(
                status_path,
                {
                    "schema_version": SCHEMA,
                    "run_id": run_id,
                    "item_id": item_id,
                    "seed": command.get("seed"),
                    "context": command.get("context"),
                    "phase": command.get("phase"),
                    "state": state,
                    "started_at_utc": started,
                    "finished_at_utc": utc_now(),
                    "returncode": returncode,
                    "command_sha256": command_sha,
                    "output_relative": command["output_relative"],
                    "output_valid_json": output_valid,
                    "output_sha256": output_hash,
                    "output_error": output_error,
                    "stdout_relative": command["stdout_relative"],
                    "stderr_relative": command["stderr_relative"],
                    "evidence": False,
                },
            )
            checksum_paths.extend([command["stdout_relative"], command["stderr_relative"]])
            if output_valid:
                checksum_paths.append(command["output_relative"])
                receipt_relative = f"status/jobs/{item_id}.execution_receipt.json"
                receipt = {
                    "schema_version": SCHEMA,
                    "run_id": run_id,
                    "job_id": item_id,
                    "mode": run_manifest["mode"],
                    "git_commit": run_manifest["git"]["commit"],
                    "contract_sha256": run_manifest["inputs"]["contract_sha256"],
                    "result_contract_sha256": run_manifest["inputs"].get(
                        "result_contract_sha256"
                    ),
                    "runner_sha256": run_manifest["inputs"]["runner_sha256"],
                    "harness_sha256": run_manifest["inputs"]["harness_sha256"],
                    "environment_sha256": run_manifest["inputs"][
                        "environment_sha256"
                    ],
                    "seed_manifest_sha256": run_manifest["inputs"][
                        "seed_manifest_sha256"
                    ],
                    "command_manifest_sha256": run_manifest["inputs"][
                        "command_manifest_sha256"
                    ],
                    "materialized_command_sha256": command_sha,
                    "output_relative": command["output_relative"],
                    "output_sha256": output_hash,
                    "validated_not_independently_audited": True,
                    "evidence": False,
                }
                atomic_write_json(run_dir / receipt_relative, receipt)
                checksum_paths.append(receipt_relative)
            for relative_key in ("runner_manifest_relative", "progress_relative"):
                relative = command.get(relative_key)
                if relative and (run_dir / relative).is_file():
                    checksum_paths.append(relative)
            checkpoint_relative = command.get("checkpoint_relative")
            if checkpoint_relative and (run_dir / checkpoint_relative).is_dir():
                checksum_paths.extend(
                    str(path.relative_to(run_dir))
                    for path in (run_dir / checkpoint_relative).rglob("*")
                    if path.is_file()
                )
            _write_heartbeat(run_dir, run_id, "running", live)
            atomic_write_json(
                paths["status"],
                {
                    "schema_version": SCHEMA,
                    "run_id": run_id,
                    "state": "running" if not failed else "failed",
                    "updated_at_utc": utc_now(),
                    "location": location,
                    "evidence": False,
                    "evidence_status": "RUNNING_NOT_EVIDENCE" if not failed else "FAILED_NOT_EVIDENCE",
                    "seed_count": seed_manifest["seed_count"],
                    "completed_seed_count": live.completed_seed_count,
                    "failed_seed_count": live.failed_seed_count,
                    "next_command_index": index + 1,
                },
            )
            if failed:
                break
    finally:
        stop.set()
        thread.join(timeout=max(1.0, interval * 2))

    completed_all = not failed and live.completed_seed_count == seed_manifest["seed_count"]
    final_state = "completed" if completed_all else "failed"
    final_evidence_status = (
        "COMPLETED_REMOTE_NOT_RETRIEVED_NOT_EVIDENCE"
        if completed_all and location == "vps"
        else "COMPLETED_HASHED_AUDIT_PENDING_NOT_EVIDENCE"
        if completed_all
        else "FAILED_NOT_EVIDENCE"
    )
    completion_receipt_relative = "status/run_completion_receipt.json"
    atomic_write_json(
        run_dir / completion_receipt_relative,
        {
            "schema_version": SCHEMA,
            "run_id": run_id,
            "state": final_state,
            "location": location,
            "seed_count": seed_manifest["seed_count"],
            "completed_seed_count": live.completed_seed_count,
            "failed_seed_count": live.failed_seed_count,
            "run_manifest_sha256": sha256_file(paths["run"]),
            "seed_manifest_sha256": sha256_file(paths["seeds"]),
            "command_manifest_sha256": sha256_file(paths["commands"]),
            "immutable_status_snapshot": True,
            "evidence": False,
            "evidence_status": final_evidence_status,
        },
    )
    checksum_paths.append(completion_receipt_relative)
    for status_group in ("seeds", "jobs"):
        status_dir = run_dir / "status" / status_group
        if status_dir.is_dir():
            checksum_paths.extend(
                str(path.relative_to(run_dir))
                for path in status_dir.rglob("*.json")
                if path.is_file()
            )

    records = _checksum_records(run_dir, checksum_paths)
    checksum_manifest = {
        "schema_version": SCHEMA,
        "run_id": run_id,
        "generated_at_utc": utc_now(),
        "records": records,
        "record_count": len(records),
        "evidence": False,
        "evidence_status": "HASHED_AUDIT_PENDING_NOT_EVIDENCE",
    }
    atomic_write_json(paths["checksums"], checksum_manifest)
    atomic_write_json(
        paths["status"],
        {
            "schema_version": SCHEMA,
            "run_id": run_id,
            "state": final_state,
            "finished_at_utc": utc_now(),
            "location": location,
            "evidence": False,
            "evidence_status": final_evidence_status,
            "seed_count": seed_manifest["seed_count"],
            "completed_seed_count": live.completed_seed_count,
            "failed_seed_count": live.failed_seed_count,
            "checksums_sha256": sha256_file(paths["checksums"]),
            "independent_scientific_audit_required": True,
        },
    )
    _write_heartbeat(run_dir, run_id, final_state, live)
    return 0 if completed_all else 1


def verify_artifacts(run_dir: Path, *, retrieved: bool) -> dict[str, Any]:
    _run_relative, run_path = _confined_checksum_path(run_dir, "run_manifest.json")
    _status_relative, status_path = _confined_checksum_path(
        run_dir, "status/run_status.json"
    )
    _checksums_relative, checksums_path = _confined_checksum_path(
        run_dir, "artifact_checksums.json"
    )
    _seeds_relative, seeds_path = _confined_checksum_path(
        run_dir, "seed_manifest.json"
    )
    _commands_relative, commands_path = _confined_checksum_path(
        run_dir, "command_manifest.json"
    )
    run_manifest = load_json(run_path)
    status = load_json(status_path)
    checksums = load_json(checksums_path)
    failures: list[str] = []
    if status.get("state") != "completed":
        failures.append("run status is not completed")
    checksum_manifest_sha256 = sha256_file(checksums_path)
    if status.get("checksums_sha256") != checksum_manifest_sha256:
        failures.append("checksum manifest hash does not match run status anchor")
    if checksums.get("schema_version") != SCHEMA:
        failures.append("checksum manifest schema mismatch")
    if checksums.get("run_id") != run_manifest.get("run_id"):
        failures.append("checksum manifest run id mismatch")
    records = checksums.get("records")
    if not isinstance(records, list):
        failures.append("checksum manifest records are missing or malformed")
        records = []
    if (
        isinstance(checksums.get("record_count"), bool)
        or not isinstance(checksums.get("record_count"), int)
        or checksums.get("record_count") != len(records)
    ):
        failures.append("checksum manifest record_count mismatch")

    checked = []
    seen_paths: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            failures.append(f"malformed checksum record {index}")
            continue
        try:
            relative, path = _confined_checksum_path(run_dir, record.get("path"))
        except HarnessError as exc:
            failures.append(str(exc))
            continue
        if relative in seen_paths:
            failures.append(f"duplicate checksum record path: {relative}")
            continue
        seen_paths.add(relative)
        expected_sha = record.get("sha256")
        expected_bytes = record.get("bytes")
        if not isinstance(expected_sha, str) or re.fullmatch(r"[0-9a-f]{64}", expected_sha) is None:
            failures.append(f"malformed checksum digest {relative}")
            continue
        if (
            isinstance(expected_bytes, bool)
            or not isinstance(expected_bytes, int)
            or expected_bytes < 0
        ):
            failures.append(f"malformed checksum byte count {relative}")
            continue
        if not path.is_file():
            failures.append(f"missing {relative}")
            continue
        actual = sha256_file(path)
        ok = actual == expected_sha and path.stat().st_size == expected_bytes
        checked.append({"path": relative, "passed": ok, "actual_sha256": actual})
        if not ok:
            failures.append(f"checksum mismatch {relative}")

    seed_manifest = load_json(seeds_path)
    command_manifest = load_json(commands_path)
    required_protected_paths = {
        "run_manifest.json",
        "seed_manifest.json",
        "command_manifest.json",
        "status/run_completion_receipt.json",
    }
    for row in seed_manifest.get("seeds", []):
        required_protected_paths.add(
            f"status/seeds/seed_{row['seed']}_{row['context']}.json"
        )
    for command in command_manifest.get("commands", []):
        item_id = command.get("job_id") or command.get("seed_id")
        if not isinstance(item_id, str) or not item_id:
            failures.append("command manifest item identity is malformed")
            continue
        if command.get("job_id"):
            required_protected_paths.add(f"status/jobs/{item_id}.json")
        required_protected_paths.add(
            f"status/jobs/{item_id}.execution_receipt.json"
        )
    missing_protected = sorted(required_protected_paths - seen_paths)
    if missing_protected:
        failures.append(
            "immutable checksum manifest omits protected status/receipt paths: "
            + ", ".join(missing_protected)
        )

    seed_statuses = sorted((run_dir / "status" / "seeds").glob("*.json"))
    expected_seed_count = seed_manifest["seed_count"]
    if len(seed_statuses) != expected_seed_count:
        failures.append("per-seed status count mismatch")
    for seed_status_path in seed_statuses:
        seed_status_relative = str(seed_status_path.relative_to(run_dir))
        try:
            _relative, confined_seed_status = _confined_checksum_path(
                run_dir, seed_status_relative
            )
        except HarnessError as exc:
            failures.append(str(exc))
            continue
        if seed_status_relative not in seen_paths:
            failures.append(
                f"per-seed status is not checksum protected: {seed_status_relative}"
            )
            continue
        seed_status = load_json(confined_seed_status)
        if seed_status.get("state") != "completed":
            failures.append(f"seed not completed: {seed_status_path.name}")
    completion_receipt_relative = "status/run_completion_receipt.json"
    _relative, completion_receipt_path = _confined_checksum_path(
        run_dir, completion_receipt_relative
    )
    if (
        completion_receipt_relative in seen_paths
        and completion_receipt_path.is_file()
    ):
        completion_receipt = load_json(completion_receipt_path)
        receipt_expected = {
            "schema_version": SCHEMA,
            "run_id": run_manifest.get("run_id"),
            "state": "completed",
            "seed_count": expected_seed_count,
            "completed_seed_count": expected_seed_count,
            "failed_seed_count": 0,
            "run_manifest_sha256": sha256_file(run_path),
            "seed_manifest_sha256": sha256_file(seeds_path),
            "command_manifest_sha256": sha256_file(commands_path),
            "immutable_status_snapshot": True,
        }
        for key, value in receipt_expected.items():
            if completion_receipt.get(key) != value:
                failures.append(f"run completion receipt mismatch: {key}")
    else:
        failures.append("run completion receipt is missing")
    result = {
        "schema_version": SCHEMA,
        "run_id": run_manifest["run_id"],
        "verified_at_utc": utc_now(),
        "retrieved": bool(retrieved),
        "checks_passed": not failures,
        "failures": failures,
        "checked_records": checked,
        "evidence": False,
        "evidence_status": (
            "RETRIEVED_HASH_VERIFIED_INDEPENDENT_AUDIT_PENDING_NOT_EVIDENCE"
            if retrieved and not failures
            else "LOCAL_HASH_VERIFIED_INDEPENDENT_AUDIT_PENDING_NOT_EVIDENCE"
            if not failures
            else "VERIFICATION_FAILED_NOT_EVIDENCE"
        ),
    }
    atomic_write_json(run_dir / "retrieval_verification.json", result)
    return result


def remote_preflight(*, output_dir: Path, host: str, remote_python: str) -> dict[str, Any]:
    if host != DEFAULT_HOST:
        raise HarnessError(f"host must be the approved SSH alias {DEFAULT_HOST!r}")
    _validate_remote_value(remote_python, "remote python")
    output_dir.mkdir(parents=True, exist_ok=True)
    code = (
        "import json,os,platform,shutil;"
        "d=shutil.disk_usage('.');"
        "m=None;"
        "exec(\"try:\\n m=int(os.sysconf('SC_PAGE_SIZE'))*int(os.sysconf('SC_PHYS_PAGES'))\\nexcept Exception:\\n pass\");"
        "print(json.dumps({'system':platform.system(),'release':platform.release(),"
        "'machine':platform.machine(),'python_version':platform.python_version(),"
        "'python_implementation':platform.python_implementation(),'cpu_count':os.cpu_count(),"
        "'physical_memory_bytes':m,'disk_total_bytes':d.total,'disk_free_bytes':d.free,"
        "'hostname_omitted_to_avoid_environment_disclosure':True},sort_keys=True))"
    )
    machine_command = f"{_remote_shell_path(remote_python)} -c {shlex.quote(code)}"
    machine_result = run_capture(
        ["ssh", "-o", "BatchMode=yes", host, machine_command]
    )
    try:
        machine = json.loads(machine_result.stdout)
    except json.JSONDecodeError as exc:
        raise HarnessError("remote machine snapshot was not valid JSON") from exc
    deps_result = run_capture(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            host,
            f"{_remote_shell_path(remote_python)} -m pip freeze --all",
        ]
    )
    deps, redactions = sanitize_dependency_snapshot(deps_result.stdout)
    deps_path = output_dir / "pip_freeze.txt"
    deps_path.write_text(deps)
    requirements_sha256 = {
        str(path.relative_to(ROOT)): sha256_file(path)
        for path in (ROOT / "requirements.txt", ROOT / "requirements-pinned.txt")
    }
    environment_code = f"""
import hashlib
import json
import platform
import sys
import sysconfig
from importlib import metadata
from pathlib import Path
import simpy.core as simpy_core
import simpy.events as simpy_events
import simpy.resources.base as simpy_resources_base
import simpy.resources.container as simpy_resources_container
import simpy.resources.resource as simpy_resources_resource

packages = {{}}
for package in {repr(("numpy", "simpy", "gymnasium", "scipy", "pandas"))}:
    try:
        packages[package] = metadata.version(package)
    except metadata.PackageNotFoundError:
        packages[package] = "MISSING"
payload = {{
    "python_implementation": platform.python_implementation(),
    "python_version": platform.python_version(),
    "python_cache_tag": sys.implementation.cache_tag,
    "python_soabi": sysconfig.get_config_var("SOABI"),
    "packages": packages,
    "requirements_sha256": {json.dumps(requirements_sha256, sort_keys=True)},
    "simpy_source_sha256": {{
        module.__name__: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        for module in (
            simpy_core,
            simpy_events,
            simpy_resources_base,
            simpy_resources_container,
            simpy_resources_resource,
        )
    }},
}}
encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
payload["environment_sha256"] = hashlib.sha256(encoded).hexdigest()
print(json.dumps(payload, sort_keys=True))
"""
    environment_result = run_capture(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            host,
            f"{_remote_shell_path(remote_python)} -c "
            f"{shlex.quote(environment_code)}",
        ]
    )
    try:
        remote_environment = json.loads(environment_result.stdout)
    except json.JSONDecodeError as exc:
        raise HarnessError("remote scientific environment was not valid JSON") from exc
    from scripts.run_paper2_bottleneck_exact_transducer import (
        certification_environment,
    )

    remote_environment = validate_scientific_environment_payload(
        remote_environment,
        local_reference=certification_environment(),
    )
    environment_path = output_dir / "scientific_environment.json"
    atomic_write_json(environment_path, remote_environment)
    atomic_write_json(output_dir / "machine.json", machine)
    result = {
        "schema_version": SCHEMA,
        "checked_at_utc": utc_now(),
        "ssh_alias": host,
        "reachable": True,
        "remote_python": remote_python,
        "machine_snapshot_sha256": sha256_file(output_dir / "machine.json"),
        "dependency_snapshot_sha256": sha256_file(deps_path),
        "dependency_snapshot_redactions": redactions,
        "scientific_environment_sha256": remote_environment[
            "environment_sha256"
        ],
        "scientific_environment_snapshot_sha256": sha256_file(
            environment_path
        ),
        "scientific_job_submitted": False,
        "evidence": False,
        "evidence_status": "REMOTE_CAPABILITY_PREFLIGHT_ONLY_NOT_EVIDENCE",
    }
    atomic_write_json(output_dir / "preflight.json", result)
    return result


def stage_vps(*, run_dir: Path, host: str, remote_root: str, repo_root: Path = ROOT) -> dict[str, Any]:
    if host != DEFAULT_HOST:
        raise HarnessError(f"host must be the approved SSH alias {DEFAULT_HOST!r}")
    _validate_remote_value(remote_root, "remote root")
    run_manifest, _, _ = _validate_prepared_inputs(
        run_dir,
        repo_root,
        allow_preflight_platform_environment=True,
    )
    if not run_manifest["git"]["scientific_source_immutable"]:
        raise HarnessError("VPS staging requires critical inputs tracked and identical to HEAD")
    run_id = run_manifest["run_id"]
    _validate_run_id(run_id)
    transport = run_dir / "transport"
    transport.mkdir(parents=True, exist_ok=True)
    # A plain ``git archive`` is insufficient: the scientific runner verifies
    # HEAD and source drift with Git on the VPS.  A single-commit bundle keeps
    # that verification available without copying local remotes or credentials.
    source_bundle = transport / "source.bundle"
    # ``git bundle create <path> <raw-oid>`` may resolve to no advertised ref
    # and Git then refuses an empty bundle.  The validator immediately above
    # proves clean live HEAD equals the prepared immutable commit, so advertise
    # HEAD while retaining the recorded OID for the remote detached checkout.
    run_capture(
        ["git", "bundle", "create", str(source_bundle), "HEAD"],
        cwd=repo_root,
    )
    control_tar = transport / "control.tar"
    with tarfile.open(control_tar, "w") as archive:
        for relative in (
            "run_manifest.json",
            "seed_manifest.json",
            "command_manifest.json",
            "authorization.json",
            "environment",
            "status",
        ):
            path = run_dir / relative
            if path.exists():
                archive.add(path, arcname=relative)
    remote_run = f"{remote_root.rstrip('/')}/{run_id}"
    remote_run_q = _remote_shell_path(remote_run)
    mkdir_command = (
        f"test ! -e {remote_run_q}/source && "
        f"mkdir -p {remote_run_q}/control"
    )
    run_capture(["ssh", "-o", "BatchMode=yes", host, mkdir_command])
    run_capture(
        [
            "rsync",
            "-a",
            str(source_bundle),
            str(control_tar),
            f"{host}:{remote_run}/",
        ]
    )
    unpack = (
        f"git clone --quiet --no-checkout {remote_run_q}/source.bundle "
        f"{remote_run_q}/source && "
        f"git -C {remote_run_q}/source checkout --quiet --detach "
        f"{shlex.quote(run_manifest['git']['commit'])} && "
        f"tar -xf {remote_run_q}/control.tar -C {remote_run_q}/control"
    )
    run_capture(["ssh", "-o", "BatchMode=yes", host, unpack])
    result_payload = {
        "schema_version": SCHEMA,
        "staged_at_utc": utc_now(),
        "ssh_alias": host,
        "remote_run": remote_run,
        "source_commit": run_manifest["git"]["commit"],
        "source_bundle_sha256": sha256_file(source_bundle),
        "control_tar_sha256": sha256_file(control_tar),
        "scientific_job_submitted": False,
        "evidence": False,
        "evidence_status": "STAGED_NOT_SUBMITTED_NOT_EVIDENCE",
    }
    atomic_write_json(run_dir / "status" / "remote_stage.json", result_payload)
    return result_payload


def launch_vps(*, run_dir: Path, host: str, remote_python: str) -> dict[str, Any]:
    if host != DEFAULT_HOST:
        raise HarnessError(f"host must be the approved SSH alias {DEFAULT_HOST!r}")
    _validate_remote_value(remote_python, "remote python")
    stage = load_json(run_dir / "status" / "remote_stage.json")
    run_manifest = load_json(run_dir / "run_manifest.json")
    if (
        run_manifest.get("mode") not in EXECUTABLE_EVIDENCE_MODES
        or run_manifest.get("execution", {}).get("sealed_for_execution") is not True
    ):
        raise HarnessError(
            "VPS launch is reserved for a sealed evidence-execution manifest"
        )
    remote_run = stage["remote_run"]
    _validate_remote_value(remote_run, "remote run path")
    harness_rel = run_manifest["inputs"]["harness_relative"]
    remote_run_q = _remote_shell_path(remote_run)
    launch_command = (
        f"cd {remote_run_q}/source && "
        f"mkdir -p {remote_run_q}/control/logs && "
        f"{{ nohup {_remote_shell_path(remote_python)} {shlex.quote(harness_rel)} execute "
        f"--run-dir {remote_run_q}/control --repo-root {remote_run_q}/source "
        f"--location vps > {remote_run_q}/control/logs/launcher.stdout.log "
        f"2> {remote_run_q}/control/logs/launcher.stderr.log < /dev/null & "
        f"echo $!; }}"
    )
    result = run_capture(["ssh", "-o", "BatchMode=yes", host, launch_command])
    pid = result.stdout.strip()
    if not pid.isdigit():
        raise HarnessError("remote launcher did not return a numeric pid")
    payload = {
        "schema_version": SCHEMA,
        "submitted_at_utc": utc_now(),
        "ssh_alias": host,
        "remote_run": remote_run,
        "remote_launcher_pid": int(pid),
        "submission_is_evidence": False,
        "evidence": False,
        "evidence_status": "SUBMITTED_NOT_EVIDENCE",
    }
    atomic_write_json(run_dir / "status" / "remote_submission.json", payload)
    return payload


def remote_status(*, run_dir: Path, host: str) -> dict[str, Any]:
    if host != DEFAULT_HOST:
        raise HarnessError(f"host must be the approved SSH alias {DEFAULT_HOST!r}")
    stage = load_json(run_dir / "status" / "remote_stage.json")
    remote_run = stage["remote_run"]
    remote_run_q = _remote_shell_path(remote_run)
    command = (
        f"for f in heartbeat.json run_status.json; do "
        f"p={remote_run_q}/control/status/$f; "
        "if [ -f \"$p\" ]; then printf '%s\\n' \"===${f}===\"; cat \"$p\"; fi; done"
    )
    result = run_capture(["ssh", "-o", "BatchMode=yes", host, command])
    payload = {
        "schema_version": SCHEMA,
        "polled_at_utc": utc_now(),
        "ssh_alias": host,
        "remote_run": remote_run,
        "raw_status": result.stdout,
        "evidence": False,
        "evidence_status": "WATCHER_STATUS_ONLY_NOT_EVIDENCE",
    }
    watcher_dir = run_dir / "watcher"
    watcher_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(watcher_dir / "latest.json", payload)
    with (watcher_dir / "watcher.log").open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
    return payload


def retrieve_vps(*, run_dir: Path, host: str) -> dict[str, Any]:
    if host != DEFAULT_HOST:
        raise HarnessError(f"host must be the approved SSH alias {DEFAULT_HOST!r}")
    stage = load_json(run_dir / "status" / "remote_stage.json")
    remote_run = stage["remote_run"]
    retrieved = run_dir / "retrieved"
    if retrieved.exists():
        raise HarnessError(f"retrieval destination already exists: {retrieved}")
    retrieved.mkdir(parents=True)
    run_capture(["rsync", "-a", f"{host}:{remote_run}/control/", f"{retrieved}/"])
    verification = verify_artifacts(retrieved, retrieved=True)
    payload = {
        "schema_version": SCHEMA,
        "retrieved_at_utc": utc_now(),
        "ssh_alias": host,
        "remote_run": remote_run,
        "retrieval_destination": str(retrieved.resolve()),
        "verification_passed": verification["checks_passed"],
        "verification_sha256": sha256_file(retrieved / "retrieval_verification.json"),
        "evidence": False,
        "evidence_status": verification["evidence_status"],
    }
    atomic_write_json(run_dir / "status" / "retrieval.json", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare")
    prepare.add_argument("--run-dir", type=Path, required=True)
    prepare.add_argument("--run-id", required=True)
    prepare.add_argument(
        "--mode",
        choices=("dry-run", "smoke", "scientific", *FROZEN_EVIDENCE_MODES),
        default="dry-run",
    )
    prepare.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    prepare.add_argument("--runner", type=Path)
    prepare.add_argument("--seed", action="append", type=parse_seed_spec)
    prepare.add_argument("--split")
    prepare.add_argument("--phase", choices=("calibration", "locked"))
    prepare.add_argument("--weeks", type=int)
    prepare.add_argument("--runner-workers", type=int, default=1)
    prepare.add_argument("--calibration-result", type=Path)
    prepare.add_argument("--batch-size", type=int, default=65_536)
    prepare.add_argument("--max-contenders", type=int, default=100_000)
    prepare.add_argument("--heartbeat-interval-seconds", type=float, default=15.0)
    prepare.add_argument("--authorization-json", type=Path)
    prepare.add_argument("--scientific-environment-json", type=Path)

    seal = sub.add_parser("seal")
    seal.add_argument("--run-dir", type=Path, required=True)
    seal.add_argument("--authorization-json", type=Path, required=True)
    seal.add_argument("--repo-root", type=Path, default=ROOT)

    execute = sub.add_parser("execute")
    execute.add_argument("--run-dir", type=Path, required=True)
    execute.add_argument("--repo-root", type=Path, default=ROOT)
    execute.add_argument("--location", choices=("local", "vps"), default="local")

    verify = sub.add_parser("verify")
    verify.add_argument("--run-dir", type=Path, required=True)
    verify.add_argument("--retrieved", action="store_true")

    preflight = sub.add_parser("remote-preflight")
    preflight.add_argument("--output-dir", type=Path, required=True)
    preflight.add_argument("--host", default=DEFAULT_HOST)
    preflight.add_argument("--remote-python", default="~/scres-ia/.venv/bin/python")

    stage = sub.add_parser("stage-vps")
    stage.add_argument("--run-dir", type=Path, required=True)
    stage.add_argument("--host", default=DEFAULT_HOST)
    stage.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT)

    launch = sub.add_parser("launch-vps")
    launch.add_argument("--run-dir", type=Path, required=True)
    launch.add_argument("--host", default=DEFAULT_HOST)
    launch.add_argument("--remote-python", default="~/scres-ia/.venv/bin/python")

    status = sub.add_parser("remote-status")
    status.add_argument("--run-dir", type=Path, required=True)
    status.add_argument("--host", default=DEFAULT_HOST)

    retrieve = sub.add_parser("retrieve-vps")
    retrieve.add_argument("--run-dir", type=Path, required=True)
    retrieve.add_argument("--host", default=DEFAULT_HOST)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "prepare":
            scientific = args.mode == "scientific"
            if scientific:
                if args.phase is None:
                    raise HarnessError("scientific prepare requires --phase")
                seeds = args.seed or _frozen_phase_seed_specs(args.phase)
                runner = (args.runner or DEFAULT_RUNNER).resolve()
                weeks = 24 if args.weeks is None else args.weeks
                split = args.phase if args.split is None else args.split
            elif args.mode in FROZEN_EVIDENCE_MODES:
                contract_payload = load_json(args.contract.resolve())
                profile = _frozen_evidence_profile(contract_payload, args.mode)
                seeds = args.seed or profile["seeds"]
                runner = (
                    args.runner
                    or ROOT / profile["runner"]
                ).resolve()
                weeks = profile["weeks"] if args.weeks is None else args.weeks
                split = profile["split"] if args.split is None else args.split
            else:
                seeds = args.seed or [(1_110_001, CONTEXTS[0])]
                runner = (args.runner or DEFAULT_SMOKE_RUNNER).resolve()
                weeks = 4 if args.weeks is None else args.weeks
                split = args.split or "harness_smoke_burned"
            result = prepare_run(
                run_dir=args.run_dir,
                run_id=args.run_id,
                mode=args.mode,
                contract_path=args.contract.resolve(),
                runner_path=runner,
                seeds=seeds,
                split=split,
                weeks=weeks,
                runner_workers=args.runner_workers,
                heartbeat_interval=args.heartbeat_interval_seconds,
                authorization_path=args.authorization_json,
                phase=args.phase,
                calibration_result_path=(
                    args.calibration_result.resolve()
                    if args.calibration_result
                    else None
                ),
                batch_size=args.batch_size,
                max_contenders=args.max_contenders,
                scientific_environment_path=(
                    args.scientific_environment_json.resolve()
                    if args.scientific_environment_json
                    else None
                ),
            )
        elif args.command == "seal":
            result = seal_run(
                run_dir=args.run_dir.resolve(),
                authorization_path=args.authorization_json.resolve(),
                repo_root=args.repo_root.resolve(),
            )
        elif args.command == "execute":
            return execute_run(
                run_dir=args.run_dir.resolve(),
                repo_root=args.repo_root.resolve(),
                location=args.location,
            )
        elif args.command == "verify":
            result = verify_artifacts(args.run_dir.resolve(), retrieved=args.retrieved)
            if not result["checks_passed"]:
                print(json.dumps(result, indent=2, sort_keys=True))
                return 1
        elif args.command == "remote-preflight":
            result = remote_preflight(
                output_dir=args.output_dir.resolve(),
                host=args.host,
                remote_python=args.remote_python,
            )
        elif args.command == "stage-vps":
            result = stage_vps(run_dir=args.run_dir.resolve(), host=args.host, remote_root=args.remote_root)
        elif args.command == "launch-vps":
            result = launch_vps(
                run_dir=args.run_dir.resolve(), host=args.host, remote_python=args.remote_python
            )
        elif args.command == "remote-status":
            result = remote_status(run_dir=args.run_dir.resolve(), host=args.host)
        elif args.command == "retrieve-vps":
            result = retrieve_vps(run_dir=args.run_dir.resolve(), host=args.host)
        else:  # pragma: no cover
            raise HarnessError(f"unsupported command {args.command}")
    except HarnessError as exc:
        print(f"FAIL_CLOSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
