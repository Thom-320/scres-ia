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
import secrets
import shlex
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from typing import Any, Callable, Iterable, Mapping, Sequence


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
ISOLATED_BOOTSTRAP = ROOT / "scripts" / "paper2_isolated_bootstrap.py"
FULL_HORIZON_CONTRACT = (
    ROOT / "contracts" / "paper2_bottleneck_full_horizon_bound_v1.json"
)
DEFAULT_HOST = "ovh-agent-lab"
DEFAULT_REMOTE_ROOT = "~/paper2-bound-runs"
SCHEMA = "paper2_bound_execution_harness_v3"
AUTH_SCHEMA = "paper2_bound_execution_authorization_v9"
CUSTODY_SCHEMA = "paper2_bound_trusted_local_custody_v3"
STAGE_CUSTODY_SCHEMA = "paper2_bound_trusted_stage_custody_v3"
LAUNCH_CUSTODY_SCHEMA = "paper2_bound_trusted_launch_custody_v3"
RUNTIME_ATTESTATION_SCHEMA = "paper2_isolated_runtime_attestation_v2"
RUNTIME_DISTRIBUTION_MANIFEST_SCHEMA = "paper2_distribution_installed_files_v1"
RUNTIME_PACKAGES = ("numpy", "simpy", "gymnasium", "scipy", "pandas")
RUNTIME_MANIFEST_EXCLUSIONS = {
    "cache": ["__pycache__", ".pyc", ".pyo"],
    "record_signatures": ["RECORD.jws", "RECORD.p7s"],
    "outside_site_packages": True,
}
REDUCED_PAIR_VERIFICATION_SCHEMA = "paper2_reduced_pair_harness_verification_v2"
SIGNED_PRELAUNCH_SCHEMA = "paper2_reduced_signed_prelaunch_v1"
SIGNED_PRELAUNCH_ACK_SCHEMA = "paper2_reduced_signed_prelaunch_ack_v1"
SIGNED_HARNESS_RECEIPT_SCHEMA = "paper2_reduced_signed_harness_receipt_v1"
SIGNED_TRANSFER_MANIFEST_SCHEMA = "paper2_reduced_signed_transfer_manifest_v1"
RETRIEVED_TRANSFER_VERIFICATION_SCHEMA = (
    "paper2_reduced_retrieved_transfer_verification_v2"
)
RETRIEVED_PAIR_VERIFICATION_SCHEMA = (
    "paper2_reduced_retrieved_pair_verification_v3"
)
AUTHORIZED_PAIR_REVERIFICATION_SCHEMA = (
    "paper2_reduced_authorized_pair_archive_reverification_v3"
)
PORTABLE_PAIR_VERIFICATION_MODE = "portable_signed_transfer_pair_v3"
CLASSIC_PAIR_VERIFICATION_MODE = "launch_host_inode_pair_v2"
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


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _semantic_equal(actual: Any, expected: Any) -> bool:
    """Compare receipt fields without allowing bool/int JSON type aliases."""
    if isinstance(expected, bool):
        return actual is expected
    if isinstance(expected, int):
        return (
            isinstance(actual, int)
            and not isinstance(actual, bool)
            and actual == expected
        )
    return actual == expected


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


def exclusive_write_json(path: Path, value: Any) -> None:
    """Create a custody artifact without following or replacing its leaf path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


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
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        list(argv),
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=env,
    )
    if check and result.returncode != 0:
        raise HarnessError(
            f"command failed ({result.returncode}): {shlex.join(argv)}: "
            f"{result.stderr.strip()}"
        )
    return result


def scientific_child_environment() -> dict[str, str]:
    """Return the complete, deterministic environment for scientific children."""
    environment = {
        "PATH": os.defpath,
        "HOME": os.environ.get("HOME", str(Path.home())),
        "TMPDIR": os.environ.get("TMPDIR", tempfile.gettempdir()),
        "LANG": "C",
        "LC_ALL": "C",
        "TZ": "UTC",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "SCRES_SCIENTIFIC_CHILD": "1",
    }
    if any(key.startswith("PYTHON") for key in environment):  # pragma: no cover
        raise HarnessError("sanitized scientific environment contains PYTHON* variables")
    return environment


def validate_runtime_attestation_payload(payload: Any) -> dict[str, Any]:
    """Validate the source-independent bootstrap attestation and its digest."""
    if not isinstance(payload, dict) or payload.get("schema_version") != RUNTIME_ATTESTATION_SCHEMA:
        raise HarnessError("scientific runtime attestation schema mismatch")
    body = dict(payload)
    runtime_sha256 = body.pop("runtime_sha256", None)
    if runtime_sha256 != canonical_json_sha256(body):
        raise HarnessError("scientific runtime attestation digest mismatch")
    portable = payload.get("portable")
    if not isinstance(portable, dict) or payload.get("portable_sha256") != canonical_json_sha256(portable):
        raise HarnessError("scientific portable-runtime digest mismatch")
    if (
        portable.get("distribution_manifest_schema")
        != RUNTIME_DISTRIBUTION_MANIFEST_SCHEMA
        or portable.get("distribution_manifest_exclusions")
        != RUNTIME_MANIFEST_EXCLUSIONS
    ):
        raise HarnessError("scientific distribution-manifest schema mismatch")
    portable_files = portable.get("portable_distribution_files")
    if not isinstance(portable_files, dict) or set(portable_files) != set(
        RUNTIME_PACKAGES
    ):
        raise HarnessError("scientific portable distribution inventory is incomplete")
    checks = payload.get("isolation_checks")
    required_checks = {
        "isolated",
        "no_site",
        "no_user_site",
        "safe_path",
        "dont_write_bytecode",
        "site_module_not_loaded",
        "customizers_absent_and_not_loaded",
        "pth_files_not_processed",
        "python_environment_absent",
    }
    if (
        not isinstance(checks, dict)
        or set(checks) != required_checks
        or any(checks[key] is not True for key in required_checks)
        or payload.get("isolation_checks_passed") is not True
    ):
        raise HarnessError("scientific runtime isolation checks failed")
    host = payload.get("host")
    if not isinstance(host, dict):
        raise HarnessError("scientific host-runtime attestation is missing")
    manifests = host.get("distribution_installed_files")
    native_summaries = host.get("host_native_distribution_files")
    if (
        not isinstance(manifests, dict)
        or set(manifests) != set(RUNTIME_PACKAGES)
        or not isinstance(native_summaries, dict)
        or set(native_summaries) != set(RUNTIME_PACKAGES)
    ):
        raise HarnessError("scientific installed-file inventory is incomplete")
    for package in RUNTIME_PACKAGES:
        manifest = manifests.get(package)
        if not isinstance(manifest, dict):
            raise HarnessError(f"scientific installed-file manifest is malformed: {package}")
        manifest_body = dict(manifest)
        claimed_manifest_sha256 = manifest_body.pop("manifest_sha256", None)
        if (
            claimed_manifest_sha256 != canonical_json_sha256(manifest_body)
            or manifest_body.get("schema_version")
            != RUNTIME_DISTRIBUTION_MANIFEST_SCHEMA
            or manifest_body.get("package") != package
            or manifest_body.get("exclusion_schema")
            != RUNTIME_MANIFEST_EXCLUSIONS
        ):
            raise HarnessError(f"scientific installed-file manifest digest mismatch: {package}")
        files = manifest_body.get("files")
        if not isinstance(files, list) or not files:
            raise HarnessError(f"scientific installed-file manifest is empty: {package}")
        for row in files:
            if (
                not isinstance(row, dict)
                or set(row)
                != {
                    "declared_path",
                    "relative_to_site_packages",
                    "bytes",
                    "sha256",
                    "classification",
                }
                or not isinstance(row["declared_path"], str)
                or not isinstance(row["relative_to_site_packages"], str)
                or not isinstance(row["bytes"], int)
                or isinstance(row["bytes"], bool)
                or row["bytes"] < 0
                or not _is_sha256(row["sha256"])
                or row["classification"]
                not in {
                    "distribution_metadata",
                    "host_native",
                    "python_source",
                    "package_data",
                }
            ):
                raise HarnessError(f"scientific installed-file row is malformed: {package}")
        portable_rows = [
            row
            for row in files
            if row["classification"] in {"python_source", "package_data"}
        ]
        native_rows = [row for row in files if row["classification"] == "host_native"]
        expected_portable = {
            "file_count": len(portable_rows),
            "files_sha256": canonical_json_sha256(portable_rows),
        }
        expected_native = {
            "file_count": len(native_rows),
            "files_sha256": canonical_json_sha256(native_rows),
        }
        if portable_files.get(package) != expected_portable:
            raise HarnessError(f"scientific portable installed-file digest mismatch: {package}")
        if native_summaries.get(package) != expected_native:
            raise HarnessError(f"scientific native installed-file digest mismatch: {package}")
    flags = host.get("flags", {})
    expected_flags = {
        "isolated": 1,
        "no_site": 1,
        "no_user_site": 1,
        "safe_path": True,
        "dont_write_bytecode": True,
    }
    if any(flags.get(key) != value for key, value in expected_flags.items()):
        raise HarnessError("scientific runtime flags are not -I -B -S")
    if host.get("forbidden_python_environment") != []:
        raise HarnessError("scientific runtime inherited PYTHON* environment")
    if any(row.get("processed") is not False for row in host.get("pth_files", [])):
        raise HarnessError("scientific runtime processed a .pth file")
    if any(
        row.get("loaded") is not False or row.get("discoverable") is not False
        for row in host.get("customizers", [])
    ):
        raise HarnessError("scientific runtime exposes a site customizer")
    return dict(payload)


def capture_runtime_attestation(
    *,
    python: str,
    repo_root: Path,
    runner_path: Path,
    output_path: Path | None = None,
) -> dict[str, Any]:
    bootstrap_path = repo_root.resolve() / "scripts" / "paper2_isolated_bootstrap.py"
    if not bootstrap_path.is_file():
        raise HarnessError("isolated runtime bootstrap is missing from repository")
    argv = [
        python,
        "-I",
        "-B",
        "-S",
        str(bootstrap_path),
        "--repo-root",
        str(repo_root.resolve()),
        "--runner",
        str(runner_path.resolve()),
        "--attest-only",
    ]
    if output_path is not None:
        argv.extend(["--attestation-output", str(output_path.resolve())])
    result = run_capture(
        argv,
        cwd=repo_root,
        env=scientific_child_environment(),
    )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise HarnessError("scientific runtime attestation was not valid JSON") from exc
    validated = validate_runtime_attestation_payload(payload)
    if output_path is not None:
        if not output_path.is_file() or load_json(output_path) != validated:
            raise HarnessError("scientific runtime attestation output differs from stdout")
    return validated


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
    result = run_capture(
        [python, "-I", "-B", "-m", "pip", "freeze", "--all"], check=True
    )
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
        "scientific_runtime": run_dir / "environment" / "scientific_runtime.json",
        "status": run_dir / "status" / "run_status.json",
        "heartbeat": run_dir / "status" / "heartbeat.json",
        "checksums": run_dir / "artifact_checksums.json",
    }


def _assert_empty_or_missing(run_dir: Path) -> None:
    if run_dir.exists() and any(run_dir.iterdir()):
        raise HarnessError(f"run directory is not empty: {run_dir}")


def _expected_tape_sha256(
    seed: int, context: str, split: str, weeks: int
) -> str:
    """Materialize the frozen exogenous tape independently of a remote run."""
    from supply_chain.paper2_bottleneck import materialize_tape

    tape = materialize_tape(int(seed), str(context), str(split), weeks=int(weeks))
    digest = tape.get("threat_sha256")
    if not _is_sha256(digest):
        raise HarnessError("deterministic tape generator returned a malformed digest")
    return str(digest)


def _seed_rows(seed_specs: Sequence[tuple[int, str]], split: str, weeks: int) -> list[dict[str, Any]]:
    if len({seed for seed, _ in seed_specs}) != len(seed_specs):
        raise HarnessError("seed manifest contains duplicate seeds")
    return [
        {
            "seed": int(seed),
            "context": context,
            "split": split,
            "weeks": int(weeks),
            "expected_tape_sha256": _expected_tape_sha256(
                int(seed), context, split, int(weeks)
            ),
            "opened_status": "BURNED_DEVELOPMENT_OR_CORRECTIVE",
        }
        for seed, context in seed_specs
    ]


def _validate_seed_tape_commitments(
    rows: Any,
    *,
    regenerate: bool,
) -> list[str]:
    """Require ordered pre-execution commitments to each deterministic tape."""
    failures: list[str] = []
    if not isinstance(rows, list):
        return ["seed tape commitments are missing or malformed"]
    identities: list[tuple[Any, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            failures.append(f"malformed seed tape commitment at index {index}")
            continue
        seed = row.get("seed")
        context = row.get("context")
        split = row.get("split")
        weeks = row.get("weeks")
        expected = row.get("expected_tape_sha256")
        if (
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or not isinstance(context, str)
            or not isinstance(split, str)
            or isinstance(weeks, bool)
            or not isinstance(weeks, int)
            or not _is_sha256(expected)
        ):
            failures.append(f"malformed seed tape commitment at index {index}")
            continue
        identities.append((seed, expected))
        if regenerate:
            try:
                regenerated = _expected_tape_sha256(seed, context, split, weeks)
            except (HarnessError, KeyError, TypeError, ValueError) as exc:
                failures.append(
                    f"deterministic tape regeneration failed at index {index}: {exc}"
                )
                continue
            if regenerated != expected:
                failures.append(
                    f"deterministic tape commitment mismatch at index {index}"
                )
    if len(identities) == len(rows) and len(set(identities)) != len(rows):
        failures.append("ordered deterministic tape commitments are not unique")
    return failures


def _isolated_runner_prefix(
    *, runner_rel: str, item_id: str, execution_role: str
) -> list[str]:
    """Build the frozen stdlib-bootstrap prefix for one scientific child."""
    return [
        "{python}",
        "-I",
        "-B",
        "-S",
        "{repo_root}/scripts/paper2_isolated_bootstrap.py",
        "--repo-root",
        "{repo_root}",
        "--runner",
        f"{{repo_root}}/{runner_rel}",
        "--attestation-output",
        f"{{run_dir}}/runtime_environment/{item_id}.scientific_runtime.json",
        "--expected-runtime-sha256",
        "{host_runtime_sha256}",
        "--execution-nonce",
        "{execution_nonce}",
        "--execution-role",
        execution_role,
        "--",
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
            *_isolated_runner_prefix(
                runner_rel=runner_rel,
                item_id=seed_id,
                execution_role="non_scientific_smoke",
            ),
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
                "execution_role": "non_scientific_smoke",
                "runtime_attestation_relative": (
                    f"runtime_environment/{seed_id}.scientific_runtime.json"
                ),
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
    argv = _isolated_runner_prefix(
        runner_rel=runner_rel,
        item_id=role,
        execution_role=role,
    )
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
            "execution_role": role,
            "runtime_attestation_relative": (
                f"runtime_environment/{role}.scientific_runtime.json"
            ),
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
        *_isolated_runner_prefix(
            runner_rel=runner_rel,
            item_id=job_id,
            execution_role=f"frontier_{phase}",
        ),
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
            "execution_role": f"frontier_{phase}",
            "runtime_attestation_relative": (
                f"runtime_environment/{job_id}.scientific_runtime.json"
            ),
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
    try:
        rel, path = _confined_checksum_path(repo_root, relative)
    except HarnessError as exc:
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
        "reduced_horizon_key_v4_certified",
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
    if authorization.get("key_schema_version") != "paper2_bottleneck_semantic_markov_key_v4":
        failures.append("authorization does not require semantic Markov key v4")
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
        "isolated_bootstrap_sha256": run_manifest["inputs"][
            "isolated_bootstrap_sha256"
        ],
        "host_runtime_sha256": run_manifest["inputs"]["host_runtime_sha256"],
        "portable_runtime_sha256": run_manifest["inputs"][
            "portable_runtime_sha256"
        ],
        "scientific_child_environment_sha256": run_manifest["inputs"][
            "scientific_child_environment_sha256"
        ],
        "execution_nonce": run_manifest["inputs"]["execution_nonce"],
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
                validate_reduced_certification_structure,
            )
            failures.extend(
                validate_reduced_certification_structure(
                    payload,
                    str(row.get("role")),
                    expected_environment_sha256=run_manifest["inputs"].get(
                        "environment_sha256"
                    ),
                )
            )
            pair_row = row.get("independent_execution_verification")
            if not isinstance(pair_row, dict):
                failures.append(
                    f"authorization lacks independent execution verification: {row.get('role')}"
                )
            else:
                _pair_path, _pair_payload = _tracked_archive_artifact(
                    repo_root, pair_row["path"], pair_row.get("sha256")
                )
                producer_fingerprint = pair_row.get(
                    "producer_public_key_fingerprint"
                )
                independent_fingerprint = pair_row.get(
                    "independent_public_key_fingerprint"
                )
                if not _is_sha256(producer_fingerprint) or not _is_sha256(
                    independent_fingerprint
                ):
                    failures.append(
                        f"authorization lacks caller-retained fingerprints: {row.get('role')}"
                    )
                else:
                    archive_reverification = reverify_authorized_reduced_pair_archive(
                        archive_root=repo_root,
                        pair_verification_path=_pair_path,
                        expected_pair_verification_sha256=str(
                            pair_row.get("sha256")
                        ),
                        verification_mode=str(
                            pair_row.get("verification_mode", "")
                        ),
                        role=str(row.get("role")),
                        certified_artifact_sha256=str(row.get("sha256")),
                        expected_producer_public_key_fingerprint=str(
                            producer_fingerprint
                        ),
                        expected_independent_public_key_fingerprint=str(
                            independent_fingerprint
                        ),
                        expected_producer_manifest_sha256=pair_row.get(
                            "producer_manifest_sha256"
                        ),
                        expected_independent_manifest_sha256=pair_row.get(
                            "independent_manifest_sha256"
                        ),
                    )
                    failures.extend(
                        f"archived pair reverification: {failure}"
                        for failure in archive_reverification.get("failures", [])
                    )
                    if archive_reverification.get("passed") is not True:
                        failures.append(
                            f"archived pair reverification failed: {row.get('role')}"
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
        "key_schema_version": "paper2_bottleneck_semantic_markov_key_v4",
        "reduced_horizon_key_v4_certified": False,
        "full_horizon_primary_acceleration_authorized": False,
        "reduced_horizon_certification_artifacts": [
            {
                "role": "w12_five_tape",
                "path": "results/paper2_bottleneck/exact_transducer_certification_w12.json",
                "sha256": "FILL_AFTER_VERIFICATION",
                "source_git_commit": "FILL_FROM_CERTIFICATE_PROVENANCE",
                "independent_execution_verification": {
                    "verification_mode": PORTABLE_PAIR_VERIFICATION_MODE,
                    "path": "results/paper2_bottleneck/exact_transducer_certification_w12_pair_verification.json",
                    "sha256": "FILL_AFTER_TWO_PROCESS_CUSTODY_VERIFICATION",
                    "producer_manifest_sha256": "FILL_FROM_CALLER_RETAINED_TRANSFER_MANIFEST",
                    "independent_manifest_sha256": "FILL_FROM_CALLER_RETAINED_TRANSFER_MANIFEST",
                    "producer_public_key_fingerprint": "FILL_FROM_CALLER_RETAINED_PRELAUNCH_RECORD",
                    "independent_public_key_fingerprint": "FILL_FROM_CALLER_RETAINED_PRELAUNCH_RECORD",
                },
            },
            {
                "role": "w16_hard_tape",
                "path": "results/paper2_bottleneck/exact_transducer_certification_w16_hard.json",
                "sha256": "FILL_AFTER_VERIFICATION",
                "source_git_commit": "FILL_FROM_CERTIFICATE_PROVENANCE",
                "independent_execution_verification": {
                    "verification_mode": PORTABLE_PAIR_VERIFICATION_MODE,
                    "path": "results/paper2_bottleneck/exact_transducer_certification_w16_hard_pair_verification.json",
                    "sha256": "FILL_AFTER_TWO_PROCESS_CUSTODY_VERIFICATION",
                    "producer_manifest_sha256": "FILL_FROM_CALLER_RETAINED_TRANSFER_MANIFEST",
                    "independent_manifest_sha256": "FILL_FROM_CALLER_RETAINED_TRANSFER_MANIFEST",
                    "producer_public_key_fingerprint": "FILL_FROM_CALLER_RETAINED_PRELAUNCH_RECORD",
                    "independent_public_key_fingerprint": "FILL_FROM_CALLER_RETAINED_PRELAUNCH_RECORD",
                },
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
        "isolated_bootstrap_sha256": run_manifest["inputs"][
            "isolated_bootstrap_sha256"
        ],
        "host_runtime_sha256": run_manifest["inputs"]["host_runtime_sha256"],
        "portable_runtime_sha256": run_manifest["inputs"][
            "portable_runtime_sha256"
        ],
        "scientific_child_environment_sha256": run_manifest["inputs"][
            "scientific_child_environment_sha256"
        ],
        "execution_nonce": run_manifest["inputs"]["execution_nonce"],
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
    for required in (contract_path, runner_path, HARNESS_PATH, ISOLATED_BOOTSTRAP):
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
        if contract.get("acceleration_proof", {}).get("required_key_schema") != "paper2_bottleneck_semantic_markov_key_v4":
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
    critical_paths = [contract_path, runner_path, HARNESS_PATH, ISOLATED_BOOTSTRAP]
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
    local_runtime_attestation = capture_runtime_attestation(
        python=sys.executable,
        repo_root=repo_root,
        runner_path=runner_path,
        output_path=paths["scientific_runtime"],
    )
    execution_nonce = secrets.token_hex(32)
    sanitized_environment = scientific_child_environment()
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
        "trust_model": (
            "caller retains this receipt digest out of band; no claim against "
            "malicious deletion of the trusted local custody store without signing or TSA"
        ),
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
            "isolated_bootstrap_relative": relative_to_root(
                ISOLATED_BOOTSTRAP, repo_root
            ),
            "isolated_bootstrap_sha256": sha256_file(ISOLATED_BOOTSTRAP),
            "seed_manifest_sha256": seed_hash,
            "command_manifest_sha256": command_hash,
            "dependency_snapshot_sha256": sha256_file(paths["dependencies"]),
            "machine_snapshot_sha256": sha256_file(paths["machine"]),
            "environment": scientific_environment,
            "environment_sha256": scientific_environment["environment_sha256"],
            "environment_source": scientific_environment_source,
            "environment_source_sha256": scientific_environment_source_sha256,
            "host_runtime_attestation_relative": str(
                paths["scientific_runtime"].relative_to(run_dir)
            ),
            "host_runtime_sha256": local_runtime_attestation["runtime_sha256"],
            "portable_runtime_sha256": local_runtime_attestation[
                "portable_sha256"
            ],
            "scientific_child_environment": sanitized_environment,
            "scientific_child_environment_sha256": canonical_json_sha256(
                sanitized_environment
            ),
            "execution_nonce": execution_nonce,
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
    stored_runtime = validate_runtime_attestation_payload(
        load_json(run_dir / expected["host_runtime_attestation_relative"])
    )
    if stored_runtime.get("runtime_sha256") != expected.get("host_runtime_sha256"):
        raise HarnessError("prepared host-runtime attestation digest mismatch")
    if stored_runtime.get("portable_sha256") != expected.get(
        "portable_runtime_sha256"
    ):
        raise HarnessError("prepared portable-runtime attestation digest mismatch")
    if stored_runtime.get("portable", {}).get(
        "bootstrap_sha256"
    ) != expected.get("isolated_bootstrap_sha256"):
        raise HarnessError("prepared runtime does not bind the isolated bootstrap")
    if stored_runtime.get("portable", {}).get("runner_sha256") != expected.get(
        "runner_sha256"
    ):
        raise HarnessError("prepared runtime does not bind the scientific runner")
    if expected.get("scientific_child_environment") != scientific_child_environment():
        raise HarnessError("sanitized scientific child environment changed")
    if expected.get("scientific_child_environment_sha256") != canonical_json_sha256(
        scientific_child_environment()
    ):
        raise HarnessError("sanitized scientific child environment digest mismatch")
    if not _is_sha256(expected.get("execution_nonce")):
        raise HarnessError("prepared execution nonce is missing or malformed")
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
        "isolated_bootstrap_sha256": sha256_file(
            repo_root / expected["isolated_bootstrap_relative"]
        ),
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
    tape_failures = _validate_seed_tape_commitments(
        seed_manifest.get("seeds"), regenerate=True
    )
    if tape_failures:
        raise HarnessError(
            "deterministic tape commitment failed closed: "
            + "; ".join(tape_failures)
        )
    mode = run_manifest.get("mode")
    if mode in EXECUTABLE_EVIDENCE_MODES:
        critical_paths = [
            repo_root / expected["contract_relative"],
            repo_root / expected["runner_relative"],
            repo_root / expected["harness_relative"],
            repo_root / expected["isolated_bootstrap_relative"],
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


def _argv_materialization_context(
    *,
    repo_root: Path,
    run_dir: Path,
    host_runtime_sha256: str | None = None,
    execution_nonce: str | None = None,
) -> dict[str, str]:
    if host_runtime_sha256 is None or execution_nonce is None:
        manifest_path = run_dir / "run_manifest.json"
        if manifest_path.is_file():
            inputs = load_json(manifest_path).get("inputs", {})
            host_runtime_sha256 = host_runtime_sha256 or inputs.get(
                "host_runtime_sha256"
            )
            execution_nonce = execution_nonce or inputs.get("execution_nonce")
    if not _is_sha256(host_runtime_sha256) or not _is_sha256(execution_nonce):
        raise HarnessError("argv materialization lacks host-runtime digest or execution nonce")
    return {
        "python_executable": sys.executable,
        "repository_root": str(repo_root.resolve()),
        "run_directory": str(run_dir.resolve()),
        "host_runtime_sha256": str(host_runtime_sha256),
        "execution_nonce": str(execution_nonce),
    }


def _materialize_argv_from_context(
    template: Sequence[str], context: Any
) -> list[str]:
    expected_keys = {
        "python_executable",
        "repository_root",
        "run_directory",
        "host_runtime_sha256",
        "execution_nonce",
    }
    if (
        not isinstance(context, dict)
        or set(context) != expected_keys
        or any(
            not isinstance(context[key], str) or not context[key]
            for key in expected_keys
        )
    ):
        raise HarnessError("materialized argv context is malformed")
    if not _is_sha256(context["host_runtime_sha256"]) or not _is_sha256(
        context["execution_nonce"]
    ):
        raise HarnessError("materialized argv runtime digest or nonce is malformed")
    replacements = {
        "{python}": context["python_executable"],
        "{repo_root}": context["repository_root"],
        "{run_dir}": context["run_directory"],
        "{host_runtime_sha256}": context["host_runtime_sha256"],
        "{execution_nonce}": context["execution_nonce"],
    }
    output = []
    for item in template:
        if not isinstance(item, str):
            raise HarnessError("materialized argv template is malformed")
        value = item
        for token, replacement in replacements.items():
            value = value.replace(token, replacement)
        output.append(value)
    return output


def _materialize_argv(
    template: Sequence[str],
    *,
    repo_root: Path,
    run_dir: Path,
    host_runtime_sha256: str = "0" * 64,
    execution_nonce: str = "0" * 64,
) -> list[str]:
    return _materialize_argv_from_context(
        template,
        _argv_materialization_context(
            repo_root=repo_root,
            run_dir=run_dir,
            host_runtime_sha256=host_runtime_sha256,
            execution_nonce=execution_nonce,
        ),
    )


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


def _validate_prelaunch_ack(
    acknowledgement: Any, prelaunch: Mapping[str, Any]
) -> dict[str, Any]:
    expected = {
        "schema_version": SIGNED_PRELAUNCH_ACK_SCHEMA,
        "prelaunch_record_sha256": prelaunch["prelaunch_record_sha256"],
        "public_key_fingerprint": prelaunch["public_key_fingerprint"],
        "authorization_sha256": prelaunch["authorization_sha256"],
        "host_runtime_sha256": prelaunch["host_runtime_sha256"],
        "acknowledged_before_child_launch": True,
    }
    if not isinstance(acknowledgement, dict) or acknowledgement != expected:
        raise HarnessError(
            "signed reduced launch requires an exact out-of-band prelaunch acknowledgement"
        )
    return dict(acknowledgement)


def execute_reduced_signed_session(
    *,
    custody_root: Path,
    role: str,
    execution_role: str,
    replay_pair_id: str,
    weeks: int,
    seeds: Sequence[tuple[int, str]],
    split: str,
    workers: int,
    output_path: Path,
    authorization_path: Path,
    exact_receipt_path: Path,
    runtime_attestation_path: Path,
    harness_receipt_path: Path,
    host_runtime_sha256: str,
    portable_runtime_sha256: str,
    harness_execution_nonce: str,
    acknowledgement_callback: Callable[[dict[str, Any]], Any],
    non_scientific_smoke: bool = False,
    max_calendars: int | None = None,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    """Create a live signer, pause for external retention, then launch once.

    The callback is the trust boundary.  A scientific CLI callback writes the
    prelaunch record to stdout and blocks on stdin; it is not allowed to infer
    or synthesize the acknowledgement from the run tree.
    """
    from scripts.run_paper2_bottleneck_exact_transducer import (
        EphemeralOpenSSLEd25519Signer,
        create_reduced_execution_launch_authorization,
        launch_reduced_execution_fresh_process,
        verify_reduced_execution_receipt_signature,
    )

    custody_root = custody_root.absolute()
    paths = {
        "output": output_path.absolute(),
        "authorization": authorization_path.absolute(),
        "exact_receipt": exact_receipt_path.absolute(),
        "runtime_attestation": runtime_attestation_path.absolute(),
        "harness_receipt": harness_receipt_path.absolute(),
    }
    if len(set(paths.values())) != len(paths):
        raise HarnessError("signed reduced custody paths are not distinct")
    for label, path in paths.items():
        try:
            path.relative_to(custody_root)
        except ValueError as exc:
            raise HarnessError(f"signed reduced {label} path escapes custody root") from exc
        if path.exists() or path.is_symlink():
            raise HarnessError(f"signed reduced {label} path already exists")
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    for name, value in (
        ("replay_pair_id", replay_pair_id),
        ("host_runtime_sha256", host_runtime_sha256),
        ("portable_runtime_sha256", portable_runtime_sha256),
        ("harness_execution_nonce", harness_execution_nonce),
    ):
        if not _is_sha256(value):
            raise HarnessError(f"signed reduced {name} is malformed")
    if execution_role not in {"producer", "independent_replay"}:
        raise HarnessError("signed reduced execution role is malformed")
    if not callable(acknowledgement_callback):
        raise HarnessError("signed reduced launch lacks an acknowledgement callback")

    signer = EphemeralOpenSSLEd25519Signer()
    try:
        authorization = create_reduced_execution_launch_authorization(
            paths["authorization"],
            paths["output"],
            paths["exact_receipt"],
            role=role,
            execution_role=execution_role,
            replay_pair_id=replay_pair_id,
            weeks=weeks,
            seeds=seeds,
            split=split,
            workers=workers,
            non_scientific_smoke=non_scientific_smoke,
            max_calendars=max_calendars,
            receipt_signer=signer,
            launch_mode="isolated_bootstrap",
            custody_root=custody_root,
            isolated_bootstrap_path=ISOLATED_BOOTSTRAP,
            runtime_attestation_path=paths["runtime_attestation"],
            host_runtime_sha256=host_runtime_sha256,
            portable_runtime_sha256=portable_runtime_sha256,
            scientific_child_environment=scientific_child_environment(),
            harness_execution_nonce=harness_execution_nonce,
            parent_launcher_path=HARNESS_PATH,
        )
        prelaunch_body = {
            "schema_version": SIGNED_PRELAUNCH_SCHEMA,
            "issued_at_utc": utc_now(),
            "role": role,
            "execution_role": execution_role,
            "replay_pair_id": replay_pair_id,
            "authorization_path": str(paths["authorization"]),
            "authorization_sha256": authorization["authorization_sha256"],
            "public_key_fingerprint": authorization[
                "prelaunch_signing_public_key_fingerprint"
            ],
            "host_runtime_sha256": host_runtime_sha256,
            "portable_runtime_sha256": portable_runtime_sha256,
            "harness_execution_nonce": harness_execution_nonce,
            "materialized_argv_sha256": canonical_json_sha256(
                authorization["materialized_argv"]
            ),
            "private_key_parent_memory_only": True,
            "child_launch_has_not_occurred": True,
            "caller_must_retain_before_ack": True,
        }
        prelaunch = {
            **prelaunch_body,
            "prelaunch_record_sha256": canonical_json_sha256(prelaunch_body),
        }
        acknowledgement = _validate_prelaunch_ack(
            acknowledgement_callback(dict(prelaunch)), prelaunch
        )
        launch = launch_reduced_execution_fresh_process(
            paths["authorization"],
            expected_authorization_sha256=authorization["authorization_sha256"],
            receipt_signer=signer,
            expected_public_key_fingerprint=prelaunch["public_key_fingerprint"],
            timeout_seconds=timeout_seconds,
        )
    except Exception:
        signer.close()
        raise
    if launch.get("passed") is not True:
        raise HarnessError(
            "signed reduced exact launch failed: "
            + "; ".join(str(value) for value in launch.get("failures", []))
        )
    exact_receipt = load_json(paths["exact_receipt"])
    signature_failures = verify_reduced_execution_receipt_signature(
        exact_receipt, prelaunch["public_key_fingerprint"]
    )
    if signature_failures:
        raise HarnessError(
            "signed reduced exact receipt failed verification: "
            + "; ".join(signature_failures)
        )
    harness_receipt_body = {
        "schema_version": SIGNED_HARNESS_RECEIPT_SCHEMA,
        "written_at_utc": utc_now(),
        "role": role,
        "execution_role": execution_role,
        "replay_pair_id": replay_pair_id,
        "parent_launcher_path": str(HARNESS_PATH),
        "parent_launcher_sha256": sha256_file(HARNESS_PATH),
        "isolated_bootstrap_path": str(ISOLATED_BOOTSTRAP),
        "isolated_bootstrap_sha256": sha256_file(ISOLATED_BOOTSTRAP),
        "scientific_child_environment": scientific_child_environment(),
        "scientific_child_environment_sha256": canonical_json_sha256(
            scientific_child_environment()
        ),
        "harness_execution_nonce": harness_execution_nonce,
        "host_runtime_sha256": host_runtime_sha256,
        "portable_runtime_sha256": portable_runtime_sha256,
        "prelaunch_record": prelaunch,
        "prelaunch_acknowledgement": acknowledgement,
        "prelaunch_acknowledgement_sha256": canonical_json_sha256(acknowledgement),
        "external_ack_received_before_child_launch": True,
        "caller_retained_fingerprint_required_for_reverification": True,
        "exact_authorization_path": str(paths["authorization"]),
        "exact_authorization_sha256": authorization["authorization_sha256"],
        "exact_materialized_argv": authorization["materialized_argv"],
        "exact_materialized_argv_sha256": canonical_json_sha256(
            authorization["materialized_argv"]
        ),
        "exact_execution_identity": authorization["execution_identity"],
        "exact_output_path": str(paths["output"]),
        "exact_output_sha256": launch["output_sha256"],
        "exact_execution_receipt_path": str(paths["exact_receipt"]),
        "exact_execution_receipt_sha256": launch["execution_receipt_sha256"],
        "runtime_attestation_path": str(paths["runtime_attestation"]),
        "runtime_attestation_file_sha256": launch[
            "runtime_attestation_file_sha256"
        ],
        "fresh_child_process": True,
        "child_pid": launch["child_pid"],
        "returncode": launch["returncode"],
        "exact_signed_receipt_verified_before_harness_receipt": True,
        "written_after_child_exit": True,
        "cross_tree_read_isolation": "NOT_ENFORCED_TRUSTED_PINNED_RUNNER_RESIDUAL",
        "evidence": False,
    }
    harness_receipt = {
        **harness_receipt_body,
        "harness_receipt_body_sha256": canonical_json_sha256(harness_receipt_body),
    }
    exclusive_write_json(paths["harness_receipt"], harness_receipt)
    return {
        "passed": True,
        "prelaunch": prelaunch,
        "launch": launch,
        "harness_receipt_path": str(paths["harness_receipt"]),
        "harness_receipt_sha256": sha256_file(paths["harness_receipt"]),
        "evidence": False,
        "evidence_status": "SIGNED_EXECUTION_COMPLETE_PAIR_REPLAY_PENDING",
    }


def execute_prepared_reduced_signed_session(
    *,
    run_dir: Path,
    execution_role: str,
    replay_pair_id: str,
    acknowledgement_callback: Callable[[dict[str, Any]], Any],
    timeout_seconds: float | None = None,
    repo_root: Path = ROOT,
) -> dict[str, Any]:
    """Execute one frozen W12/W16 manifest through the interactive signer."""
    run_dir = run_dir.absolute()
    run_manifest, seed_manifest, command_manifest = _validate_prepared_inputs(
        run_dir, repo_root
    )
    mode = run_manifest.get("mode")
    if mode not in REDUCED_MODE_ROLE:
        raise HarnessError("signed reduced session requires a frozen W12/W16 run")
    commands = command_manifest.get("commands")
    if not isinstance(commands, list) or len(commands) != 1:
        raise HarnessError("signed reduced session requires exactly one command")
    command = commands[0]
    role = REDUCED_MODE_ROLE[mode]
    if command.get("evidence_role") != role:
        raise HarnessError("signed reduced command role differs from frozen profile")
    rows = seed_manifest.get("seeds")
    if not isinstance(rows, list) or not rows:
        raise HarnessError("signed reduced seed manifest is empty")
    seeds = [(int(row["seed"]), str(row["context"])) for row in rows]
    runtime_relative = command.get("runtime_attestation_relative")
    if not isinstance(runtime_relative, str):
        raise HarnessError("signed reduced runtime path is missing")
    custody_dir = run_dir / "signed_custody" / execution_role
    return execute_reduced_signed_session(
        custody_root=run_dir,
        role=role,
        execution_role=execution_role,
        replay_pair_id=replay_pair_id,
        weeks=int(run_manifest["execution"]["weeks"]),
        seeds=seeds,
        split=str(run_manifest["execution"]["split"]),
        workers=int(run_manifest["execution"]["runner_workers_per_seed"]),
        output_path=run_dir / str(command["output_relative"]),
        authorization_path=custody_dir / "launch_authorization.json",
        exact_receipt_path=custody_dir / "exact_execution_receipt.json",
        runtime_attestation_path=run_dir / runtime_relative,
        harness_receipt_path=custody_dir / "harness_execution_receipt.json",
        host_runtime_sha256=str(run_manifest["inputs"]["host_runtime_sha256"]),
        portable_runtime_sha256=str(
            run_manifest["inputs"]["portable_runtime_sha256"]
        ),
        harness_execution_nonce=str(run_manifest["inputs"]["execution_nonce"]),
        acknowledgement_callback=acknowledgement_callback,
        timeout_seconds=timeout_seconds,
    )


def _interactive_prelaunch_acknowledgement(
    prelaunch: dict[str, Any]
) -> dict[str, Any]:
    print("PAPER2_PRELAUNCH " + json.dumps(prelaunch, sort_keys=True), flush=True)
    line = sys.stdin.readline()
    if not line:
        raise HarnessError("prelaunch acknowledgement channel closed before ACK")
    try:
        acknowledgement = json.loads(line)
    except json.JSONDecodeError as exc:
        raise HarnessError("prelaunch acknowledgement is not valid JSON") from exc
    if not isinstance(acknowledgement, dict):
        raise HarnessError("prelaunch acknowledgement is not a JSON object")
    return acknowledgement


def create_reduced_signed_transfer_manifest(
    *,
    launch_root: Path,
    harness_receipt_path: Path,
    expected_harness_receipt_sha256: str,
    role: str,
    expected_public_key_fingerprint: str,
    manifest_path: Path,
) -> dict[str, Any]:
    """Verify launch-host inode custody, then freeze a relocatable hash index."""
    from scripts.run_paper2_bottleneck_exact_transducer import (
        _load_bound_execution,
    )

    launch_root = launch_root.resolve()

    def checked_launch_descendant(path: Path, label: str) -> tuple[str, Path]:
        lexical = path.absolute()
        try:
            relative = str(lexical.relative_to(launch_root))
        except ValueError as exc:
            raise HarnessError(f"{label} escapes launch root") from exc
        canonical, checked = _confined_checksum_path(launch_root, relative)
        if checked != lexical:
            raise HarnessError(f"{label} is not a canonical launch-root descendant")
        return canonical, checked

    harness_relative, harness_receipt_path = checked_launch_descendant(
        harness_receipt_path, "harness receipt"
    )
    _manifest_relative, manifest_path = checked_launch_descendant(
        manifest_path, "transfer manifest"
    )
    if manifest_path.exists() or manifest_path.is_symlink():
        raise HarnessError("signed transfer manifest path already exists")
    if harness_receipt_path.is_symlink() or not harness_receipt_path.is_file():
        raise HarnessError("signed harness receipt is missing or a symlink")
    if not _is_sha256(expected_harness_receipt_sha256) or sha256_file(
        harness_receipt_path
    ) != expected_harness_receipt_sha256:
        raise HarnessError("caller-retained harness receipt digest mismatch")
    if not _is_sha256(expected_public_key_fingerprint):
        raise HarnessError("caller-retained signing fingerprint is malformed")
    harness_receipt = load_json(harness_receipt_path)
    if harness_receipt.get("schema_version") != SIGNED_HARNESS_RECEIPT_SCHEMA:
        raise HarnessError("signed harness receipt schema mismatch")
    output_path = Path(str(harness_receipt.get("exact_output_path", ""))).absolute()
    runtime_path = Path(
        str(harness_receipt.get("runtime_attestation_path", ""))
    ).absolute()
    authorization_path = Path(
        str(harness_receipt.get("exact_authorization_path", ""))
    ).absolute()
    exact_receipt_path = Path(
        str(harness_receipt.get("exact_execution_receipt_path", ""))
    ).absolute()
    _output_relative, output_path = checked_launch_descendant(
        output_path, "signed output"
    )
    _runtime_relative, runtime_path = checked_launch_descendant(
        runtime_path, "runtime attestation"
    )
    _authorization_relative, authorization_path = checked_launch_descendant(
        authorization_path, "launch authorization"
    )
    _exact_receipt_relative, exact_receipt_path = checked_launch_descendant(
        exact_receipt_path, "exact execution receipt"
    )
    preverified_exact_receipt = load_json(exact_receipt_path)
    for stream_name in ("stdout", "stderr"):
        checked_launch_descendant(
            Path(
                str(preverified_exact_receipt.get(f"{stream_name}_path", ""))
            ),
            f"{stream_name} custody log",
        )
    _output, exact_receipt, launch_failures = _load_bound_execution(
        output_path,
        harness_receipt_path,
        expected_output_sha256=str(harness_receipt.get("exact_output_sha256", "")),
        expected_receipt_sha256=expected_harness_receipt_sha256,
        expected_authorization_sha256=str(
            harness_receipt.get("exact_authorization_sha256", "")
        ),
        expected_public_key_fingerprint=expected_public_key_fingerprint,
        runtime_attestation_path=runtime_path,
        expected_runtime_attestation_sha256=str(
            harness_receipt.get("runtime_attestation_file_sha256", "")
        ),
        role=role,
        label="launch_host",
    )
    if launch_failures:
        raise HarnessError(
            "launch-host exact custody verification failed: "
            + "; ".join(launch_failures)
        )
    authorization = load_json(authorization_path)
    transport_plan = authorization.get("transport_archive_plan")
    if (
        not isinstance(transport_plan, dict)
        or transport_plan.get("schema_version")
        != "paper2_reduced_transport_archive_plan_v1"
        or exact_receipt.get("transport_archive_plan") != transport_plan
        or transport_plan.get("custody_root") != str(launch_root)
    ):
        raise HarnessError("signed exact transport archive plan is invalid")
    relative_artifacts = transport_plan.get("relative_artifacts")
    if not isinstance(relative_artifacts, dict) or set(relative_artifacts) != {
        "authorization",
        "output",
        "runtime_attestation",
        "exact_receipt",
        "stdout_log",
        "stderr_log",
    }:
        raise HarnessError("signed exact relative-artifact plan is malformed")

    expected_launch_paths = {
        "authorization": authorization_path,
        "output": output_path,
        "runtime_attestation": runtime_path,
        "exact_receipt": exact_receipt_path,
        "stdout_log": Path(str(exact_receipt.get("stdout_path", ""))).absolute(),
        "stderr_log": Path(str(exact_receipt.get("stderr_path", ""))).absolute(),
    }
    signed_hashes = {
        "authorization": str(exact_receipt.get("authorization_sha256", "")),
        "output": str(exact_receipt.get("output_sha256", "")),
        "runtime_attestation": str(
            exact_receipt.get("runtime_attestation_file_sha256", "")
        ),
        "stdout_log": str(exact_receipt.get("stdout_sha256", "")),
        "stderr_log": str(exact_receipt.get("stderr_sha256", "")),
    }
    artifacts: dict[str, Any] = {}
    for label, launch_path in expected_launch_paths.items():
        relative = str(relative_artifacts.get(label, ""))
        _canonical, planned = _confined_checksum_path(launch_root, relative)
        _launch_relative, checked_launch_path = checked_launch_descendant(
            launch_path, f"signed transport artifact {label}"
        )
        if planned != checked_launch_path:
            raise HarnessError(f"signed transport plan path mismatch: {label}")
        actual_sha256 = sha256_file(planned)
        signed_sha256 = (
            sha256_file(exact_receipt_path)
            if label == "exact_receipt"
            else signed_hashes[label]
        )
        if not _is_sha256(signed_sha256) or actual_sha256 != signed_sha256:
            raise HarnessError(f"signed transport content mismatch: {label}")
        artifacts[label] = {
            "relative_path": relative,
            "sha256": actual_sha256,
            "bytes": planned.stat().st_size,
            "hash_signed_inside_exact_receipt": label != "exact_receipt",
        }
    artifacts["harness_receipt"] = {
        "relative_path": harness_relative,
        "sha256": expected_harness_receipt_sha256,
        "bytes": harness_receipt_path.stat().st_size,
        "hash_signed_inside_exact_receipt": False,
    }
    body = {
        "schema_version": SIGNED_TRANSFER_MANIFEST_SCHEMA,
        "created_at_utc": utc_now(),
        "role": role,
        "execution_role": exact_receipt.get("execution_role"),
        "replay_pair_id": exact_receipt.get("replay_pair_id"),
        "public_key_fingerprint": expected_public_key_fingerprint,
        "signed_transport_archive_plan": transport_plan,
        "artifacts": artifacts,
        "exact_receipt_signature_scheme": exact_receipt.get(
            "receipt_signature_scheme"
        ),
        "exact_receipt_signed_body_sha256": exact_receipt.get(
            "receipt_signed_body_sha256"
        ),
        "exact_receipt_signature_ed25519": exact_receipt.get(
            "receipt_signature_ed25519"
        ),
        "launch_host_absolute_path_and_inode_verification_passed": True,
        "retrieval_must_use_relative_paths_and_content_hashes": True,
        "manifest_requires_caller_retained_sha256": True,
        "evidence": False,
    }
    manifest = {**body, "manifest_body_sha256": canonical_json_sha256(body)}
    exclusive_write_json(manifest_path, manifest)
    return {
        "passed": True,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "public_key_fingerprint": expected_public_key_fingerprint,
        "launch_host_inode_verification_passed": True,
        "evidence": False,
    }


def verify_retrieved_reduced_transfer(
    *,
    retrieved_root: Path,
    manifest_relative: str,
    expected_manifest_sha256: str,
    expected_public_key_fingerprint: str,
) -> dict[str, Any]:
    """Verify relocated bytes without replaying launch-host inode assertions."""
    from scripts.run_paper2_bottleneck_exact_transducer import (
        _execution_authorization_failures,
        _execution_field_failures,
        _execution_identity_from_authorization,
        _reduced_execution_witness,
        verify_reduced_execution_receipt_signature,
    )

    retrieved_root = retrieved_root.resolve()
    _canonical_manifest, manifest_path = _confined_checksum_path(
        retrieved_root, manifest_relative
    )
    failures: list[str] = []
    if manifest_path.is_symlink() or not manifest_path.is_file():
        return {
            "schema_version": RETRIEVED_TRANSFER_VERIFICATION_SCHEMA,
            "passed": False,
            "failures": ["retrieved transfer manifest is missing or a symlink"],
        }
    if not _is_sha256(expected_manifest_sha256) or sha256_file(
        manifest_path
    ) != expected_manifest_sha256:
        failures.append("caller-retained transfer-manifest digest mismatch")
    manifest = load_json(manifest_path)
    body = dict(manifest) if isinstance(manifest, dict) else {}
    claimed_manifest_body_sha256 = body.pop("manifest_body_sha256", None)
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != SIGNED_TRANSFER_MANIFEST_SCHEMA
        or claimed_manifest_body_sha256 != canonical_json_sha256(body)
    ):
        failures.append("retrieved transfer-manifest body is invalid")
        manifest = {}
    if manifest.get("public_key_fingerprint") != expected_public_key_fingerprint:
        failures.append("caller-retained transfer signing fingerprint mismatch")
    artifacts = manifest.get("artifacts")
    required_artifacts = {
        "authorization",
        "output",
        "runtime_attestation",
        "exact_receipt",
        "stdout_log",
        "stderr_log",
        "harness_receipt",
    }
    if not isinstance(artifacts, dict) or set(artifacts) != required_artifacts:
        failures.append("retrieved transfer artifact index is malformed")
        artifacts = {}
    relocated: dict[str, tuple[Path, bytes]] = {}
    for label in required_artifacts:
        row = artifacts.get(label)
        if not isinstance(row, dict):
            failures.append(f"retrieved transfer artifact is missing: {label}")
            continue
        try:
            _canonical, path = _confined_checksum_path(
                retrieved_root, row.get("relative_path")
            )
        except HarnessError as exc:
            failures.append(f"retrieved transfer artifact path invalid: {label}: {exc}")
            continue
        if path.is_symlink() or not path.is_file():
            failures.append(f"retrieved transfer artifact missing/symlink: {label}")
            continue
        data = path.read_bytes()
        if (
            not _is_sha256(row.get("sha256"))
            or hashlib.sha256(data).hexdigest() != row.get("sha256")
            or len(data) != row.get("bytes")
        ):
            failures.append(f"retrieved transfer artifact digest/size mismatch: {label}")
        relocated[label] = (path, data)
    parsed: dict[str, dict[str, Any]] = {}
    for label in ("authorization", "output", "runtime_attestation", "exact_receipt", "harness_receipt"):
        if label not in relocated:
            continue
        try:
            candidate = json.loads(relocated[label][1])
        except json.JSONDecodeError:
            failures.append(f"retrieved transfer JSON is invalid: {label}")
            continue
        if not isinstance(candidate, dict):
            failures.append(f"retrieved transfer JSON is not an object: {label}")
            continue
        parsed[label] = candidate
    authorization = parsed.get("authorization", {})
    output = parsed.get("output", {})
    runtime = parsed.get("runtime_attestation", {})
    exact_receipt = parsed.get("exact_receipt", {})
    harness_receipt = parsed.get("harness_receipt", {})
    failures.extend(
        verify_reduced_execution_receipt_signature(
            exact_receipt, expected_public_key_fingerprint
        )
    )
    authorization_sha256 = (
        hashlib.sha256(relocated["authorization"][1]).hexdigest()
        if "authorization" in relocated
        else ""
    )
    failures.extend(
        "retrieved authorization: " + failure
        for failure in _execution_authorization_failures(
            authorization,
            authorization_sha256,
            enforce_current_interpreter=False,
            enforce_current_openssl=False,
            expected_environment_sha256=authorization.get("environment_sha256"),
        )
    )
    identity = _execution_identity_from_authorization(
        authorization, authorization_sha256
    )
    if output.get("execution_identity") != identity:
        failures.append("retrieved output execution identity mismatch")
    for key, value in identity.items():
        if exact_receipt.get(key) != value:
            failures.append(f"retrieved exact receipt identity mismatch: {key}")
    plan = authorization.get("transport_archive_plan")
    if (
        plan != exact_receipt.get("transport_archive_plan")
        or plan != manifest.get("signed_transport_archive_plan")
    ):
        failures.append("retrieved signed transport plan mismatch")
    relative_plan = plan.get("relative_artifacts") if isinstance(plan, dict) else {}
    for label in (
        "authorization",
        "output",
        "runtime_attestation",
        "exact_receipt",
        "stdout_log",
        "stderr_log",
    ):
        if artifacts.get(label, {}).get("relative_path") != relative_plan.get(label):
            failures.append(f"retrieved relative archive plan mismatch: {label}")
    signed_hash_fields = {
        "authorization": "authorization_sha256",
        "output": "output_sha256",
        "runtime_attestation": "runtime_attestation_file_sha256",
        "stdout_log": "stdout_sha256",
        "stderr_log": "stderr_sha256",
    }
    for label, field in signed_hash_fields.items():
        row = artifacts.get(label, {})
        if exact_receipt.get(field) != row.get("sha256"):
            failures.append(f"retrieved exact signature does not bind artifact: {label}")
    runtime_body = dict(runtime)
    runtime_sha256 = runtime_body.pop("runtime_sha256", None)
    if not (
        runtime.get("schema_version") == RUNTIME_ATTESTATION_SCHEMA
        and runtime_sha256 == canonical_json_sha256(runtime_body)
        and runtime_sha256 == authorization.get("host_runtime_sha256")
        and runtime.get("portable_sha256")
        == authorization.get("portable_runtime_sha256")
        and runtime.get("isolation_checks_passed") is True
    ):
        failures.append("retrieved runtime attestation content mismatch")
    harness_body = dict(harness_receipt)
    harness_digest = harness_body.pop("harness_receipt_body_sha256", None)
    if not (
        harness_receipt.get("schema_version") == SIGNED_HARNESS_RECEIPT_SCHEMA
        and harness_digest == canonical_json_sha256(harness_body)
        and harness_receipt.get("exact_authorization_sha256")
        == authorization_sha256
        and harness_receipt.get("exact_output_sha256")
        == artifacts.get("output", {}).get("sha256")
        and harness_receipt.get("exact_execution_receipt_sha256")
        == artifacts.get("exact_receipt", {}).get("sha256")
        and harness_receipt.get("runtime_attestation_file_sha256")
        == artifacts.get("runtime_attestation", {}).get("sha256")
    ):
        failures.append("retrieved harness-to-exact custody chain mismatch")
    prelaunch = harness_receipt.get("prelaunch_record")
    acknowledgement = harness_receipt.get("prelaunch_acknowledgement")
    if not isinstance(prelaunch, dict):
        prelaunch = {}
        failures.append("retrieved harness prelaunch record is missing")
    if not isinstance(acknowledgement, dict):
        acknowledgement = {}
        failures.append("retrieved harness prelaunch acknowledgement is missing")
    prelaunch_body = dict(prelaunch)
    prelaunch_sha256 = prelaunch_body.pop("prelaunch_record_sha256", None)
    expected_ack = {
        "schema_version": SIGNED_PRELAUNCH_ACK_SCHEMA,
        "prelaunch_record_sha256": prelaunch.get("prelaunch_record_sha256"),
        "public_key_fingerprint": expected_public_key_fingerprint,
        "authorization_sha256": authorization_sha256,
        "host_runtime_sha256": harness_receipt.get("host_runtime_sha256"),
        "acknowledged_before_child_launch": True,
    }
    if not (
        prelaunch_sha256 == canonical_json_sha256(prelaunch_body)
        and acknowledgement == expected_ack
        and harness_receipt.get("prelaunch_acknowledgement_sha256")
        == canonical_json_sha256(acknowledgement)
        and prelaunch.get("child_launch_has_not_occurred") is True
        and prelaunch.get("caller_must_retain_before_ack") is True
    ):
        failures.append("retrieved prelaunch acknowledgement chain mismatch")
    role = str(manifest.get("role", ""))
    if not (
        role == exact_receipt.get("role")
        and manifest.get("execution_role") == exact_receipt.get("execution_role")
        and manifest.get("replay_pair_id") == exact_receipt.get("replay_pair_id")
        and manifest.get("exact_receipt_signature_scheme")
        == exact_receipt.get("receipt_signature_scheme")
        and manifest.get("exact_receipt_signed_body_sha256")
        == exact_receipt.get("receipt_signed_body_sha256")
        and manifest.get("exact_receipt_signature_ed25519")
        == exact_receipt.get("receipt_signature_ed25519")
        and exact_receipt.get("fresh_child_process") is True
        and exact_receipt.get("returncode") == 0
        and exact_receipt.get("written_after_child_exit_and_output_validation")
        is True
    ):
        failures.append("retrieved manifest/exact receipt execution binding mismatch")
    failures.extend(_execution_field_failures(output, role or "retrieved"))
    witness_sha256 = (
        canonical_json_sha256(_reduced_execution_witness(output)) if output else None
    )
    portable_scope = {
        key: exact_receipt.get(key)
        for key in (
            "role",
            "scientific_run",
            "weeks",
            "split",
            "workers",
            "max_calendars",
            "source_commit",
            "runner_sha256",
            "contract_sha256",
            "seed_identity",
            "portable_runtime_sha256",
            "isolated_bootstrap_sha256",
            "scientific_child_environment_sha256",
        )
    }
    result = {
        "schema_version": RETRIEVED_TRANSFER_VERIFICATION_SCHEMA,
        "role": role,
        "execution_role": manifest.get("execution_role"),
        "replay_pair_id": manifest.get("replay_pair_id"),
        "manifest_sha256": expected_manifest_sha256,
        "public_key_fingerprint": expected_public_key_fingerprint,
        "scientific_run": exact_receipt.get("scientific_run"),
        "authorization_sha256": authorization_sha256,
        "output_sha256": artifacts.get("output", {}).get("sha256"),
        "exact_receipt_sha256": artifacts.get("exact_receipt", {}).get("sha256"),
        "trusted_parent_pid": exact_receipt.get("trusted_parent_pid"),
        "child_pid": exact_receipt.get("child_pid"),
        "host_runtime_sha256": exact_receipt.get("host_runtime_sha256"),
        "portable_scientific_scope": portable_scope,
        "execution_witness_sha256": witness_sha256,
        "signed_exact_receipt_verified": not any(
            "signature" in failure or "fingerprint" in failure for failure in failures
        ),
        "relative_archive_layout_verified": len(relocated) == len(required_artifacts),
        "lexical_no_symlink_archive_paths_verified": (
            len(relocated) == len(required_artifacts)
            and not any("path invalid" in failure for failure in failures)
        ),
        "launch_host_inode_claims_preserved_not_reperformed": True,
        "retrieved_inode_equality_not_required": True,
        "all_content_hashes_recomputed": True,
        "failures": failures,
        "passed": not failures,
    }
    result["verification_sha256"] = canonical_json_sha256(result)
    return result


def verify_retrieved_reduced_pair(
    *,
    archive_root: Path,
    producer_root_relative: str,
    independent_root_relative: str,
    producer_manifest_relative: str,
    independent_manifest_relative: str,
    expected_producer_manifest_sha256: str,
    expected_independent_manifest_sha256: str,
    expected_producer_public_key_fingerprint: str,
    expected_independent_public_key_fingerprint: str,
    role: str,
) -> dict[str, Any]:
    """Compare two relocated signed executions without replaying host inodes."""
    archive_root = archive_root.resolve()

    def relocated_root(relative: str, label: str) -> Path:
        _canonical, root = _confined_checksum_path(archive_root, relative)
        if not root.is_dir():
            raise HarnessError(f"retrieved {label} run root is missing")
        return root

    producer_root = relocated_root(producer_root_relative, "producer")
    independent_root = relocated_root(
        independent_root_relative, "independent"
    )
    failures: list[str] = []
    if producer_root == independent_root:
        failures.append("retrieved producer and independent roots are identical")
    producer = verify_retrieved_reduced_transfer(
        retrieved_root=producer_root,
        manifest_relative=producer_manifest_relative,
        expected_manifest_sha256=expected_producer_manifest_sha256,
        expected_public_key_fingerprint=expected_producer_public_key_fingerprint,
    )
    independent = verify_retrieved_reduced_transfer(
        retrieved_root=independent_root,
        manifest_relative=independent_manifest_relative,
        expected_manifest_sha256=expected_independent_manifest_sha256,
        expected_public_key_fingerprint=expected_independent_public_key_fingerprint,
    )
    failures.extend(
        f"producer: {failure}" for failure in producer.get("failures", [])
    )
    failures.extend(
        f"independent: {failure}" for failure in independent.get("failures", [])
    )
    if producer.get("role") != role or independent.get("role") != role:
        failures.append("retrieved pair role differs from requested certificate")
    if producer.get("execution_role") != "producer":
        failures.append("retrieved producer execution role is invalid")
    if independent.get("execution_role") != "independent_replay":
        failures.append("retrieved independent execution role is invalid")
    pair_id = producer.get("replay_pair_id")
    if not _is_sha256(pair_id) or independent.get("replay_pair_id") != pair_id:
        failures.append("retrieved executions do not share one replay-pair id")
    if producer.get("scientific_run") is not True or independent.get(
        "scientific_run"
    ) is not True:
        failures.append("retrieved pair is not two scientific executions")
    for label, left, right in (
        (
            "signing-key fingerprint",
            expected_producer_public_key_fingerprint,
            expected_independent_public_key_fingerprint,
        ),
        (
            "transfer-manifest digest",
            expected_producer_manifest_sha256,
            expected_independent_manifest_sha256,
        ),
        (
            "authorization digest",
            producer.get("authorization_sha256"),
            independent.get("authorization_sha256"),
        ),
        (
            "exact-receipt digest",
            producer.get("exact_receipt_sha256"),
            independent.get("exact_receipt_sha256"),
        ),
        (
            "output digest",
            producer.get("output_sha256"),
            independent.get("output_sha256"),
        ),
    ):
        if left == right:
            failures.append(f"retrieved pair reuses one {label}")
    if producer.get("portable_scientific_scope") != independent.get(
        "portable_scientific_scope"
    ):
        failures.append("retrieved pair portable scientific scopes differ")
    if producer.get("execution_witness_sha256") != independent.get(
        "execution_witness_sha256"
    ):
        failures.append("retrieved pair exact execution witnesses differ")
    if (
        producer.get("host_runtime_sha256")
        == independent.get("host_runtime_sha256")
        and producer.get("trusted_parent_pid")
        == independent.get("trusted_parent_pid")
    ):
        failures.append(
            "retrieved pair reuses one parent PID in the same signed host runtime"
        )
    result = {
        "schema_version": RETRIEVED_PAIR_VERIFICATION_SCHEMA,
        "role": role,
        "replay_pair_id": pair_id,
        "archive_root_identity": "CALLER_SUPPLIED_ROOT_NOT_EMBEDDED",
        "portable_archived_artifacts": {
            "producer": {
                "run_root_relative": producer_root_relative,
                "manifest_relative": producer_manifest_relative,
            },
            "independent": {
                "run_root_relative": independent_root_relative,
                "manifest_relative": independent_manifest_relative,
            },
        },
        "caller_retained_inputs": {
            "producer_manifest_sha256": expected_producer_manifest_sha256,
            "independent_manifest_sha256": expected_independent_manifest_sha256,
            "producer_public_key_fingerprint": (
                expected_producer_public_key_fingerprint
            ),
            "independent_public_key_fingerprint": (
                expected_independent_public_key_fingerprint
            ),
        },
        "producer_transfer_verification": producer,
        "independent_transfer_verification": independent,
        "two_distinct_signing_keys": (
            expected_producer_public_key_fingerprint
            != expected_independent_public_key_fingerprint
        ),
        "portable_scientific_scope_match": producer.get(
            "portable_scientific_scope"
        )
        == independent.get("portable_scientific_scope"),
        "exact_execution_witness_match": producer.get(
            "execution_witness_sha256"
        )
        == independent.get("execution_witness_sha256"),
        "launch_host_inode_claims_preserved_not_reperformed": True,
        "relocated_pair_verification": True,
        "lexical_no_symlink_archive_paths_verified": (
            producer.get("lexical_no_symlink_archive_paths_verified") is True
            and independent.get("lexical_no_symlink_archive_paths_verified") is True
        ),
        "persisted_pass_not_authoritative": True,
        "single_payload_validation_remains_fail_closed": True,
        "failures": failures,
        "passed": not failures,
    }
    result["verification_sha256"] = canonical_json_sha256(result)
    return result


def reverify_retrieved_reduced_pair_archive(
    *,
    archive_root: Path,
    pair_verification_path: Path,
    expected_pair_verification_sha256: str,
    role: str,
    certified_artifact_sha256: str,
    expected_producer_manifest_sha256: str,
    expected_independent_manifest_sha256: str,
    expected_producer_public_key_fingerprint: str,
    expected_independent_public_key_fingerprint: str,
) -> dict[str, Any]:
    """Reopen a portable pair and both signed transfer chains from bytes.

    The persisted pair verdict supplies only its relative archive layout.  Its
    cached transfer verifications and PASS flag are deliberately ignored; all
    manifest digests and signing-key fingerprints are caller-retained inputs.
    Launch-host absolute paths and inode numbers are neither replayed nor
    trusted after transport.
    """
    archive_root = archive_root.resolve()
    try:
        pair_relative = str(
            pair_verification_path.absolute().relative_to(archive_root)
        )
        _canonical_pair, pair_path = _confined_checksum_path(
            archive_root, pair_relative
        )
    except (ValueError, HarnessError) as exc:
        raise HarnessError("portable pair verification escapes archive root") from exc
    if not pair_path.is_file():
        raise HarnessError("portable pair verification is missing")
    if not _is_sha256(expected_pair_verification_sha256) or sha256_file(
        pair_path
    ) != expected_pair_verification_sha256:
        raise HarnessError("caller-retained portable pair digest mismatch")

    payload = load_json(pair_path)
    failures: list[str] = []
    if not isinstance(payload, dict):
        raise HarnessError("portable pair verification is not an object")
    body = dict(payload)
    claimed_body_sha256 = body.pop("verification_sha256", None)
    if (
        payload.get("schema_version") != RETRIEVED_PAIR_VERIFICATION_SCHEMA
        or claimed_body_sha256 != canonical_json_sha256(body)
        or payload.get("role") != role
    ):
        raise HarnessError("portable pair verification identity is invalid")
    if payload.get("archive_root_identity") != "CALLER_SUPPLIED_ROOT_NOT_EMBEDDED":
        failures.append("portable pair embeds or misstates the archive-root identity")
    if payload.get("persisted_pass_not_authoritative") is not True:
        failures.append("portable pair does not disclaim its persisted PASS flag")
    if payload.get("single_payload_validation_remains_fail_closed") is not True:
        failures.append("portable pair weakens single-payload fail closure")
    if payload.get("lexical_no_symlink_archive_paths_verified") is not True:
        failures.append("portable pair lacks lexical no-symlink path assurance")

    expected_inputs = {
        "producer_manifest_sha256": expected_producer_manifest_sha256,
        "independent_manifest_sha256": expected_independent_manifest_sha256,
        "producer_public_key_fingerprint": (
            expected_producer_public_key_fingerprint
        ),
        "independent_public_key_fingerprint": (
            expected_independent_public_key_fingerprint
        ),
    }
    if any(not _is_sha256(value) for value in expected_inputs.values()):
        failures.append("caller-retained portable pair input is malformed")
    if payload.get("caller_retained_inputs") != expected_inputs:
        failures.append("portable pair caller-retained inputs mismatch")

    archived = payload.get("portable_archived_artifacts")
    required_row_keys = {"run_root_relative", "manifest_relative"}
    if not isinstance(archived, dict) or set(archived) != {
        "producer",
        "independent",
    }:
        failures.append("portable pair artifact index is malformed")
        archived = {}
    for label in ("producer", "independent"):
        row = archived.get(label)
        if not isinstance(row, dict) or set(row) != required_row_keys:
            failures.append(f"portable {label} artifact index is malformed")

    current: dict[str, Any]
    if failures:
        current = {
            "schema_version": RETRIEVED_PAIR_VERIFICATION_SCHEMA,
            "passed": False,
            "failures": ["portable archive precheck failed"],
        }
    else:
        producer = archived["producer"]
        independent = archived["independent"]
        try:
            current = verify_retrieved_reduced_pair(
                archive_root=archive_root,
                producer_root_relative=str(producer["run_root_relative"]),
                independent_root_relative=str(
                    independent["run_root_relative"]
                ),
                producer_manifest_relative=str(producer["manifest_relative"]),
                independent_manifest_relative=str(
                    independent["manifest_relative"]
                ),
                expected_producer_manifest_sha256=(
                    expected_producer_manifest_sha256
                ),
                expected_independent_manifest_sha256=(
                    expected_independent_manifest_sha256
                ),
                expected_producer_public_key_fingerprint=(
                    expected_producer_public_key_fingerprint
                ),
                expected_independent_public_key_fingerprint=(
                    expected_independent_public_key_fingerprint
                ),
                role=role,
            )
        except (HarnessError, OSError, ValueError, TypeError) as exc:
            current = {
                "schema_version": RETRIEVED_PAIR_VERIFICATION_SCHEMA,
                "passed": False,
                "failures": [
                    "portable archive reconstruction failed: "
                    f"{type(exc).__name__}: {exc}"
                ],
            }
        failures.extend(
            f"portable pair reverification: {failure}"
            for failure in current.get("failures", [])
        )
        if current.get("passed") is not True:
            failures.append("portable pair reverification did not pass")

    if not _is_sha256(certified_artifact_sha256):
        failures.append("certified portable artifact digest is malformed")
    current_output_hashes = {
        current.get("producer_transfer_verification", {}).get("output_sha256"),
        current.get("independent_transfer_verification", {}).get("output_sha256"),
    }
    if certified_artifact_sha256 not in current_output_hashes:
        failures.append("certified artifact is absent from reverified portable pair")

    result = {
        "schema_version": AUTHORIZED_PAIR_REVERIFICATION_SCHEMA,
        "verification_mode": PORTABLE_PAIR_VERIFICATION_MODE,
        "pair_verification_path": str(pair_path),
        "pair_verification_sha256": expected_pair_verification_sha256,
        "persisted_pass_claim_ignored": True,
        "cached_transfer_verifications_ignored": True,
        "both_signed_transfer_chains_reopened": True,
        "lexical_no_symlink_archive_paths_reverified": (
            current.get("lexical_no_symlink_archive_paths_verified") is True
        ),
        "launch_host_inode_claims_not_replayed": True,
        "caller_retained_manifest_digests_reapplied": True,
        "caller_retained_fingerprints_reapplied": True,
        "current_verification": current,
        "failures": failures,
        "passed": not failures and current.get("passed") is True,
    }
    result["verification_sha256"] = canonical_json_sha256(result)
    return result


def verify_reduced_evidence_pair(
    *,
    archive_root: Path,
    producer_run_dir: Path,
    independent_run_dir: Path,
    role: str,
    producer_output_relative: str,
    independent_output_relative: str,
    expected_producer_output_sha256: str,
    expected_independent_output_sha256: str,
    producer_receipt_relative: str,
    independent_receipt_relative: str,
    expected_producer_receipt_sha256: str,
    expected_independent_receipt_sha256: str,
    producer_authorization_relative: str,
    independent_authorization_relative: str,
    expected_producer_authorization_sha256: str,
    expected_independent_authorization_sha256: str,
    expected_producer_public_key_fingerprint: str,
    expected_independent_public_key_fingerprint: str,
    producer_runtime_attestation_relative: str,
    independent_runtime_attestation_relative: str,
    expected_producer_runtime_attestation_sha256: str,
    expected_independent_runtime_attestation_sha256: str,
    output_path: Path | None,
    expected_environment_sha256: str | None = None,
) -> dict[str, Any]:
    """Verify two fresh reduced executions with caller-retained launch custody."""
    archive_root = archive_root.resolve()
    producer_run_dir = producer_run_dir.resolve()
    independent_run_dir = independent_run_dir.resolve()
    failures: list[str] = []
    if producer_run_dir == independent_run_dir:
        failures.append("producer and independent run trees are identical")
    for label, run_dir in (
        ("producer", producer_run_dir),
        ("independent", independent_run_dir),
    ):
        try:
            run_dir.relative_to(archive_root)
        except ValueError as exc:
            raise HarnessError(f"{label} run tree escapes archive root") from exc
    if output_path is not None and output_path.exists():
        raise HarnessError("reduced pair verification output already exists")
    for label, value in (
        ("producer output", expected_producer_output_sha256),
        ("independent output", expected_independent_output_sha256),
        ("producer receipt", expected_producer_receipt_sha256),
        ("independent receipt", expected_independent_receipt_sha256),
        ("producer launch authorization", expected_producer_authorization_sha256),
        (
            "independent launch authorization",
            expected_independent_authorization_sha256,
        ),
        (
            "producer public-key fingerprint",
            expected_producer_public_key_fingerprint,
        ),
        (
            "independent public-key fingerprint",
            expected_independent_public_key_fingerprint,
        ),
        (
            "producer runtime attestation",
            expected_producer_runtime_attestation_sha256,
        ),
        (
            "independent runtime attestation",
            expected_independent_runtime_attestation_sha256,
        ),
    ):
        if not _is_sha256(value):
            failures.append(f"{label} caller-retained digest is malformed")

    def confined(run_dir: Path, relative: str, label: str) -> Path:
        try:
            _canonical, path = _confined_checksum_path(run_dir, relative)
        except HarnessError as exc:
            failures.append(f"{label}: {exc}")
            return run_dir / "__invalid__"
        if not path.is_file():
            failures.append(f"{label} is missing")
        return path

    producer_output = confined(
        producer_run_dir, producer_output_relative, "producer output"
    )
    independent_output = confined(
        independent_run_dir, independent_output_relative, "independent output"
    )
    producer_receipt = confined(
        producer_run_dir, producer_receipt_relative, "producer execution receipt"
    )
    independent_receipt = confined(
        independent_run_dir,
        independent_receipt_relative,
        "independent execution receipt",
    )
    producer_authorization = confined(
        producer_run_dir,
        producer_authorization_relative,
        "producer launch authorization",
    )
    independent_authorization = confined(
        independent_run_dir,
        independent_authorization_relative,
        "independent launch authorization",
    )
    producer_runtime = confined(
        producer_run_dir,
        producer_runtime_attestation_relative,
        "producer runtime attestation",
    )
    independent_runtime = confined(
        independent_run_dir,
        independent_runtime_attestation_relative,
        "independent runtime attestation",
    )
    for label, path, expected in (
        ("producer output", producer_output, expected_producer_output_sha256),
        (
            "independent output",
            independent_output,
            expected_independent_output_sha256,
        ),
        (
            "producer execution receipt",
            producer_receipt,
            expected_producer_receipt_sha256,
        ),
        (
            "independent execution receipt",
            independent_receipt,
            expected_independent_receipt_sha256,
        ),
        (
            "producer launch authorization",
            producer_authorization,
            expected_producer_authorization_sha256,
        ),
        (
            "independent launch authorization",
            independent_authorization,
            expected_independent_authorization_sha256,
        ),
        (
            "producer runtime attestation",
            producer_runtime,
            expected_producer_runtime_attestation_sha256,
        ),
        (
            "independent runtime attestation",
            independent_runtime,
            expected_independent_runtime_attestation_sha256,
        ),
    ):
        if path.is_file() and sha256_file(path) != expected:
            failures.append(f"{label} caller-retained digest mismatch")

    for label, receipt_path, authorization_path, expected_authorization in (
        (
            "producer",
            producer_receipt,
            producer_authorization,
            expected_producer_authorization_sha256,
        ),
        (
            "independent",
            independent_receipt,
            independent_authorization,
            expected_independent_authorization_sha256,
        ),
    ):
        if not receipt_path.is_file() or not authorization_path.is_file():
            continue
        try:
            receipt = load_json(receipt_path)
        except HarnessError as exc:
            failures.append(f"{label} execution receipt is unreadable: {exc}")
            continue
        if receipt.get("schema_version") == SIGNED_HARNESS_RECEIPT_SCHEMA:
            receipt_authorization_path = receipt.get("exact_authorization_path")
            receipt_authorization_sha256 = receipt.get(
                "exact_authorization_sha256"
            )
        else:
            receipt_authorization_path = receipt.get("authorization_path")
            receipt_authorization_sha256 = receipt.get("authorization_sha256")
        if receipt_authorization_path != str(authorization_path.resolve()):
            failures.append(f"{label} receipt launch-authorization path mismatch")
        if receipt_authorization_sha256 != expected_authorization:
            failures.append(f"{label} receipt launch-authorization digest mismatch")

    if expected_producer_authorization_sha256 == expected_independent_authorization_sha256:
        failures.append("producer and independent launch-authorization digests collide")
    if expected_producer_receipt_sha256 == expected_independent_receipt_sha256:
        failures.append("producer and independent execution-receipt digests collide")
    if (
        expected_producer_public_key_fingerprint
        == expected_independent_public_key_fingerprint
    ):
        failures.append("producer and independent signing-key fingerprints collide")

    exact_verification: dict[str, Any]
    if failures:
        exact_verification = {
            "passed": False,
            "failures": ["harness custody precheck failed"],
        }
    else:
        from scripts.run_paper2_bottleneck_exact_transducer import (
            verify_independent_reduced_execution,
        )

        exact_verification = verify_independent_reduced_execution(
            producer_output,
            independent_output,
            role,
            expected_producer_sha256=expected_producer_output_sha256,
            expected_independent_sha256=expected_independent_output_sha256,
            expected_producer_authorization_sha256=(
                expected_producer_authorization_sha256
            ),
            expected_independent_authorization_sha256=(
                expected_independent_authorization_sha256
            ),
            producer_receipt_path=producer_receipt,
            expected_producer_receipt_sha256=expected_producer_receipt_sha256,
            independent_receipt_path=independent_receipt,
            expected_independent_receipt_sha256=expected_independent_receipt_sha256,
            expected_producer_public_key_fingerprint=(
                expected_producer_public_key_fingerprint
            ),
            expected_independent_public_key_fingerprint=(
                expected_independent_public_key_fingerprint
            ),
            producer_runtime_attestation_path=producer_runtime,
            expected_producer_runtime_attestation_sha256=(
                expected_producer_runtime_attestation_sha256
            ),
            independent_runtime_attestation_path=independent_runtime,
            expected_independent_runtime_attestation_sha256=(
                expected_independent_runtime_attestation_sha256
            ),
            expected_environment_sha256=expected_environment_sha256,
        )
        if exact_verification.get("passed") is not True:
            failures.extend(
                f"exact verifier: {failure}"
                for failure in exact_verification.get("failures", [])
            )
    result = {
        "schema_version": REDUCED_PAIR_VERIFICATION_SCHEMA,
        "role": role,
        "archive_root_identity": "CALLER_SUPPLIED_ROOT_NOT_EMBEDDED",
        "producer_run_dir": str(producer_run_dir),
        "independent_run_dir": str(independent_run_dir),
        "producer_output_sha256": expected_producer_output_sha256,
        "independent_output_sha256": expected_independent_output_sha256,
        "producer_execution_receipt_sha256": expected_producer_receipt_sha256,
        "independent_execution_receipt_sha256": expected_independent_receipt_sha256,
        "producer_launch_authorization_sha256": expected_producer_authorization_sha256,
        "independent_launch_authorization_sha256": (
            expected_independent_authorization_sha256
        ),
        "producer_public_key_fingerprint": expected_producer_public_key_fingerprint,
        "independent_public_key_fingerprint": (
            expected_independent_public_key_fingerprint
        ),
        "producer_runtime_attestation_sha256": (
            expected_producer_runtime_attestation_sha256
        ),
        "independent_runtime_attestation_sha256": (
            expected_independent_runtime_attestation_sha256
        ),
        "expected_environment_sha256": expected_environment_sha256,
        "archived_artifacts": {
            "producer": {
                "run_dir_relative": str(producer_run_dir.relative_to(archive_root)),
                "output_relative": producer_output_relative,
                "harness_receipt_relative": producer_receipt_relative,
                "authorization_relative": producer_authorization_relative,
                "runtime_attestation_relative": producer_runtime_attestation_relative,
            },
            "independent": {
                "run_dir_relative": str(independent_run_dir.relative_to(archive_root)),
                "output_relative": independent_output_relative,
                "harness_receipt_relative": independent_receipt_relative,
                "authorization_relative": independent_authorization_relative,
                "runtime_attestation_relative": independent_runtime_attestation_relative,
            },
        },
        "exact_verification": exact_verification,
        "caller_retained_fingerprints_are_external_inputs": True,
        "single_payload_validation_remains_fail_closed": True,
        "failures": failures,
        "passed": not failures and exact_verification.get("passed") is True,
    }
    result["verification_sha256"] = canonical_json_sha256(result)
    if output_path is not None:
        output_absolute = output_path.absolute()
        try:
            output_absolute.relative_to(archive_root)
        except ValueError as exc:
            raise HarnessError("reduced pair verification output escapes archive root") from exc
        exclusive_write_json(output_absolute, result)
    return result


def reverify_reduced_pair_archive(
    *,
    archive_root: Path,
    pair_verification_path: Path,
    expected_pair_verification_sha256: str,
    role: str,
    certified_artifact_sha256: str,
    expected_producer_public_key_fingerprint: str,
    expected_independent_public_key_fingerprint: str,
) -> dict[str, Any]:
    """Ignore persisted PASS and rerun the complete archived custody verifier."""
    archive_root = archive_root.resolve()
    try:
        pair_relative = str(
            pair_verification_path.absolute().relative_to(archive_root)
        )
        _canonical_pair, pair_path = _confined_checksum_path(
            archive_root, pair_relative
        )
    except (ValueError, HarnessError) as exc:
        raise HarnessError("archived pair verification escapes archive root") from exc
    if not pair_path.is_file():
        raise HarnessError("archived pair verification is missing")
    if not _is_sha256(expected_pair_verification_sha256) or sha256_file(
        pair_path
    ) != expected_pair_verification_sha256:
        raise HarnessError("caller-retained archived pair digest mismatch")
    payload = load_json(pair_path)
    if not isinstance(payload, dict):
        raise HarnessError("archived pair verification is not an object")
    body = dict(payload)
    claimed_body_sha256 = body.pop("verification_sha256", None)
    if (
        payload.get("schema_version") != REDUCED_PAIR_VERIFICATION_SCHEMA
        or claimed_body_sha256 != canonical_json_sha256(body)
        or payload.get("role") != role
    ):
        raise HarnessError("archived pair verification identity is invalid")
    if (
        payload.get("producer_public_key_fingerprint")
        != expected_producer_public_key_fingerprint
        or payload.get("independent_public_key_fingerprint")
        != expected_independent_public_key_fingerprint
    ):
        raise HarnessError("caller-retained archived signing fingerprint mismatch")
    if certified_artifact_sha256 not in {
        payload.get("producer_output_sha256"),
        payload.get("independent_output_sha256"),
    }:
        raise HarnessError("certified artifact is absent from archived pair")
    archived = payload.get("archived_artifacts")
    if not isinstance(archived, dict) or set(archived) != {
        "producer",
        "independent",
    }:
        raise HarnessError("archived pair artifact index is malformed")

    def archived_row(label: str) -> tuple[Path, dict[str, Any]]:
        row = archived.get(label)
        required = {
            "run_dir_relative",
            "output_relative",
            "harness_receipt_relative",
            "authorization_relative",
            "runtime_attestation_relative",
        }
        if not isinstance(row, dict) or set(row) != required:
            raise HarnessError(f"archived {label} artifact index is malformed")
        run_relative = str(row["run_dir_relative"])
        _canonical, run_dir = _confined_checksum_path(archive_root, run_relative)
        if not run_dir.is_dir():
            raise HarnessError(f"archived {label} run tree is missing")
        return run_dir, row

    producer_dir, producer = archived_row("producer")
    independent_dir, independent = archived_row("independent")
    current = verify_reduced_evidence_pair(
        archive_root=archive_root,
        producer_run_dir=producer_dir,
        independent_run_dir=independent_dir,
        role=role,
        producer_output_relative=str(producer["output_relative"]),
        independent_output_relative=str(independent["output_relative"]),
        expected_producer_output_sha256=str(payload["producer_output_sha256"]),
        expected_independent_output_sha256=str(
            payload["independent_output_sha256"]
        ),
        producer_receipt_relative=str(producer["harness_receipt_relative"]),
        independent_receipt_relative=str(
            independent["harness_receipt_relative"]
        ),
        expected_producer_receipt_sha256=str(
            payload["producer_execution_receipt_sha256"]
        ),
        expected_independent_receipt_sha256=str(
            payload["independent_execution_receipt_sha256"]
        ),
        producer_authorization_relative=str(producer["authorization_relative"]),
        independent_authorization_relative=str(
            independent["authorization_relative"]
        ),
        expected_producer_authorization_sha256=str(
            payload["producer_launch_authorization_sha256"]
        ),
        expected_independent_authorization_sha256=str(
            payload["independent_launch_authorization_sha256"]
        ),
        expected_producer_public_key_fingerprint=(
            expected_producer_public_key_fingerprint
        ),
        expected_independent_public_key_fingerprint=(
            expected_independent_public_key_fingerprint
        ),
        producer_runtime_attestation_relative=str(
            producer["runtime_attestation_relative"]
        ),
        independent_runtime_attestation_relative=str(
            independent["runtime_attestation_relative"]
        ),
        expected_producer_runtime_attestation_sha256=str(
            payload["producer_runtime_attestation_sha256"]
        ),
        expected_independent_runtime_attestation_sha256=str(
            payload["independent_runtime_attestation_sha256"]
        ),
        output_path=None,
        expected_environment_sha256=payload.get("expected_environment_sha256"),
    )
    return {
        "schema_version": "paper2_reduced_pair_archive_reverification_v1",
        "verification_mode": CLASSIC_PAIR_VERIFICATION_MODE,
        "pair_verification_path": str(pair_path),
        "pair_verification_sha256": expected_pair_verification_sha256,
        "persisted_pass_claim_ignored": True,
        "all_archived_artifacts_reopened": True,
        "caller_retained_fingerprints_reapplied": True,
        "current_verification": current,
        "failures": list(current.get("failures", [])),
        "passed": current.get("passed") is True,
    }


def reverify_authorized_reduced_pair_archive(
    *,
    archive_root: Path,
    pair_verification_path: Path,
    expected_pair_verification_sha256: str,
    verification_mode: str,
    role: str,
    certified_artifact_sha256: str,
    expected_producer_public_key_fingerprint: str,
    expected_independent_public_key_fingerprint: str,
    expected_producer_manifest_sha256: str | None = None,
    expected_independent_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Dispatch an explicit v8 authorization to local or portable custody."""
    if verification_mode == CLASSIC_PAIR_VERIFICATION_MODE:
        return reverify_reduced_pair_archive(
            archive_root=archive_root,
            pair_verification_path=pair_verification_path,
            expected_pair_verification_sha256=(
                expected_pair_verification_sha256
            ),
            role=role,
            certified_artifact_sha256=certified_artifact_sha256,
            expected_producer_public_key_fingerprint=(
                expected_producer_public_key_fingerprint
            ),
            expected_independent_public_key_fingerprint=(
                expected_independent_public_key_fingerprint
            ),
        )
    if verification_mode == PORTABLE_PAIR_VERIFICATION_MODE:
        if not isinstance(expected_producer_manifest_sha256, str) or not isinstance(
            expected_independent_manifest_sha256, str
        ):
            raise HarnessError(
                "portable pair authorization lacks external manifest digests"
            )
        return reverify_retrieved_reduced_pair_archive(
            archive_root=archive_root,
            pair_verification_path=pair_verification_path,
            expected_pair_verification_sha256=(
                expected_pair_verification_sha256
            ),
            role=role,
            certified_artifact_sha256=certified_artifact_sha256,
            expected_producer_manifest_sha256=(
                expected_producer_manifest_sha256
            ),
            expected_independent_manifest_sha256=(
                expected_independent_manifest_sha256
            ),
            expected_producer_public_key_fingerprint=(
                expected_producer_public_key_fingerprint
            ),
            expected_independent_public_key_fingerprint=(
                expected_independent_public_key_fingerprint
            ),
        )
    raise HarnessError(
        "reduced pair authorization has an unknown or legacy verification mode"
    )


def validate_reduced_pair_verification_payload(
    payload: Any,
    *,
    role: str,
    certified_artifact_sha256: str,
) -> list[str]:
    """Validate a persisted two-process verification before authorization."""
    failures: list[str] = []
    if not isinstance(payload, dict):
        return ["reduced pair verification is not an object"]
    if payload.get("schema_version") != REDUCED_PAIR_VERIFICATION_SCHEMA:
        failures.append("reduced pair verification schema mismatch")
    body = dict(payload)
    claimed = body.pop("verification_sha256", None)
    if claimed != canonical_json_sha256(body):
        failures.append("reduced pair verification body digest mismatch")
    if payload.get("role") != role:
        failures.append("reduced pair verification role mismatch")
    if payload.get("passed") is not True or payload.get("failures") != []:
        failures.append("reduced pair verification did not pass")
    if payload.get("single_payload_validation_remains_fail_closed") is not True:
        failures.append("reduced pair verification weakens single-payload fail closure")
    if payload.get("caller_retained_fingerprints_are_external_inputs") is not True:
        failures.append("reduced pair verification lacks external fingerprint inputs")
    exact = payload.get("exact_verification")
    if not isinstance(exact, dict) or exact.get("passed") is not True:
        failures.append("exact independent execution verification did not pass")
    if certified_artifact_sha256 not in {
        payload.get("producer_output_sha256"),
        payload.get("independent_output_sha256"),
    }:
        failures.append("authorized reduced artifact is absent from verified pair")
    for key in (
        "producer_execution_receipt_sha256",
        "independent_execution_receipt_sha256",
        "producer_launch_authorization_sha256",
        "independent_launch_authorization_sha256",
        "producer_public_key_fingerprint",
        "independent_public_key_fingerprint",
        "producer_runtime_attestation_sha256",
        "independent_runtime_attestation_sha256",
    ):
        if not _is_sha256(payload.get(key)):
            failures.append(f"reduced pair custody digest is malformed: {key}")
    failures.append(
        "authorization must reopen archived outputs, launch authorizations, signed "
        "execution receipts, runtime attestations, harness custody receipts, and "
        "caller-retained public-key fingerprints and rerun verification; a persisted "
        "pair verdict cannot authorize itself"
    )
    return failures


def _validate_scientific_result(
    payload: dict[str, Any],
    run_manifest: dict[str, Any],
    runner_manifest: dict[str, Any] | None = None,
    seed_manifest: dict[str, Any] | None = None,
) -> list[str]:
    """Validate execution assurances without interpreting H_PI as promotion."""
    failures = []
    expected = run_manifest["inputs"]
    result_schema = payload.get("schema_version")
    assurance = payload.get("execution_assurance")
    if (
        run_manifest.get("mode") == "scientific"
        and result_schema != "paper2_bottleneck_full_frontier_v2"
    ):
        failures.append("scientific run result is not the native frontier schema")
    if (
        result_schema == "paper2_bottleneck_full_frontier_v2"
        and "execution_assurance" in payload
    ):
        failures.append(
            "native frontier payload mixes a foreign execution assurance"
        )
        assurance = None
    if isinstance(assurance, dict):
        if result_schema is not None:
            failures.append(
                "execution-assurance payload has an unsupported foreign schema"
            )
        if assurance.get("key_schema_version") != "paper2_bottleneck_semantic_markov_key_v4":
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
    elif result_schema == "paper2_bottleneck_full_frontier_v2":
        if runner_manifest is None:
            failures.append("native frontier result lacks its runner manifest")
        if seed_manifest is None:
            failures.append("native frontier result lacks its harness seed manifest")
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
        if authorization.get("key_schema_version") != "paper2_bottleneck_semantic_markov_key_v4":
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
        certificate_identities = [
            (row.get("seed"), row.get("tape_sha256"))
            for row in coverage_rows
            if isinstance(row, dict)
        ]
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
            and len(set(certificate_identities)) == expected_certificate_count
            and certificate_coverage.get("rows_sha256")
            == canonical_json_sha256(coverage_rows)
            and all(
                row.get("index") == index
                and row.get("complete") is True
                and isinstance(row.get("certificate_sha256"), str)
                and len(row["certificate_sha256"]) == 64
                for index, row in enumerate(coverage_rows)
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
            if (
                runner_manifest.get("schema_version")
                != "paper2_bottleneck_full_frontier_manifest_v2"
            ):
                failures.append("runner manifest schema mismatch")
            if runner_manifest.get("phase_execution_complete") is not True:
                failures.append("runner manifest phase did not complete")
            if runner_manifest.get("full_execution_was_explicitly_invoked") is not True:
                failures.append("runner manifest does not record explicit full execution")
            input_artifacts = runner_manifest.get("input_artifacts", {})
            if input_artifacts.get("primary_contract", {}).get("sha256") != expected["contract_sha256"]:
                failures.append("runner manifest contract hash mismatch")
            if input_artifacts.get("authorization", {}).get("sha256") != expected.get("authorization_sha256"):
                failures.append("runner manifest authorization hash mismatch")
            code_hashes = runner_manifest.get("code_sha256", {})
            if code_hashes.get(expected["runner_relative"]) != expected["runner_sha256"]:
                failures.append("runner manifest code hash mismatch")
            if runner_manifest.get("key_schema_version") != "paper2_bottleneck_semantic_markov_key_v4":
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
            harness_seed_rows = (
                seed_manifest.get("seeds", [])
                if isinstance(seed_manifest, dict)
                else []
            )
            runner_seed_rows = runner_manifest.get("seed_manifest", [])
            transducer_rows = payload.get("transducers", [])
            runner_checkpoint_rows = runner_manifest.get(
                "checkpoint_artifacts", []
            )

            ordered_sources = {
                "harness seed manifest": harness_seed_rows,
                "runner seed manifest": runner_seed_rows,
                "result transducers": transducer_rows,
                "result checkpoints": checkpoints,
                "collision-certificate coverage": coverage_rows,
                "runner checkpoint artifacts": runner_checkpoint_rows,
            }
            for label, rows in ordered_sources.items():
                if not isinstance(rows, list) or len(rows) != expected_certificate_count:
                    failures.append(
                        f"frontier ordered identity source has wrong count: {label}"
                    )

            ordered_harness = []
            ordered_runner = []
            if (
                isinstance(harness_seed_rows, list)
                and isinstance(runner_seed_rows, list)
                and len(harness_seed_rows) == expected_certificate_count
                and len(runner_seed_rows) == expected_certificate_count
            ):
                for index, (harness_row, runner_row) in enumerate(
                    zip(harness_seed_rows, runner_seed_rows)
                ):
                    if not isinstance(harness_row, dict) or not isinstance(
                        runner_row, dict
                    ):
                        failures.append(
                            f"frontier malformed ordered seed identity at index {index}"
                        )
                        continue
                    if (
                        runner_row.get("seed") != harness_row.get("seed")
                        or runner_row.get("context_0") != harness_row.get("context")
                        or runner_row.get("split") != harness_row.get("split")
                        or runner_row.get("tape_sha256")
                        != harness_row.get("expected_tape_sha256")
                    ):
                        failures.append(
                            f"frontier harness/runner seed identity mismatch at index {index}"
                        )
                    ordered_harness.append(harness_row.get("seed"))
                    ordered_runner.append(
                        (runner_row.get("seed"), runner_row.get("tape_sha256"))
                    )
                if len(set(ordered_harness)) != expected_certificate_count:
                    failures.append("frontier harness seed identities are not unique")
                if len(set(ordered_runner)) != expected_certificate_count:
                    failures.append("frontier runner tape identities are not unique")
                if (
                    isinstance(checkpoints, list)
                    and len(checkpoints) == expected_certificate_count
                ):
                    for index, (harness_row, checkpoint_row) in enumerate(
                        zip(harness_seed_rows, checkpoints)
                    ):
                        if not isinstance(harness_row, dict) or not isinstance(
                            checkpoint_row, dict
                        ):
                            failures.append(
                                f"frontier malformed checkpoint scope identity at index {index}"
                            )
                            continue
                        if (
                            checkpoint_row.get("index") != index
                            or checkpoint_row.get("seed") != harness_row.get("seed")
                            or checkpoint_row.get("context")
                            != harness_row.get("context")
                            or checkpoint_row.get("split")
                            != harness_row.get("split")
                            or checkpoint_row.get("weeks")
                            != harness_row.get("weeks")
                        ):
                            failures.append(
                                f"frontier checkpoint scope identity mismatch at index {index}"
                            )

            def result_identity_rows(
                rows: Any,
                *,
                source: str,
                checkpoint: bool = False,
            ) -> list[tuple[Any, Any, Any, Any]]:
                identities: list[tuple[Any, Any, Any, Any]] = []
                if not isinstance(rows, list):
                    return identities
                for index, row in enumerate(rows):
                    if not isinstance(row, dict):
                        failures.append(
                            f"frontier malformed {source} identity at index {index}"
                        )
                        continue
                    if checkpoint:
                        proof = row.get("source_transducer_proof", {})
                        certificate = (
                            proof.get("collision_bisimulation", {}).get(
                                "certificate_sha256"
                            )
                            if isinstance(proof, dict)
                            else None
                        )
                    else:
                        certificate = row.get("collision_certificate_sha256")
                    identity = (
                        row.get("seed"),
                        row.get("tape_sha256"),
                        certificate,
                        row.get("score_table_sha256"),
                    )
                    if not all(_is_sha256(value) for value in identity[1:]):
                        failures.append(
                            f"frontier malformed {source} digest identity at index {index}"
                        )
                    identities.append(identity)
                return identities

            transducer_identities = result_identity_rows(
                transducer_rows, source="result transducer"
            )
            checkpoint_identities = result_identity_rows(
                checkpoints, source="result checkpoint", checkpoint=True
            )
            coverage_identities = [
                (
                    row.get("seed"),
                    row.get("tape_sha256"),
                    row.get("certificate_sha256"),
                )
                for row in coverage_rows
                if isinstance(row, dict)
            ]
            runner_checkpoint_identities = [
                (
                    row.get("seed"),
                    row.get("collision_certificate_sha256"),
                    row.get("score_table_sha256"),
                )
                for row in runner_checkpoint_rows
                if isinstance(row, dict)
            ]
            if any(
                not _is_sha256(value)
                for identity in coverage_identities
                for value in identity[1:]
            ):
                failures.append("frontier collision coverage has malformed identity digests")
            if any(
                not _is_sha256(value)
                for identity in runner_checkpoint_identities
                for value in identity[1:]
            ):
                failures.append("frontier runner checkpoints have malformed identity digests")

            if (
                len(ordered_runner) == expected_certificate_count
                and len(transducer_identities) == expected_certificate_count
                and [identity[:2] for identity in transducer_identities]
                != ordered_runner
            ):
                failures.append("frontier ordered tape identities mismatch: transducers")
            if (
                len(transducer_identities) == expected_certificate_count
                and len(checkpoint_identities) == expected_certificate_count
                and checkpoint_identities != transducer_identities
            ):
                failures.append(
                    "frontier ordered tape/certificate identities mismatch: checkpoints"
                )
            if (
                len(transducer_identities) == expected_certificate_count
                and len(coverage_identities) == expected_certificate_count
                and coverage_identities
                != [identity[:3] for identity in transducer_identities]
            ):
                failures.append(
                    "frontier ordered tape/certificate identities mismatch: coverage"
                )
            if (
                len(transducer_identities) == expected_certificate_count
                and len(runner_checkpoint_identities) == expected_certificate_count
                and runner_checkpoint_identities
                != [
                    (identity[0], identity[2], identity[3])
                    for identity in transducer_identities
                ]
            ):
                failures.append(
                    "frontier ordered certificate identities mismatch: runner checkpoints"
                )

            tape_by_seed = dict(ordered_runner)
            for index, replay in enumerate(replays):
                if (
                    not isinstance(replay, dict)
                    or replay.get("seed") not in tape_by_seed
                    or replay.get("tape_sha256")
                    != tape_by_seed.get(replay.get("seed"))
                ):
                    failures.append(
                        f"frontier selected replay tape identity mismatch at index {index}"
                    )
                    break
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
        canonical, path = _confined_checksum_path(run_dir, relative)
        if not path.is_file():
            raise HarnessError(f"expected artifact missing: {canonical}")
        records.append(
            {
                "path": canonical,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    return records


def _confined_checksum_path(run_dir: Path, relative: Any) -> tuple[str, Path]:
    """Return one lexical descendant after rejecting every descendant symlink.

    ``run_dir`` is the explicit trust anchor and is resolved once.  Its own
    ancestors may therefore be platform symlinks (for example ``/tmp`` on some
    systems).  Every existing component *beneath* that resolved root, including
    the leaf, is checked with ``lstat`` before any candidate resolution.  Missing
    leaves remain valid for callers that are about to create them.
    """
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise HarnessError("checksum record path is not a canonical relative path")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise HarnessError(f"checksum record path escapes run directory: {relative!r}")
    canonical = pure.as_posix()
    if canonical != relative:
        raise HarnessError(f"checksum record path is not canonical: {relative!r}")
    root = run_dir.resolve()
    candidate = root
    missing_ancestor = False
    for part in pure.parts:
        candidate = candidate / part
        if missing_ancestor:
            continue
        try:
            component_stat = candidate.lstat()
        except FileNotFoundError:
            missing_ancestor = True
            continue
        except OSError as exc:
            raise HarnessError(
                f"checksum record component cannot be inspected: {relative!r}: "
                f"{type(exc).__name__}"
            ) from exc
        if stat.S_ISLNK(component_stat.st_mode):
            raise HarnessError(
                "checksum record path contains a symlink below the trusted root: "
                f"{relative!r}"
            )
    resolved_candidate = candidate.resolve(strict=False)
    try:
        resolved_candidate.relative_to(root)
    except ValueError as exc:
        raise HarnessError(
            f"checksum record path escapes run directory: {relative!r}"
        ) from exc
    return canonical, candidate


def _custody_anchor_path(run_dir: Path, run_id: str) -> Path:
    """Return a local sidecar path that is never part of the transported tree."""
    return run_dir.resolve().parent / (
        f".{run_dir.resolve().name}.{run_id}.trusted-local-custody.json"
    )


def _stage_custody_path(run_dir: Path, run_id: str) -> Path:
    return run_dir.resolve().parent / (
        f".{run_dir.resolve().name}.{run_id}.trusted-stage-custody.json"
    )


def _launch_custody_path(run_dir: Path, run_id: str) -> Path:
    return run_dir.resolve().parent / (
        f".{run_dir.resolve().name}.{run_id}.trusted-launch-custody.json"
    )


def _write_external_once(
    path: Path,
    payload: dict[str, Any],
    *,
    run_dir: Path,
    label: str,
) -> tuple[Path, str]:
    resolved = path.resolve()
    try:
        resolved.relative_to(run_dir.resolve())
    except ValueError:
        pass
    else:
        raise HarnessError(f"{label} must be outside the run tree")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if resolved.exists():
        if not resolved.is_file() or resolved.read_text() != encoded:
            raise HarnessError(f"existing {label} differs and will not be overwritten")
    else:
        try:
            with resolved.open("x") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
        except FileExistsError as exc:
            raise HarnessError(
                f"{label} appeared concurrently and was not overwritten"
            ) from exc
    return resolved, sha256_file(resolved)


def _custody_payload(run_dir: Path) -> dict[str, Any]:
    run_manifest = load_json(run_dir / "run_manifest.json")
    seed_manifest = load_json(run_dir / "seed_manifest.json")
    command_manifest = load_json(run_dir / "command_manifest.json")
    mode = run_manifest.get("mode")
    controlled_paths = (
        "run_manifest.json",
        "seed_manifest.json",
        "command_manifest.json",
        "authorization.json",
        "environment/pip_freeze.txt",
        "environment/machine.json",
        "environment/scientific_runtime.json",
        "environment/scientific_runtime.json",
    )
    required_paths = {
        "run_manifest.json",
        "seed_manifest.json",
        "command_manifest.json",
        "environment/pip_freeze.txt",
        "environment/machine.json",
        "environment/scientific_runtime.json",
    }
    if mode == "scientific":
        required_paths.add("authorization.json")
    missing_required = [
        relative
        for relative in sorted(required_paths)
        if not (run_dir / relative).is_file()
    ]
    if missing_required:
        raise HarnessError(
            "trusted custody control inputs are missing: "
            + ", ".join(missing_required)
        )
    records = []
    for relative in controlled_paths:
        path = run_dir / relative
        if path.is_file():
            records.append(
                {
                    "path": relative,
                    "present": True,
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
            )
        else:
            records.append({"path": relative, "present": False})
    seed_rows = seed_manifest.get("seeds", [])
    command_rows = command_manifest.get("commands", [])
    payload = {
        "schema_version": CUSTODY_SCHEMA,
        "run_id": run_manifest.get("run_id"),
        "mode": mode,
        "records": records,
        "prepared_inputs": run_manifest.get("inputs"),
        "prepared_inputs_sha256": canonical_json_sha256(
            run_manifest.get("inputs")
        ),
        "source_commit": run_manifest.get("git", {}).get("commit"),
        "ordered_seed_identities": [
            {
                "index": index,
                "seed": row.get("seed"),
                "context": row.get("context"),
                "split": row.get("split"),
                "weeks": row.get("weeks"),
                "expected_tape_sha256": row.get("expected_tape_sha256"),
            }
            for index, row in enumerate(seed_rows)
            if isinstance(row, dict)
        ],
        "command_templates": [
            {
                "index": index,
                "item_id": row.get("job_id") or row.get("seed_id"),
                "covered_seed_ids": row.get("covered_seed_ids"),
                "argv_template": row.get("argv_template"),
                "argv_template_sha256": row.get("argv_template_sha256"),
            }
            for index, row in enumerate(command_rows)
            if isinstance(row, dict)
        ],
        "materialization_policy": {
            "schema_version": "paper2_argv_materialization_v1",
            "substitutions": {
                "{python}": "python_executable",
                "{repo_root}": "repository_root",
                "{run_dir}": "run_directory",
                "{host_runtime_sha256}": "host_runtime_sha256",
                "{execution_nonce}": "execution_nonce",
            },
            "context_keys_exact": [
                "python_executable",
                "repository_root",
                "run_directory",
                "host_runtime_sha256",
                "execution_nonce",
            ],
        },
        "transported": False,
        "retrieval_may_overwrite": False,
    }
    payload["anchor_body_sha256"] = canonical_json_sha256(payload)
    return payload


def create_trusted_local_custody_anchor(
    run_dir: Path, *, anchor_path: Path | None = None
) -> tuple[Path, str]:
    run_manifest = load_json(run_dir / "run_manifest.json")
    status = load_json(run_dir / "status" / "run_status.json")
    if (
        status.get("state") != "prepared"
        or status.get("sealed_for_execution") is not True
        or (run_dir / "artifact_checksums.json").exists()
        or (run_dir / "status" / "run_completion_receipt.json").exists()
        or (run_dir / "retrieved").exists()
    ):
        raise HarnessError(
            "trusted custody anchor can only be minted for a sealed pre-execution run"
        )
    path = (
        anchor_path.resolve()
        if anchor_path is not None
        else _custody_anchor_path(run_dir, str(run_manifest.get("run_id")))
    )
    try:
        path.relative_to(run_dir.resolve())
    except ValueError:
        pass
    else:
        raise HarnessError("trusted custody anchor must be outside the run tree")
    expected = _custody_payload(run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or load_json(path) != expected:
            raise HarnessError(
                "existing trusted custody anchor differs and will not be overwritten"
            )
    else:
        encoded = json.dumps(expected, indent=2, sort_keys=True) + "\n"
        try:
            with path.open("x") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
        except FileExistsError as exc:
            raise HarnessError(
                "trusted custody anchor appeared concurrently and was not overwritten"
            ) from exc
    return path, sha256_file(path)


def create_trusted_stage_custody_receipt(
    run_dir: Path,
    *,
    source_bundle: Path,
    control_tar: Path,
    host: str,
    remote_run: str,
    custody_path: Path,
    custody_sha256: str,
) -> tuple[Path, str]:
    """Seal prepared bytes and transport archives before any remote transfer."""
    if host != DEFAULT_HOST:
        raise HarnessError(f"host must be the approved SSH alias {DEFAULT_HOST!r}")
    _validate_remote_value(remote_run, "remote run path")
    if not source_bundle.is_file() or not control_tar.is_file():
        raise HarnessError("stage custody requires completed source/control archives")
    run_manifest = load_json(run_dir / "run_manifest.json")
    resolved_custody = custody_path.resolve()
    if not resolved_custody.is_file() or sha256_file(resolved_custody) != custody_sha256:
        raise HarnessError("original trusted custody anchor is missing or changed")
    payload = {
        "schema_version": STAGE_CUSTODY_SCHEMA,
        "run_id": run_manifest.get("run_id"),
        "mode": run_manifest.get("mode"),
        "source_commit": run_manifest.get("git", {}).get("commit"),
        "trusted_local_custody": {
            "path": str(resolved_custody),
            "sha256": custody_sha256,
        },
        "source_bundle": {
            "sha256": sha256_file(source_bundle),
            "bytes": source_bundle.stat().st_size,
        },
        "control_tar": {
            "sha256": sha256_file(control_tar),
            "bytes": control_tar.stat().st_size,
        },
        "remote": {"ssh_alias": host, "remote_run": remote_run},
        "created_before_transfer": True,
        "custody_nonce": secrets.token_hex(32),
        "created_at_utc": utc_now(),
        "trust_model": (
            "caller retains this launch receipt digest out of band; no claim against "
            "malicious deletion of the trusted local custody store without signing or TSA"
        ),
        "transported": False,
        "retrieval_may_overwrite": False,
    }
    payload["stage_body_sha256"] = canonical_json_sha256(payload)
    return _write_external_once(
        _stage_custody_path(run_dir, str(run_manifest.get("run_id"))),
        payload,
        run_dir=run_dir,
        label="trusted stage custody receipt",
    )


def create_trusted_launch_custody_receipt(
    run_dir: Path,
    *,
    stage_path: Path,
    stage_sha256: str,
    materialization_context: dict[str, str],
    remote_python_requested: str,
    remote_launch_shell_command: str,
) -> tuple[Path, str]:
    """Seal the exact remote runtime and argv before process submission."""
    resolved_stage = stage_path.resolve()
    if not resolved_stage.is_file() or sha256_file(resolved_stage) != stage_sha256:
        raise HarnessError("trusted stage custody receipt is missing or changed")
    stage = load_json(resolved_stage)
    if stage.get("schema_version") != STAGE_CUSTODY_SCHEMA:
        raise HarnessError("trusted stage custody receipt schema mismatch")
    stage_body = dict(stage)
    stage_body_sha256 = stage_body.pop("stage_body_sha256", None)
    if stage_body_sha256 != canonical_json_sha256(stage_body):
        raise HarnessError("trusted stage custody receipt body digest mismatch")
    command_manifest = load_json(run_dir / "command_manifest.json")
    materialized_commands = []
    for index, command in enumerate(command_manifest.get("commands", [])):
        if not isinstance(command, dict):
            raise HarnessError("launch custody command manifest is malformed")
        argv = _materialize_argv_from_context(
            command.get("argv_template"), materialization_context
        )
        materialized_commands.append(
            {
                "index": index,
                "item_id": command.get("job_id") or command.get("seed_id"),
                "materialized_argv": argv,
                "materialized_command_sha256": canonical_json_sha256(argv),
            }
        )
    run_manifest = load_json(run_dir / "run_manifest.json")
    payload = {
        "schema_version": LAUNCH_CUSTODY_SCHEMA,
        "run_id": run_manifest.get("run_id"),
        "mode": run_manifest.get("mode"),
        "source_commit": run_manifest.get("git", {}).get("commit"),
        "trusted_stage_custody": {
            "path": str(resolved_stage),
            "sha256": stage_sha256,
        },
        "trusted_local_custody": stage.get("trusted_local_custody"),
        "source_bundle": stage.get("source_bundle"),
        "control_tar": stage.get("control_tar"),
        "remote": stage.get("remote"),
        "remote_python_requested": remote_python_requested,
        "materialization_context": materialization_context,
        "execution_nonce": materialization_context["execution_nonce"],
        "host_runtime_sha256": materialization_context["host_runtime_sha256"],
        "portable_runtime_sha256": run_manifest.get("inputs", {}).get(
            "portable_runtime_sha256"
        ),
        "scientific_child_environment_sha256": run_manifest.get("inputs", {}).get(
            "scientific_child_environment_sha256"
        ),
        "isolated_bootstrap_sha256": run_manifest.get("inputs", {}).get(
            "isolated_bootstrap_sha256"
        ),
        "materialized_commands": materialized_commands,
        "remote_launch_shell_command": remote_launch_shell_command,
        "remote_launch_shell_command_sha256": sha256_bytes(
            remote_launch_shell_command.encode()
        ),
        "created_before_submission": True,
        "custody_nonce": secrets.token_hex(32),
        "created_at_utc": utc_now(),
        "transported": False,
        "retrieval_may_overwrite": False,
    }
    payload["launch_body_sha256"] = canonical_json_sha256(payload)
    return _write_external_once(
        _launch_custody_path(run_dir, str(run_manifest.get("run_id"))),
        payload,
        run_dir=run_dir,
        label="trusted launch custody receipt",
    )


def _verify_trusted_local_custody(
    run_dir: Path,
    *,
    anchor_path: Path | None,
    anchor_sha256: str | None,
) -> tuple[dict[str, Any] | None, list[str]]:
    failures: list[str] = []
    if anchor_path is None or not _is_sha256(anchor_sha256):
        return None, [
            "retrieved verification requires an explicit trusted local custody anchor and digest"
        ]
    resolved_anchor = anchor_path.resolve()
    try:
        resolved_anchor.relative_to(run_dir.resolve())
    except ValueError:
        pass
    else:
        return None, ["trusted local custody anchor is inside the retrieved tree"]
    if not resolved_anchor.is_file():
        return None, ["trusted local custody anchor is missing"]
    if sha256_file(resolved_anchor) != anchor_sha256:
        return None, ["trusted local custody anchor digest mismatch"]
    try:
        anchor = load_json(resolved_anchor)
    except HarnessError as exc:
        return None, [f"trusted local custody anchor is unreadable: {exc}"]
    if not isinstance(anchor, dict) or anchor.get("schema_version") != CUSTODY_SCHEMA:
        return None, ["trusted local custody anchor schema mismatch"]
    anchor_body = dict(anchor)
    anchor_body_sha256 = anchor_body.pop("anchor_body_sha256", None)
    if anchor_body_sha256 != canonical_json_sha256(anchor_body):
        failures.append("trusted local custody anchor body digest mismatch")
    records = anchor.get("records")
    if not isinstance(records, list):
        return anchor, [*failures, "trusted local custody records are malformed"]
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            failures.append("trusted local custody record is malformed")
            continue
        try:
            relative, path = _confined_checksum_path(run_dir, record["path"])
        except HarnessError as exc:
            failures.append(str(exc))
            continue
        expected_present = record.get("present") is True
        if path.is_file() != expected_present:
            failures.append(f"trusted local custody presence mismatch: {relative}")
            continue
        if expected_present and (
            record.get("sha256") != sha256_file(path)
            or record.get("bytes") != path.stat().st_size
        ):
            failures.append(f"trusted local custody byte mismatch: {relative}")
    if failures:
        return anchor, failures
    try:
        recomputed = _custody_payload(run_dir)
    except (HarnessError, TypeError, AttributeError, ValueError) as exc:
        return anchor, [f"retrieved custody reconstruction failed: {exc}"]
    if recomputed != anchor:
        failures.append(
            "retrieved command templates, ordered seeds, inputs, or materialization policy differ from trusted local custody"
        )
    return anchor, failures


def _load_external_receipt(
    path: Path | None,
    digest: str | None,
    *,
    run_dir: Path,
    schema: str,
    body_digest_key: str,
    label: str,
) -> tuple[dict[str, Any] | None, list[str]]:
    if path is None or not _is_sha256(digest):
        return None, [f"retrieved verification requires an explicit {label} and digest"]
    resolved = path.resolve()
    try:
        resolved.relative_to(run_dir.resolve())
    except ValueError:
        pass
    else:
        return None, [f"{label} is inside the retrieved tree"]
    if not resolved.is_file():
        return None, [f"{label} is missing"]
    if sha256_file(resolved) != digest:
        return None, [f"{label} digest mismatch"]
    try:
        payload = load_json(resolved)
    except HarnessError as exc:
        return None, [f"{label} is unreadable: {exc}"]
    if not isinstance(payload, dict) or payload.get("schema_version") != schema:
        return None, [f"{label} schema mismatch"]
    nonce = payload.get("custody_nonce")
    if (
        not isinstance(nonce, str)
        or len(nonce) != 64
        or any(character not in "0123456789abcdef" for character in nonce)
    ):
        return payload, [f"{label} custody nonce is missing or malformed"]
    body = dict(payload)
    body_digest = body.pop(body_digest_key, None)
    if body_digest != canonical_json_sha256(body):
        return payload, [f"{label} body digest mismatch"]
    return payload, []


def _verify_trusted_launch_custody(
    run_dir: Path,
    *,
    launch_path: Path | None,
    launch_sha256: str | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, list[str]]:
    """Verify the external pre-transfer -> pre-launch custody chain."""
    launch, failures = _load_external_receipt(
        launch_path,
        launch_sha256,
        run_dir=run_dir,
        schema=LAUNCH_CUSTODY_SCHEMA,
        body_digest_key="launch_body_sha256",
        label="trusted launch custody receipt",
    )
    if launch is None or failures:
        return launch, None, failures
    stage_ref = launch.get("trusted_stage_custody", {})
    stage_path_value = stage_ref.get("path") if isinstance(stage_ref, dict) else None
    stage_sha256 = stage_ref.get("sha256") if isinstance(stage_ref, dict) else None
    stage, stage_failures = _load_external_receipt(
        Path(stage_path_value) if isinstance(stage_path_value, str) else None,
        stage_sha256,
        run_dir=run_dir,
        schema=STAGE_CUSTODY_SCHEMA,
        body_digest_key="stage_body_sha256",
        label="trusted stage custody receipt",
    )
    failures.extend(stage_failures)
    if stage is None or failures:
        return launch, stage, failures
    for key in (
        "run_id",
        "mode",
        "source_commit",
        "trusted_local_custody",
        "source_bundle",
        "control_tar",
        "remote",
    ):
        if launch.get(key) != stage.get(key):
            failures.append(f"launch/stage custody mismatch: {key}")
    local_ref = stage.get("trusted_local_custody", {})
    local_path_value = local_ref.get("path") if isinstance(local_ref, dict) else None
    local_sha256 = local_ref.get("sha256") if isinstance(local_ref, dict) else None
    _anchor, local_failures = _verify_trusted_local_custody(
        run_dir,
        anchor_path=(
            Path(local_path_value) if isinstance(local_path_value, str) else None
        ),
        anchor_sha256=local_sha256,
    )
    failures.extend(local_failures)
    if failures:
        return launch, stage, failures

    run_manifest = load_json(run_dir / "run_manifest.json")
    if launch.get("run_id") != run_manifest.get("run_id"):
        failures.append("launch custody run id mismatch")
    if launch.get("mode") != run_manifest.get("mode"):
        failures.append("launch custody mode mismatch")
    if launch.get("source_commit") != run_manifest.get("git", {}).get("commit"):
        failures.append("launch custody source commit mismatch")
    launch_expected = {
        "execution_nonce": run_manifest.get("inputs", {}).get("execution_nonce"),
        "portable_runtime_sha256": run_manifest.get("inputs", {}).get(
            "portable_runtime_sha256"
        ),
        "scientific_child_environment_sha256": run_manifest.get("inputs", {}).get(
            "scientific_child_environment_sha256"
        ),
        "isolated_bootstrap_sha256": run_manifest.get("inputs", {}).get(
            "isolated_bootstrap_sha256"
        ),
    }
    for key, expected in launch_expected.items():
        if launch.get(key) != expected:
            failures.append(f"launch custody runtime identity mismatch: {key}")

    command_manifest = load_json(run_dir / "command_manifest.json")
    commands = command_manifest.get("commands", [])
    committed = launch.get("materialized_commands", [])
    if not isinstance(commands, list) or not isinstance(committed, list):
        failures.append("launch custody materialized commands are malformed")
        return launch, stage, failures
    if len(commands) != len(committed):
        failures.append("launch custody command count mismatch")
        return launch, stage, failures
    context = launch.get("materialization_context")
    if (
        not isinstance(context, dict)
        or launch.get("host_runtime_sha256") != context.get("host_runtime_sha256")
    ):
        failures.append("launch custody host runtime differs from materialization context")
    try:
        expected_commands = [
            {
                "index": index,
                "item_id": command.get("job_id") or command.get("seed_id"),
                "materialized_argv": _materialize_argv_from_context(
                    command.get("argv_template"), context
                ),
            }
            for index, command in enumerate(commands)
            if isinstance(command, dict)
        ]
    except (HarnessError, TypeError) as exc:
        failures.append(f"launch custody argv reconstruction failed: {exc}")
        return launch, stage, failures
    for row in expected_commands:
        row["materialized_command_sha256"] = canonical_json_sha256(
            row["materialized_argv"]
        )
    if expected_commands != committed:
        failures.append("launch custody exact materialized argv mismatch")
        return launch, stage, failures

    for row in committed:
        item_id = row.get("item_id")
        if not isinstance(item_id, str):
            failures.append("launch custody command identity is malformed")
            continue
        receipt_path = run_dir / "status" / "jobs" / f"{item_id}.execution_receipt.json"
        if not receipt_path.is_file():
            failures.append(f"launch-custody execution receipt is missing: {item_id}")
            continue
        receipt = load_json(receipt_path)
        for key, expected in (
            ("materialization_context", context),
            ("materialized_argv", row.get("materialized_argv")),
            (
                "materialized_command_sha256",
                row.get("materialized_command_sha256"),
            ),
            ("execution_nonce", launch.get("execution_nonce")),
            ("host_runtime_sha256", launch.get("host_runtime_sha256")),
            (
                "portable_runtime_sha256",
                launch.get("portable_runtime_sha256"),
            ),
            (
                "scientific_child_environment_sha256",
                launch.get("scientific_child_environment_sha256"),
            ),
            (
                "isolated_bootstrap_sha256",
                launch.get("isolated_bootstrap_sha256"),
            ),
        ):
            if receipt.get(key) != expected:
                failures.append(f"launch custody execution mismatch: {item_id}:{key}")
    return launch, stage, failures


def execute_run(
    *,
    run_dir: Path,
    repo_root: Path = ROOT,
    location: str = "local",
    expected_host_runtime_sha256: str | None = None,
) -> int:
    run_manifest, seed_manifest, command_manifest = _validate_prepared_inputs(run_dir, repo_root)
    if run_manifest["mode"] == "dry-run":
        raise HarnessError("dry-run manifest cannot execute")
    if run_manifest["mode"] == "smoke":
        if run_manifest["execution"]["weeks"] > 4 or len(seed_manifest["seeds"]) != 1:
            raise HarnessError("smoke execution exceeded one-tape W4 ceiling")
    run_id = run_manifest["run_id"]
    host_runtime_sha256 = (
        expected_host_runtime_sha256
        or run_manifest["inputs"]["host_runtime_sha256"]
    )
    if not _is_sha256(host_runtime_sha256):
        raise HarnessError("execution host-runtime digest is malformed")
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
            materialization_context = _argv_materialization_context(
                repo_root=repo_root,
                run_dir=run_dir,
                host_runtime_sha256=host_runtime_sha256,
                execution_nonce=run_manifest["inputs"]["execution_nonce"],
            )
            argv = _materialize_argv_from_context(
                command["argv_template"], materialization_context
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
                    env=scientific_child_environment(),
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
            runtime_attestation_relative = command.get(
                "runtime_attestation_relative"
            )
            runtime_attestation_path = (
                run_dir / runtime_attestation_relative
                if isinstance(runtime_attestation_relative, str)
                else None
            )
            runtime_attestation = None
            runtime_attestation_file_sha256 = None
            if runtime_attestation_path is not None and runtime_attestation_path.is_file():
                try:
                    runtime_attestation = validate_runtime_attestation_payload(
                        load_json(runtime_attestation_path)
                    )
                    runtime_attestation_file_sha256 = sha256_file(
                        runtime_attestation_path
                    )
                    if runtime_attestation.get("runtime_sha256") != materialization_context[
                        "host_runtime_sha256"
                    ]:
                        output_error = "child host-runtime attestation digest mismatch"
                except HarnessError as exc:
                    output_error = str(exc)
            if output_path.is_file():
                try:
                    parsed = load_json(output_path)
                    output_hash = sha256_file(output_path)
                    output_valid = isinstance(parsed, dict)
                    if runtime_attestation is None or output_error is not None:
                        output_valid = False
                    if output_valid and run_manifest["mode"] == "scientific":
                        native_manifest = None
                        runner_manifest_relative = command.get("runner_manifest_relative")
                        if runner_manifest_relative:
                            native_manifest_path = run_dir / runner_manifest_relative
                            if native_manifest_path.is_file():
                                native_manifest = load_json(native_manifest_path)
                        assurance_failures = _validate_scientific_result(
                            parsed, run_manifest, native_manifest, seed_manifest
                        )
                        if (
                            isinstance(native_manifest, dict)
                            and native_manifest.get("result_sha256") != output_hash
                        ):
                            assurance_failures.append(
                                "runner manifest result hash mismatch"
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
                except (HarnessError, TypeError, AttributeError, ValueError) as exc:
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
                    "scientific_child_environment_sha256": run_manifest["inputs"][
                        "scientific_child_environment_sha256"
                    ],
                    "execution_nonce": run_manifest["inputs"]["execution_nonce"],
                    "execution_role": command.get("execution_role"),
                    "isolated_bootstrap_sha256": run_manifest["inputs"][
                        "isolated_bootstrap_sha256"
                    ],
                    "host_runtime_sha256": runtime_attestation.get(
                        "runtime_sha256"
                    ),
                    "portable_runtime_sha256": runtime_attestation.get(
                        "portable_sha256"
                    ),
                    "runtime_attestation_relative": runtime_attestation_relative,
                    "runtime_attestation_file_sha256": runtime_attestation_file_sha256,
                    "seed_manifest_sha256": run_manifest["inputs"][
                        "seed_manifest_sha256"
                    ],
                    "command_manifest_sha256": run_manifest["inputs"][
                        "command_manifest_sha256"
                    ],
                    "materialization_context": materialization_context,
                    "materialized_argv": argv,
                    "materialized_command_sha256": command_sha,
                    "output_relative": command["output_relative"],
                    "output_sha256": output_hash,
                    "validated_not_independently_audited": True,
                    "evidence": False,
                }
                atomic_write_json(run_dir / receipt_relative, receipt)
                checksum_paths.append(receipt_relative)
                if runtime_attestation_relative:
                    checksum_paths.append(runtime_attestation_relative)
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


def verify_artifacts(
    run_dir: Path,
    *,
    retrieved: bool,
    trusted_local_anchor: Path | None = None,
    trusted_local_anchor_sha256: str | None = None,
    trusted_launch_receipt: Path | None = None,
    trusted_launch_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    trusted_anchor = None
    trusted_launch = None
    if retrieved:
        trusted_launch, _trusted_stage, custody_failures = (
            _verify_trusted_launch_custody(
                run_dir,
                launch_path=trusted_launch_receipt,
                launch_sha256=trusted_launch_receipt_sha256,
            )
        )
        local_ref = (
            trusted_launch.get("trusted_local_custody", {})
            if isinstance(trusted_launch, dict)
            else {}
        )
        if trusted_local_anchor is not None and (
            str(trusted_local_anchor.resolve()) != local_ref.get("path")
            or trusted_local_anchor_sha256 != local_ref.get("sha256")
        ):
            custody_failures.append(
                "explicit local anchor differs from trusted launch custody"
            )
        trusted_anchor, local_failures = _verify_trusted_local_custody(
            run_dir,
            anchor_path=(
                Path(local_ref["path"])
                if isinstance(local_ref, dict)
                and isinstance(local_ref.get("path"), str)
                else None
            ),
            anchor_sha256=(
                local_ref.get("sha256") if isinstance(local_ref, dict) else None
            ),
        )
        if not custody_failures:
            custody_failures.extend(local_failures)
        if custody_failures:
            result = {
                "schema_version": SCHEMA,
                "run_id": (
                    trusted_anchor.get("run_id")
                    if isinstance(trusted_anchor, dict)
                    else "UNTRUSTED_RETRIEVED_BUNDLE"
                ),
                "verified_at_utc": utc_now(),
                "retrieved": True,
                "checks_passed": False,
                "failures": custody_failures,
                "checked_records": [],
                "trusted_local_anchor_sha256": trusted_local_anchor_sha256,
                "trusted_launch_receipt_sha256": trusted_launch_receipt_sha256,
                "evidence": False,
                "evidence_status": "VERIFICATION_FAILED_NOT_EVIDENCE",
            }
            atomic_write_json(run_dir / "retrieval_verification.json", result)
            return result
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
    run_id = run_manifest.get("run_id")
    for label, document in (
        ("run manifest", run_manifest),
        ("seed manifest", seed_manifest),
        ("command manifest", command_manifest),
        ("run status", status),
    ):
        if document.get("schema_version") != SCHEMA:
            failures.append(f"{label} schema mismatch")
        if document.get("run_id") != run_id:
            failures.append(f"{label} run id mismatch")
    if seed_manifest.get("mode") != run_manifest.get("mode"):
        failures.append("seed manifest mode mismatch")
    if command_manifest.get("mode") != run_manifest.get("mode"):
        failures.append("command manifest mode mismatch")
    if run_manifest.get("inputs", {}).get("seed_manifest_sha256") != sha256_file(
        seeds_path
    ):
        failures.append("run manifest seed-manifest hash mismatch")
    if run_manifest.get("inputs", {}).get(
        "command_manifest_sha256"
    ) != sha256_file(commands_path):
        failures.append("run manifest command-manifest hash mismatch")

    seed_rows = seed_manifest.get("seeds", [])
    if not isinstance(seed_rows, list):
        failures.append("seed manifest rows are missing or malformed")
        seed_rows = []
    failures.extend(
        f"independent tape audit failed: {failure}"
        for failure in _validate_seed_tape_commitments(
            seed_rows,
            regenerate=True,
        )
    )
    expected_seed_count = seed_manifest.get("seed_count")
    if (
        isinstance(expected_seed_count, bool)
        or not isinstance(expected_seed_count, int)
        or expected_seed_count != len(seed_rows)
    ):
        failures.append("seed manifest seed_count mismatch")
        expected_seed_count = len(seed_rows)
    expected_seed_rows: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(seed_rows):
        if not isinstance(row, dict):
            failures.append(f"malformed seed manifest row {index}")
            continue
        seed = row.get("seed")
        context = row.get("context")
        if isinstance(seed, bool) or not isinstance(seed, int) or not isinstance(
            context, str
        ):
            failures.append(f"malformed seed identity at index {index}")
            continue
        seed_id = f"seed_{seed}_{context}"
        if seed_id in expected_seed_rows:
            failures.append(f"duplicate seed identity: {seed_id}")
            continue
        expected_seed_rows[seed_id] = row

    command_rows = command_manifest.get("commands", [])
    if not isinstance(command_rows, list):
        failures.append("command manifest rows are missing or malformed")
        command_rows = []
    commands_by_item: dict[str, dict[str, Any]] = {}
    seed_owner: dict[str, str] = {}
    for index, command in enumerate(command_rows):
        if not isinstance(command, dict):
            failures.append(f"malformed command manifest row {index}")
            continue
        item_id = command.get("job_id") or command.get("seed_id")
        if not isinstance(item_id, str) or not item_id:
            failures.append("command manifest item identity is malformed")
            continue
        if item_id in commands_by_item:
            failures.append(f"duplicate command manifest item identity: {item_id}")
            continue
        commands_by_item[item_id] = command
        if command.get("argv_template_sha256") != canonical_json_sha256(
            command.get("argv_template")
        ):
            failures.append(f"command template hash mismatch: {item_id}")
        covered_seed_ids = (
            command.get("covered_seed_ids")
            if command.get("job_id")
            else [command.get("seed_id")]
        )
        if not isinstance(covered_seed_ids, list) or not covered_seed_ids:
            failures.append(f"command seed coverage is malformed: {item_id}")
            continue
        for seed_id in covered_seed_ids:
            if seed_id not in expected_seed_rows:
                failures.append(
                    f"command covers unknown seed identity: {item_id}:{seed_id}"
                )
                continue
            if seed_id in seed_owner:
                failures.append(f"seed identity has multiple command owners: {seed_id}")
                continue
            seed_owner[seed_id] = item_id
    if set(seed_owner) != set(expected_seed_rows):
        failures.append("command manifest does not cover the exact seed manifest")

    required_protected_paths = {
        "run_manifest.json",
        "seed_manifest.json",
        "command_manifest.json",
        "status/run_completion_receipt.json",
    }
    for row in seed_rows:
        if not isinstance(row, dict) or "seed" not in row or "context" not in row:
            continue
        required_protected_paths.add(
            f"status/seeds/seed_{row['seed']}_{row['context']}.json"
        )
    for command in command_rows:
        if not isinstance(command, dict):
            continue
        item_id = command.get("job_id") or command.get("seed_id")
        if not isinstance(item_id, str) or not item_id:
            continue
        if command.get("job_id"):
            required_protected_paths.add(f"status/jobs/{item_id}.json")
        required_protected_paths.add(
            f"status/jobs/{item_id}.execution_receipt.json"
        )
        for relative_key in (
            "output_relative",
            "stdout_relative",
            "stderr_relative",
            "runner_manifest_relative",
        ):
            relative = command.get(relative_key)
            if isinstance(relative, str):
                required_protected_paths.add(relative)
        if (
            run_manifest.get("mode") == "scientific"
            and not isinstance(command.get("runner_manifest_relative"), str)
        ):
            failures.append(
                f"scientific command lacks a runner manifest path: {item_id}"
            )
    missing_protected = sorted(required_protected_paths - seen_paths)
    if missing_protected:
        failures.append(
            "immutable checksum manifest omits protected status/receipt paths: "
            + ", ".join(missing_protected)
        )

    jobs_dir = run_dir / "status" / "jobs"
    actual_receipts = {
        path.name
        for path in jobs_dir.glob("*.execution_receipt.json")
        if path.is_file()
    }
    expected_receipts = {
        f"{item_id}.execution_receipt.json" for item_id in commands_by_item
    }
    if actual_receipts != expected_receipts:
        failures.append("execution-receipt file identities mismatch command manifest")
    actual_job_statuses = {
        path.name
        for path in jobs_dir.glob("*.json")
        if path.is_file() and not path.name.endswith(".execution_receipt.json")
    }
    expected_job_statuses = {
        f"{item_id}.json"
        for item_id, command in commands_by_item.items()
        if command.get("job_id")
    }
    if actual_job_statuses != expected_job_statuses:
        failures.append("job-status file identities mismatch command manifest")

    command_output_sha256: dict[str, str] = {}
    for item_id, command in commands_by_item.items():
        try:
            output_relative, output_path = _confined_checksum_path(
                run_dir, command.get("output_relative")
            )
        except HarnessError as exc:
            failures.append(str(exc))
            continue
        if not output_path.is_file():
            failures.append(f"command output is missing: {item_id}")
            continue
        output_sha256 = sha256_file(output_path)
        command_output_sha256[item_id] = output_sha256
        if output_relative not in seen_paths:
            failures.append(f"command output is not checksum protected: {item_id}")

        status_relative = (
            f"status/jobs/{item_id}.json"
            if command.get("job_id")
            else f"status/seeds/{item_id}.json"
        )
        _relative, command_status_path = _confined_checksum_path(
            run_dir, status_relative
        )
        command_status = (
            load_json(command_status_path) if command_status_path.is_file() else {}
        )
        status_expected = {
            "schema_version": SCHEMA,
            "run_id": run_id,
            "item_id": item_id,
            "seed": command.get("seed"),
            "context": command.get("context"),
            "phase": command.get("phase"),
            "state": "completed",
            "returncode": 0,
            "output_relative": command.get("output_relative"),
            "output_valid_json": True,
            "output_sha256": output_sha256,
            "output_error": None,
            "stdout_relative": command.get("stdout_relative"),
            "stderr_relative": command.get("stderr_relative"),
            "evidence": False,
        }
        for key, value in status_expected.items():
            if key not in command_status or not _semantic_equal(
                command_status.get(key), value
            ):
                failures.append(f"command/job status mismatch: {item_id}:{key}")
        command_sha256 = command_status.get("command_sha256")
        if not _is_sha256(command_sha256):
            failures.append(f"command/job status has malformed command hash: {item_id}")

        receipt_relative = f"status/jobs/{item_id}.execution_receipt.json"
        _relative, receipt_path = _confined_checksum_path(run_dir, receipt_relative)
        receipt = load_json(receipt_path) if receipt_path.is_file() else {}
        materialization_context = receipt.get("materialization_context")
        materialized_argv = receipt.get("materialized_argv")
        expected_materialized_argv = None
        recomputed_command_sha256 = None
        try:
            expected_materialized_argv = _materialize_argv_from_context(
                command.get("argv_template"), materialization_context
            )
        except (HarnessError, TypeError) as exc:
            failures.append(
                f"execution receipt materialization context mismatch: {item_id}: {exc}"
            )
        if (
            not isinstance(materialized_argv, list)
            or not materialized_argv
            or any(not isinstance(value, str) for value in materialized_argv)
        ):
            failures.append(
                f"execution receipt lacks canonical materialized argv: {item_id}"
            )
        elif expected_materialized_argv is not None:
            if materialized_argv != expected_materialized_argv:
                failures.append(
                    f"execution receipt argv does not match command template: {item_id}"
                )
            recomputed_command_sha256 = canonical_json_sha256(
                expected_materialized_argv
            )
            if command_sha256 != recomputed_command_sha256:
                failures.append(
                    f"command/job status command hash is not recomputable: {item_id}"
                )
        receipt_expected = {
            "schema_version": SCHEMA,
            "run_id": run_id,
            "job_id": item_id,
            "mode": run_manifest.get("mode"),
            "git_commit": run_manifest.get("git", {}).get("commit"),
            "contract_sha256": run_manifest.get("inputs", {}).get(
                "contract_sha256"
            ),
            "result_contract_sha256": run_manifest.get("inputs", {}).get(
                "result_contract_sha256"
            ),
            "runner_sha256": run_manifest.get("inputs", {}).get("runner_sha256"),
            "harness_sha256": run_manifest.get("inputs", {}).get(
                "harness_sha256"
            ),
            "environment_sha256": run_manifest.get("inputs", {}).get(
                "environment_sha256"
            ),
            "seed_manifest_sha256": sha256_file(seeds_path),
            "command_manifest_sha256": sha256_file(commands_path),
            "materialization_context": materialization_context,
            "materialized_argv": expected_materialized_argv,
            "materialized_command_sha256": recomputed_command_sha256,
            "output_relative": command.get("output_relative"),
            "output_sha256": output_sha256,
            "validated_not_independently_audited": True,
            "evidence": False,
        }
        for key, value in receipt_expected.items():
            if key not in receipt or not _semantic_equal(receipt.get(key), value):
                failures.append(f"execution receipt mismatch: {item_id}:{key}")

        if run_manifest.get("mode") == "scientific":
            runner_manifest_relative = command.get("runner_manifest_relative")
            try:
                runner_relative, runner_manifest_path = _confined_checksum_path(
                    run_dir, runner_manifest_relative
                )
            except HarnessError as exc:
                failures.append(str(exc))
                runner_relative = None
                runner_manifest_path = None
            if runner_relative is not None and runner_relative not in seen_paths:
                failures.append(
                    f"scientific runner manifest is not checksum protected: {item_id}"
                )
            if runner_manifest_path is None or not runner_manifest_path.is_file():
                failures.append(f"scientific runner manifest is missing: {item_id}")
            else:
                try:
                    scientific_payload = load_json(output_path)
                    scientific_runner_manifest = load_json(runner_manifest_path)
                    if not isinstance(scientific_payload, dict) or not isinstance(
                        scientific_runner_manifest, dict
                    ):
                        failures.append(
                            "scientific result revalidation failed: result or runner manifest is not an object"
                        )
                    else:
                        scientific_failures = _validate_scientific_result(
                            scientific_payload,
                            run_manifest,
                            scientific_runner_manifest,
                            seed_manifest,
                        )
                        failures.extend(
                            f"scientific result revalidation failed: {failure}"
                            for failure in scientific_failures
                        )
                        if (
                            scientific_runner_manifest.get("result_sha256")
                            != output_sha256
                        ):
                            failures.append(
                                "scientific runner manifest result hash mismatch"
                            )
                except (HarnessError, TypeError, AttributeError, ValueError) as exc:
                    failures.append(
                        f"scientific result revalidation failed: {exc}"
                    )

    seed_statuses = {
        path.name: path
        for path in (run_dir / "status" / "seeds").glob("*.json")
        if path.is_file()
    }
    expected_seed_status_names = {
        f"{seed_id}.json" for seed_id in expected_seed_rows
    }
    if set(seed_statuses) != expected_seed_status_names:
        failures.append("per-seed status identities mismatch seed manifest")
    for seed_id, seed_row in expected_seed_rows.items():
        seed_status_name = f"{seed_id}.json"
        seed_status_path = seed_statuses.get(seed_status_name)
        if seed_status_path is None:
            continue
        seed_status_relative = str(seed_status_path.relative_to(run_dir))
        if seed_status_relative not in seen_paths:
            failures.append(
                f"per-seed status is not checksum protected: {seed_status_relative}"
            )
        seed_status = load_json(seed_status_path)
        owner_id = seed_owner.get(seed_id)
        owner = commands_by_item.get(owner_id, {})
        grouped = bool(owner.get("job_id"))
        expected_output_sha256 = command_output_sha256.get(owner_id)
        status_expected = {
            "schema_version": SCHEMA,
            "run_id": run_id,
            "seed": seed_row.get("seed"),
            "context": seed_row.get("context"),
            "state": "completed",
            "returncode": 0,
            "output_sha256": expected_output_sha256,
            "evidence": False,
        }
        if grouped:
            status_expected.update(
                {"seed_id": seed_id, "phase": seed_row.get("split")}
            )
        else:
            status_expected.update(
                {"item_id": seed_id, "phase": owner.get("phase")}
            )
        for key, value in status_expected.items():
            if key not in seed_status or not _semantic_equal(
                seed_status.get(key), value
            ):
                failures.append(f"per-seed status mismatch: {seed_id}:{key}")
    completion_receipt_relative = "status/run_completion_receipt.json"
    _relative, completion_receipt_path = _confined_checksum_path(
        run_dir, completion_receipt_relative
    )
    if (
        completion_receipt_relative in seen_paths
        and completion_receipt_path.is_file()
    ):
        completion_receipt = load_json(completion_receipt_path)
        receipt_location = completion_receipt.get("location")
        expected_evidence_status = {
            "local": "COMPLETED_HASHED_AUDIT_PENDING_NOT_EVIDENCE",
            "vps": "COMPLETED_REMOTE_NOT_RETRIEVED_NOT_EVIDENCE",
        }.get(receipt_location)
        if expected_evidence_status is None:
            failures.append("run completion receipt has invalid location")
        receipt_expected = {
            "schema_version": SCHEMA,
            "run_id": run_id,
            "state": "completed",
            "location": receipt_location,
            "seed_count": expected_seed_count,
            "completed_seed_count": expected_seed_count,
            "failed_seed_count": 0,
            "run_manifest_sha256": sha256_file(run_path),
            "seed_manifest_sha256": sha256_file(seeds_path),
            "command_manifest_sha256": sha256_file(commands_path),
            "immutable_status_snapshot": True,
            "evidence": False,
            "evidence_status": expected_evidence_status,
        }
        for key, value in receipt_expected.items():
            if key not in completion_receipt or not _semantic_equal(
                completion_receipt.get(key), value
            ):
                failures.append(f"run completion receipt mismatch: {key}")
        status_expected = {
            "schema_version": SCHEMA,
            "run_id": run_id,
            "state": "completed",
            "location": receipt_location,
            "evidence": False,
            "evidence_status": expected_evidence_status,
            "seed_count": expected_seed_count,
            "completed_seed_count": expected_seed_count,
            "failed_seed_count": 0,
            "independent_scientific_audit_required": True,
        }
        for key, value in status_expected.items():
            if key not in status or not _semantic_equal(status.get(key), value):
                failures.append(f"run status semantic mismatch: {key}")
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
        "trusted_local_anchor_sha256": (
            (
                trusted_launch.get("trusted_local_custody", {}).get("sha256")
                if isinstance(trusted_launch, dict)
                else None
            )
            if retrieved
            else None
        ),
        "trusted_launch_receipt_sha256": (
            trusted_launch_receipt_sha256 if retrieved else None
        ),
        "evidence": False,
        "evidence_status": (
            "RETRIEVED_CUSTODY_AND_HASH_VERIFIED_INDEPENDENT_AUDIT_PENDING_NOT_EVIDENCE"
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
    machine_command = (
        f"{_remote_shell_path(remote_python)} -I -B -S -c {shlex.quote(code)}"
    )
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
            f"{_remote_shell_path(remote_python)} -I -B -m pip freeze --all",
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
            f"{_remote_shell_path(remote_python)} -I -B -c "
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
    custody_path, custody_sha256 = create_trusted_local_custody_anchor(run_dir)
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
    stage_custody_path, stage_custody_sha256 = (
        create_trusted_stage_custody_receipt(
            run_dir,
            source_bundle=source_bundle,
            control_tar=control_tar,
            host=host,
            remote_run=remote_run,
            custody_path=custody_path,
            custody_sha256=custody_sha256,
        )
    )
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
    remote_hashes = run_capture(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            host,
            (
                f"sha256sum {remote_run_q}/source.bundle "
                f"{remote_run_q}/control.tar"
            ),
        ]
    )
    uploaded = [
        line.split()[0]
        for line in remote_hashes.stdout.splitlines()
        if line.split()
    ]
    expected_uploaded = [sha256_file(source_bundle), sha256_file(control_tar)]
    if uploaded != expected_uploaded:
        raise HarnessError("remote source/control archive digest mismatch")
    stage_after_upload = load_json(stage_custody_path)
    if (
        stage_after_upload.get("source_bundle", {}).get("sha256")
        != expected_uploaded[0]
        or stage_after_upload.get("control_tar", {}).get("sha256")
        != expected_uploaded[1]
        or sha256_file(stage_custody_path) != stage_custody_sha256
    ):
        raise HarnessError("local staged archives changed after custody sealing")
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
        "trusted_local_anchor_path": str(custody_path),
        "trusted_local_anchor_sha256": custody_sha256,
        "trusted_local_anchor_transported": False,
        "trusted_stage_custody_path": str(stage_custody_path),
        "trusted_stage_custody_sha256": stage_custody_sha256,
        "trusted_stage_custody_transported": False,
        "out_of_band_custody_record": {
            "schema_version": "paper2_bound_out_of_band_custody_record_v1",
            "record_type": "stage",
            "run_id": run_id,
            "receipt_path": str(stage_custody_path),
            "receipt_sha256": stage_custody_sha256,
            "retain_outside_run_tree": True,
        },
        "remote_archive_digests_verified": True,
        "scientific_job_submitted": False,
        "evidence": False,
        "evidence_status": "STAGED_NOT_SUBMITTED_NOT_EVIDENCE",
    }
    atomic_write_json(run_dir / "status" / "remote_stage.json", result_payload)
    return result_payload


def _remote_materialization_context(
    *,
    host: str,
    remote_python: str,
    remote_run: str,
    runner_relative: str,
    expected_portable_runtime_sha256: str,
    execution_nonce: str,
) -> dict[str, str]:
    """Ask the staged runtime for the exact paths receipts will record."""
    remote_run_q = _remote_shell_path(remote_run)
    probe = (
        f"{_remote_shell_path(remote_python)} -I -B -S -c "
        + shlex.quote(
            "import json,pathlib,sys; "
            "print(json.dumps({'python_executable':sys.executable,"
            "'repository_root':str(pathlib.Path(sys.argv[1]).resolve()),"
            "'run_directory':str(pathlib.Path(sys.argv[2]).resolve())},sort_keys=True))"
        )
        + f" {remote_run_q}/source {remote_run_q}/control"
    )
    result = run_capture(["ssh", "-o", "BatchMode=yes", host, probe])
    try:
        payload = json.loads(result.stdout)
    except (json.JSONDecodeError, TypeError) as exc:
        raise HarnessError("remote materialization context probe was malformed") from exc
    basic_context = _argv_materialization_context_from_remote(
        {
            **payload,
            "host_runtime_sha256": "0" * 64,
            "execution_nonce": execution_nonce,
        }
    )
    bootstrap = f"{remote_run_q}/source/scripts/paper2_isolated_bootstrap.py"
    runner = f"{remote_run_q}/source/{shlex.quote(runner_relative)}"
    sanitized_prefix = (
        'env -i HOME="$HOME" PATH=/usr/bin:/bin TMPDIR=/tmp LANG=C LC_ALL=C '
        "TZ=UTC OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
        "VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 SCRES_SCIENTIFIC_CHILD=1"
    )
    attest_command = (
        f"{sanitized_prefix} {_remote_shell_path(basic_context['python_executable'])} "
        f"-I -B -S {bootstrap} --repo-root {remote_run_q}/source "
        f"--runner {runner} --attest-only"
    )
    attested = run_capture(
        ["ssh", "-o", "BatchMode=yes", host, attest_command]
    )
    try:
        runtime = validate_runtime_attestation_payload(json.loads(attested.stdout))
    except (json.JSONDecodeError, TypeError) as exc:
        raise HarnessError("remote host-runtime attestation was malformed") from exc
    if runtime.get("portable_sha256") != expected_portable_runtime_sha256:
        raise HarnessError("remote portable runtime differs from preparation")
    return {
        **basic_context,
        "host_runtime_sha256": runtime["runtime_sha256"],
    }


def _argv_materialization_context_from_remote(payload: Any) -> dict[str, str]:
    """Validate a remote context without normalizing attacker-controlled values."""
    if not isinstance(payload, dict):
        raise HarnessError("remote materialization context is not an object")
    expected = {
        "python_executable",
        "repository_root",
        "run_directory",
        "host_runtime_sha256",
        "execution_nonce",
    }
    if set(payload) != expected or any(
        not isinstance(payload[key], str)
        or not payload[key].startswith("/")
        or not SAFE_REMOTE_RE.fullmatch(payload[key])
        for key in ("python_executable", "repository_root", "run_directory")
    ):
        raise HarnessError("remote materialization context is malformed or unsafe")
    if not _is_sha256(payload["host_runtime_sha256"]) or not _is_sha256(
        payload["execution_nonce"]
    ):
        raise HarnessError("remote runtime digest or execution nonce is malformed")
    return {key: payload[key] for key in sorted(expected)}


def launch_vps(*, run_dir: Path, host: str, remote_python: str) -> dict[str, Any]:
    if host != DEFAULT_HOST:
        raise HarnessError(f"host must be the approved SSH alias {DEFAULT_HOST!r}")
    _validate_remote_value(remote_python, "remote python")
    run_manifest = load_json(run_dir / "run_manifest.json")
    if (
        run_manifest.get("mode") not in EXECUTABLE_EVIDENCE_MODES
        or run_manifest.get("execution", {}).get("sealed_for_execution") is not True
    ):
        raise HarnessError(
            "VPS launch is reserved for a sealed evidence-execution manifest"
        )
    stage_path = _stage_custody_path(run_dir, str(run_manifest.get("run_id")))
    stage_sha256 = sha256_file(stage_path) if stage_path.is_file() else None
    stage, stage_failures = _load_external_receipt(
        stage_path,
        stage_sha256,
        run_dir=run_dir,
        schema=STAGE_CUSTODY_SCHEMA,
        body_digest_key="stage_body_sha256",
        label="trusted stage custody receipt",
    )
    if stage is None or stage_failures:
        raise HarnessError("trusted stage custody failed: " + "; ".join(stage_failures))
    remote = stage.get("remote", {})
    if remote.get("ssh_alias") != host or not isinstance(remote.get("remote_run"), str):
        raise HarnessError("trusted stage custody remote identity mismatch")
    remote_run = remote["remote_run"]
    _validate_remote_value(remote_run, "remote run path")
    harness_rel = run_manifest["inputs"]["harness_relative"]
    remote_run_q = _remote_shell_path(remote_run)
    materialization_context = _remote_materialization_context(
        host=host,
        remote_python=remote_python,
        remote_run=remote_run,
        runner_relative=run_manifest["inputs"]["runner_relative"],
        expected_portable_runtime_sha256=run_manifest["inputs"][
            "portable_runtime_sha256"
        ],
        execution_nonce=run_manifest["inputs"]["execution_nonce"],
    )
    resolved_python = materialization_context["python_executable"]
    launch_command = (
        f"cd {remote_run_q}/source && "
        f"mkdir -p {remote_run_q}/control/logs && "
        f"{{ nohup {_remote_shell_path(resolved_python)} -I -B {shlex.quote(harness_rel)} execute "
        f"--run-dir {remote_run_q}/control --repo-root {remote_run_q}/source "
        f"--expected-host-runtime-sha256 {materialization_context['host_runtime_sha256']} "
        f"--location vps > {remote_run_q}/control/logs/launcher.stdout.log "
        f"2> {remote_run_q}/control/logs/launcher.stderr.log < /dev/null & "
        f"echo $!; }}"
    )
    launch_custody_path, launch_custody_sha256 = (
        create_trusted_launch_custody_receipt(
            run_dir,
            stage_path=stage_path,
            stage_sha256=str(stage_sha256),
            materialization_context=materialization_context,
            remote_python_requested=remote_python,
            remote_launch_shell_command=launch_command,
        )
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
        "trusted_launch_custody_path": str(launch_custody_path),
        "trusted_launch_custody_sha256": launch_custody_sha256,
        "trusted_launch_custody_transported": False,
        "out_of_band_custody_record": {
            "schema_version": "paper2_bound_out_of_band_custody_record_v1",
            "record_type": "launch",
            "run_id": run_manifest.get("run_id"),
            "receipt_path": str(launch_custody_path),
            "receipt_sha256": launch_custody_sha256,
            "retain_outside_run_tree": True,
            "required_for_retrieval": True,
        },
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


def retrieve_vps(
    *,
    run_dir: Path,
    host: str,
    trusted_launch_receipt: Path | None = None,
    trusted_launch_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    if host != DEFAULT_HOST:
        raise HarnessError(f"host must be the approved SSH alias {DEFAULT_HOST!r}")
    if trusted_launch_receipt is None or not _is_sha256(
        trusted_launch_receipt_sha256
    ):
        raise HarnessError(
            "retrieval requires the caller-retained trusted launch receipt path and digest"
        )
    launch_path = trusted_launch_receipt.resolve()
    launch_sha256 = str(trusted_launch_receipt_sha256)
    launch, launch_failures = _load_external_receipt(
        launch_path,
        launch_sha256,
        run_dir=run_dir,
        schema=LAUNCH_CUSTODY_SCHEMA,
        body_digest_key="launch_body_sha256",
        label="trusted launch custody receipt",
    )
    if launch is None or launch_failures:
        raise HarnessError("trusted launch custody failed: " + "; ".join(launch_failures))
    remote = launch.get("remote", {})
    if remote.get("ssh_alias") != host or not isinstance(remote.get("remote_run"), str):
        raise HarnessError("trusted launch custody remote identity mismatch")
    remote_run = remote["remote_run"]
    _validate_remote_value(remote_run, "remote run path")
    retrieved = run_dir / "retrieved"
    if retrieved.exists():
        raise HarnessError(f"retrieval destination already exists: {retrieved}")
    retrieved.mkdir(parents=True)
    run_capture(["rsync", "-a", f"{host}:{remote_run}/control/", f"{retrieved}/"])
    verification = verify_artifacts(
        retrieved,
        retrieved=True,
        trusted_launch_receipt=launch_path,
        trusted_launch_receipt_sha256=launch_sha256,
    )
    payload = {
        "schema_version": SCHEMA,
        "retrieved_at_utc": utc_now(),
        "ssh_alias": host,
        "remote_run": remote_run,
        "retrieval_destination": str(retrieved.resolve()),
        "verification_passed": verification["checks_passed"],
        "verification_sha256": sha256_file(retrieved / "retrieval_verification.json"),
        "trusted_launch_custody_path": str(launch_path),
        "trusted_launch_custody_sha256": launch_sha256,
        "caller_retained_launch_digest_required": True,
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
    execute.add_argument("--expected-host-runtime-sha256")

    signed_reduced = sub.add_parser("signed-reduced-launch")
    signed_reduced.add_argument("--run-dir", type=Path, required=True)
    signed_reduced.add_argument(
        "--execution-role",
        choices=("producer", "independent_replay"),
        required=True,
    )
    signed_reduced.add_argument("--replay-pair-id", required=True)
    signed_reduced.add_argument("--timeout-seconds", type=float)

    verify = sub.add_parser("verify")
    verify.add_argument("--run-dir", type=Path, required=True)
    verify.add_argument("--retrieved", action="store_true")
    verify.add_argument("--trusted-local-anchor", type=Path)
    verify.add_argument("--trusted-local-anchor-sha256")
    verify.add_argument("--trusted-launch-receipt", type=Path)
    verify.add_argument("--trusted-launch-receipt-sha256")

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
    retrieve.add_argument("--trusted-launch-receipt", type=Path, required=True)
    retrieve.add_argument(
        "--trusted-launch-receipt-sha256", required=True
    )

    pair = sub.add_parser("verify-reduced-pair")
    pair.add_argument("--archive-root", type=Path, required=True)
    pair.add_argument("--producer-run-dir", type=Path, required=True)
    pair.add_argument("--independent-run-dir", type=Path, required=True)
    pair.add_argument("--role", required=True)
    pair.add_argument("--producer-output-relative", required=True)
    pair.add_argument("--independent-output-relative", required=True)
    pair.add_argument("--producer-output-sha256", required=True)
    pair.add_argument("--independent-output-sha256", required=True)
    pair.add_argument("--producer-receipt-relative", required=True)
    pair.add_argument("--independent-receipt-relative", required=True)
    pair.add_argument("--producer-receipt-sha256", required=True)
    pair.add_argument("--independent-receipt-sha256", required=True)
    pair.add_argument("--producer-authorization-relative", required=True)
    pair.add_argument("--independent-authorization-relative", required=True)
    pair.add_argument("--producer-authorization-sha256", required=True)
    pair.add_argument("--independent-authorization-sha256", required=True)
    pair.add_argument("--producer-public-key-fingerprint", required=True)
    pair.add_argument("--independent-public-key-fingerprint", required=True)
    pair.add_argument("--producer-runtime-attestation-relative", required=True)
    pair.add_argument("--independent-runtime-attestation-relative", required=True)
    pair.add_argument("--producer-runtime-attestation-sha256", required=True)
    pair.add_argument("--independent-runtime-attestation-sha256", required=True)
    pair.add_argument("--expected-environment-sha256")
    pair.add_argument("--output", type=Path, required=True)
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
                expected_host_runtime_sha256=args.expected_host_runtime_sha256,
            )
        elif args.command == "signed-reduced-launch":
            result = execute_prepared_reduced_signed_session(
                run_dir=args.run_dir,
                execution_role=args.execution_role,
                replay_pair_id=args.replay_pair_id,
                acknowledgement_callback=_interactive_prelaunch_acknowledgement,
                timeout_seconds=args.timeout_seconds,
            )
        elif args.command == "verify":
            result = verify_artifacts(
                args.run_dir.resolve(),
                retrieved=args.retrieved,
                trusted_local_anchor=(
                    args.trusted_local_anchor.resolve()
                    if args.trusted_local_anchor
                    else None
                ),
                trusted_local_anchor_sha256=args.trusted_local_anchor_sha256,
                trusted_launch_receipt=(
                    args.trusted_launch_receipt.resolve()
                    if args.trusted_launch_receipt
                    else None
                ),
                trusted_launch_receipt_sha256=(
                    args.trusted_launch_receipt_sha256
                ),
            )
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
            result = retrieve_vps(
                run_dir=args.run_dir.resolve(),
                host=args.host,
                trusted_launch_receipt=args.trusted_launch_receipt.resolve(),
                trusted_launch_receipt_sha256=(
                    args.trusted_launch_receipt_sha256
                ),
            )
        elif args.command == "verify-reduced-pair":
            result = verify_reduced_evidence_pair(
                archive_root=args.archive_root,
                producer_run_dir=args.producer_run_dir,
                independent_run_dir=args.independent_run_dir,
                role=args.role,
                producer_output_relative=args.producer_output_relative,
                independent_output_relative=args.independent_output_relative,
                expected_producer_output_sha256=args.producer_output_sha256,
                expected_independent_output_sha256=args.independent_output_sha256,
                producer_receipt_relative=args.producer_receipt_relative,
                independent_receipt_relative=args.independent_receipt_relative,
                expected_producer_receipt_sha256=args.producer_receipt_sha256,
                expected_independent_receipt_sha256=args.independent_receipt_sha256,
                producer_authorization_relative=args.producer_authorization_relative,
                independent_authorization_relative=args.independent_authorization_relative,
                expected_producer_authorization_sha256=(
                    args.producer_authorization_sha256
                ),
                expected_independent_authorization_sha256=(
                    args.independent_authorization_sha256
                ),
                expected_producer_public_key_fingerprint=(
                    args.producer_public_key_fingerprint
                ),
                expected_independent_public_key_fingerprint=(
                    args.independent_public_key_fingerprint
                ),
                producer_runtime_attestation_relative=(
                    args.producer_runtime_attestation_relative
                ),
                independent_runtime_attestation_relative=(
                    args.independent_runtime_attestation_relative
                ),
                expected_producer_runtime_attestation_sha256=(
                    args.producer_runtime_attestation_sha256
                ),
                expected_independent_runtime_attestation_sha256=(
                    args.independent_runtime_attestation_sha256
                ),
                output_path=args.output,
                expected_environment_sha256=args.expected_environment_sha256,
            )
            if not result["passed"]:
                print(json.dumps(result, indent=2, sort_keys=True))
                return 1
        else:  # pragma: no cover
            raise HarnessError(f"unsupported command {args.command}")
    except HarnessError as exc:
        print(f"FAIL_CLOSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
