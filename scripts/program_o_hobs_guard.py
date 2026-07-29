"""Immutable freeze verification for Program O's H_obs fit stage."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parent.parent


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def tracked_bytes(path: Path, commit: str = "HEAD") -> bytes:
    relative = path.resolve().relative_to(ROOT).as_posix()
    return subprocess.check_output(
        ["git", "show", f"{commit}:{relative}"], cwd=ROOT
    )


def verify_hobs_fit_freeze(
    *,
    freeze_path: Path,
    contract_path: Path,
    run_id: str,
    run_dir: Path,
    seed_range: Sequence[int],
    expected_commit: str,
) -> dict[str, Any]:
    freeze_path = freeze_path.resolve()
    contract_path = contract_path.resolve()
    freeze = json.loads(freeze_path.read_text())
    commit = git_commit()
    failures = []
    if commit != str(expected_commit):
        failures.append("scientific commit")
    try:
        if tracked_bytes(freeze_path) != freeze_path.read_bytes():
            failures.append("freeze differs from tracked HEAD blob")
    except (subprocess.CalledProcessError, ValueError):
        failures.append("freeze is not tracked at HEAD")
    if freeze.get("status") != "AUTHORIZED_PROGRAM_O_HOBS_FIT_ONLY":
        failures.append("freeze status")
    authorization = freeze.get("execution_authorization", {})
    if authorization.get("stage") != "fit":
        failures.append("stage")
    if authorization.get("run_id") != str(run_id):
        failures.append("run id")
    if Path(os.path.abspath(str(authorization.get("run_dir", "")))) != Path(
        os.path.abspath(str(run_dir))
    ):
        failures.append("run dir")
    if authorization.get("seed_range") != list(map(int, seed_range)):
        failures.append("seed range")
    if freeze.get("contract_sha256") != sha256(contract_path):
        failures.append("contract hash")
    for relative, expected_hash in freeze.get("source_hashes", {}).items():
        source = ROOT / relative
        if not source.is_file() or sha256(source) != str(expected_hash):
            failures.append(f"source hash: {relative}")
    if failures:
        raise RuntimeError("H_obs fit freeze verification failed: " + "; ".join(failures))
    return {
        "scientific_commit": commit,
        "freeze_sha256": sha256(freeze_path),
        "run_id": str(run_id),
        "run_dir": str(Path(os.path.abspath(str(run_dir)))),
        "stage": "fit",
        "seed_range": list(map(int, seed_range)),
    }
