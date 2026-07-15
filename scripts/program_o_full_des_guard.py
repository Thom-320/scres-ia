"""Fail-closed authorization and one-use seed custody for Program O."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parent.parent


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _tracked_bytes(path: Path) -> bytes:
    relative = path.resolve().relative_to(ROOT).as_posix()
    return subprocess.check_output(["git", "show", f"HEAD:{relative}"], cwd=ROOT)


def verify_tracked_freeze(
    *,
    freeze_path: Path,
    contract_path: Path,
    stage: str,
    run_id: str,
    run_dir: Path,
    seed_range: Sequence[int],
    expected_commit: str | None = None,
) -> dict[str, Any]:
    """Verify that HEAD contains the exact freeze and all bound source blobs."""
    freeze_path = freeze_path.resolve()
    contract_path = contract_path.resolve()
    freeze = json.loads(freeze_path.read_text())
    failures: list[str] = []
    commit = git_commit()
    if expected_commit is not None and commit != str(expected_commit):
        failures.append("scientific commit")
    try:
        if _tracked_bytes(freeze_path) != freeze_path.read_bytes():
            failures.append("freeze differs from tracked HEAD blob")
    except (subprocess.CalledProcessError, ValueError):
        failures.append("freeze is not tracked in HEAD")
    expected_status = {
        "development": "AUTHORIZED_PROGRAM_O_FULL_DES_HPI_DEVELOPMENT_ONLY",
        "validation": "AUTHORIZED_PROGRAM_O_FULL_DES_VALIDATION",
    }[str(stage)]
    if freeze.get("status") != expected_status:
        failures.append("freeze status")
    execution = freeze.get("execution_authorization", {})
    if execution.get("stage") != str(stage):
        failures.append("authorized stage")
    if execution.get("run_id") != str(run_id):
        failures.append("authorized run id")
    if Path(str(execution.get("run_dir", ""))) != run_dir.resolve():
        failures.append("authorized run dir")
    if execution.get("seed_range") != list(map(int, seed_range)):
        failures.append("authorized seed range")
    if freeze.get("contract", {}).get("sha256") != sha256(contract_path):
        failures.append("contract hash")
    for relative, expected_hash in freeze.get("source_hashes", {}).items():
        source = ROOT / relative
        if not source.is_file() or sha256(source) != str(expected_hash):
            failures.append(f"source hash: {relative}")
    if failures:
        raise RuntimeError("freeze verification failed: " + "; ".join(failures))
    return {
        "scientific_commit": commit,
        "freeze_sha256": sha256(freeze_path),
        "freeze_path": str(freeze_path),
        "run_id": str(run_id),
        "run_dir": str(run_dir.resolve()),
        "stage": str(stage),
        "seed_range": list(map(int, seed_range)),
    }


def seed_claim_path(claim_root: Path, *, stage: str, seed_range: Sequence[int]) -> Path:
    start, end = map(int, seed_range)
    return claim_root.resolve() / f"program_o_{stage}_{start}_{end}.json"


def create_seed_claim(
    *,
    claim_root: Path,
    authorization: Mapping[str, Any],
    contract_sha256: str,
) -> Path:
    """Claim a seed block once on the execution host using O_EXCL."""
    claim_root = claim_root.resolve()
    claim_root.mkdir(parents=True, exist_ok=True)
    destination = seed_claim_path(
        claim_root,
        stage=str(authorization["stage"]),
        seed_range=authorization["seed_range"],
    )
    payload = {
        "schema_version": "program_o_seed_claim_v1",
        "claimed_at_utc": now_utc(),
        **dict(authorization),
        "contract_sha256": str(contract_sha256),
    }
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(destination, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return destination


def verify_seed_claim(
    *,
    claim_path: Path,
    authorization: Mapping[str, Any],
    contract_sha256: str,
) -> dict[str, Any]:
    claim_path = claim_path.resolve()
    claim = json.loads(claim_path.read_text())
    failures = []
    for key in (
        "scientific_commit",
        "freeze_sha256",
        "run_id",
        "run_dir",
        "stage",
        "seed_range",
    ):
        if claim.get(key) != authorization.get(key):
            failures.append(key)
    if claim.get("contract_sha256") != str(contract_sha256):
        failures.append("contract_sha256")
    expected_path = seed_claim_path(
        claim_path.parent,
        stage=str(authorization["stage"]),
        seed_range=authorization["seed_range"],
    )
    if claim_path != expected_path:
        failures.append("claim path")
    if failures:
        raise RuntimeError("seed claim verification failed: " + ", ".join(failures))
    return claim
