"""Fail-closed source and genealogy checks for Program O-R execution."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


FREEZE_RELATIVE_PATH = Path(
    "research/paper2_exhaustive_search/program_o_ret_only_learner_execution_freeze_20260717.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_execution_freeze(root: Path, contract_path: Path) -> dict[str, Any]:
    freeze_path = root / FREEZE_RELATIVE_PATH
    if not freeze_path.exists():
        raise RuntimeError(f"missing execution freeze: {freeze_path}")
    freeze = json.loads(freeze_path.read_text())
    if freeze.get("status") != "FROZEN_BEFORE_748_SCIENTIFIC_SEEDS":
        raise RuntimeError("execution freeze is not authorized")
    if sha256(contract_path) != freeze.get("contract_sha256"):
        raise RuntimeError("contract hash differs from execution freeze")
    contract = json.loads(contract_path.read_text())
    expected = contract.get("source_hashes", {})
    if not expected or expected != freeze.get("source_hashes"):
        raise RuntimeError("source hash manifest missing or inconsistent")
    for relative, digest in expected.items():
        path = root / relative
        if not path.is_file() or sha256(path) != digest:
            raise RuntimeError(f"frozen source mismatch: {relative}")
    scientific_commit = str(freeze["scientific_source_commit"])
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", scientific_commit, "HEAD"],
        cwd=root,
        check=False,
    )
    if ancestry.returncode != 0:
        raise RuntimeError("frozen scientific source commit is not an ancestor of HEAD")
    return freeze
