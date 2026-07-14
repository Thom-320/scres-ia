#!/usr/bin/env python3
"""Verify that neither Paper-2 return path is declared without all deliverables."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.verify_paper2_exhaustion import validate_boundary_family_proof_ledger


ROOT = Path(__file__).resolve().parent.parent
SEARCH = ROOT / "research" / "paper2_exhaustive_search"
SCHEMA = "paper2_terminal_return_verification_v1"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_readiness(readiness: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    rows = readiness.get("required_outputs", [])
    ids = [row.get("id") for row in rows]
    if ids != list(range(1, 14)):
        failures.append("required outputs must be exactly ordered ids 1 through 13")
    hashes_checked = 0
    for row in rows:
        artifacts = row.get("artifacts")
        if not artifacts:
            failures.append(f"output {row.get('id')} has no artifact routing")
            continue
        if row.get("terminal_ready") is True:
            for artifact in artifacts:
                if not isinstance(artifact, dict):
                    failures.append(
                        f"terminal-ready output {row.get('id')} has unhashed artifact"
                    )
                    continue
                path = ROOT / artifact.get("path", "")
                expected = artifact.get("sha256")
                if not path.is_file():
                    failures.append(f"missing artifact: {artifact.get('path')}")
                elif not expected or sha256(path) != expected:
                    failures.append(f"artifact hash mismatch: {artifact.get('path')}")
                else:
                    hashes_checked += 1
    terminal_ids = [row["id"] for row in rows if row.get("terminal_ready") is True]
    nonterminal_ids = [row["id"] for row in rows if row.get("terminal_ready") is not True]
    summary = readiness.get("summary", {})
    if (
        summary.get("required_output_count") != len(rows)
        or summary.get("terminal_ready_count") != len(terminal_ids)
        or summary.get("nonterminal_output_ids") != nonterminal_ids
    ):
        failures.append("readiness summary mismatch")
    return {
        "passed": not failures,
        "failures": failures,
        "hashes_checked": hashes_checked,
        "terminal_ready_ids": terminal_ids,
        "nonterminal_ids": nonterminal_ids,
        "all_outputs_terminal_ready": len(rows) == 13 and not nonterminal_ids,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=SEARCH / "terminal_return_verification.json",
    )
    args = parser.parse_args()
    readiness = load(SEARCH / "terminal_return_readiness.json")
    registry = load(SEARCH / "approach_registry.json")
    proof_ledger = load(SEARCH / "boundary_family_proof_ledger.json")
    readiness_validation = validate_readiness(readiness)
    family_validation = validate_boundary_family_proof_ledger(
        registry, proof_ledger, root=ROOT
    )

    return_a = (
        readiness_validation["passed"]
        and readiness_validation["all_outputs_terminal_ready"]
        and readiness.get("paper2_confirmed") is True
        and readiness.get("paper3_authorized") is True
        and readiness.get("return_a_supported") is True
    )
    return_b = (
        readiness_validation["passed"]
        and readiness_validation["all_outputs_terminal_ready"]
        and family_validation["all_families_terminal_b_eligible"]
        and readiness.get("return_b_supported") is True
    )
    declarations_consistent = (
        readiness.get("return_a_supported") is return_a
        and readiness.get("return_b_supported") is return_b
    )
    result = {
        "schema_version": SCHEMA,
        "status": (
            "TERMINAL_RETURN_A_VERIFIED" if return_a
            else "TERMINAL_RETURN_B_VERIFIED" if return_b
            else "NONTERMINAL_REQUIRED_OUTPUTS_OR_FAMILY_PROOFS_INCOMPLETE"
        ),
        "return_a_verified": return_a,
        "return_b_verified": return_b,
        "declarations_consistent": declarations_consistent,
        "readiness_validation": readiness_validation,
        "family_proof_validation": family_validation,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if readiness_validation["passed"] and declarations_consistent else 1


if __name__ == "__main__":
    raise SystemExit(main())
