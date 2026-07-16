#!/usr/bin/env python3
"""Verify the content-addressed wartime GSA execution preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_war_stress_policy_manifest import build_manifest  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(repo_root: Path) -> dict[str, Any]:
    contract_path = repo_root / "contracts/war_stress_gsa_execution_preflight_v1.json"
    contract = json.loads(contract_path.read_text())
    failures: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    overlay = contract["gsa_overlay"]
    require(
        _sha256(repo_root / overlay["path"]) == overlay["sha256"],
        "GSA overlay hash mismatch",
    )
    require(
        _sha256(repo_root / overlay["configuration_manifest_path"])
        == overlay["configuration_manifest_sha256"],
        "GSA configuration manifest hash mismatch",
    )
    policy = contract["policy_family"]
    policy_path = repo_root / policy["manifest_path"]
    require(_sha256(policy_path) == policy["manifest_sha256"], "policy manifest hash mismatch")
    regenerated = build_manifest()
    committed = json.loads(policy_path.read_text())
    require(regenerated == committed, "policy manifest is not exactly reproducible")
    require(
        committed["ordered_template_rows_sha256"]
        == policy["ordered_template_rows_sha256"],
        "ordered policy row hash mismatch",
    )
    for section in contract["executor"].values():
        path = repo_root / section["path"]
        require(path.is_file(), f"missing executor artifact {path}")
        if path.is_file():
            require(_sha256(path) == section["sha256"], f"hash mismatch for {path}")
    benchmark = contract["benchmark"]
    benchmark_path = repo_root / benchmark["receipt_path"]
    require(
        _sha256(benchmark_path) == benchmark["receipt_sha256"],
        "benchmark receipt hash mismatch",
    )
    receipt = json.loads(benchmark_path.read_text())
    require(receipt["scientific_seeds_opened"] is False, "benchmark opened scientific seeds")
    require(receipt["risk_tape"]["r3_event_count"] == 0, "R3 entered primary benchmark")
    require(receipt["runtime"]["compute_gate_pass"] is False, "compute STOP disappeared")
    authorization = contract["authorization"]
    require(authorization["open_7470001_through_7470012"] is False, "seeds authorized")
    require(authorization["run_Morris"] is False, "Morris authorized")
    require(authorization["train_learner"] is False, "learner authorized")
    return {
        "schema_version": "war_stress_gsa_execution_preflight_verification_v1",
        "status": (
            "PASS_EXECUTOR_PREFLIGHT_SCIENTIFIC_BLOCKED"
            if not failures
            else "FAIL_EXECUTOR_PREFLIGHT"
        ),
        "failures": failures,
        "contract_sha256": _sha256(contract_path),
        "policy_templates": committed["total_policy_templates"],
        "projected_scientific_episodes": benchmark["projected_scientific_episodes"],
        "scientific_seeds_opened": False,
        "scientific_execution_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "research/paper2_exhaustive_search/war_stress_gsa_execution_preflight_verification_20260716.json"
        ),
    )
    args = parser.parse_args()
    result = verify(args.repo_root.resolve())
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not result["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
