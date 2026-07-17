#!/usr/bin/env python3
"""Fail-closed Program O-R calibration adjudication; never self-authorizes confirmation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from supply_chain.program_o_eval_custody import (  # noqa: E402
    sha256,
    verify_sha256_manifest,
)
from supply_chain.program_o_ret_freeze import verify_execution_freeze  # noqa: E402


CONTRACT = ROOT / "contracts/program_o_ret_only_learner_v1.json"


def adjudicate(calibration_path: Path, direct_audit_path: Path) -> dict[str, object]:
    """Validate the complete calibration chain and return a non-authorizing verdict."""
    calibration_root = calibration_path.parent
    evaluation_manifest = verify_sha256_manifest(
        calibration_root, calibration_root / "evaluation_files.sha256"
    )
    if "result.json" not in evaluation_manifest:
        raise ValueError("result.json absent from evaluation manifest")
    audit_manifest = verify_sha256_manifest(
        direct_audit_path.parent, direct_audit_path.parent / "audit_files.sha256"
    )
    if direct_audit_path.name not in audit_manifest:
        raise ValueError("direct audit absent from audit manifest")

    calibration = json.loads(calibration_path.read_text())
    direct_audit = json.loads(direct_audit_path.read_text())
    calibration_sha = sha256(calibration_path)
    direct_audit_sha = sha256(direct_audit_path)
    checks = {
        "calibration_phase": calibration.get("phase") == "calibration",
        "provisional_primary_pass": calibration.get("provisional_primary_pass") is True,
        "all_amendment_gates": bool(calibration.get("amendment_gates"))
        and all(calibration["amendment_gates"].values()),
        "direct_full_des_pass": direct_audit.get("passed") is True,
        "direct_full_des_phase": direct_audit.get("phase") == "calibration",
        "direct_audit_binds_result": (
            direct_audit.get("evaluation_result_sha256") == calibration_sha
        ),
        "evaluation_manifest_verified": True,
        "audit_manifest_verified": True,
    }
    eligible = all(checks.values())
    return {
        "schema_version": "program_o_ret_only_calibration_adjudication_v1_2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": (
            "ELIGIBLE_FOR_INDEPENDENT_AUTHORIZATION"
            if eligible
            else "STOP_CALIBRATION_NOT_ELIGIBLE"
        ),
        "checks": checks,
        "calibration_result": str(calibration_path),
        "calibration_result_sha256": calibration_sha,
        "direct_audit": str(direct_audit_path),
        "direct_audit_sha256": direct_audit_sha,
        "contract_sha256": sha256(CONTRACT),
        "independent_authorization_created": False,
        "claim_boundary": (
            "Eligibility is not independent authorization. Confirmation remains closed until "
            "a separate auditor signs all bound hashes."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-result", type=Path, required=True)
    parser.add_argument("--direct-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    verify_execution_freeze(ROOT, CONTRACT)
    verdict = adjudicate(args.calibration_result, args.direct_audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n")
    return 0 if verdict["status"] == "ELIGIBLE_FOR_INDEPENDENT_AUTHORIZATION" else 1


if __name__ == "__main__":
    raise SystemExit(main())
