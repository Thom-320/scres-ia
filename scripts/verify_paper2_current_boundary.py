#!/usr/bin/env python3
"""Verify the current Paper-2 boundary certificate against live artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def verify(certificate_path: Path, repo_root: Path) -> dict[str, Any]:
    certificate = _load(certificate_path)
    failures: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    def artifact(section: dict[str, Any], path_key: str, hash_key: str) -> Path:
        path = repo_root / str(section[path_key])
        require(path.is_file(), f"missing artifact: {path}")
        if path.is_file():
            require(_sha256(path) == section[hash_key], f"hash mismatch: {path}")
        return path

    require(
        certificate.get("status")
        == "CURRENT_IMPLEMENTED_PORTFOLIO_EXHAUSTED_NO_LEARNER_AUTHORIZED",
        "unexpected certificate status",
    )
    require(not bool(certificate.get("global_impossibility_claimed")), "global overclaim")

    program_o = certificate["program_o"]
    program_o_path = artifact(program_o, "audit_path", "audit_sha256")
    if program_o_path.is_file():
        program_o_audit = _load(program_o_path)
        require(
            program_o_audit.get("audited_status")
            == "STOP_PROGRAM_O_AFTER_CORRECTIVE_VALIDATION",
            "Program O terminal status mismatch",
        )
        boundary = program_o_audit.get("claim_boundary", {})
        require(not bool(boundary.get("safe_joint_h_obs_contract_confirmed")), "Program O safe H_obs overclaim")
        require(bool(boundary.get("program_o_closed")), "Program O not closed")
        require(bool(boundary.get("second_rescue_forbidden")), "Program O rescue prohibition missing")

    risk = certificate["garrido_risk_sensitivity"]
    result_path = artifact(risk, "result_path", "result_sha256")
    audit_path = artifact(risk, "audit_path", "audit_sha256")
    artifact(risk, "retrieved_package_path", "retrieved_package_sha256")
    if result_path.is_file() and audit_path.is_file():
        result = _load(result_path)
        audit = _load(audit_path)
        summaries = [
            row
            for rows in result.get("group_budget_summaries", {}).values()
            for row in rows
        ]
        require(result.get("status") == "DEVELOPMENT_NO_DOOR_UNDER_TESTED_FRONTIER", "risk result status mismatch")
        require(result.get("passing_doors") == [], "risk result contains a passing door")
        require(not bool(result.get("black_swan_R3_scaled")), "R3 scaled in risk result")
        require(audit.get("status") == "PASS_GARRIDO_RISK_AUDIT", "independent risk audit failed")
        require(audit.get("failures") == [], "independent risk audit has failures")
        require(int(audit.get("n_rows", -1)) == 4860, "risk row count mismatch")
        raw_max = max(float(row["H_profile_raw"]) for row in summaries)
        safe_max = max(float(row["H_profile_safe"]) for row in summaries)
        require(math.isclose(raw_max, float(risk["maximum_H_profile_raw"]), abs_tol=1e-15), "raw maximum mismatch")
        require(math.isclose(safe_max, float(risk["maximum_H_profile_safe"]), abs_tol=1e-15), "safe maximum mismatch")
        require(sum(bool(row["door_pass"]) for row in summaries) == 0, "recomputed passing door count is nonzero")

    timing = certificate["restricted_timing"]
    timing_path = artifact(timing, "custody_path", "custody_sha256")
    if timing_path.is_file():
        custody = _load(timing_path)
        require(
            custody.get("status") == "CLOSED_UNOPENED_AFTER_GARRIDO_RISK_GATE_FAIL",
            "timing custody status mismatch",
        )
        require(not bool(custody.get("opened")), "timing seeds were opened")
        require(bool(custody.get("closed_without_access")), "timing closure not attested")

    questions = certificate["external_domain_questions"]
    question_path = artifact(questions, "json_path", "json_sha256")
    artifact(questions, "document_path", "document_sha256")
    if question_path.is_file():
        payload = _load(question_path)
        ids = [str(row["id"]) for row in payload.get("questions", [])]
        require(ids == list(questions["question_ids"]), "domain question IDs mismatch")

    authorization = certificate["authorization"]
    for key in (
        "classical_safe_H_obs_confirmed",
        "learner_authorized",
        "paper2_confirmed",
        "paper3_authorized",
    ):
        require(not bool(authorization.get(key)), f"unauthorized positive flag: {key}")

    return {
        "schema_version": "paper2_current_boundary_verification_v1",
        "status": "PASS_CURRENT_BOUNDARY_CERTIFICATE" if not failures else "FAIL_CURRENT_BOUNDARY_CERTIFICATE",
        "certificate_sha256": _sha256(certificate_path),
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--certificate",
        type=Path,
        default=Path("research/paper2_exhaustive_search/paper2_current_boundary_certificate_20260716.json"),
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    verdict = verify(args.certificate, args.repo_root)
    rendered = json.dumps(verdict, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if verdict["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
