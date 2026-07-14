#!/usr/bin/env python3
"""Fail closed on exact calibration ties at the <=4-switch boundary.

The frozen producer selects the minimum-index exact maximizer.  That tie break
is deterministic, but an interior selected calendar does not imply an interior
frontier if another exact maximizer uses four switches.  This post-result audit
derives every exact maximizer from the content-addressed candidate rows and
binds the interpretation to passed custody and independent deep replay.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
SCHEMA = "paper2_switch4_exact_tie_audit_v1"


def load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def audit_exact_ties(
    result: dict[str, Any],
    *,
    boundary_switch_count: int = 4,
) -> dict[str, Any]:
    failures: list[str] = []
    rows = result.get("candidate_family", {}).get("rows")
    calibration = result.get("calibration", {})
    if not isinstance(rows, list) or not rows:
        return {
            "passed": False,
            "failures": ["candidate rows missing or empty"],
            "primary_tie_indices": [],
            "primary_tie_switch_counts": [],
            "tie_spans_boundary": None,
        }

    indices = [row.get("candidate_index") for row in rows]
    if indices != list(range(len(rows))):
        failures.append("candidate indices are not unique contiguous producer order")

    exact_values: list[Fraction] = []
    for index, row in enumerate(rows):
        try:
            exact_values.append(
                Fraction(
                    int(row["exact_sum_numerator"]),
                    int(row["exact_sum_denominator"]),
                )
            )
            int(row["switch_count"])
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            failures.append(f"invalid exact candidate row at {index}")
            break
    if failures:
        return {
            "passed": False,
            "failures": sorted(set(failures)),
            "primary_tie_indices": [],
            "primary_tie_switch_counts": [],
            "tie_spans_boundary": None,
        }

    best = max(exact_values)
    tie_indices = [index for index, value in enumerate(exact_values) if value == best]
    tie_switch_counts = [int(rows[index]["switch_count"]) for index in tie_indices]
    selected_index = calibration.get("selected_index")
    expected_selected = min(tie_indices)
    if selected_index != expected_selected:
        failures.append("selected index is not the minimum-index exact maximizer")
    if calibration.get("primary_tie_count") != len(tie_indices):
        failures.append("producer primary_tie_count mismatch")
    if str(calibration.get("selected_exact_sum_numerator")) != str(best.numerator):
        failures.append("selected exact numerator mismatch")
    if str(calibration.get("selected_exact_sum_denominator")) != str(best.denominator):
        failures.append("selected exact denominator mismatch")

    tie_spans_boundary = boundary_switch_count in tie_switch_counts
    expected_status = (
        "CALIBRATION_SWITCH4_BOUNDARY_ACTIVE"
        if tie_spans_boundary
        else "CALIBRATION_SWITCH4_INTERIOR"
    )
    source_status_consistent = (
        result.get("scientific_status") == expected_status
        and result.get("boundary_hit") is tie_spans_boundary
    )
    if not source_status_consistent:
        failures.append("producer status ignores an exact boundary-spanning tie")

    return {
        "passed": not failures,
        "failures": sorted(set(failures)),
        "candidate_count": len(rows),
        "boundary_switch_count": boundary_switch_count,
        "primary_tie_count": len(tie_indices),
        "primary_tie_indices": tie_indices,
        "primary_tie_switch_counts": tie_switch_counts,
        "tie_spans_boundary": tie_spans_boundary,
        "selected_index": selected_index,
        "selected_switch_count": calibration.get("selected_switch_count"),
        "exact_maximum_numerator": str(best.numerator),
        "exact_maximum_denominator": str(best.denominator),
        "source_scientific_status": result.get("scientific_status"),
        "source_boundary_hit": result.get("boundary_hit"),
        "source_status_consistent_with_all_exact_ties": source_status_consistent,
        "effective_scientific_status": expected_status,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--custody-audit", type=Path, required=True)
    parser.add_argument("--deep-verification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result_path = args.result.resolve(strict=True)
    custody_path = args.custody_audit.resolve(strict=True)
    deep_path = args.deep_verification.resolve(strict=True)
    result = load(result_path)
    custody = load(custody_path)
    deep = load(deep_path)
    result_sha = sha256(result_path)

    tie_audit = audit_exact_ties(result)
    upstream_checks = {
        "custody_passed": custody.get("passed") is True,
        "custody_result_hash_bound": custody.get("result_sha256") == result_sha,
        "deep_verification_passed": deep.get("passed") is True,
        "deep_result_hash_bound": deep.get("result_sha256") == result_sha,
        "deep_candidate_rollouts": deep.get("deep_replay", {}).get(
            "candidate_rollouts_replayed"
        ) == 5_347_860,
        "deep_selected_ledger_rollouts": deep.get("deep_replay", {}).get(
            "selected_ledger_rollouts_replayed"
        ) == 60,
    }
    passed = all(upstream_checks.values()) and tie_audit["passed"] is True
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    payload = {
        "schema_version": SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "verification_git_head": head,
        "result_path": str(result_path),
        "result_sha256": result_sha,
        "custody_audit_path": str(custody_path),
        "custody_audit_sha256": sha256(custody_path),
        "deep_verification_path": str(deep_path),
        "deep_verification_sha256": sha256(deep_path),
        "upstream_checks": upstream_checks,
        "tie_audit": tie_audit,
        "passed": passed,
        "claim_limit": "Calibration comparator development only; even an exact interior <=4 result is not a full-horizon H_PI or H_obs result.",
    }
    payload["content_sha256"] = json_sha256(payload)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite audit: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
