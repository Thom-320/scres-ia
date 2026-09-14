#!/usr/bin/env python3
"""Fail-closed validator for the prospective Paper B metric gate."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

DEFAULT_CONTRACT = Path("contracts/paper_b_metric_gate_v1.json")


def unresolved_decisions(contract: dict[str, object]) -> list[str]:
    decisions = contract.get("blocking_decisions")
    if not isinstance(decisions, dict):
        raise ValueError("blocking_decisions must be an object")
    return sorted(key for key, value in decisions.items() if value is None)


def validate_contract(contract: dict[str, object]) -> list[str]:
    errors: list[str] = []
    if contract.get("scientific_execution_authorized") is not False:
        errors.append("scientific_execution_authorized must remain false before PASS")
    if contract.get("fresh_roots_opened") is not False:
        errors.append("fresh_roots_opened must be false")
    if contract.get("fresh_tapes_opened") is not False:
        errors.append("fresh_tapes_opened must be false")
    status = contract.get("status")
    pass_status = contract.get("pass_status")
    if status not in {"DRAFT_UNRESOLVED_METRIC_AUTHORITY", pass_status}:
        errors.append("unexpected metric-gate status")
    unresolved = unresolved_decisions(contract)
    if status == "DRAFT_UNRESOLVED_METRIC_AUTHORITY" and not unresolved:
        errors.append("draft metric gate unexpectedly has no unresolved decisions")
    if status == pass_status and unresolved:
        errors.append("passed metric gate still has unresolved decisions")
    source_evidence = contract.get("source_evidence")
    if not isinstance(source_evidence, dict) or not source_evidence:
        errors.append("source_evidence must be a non-empty object")
    elif any(
        not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
        for value in source_evidence.values()
    ):
        errors.append("every source evidence entry must be a sha256 digest")
    return errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument(
        "--require-pass",
        action="store_true",
        help="Return nonzero while any blocking decision remains unresolved.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    errors = validate_contract(contract)
    unresolved = unresolved_decisions(contract)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 2
    print(f"metric_gate_status={contract['status']}")
    print(f"unresolved_decision_count={len(unresolved)}")
    for decision in unresolved:
        print(f"UNRESOLVED: {decision}")
    if args.require_pass and (
        unresolved or contract.get("status") != contract.get("pass_status")
    ):
        print(contract["stop_status"])
        return 3
    print("PASS_DRAFT_METRIC_GATE_INTEGRITY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
