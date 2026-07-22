#!/usr/bin/env python3
"""Freeze the predetermined ReT-first comparator if burned convergence passes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


EXPECTED_ROOTS = [7_570_801, 7_570_824]
EXPECTED_SIGNATURE = [4, "scenario", 0.0, "expected"]
EXPECTED_TOLERANCE = 0.002
EXPECTED_TIE_BREAKER = "service"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(payload: dict[str, object], *, receipt_path: str, receipt_sha256: str) -> dict[str, object]:
    if payload.get("claim_status") != "BURNED_DEVELOPMENT_NO_CLAIM":
        raise ValueError("only burned convergence may freeze a comparator")
    if payload.get("phase") != "convergence":
        raise ValueError("receipt is not a merged convergence result")
    if payload.get("history_roots") != EXPECTED_ROOTS:
        raise ValueError("receipt does not cover the complete burned root block")
    if float(payload.get("value_indifference_tolerance", -1.0)) != EXPECTED_TOLERANCE:
        raise ValueError("unexpected ReT indifference tolerance")
    if payload.get("tie_breaker") != EXPECTED_TIE_BREAKER:
        raise ValueError("unexpected tie breaker")
    for key in (
        "selection_performed",
        "learner_return_used",
        "retained_minus_reset_used_for_selection",
    ):
        if payload.get(key) is not False:
            raise ValueError(f"invalid selection provenance: {key}")
    rows = [
        row
        for row in payload.get("convergence", [])
        if row.get("signature") == EXPECTED_SIGNATURE
    ]
    if len(rows) != 1:
        raise ValueError("predetermined universal comparator is missing or duplicated")
    selected = rows[0]
    if selected.get("convergence_pass") is not True:
        raise ValueError("predetermined universal comparator failed convergence")
    low_id = str(selected["low_config"])
    config = {
        "horizon": 4,
        "conditional_paths": 64,
        "mode": "scenario",
        "worst_product_floor": 0.0,
        "max_unresolved_orders": None,
        "tail_alpha": 0.10,
        "service_statistic": "expected",
        "value_indifference_tolerance": EXPECTED_TOLERANCE,
        "tie_breaker": EXPECTED_TIE_BREAKER,
    }
    return {
        "schema_version": "q_r1_comparator_v2_freeze_v1",
        "status": "FROZEN_BURNED_CALIBRATION_NO_FRESH_SEEDS",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scientific_role": "strongest_tested_universal_structured_comparator_not_optimality_claim",
        "primary_objective": "early_ret_complete_cohort",
        "secondary_disclosures": [
            "early_ret_visible",
            "ret_total",
            "worst_product_fill",
            "unresolved_orders",
            "lost_orders",
            "resources",
        ],
        "config_id": low_id,
        "config": config,
        "convergence_receipt": receipt_path,
        "convergence_sha256": receipt_sha256,
        "convergence_gate": {
            "first_action_agreement_min": 0.95,
            "mean_value_error_max": 0.005,
            "q95_value_error_max": 0.01,
            "abstentions_max": 0,
        },
        "observed_convergence": selected,
        "selection_used_learner_return": False,
        "selection_used_retained_minus_reset": False,
        "fresh_roots_assigned": None,
        "execution_authority": "BURNED_PARETO_AND_POWER_AUDIT_ONLY",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    payload = json.loads(args.receipt.read_text())
    result = freeze(
        payload,
        receipt_path=str(args.receipt),
        receipt_sha256=sha256(args.receipt),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
