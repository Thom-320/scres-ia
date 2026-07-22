#!/usr/bin/env python3
"""Freeze the predeclared pure-ReT comparator (corrective amendment v1).

Supersedes ``scripts/freeze_q_r1_comparator_v2.py``, which is preserved unmodified as
evidence: it targets the service tie-break candidate that commit 51957969 rejected for
ranking instability, and it hardcodes ``conditional_paths = 64`` while its expected
signature carries no particle count -- so relaxing its tolerance/tie-breaker guards would
emit a c256 comparator mislabelled as c64.

Authority: commit 51957969 (2026-07-22 13:43:09 -0500), recorded BEFORE the c256/c1024
result, step 5-6: return to the pure-ReT selector, run the c256/c1024 convergence check
with the outcome gate unchanged, and freeze that comparator if and only if the original
thresholds pass.  See docs/Q_R1_COMPARATOR_V2_FREEZE_AMENDMENT_V1_2026-07-22.md.

This instrument introduces no new selection rule.  Every emitted field is either derived
from the receipt or copied from the pre-result authority; nothing is asserted by hand.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re


AMENDMENT_DOC = "docs/Q_R1_COMPARATOR_V2_FREEZE_AMENDMENT_V1_2026-07-22.md"
AUTHORITY_COMMIT = "51957969"

EXPECTED_ROOTS = [7_570_801, 7_570_824]
EXPECTED_CONFIG_ID = "qr1_v2_scenario_h4_c256_wf0.00_unone_expected_tol0.0000_legacy"
EXPECTED_HIGH_CONFIG_ID = "qr1_v2_scenario_h4_c1024_wf0.00_unone_expected_tol0.0000_legacy"
EXPECTED_SIGNATURE = [4, "scenario", 0.0, "expected"]
EXPECTED_TOLERANCE = 0.0
EXPECTED_TIE_BREAKER = "legacy"

# Unchanged from the original gate -- this amendment relaxes nothing.
GATE = {
    "first_action_agreement_min": 0.95,
    "mean_value_error_max": 0.005,
    "q95_value_error_max": 0.01,
    "abstentions_max": 0,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _paths_from_config_id(config_id: str) -> int:
    """Derive the particle budget from the config id instead of hardcoding it.

    The receipt's ``signature`` field does not carry the particle count, so matching on
    the signature alone cannot distinguish c64 from c256.  Deriving the count here is what
    makes a c64/c256 mislabel impossible.
    """
    match = re.search(r"_c(\d+)_", config_id)
    if match is None:
        raise ValueError(f"cannot derive conditional path budget from {config_id!r}")
    return int(match.group(1))


def freeze(payload: dict[str, object], *, receipt_path: str, receipt_sha256: str) -> dict[str, object]:
    if payload.get("claim_status") != "BURNED_DEVELOPMENT_NO_CLAIM":
        raise ValueError("only burned convergence may freeze a comparator")
    if payload.get("phase") != "convergence":
        raise ValueError("receipt is not a merged convergence result")
    if payload.get("history_roots") != EXPECTED_ROOTS:
        raise ValueError("receipt does not cover the complete burned root block")
    if float(payload.get("value_indifference_tolerance", -1.0)) != EXPECTED_TOLERANCE:
        raise ValueError("unexpected ReT indifference tolerance (amendment expects the legacy 0.0 band)")
    if payload.get("tie_breaker") != EXPECTED_TIE_BREAKER:
        raise ValueError("unexpected tie breaker (amendment expects the legacy tie breaker)")
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
        if str(row.get("low_config")) == EXPECTED_CONFIG_ID
    ]
    if len(rows) != 1:
        raise ValueError("predeclared pure-ReT c256 comparator is missing or duplicated")
    selected = rows[0]

    if list(selected.get("signature", [])) != EXPECTED_SIGNATURE:
        raise ValueError("selected row signature does not match the predeclared family")
    if str(selected.get("high_config")) != EXPECTED_HIGH_CONFIG_ID:
        raise ValueError("convergence reference is not the predeclared c1024 high budget")

    low_paths = _paths_from_config_id(str(selected["low_config"]))
    high_paths = _paths_from_config_id(str(selected["high_config"]))
    if high_paths <= low_paths:
        raise ValueError("high-budget reference must strictly exceed the frozen budget")

    # Re-verify the four gate criteria numerically; do not trust the boolean alone.
    agreement = float(selected["first_action_agreement"])
    mean_err = float(selected["mean_abs_planning_value_error"])
    q95_err = float(selected["q95_abs_planning_value_error"])
    abstentions = int(selected["low_abstentions"]) + int(selected["high_abstentions"])
    gate_checks = {
        "first_action_agreement": agreement >= GATE["first_action_agreement_min"],
        "mean_value_error": mean_err <= GATE["mean_value_error_max"],
        "q95_value_error": q95_err <= GATE["q95_value_error_max"],
        "abstentions": abstentions <= GATE["abstentions_max"],
    }
    if not all(gate_checks.values()):
        failed = sorted(k for k, ok in gate_checks.items() if not ok)
        raise ValueError(f"predeclared comparator failed convergence gate: {failed}")
    if selected.get("convergence_pass") is not True:
        raise ValueError("receipt boolean disagrees with recomputed gate")

    config = {
        "horizon": EXPECTED_SIGNATURE[0],
        "conditional_paths": low_paths,
        "mode": EXPECTED_SIGNATURE[1],
        "worst_product_floor": EXPECTED_SIGNATURE[2],
        "max_unresolved_orders": None,
        "tail_alpha": 0.10,
        "service_statistic": EXPECTED_SIGNATURE[3],
        "value_indifference_tolerance": EXPECTED_TOLERANCE,
        "tie_breaker": EXPECTED_TIE_BREAKER,
    }
    return {
        "schema_version": "q_r1_comparator_v2_freeze_v1",
        "status": "FROZEN_BURNED_CALIBRATION_NO_FRESH_SEEDS",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "amendment": AMENDMENT_DOC,
        "authority_commit": AUTHORITY_COMMIT,
        "supersedes_instrument": "scripts/freeze_q_r1_comparator_v2.py",
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
        "config_id": str(selected["low_config"]),
        "config": config,
        "convergence_receipt": receipt_path,
        "convergence_sha256": receipt_sha256,
        "convergence_gate": dict(GATE),
        "convergence_gate_recomputed": gate_checks,
        "observed_convergence": selected,
        "smallest_tested_convergent_budget": True,
        "untested_budgets_between_failed_and_frozen": [128],
        "superseded_by_coverage_expansion": {
            "config_id": "qr1_v2_scenario_h4_c64_wf0.00_unone_expected",
            "passed_at_states": 16,
            "failed_at_states": 96,
            "agreement_at_96_states": 0.90625,
        },
        "constraint_aware_family_frozen": False,
        "constraint_aware_family_status": "NO_CONVERGENT_BUDGET_TESTED_C256_ESCALATION_NOT_RUN",
        "selection_used_learner_return": False,
        "selection_used_retained_minus_reset": False,
        "fresh_roots_assigned": None,
        "learner_authorized": False,
        "execution_authority": "BURNED_PARETO_AND_POWER_AUDIT_ONLY",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate the receipt and print the contract without writing it",
    )
    args = parser.parse_args()
    if args.output.exists() and not args.dry_run:
        raise SystemExit(f"refusing to overwrite {args.output}")
    payload = json.loads(args.receipt.read_text())
    result = freeze(
        payload,
        receipt_path=str(args.receipt),
        receipt_sha256=sha256(args.receipt),
    )
    if not args.dry_run:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
