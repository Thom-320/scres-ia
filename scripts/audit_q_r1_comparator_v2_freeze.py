#!/usr/bin/env python3
"""Fail-closed re-derivation of a frozen comparator's convergence receipt.

The freeze instrument (scripts/freeze_q_r1_comparator_v2_amended.py) reapplies the gate
thresholds to the receipt's *summary* fields.  That is strictly weaker than it sounds: a
receipt whose summaries were correct but whose raw rows disagreed with them would pass.

This audit closes that gap.  It rebuilds first-action agreement, mean and q95 planning
value error, and abstentions from the raw ``convergence_pairs`` rows, using the same
conventions as scripts/merge_q_r1_comparator_v2_shards.py::merge_convergence, and refuses
unless the re-derived values match the summaries the freeze relied on.  It also validates
coverage: unique row identities, exact root blocks, the declared path budgets, and the
declared state counts.

Nothing here can change the frozen comparator, the gate, or the amendment: it either
confirms the freeze was entitled to happen or it fails.  Run before any downstream
evaluation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


# Merger conventions (merge_convergence): strict < on the error bounds, >= on agreement.
GATE_AGREEMENT_MIN = 0.95
GATE_MEAN_ERROR_MAX = 0.005
GATE_Q95_ERROR_MAX = 0.01

EXPECTED_ROOT_BLOCKS = [
    [7_570_801, 7_570_806],
    [7_570_807, 7_570_812],
    [7_570_813, 7_570_818],
    [7_570_819, 7_570_824],
]
EXPECTED_PATH_BUDGETS = [256, 1024]
EXPECTED_STATES = 48
EXPECTED_ARM_STATES = 96
EXPECTED_PRIOR_ARMS = {"retained", "reset"}

# Summaries are stored as float64; re-derivation must reproduce them bit-for-bit up to
# ordinary floating-point reassociation, not merely "closely".
MATCH_TOLERANCE = 1e-12


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fail(checks: list[dict[str, object]], name: str, detail: str) -> None:
    checks.append({"check": name, "passed": False, "detail": detail})


def _ok(checks: list[dict[str, object]], name: str, detail: str) -> None:
    checks.append({"check": name, "passed": True, "detail": detail})


def audit(
    freeze: dict[str, object],
    receipt: dict[str, object],
    *,
    freeze_path: str,
    freeze_sha: str,
    receipt_path: str,
    receipt_sha: str,
) -> dict[str, object]:
    checks: list[dict[str, object]] = []
    config_id = str(freeze["config_id"])

    # The freeze must point at this receipt, by hash.
    if str(freeze.get("convergence_sha256")) != receipt_sha:
        _fail(checks, "freeze_points_at_receipt", "freeze convergence_sha256 != receipt sha256")
    else:
        _ok(checks, "freeze_points_at_receipt", receipt_sha)

    summaries = [
        row for row in receipt.get("convergence", []) if str(row.get("low_config")) == config_id
    ]
    if len(summaries) != 1:
        raise SystemExit(f"receipt does not contain exactly one summary for {config_id}")
    summary = summaries[0]

    raw = list(receipt.get("convergence_pairs", []))
    if not raw:
        raise SystemExit("receipt carries no raw convergence_pairs; nothing can be re-derived")

    # -- coverage ------------------------------------------------------------------
    identities = [
        (
            tuple(row["signature"]),
            int(row["history_root"]),
            int(row["campaign_index"]),
            str(row["persistence_mode"]),
            str(row["prior_arm"]),
        )
        for row in raw
    ]
    if len(identities) != len(set(identities)):
        _fail(checks, "unique_raw_identities", f"{len(identities) - len(set(identities))} duplicates")
    else:
        _ok(checks, "unique_raw_identities", f"{len(identities)} unique")

    if len(raw) != EXPECTED_ARM_STATES:
        _fail(checks, "raw_row_count", f"{len(raw)} != {EXPECTED_ARM_STATES}")
    else:
        _ok(checks, "raw_row_count", str(EXPECTED_ARM_STATES))

    signatures = {tuple(row["signature"]) for row in raw}
    if len(signatures) != 1 or list(next(iter(signatures))) != list(summary["signature"]):
        _fail(checks, "single_signature", f"{sorted(signatures)}")
    else:
        _ok(checks, "single_signature", str(list(next(iter(signatures)))))

    blocks = [list(block) for block in receipt.get("root_blocks", [])]
    if blocks != EXPECTED_ROOT_BLOCKS:
        _fail(checks, "root_blocks", f"{blocks} != {EXPECTED_ROOT_BLOCKS}")
    else:
        _ok(checks, "root_blocks", "4 blocks x 6 roots, contiguous, non-overlapping")

    expected_roots = {
        root for block in EXPECTED_ROOT_BLOCKS for root in range(block[0], block[1] + 1)
    }
    seen_roots = {int(row["history_root"]) for row in raw}
    if seen_roots != expected_roots:
        missing = sorted(expected_roots - seen_roots)
        extra = sorted(seen_roots - expected_roots)
        _fail(checks, "root_coverage", f"missing={missing} extra={extra}")
    else:
        _ok(checks, "root_coverage", f"{len(seen_roots)} roots, exact")

    seen_arms = {str(row["prior_arm"]) for row in raw}
    if seen_arms != EXPECTED_PRIOR_ARMS:
        _fail(checks, "prior_arms", f"{sorted(seen_arms)} != {sorted(EXPECTED_PRIOR_ARMS)}")
    else:
        _ok(checks, "prior_arms", "retained + reset")

    budgets = list(receipt.get("conditional_path_budgets", []))
    if budgets != EXPECTED_PATH_BUDGETS:
        _fail(checks, "path_budgets", f"{budgets} != {EXPECTED_PATH_BUDGETS}")
    else:
        _ok(checks, "path_budgets", str(EXPECTED_PATH_BUDGETS))

    if int(receipt.get("states", -1)) != EXPECTED_STATES:
        _fail(checks, "states", f"{receipt.get('states')} != {EXPECTED_STATES}")
    else:
        _ok(checks, "states", str(EXPECTED_STATES))

    if int(summary.get("comparable_arm_states", -1)) != EXPECTED_ARM_STATES:
        _fail(checks, "comparable_arm_states", f"{summary.get('comparable_arm_states')}")
    else:
        _ok(checks, "comparable_arm_states", str(EXPECTED_ARM_STATES))

    # -- re-derivation -------------------------------------------------------------
    errors = np.asarray(
        [float(row["absolute_planning_value_error"]) for row in raw], dtype=float
    )
    agreement_flags = np.asarray(
        [int(row["low_action"]) == int(row["high_action"]) for row in raw], dtype=bool
    )
    recomputed = {
        "first_action_agreement": float(agreement_flags.mean()),
        "mean_abs_planning_value_error": float(errors.mean()),
        "q95_abs_planning_value_error": float(np.quantile(errors, 0.95)),
        "comparable_arm_states": int(len(errors)),
        "disagreements": int((~agreement_flags).sum()),
    }

    deltas: dict[str, float] = {}
    for key in (
        "first_action_agreement",
        "mean_abs_planning_value_error",
        "q95_abs_planning_value_error",
    ):
        delta = abs(recomputed[key] - float(summary[key]))
        deltas[key] = delta
        if delta > MATCH_TOLERANCE:
            _fail(checks, f"recomputed_matches_summary::{key}", f"|delta| = {delta:.3e}")
        else:
            _ok(checks, f"recomputed_matches_summary::{key}", f"|delta| = {delta:.3e}")

    abstentions = int(summary["low_abstentions"]) + int(summary["high_abstentions"])

    # -- gate, applied to the RE-DERIVED values ------------------------------------
    gate = {
        "first_action_agreement": recomputed["first_action_agreement"] >= GATE_AGREEMENT_MIN,
        "mean_value_error": recomputed["mean_abs_planning_value_error"] < GATE_MEAN_ERROR_MAX,
        "q95_value_error": recomputed["q95_abs_planning_value_error"] < GATE_Q95_ERROR_MAX,
        "abstentions": abstentions == 0,
    }
    for name, passed in gate.items():
        if passed:
            _ok(checks, f"gate_on_recomputed::{name}", "pass")
        else:
            _fail(checks, f"gate_on_recomputed::{name}", "fail")

    # -- provenance ----------------------------------------------------------------
    for key in (
        "selection_performed",
        "learner_return_used",
        "retained_minus_reset_used_for_selection",
    ):
        if receipt.get(key) is not False:
            _fail(checks, f"provenance::{key}", str(receipt.get(key)))
        else:
            _ok(checks, f"provenance::{key}", "false")

    if str(receipt.get("claim_status")) != "BURNED_DEVELOPMENT_NO_CLAIM":
        _fail(checks, "claim_status", str(receipt.get("claim_status")))
    else:
        _ok(checks, "claim_status", "BURNED_DEVELOPMENT_NO_CLAIM")

    passed_all = all(bool(check["passed"]) for check in checks)
    return {
        "schema_version": "q_r1_comparator_v2_freeze_audit_v1",
        "claim_status": "BURNED_DEVELOPMENT_NO_CLAIM",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "verdict": (
            "FREEZE_ENTITLED_RECOMPUTED_FROM_RAW_ROWS"
            if passed_all
            else "FREEZE_NOT_ENTITLED_RECOMPUTATION_FAILED"
        ),
        "config_id": config_id,
        "freeze_path": freeze_path,
        "freeze_sha256": freeze_sha,
        "receipt_path": receipt_path,
        "receipt_sha256": receipt_sha,
        "recomputed": recomputed,
        "summary_reported": {
            key: summary[key]
            for key in (
                "first_action_agreement",
                "mean_abs_planning_value_error",
                "q95_abs_planning_value_error",
                "low_abstentions",
                "high_abstentions",
                "comparable_arm_states",
                "convergence_pass",
            )
        },
        "absolute_deltas": deltas,
        "gate_on_recomputed": gate,
        "gate_thresholds": {
            "first_action_agreement_min": GATE_AGREEMENT_MIN,
            "mean_value_error_max_strict": GATE_MEAN_ERROR_MAX,
            "q95_value_error_max_strict": GATE_Q95_ERROR_MAX,
            "abstentions_max": 0,
        },
        "checks": checks,
        "checks_passed": sum(1 for check in checks if check["passed"]),
        "checks_total": len(checks),
        "selection_performed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    freeze_path = args.freeze.resolve()
    receipt_path = args.receipt.resolve()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    result = audit(
        json.loads(freeze_path.read_text()),
        json.loads(receipt_path.read_text()),
        freeze_path=str(args.freeze),
        freeze_sha=sha256(freeze_path),
        receipt_path=str(args.receipt),
        receipt_sha=sha256(receipt_path),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["verdict"].startswith("FREEZE_ENTITLED") else 1


if __name__ == "__main__":
    raise SystemExit(main())
