#!/usr/bin/env python3
"""Schema-only successor for a frozen comparator's secondary-disclosure list.

Two defects in the v1 disclosure list, both inherited verbatim from the superseded
freeze utility:

1. ``ret_total`` does not exist.  ``evaluate_calendar`` emits ``ret_full``; the only
   ``ret_total`` in the tree is an unrelated internal accumulator in
   scripts/benchmark_ret_ablation_static.py.  A disclosure named after a nonexistent key
   is silently dropped by any table builder that looks it up.

2. Renaming alone would be worse than the bug.  On this evaluation path ``ret_full`` is
   identically 0.0 -- ``full_order_values`` is gated on ``completed`` at score_time
   (program_o_full_des_transducer.py:731-742), which is empty for the early cohort -- and
   ``lost_orders`` is hardcoded ``np.zeros`` (program_o_full_des_transducer.py:795).
   Listing either as a live guardrail would put a dead column in every downstream table.

This script does NOT mutate the frozen contract.  The v1 file is the object the burned
Pareto was evaluated against and its hash is recorded in those shard outputs, so it stays
byte-identical.  The successor carries the identical ``config`` and convergence receipt,
corrects the disclosure names, and marks the degenerate ones as degenerate.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


CORRECTED_DISCLOSURES = [
    "early_ret_visible",
    "ret_visible",
    "worst_product_fill",
    "unresolved_orders",
    "resources",
]

DEGENERATE_DISCLOSURES = {
    "ret_full": (
        "identically 0.0 on this evaluation path: full_order_values is gated on "
        "`completed` at score_time (program_o_full_des_transducer.py:731-742), which is "
        "empty for the early cohort. Report only with this caveat; never as a live guardrail."
    ),
    "lost_orders": (
        "hardcoded np.zeros in the fast-path transducer "
        "(program_o_full_des_transducer.py:795). Structurally incapable of showing a loss "
        "on this path; never as a live guardrail."
    ),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def correct(payload: dict[str, object], *, source_path: str, source_sha: str) -> dict[str, object]:
    if payload.get("status") != "FROZEN_BURNED_CALIBRATION_NO_FRESH_SEEDS":
        raise SystemExit("source is not an executable freeze")
    if payload.get("secondary_disclosures") == CORRECTED_DISCLOSURES:
        raise SystemExit("source already carries the corrected disclosure list")

    successor = dict(payload)
    successor["schema_version"] = "q_r1_comparator_v2_freeze_v1_1"
    successor["created_at"] = datetime.now(timezone.utc).isoformat()
    successor["supersedes"] = source_path
    successor["supersedes_sha256"] = source_sha
    successor["correction_scope"] = "SCHEMA_ONLY_SECONDARY_DISCLOSURE_NAMES"
    successor["correction_note"] = (
        "config, config_id, convergence receipt, gate and execution authority are "
        "byte-identical to the superseded freeze; only the secondary-disclosure schema "
        "changed. The burned Pareto in results/q_r1/comparator_v2_frozen_pareto_c256_v1 "
        "was evaluated against the superseded file and its recorded hash remains valid."
    )
    successor["secondary_disclosures"] = list(CORRECTED_DISCLOSURES)
    successor["degenerate_disclosures"] = dict(DEGENERATE_DISCLOSURES)

    # The scientific object must be untouched.
    for key in ("config", "config_id", "convergence_receipt", "convergence_sha256",
                "convergence_gate", "primary_objective", "execution_authority",
                "fresh_roots_assigned", "learner_authorized"):
        if successor.get(key) != payload.get(key):
            raise SystemExit(f"schema-only correction altered {key}")
    return successor


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    result = correct(
        json.loads(source.read_text()),
        source_path=str(args.source),
        source_sha=sha256(source),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
