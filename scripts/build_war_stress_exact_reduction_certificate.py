#!/usr/bin/env python3
"""Certify the only currently proven exact policy-family reduction."""

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

from scripts.build_war_stress_policy_manifest import iter_policy_templates


def _update(digest: Any, row: dict[str, Any]) -> None:
    digest.update(json.dumps(row, sort_keys=True, separators=(",", ":")).encode())
    digest.update(b"\n")


def build_certificate() -> dict[str, Any]:
    full_digest = hashlib.sha256()
    reduced_digest = hashlib.sha256()
    full_count = reduced_count = 0
    removed: list[dict[str, str]] = []
    for row in iter_policy_templates():
        _update(full_digest, row)
        full_count += 1
        if row["family"] == "open_loop_8week_periodic" and row["payload"] == 0:
            # Open-loop environments initialize at `low`. Calendar 00000000
            # therefore issues HOLD forever and is trajectory-identical to the
            # corresponding constant-low policy. Calendar 11111111 is *not*
            # removed: it intervenes from low to high after reset and may face
            # shift ramp/buffer commitment lags unlike constant-high.
            removed.append(
                {
                    "removed_policy_id": row["policy_id"],
                    "retained_policy_id": f"constant::{row['low']}",
                    "proof_class": "IDENTICAL_INITIAL_POSTURE_AND_ALL_ZERO_WEEKLY_ACTIONS",
                }
            )
            continue
        _update(reduced_digest, row)
        reduced_count += 1
    return {
        "schema_version": "war_stress_exact_policy_reduction_certificate_v1",
        "status": "EXACT_DUPLICATE_ONLY_REDUCTION_PASS__INSUFFICIENT_FOR_COMPUTE",
        "full_policy_count": full_count,
        "reduced_policy_count": reduced_count,
        "removed_duplicate_count": len(removed),
        "reduction_fraction": (full_count - reduced_count) / full_count,
        "full_ordered_rows_sha256": full_digest.hexdigest(),
        "reduced_ordered_rows_sha256": reduced_digest.hexdigest(),
        "duplicate_mapping_sha256": hashlib.sha256(
            json.dumps(removed, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "duplicate_mappings": removed,
        "vector_dominance_used": False,
        "branch_and_bound_used": False,
        "claim_boundary": (
            "Only exact trajectory duplicates are removed. No ReT, guardrail, "
            "resource, dominance, bound or learned prediction is used for pruning."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "research/paper2_exhaustive_search/war_stress_exact_policy_reduction_20260716.json"
        ),
    )
    args = parser.parse_args()
    payload = build_certificate()
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "duplicate_mappings"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
