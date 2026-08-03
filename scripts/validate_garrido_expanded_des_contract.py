#!/usr/bin/env python3
"""Fail-closed validator for the design-only Garrido E* contract.

This validator checks contract and custody state only.  It never opens roots,
allocates seeds, runs the DES, launches a comparator, or trains a learner.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "contracts" / "garrido_expanded_des_e_star_v1.json"
SEED_REGISTRY_PATH = ROOT / "research" / "seed_custody_registry.json"


DESIGN_ONLY_STATUS = "DESIGN_ONLY_NOT_AUTHORIZED_UNTIL_SUBMISSION_A_RECEIPT"


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def validate_contract(
    contract: dict[str, Any], seed_registry: dict[str, Any]
) -> dict[str, Any]:
    errors: list[str] = []
    if contract.get("status") != DESIGN_ONLY_STATUS:
        errors.append("contract is not design-only")

    authority = contract.get("authority", {})
    for field in (
        "scientific_execution_authorized",
        "fresh_roots_opened",
        "fresh_tapes_opened",
        "optimizer_seeds_assigned",
        "neural_training_authorized",
    ):
        if authority.get(field) is not False:
            errors.append(f"authority.{field} must be false")

    seed_policy = contract.get("seed_and_execution_policy", {})
    if seed_policy.get("new_seed_opening") is not False:
        errors.append("contract allows new seed opening")

    if seed_registry.get("new_seed_opening") is not False:
        errors.append("seed registry allows new seed opening")
    if seed_registry.get("scientific_execution_authorized") is not False:
        errors.append("seed registry authorizes scientific execution")

    metrics = contract.get("metric_hierarchy", {})
    allowed = {
        item.get("id")
        for item in metrics.get("allowed_primary_endpoints", [])
        if isinstance(item, dict)
    }
    if metrics.get("primary_selection_status") != (
        "PENDING_PI_AND_GARRIDO_SIGNATURE_BEFORE_FRESH_DATA"
    ):
        errors.append("primary metric selection is not pending signature")
    if allowed != {"ret_excel_request_snapshot_v2", "cobb_douglas_index"}:
        errors.append("primary endpoint set changed unexpectedly")

    cvar = metrics.get("cvar", {})
    for field in ("may_be_primary", "may_promote_alone", "may_block_alone"):
        if cvar.get(field) is not False:
            errors.append(f"metric_hierarchy.cvar.{field} must be false")

    if len(contract.get("factorial_masks", [])) != 8:
        errors.append("E* factorial must contain all eight masks")

    return {"ok": not errors, "errors": errors}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    # --contract is REQUIRED: a default is how three artifacts got sealed against
    # the wrong document. Previous default was CONTRACT_PATH
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--seed-registry", type=Path, default=SEED_REGISTRY_PATH)
    args = parser.parse_args()

    result = validate_contract(
        load_json(args.contract), load_json(args.seed_registry)
    )
    print(json.dumps({"status": "PASS_DESIGN_ONLY" if result["ok"] else "FAIL", **result}, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
