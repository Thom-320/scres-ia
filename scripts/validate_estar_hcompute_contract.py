#!/usr/bin/env python3
"""Fail-closed validator for the design-only E* H_compute amendment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from supply_chain.estar_kernel import MASKS, NODE_IDS


DESIGN_ONLY = "DESIGN_ONLY_NOT_AUTHORIZED_UNTIL_SUBMISSION_A_RECEIPT"
REQUIRED_TERMINALS = {
    "STOP_ESTAR_FLAGS_OFF_NON_EQUIVALENT",
    "STOP_ESTAR_DES_BRIDGE_NOT_READY",
    "STOP_ESTAR_PLANNER_NOT_BINDING",
    "NEURAL_AMORTIZATION_PREMIUM",
}


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def validate_contract(contract: dict[str, Any], registry: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if contract.get("status") != DESIGN_ONLY:
        errors.append("contract must remain design-only")
    authority = contract.get("authority", {})
    for key in (
        "scientific_execution_authorized",
        "fresh_roots_opened",
        "fresh_tapes_opened",
        "optimizer_seeds_assigned",
        "neural_training_authorized",
    ):
        if authority.get(key) is not False:
            errors.append(f"authority.{key} must be false")
    if contract.get("scope", {}).get("immutable_programs") != [
        "Program Q",
        "Program O",
        "thesis_1to1",
    ]:
        errors.append("Program Q/O/thesis_1to1 immutability is not explicit")
    if tuple(contract.get("scope", {}).get("approved_nodes", ())) != NODE_IDS:
        errors.append("approved node vocabulary differs from EStarKernel")
    bridge = contract.get("flags_off_bridge", {})
    if bridge.get("status") != "GOLDEN_VECTOR_REQUIRED_BEFORE_H_COMPUTE":
        errors.append("flags-off bridge must require a golden vector")
    if not isinstance(bridge.get("golden_payload_sha256"), str) or len(
        bridge.get("golden_payload_sha256", "")
    ) != 64:
        errors.append("flags-off bridge golden digest is missing or malformed")
    expanded_bridge = contract.get("expanded_des_bridge", {})
    if expanded_bridge.get("status") != "PENDING_SOURCE_CONSERVING_ADAPTER":
        errors.append("expanded DES bridge must remain pending before implementation")
    if expanded_bridge.get("required_before_h_compute_adjudication") is not True:
        errors.append("expanded DES bridge must gate H_compute adjudication")
    masks = contract.get("factorial_masks", [])
    ids = [row.get("mask_id") for row in masks if isinstance(row, dict)]
    if ids != list(MASKS):
        errors.append("factorial masks are incomplete or reordered")
    h_compute = contract.get("h_compute", {})
    if h_compute.get("burned_only") is not True:
        errors.append("h_compute must be burned-only")
    if int(h_compute.get("timing_repetitions", 0)) < int(
        h_compute.get("warmup_repetitions", 0)
    ):
        errors.append("timing repetitions must include warmup repetitions")
    if float(h_compute.get("native_cadence_fraction_budget", 0.0)) != 0.10:
        errors.append("native cadence budget must remain 0.10")
    if float(h_compute.get("relative_call_budget_vs_m000", 0.0)) != 10.0:
        errors.append("relative call budget must remain 10.0")
    metrics = contract.get("metric_hierarchy", {})
    if metrics.get("recommended_primary_endpoint") != "ret_excel_request_snapshot_v2":
        errors.append("recommended primary endpoint must remain Excel/ReT")
    cvar = metrics.get("cvar", {})
    for key in ("may_be_primary", "may_promote_alone", "may_block_alone"):
        if cvar.get(key) is not False:
            errors.append(f"metric_hierarchy.cvar.{key} must be false")
    if registry.get("new_seed_opening") is not False:
        errors.append("seed registry allows new seed opening")
    if registry.get("scientific_execution_authorized") is not False:
        errors.append("seed registry authorizes scientific execution")
    missing = sorted(REQUIRED_TERMINALS - set(contract.get("terminal_states", [])))
    if missing:
        errors.append(f"missing terminal states: {missing}")
    return {"ok": not errors, "errors": errors}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--seed-registry", type=Path, default=ROOT / "research/seed_custody_registry.json")
    args = parser.parse_args()
    result = validate_contract(load(args.contract), load(args.seed_registry))
    print(json.dumps({"status": "PASS" if result["ok"] else "FAIL", **result}, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
