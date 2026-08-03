#!/usr/bin/env python3
"""Audit G3-obs f2 from an already sealed result, without rerunning the DES.

The G3-obs preregistration requires the mean ordering

    real windowed signal > delayed signal > uninformed placebo > wrong claimant

The original runner only checked the real signal against the placebo and wrong-claimant arms.
This auditor reads the stored test-arm summaries, checks the complete ordering, and seals a
receipt.  It deliberately records a source-contract mismatch instead of treating an audit as a
retroactive rerun under the newer contract.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from supply_chain.arm_runner import seal_and_write  # noqa: E402


REQUIRED_ORDER = (
    "threshold_windowed",
    "threshold_delayed",
    "uninformed_placebo",
    "wrong_claimant",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def payload_self_sha256(payload: dict[str, Any]) -> str:
    body = dict(payload)
    body.pop("self_sha256", None)
    encoded = json.dumps(body, indent=1, sort_keys=True, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected an object in {path}")
    return value


def compare_order(means: dict[str, float]) -> dict[str, Any]:
    values = [float(means[name]) for name in REQUIRED_ORDER]
    comparisons = [
        {
            "left": REQUIRED_ORDER[i],
            "right": REQUIRED_ORDER[i + 1],
            "left_mean": values[i],
            "right_mean": values[i + 1],
            "strictly_greater": bool(values[i] > values[i + 1]),
        }
        for i in range(len(values) - 1)
    ]
    return {
        "required_order": list(REQUIRED_ORDER),
        "means": {name: float(means[name]) for name in REQUIRED_ORDER},
        "comparisons": comparisons,
        "passed": all(item["strictly_greater"] for item in comparisons),
    }


def audit(source: Path, intended_contract: Path) -> tuple[dict[str, Any], Path]:
    source_payload = load_json(source)
    stored_source_self_sha = source_payload.get("self_sha256")
    if not isinstance(stored_source_self_sha, str):
        raise ValueError("source artifact has no self_sha256")
    computed_source_self_sha = payload_self_sha256(source_payload)
    source_self_valid = stored_source_self_sha == computed_source_self_sha
    if not source_self_valid:
        raise ValueError(
            "source artifact self_sha256 is invalid: "
            f"stored={stored_source_self_sha}, computed={computed_source_self_sha}"
        )

    source_contract_path = ROOT / str(source_payload.get("contract_path", ""))
    source_contract_sha = str(source_payload.get("contract_sha256", ""))
    if not source_contract_path.exists():
        raise ValueError(f"source contract is missing: {source_contract_path}")
    source_contract_actual_sha = file_sha256(source_contract_path)
    source_contract_hash_valid = source_contract_sha == source_contract_actual_sha

    intended_contract_sha = file_sha256(intended_contract)
    source_contract_matches_intended = source_contract_sha == intended_contract_sha

    results = source_payload.get("results")
    if not isinstance(results, dict) or not results:
        raise ValueError("source artifact has no results mapping")

    cells: dict[str, Any] = {}
    for cell, result in sorted(results.items()):
        if not isinstance(result, dict):
            raise ValueError(f"result cell is not an object: {cell}")
        vs_constant = result.get("vs_constant")
        if not isinstance(vs_constant, dict):
            raise ValueError(f"cell has no vs_constant mapping: {cell}")
        means: dict[str, float] = {}
        confidence_intervals: dict[str, dict[str, float]] = {}
        for arm in REQUIRED_ORDER:
            summary = vs_constant.get(arm)
            if not isinstance(summary, dict) or "mean" not in summary:
                raise ValueError(f"cell {cell} is missing mean for {arm}")
            means[arm] = float(summary["mean"])
            confidence_intervals[arm] = {
                key: float(summary[key])
                for key in ("lcb95", "mean", "ucb95")
                if key in summary
            }
        checked = compare_order(means)
        checked["confidence_intervals"] = confidence_intervals
        checked["test_seeds"] = list(result.get("test_seeds", []))
        cells[str(cell)] = checked

    f2_pass = all(bool(cell["passed"]) for cell in cells.values())
    if not source_contract_matches_intended:
        claim_status = (
            "F2_ORDER_HOLDS_SOURCE_CONTRACT_MISMATCH"
            if f2_pass
            else "F2_ORDER_FAILS_SOURCE_CONTRACT_MISMATCH"
        )
        promotion_status = "BLOCKED_SOURCE_ARTIFACT_NOT_SEALED_UNDER_INTENDED_CONTRACT"
    else:
        claim_status = "F2_ORDER_RECONCILED_NO_NEW_SEEDS" if f2_pass else "F2_ORDER_FAILS"
        promotion_status = "F2_RECONCILED" if f2_pass else "HALTED_F2_FAILED"

    payload = {
        "schema_version": "g3_obs_f2_audit_v1",
        "audit_status": "AUDIT_ONLY_NO_NEW_SEEDS_NO_DES_RERUN",
        "claim_status": claim_status,
        "promotion_status": promotion_status,
        "source_artifact_path": str(source),
        "source_artifact_file_sha256": file_sha256(source),
        "source_artifact_self_sha256": stored_source_self_sha,
        "source_artifact_self_sha256_valid": source_self_valid,
        "source_schema_version": source_payload.get("schema_version"),
        "source_claim_status": source_payload.get("claim_status"),
        "source_contract_path": str(source_contract_path.relative_to(ROOT)),
        "source_contract_sha256": source_contract_sha,
        "source_contract_hash_valid": source_contract_hash_valid,
        "intended_contract_path": str(intended_contract),
        "intended_contract_sha256": intended_contract_sha,
        "source_contract_matches_intended": source_contract_matches_intended,
        "f2_required_order": list(REQUIRED_ORDER),
        "f2_all_cells_passed": f2_pass,
        "cells": cells,
        "seeds": list(source_payload.get("seeds", [])),
        "n_seeds": len(source_payload.get("seeds", [])),
        "new_seeds_opened": False,
        "des_rerun": False,
        "interpretation": (
            "The complete f2 ordering is present in the stored test summaries. This receipt "
            "does not retroactively change the source run's sealed contract; the source was "
            "sealed under the earlier contract and must remain labeled accordingly."
        ),
        "created_at_audit_utc": datetime.now(timezone.utc).isoformat(),
    }
    return payload, source


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("results/headroom/g3_obs_conversion_v2/result.json"),
    )
    parser.add_argument(
        "--intended-contract",
        type=Path,
        default=Path("docs/PREREGISTRO_G3_OBS_V2_POTENCIA_2026-08-02.md"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/headroom/g3_obs_conversion_v2/f2_audit_result.json"),
    )
    args = parser.parse_args()

    source = args.source if args.source.is_absolute() else ROOT / args.source
    intended_contract = (
        args.intended_contract
        if args.intended_contract.is_absolute()
        else ROOT / args.intended_contract
    )
    output = args.output if args.output.is_absolute() else ROOT / args.output

    payload, source_for_reference = audit(source, intended_contract)
    digest = seal_and_write(
        payload,
        output,
        contract=intended_contract,
        reference=source_for_reference,
    )
    print(json.dumps({
        "claim_status": payload["claim_status"],
        "promotion_status": payload["promotion_status"],
        "f2_all_cells_passed": payload["f2_all_cells_passed"],
        "source_contract_matches_intended": payload["source_contract_matches_intended"],
        "new_seeds_opened": payload["new_seeds_opened"],
        "des_rerun": payload["des_rerun"],
        "output": str(output),
        "self_sha256": digest,
    }, indent=2))
    return 0 if payload["f2_all_cells_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
