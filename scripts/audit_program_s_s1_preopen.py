#!/usr/bin/env python3
"""Audit Program S S1 readiness while preserving Program Q VPS priority."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "contract": ROOT / "contracts/program_s_product_mix_risk_interaction_gsa_v1.json",
    "seed_manifest": ROOT / "research/paper2_exhaustive_search/program_s_seed_manifest_v1.json",
    "design_freeze": ROOT / "research/paper2_exhaustive_search/program_s_morris_design_freeze_v1.json",
    "design": ROOT / "research/paper2_exhaustive_search/program_s_morris_design_v1.json",
    "s0": ROOT / "results/program_s/s0_preflight_v1/result.json",
    "s0_audit": ROOT / "results/program_s/s0_preflight_v1/independent_audit_v1.json",
    "transducer": ROOT / "results/program_s/s1_transducer_preflight_v1/result.json",
    "program_q": ROOT / "contracts/program_q_frozen_policy_replication_v1.json",
}
OUT = ROOT / "research/paper2_exhaustive_search/program_s_s1_preopen_audit_v1.json"


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def audit() -> dict:
    payloads = {name: json.loads(path.read_text()) for name, path in FILES.items()}
    design = payloads["design"]
    freeze = payloads["design_freeze"]
    generated_design = dict(design)
    declared_hash = generated_design.pop("design_sha256")
    result_files = list((ROOT / "results").rglob("*"))
    suspicious_751 = [
        str(path.relative_to(ROOT))
        for path in result_files
        if path.is_file() and any(str(seed) in path.name for seed in range(7510001, 7510013))
    ]
    groups = design["groups"]
    checks = {
        "s0_passed": payloads["s0"]["pass"] is True,
        "s0_audit_passed": payloads["s0_audit"]["pass"] is True,
        "transducer_all_masks_exact": set(payloads["transducer"]["eligible_masks"])
        == set(payloads["contract"]["physical_masks"]),
        "transducer_preflight_passed": payloads["transducer"]["pass"] is True,
        "design_hash_valid": digest(generated_design) == declared_hash,
        "six_mask_stratum_groups": len(groups) == 6,
        "ten_trajectories_per_group": all(len(group["trajectories"]) == 10 for group in groups),
        "capacity_one_anchor_per_native_and_coupling": all(
            all(anchor["baseline_capacity_multiplier"] == 1.0 for anchor in group["mandatory_capacity_1_anchors"])
            and len(group["mandatory_capacity_1_anchors"])
            == (1 if group["stratum"] == "THESIS_NATIVE_INDEPENDENT" else 3)
            for group in groups
        ),
        "seed_block_declarations_match": freeze["seed_block"] == [7510001, 7510012]
        and payloads["seed_manifest"]["reserved_unopened"]["S1_Morris"]
        == [7510001, 7510012],
        "no_751_result_file": not suspicious_751,
        "program_q_still_pending_preopen": payloads["program_q"]["status"]
        == "FROZEN_POWER_PASS_N_256_PENDING_SEED_AUTHORIZATION",
    }
    technically_ready = all(
        value for key, value in checks.items() if key != "program_q_still_pending_preopen"
    )
    q_has_priority = checks["program_q_still_pending_preopen"]
    return {
        "schema_version": "program_s_s1_preopen_audit_v1",
        "checks": checks,
        "suspicious_751_result_files": suspicious_751,
        "technically_ready": technically_ready,
        "program_q_vps_priority_active": q_has_priority,
        "scientific_seed_authorization": bool(technically_ready and not q_has_priority),
        "verdict": (
            "HOLD_S1_TECHNICALLY_READY_PROGRAM_Q_HAS_VPS_PRIORITY"
            if technically_ready and q_has_priority
            else "PASS_S1_PREOPEN_AUTHORIZED"
            if technically_ready
            else "STOP_S1_PREOPEN_AUDIT_FAILURE"
        ),
        "file_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in FILES.items()
        },
    }


def main() -> int:
    payload = audit()
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["technically_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

