#!/usr/bin/env python3
"""Fail-closed structural audit of the independently produced Program S S0 result."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "contracts/program_s_product_mix_risk_interaction_gsa_v1.json"
RESULT_PATH = ROOT / "results/program_s/s0_preflight_v1/result.json"
OUT_PATH = ROOT / "results/program_s/s0_preflight_v1/independent_audit_v1.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def audit() -> dict:
    contract = json.loads(CONTRACT_PATH.read_text())
    result = json.loads(RESULT_PATH.read_text())
    fixtures = result["deterministic_liveness"]["fixtures"]
    checks = {
        "contract_frozen_before_scientific_seeds": contract["status"]
        == "FROZEN_S0_IMPLEMENTATION_NO_SCIENTIFIC_SEEDS_OPENED",
        "no_scientific_seeds_opened": result["scientific_seed_blocks_opened"] == [],
        "producer_verdict_exact": result["verdict"]
        == "PASS_S0_RISK_ADAPTER_LIVE_AND_RISKOFF_IDENTICAL",
        "adapter_defaults_bitwise_equal": result["riskoff_identity"][
            "adapter_defaults_bitwise_equal"
        ]
        is True,
        "custody_numeric_drift_bounded": max(
            row["abs_diff"] for row in result["riskoff_identity"]["checks"].values()
        )
        <= float(result["riskoff_identity"]["custodied_numeric_tolerance"]),
        "r11_both_targets": fixtures["R11_op5"]["pass"]
        and fixtures["R11_op6"]["pass"],
        "r14_exact_rework_and_conservation": fixtures["R14_op7_rework"]["pass"]
        and fixtures["R14_op7_rework"]["rework_started_quantity"]
        == fixtures["R14_op7_rework"]["rework_returned_quantity"],
        "r21_simultaneous_exact_ops": fixtures["R21_simultaneous"]["pass"]
        and fixtures["R21_simultaneous"]["expected_ops"] == [3, 5, 6, 7, 9],
        "r22_all_routes_separately_live": all(
            fixtures[f"R22_op{op_id}"]["pass"] for op_id in (4, 8, 10, 12)
        ),
        "r23_live": fixtures["R23_op11"]["pass"],
        "r24_product_label_preserved": fixtures["R24_product_preservation"]["pass"]
        and bool(fixtures["R24_product_preservation"]["contingent_product_labels"]),
        "incidence_separate_from_liveness": result["thesis_incidence_report_only"][
            "rare_zero_events_do_not_invalidate_deterministic_liveness"
        ]
        is True,
        "r21_native_zero_disclosed": result["thesis_incidence_report_only"]["masks"][
            "CROSS_ECHELON_SURGE"
        ]["R21"]["total"]
        == 0,
    }
    passed = all(checks.values())
    return {
        "schema_version": "program_s_s0_independent_audit_v1",
        "contract_sha256": sha256(CONTRACT_PATH),
        "producer_result_sha256": sha256(RESULT_PATH),
        "checks": checks,
        "pass": passed,
        "verdict": (
            "PASS_S0_RISK_ADAPTER_LIVE_AND_RISKOFF_IDENTICAL"
            if passed
            else "STOP_S0_RISK_ADAPTER_OR_PARITY_FAILURE"
        ),
        "authority": (
            "S1_IMPLEMENTATION_AND_PREOPEN_AUDIT_ONLY"
            if passed
            else "NO_PROGRAM_S_SCREEN_AUTHORIZED"
        ),
        "scientific_seed_authorization": False,
    }


def main() -> int:
    payload = audit()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
