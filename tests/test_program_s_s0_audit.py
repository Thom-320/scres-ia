from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_program_s_s0 import audit


ROOT = Path(__file__).resolve().parents[1]


def test_live_s0_result_passes_fail_closed_audit_without_seed_authority() -> None:
    payload = audit()
    assert payload["pass"] is True
    assert payload["verdict"] == "PASS_S0_RISK_ADAPTER_LIVE_AND_RISKOFF_IDENTICAL"
    assert payload["scientific_seed_authorization"] is False
    assert all(payload["checks"].values())


def test_s0_result_discloses_native_r21_zero_incidence() -> None:
    result = json.loads(
        (ROOT / "results/program_s/s0_preflight_v1/result.json").read_text()
    )
    r21 = result["thesis_incidence_report_only"]["masks"]["CROSS_ECHELON_SURGE"]["R21"]
    assert r21["total"] == 0
    assert r21["zero_tapes"] == 12
    assert result["deterministic_liveness"]["fixtures"]["R21_simultaneous"]["pass"]

