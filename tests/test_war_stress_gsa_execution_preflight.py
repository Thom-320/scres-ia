from __future__ import annotations

import json
from pathlib import Path

from scripts.verify_war_stress_gsa_execution_preflight import verify


ROOT = Path(__file__).resolve().parents[1]


def test_execution_preflight_passes_but_keeps_scientific_seeds_closed() -> None:
    result = verify(ROOT)
    assert result["status"] == "PASS_EXECUTOR_PREFLIGHT_SCIENTIFIC_BLOCKED"
    assert result["scientific_seeds_opened"] is False
    assert result["scientific_execution_authorized"] is False
    assert result["policy_templates"] == 50_202
    assert result["reduced_policy_templates"] == 50_049
    assert result["projected_scientific_episodes"] == 86_484_672


def test_command_manifest_has_watcher_first_and_no_scientific_authorization() -> None:
    payload = json.loads(
        (
            ROOT
            / "research/paper2_exhaustive_search/war_stress_gsa_command_manifest_20260716.json"
        ).read_text()
    )
    assert payload["benchmark"]["authorized"] is True
    assert payload["scientific"]["authorized"] is False
    assert payload["watcher"]["terminal_condition"].startswith("producer PGID/SID empty")


def test_des_crn_calibration_is_populated_but_does_not_authorize_interaction() -> None:
    payload = json.loads(
        (
            ROOT
            / "research/paper2_exhaustive_search/war_stress_crn_noise_calibration_20260716.json"
        ).read_text()
    )
    assert payload["status"] == "NONSCIENTIFIC_DES_CRN_DECOMPOSITION_FIXTURE_PASS"
    assert payload["scientific_seeds_opened"] is False
    assert len(payload["raw_rows"]) == 16
    assert all(row["r3_event_count"] == 0 for row in payload["raw_rows"])
    audit = payload["crn_two_way_audit"]
    assert audit["configuration_signal_variance"] >= 0.0
    assert audit["tape_main_effect_variance"] >= 0.0
    assert audit["configuration_by_tape_variance"] >= 0.0
    assert 0.0 <= audit["monte_carlo_fraction"] <= 1.0
    assert "does not calibrate an interaction" in payload["claim_boundary"]
