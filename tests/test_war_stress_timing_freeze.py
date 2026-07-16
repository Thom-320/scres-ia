from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.verify_war_stress_timing_freeze import verify


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "contracts/war_stress_timing_atlas_v1.json"
CUSTODY = ROOT / "research/paper2_exhaustive_search/war_stress_timing_seed_custody_20260716.json"
AUTHORIZATION = ROOT / "research/paper2_exhaustive_search/war_stress_timing_reopening_authorization_20260716.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_war_stress_timing_freeze_is_complete_and_unopened() -> None:
    verdict = verify(CONTRACT, CUSTODY, ROOT)
    assert verdict["status"] == "PASS_WAR_STRESS_TIMING_FREEZE"
    assert verdict["expanded_primary_cell_count"] == 144
    assert verdict["scientific_seeds_opened"] is False
    assert verdict["failures"] == []


def test_freeze_rejects_cross_regime_reversal_prerequisite(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["restricted_privileged_timing_family"][
        "cross_regime_posture_reversal_required"
    ] = True
    modified = tmp_path / "contract.json"
    modified.write_text(json.dumps(payload), encoding="utf-8")

    verdict = verify(modified, CUSTODY, ROOT)
    assert verdict["status"] == "FAIL_WAR_STRESS_TIMING_FREEZE"
    assert "obsolete cross-regime reversal prerequisite remains" in verdict["failures"]


def test_freeze_rejects_r3_in_primary_grid(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["primary_grid"]["masks"][0]["enabled_risks"].append("R3")
    modified = tmp_path / "contract.json"
    modified.write_text(json.dumps(payload), encoding="utf-8")

    verdict = verify(modified, CUSTODY, ROOT)
    assert verdict["status"] == "FAIL_WAR_STRESS_TIMING_FREEZE"
    assert any("R3" in failure or "risk masks" in failure for failure in verdict["failures"])


def test_reopening_authorization_is_content_addressed_and_preflight_only() -> None:
    payload = json.loads(AUTHORIZATION.read_text(encoding="utf-8"))
    assert payload["status"] == "AUTHORIZE_IMPLEMENTATION_AND_PREFLIGHT_ONLY"
    new = payload["new_contract"]
    for path_key, hash_key in (
        ("path", "sha256"),
        ("preregistration_path", "preregistration_sha256"),
        ("custody_path", "custody_sha256"),
        ("freeze_verification_path", "freeze_verification_sha256"),
    ):
        path = ROOT / new[path_key]
        assert path.is_file()
        assert _sha256(path) == new[hash_key]
    overlay = payload["gsa_overlay"]
    for path_key, hash_key in (
        ("contract_path", "contract_sha256"),
        ("preregistration_path", "preregistration_sha256"),
        ("verification_path", "verification_sha256"),
    ):
        path = ROOT / overlay[path_key]
        assert path.is_file()
        assert _sha256(path) == overlay[hash_key]
    assert any("open seed 7470001" in item for item in payload["not_authorized_now"])
    assert any("train a learner" in item for item in payload["not_authorized_now"])
