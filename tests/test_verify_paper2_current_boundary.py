from __future__ import annotations

import json
from pathlib import Path

from scripts.verify_paper2_current_boundary import verify


REPO_ROOT = Path(__file__).resolve().parents[1]
CERTIFICATE = REPO_ROOT / "research/paper2_exhaustive_search/paper2_current_boundary_certificate_20260716.json"


def test_current_boundary_certificate_matches_live_artifacts() -> None:
    verdict = verify(CERTIFICATE, REPO_ROOT)
    assert verdict["status"] == "PASS_CURRENT_BOUNDARY_CERTIFICATE"
    assert verdict["failures"] == []


def test_current_boundary_certificate_rejects_positive_authorization(tmp_path: Path) -> None:
    payload = json.loads(CERTIFICATE.read_text(encoding="utf-8"))
    payload["authorization"]["learner_authorized"] = True
    modified = tmp_path / "boundary.json"
    modified.write_text(json.dumps(payload), encoding="utf-8")

    verdict = verify(modified, REPO_ROOT)
    assert verdict["status"] == "FAIL_CURRENT_BOUNDARY_CERTIFICATE"
    assert "unauthorized positive flag: learner_authorized" in verdict["failures"]
