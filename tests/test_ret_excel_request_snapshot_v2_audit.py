import json
from pathlib import Path

from scripts.audit_ret_excel_request_snapshot_v2 import json_sha256


ROOT = Path(__file__).resolve().parent.parent
AUDIT = (
    ROOT
    / "research/paper2_exhaustive_search/ret_excel_request_snapshot_v2_implementation_audit_20260714.json"
)


def test_v2_implementation_audit_is_hashed_and_exact():
    payload = json.loads(AUDIT.read_text())
    expected = payload.pop("content_sha256")

    assert json_sha256(payload) == expected
    assert payload["canonical_development_contract"] == (
        "ret_excel_request_snapshot_v2"
    )
    replay = payload["canonical_aggregator_workbook_replay"]
    assert replay["formula_rows"] == 47_546
    assert replay["mismatches"] == 0
    assert replay["max_abs_diff"] == 0.0
    assert replay["passes"] is True


def test_v2_audit_keeps_prior_results_and_papers_blocked():
    payload = json.loads(AUDIT.read_text())
    authorization = payload["scientific_authorization"]

    assert payload["superseded_contract"]["disposition"].startswith(
        "QUARANTINED"
    )
    assert authorization["paper2_confirmed"] is False
    assert authorization["paper3_authorized"] is False
    assert authorization["prior_h_j_mtr_results_restored"] is False
