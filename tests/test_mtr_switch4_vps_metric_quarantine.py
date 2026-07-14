import json
from pathlib import Path

from scripts.record_mtr_switch4_metric_quarantine import json_sha256


ROOT = Path(__file__).resolve().parent.parent
AUDIT = (
    ROOT
    / "research/paper2_exhaustive_search/mtr_switch4_vps_metric_quarantine_20260714.json"
)


def test_metric_quarantine_is_hashed_incomplete_and_non_evidence():
    payload = json.loads(AUDIT.read_text())
    expected = payload.pop("content_sha256")

    assert json_sha256(payload) == expected
    assert payload["scientific_status"] == (
        "CONTROLLED_TERMINATION_OBSOLETE_METRIC_NO_RESULT"
    )
    assert payload["metric_disposition"] == "QUARANTINED_NOT_CANONICAL"
    assert payload["final_watcher_observation"]["progress"]["completed"] == 5
    assert payload["final_watcher_observation"]["progress"]["total"] == 60
    assert payload["final_watcher_observation"]["result_exists"] is False
    custody = payload["watcher_custody_assessment"]
    assert custody["prestart_and_parent_pid_monitoring_verified"] is True
    assert custody["whole_scientific_session_termination_verified"] is False
    assert len(payload["orphan_worker_remediation"]["worker_pids"]) == 6
    assert payload["orphan_worker_remediation"]["signal"] == "SIGTERM"
    assert payload["virgin_tapes_opened"] is False
    assert payload["scientific_evidence_value"].startswith("NONE")
