from copy import deepcopy
import json
from pathlib import Path

from scripts.audit_paper2_switch4_exact_ties import audit_exact_ties


ROOT = Path(__file__).resolve().parent.parent


def _payload():
    payload = json.loads(
        (
            ROOT
            / "results/paper2_bound_harness/switch_complexity_screen/"
            "9672e21_local_20260713_sieve1/result.json"
        ).read_text()
    )
    # Reuse the exact <=3 candidate rows as a compact fixture while presenting
    # the status vocabulary emitted by the dedicated switch4 producer.
    payload["scientific_status"] = "CALIBRATION_SWITCH4_BOUNDARY_ACTIVE"
    payload["boundary_hit"] = True
    return payload


def test_unique_switch3_winner_is_boundary_active_for_its_restricted_family():
    audit = audit_exact_ties(_payload(), boundary_switch_count=3)

    assert audit["passed"] is True
    assert audit["primary_tie_count"] == 1
    assert audit["primary_tie_switch_counts"] == [3]
    assert audit["tie_spans_boundary"] is True


def test_interior_selected_tie_with_boundary_maximizer_fails_closed():
    payload = deepcopy(_payload())
    rows = payload["candidate_family"]["rows"]
    selected = payload["calibration"]["selected_index"]
    interior = next(
        index for index, row in enumerate(rows)
        if row["switch_count"] < 3 and index < selected
    )
    rows[interior]["exact_sum_numerator"] = rows[selected]["exact_sum_numerator"]
    rows[interior]["exact_sum_denominator"] = rows[selected]["exact_sum_denominator"]
    payload["calibration"]["selected_index"] = min(interior, selected)
    payload["calibration"]["primary_tie_count"] = 2
    payload["calibration"]["selected_switch_count"] = rows[
        payload["calibration"]["selected_index"]
    ]["switch_count"]
    payload["scientific_status"] = "CALIBRATION_SWITCH4_INTERIOR"
    payload["boundary_hit"] = False

    audit = audit_exact_ties(payload, boundary_switch_count=3)

    assert audit["tie_spans_boundary"] is True
    assert audit["effective_scientific_status"] == "CALIBRATION_SWITCH4_BOUNDARY_ACTIVE"
    assert audit["passed"] is False
    assert "producer status ignores an exact boundary-spanning tie" in audit["failures"]


def test_primary_tie_count_mismatch_is_rejected():
    payload = deepcopy(_payload())
    payload["calibration"]["primary_tie_count"] = 2

    audit = audit_exact_ties(payload, boundary_switch_count=3)

    assert audit["passed"] is False
    assert "producer primary_tie_count mismatch" in audit["failures"]
