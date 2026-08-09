from pathlib import Path

from scripts.verify_procurement_overorder_source_v2 import build_payload


ROOT = Path(__file__).resolve().parent.parent


def test_full_thesis_independently_confirms_per_raw_material_text() -> None:
    payload = build_payload((ROOT / "thesis.txt").read_text(encoding="utf-8"))

    assert payload["claim_status"] == (
        "SOURCE_EXPLICIT_PER_RAW_MATERIAL__INDEPENDENTLY_CONFIRMED"
    )
    assert payload["source_evidence"][
        "operation_2_enumerates_rm1_through_rm12"
    ] is True
    assert payload["source_evidence"][
        "table_6_20_op2_each_rm_occurrences"
    ] == 3
    assert payload["source_evidence"][
        "table_6_20_op3_shift_levels_present"
    ] is True
    assert payload["source_evidence"]["textual_evidence_all_passed"] is True
    assert payload["source_evidence"]["textual_status_depends_only_on"] == [
        "f1_operation_2_enumerates_rm1_through_rm12",
        "f2_table_6_20_repeats_each_rm_for_all_three_shift_levels",
    ]
    assert payload["derived"]["op2_reorder_period_weeks"] == 4.0
    assert payload["derived"]["op2_procurement_components_week"] == 570_000.0
    assert payload["derived"]["op2_over_mean_s1_demand"] == 19.0 / 6.0
    assert payload["falsifier_summary"]["all_passed"] is True


def test_partial_phrase_does_not_pass_the_source_gate() -> None:
    payload = build_payload("Op2 supplies 190,000 units monthly.")

    assert payload["claim_status"] == "SOURCE_TEXT_NOT_INDEPENDENTLY_CONFIRMED"
    assert payload["falsifier_summary"]["n_failed"] >= 3


def test_matching_phrases_outside_the_named_sections_do_not_pass() -> None:
    misleading_text = " ".join(
        (
            "Q = {190,000 rm1, 190,000 rm12}",
            "190,000 units of each rm " * 3,
            "Operation 2 (Op2, j): no quantity is specified here.",
            "Operation 3 (Op3, j): follows.",
            "Table 6.20 Short-term manufacturing capacity (S) of the MFSC",
            "no Op2 rows are present here.",
            "Table 6.21 Design matrix with short-term manufacturing capacity",
            "190,000 units of each rm " * 3,
        )
    )

    payload = build_payload(misleading_text)

    assert payload["claim_status"] == "SOURCE_TEXT_NOT_INDEPENDENTLY_CONFIRMED"
    assert payload["source_evidence"][
        "operation_2_enumerates_rm1_through_rm12"
    ] is False
    assert payload["source_evidence"][
        "table_6_20_op2_each_rm_occurrences"
    ] == 0


def test_derived_checks_do_not_control_the_textual_status() -> None:
    direct_text_without_op3_context = " ".join(
        (
            "Operation 2 (Op2, j):",
            "Q = {190,000 rm1, 190,000 rm12}",
            "Operation 3 (Op3, j):",
            "Table 6.20 Short-term manufacturing capacity (S) of the MFSC",
            "190,000 units of each rm " * 3,
            "Table 6.21 Design matrix with short-term manufacturing capacity",
        )
    )

    payload = build_payload(direct_text_without_op3_context)

    assert payload["claim_status"] == (
        "SOURCE_EXPLICIT_PER_RAW_MATERIAL__INDEPENDENTLY_CONFIRMED"
    )
    assert payload["falsifiers"][
        "f3_table_6_20_contains_the_shift_scaled_op3_levels"
    ]["passed"] is False
    for name in (
        "f3_table_6_20_contains_the_shift_scaled_op3_levels",
        "f4_published_op2_exceeds_mean_s1_demand_threefold",
        "f5_op2_weekly_rate_is_close_to_the_published_s3_flow",
    ):
        assert payload["falsifiers"][name]["textual_status_gate"] is False
        assert payload["falsifiers"][name]["evidence_class"].startswith(
            "DERIVED_"
        )


def test_audit_is_complementary_to_canonical_v1() -> None:
    payload = build_payload((ROOT / "thesis.txt").read_text(encoding="utf-8"))

    assert payload["relationship_to_canonical_v1"] == {
        "artifact": "results/procurement_overorder_source/result.json",
        "relationship": "COMPLEMENTARY_INDEPENDENT_SOURCE_ONLY_AUDIT",
        "canonical_v1_d5_status": "MATCHES-PUBLISHED",
        "does_not_supersede_v1": True,
        "does_not_rerun_v1_simulation": True,
    }
    assert "supersedes" not in payload
