import hashlib
import json
from copy import deepcopy
from pathlib import Path

from scripts.verify_paper2_exhaustion import (
    validate_boundary_family_proof_ledger,
    validate_metric_governance,
    validate_paper3_claim_supersession,
)
from scripts.validate_phase0_failure_taxonomy import validate as validate_phase0_taxonomy


ROOT = Path(__file__).resolve().parent.parent
SEARCH = ROOT / "research" / "paper2_exhaustive_search"


def load(name: str):
    return json.loads((SEARCH / name).read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_failure_taxonomy_is_unique_and_keeps_corrective_k3_truth():
    taxonomy = load("phase0_failure_taxonomy.json")
    rows = {row["family_id"]: row for row in taxonomy["decision_families"]}
    assert len(rows) == len(taxonomy["decision_families"])
    assert set("ABC").issubset({family_id[0] for family_id in rows})
    assert rows["K3_budgeted_replenishment"]["final_verdict"] == (
        "RETRACT_K3_ADAPTIVE_AND_NEURAL_CLAIMS_STATIC_PERIOD8_CONFOUND"
    )
    assert rows["K3_budgeted_replenishment"]["retained_value"] is None


def test_phase0_taxonomy_has_canonical_metric_exact_failure_and_hashed_evidence():
    taxonomy = load("phase0_failure_taxonomy.json")
    validation = validate_phase0_taxonomy(taxonomy)

    assert validation["passed"] is True
    assert validation["failures"] == []
    assert validation["family_count"] == 17
    assert validation["evidence_hashes_checked"] >= 34
    assert taxonomy["canonical_metric"] == "ret_excel_request_snapshot_v2"
    assert "cannot establish" in taxonomy["metric_quarantine_rule"]
    assert all(
        row["primary_metric"] and row["exact_failure"]
        for row in taxonomy["decision_families"]
    )


def test_taxonomy_does_not_mislabel_tested_policies_as_optimized_hobs():
    taxonomy = load("phase0_failure_taxonomy.json")
    for row in taxonomy["decision_families"]:
        estimate = row.get("h_obs")
        if estimate and "policy" in estimate:
            assert estimate["estimate_kind"].startswith("tested_policy_delta")
    dra2 = next(
        row for row in taxonomy["decision_families"]
        if row["family_id"] == "DRA2_DRA2b_finite_convoy"
    )
    assert dra2["h_pi"]["estimate_kind"] == "restricted_10_14_day_oracle_not_full_horizon_H_PI"


def test_no_candidate_is_promotable_and_paper3_is_closed():
    registry = load("approach_registry.json")
    assert registry["registry_summary"]["promotable"] == 0
    assert registry["registry_summary"]["paper2_confirmed"] is False
    assert registry["registry_summary"]["paper3_authorized"] is False
    assert not any(row["state"] == "promotable" for row in registry["approaches"])


def test_boundary_family_proof_ledger_is_complete_but_explicitly_nonterminal():
    registry = load("approach_registry.json")
    ledger = load("boundary_family_proof_ledger.json")
    validation = validate_boundary_family_proof_ledger(registry, ledger, root=ROOT)

    assert validation["schema_ok"] is True
    assert validation["coverage_ok"] is True
    assert validation["summary_consistent"] is True
    assert validation["failures"] == []
    assert validation["all_families_terminal_b_eligible"] is False
    assert validation["terminal_family_ids"] == []
    assert set(validation["nonterminal_family_ids"]) == {
        row["family_id"] for row in registry["approaches"]
    }
    assert ledger["terminal_b_supported"] is False


def test_canonical_v2_metric_governance_quarantines_every_visible_v1_claim():
    governance = load("metric_governance_audit.json")
    source_semantics = load(
        "ret_excel_visible_v1_source_semantics_audit_20260714.json"
    )
    implementation = load(
        "ret_excel_request_snapshot_v2_implementation_audit_20260714.json"
    )
    excel_reaudit = load("excel_metric_reaudit_20260713.json")

    validation = validate_metric_governance(
        governance,
        source_semantics,
        implementation,
        excel_reaudit,
        root=ROOT,
    )

    assert validation["passed"] is True
    assert validation["failures"] == []
    assert validation["hashes_checked"] == 3
    assert validation["implementation_source_hashes_checked"] == 5
    assert validation["canonical_contract"] == "ret_excel_request_snapshot_v2"
    assert validation["visible_v1_disposition"] == (
        "QUARANTINED_METRIC_DEVELOPMENT_ONLY"
    )
    assert validation["quarantined_scopes"] == [
        "all_ret_excel_visible_v1_oat_ledger_outputs",
        "mtr_visible_v1_switch_frontiers",
        "program_h_visible_v1",
        "program_j_visible_v1",
    ]
    assert validation["paper2_positive_confirmed"] is False
    assert validation["paper2_null_confirmed"] is False
    assert validation["paper2_ceiling_confirmed"] is False
    assert validation["prior_h_j_mtr_results_restored"] is False


def test_metric_governance_fails_if_visible_v1_or_same_time_is_promoted():
    governance = deepcopy(load("metric_governance_audit.json"))
    source_semantics = load(
        "ret_excel_visible_v1_source_semantics_audit_20260714.json"
    )
    implementation = load(
        "ret_excel_request_snapshot_v2_implementation_audit_20260714.json"
    )
    excel_reaudit = load("excel_metric_reaudit_20260713.json")

    governance["canonical_endpoint"]["request_snapshot_semantics"][
        "same_timestamp_authority"
    ] = "CONFIRMED_WITHOUT_DOMAIN_RESPONSE"
    governance["superseded_contracts"]["ret_excel_visible_v1"][
        "status"
    ] = "CANONICAL"
    governance["paper2_authorization"]["paper2_null_confirmed"] = True
    governance["quarantine_registry"] = [
        row
        for row in governance["quarantine_registry"]
        if row["scope_id"] != "mtr_visible_v1_switch_frontiers"
    ]

    validation = validate_metric_governance(
        governance,
        source_semantics,
        implementation,
        excel_reaudit,
        root=ROOT,
    )

    assert validation["passed"] is False
    assert "same-time convention was promoted without Garrido authority" in validation[
        "failures"
    ]
    assert "visible-v1 is not quarantined" in validation["failures"]
    assert "H/J/MTR or global visible-v1 quarantine is missing" in validation[
        "failures"
    ]
    assert "scientific authorization must remain false: paper2_null_confirmed" in (
        validation["failures"]
    )


def test_historical_noncanonical_bounds_cannot_masquerade_as_visible_v1_ceilings():
    registry = load("approach_registry.json")
    ledger = load("boundary_family_proof_ledger.json")
    rows = {row["family_id"]: row for row in ledger["families"]}

    program_h = rows["multi_echelon_information_lag_only"]
    program_j = rows["alarm_signal_finite_maintenance"]
    assert program_h["governing_metric"] == "ret_excel_request_snapshot_v2"
    assert program_h["evidence_metric"] == "ret_excel_full_ledger_order_adapter"
    assert program_j["governing_metric"] == "ret_excel_request_snapshot_v2"
    assert program_j["evidence_metric"].startswith("ret_excel_request_snapshot_v2")
    assert program_h["terminal_b_eligible"] is False
    assert program_j["terminal_b_eligible"] is False

    forged = deepcopy(ledger)
    forged_h = next(
        row
        for row in forged["families"]
        if row["family_id"] == "multi_echelon_information_lag_only"
    )
    forged_h["terminal_b_eligible"] = True
    forged_h["closure_kind"] = "certified_quantitative_global_upper_bound"
    forged_h["quantitative_ceiling"]["ucb95"] = 0.009
    forged_h["scope_closes_registered_family"] = True
    forged["summary"]["terminal_b_eligible_families"] = 1
    forged["summary"]["nonterminal_families"] = 16

    validation = validate_boundary_family_proof_ledger(registry, forged, root=ROOT)
    family_failure = next(
        failure
        for failure in validation["failures"]
        if failure.get("family_id") == forged_h["family_id"]
    )
    assert "evidence_metric_does_not_match_governing_metric" in family_failure[
        "reasons"
    ]


def test_historical_visible_v1_ceiling_audit_is_content_addressed_and_nonterminal():
    audit = load("historical_visible_v1_ceiling_audit_20260714.json")

    assert audit["governing_metric"] == "ret_excel_visible_v1"
    assert audit["summary"]["lanes_audited"] == len(audit["lanes"]) == 10
    assert audit["summary"][
        "existing_family_wide_visible_v1_matched_resource_ceilings"
    ] == 0
    assert audit["summary"]["paper2_confirmed"] is False
    assert audit["summary"]["terminal_boundary_supported"] is False
    for row in audit["lanes"]:
        artifacts = row.get("artifacts", [row.get("artifact")])
        for artifact in artifacts:
            assert artifact is not None
            path = ROOT / artifact["path"]
            assert path.is_file(), artifact["path"]
            assert sha256(path) == artifact["sha256"]

    by_lane = {row["lane"]: row for row in audit["lanes"]}
    assert by_lane["Program H"]["evidence_metric"] == (
        "ret_excel_full_ledger_order_adapter"
    )
    assert by_lane["Program J"]["evidence_metric"] == "ret_excel_full_ledger"


def test_global_sensitivity_inventory_reports_every_executed_design_and_scope():
    inventory = load("global_sensitivity_portfolio_inventory.json")

    assert inventory["governing_metric"] == "ret_excel_request_snapshot_v2"
    assert len(inventory["executed_studies"]) == 5
    assert len(inventory["active_unscreened_or_incomplete"]) == 4
    assert "neither Return A nor Return B" in inventory["portfolio_conclusion"]
    for study in inventory["executed_studies"]:
        assert study["rows_complete"] is True
        artifacts = study.get("artifacts", [study.get("artifact")])
        for artifact in artifacts:
            assert artifact is not None
            path = ROOT / artifact["path"]
            assert path.is_file(), artifact["path"]
            assert sha256(path) == artifact["sha256"]

    reconstructed = next(
        row
        for row in inventory["executed_studies"]
        if row["study"].startswith("Historical Program-I")
    )
    assert reconstructed["total_reported_evaluations"] == 97
    assert "not full-DES ret_excel_visible_v1" in reconstructed["metric_scope"]


def test_registry_relabeling_cannot_manufacture_a_terminal_boundary():
    registry = deepcopy(load("approach_registry.json"))
    ledger = load("boundary_family_proof_ledger.json")
    for row in registry["approaches"]:
        row["state"] = "falsified_current_contract"

    validation = validate_boundary_family_proof_ledger(registry, ledger, root=ROOT)

    assert validation["all_families_terminal_b_eligible"] is False
    assert any(
        "registry_state_mismatch" in failure.get("reasons", [])
        for failure in validation["failures"]
    )


def test_terminal_label_without_complete_hashed_proof_is_rejected():
    registry = load("approach_registry.json")
    ledger = deepcopy(load("boundary_family_proof_ledger.json"))
    row = ledger["families"][0]
    row["terminal_b_eligible"] = True
    row["closure_kind"] = "exact_global_zero"
    row["comparator_complete"] = True
    row["resource_matching_certified"] = True
    row["scope_closes_registered_family"] = True
    row["unresolved_extension_or_domain_fact"] = False
    row["proof_artifacts"] = []
    ledger["summary"]["terminal_b_eligible_families"] = 1
    ledger["summary"]["nonterminal_families"] = 16

    validation = validate_boundary_family_proof_ledger(registry, ledger, root=ROOT)

    assert validation["all_families_terminal_b_eligible"] is False
    family_failure = next(
        failure for failure in validation["failures"]
        if failure.get("family_id") == row["family_id"]
    )
    assert "missing_content_addressed_proof" in family_failure["reasons"]


def test_reproducibility_manifest_hashes_every_listed_artifact_and_source():
    manifest = load("reproducibility_manifest.json")
    assert manifest["paper2_confirmed"] is False
    assert manifest["paper3_authorized"] is False
    assert manifest["scientific_status"] == (
        "HOLD_CANONICAL_V2_RESCORE_AND_DOMAIN_CONFIRMATION_REQUIRED"
    )
    for relative, expected in manifest["artifact_hashes"].items():
        path = Path(relative)
        if not path.is_absolute():
            path = ROOT / path
        assert path.is_file(), relative
        assert sha256(path) == expected, relative
    for source, expected in manifest["source_hashes"].items():
        path = Path(source)
        assert path.is_file(), source
        assert sha256(path) == expected, source
    coverage = manifest["required_set_coverage"]
    assert coverage["passed"] is True
    assert coverage["missing_required_artifacts"] == []
    assert coverage["missing_required_sources"] == []
    assert coverage["required_artifact_count"] >= 27
    assert coverage["required_source_count"] == 8
    assert {
        "/Users/thom/Downloads/Raw_data1+Re.xlsx",
        "/Users/thom/Downloads/Raw_data2+Re.xlsx",
        "/Users/thom/Downloads/Rsult_1.xlsx",
    }.issubset(manifest["source_hashes"])


def test_domain_blocked_families_have_questions_and_state_change_evidence():
    registry = load("approach_registry.json")
    blocked = [row for row in registry["approaches"] if row["state"].startswith("blocked")]
    assert blocked
    for row in blocked:
        assert row["exact_garrido_question"].endswith("?")
        assert row["evidence_required_to_change_state"]


def test_current_action_absence_is_not_mislabeled_as_global_impossibility():
    registry = load("approach_registry.json")
    assert "universal boundary" in registry["searched_envelope"]["not_claimed"].lower()
    rows = {row["family_id"]: row for row in registry["approaches"]}
    product_mix = rows["multi_ration_product_mix_setup_substitution"]
    reservation = rows["regime_lead_time_advance_transport_reservation"]
    assert product_mix["state"] == "blocked_domain_fact"
    assert reservation["state"] == "blocked_frozen_contract_hpi_region_fail"
    assert reservation["current_physics_ceiling"]["h_pi"] == 0.0
    assert reservation["program_m_extension"]["h_pi"] is None
    assert reservation["program_m_extension"]["h_obs"] is None


def test_mission_loadout_is_new_but_current_kernel_null():
    registry = load("approach_registry.json")
    rows = {row["family_id"]: row for row in registry["approaches"]}
    loadout = rows["mission_loadout_carried_autonomy_allocation"]
    assert loadout["state"] == "blocked_domain_fact"
    assert loadout["current_physics_ceiling"]["kind"] == "action_absent_exact_null"
    assert loadout["current_physics_ceiling"]["h_pi"] == 0.0
    assert loadout["current_physics_ceiling"]["h_obs"] == 0.0
    assert loadout["exact_garrido_question"].endswith("?")


def test_canonical_ret_ceiling_does_not_assume_global_unit_bound():
    review = (SEARCH / "primary_source_literature_review.md").read_text()
    assert "Do not assume per-row ReT is capped at one" in review


def test_prelearner_contract_keeps_learning_closed_until_every_gate_passes():
    contract = load("prelearner_contract.json")
    registry = load("approach_registry.json")
    statuses = contract["candidate_entry_status"]
    assert contract["status"] == "NO_CURRENT_CANDIDATE_ELIGIBLE"
    assert contract["primary_endpoint"].startswith("ret_excel_request_snapshot_v2")
    assert contract["metric_readiness"] == {
        "status": "HOLD_CANONICAL_V2_RESCORE_AND_DOMAIN_CONFIRMATION_REQUIRED",
        "same_timestamp_garrido_confirmation": False,
        "prior_visible_v1_results_restored": False,
        "paper2_null_positive_or_ceiling_authorized": False,
    }
    assert contract["learner_authorized"] is False
    assert contract["paper3_authorized"] is False
    assert len(contract["mandatory_gates"]) == 15
    assert set(statuses) == {row["family_id"] for row in registry["approaches"]}
    assert statuses["integrated_production_maintenance_routing_recovery_resource"] == "active_for_bound"


def test_historical_c12_is_bounded_evidence_not_current_paper3_authorization():
    supersession = load("paper3_claim_supersession.json")
    validation = validate_paper3_claim_supersession(supersession, root=ROOT)

    assert validation["passed"] is True
    assert validation["failures"] == []
    assert validation["hashes_checked"] == 6
    assert validation["historical_claim_preserved"] is True
    assert validation["paper3_authorized"] is False
    assert supersession["historical_claim"]["authorization_transferable_to_current_paper3"] is False
    assert supersession["current_task_authorization_gate"]["canonical_endpoint"] == (
        "ret_excel_request_snapshot_v2"
    )
    assert supersession["current_task_authorization_gate"][
        "paper2_learned_adaptive_value_confirmed"
    ] is False


def test_historical_supported_label_cannot_be_flipped_into_paper3_authority():
    supersession = deepcopy(load("paper3_claim_supersession.json"))
    supersession["historical_claim"][
        "authorization_transferable_to_current_paper3"
    ] = True
    supersession["effective_current_disposition"]["paper3_authorized"] = True
    supersession["effective_current_disposition"][
        "new_retained_learning_execution_authorized"
    ] = True

    validation = validate_paper3_claim_supersession(supersession, root=ROOT)

    assert validation["passed"] is False
    assert validation["paper3_authorized"] is True
    assert "historical C12 was made transferable to current Paper 3" in validation[
        "failures"
    ]
    assert "invalid effective disposition: paper3_authorized" in validation[
        "failures"
    ]
    assert (
        "invalid effective disposition: new_retained_learning_execution_authorized"
        in validation["failures"]
    )


def test_intervention_ledger_is_complete_and_claim_limited():
    registry = load("approach_registry.json")
    ledger = load("candidate_intervention_ledger.json")
    rows = {row["family_id"]: row for row in ledger["interventions"]}
    assert set(rows) == {row["family_id"] for row in registry["approaches"]}
    for row in rows.values():
        assert row["provenance_class"]
        assert row["units_and_conservation"]
        assert row["plausible_envelope_status"]
        assert row["null_or_falsification_regime"]
        assert row["claim_limit"]


def test_all_catalogued_decision_rights_are_routed_once():
    catalog = json.loads((ROOT / "contracts" / "decision_right_catalog_v1.json").read_text())
    coverage = load("decision_right_catalog_coverage.json")
    catalog_ids = {
        row["id"] for row in catalog["factors"] if row["class"] == "decision_right"
    }
    rows = {row["factor_id"]: row for row in coverage["rows"]}
    assert len(rows) == 32
    assert set(rows) == catalog_ids
    assert coverage["new_executable_source_native_candidate_count"] == 0
    assert coverage["exact_current_kernel_zero_count"] == 12
    assert rows["op7_release_period"]["catalog_status"] == "implemented"
    assert rows["op7_release_period"]["disposition"] == "transition_dead_configuration_field"
    assert rows["op7_release_period"]["current_kernel_h_pi_ceiling"] == 0.0
    assert rows["op6_rework_rule"]["disposition"] == (
        "transition_live_fidelity_configuration_not_adaptive_action"
    )
    assert rows["op6_rework_rule"]["current_kernel_h_pi_ceiling"] is None
    assert "op7_rop" in (ROOT / "supply_chain" / "config.py").read_text()
    assert "op7_rop" not in (ROOT / "supply_chain" / "supply_chain.py").read_text()


def test_constants_only_statistic_does_not_close_bottleneck_family():
    registry = load("approach_registry.json")
    row = next(
        item for item in registry["approaches"]
        if item["family_id"] == "integrated_production_maintenance_routing_recovery_resource"
    )
    rejection = (SEARCH / "rejected_concurrent_artifacts.md").read_text()
    assert row["state"] == "active_for_bound"
    assert row["current_physics_ceiling"]["available"] is False
    assert "lower bound" in rejection
    assert "LCB" in rejection


def test_mtr_resource_semantics_separate_budget_from_endogenous_flows():
    verdict = load("mtr_resource_semantics_verdict_20260714.json")
    registry = load("approach_registry.json")
    transition = load("garrido_family_question_state_transitions.json")
    response_ledger = (
        ROOT / "docs" / "GARRIDO_FACE_VALIDATION_RESPONSES_2026-07-13.md"
    ).read_text()
    question_text = (SEARCH / "garrido_face_validation_questions.md").read_text()

    assert verdict["named_budget"]["total_episode_hours"] == 4032
    assert {row["field"] for row in verdict["endogenous_outcome_flows_not_named_budgets_in_v1"]} == {
        "reserve_units_issued",
        "reserve_units_replenished",
        "reserve_replenishment_requests",
    }
    assert verdict["domain_validity"]["status"] == "UNANSWERED_GARRIDO_FACT"
    assert verdict["claim_boundary"]["h_pi_computed"] is False
    assert verdict["claim_boundary"]["h_obs_computed"] is False
    assert verdict["claim_boundary"]["learner_authorized"] is False
    assert verdict["claim_boundary"]["paper3_authorized"] is False

    integrated = next(
        row for row in registry["approaches"]
        if row["family_id"] == "integrated_production_maintenance_routing_recovery_resource"
    )
    assert integrated["state"] == "active_for_bound"
    assert "endogenous outcomes" in integrated["main_threat_to_validity"]
    q6 = next(
        row for row in transition["family_mappings"]
        if row["family_id"] == "integrated_production_maintenance_routing_recovery_resource"
    )
    assert any("componentwise budgeted" in item for item in q6["positive_conditions"])
    assert "vehicle-hour" in question_text
    assert "Initial 10,000-unit prepositioning" in response_ledger


def test_effect_quotient_is_not_promoted_as_a_completed_ceiling():
    result = json.loads(
        (ROOT / "results" / "paper2_bottleneck" / "effect_quotient_audit.json").read_text()
    )
    assert result["scientific_status"] == "EXACT_ACCELERATION_DESIGN_NOT_H_PI_RESULT"
    assert result["totals"]["effect_quotient_des_runs"] == 88_684_583
    assert result["not_done"]


def test_full_horizon_bound_contract_excludes_algorithm_development_tape():
    contract = json.loads(
        (ROOT / "contracts" / "paper2_bottleneck_full_horizon_bound_v1.json").read_text()
    )
    assert contract["status"] == "RETROSPECTIVE_BOUND_PROTOCOL_FROZEN_BEFORE_FULL_EXECUTION"
    assert contract["complete_open_loop_family"]["effective_calendar_count"] == 11_184_811
    assert contract["complete_open_loop_family"]["locked_bound"]["algorithm_development_seed_excluded"] == 1_110_001
    assert contract["complete_open_loop_family"]["locked_bound"]["n_tapes"] == 119
    assert contract["inference"]["learner_authorized"] is False
    assert contract["inference"]["paper3_authorized"] is False


def test_primary_bound_v2_is_frozen_before_24_week_results_and_fail_closed():
    contract = json.loads(
        (ROOT / "contracts" / "paper2_bottleneck_primary_bound_v2.json").read_text()
    )
    assert contract["status"] == (
        "RETROSPECTIVE_PRIMARY_BOUND_PROTOCOL_FROZEN_BEFORE_ANY_24_WEEK_RESULT"
    )
    assert contract["physics"]["effective_calendar_count"] == 11_184_811
    assert contract["seed_blocks"]["algorithm_development_excluded"] == [1_110_001]
    assert contract["seed_blocks"]["calibration"]["n"] == 60
    assert contract["seed_blocks"]["locked_bound"]["n"] == 119
    assert contract["acceleration_proof"]["required_key_schema"].endswith("_v4")
    assert "UCB95 is strictly below 0.01" in contract["decision_rules"]["boundary_close"]
    assert contract["decision_rules"]["learner_authorized"] is False
    assert contract["decision_rules"]["paper3_authorized"] is False


def test_loose_metric_bound_is_not_mislabeled_as_closing_evidence():
    result = json.loads(
        (ROOT / "results" / "paper2_bottleneck" / "loose_canonical_upper_bound.json").read_text()
    )
    assert result["scientific_status"] == "VALID_EMPIRICAL_TAPE_BOUND_TOO_LOOSE_TO_CLOSE"
    assert result["summary"]["tapes_with_upper_gap_at_most_0_01"] == 0
    assert result["summary"]["empirical_upper_gap_ci95"][0] > 1.0


def test_memoryless_signal_mapping_screen_selects_constant_M():
    result = json.loads(
        (ROOT / "results" / "paper2_bottleneck" / "signal_mapping_audit.json").read_text()
    )
    assert result["calibration"]["candidate_count"] == 27
    assert result["calibration"]["selected_mapping_equipment_transport_mission"] == "MMM"
    assert result["locked"]["ret_minus_constant_M_ci95"] == [0.0, 0.0, 0.0]


def test_concurrent_terminal_boundary_claim_is_superseded():
    audit = load("concurrent_boundary_commit_audit.json")
    certificate = json.loads(
        (ROOT / "results" / "paper2_search" / "boundary_certificate.json").read_text()
    )
    atlas = json.loads(
        (ROOT / "results" / "paper2_search" / "voi_ceiling_atlas.json").read_text()
    )
    seed_ledger_text = (
        ROOT / "results" / "paper2_search" / "seed_burn_ledger.json"
    ).read_text()
    corrected_seed_ledger = load("seed_burn_ledger_correction.json")
    registry = load("approach_registry.json")
    integrated = next(
        row for row in registry["approaches"]
        if row["family_id"] == "integrated_production_maintenance_routing_recovery_resource"
    )
    assert certificate["terminal_return"] == (
        "RETRACTED_PENDING_EXACT_MTR_BOUND_AND_DOMAIN_FACTS"
    )
    assert certificate["scientific_status"].startswith("NONTERMINAL_LEGACY")
    assert certificate["program_l_route_recourse_followup"]["verdict"].startswith(
        "REJECT_PROGRAM_L_TERMINAL_REAFFIRMATION"
    )
    assert certificate["evidence_budget_discharged"]["families_with_quantitative_bounds"].startswith(">=3")
    assert "NOT full Op1-Op13 DES" in atlas["contract"]
    assert "7_200_001+" in atlas["design"]
    assert "7200001" not in seed_ledger_text
    atlas_burn = next(
        row for row in corrected_seed_ledger["blocks"]
        if row["label"] == "concurrent_stylized_voi_atlas"
    )
    assert atlas_burn["range"] == "7200001-7299999"
    assert atlas_burn["status"] == "burned_conservative_reservation"
    assert audit["verdict"] == "SUPERSEDED_NONTERMINAL_INSUFFICIENT_FOR_RETURN_B"
    assert audit["terminal_boundary_certified"] is False
    assert len(audit["failed_terminal_B_requirements"]) >= 10
    assert integrated["state"] == "active_for_bound"
    assert integrated["current_physics_ceiling"]["available"] is False


def test_concurrent_atlas_positive_cells_fail_canonical_replay():
    result = json.loads(
        (ROOT / "results" / "paper2_search" / "voi_ceiling_atlas_corrective_audit.json").read_text()
    )
    assert result["positive_cell_count_in_source"] == 2
    assert result["all_cell_summary"]["n_cells"] == 64
    assert result["all_cell_summary"]["stored_statistic_reproduced_cells"] == 64
    assert result["all_cell_summary"]["unguarded_sparse_visible_metric_H_PI_exact_zero_cells"] == 64
    assert result["all_cell_summary"]["unguarded_sparse_visible_metric_tested_belief_positive_cells"] == 0
    assert all(row["stored_statistic_reproduced"] for row in result["positive_cells"])
    assert all(
        row["full_order_formula"]["tested_belief_minus_static"] >= 0.01
        and row["full_order_formula"]["belief_minus_static_lost_ci95"][1] > 0
        and row["full_order_formula"]["belief_lost_better_tapes"] == 0
        for row in result["positive_cells"]
    )
    assert all(
        row["visible_ledger"]["H_PI"] == 0.0
        and row["visible_ledger"]["best_static_sequence"] == ["HOLD"] * 4
        and row["visible_ledger"]["best_static_mean_lost_orders"] == 48.0
        and row["visible_ledger"]["best_static_mean_visible_rows"] == 0.0
        and row["visible_ledger"]["project_H_PI_valid"] is False
        for row in result["positive_cells"]
    )
    assert result["r22_inertness"]["maximum_score_differences"] == {
        "full_order_formula": 0.0,
        "visible_ledger": 0.0,
    }
    assert result["terminal_boundary_supported"] is False
