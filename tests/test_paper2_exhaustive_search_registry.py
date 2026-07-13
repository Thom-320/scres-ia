import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SEARCH = ROOT / "research" / "paper2_exhaustive_search"


def load(name: str):
    return json.loads((SEARCH / name).read_text())


def test_failure_taxonomy_is_unique_and_keeps_corrective_k3_truth():
    taxonomy = load("phase0_failure_taxonomy.json")
    rows = {row["family_id"]: row for row in taxonomy["decision_families"]}
    assert len(rows) == len(taxonomy["decision_families"])
    assert set("ABC").issubset({family_id[0] for family_id in rows})
    assert rows["K3_budgeted_replenishment"]["final_verdict"] == (
        "RETRACT_K3_ADAPTIVE_AND_NEURAL_CLAIMS_STATIC_PERIOD8_CONFOUND"
    )
    assert rows["K3_budgeted_replenishment"]["retained_value"] is None


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
    assert reservation["state"] == "blocked_domain_fact"
    assert reservation["current_physics_ceiling"]["h_pi"] == 0.0


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
    assert contract["learner_authorized"] is False
    assert contract["paper3_authorized"] is False
    assert len(contract["mandatory_gates"]) == 15
    assert set(statuses) == {row["family_id"] for row in registry["approaches"]}
    assert statuses["integrated_production_maintenance_routing_recovery_resource"] == "active_for_bound"


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
    assert coverage["exact_current_kernel_zero_count"] == 13
    assert rows["op7_release_period"]["catalog_status"] == "implemented"
    assert rows["op7_release_period"]["disposition"] == "transition_dead_configuration_field"
    assert rows["op7_release_period"]["current_kernel_h_pi_ceiling"] == 0.0
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
    assert contract["acceleration_proof"]["required_key_schema"].endswith("_v2")
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
