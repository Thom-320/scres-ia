#!/usr/bin/env python3
"""Fail-closed verifier for the Paper-2 evidence and approach registries.

This verifier does not turn an unbounded researcher-extension class into a
scientific impossibility result.  It checks that the versioned evidence is
internally consistent, that superseded positive claims are not promoted, and
that unresolved/domain-blocked families remain visibly nonterminal.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
SEARCH = ROOT / "research" / "paper2_exhaustive_search"

SOURCE_HASHES = {
    Path("/Users/thom/Downloads/Raw_data1+Re.xlsx"):
        "30b88c9b9fe68ef527dbfcc70d8e653ea7bd152ab891b3fc0ecf53cb6f043486",
    Path("/Users/thom/Downloads/Raw_data2+Re.xlsx"):
        "4bd462771fefff16fc5666a851256b3780198d474832dec1423c0b6f94be86b0",
    Path("/Users/thom/Downloads/Rsult_1.xlsx"):
        "1901f683f6014cf75237c17233b8eba04f541b956f2d19dcecf2edc00e83b00a",
    Path("/Users/thom/Downloads/garrido et al 2024 factory resilience.pdf"):
        "1260863dc295232faf24b820e1f67d53f25f81ffa2d221f7ef02a02310519c43",
    Path("/Users/thom/Downloads/v.0_neuralNet-scres.docx"):
        "b111070a05c8f4d1afa058454138bed9b4b74900ab87eaaf6eb5186b6e8293f2",
    Path("/Users/thom/Downloads/v.0_neuralNet-scres.pdf"):
        "521b12770e94f3e70c4c88ce1e38613f4e0aad3e1dab114632c9c89dbfad182d",
    Path("/Users/thom/Library/CloudStorage/GoogleDrive-chisicathomas@gmail.com/My Drive/Supernote/Document/20_RESEARCH/PhD-Papers/garrido2024 scres+AI.pdf"):
        "3e3bc8f82e20b891ee163fb8a035dd37be4312fa11f58dde77452dc1bb903ae6",
    Path("/Users/thom/Library/CloudStorage/GoogleDrive-chisicathomas@gmail.com/My Drive/Archive/Misc_Unsorted/Unsorted/WRAP_Theses_Garrido_Rios_2017.pdf"):
        "de9192d233b0c728ece6156b754fc64543146868121358b8a95c73b3edaa55cf",
}


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def close(actual: float, expected: float, tol: float = 1e-12) -> bool:
    return math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tol)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=SEARCH / "boundary_verification.json",
    )
    args = parser.parse_args()

    checks: list[dict[str, Any]] = []

    def record(name: str, passed: bool, detail: Any) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    taxonomy = load(SEARCH / "phase0_failure_taxonomy.json")
    registry = load(SEARCH / "approach_registry.json")
    prelearner = load(SEARCH / "prelearner_contract.json")
    interventions = load(SEARCH / "candidate_intervention_ledger.json")
    decision_coverage = load(SEARCH / "decision_right_catalog_coverage.json")
    decision_catalog = load(ROOT / "contracts" / "decision_right_catalog_v1.json")

    record("taxonomy_schema", taxonomy.get("schema_version") == "paper2_failure_taxonomy_v1",
           taxonomy.get("schema_version"))
    family_ids = [row["family_id"] for row in taxonomy["decision_families"]]
    record("taxonomy_unique_families", len(family_ids) == len(set(family_ids)), len(family_ids))
    record("taxonomy_covers_A_through_K3", len(family_ids) >= 17,
           {"n": len(family_ids), "first": family_ids[:3], "last": family_ids[-3:]})

    approach_ids = {row["family_id"] for row in registry["approaches"]}
    intervention_ids = {row["family_id"] for row in interventions["interventions"]}
    record(
        "intervention_ledger_covers_registry",
        approach_ids == intervention_ids,
        {"approaches": len(approach_ids), "interventions": len(intervention_ids)},
    )
    record(
        "prelearner_gate_is_fail_closed",
        prelearner["status"] == "NO_CURRENT_CANDIDATE_ELIGIBLE"
        and prelearner["learner_authorized"] is False
        and prelearner["paper3_authorized"] is False
        and set(prelearner["candidate_entry_status"]) == approach_ids,
        {
            "status": prelearner["status"],
            "mandatory_gates": len(prelearner["mandatory_gates"]),
            "learner_authorized": prelearner["learner_authorized"],
        },
    )
    catalog_decision_ids = {
        row["id"] for row in decision_catalog["factors"]
        if row["class"] == "decision_right"
    }
    coverage_ids = {row["factor_id"] for row in decision_coverage["rows"]}
    coverage_evidence_paths = {
        evidence
        for row in decision_coverage["rows"]
        for evidence in row["evidence"]
    }
    op7_release = next(
        row for row in decision_coverage["rows"]
        if row["factor_id"] == "op7_release_period"
    )
    record(
        "decision_right_catalog_is_exhaustively_routed",
        decision_coverage["decision_right_count"] == 32
        and decision_coverage["all_decision_rights_covered_once"] is True
        and decision_coverage["new_executable_source_native_candidate_count"] == 0
        and decision_coverage["exact_current_kernel_zero_count"] == 13
        and coverage_ids == catalog_decision_ids
        and all((ROOT / path).exists() for path in coverage_evidence_paths)
        and op7_release["catalog_status"] == "implemented"
        and op7_release["disposition"] == "transition_dead_configuration_field"
        and op7_release["current_kernel_h_pi_ceiling"] == 0.0
        and "op7_rop" in (ROOT / "supply_chain" / "config.py").read_text()
        and "op7_rop" not in (ROOT / "supply_chain" / "supply_chain.py").read_text(),
        {
            "catalog_count": len(catalog_decision_ids),
            "coverage_count": len(coverage_ids),
            "exact_current_kernel_zero_count": decision_coverage["exact_current_kernel_zero_count"],
            "new_executable_source_native_candidate_count": decision_coverage["new_executable_source_native_candidate_count"],
            "op7_release_period": op7_release,
        },
    )

    source_details = {}
    source_ok = True
    for path, expected in SOURCE_HASHES.items():
        actual = sha256(path) if path.exists() else None
        source_details[str(path)] = {"expected": expected, "actual": actual}
        source_ok &= actual == expected
    record("source_hashes", source_ok, source_details)

    metric_audit = load(SEARCH / "metric_governance_audit.json")
    metric_lock = metric_audit["canonical_endpoint"]["fresh_reaudit"]
    record(
        "canonical_metric_and_cd_claim_boundary_locked",
        metric_audit["status"]
        == "CANONICAL_RET_LOCKED_CD_SECONDARY_ONLY_FOR_CURRENT_PAPER2"
        and metric_audit["canonical_endpoint"]["contract_id"]
        == "ret_excel_visible_v1"
        and metric_lock["raw_cf_sheets"] == 20
        and metric_lock["formula_rows"] == 47_546
        and metric_lock["mismatches"] == 0
        and metric_lock["max_abs_diff"] == 0.0
        and metric_audit["paper2_authorization"]["learner_authorized_by_cd"]
        is False
        and metric_audit["paper2_authorization"]["paper3_authorized"] is False,
        {
            "status": metric_audit["status"],
            "canonical_contract": metric_audit["canonical_endpoint"][
                "contract_id"
            ],
            "fresh_reaudit": metric_lock,
            "cd_current_paper2": metric_audit["metric_decision"]["current_paper2"],
        },
    )

    ancestor = "ef6b53b7cac9c1cbdcdc4347a31c8300d1941fc0"
    is_ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, "HEAD"],
        cwd=ROOT,
        check=False,
    ).returncode == 0
    record("k3_corrective_commit_is_ancestor", is_ancestor,
           {"corrective_commit": ancestor, "head": git("rev-parse", "HEAD")})

    k3_path = ROOT / "results" / "k3" / "open_loop_confound_audit.json"
    k3 = load(k3_path)
    k3_ok = (
        k3["verdict"] == "RETRACT_K3_ADAPTIVE_AND_NEURAL_CLAIMS_STATIC_PERIOD8_CONFOUND"
        and k3["paper2_adaptive_confirmed"] is False
        and k3["paper3_neural_retention_authorized"] is False
        and k3["ppo_seed0_unique_test_sequences"] == 1
        and k3["ppo_seed0_modal_sequence"] == [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.0, 0.0]
        and k3["ppo_seed0_minus_fixed_ret"] == [0.0, 0.0, 0.0]
        and k3["learner_test_6900001_6900120"]["fixed_minus_mpc_ordered_D0"] == [0.0, 0.0, 0.0]
    )
    record("k3_static_period8_confound", k3_ok,
           {"sha256": sha256(k3_path), "verdict": k3["verdict"]})

    track_b = load(ROOT / "outputs" / "experiments" /
                   "track_b_same_contract_challenge_2026-07-10" / "summary.json")
    track_b_primary = track_b["ret_excel"]["canonical_joint_minus_best_full_static"]
    record(
        "track_b_same_contract_reversal",
        track_b["stop_rule"]["passed"] is False
        and track_b_primary["two_way_ci95"][1] < 0
        and track_b_primary["tapes_positive"] == 2,
        track_b_primary,
    )

    program_h = load(ROOT / "results" / "program_h" / "bound" / "verdict.json")
    h_bound = program_h["certified_upper_bound_J_PI_minus_ABAB"]
    record(
        "program_h_bound_remains_above_mcid",
        close(h_bound["mean"], 0.01641384654689033)
        and h_bound["ci95"][1] > 0.01,
        h_bound,
    )

    program_i = load(ROOT / "results" / "program_i" / "branching" / "verdict.json")
    i_transport = program_i["families"]["transport"]["horizons"]["8"]
    record(
        "program_i_resource_equal_transport_tiny",
        i_transport["gates"]["resource_equivalence"] is True
        and i_transport["oracle_delta"]["ci95"][1] < 0.01,
        i_transport["oracle_delta"],
    )

    maintenance = load(ROOT / "results" / "paper2_maintenance" / "terminal_verdict.json")
    record(
        "maintenance_prelearner_stop",
        maintenance["verdict"] == "STOP_NO_OBSERVABLE_MAINTENANCE_HEADROOM"
        and maintenance["paper2_learner_authorized"] is False
        and maintenance["paper3_authorized"] is False
        and maintenance["central_screen"]["oracle_ret_delta_ci95"][1] < 0.01,
        maintenance["central_screen"],
    )

    bottleneck = load(ROOT / "results" / "paper2_bottleneck" /
                      "locked_confirmation" / "verdict.json")
    record(
        "bottleneck_observable_policy_stop",
        bottleneck["verdict"] == "STOP_NO_ADAPTIVE_BOTTLENECK_VALUE"
        and bottleneck["ppo_trained"] is False
        and bottleneck["gates"]["equal_team_hours"] is True
        and bottleneck["gates"]["crn"] is True,
        bottleneck["signal_policy_minus_best_constant"],
    )
    bottleneck_corrective = load(
        ROOT / "results" / "paper2_bottleneck" / "corrective_completeness_audit.json"
    )
    effect_quotient = load(
        ROOT / "results" / "paper2_bottleneck" / "effect_quotient_audit.json"
    )
    loose_bound = load(
        ROOT / "results" / "paper2_bottleneck" / "loose_canonical_upper_bound.json"
    )
    signal_mapping = load(
        ROOT / "results" / "paper2_bottleneck" / "signal_mapping_audit.json"
    )
    bottleneck_bound_contract = load(
        ROOT / "contracts" / "paper2_bottleneck_full_horizon_bound_v1.json"
    )
    record(
        "bottleneck_family_bound_gap_is_explicit",
        bottleneck_corrective["status"] == "ACTIVE_FOR_BOUND_NOT_TERMINAL"
        and bottleneck_corrective["calendar_frontier"]["effective_full_horizon_calendar_count"] == 11_184_811
        and bottleneck_corrective["calendar_frontier"]["exact_bound_available"] is False
        and bottleneck_corrective["component_resource_audit"]["total_team_hours_equal"] is True
        and bottleneck_corrective["component_resource_audit"]["allocation_destination_hours_are_separate_contract_resources"] is False
        and bottleneck_corrective["component_resource_audit"]["reserve_resource_semantics_resolved"] is False
        and bottleneck_corrective["null_physics"]["exact_selected_metric_equivalence"] is True
        and bottleneck_corrective["null_physics"]["n_retrospective_tapes"] == 3
        and bottleneck_corrective["trajectory_audit"]["locked_unique_signal_sequences"] == 120,
        {
            "calendar_frontier": bottleneck_corrective["calendar_frontier"],
            "component_resource_audit": bottleneck_corrective["component_resource_audit"],
            "null_physics": bottleneck_corrective["null_physics"],
        },
    )
    record(
        "bottleneck_exact_effect_quotient_is_only_an_acceleration",
        effect_quotient["scientific_status"] == "EXACT_ACCELERATION_DESIGN_NOT_H_PI_RESULT"
        and effect_quotient["full_calendar_count"] == 11_184_811
        and effect_quotient["splits"]["calibration"]["distinct_effect_executions"] == 30_765_821
        and effect_quotient["splits"]["locked"]["distinct_effect_executions"] == 57_918_762
        and effect_quotient["totals"]["effect_quotient_des_runs"] == 88_684_583
        and effect_quotient["collision_validation"]["selected_outcomes_exactly_equal"] is True
        and bool(effect_quotient["not_done"]),
        {
            "scientific_status": effect_quotient["scientific_status"],
            "totals": effect_quotient["totals"],
            "not_done": effect_quotient["not_done"],
        },
    )
    record(
        "bottleneck_full_horizon_bound_protocol_is_frozen_fail_closed",
        bottleneck_bound_contract["status"] == "RETROSPECTIVE_BOUND_PROTOCOL_FROZEN_BEFORE_FULL_EXECUTION"
        and bottleneck_bound_contract["complete_open_loop_family"]["effective_calendar_count"] == 11_184_811
        and bottleneck_bound_contract["complete_open_loop_family"]["locked_bound"]["algorithm_development_seed_excluded"] == 1_110_001
        and bottleneck_bound_contract["complete_open_loop_family"]["locked_bound"]["n_tapes"] == 119
        and bottleneck_bound_contract["exact_acceleration_requirements"]["failure_rule"].startswith("Any state-key collision")
        and bottleneck_bound_contract["inference"]["learner_authorized"] is False
        and bottleneck_bound_contract["inference"]["paper3_authorized"] is False,
        {
            "contract_id": bottleneck_bound_contract["contract_id"],
            "locked_bound": bottleneck_bound_contract["complete_open_loop_family"]["locked_bound"],
            "state_merge_rule": bottleneck_bound_contract["exact_acceleration_requirements"]["state_merge_rule"],
        },
    )
    record(
        "bottleneck_cheap_canonical_bound_is_valid_but_vacuous",
        loose_bound["scientific_status"] == "VALID_EMPIRICAL_TAPE_BOUND_TOO_LOOSE_TO_CLOSE"
        and loose_bound["summary"]["n_orders_unique"] == [143]
        and loose_bound["summary"]["tapes_with_upper_gap_at_most_0_01"] == 0
        and loose_bound["summary"]["empirical_upper_gap_ci95"][0] > 1.0,
        {
            "scientific_status": loose_bound["scientific_status"],
            "summary": loose_bound["summary"],
        },
    )
    record(
        "bottleneck_memoryless_signal_mapping_reduces_to_constant_M",
        signal_mapping["calibration"]["candidate_count"] == 27
        and signal_mapping["calibration"]["selected_mapping_equipment_transport_mission"] == "MMM"
        and signal_mapping["locked"]["ret_minus_constant_M_ci95"] == [0.0, 0.0, 0.0]
        and "not optimized H_obs" in signal_mapping["scientific_use"],
        {
            "selected_mapping": signal_mapping["calibration"]["selected_mapping_equipment_transport_mission"],
            "locked": signal_mapping["locked"],
        },
    )

    # A concurrently generated local artifact incorrectly labels the maximum of
    # three constant policies as a full-horizon PI ceiling.  That statistic is a
    # lower bound on the weekly oracle, and its bootstrap LCB cannot certify an
    # upper ceiling.  Keep it outside the evidence allowlist and prove that it
    # did not change the registry state.
    invalid_ceiling = SEARCH / "bottleneck_pi_ceiling.json"
    invalid_detail: dict[str, Any] = {
        "present": invalid_ceiling.exists(),
        "reason": "best-of-three constants is not an upper bound on the 24-week oracle; LCB<0.01 is not a ceiling test",
    }
    if invalid_ceiling.exists():
        invalid_detail.update({
            "sha256": sha256(invalid_ceiling),
            "local_unverified_verdict": load(invalid_ceiling).get("verdict"),
        })
    integrated = next(
        row for row in registry["approaches"]
        if row["family_id"] == "integrated_production_maintenance_routing_recovery_resource"
    )
    record(
        "invalid_constants_only_PI_claim_is_excluded",
        integrated["state"] == "active_for_bound"
        and integrated["current_physics_ceiling"]["available"] is False,
        invalid_detail,
    )

    # Commit a91890bf was pushed concurrently with a terminal Boundary-B label.
    # The tracked certificate is now explicitly retracted; its underlying atlas
    # remains useful only as corrective exploratory evidence.
    concurrent_audit = load(SEARCH / "concurrent_boundary_commit_audit.json")
    concurrent_certificate = load(
        ROOT / "results" / "paper2_search" / "boundary_certificate.json"
    )
    concurrent_atlas = load(
        ROOT / "results" / "paper2_search" / "voi_ceiling_atlas.json"
    )
    concurrent_atlas_corrective = load(
        ROOT / "results" / "paper2_search" / "voi_ceiling_atlas_corrective_audit.json"
    )
    concurrent_seed_ledger_path = (
        ROOT / "results" / "paper2_search" / "seed_burn_ledger.json"
    )
    concurrent_seed_ledger = load(concurrent_seed_ledger_path)
    concurrent_seed_ledger_text = concurrent_seed_ledger_path.read_text()
    corrected_seed_ledger = load(SEARCH / "seed_burn_ledger_correction.json")
    atlas_burn = next(
        row for row in corrected_seed_ledger["blocks"]
        if row["label"] == "concurrent_stylized_voi_atlas"
    )
    missing_referenced_files = [
        str(path.relative_to(ROOT))
        for path in (
            ROOT / "results" / "paper2_search" / "voi_ceiling_atlas_figure.py",
            ROOT / "tests" / "test_voi_ceiling_atlas.py",
        )
        if not path.exists()
    ]
    record(
        "concurrent_a918_terminal_boundary_is_superseded",
        concurrent_audit["audited_commit"].startswith("a91890bf")
        and concurrent_audit["verdict"] == "SUPERSEDED_NONTERMINAL_INSUFFICIENT_FOR_RETURN_B"
        and concurrent_audit["terminal_boundary_certified"] is False
        and concurrent_certificate["terminal_return"]
        == "RETRACTED_PENDING_EXACT_MTR_BOUND_AND_DOMAIN_FACTS"
        and concurrent_certificate["scientific_status"].startswith(
            "NONTERMINAL_LEGACY"
        )
        and concurrent_certificate["program_l_route_recourse_followup"][
            "verdict"
        ].startswith("REJECT_PROGRAM_L_TERMINAL_REAFFIRMATION")
        and concurrent_certificate["evidence_budget_discharged"]["families_with_quantitative_bounds"].startswith(">=3")
        and "NOT full Op1-Op13 DES" in concurrent_atlas["contract"]
        and "7_200_001+" in concurrent_atlas["design"]
        and "7200001" not in concurrent_seed_ledger_text
        and concurrent_seed_ledger["known_virgin_blocks_from_source_of_truth"]["k3_confirmation"] == "6800001-6800120"
        and concurrent_seed_ledger["known_virgin_blocks_from_source_of_truth"]["k3_learner_test"] == "6900001-6900120"
        and atlas_burn["range"] == "7200001-7299999"
        and atlas_burn["status"] == "burned_conservative_reservation"
        and set(missing_referenced_files) == {
            "results/paper2_search/voi_ceiling_atlas_figure.py",
            "tests/test_voi_ceiling_atlas.py",
        }
        and integrated["state"] == "active_for_bound"
        and integrated["current_physics_ceiling"]["available"] is False,
        {
            "concurrent_claim": concurrent_certificate["terminal_return"],
            "reported_quantitative_bounds": concurrent_certificate["evidence_budget_discharged"]["families_with_quantitative_bounds"],
            "atlas_contract": concurrent_atlas["contract"],
            "opened_unledgered_seed_block": "7200001+",
            "incorrectly_listed_known_virgin_k3_blocks": {
                "confirmation": concurrent_seed_ledger["known_virgin_blocks_from_source_of_truth"]["k3_confirmation"],
                "learner_test": concurrent_seed_ledger["known_virgin_blocks_from_source_of_truth"]["k3_learner_test"],
            },
            "corrected_conservative_burn": atlas_burn,
            "missing_referenced_files": missing_referenced_files,
            "governing_status": concurrent_audit["governing_current_status"],
        },
    )
    program_g_text = (ROOT / "supply_chain" / "program_g.py").read_text()
    simulate_orders_source = program_g_text.split("def simulate_orders", 1)[1].split(
        "def ret_order_metrics", 1
    )[0]
    ret_order_source = program_g_text.split("def ret_order_metrics", 1)[1].split(
        "# Cobb-Douglas", 1
    )[0]
    headroom_source = (ROOT / "supply_chain" / "headroom_sensitivity.py").read_text()
    positive_atlas_cells = [
        cell for cell in concurrent_atlas["cells"] if cell["H_obs"] >= 0.01
    ]
    atlas_cell_keys = set().union(*(cell.keys() for cell in concurrent_atlas["cells"]))
    record(
        "concurrent_atlas_is_not_a_canonical_Hobs_ceiling",
        "compute_order_level_ret_excel_formula" in ret_order_source
        and "compute_order_level_ret_excel_visible_ledger" not in ret_order_source
        and "tape.r22" not in simulate_orders_source
        and "obs = np.array([_ret(t, _belief_policy(t))" in headroom_source
        and len(positive_atlas_cells) == 2
        and not ({"lost", "worst_cssu", "quantity_ret", "tail_risk", "resource_ledger"} & atlas_cell_keys),
        {
            "metric_used": "compute_order_level_ret_excel_formula_not_visible_ledger",
            "r22_consumed_by_scored_simulate_orders": False,
            "Hobs_estimator": "one_belief_policy_tested_delta_not_maximum_or_ceiling",
            "cells_Hobs_ge_0_01": len(positive_atlas_cells),
            "missing_guardrail_fields": sorted(
                {"lost", "worst_cssu", "quantity_ret", "tail_risk", "resource_ledger"}
                - atlas_cell_keys
            ),
        },
    )
    corrected_positive = concurrent_atlas_corrective["positive_cells"]
    corrected_atlas_summary = concurrent_atlas_corrective["all_cell_summary"]
    record(
        "concurrent_atlas_positive_cells_fail_lost_guardrail_and_visible_optimum_is_degenerate",
        concurrent_atlas_corrective["scientific_status"] == "EXPLORATORY_ATLAS_NOT_CANONICAL_HOBS_OR_BOUND"
        and corrected_atlas_summary["n_cells"] == 64
        and corrected_atlas_summary["stored_statistic_reproduced_cells"] == 64
        and corrected_atlas_summary["unguarded_sparse_visible_metric_H_PI_exact_zero_cells"] == 64
        and corrected_atlas_summary["unguarded_sparse_visible_metric_tested_belief_positive_cells"] == 0
        and concurrent_atlas_corrective["positive_cell_count_in_source"] == 2
        and all(row["stored_statistic_reproduced"] for row in corrected_positive)
        and all(row["full_order_formula"]["tested_belief_minus_static"] >= 0.01 for row in corrected_positive)
        and all(row["full_order_formula"]["belief_minus_static_lost_ci95"][1] > 0 for row in corrected_positive)
        and all(row["full_order_formula"]["belief_lost_better_tapes"] == 0 for row in corrected_positive)
        and all(close(row["visible_ledger"]["H_PI"], 0.0) for row in corrected_positive)
        and all(row["visible_ledger"]["best_static_sequence"] == ["HOLD"] * 4 for row in corrected_positive)
        and all(row["visible_ledger"]["best_static_mean_lost_orders"] == 48.0 for row in corrected_positive)
        and all(row["visible_ledger"]["best_static_mean_visible_rows"] == 0.0 for row in corrected_positive)
        and all(row["visible_ledger"]["project_H_PI_valid"] is False for row in corrected_positive)
        and concurrent_atlas_corrective["r22_inertness"]["r22_arrays_differ_on_at_least_one_tape"] is True
        and concurrent_atlas_corrective["r22_inertness"]["maximum_score_differences"] == {
            "full_order_formula": 0.0,
            "visible_ledger": 0.0,
        }
        and concurrent_atlas_corrective["terminal_boundary_supported"] is False,
        {
            "source_seed_pattern": concurrent_atlas_corrective["source_seed_pattern_reconstructed"],
            "all_cell_summary": corrected_atlas_summary,
            "cells": [
                {
                    "source_cell_index": row["source_cell_index"],
                    "stored_reproduced": row["stored_statistic_reproduced"],
                    "full_formula_tested_delta": row["full_order_formula"]["tested_belief_minus_static"],
                    "full_formula_lost_delta_ci95": row["full_order_formula"]["belief_minus_static_lost_ci95"],
                    "lost_guardrail_pass": row["full_order_formula"]["belief_minus_static_lost_ci95"][1] <= 0,
                    "unguarded_sparse_visible_metric_H_PI": row["visible_ledger"]["H_PI"],
                    "unguarded_best_static_lost_mean": row["visible_ledger"]["best_static_mean_lost_orders"],
                    "project_H_PI_valid": row["visible_ledger"]["project_H_PI_valid"],
                }
                for row in corrected_positive
            ],
            "r22_inertness": concurrent_atlas_corrective["r22_inertness"],
        },
    )

    states = [row["state"] for row in registry["approaches"]]
    active = [row["family_id"] for row in registry["approaches"]
              if row["state"] == "active_for_bound"]
    blocked = [row["family_id"] for row in registry["approaches"]
               if row["state"].startswith("blocked")]
    record("registry_has_no_promotable_family", registry["registry_summary"]["promotable"] == 0,
           {"states": sorted(set(states))})
    record("paper3_not_authorized", registry["registry_summary"]["paper3_authorized"] is False,
           registry["registry_summary"])
    record("unresolved_families_are_explicit", bool(active or blocked),
           {"active_for_bound": active, "blocked": blocked})

    failed = [check for check in checks if not check["passed"]]
    terminal_boundary = not failed and not active and not blocked
    if failed:
        status = "FAIL_EVIDENCE_INCONSISTENT"
    elif active:
        status = "OPEN_ACTIVE_BOUND_REQUIRED"
    elif blocked:
        status = "BOUNDARY_CONDITIONAL_ON_DOMAIN_FACTS"
    else:
        status = "TERMINAL_FINITE_ENVELOPE_BOUNDARY"

    result = {
        "schema_version": "paper2_boundary_verification_v1",
        "repository_head": git("rev-parse", "HEAD"),
        "status": status,
        "terminal_boundary": terminal_boundary,
        "paper2_confirmed": False,
        "paper3_authorized": False,
        "checks_passed": len(checks) - len(failed),
        "checks_total": len(checks),
        "active_for_bound": active,
        "blocked_domain_or_claim_families": blocked,
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
