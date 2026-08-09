from copy import deepcopy
import json
from pathlib import Path

from scripts.validate_program_x_o_scale_contract import (
    action_count,
    build_payload,
    calendar_count,
    weak_compositions,
)


ROOT = Path(__file__).resolve().parent.parent


def _contract() -> dict:
    return json.loads(
        (ROOT / "contracts/program_x_o_scale_amortized_control_v1.json").read_text(
            encoding="utf-8"
        )
    )


def test_action_space_counts_and_conserves_batch_rights() -> None:
    expected = {2: 4, 4: 20, 8: 120}
    for n_products, count in expected.items():
        actions = list(weak_compositions(3, n_products))
        assert len(actions) == count == action_count(n_products, 3)
        assert len(set(actions)) == len(actions)
        assert all(sum(action) == 3 for action in actions)


def test_calendar_complexity_is_cardinality_not_program_o_parity() -> None:
    assert calendar_count(2, 3, 8) == 65_536
    assert calendar_count(4, 3, 8) == 25_600_000_000
    assert calendar_count(8, 3, 8) == 42_998_169_600_000_000

    payload = build_payload(_contract())
    check_name = "c5_n2_has_cardinality_only_and_parity_remains_pending"
    assert check_name in payload["consistency_checks"]
    assert payload["consistency_checks"][check_name]["passed"] is True
    assert payload["claim_boundary"]["n2_cardinality_established"] is True
    assert payload["claim_boundary"]["n2_physical_parity_established"] is False
    assert all(
        "embeds_program_o" not in name
        for name in payload["consistency_checks"]
    )


def test_contract_preflight_is_eight_consistency_checks_and_opens_no_gate() -> None:
    payload = build_payload(_contract())
    assert payload["claim_status"] == (
        "DESIGN_CONSISTENCY_PASS__NO_SCIENTIFIC_GATE_OPENED"
    )
    assert payload["consistency_summary"] == {
        "all_passed": True,
        "n_computed": 8,
        "n_failed": 0,
        "failed": [],
    }
    assert payload["claim_boundary"] == {
        "n2_cardinality_established": True,
        "n2_physical_parity_established": False,
        "headroom_established": False,
        "history_value_established": False,
        "neural_premium_established": False,
        "neural_training_authorized": False,
        "fresh_seeds_opened": False,
    }


def test_architecture_decoder_planner_and_metric_roles_are_explicit() -> None:
    contract = _contract()
    architecture = contract["learner_ladder"]["primary_architecture"]
    decoder = contract["mechanism"]["action_decoder"]
    metric = contract["metric_panel"]
    claim = contract["claim_tiers"]["T2_neural_amortization"]

    assert architecture["policy_symmetry"] == (
        "permutation-equivariant over product labels"
    )
    assert architecture["value_symmetry"] == (
        "permutation-invariant over product labels"
    )
    assert decoder["nonnegative_integer"] is True
    assert decoder["sum_equals_batches_per_week"] is True
    assert "natural online cost is measured" in (
        contract["planner_roles"]["teacher_high_budget"]
    )
    assert metric["primary_physical_endpoint"]["direction"] == "higher_is_better"
    assert metric["secondary_status"] == (
        "PRESPECIFIED_REPORT_ONLY_CANNOT_PROMOTE_RESCUE_OR_BLOCK_"
        "THE_PRIMARY_PHYSICAL_CLAIM"
    )
    assert "absolute operational SLA" in claim
    assert "10x lower" in claim
    assert "cannot replace either latency condition" in claim
    assert " or DES calls" not in claim


def test_hmm_warning_iid_order_and_h4_identification_are_explicit() -> None:
    contract = _contract()
    demand = contract["demand_and_information"]
    transition = demand["latent_regime"]["transition_kernel"]
    warning = demand["warning_kernel"]

    assert demand["latent_regime"]["initial_prior"] == (
        "P(Z_0=i)=1/N for every i"
    )
    assert "(1-rho)/(N-1)" in transition["different_state"]
    assert "(1-q)/(N-1)" in warning["incorrect_label"]
    assert "q=1/N" in warning["independence_null"]
    assert demand["iid_regime_null"]["equivalent_symmetric_parameter"] == (
        "rho=1/N"
    )
    assert "byte-identical physical" in demand["h4_identification"]
    assert "RNG state" in demand["h4_identification"]
    assert build_payload(contract)["consistency_checks"][
        "c6_hmm_warning_causality_and_h4_are_explicit"
    ]["passed"] is True


def test_authorization_gates_are_branched() -> None:
    branches = _contract()["authorization_branches"]
    assert any("H_ret is not required" in row for row in branches["amortization"])
    assert any(
        "conditional-history" in row
        for row in branches["recurrent_representation"]
    )
    assert any(
        "observable headroom" in row
        for row in branches["quality_residual_rl"]
    )
    assert any(
        "failure does not block the amortization branch" in row
        for row in branches["quality_residual_rl"]
    )


def test_consistency_checks_fail_closed_on_scientific_boundary_mutations() -> None:
    contract = _contract()

    bad_hmm = deepcopy(contract)
    bad_hmm["demand_and_information"]["warning_kernel"][
        "incorrect_label"
    ] = "accuracy scalar only"
    assert build_payload(bad_hmm)["consistency_checks"][
        "c6_hmm_warning_causality_and_h4_are_explicit"
    ]["passed"] is False

    bad_architecture = deepcopy(contract)
    bad_architecture["learner_ladder"]["primary_architecture"][
        "policy_symmetry"
    ] = "permutation-invariant"
    assert build_payload(bad_architecture)["consistency_checks"][
        "c7_architecture_metrics_and_compute_claims_are_bounded"
    ]["passed"] is False

    bad_resources = deepcopy(contract)
    bad_resources["mechanism"]["action_decoder"][
        "sum_equals_batches_per_week"
    ] = False
    assert build_payload(bad_resources)["consistency_checks"][
        "c3_actions_decoder_and_resource_rights_are_consistent"
    ]["passed"] is False

    bad_custody = deepcopy(contract)
    bad_custody["execution_policy"]["fresh_seeds_opened"] = True
    assert build_payload(bad_custody)["consistency_checks"][
        "c8_no_seed_or_scientific_execution_is_authorized"
    ]["passed"] is False
