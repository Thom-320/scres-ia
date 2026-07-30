from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
CONTRACT = ROOT / "contracts" / "paper_b_independent_calibration_v1.json"
ECONOMIC_CONTRACT = (
    ROOT / "contracts" / "cobb_douglas_economic_sensitivity_v2.json"
)
ECONOMIC_RESULT = (
    ROOT / "results" / "cobb_douglas" / "economic_sensitivity_v2" / "result.json"
)


def _contract() -> dict:
    return json.loads(CONTRACT.read_text())


def test_independent_decision_is_not_an_external_blocker() -> None:
    contract = _contract()
    assert contract["status"] == "METHOD_DECISION_FREEZE_NO_EXTERNAL_INPUT_REQUIRED"
    assert contract["external_signoff_required"] is False
    assert contract["historical_results_modified"] is False
    assert contract["authorization"]["external_email_or_message_authorized"] is False
    assert contract["authorization"]["neural_or_kan_authorized"] is False


def test_delay_grid_is_derived_from_the_frozen_lead_time() -> None:
    contract = _contract()
    decision = contract["fulfillment_delay_decision"]
    grid = decision["prospective_robustness_grid"]
    lead_time = grid["lead_time_hours"]
    assert lead_time == contract["thesis_evidence"]["lead_time_hours"] == 48.0
    assert grid["delay_hours"] == [
        lead_time + slack for slack in grid["signed_slack_hours"]
    ]
    assert decision["historical_reproduction_lane_hours"] in grid["delay_hours"]
    assert contract["thesis_evidence"]["fulfillment_delay_54_specified_by_thesis"] is False


def test_cost_decision_cannot_select_a_policy() -> None:
    contract = _contract()
    economic = contract["economic_decision"]
    assert economic["primary_resource_comparison"]["method"] == "PARETO_NO_SCALARIZATION"
    assert economic["cobb_douglas_role"] == "SECONDARY_SENSITIVITY_ONLY"
    assert economic["monetary_calibration_required_for_current_methodological_claims"] is False
    assert economic["relative_price_grid"]["one_factor_multipliers"] == [
        0.5,
        1.0,
        2.0,
        5.0,
    ]
    assert set(economic["relative_price_grid"]["active_static_terms"]) == {
        "c_p",
        "c_u",
        "c_i",
        "c_b",
    }


def test_executable_economic_grid_matches_the_independent_decision() -> None:
    decision = _contract()["economic_decision"]["relative_price_grid"]
    economic = json.loads(ECONOMIC_CONTRACT.read_text())
    assert economic["primary_endpoint_authorized"] is False
    assert economic["policy_selection_authorized"] is False
    assert economic["domain_calibration_status"] == (
        "NOT_REQUIRED_FOR_BOUNDED_METHODOLOGICAL_CLAIMS"
    )
    scenarios = economic["one_factor_scenarios"]
    assert {
        spec["coefficient"] for spec in scenarios.values()
    } == set(decision["active_static_terms"])
    assert {
        spec["multiplier"] for spec in scenarios.values()
    } == {0.5, 2.0, 5.0}


def test_economic_result_remains_nonselective_and_contract_bound() -> None:
    result = json.loads(ECONOMIC_RESULT.read_text())
    assert result["schema_version"] == "cobb_douglas_economic_sensitivity_v2"
    assert result["primary_endpoint_authorized"] is False
    assert result["policy_selection_authorized"] is False
    assert result["contract_path"] == (
        "contracts/cobb_douglas_economic_sensitivity_v2.json"
    )
    assert result["families"]["R1r"]["winner_stable_across_grid"] is True
    assert result["families"]["R2r"]["winner_stable_across_grid"] is False
