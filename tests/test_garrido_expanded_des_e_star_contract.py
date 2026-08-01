from __future__ import annotations

import json
from pathlib import Path

from scripts.validate_garrido_expanded_des_contract import validate_contract


ROOT = Path(__file__).resolve().parents[1]


def _contract() -> dict:
    return json.loads(
        (ROOT / "contracts" / "garrido_expanded_des_e_star_v1.json").read_text()
    )


def test_e_star_is_design_only_and_inherits_fail_closed_authority() -> None:
    contract = _contract()
    assert contract["status"] == "DESIGN_ONLY_NOT_AUTHORIZED_UNTIL_SUBMISSION_A_RECEIPT"
    authority = contract["authority"]
    assert authority["source"] == "main:contracts/authority_ladder_v1.json"
    assert authority["scientific_execution_authorized"] is False
    assert authority["fresh_roots_opened"] is False
    assert authority["fresh_tapes_opened"] is False
    assert authority["optimizer_seeds_assigned"] is False
    assert authority["neural_training_authorized"] is False


def test_e_star_has_one_kernel_and_complete_decision_mask_factorial() -> None:
    contract = _contract()
    assert contract["scope"]["prospective_kernel"] == "E_star"
    masks = contract["factorial_masks"]
    assert [mask["mask_id"] for mask in masks] == [
        "M000", "M100", "M010", "M001",
        "M110", "M101", "M011", "M111",
    ]
    assert len({tuple(mask[key] for key in ("P", "U", "D")) for mask in masks}) == 8


def test_primary_metric_requires_a_pre_execution_choice_and_reports_both_lenses() -> None:
    metrics = _contract()["metric_hierarchy"]
    assert metrics["primary_selection_status"] == (
        "PENDING_PI_AND_GARRIDO_SIGNATURE_BEFORE_FRESH_DATA"
    )
    allowed = {item["id"] for item in metrics["allowed_primary_endpoints"]}
    assert allowed == {"ret_excel_request_snapshot_v2", "cobb_douglas_index"}
    assert "ret_excel_request_snapshot_v2" in metrics["always_reported_resilience_panel"]
    assert "cobb_douglas_index" in metrics["always_reported_resilience_panel"]
    assert metrics["scalarization"]["unregistered_weighted_sum_forbidden"] is True


def test_cvar_is_secondary_and_cannot_promote_or_block_alone() -> None:
    cvar = _contract()["metric_hierarchy"]["cvar"]
    assert cvar["role"] == "secondary_non_promoting_tail_diagnostic"
    assert cvar["may_be_primary"] is False
    assert cvar["may_promote_alone"] is False
    assert cvar["may_block_alone"] is False


def test_no_neural_gate_before_structured_residual() -> None:
    contract = _contract()
    assert contract["gates"]["G3"].startswith("observable residual")
    assert contract["neural_ladder"]["r2_not_a_control_gate"] is True
    assert contract["seed_and_execution_policy"]["new_seed_opening"] is False


def test_design_validator_is_fail_closed() -> None:
    contract = _contract()
    registry = json.loads(
        (ROOT / "research" / "seed_custody_registry.json").read_text()
    )
    result = validate_contract(contract, registry)
    assert result == {"ok": True, "errors": []}


def test_design_validator_rejects_a_cvar_primary() -> None:
    contract = _contract()
    registry = json.loads(
        (ROOT / "research" / "seed_custody_registry.json").read_text()
    )
    contract["metric_hierarchy"]["cvar"]["may_be_primary"] = True
    result = validate_contract(contract, registry)
    assert result["ok"] is False
    assert "metric_hierarchy.cvar.may_be_primary must be false" in result["errors"]
