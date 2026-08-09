from scripts.audit_g3a_boundary_v2_contract_v1 import audit


def test_latest_g3a_negative_and_contract_limits_are_detected():
    result = audit()
    facts = result["facts"]
    assert facts["claim_status"] == "G3A_DID_NOT_REPRODUCE"
    assert facts["n_controllers"] == 34
    assert facts["missing_preregistered_falsifier_ids"] == ["f3", "f5"]
    assert facts["f9_uses_nonnegative_threshold"]
    assert facts["held_evaluation_arrays_complete"]
    assert not facts["per_seed_selection_rows_persisted"]
    assert facts["full34_reuses_predecessor_seed_block"]
