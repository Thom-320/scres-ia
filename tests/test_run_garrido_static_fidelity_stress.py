from scripts.run_garrido_static_fidelity_stress import policy_candidates


def test_matched_only_policy_set_contains_only_garrido_baseline() -> None:
    policies = policy_candidates("matched_only")

    assert [policy["name"] for policy in policies] == ["garrido_matched_DOE_baseline"]
    assert policies[0]["kind"] == "matched_doe"


def test_minimal_policy_set_keeps_static_comparison_controls() -> None:
    names = {policy["name"] for policy in policy_candidates("minimal")}

    assert "garrido_matched_DOE_baseline" in names
    assert "pure_inventory_I0_S1" in names
    assert "pure_inventory_I672_S1" in names
    assert "pure_capacity_I0_S1" in names
    assert "pure_capacity_I0_S3" in names
