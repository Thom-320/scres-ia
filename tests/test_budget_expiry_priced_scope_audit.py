from scripts.audit_budget_expiry_priced_scope_v1 import audit


def test_priced_closure_is_static_and_did_not_persist_raw_matrices():
    facts = audit()["facts"]
    assert facts["n_postures"] == 27
    assert facts["same_action_reused_at_every_step"] is True
    assert facts["enumerates_within_episode_schedules"] is False
    assert facts["all_24_reported_distinct_static_optima_equal_one"] is True
    assert facts["raw_tape_by_posture_matrices_persisted"] is False
    assert facts["cost_scale_uses_train_and_test_max"] is True
