"""The G3a contract audit, and what it should say once its findings are acted on.

This test arrived asserting that the runner was OUT of contract compliance: two preregistered
falsifiers missing, `f9` written as a non-negativity check that could not fail, and per-seed
selection rows not persisted. All four findings were correct.

They have been fixed, so the assertions are inverted rather than deleted. An audit test that keeps
asserting a defect after the defect is gone stops testing the runner and starts testing that nobody
improved it. What still has to hold is the AUDITOR's ability to see a regression: if a preregistered
falsifier is dropped again, or `f9` goes back to `>= 0`, or the selection rows stop being written,
these assertions fail.

The one finding not fixable in code is kept as-is: the 34-arm run reuses the 14-arm seed block. It
is a scope amendment rather than an independent replication, and the artifact should keep saying so.
"""
from scripts.audit_g3a_boundary_v2_contract_v1 import audit


def test_latest_g3a_negative_stands():
    facts = audit()["facts"]
    assert facts["claim_status"] == "G3A_DID_NOT_REPRODUCE"
    assert facts["n_controllers"] == 34
    h = facts["persistent_uniform_hard_quota_h_obs"]
    # The reported target is 0.0963 [0.0682, 0.1245]; this straddles zero an order of magnitude
    # below it. The interval crossing zero is the finding, not the point estimate.
    assert h["lcb95"] < 0.0 < h["ucb95"]
    assert h["mean"] < 0.01


def test_preregistered_falsifiers_are_all_present():
    """f3 and f5 were enumerated in the contract and absent from the runner. Both now exist."""
    facts = audit()["facts"]
    assert facts["contract_declares_f1_through_f9"]
    assert facts["missing_preregistered_falsifier_ids"] == []
    assert not facts["mass_falsifier_absent"]
    assert not facts["common_belief_model_falsifier_absent"]


def test_f9_is_no_longer_a_check_that_cannot_fail():
    """`forfeited >= 0` is true of every number a counter can hold, and also true when the
    attribute is missing and `getattr` returns its default. It now asks that forfeiture SEPARATES
    the contracts, which a work-conserving cell can violate."""
    facts = audit()["facts"]
    assert not facts["f9_uses_nonnegative_threshold"]


def test_selection_is_re_derivable_from_the_artifact():
    facts = audit()["facts"]
    assert facts["held_evaluation_arrays_complete"]
    assert facts["per_seed_selection_rows_persisted"]


def test_scope_amendment_is_still_declared_as_one():
    """Not a defect and not fixable in code: the enlarged library reused the same seeds, so it
    widens the class rather than replicating the result. The artifact must keep saying so."""
    facts = audit()["facts"]
    assert facts["full34_reuses_predecessor_seed_block"]
    assert facts["full34_seed_count"] == 60
