from __future__ import annotations

import numpy as np

from supply_chain.g3a_forensic import make_tape, policies, simulate


def test_policy_library_has_the_manuscript_width_and_unique_names():
    library = policies()
    assert len(library) == 34
    assert len({policy.name for policy in library}) == 34
    assert sum(policy.family == "constant" for policy in library) == 9
    assert sum(not policy.deployable for policy in library) == 1


def test_same_tape_is_policy_independent_and_global_fifo_is_action_invariant():
    tape = make_tape(8701001, "persistent_uniform")
    outcomes = [simulate(tape, "global_fifo", policy) for policy in policies()]
    assert len({outcome["tape_sha256"] for outcome in outcomes}) == 1
    assert max(outcome["primary_service"] for outcome in outcomes) == min(
        outcome["primary_service"] for outcome in outcomes)
    assert max(abs(outcome["flow_residual"]) for outcome in outcomes) < 1e-8


def test_capacity_contracts_have_the_declared_forfeiture_semantics():
    tape = make_tape(8701002, "persistent_uniform")
    policy = next(policy for policy in policies() if policy.name == "constant_0.9")
    hard = simulate(tape, "hard_quota", policy)
    spare = simulate(tape, "spare_reallocation", policy)
    pooled = simulate(tape, "global_fifo", policy)
    assert hard["forfeited_capacity"] > 0
    assert spare["forfeited_capacity"] == 0
    assert pooled["forfeited_capacity"] == 0
    assert np.isfinite([hard["primary_service"], spare["primary_service"], pooled["primary_service"]]).all()
