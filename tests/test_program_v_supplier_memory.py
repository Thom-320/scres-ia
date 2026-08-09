from supply_chain.program_v_supplier_memory import (
    ACTIONS, WEEKLY_ORDER, HORIZON, avoid_action, make_tape, policy_library, simulate,
    update_posterior,
)
import numpy as np
from scripts.run_program_v_prelearner_gate_v1 import paired_interval


def test_action_grid_and_policy_library_are_fixed():
    assert len(ACTIONS) == 6
    assert all(sum(action) == 1.0 and max(action) <= 0.5 for action in ACTIONS)
    assert len(policy_library()) == 13


def test_privileged_action_reverses_with_each_disrupted_supplier():
    assert len({avoid_action(i) for i in range(3)}) == 3
    for i in range(3):
        assert avoid_action(i)[i] == 0.0


def test_tape_is_common_and_mass_and_order_rights_close():
    tape = make_tape(8701001)
    rows = [simulate(tape, policy) for policy in policy_library()]
    assert len({row["tape_sha256"] for row in rows}) == 1
    assert max(abs(row["mass_residual"]) for row in rows) < 1e-9
    assert {row["ordered"] for row in rows} == {WEEKLY_ORDER * HORIZON}


def test_paired_interval_uses_seed_pairs():
    rows = [
        {"seed": 1, "policy": "a", "service": 0.8},
        {"seed": 1, "policy": "b", "service": 0.7},
        {"seed": 2, "policy": "a", "service": 0.9},
        {"seed": 2, "policy": "b", "service": 0.8},
    ]
    result = paired_interval(rows, "a", "b")
    assert result["n_pairs"] == 2
    assert abs(result["mean"] - 0.1) < 1e-12


def test_unobserved_yield_does_not_change_belief():
    prior = np.array([0.7, 0.2, 0.1])
    mask = np.array([True, False, True])
    lhs = update_posterior(prior, 1, np.array([0.1, 0.0, 1.0]), mask)
    rhs = update_posterior(prior, 1, np.array([0.1, 1.0, 1.0]), mask)
    np.testing.assert_allclose(lhs, rhs)
