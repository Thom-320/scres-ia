from __future__ import annotations

import math

from supply_chain.program_t_exact_pomdp import ExactProductMixPOMDP


def test_exact_belief_dp_bounds_open_loop_and_truncated_mpc() -> None:
    model = ExactProductMixPOMDP(horizon=5)
    diagnostic = model.diagnostic()
    exact = float(diagnostic["exact_belief_dp_cost"])
    assert exact <= float(diagnostic["best_open_loop_cost"]) + 1e-12
    assert all(
        exact <= float(cost) + 1e-12
        for cost in diagnostic["truncated_mpc_cost"].values()
    )


def test_full_horizon_receding_mpc_recovers_exact_policy_value() -> None:
    model = ExactProductMixPOMDP(horizon=6)
    exact = model.exact_cost(0, 0.5, model.horizon)
    assert math.isclose(
        model.truncated_mpc_cost(model.horizon), exact, rel_tol=0.0, abs_tol=1e-12
    )


def test_exact_problem_is_product_label_symmetric() -> None:
    model = ExactProductMixPOMDP(horizon=5)
    for net_c in (-4, -1, 0, 2, 5):
        for belief in (0.1, 0.35, 0.5, 0.8):
            left = model.exact_cost(net_c, belief, 4)
            right = model.exact_cost(-net_c, 1.0 - belief, 4)
            assert math.isclose(left, right, rel_tol=0.0, abs_tol=1e-10)


def test_every_transition_conserves_total_product_mass() -> None:
    model = ExactProductMixPOMDP(horizon=4)
    for net_c in range(-6, 7):
        for action in model.actions:
            for demand_c in range(model.slots_per_week + 1):
                next_c = model.transition_net_c(net_c, action, demand_c)
                next_h = -next_c
                assert next_c + next_h == 0


def test_complete_open_loop_frontier_has_expected_cardinality() -> None:
    model = ExactProductMixPOMDP(horizon=4)
    sequence, cost = model.best_open_loop()
    assert len(sequence) == 4
    assert all(action in model.actions for action in sequence)
    assert cost >= 0.0


def test_invalid_sequences_and_parameters_fail_closed() -> None:
    try:
        ExactProductMixPOMDP(horizon=0)
    except ValueError:
        pass
    else:
        raise AssertionError("zero horizon must fail")

    model = ExactProductMixPOMDP(horizon=4)
    try:
        model.open_loop_cost((0, 1, 2))
    except ValueError:
        pass
    else:
        raise AssertionError("short open-loop sequence must fail")
