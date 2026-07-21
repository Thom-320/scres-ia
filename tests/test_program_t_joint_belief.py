from __future__ import annotations

import math

import numpy as np

from supply_chain.program_t_joint_belief import ExactJointBelief, THETA_GRID, weekly_product_counts


def test_joint_belief_normalizes_and_learns_extreme_counts() -> None:
    belief = ExactJointBelief.uniform()
    prior = belief.theta_marginal
    belief.observe_previous_week(6)
    assert math.isclose(float(belief.probability.sum()), 1.0)
    assert belief.probability_regime_c > 0.5
    assert belief.theta_marginal != prior


def test_oracle_parameters_never_reveal_current_regime() -> None:
    belief = ExactJointBelief.oracle_parameters((0.90, 0.75))
    assert belief.theta_marginal == (0.0, 1.0, 0.0)
    assert belief.probability_regime_c == 0.5
    samples = belief.sample_states(count=100, seed=7)
    assert {(rho, share) for rho, share, _regime in samples} == {(0.90, 0.75)}
    assert {regime for _rho, _share, regime in samples} == {False, True}


def test_exact_update_matches_manual_six_state_calculation() -> None:
    belief = ExactJointBelief.uniform()
    belief.observe_previous_week(5)
    manual = np.zeros((3, 2), dtype=float)
    for i, (rho, share) in enumerate(THETA_GRID):
        for z in (0, 1):
            p = share if z else 1.0 - share
            likelihood = math.comb(6, 5) * p**5 * (1.0 - p)
            for z_next in (0, 1):
                transition = rho if z_next == z else 1.0 - rho
                manual[i, z_next] += (1.0 / 6.0) * likelihood * transition
    manual /= manual.sum()
    np.testing.assert_allclose(belief.probability, manual, atol=1e-15)


def test_weekly_counts_use_only_completed_week_rows() -> None:
    counts = weekly_product_counts(
        order_times=(30.0, 54.0, 198.0, 390.0),
        order_products=("P_C", "P_H", "P_C", "P_C"),
        decision_start=0.0,
        weeks=3,
    )
    assert counts == (1, 1, 1)
