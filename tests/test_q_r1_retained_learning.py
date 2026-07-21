from __future__ import annotations

import numpy as np
import pytest

from scripts.evaluate_program_q_replication import scheduler
from supply_chain.program_t_joint_belief import ExactJointBelief
from supply_chain.q_r1_retained_learning import (
    PERSISTENCE_MODES,
    build_parameter_history,
    common_continuation_calendar,
    controller_prefix,
    retained_belief_path,
    transition_theta,
    wrong_belief,
)


def test_early_metric_accepts_direct_order_objects():
    from types import SimpleNamespace
    from supply_chain.q_r1_retained_learning import early_cohort_metrics_from_orders

    orders = [
        SimpleNamespace(
            j=1, OPTj=10.0, OATj=40.0, LTj=48.0, lost=False,
            lost_time=None, metrics_excluded=False, quantity=10.0,
            requested_product_id="P_C",
        ),
        SimpleNamespace(
            j=2, OPTj=20.0, OATj=None, LTj=48.0, lost=False,
            lost_time=None, metrics_excluded=False, quantity=10.0,
            requested_product_id="P_H",
        ),
    ]
    scored = early_cohort_metrics_from_orders(
        orders=orders, decision_start=0.0, score_time=500.0
    )
    assert scored["early_generated_orders"] == 2.0
    assert scored["early_unresolved_orders"] == 1.0
    assert scored["early_omitted_rows"] == 1.0


def test_risk_level_history_is_reproducible_and_binary():
    from scripts.run_q_r1_d2_risk_memory_bound import risk_level_history

    first = risk_level_history(7570601, 12, "persistent_0p90", "R24")
    second = risk_level_history(7570601, 12, "persistent_0p90", "R24")
    assert first == second
    assert len(first) == 12
    assert set(first) <= {0, 1}


def test_risk_cells_change_only_the_declared_native_frequency():
    from scripts.run_q_r1_d2_risk_memory_bound import cell_for

    current = cell_for("R24", 0)
    increased = cell_for("R24", 1)
    assert current.phi_by_risk["R22"] == increased.phi_by_risk["R22"] == 1.0
    assert current.phi_by_risk["R24"] == 1.0
    assert increased.phi_by_risk["R24"] == 2.0


def test_iid_theta_transition_is_uniform() -> None:
    rng = np.random.default_rng(7123)
    counts = np.zeros(3, dtype=int)
    current = 0
    for _ in range(60_000):
        current = transition_theta(
            current, persistence=PERSISTENCE_MODES["iid"], rng=rng
        )
        counts[current] += 1
    np.testing.assert_allclose(counts / counts.sum(), (1 / 3, 1 / 3, 1 / 3), atol=0.01)


def test_physical_history_and_knowledge_are_separate() -> None:
    history = build_parameter_history(
        history_root=7_570_201,
        campaigns=3,
        persistence_mode="persistent_0p90",
        scheduler=scheduler(),
    )
    path = retained_belief_path(history)
    assert len(path) == len(history) == 3
    assert path[0].theta_marginal == (1 / 3, 1 / 3, 1 / 3)
    assert history[0].reset_physical() is history[0]
    assert history[0].skeleton.prefix_state_hash
    assert path[1].probability_regime_c == pytest.approx(0.5)


def test_common_continuation_changes_only_first_two_decisions() -> None:
    arm = (0, 1, 2, 3, 0, 1, 2, 3)
    reset = (3, 2, 1, 0, 3, 2, 1, 0)
    combined = common_continuation_calendar(arm, reset)
    assert combined == (0, 1, 1, 0, 3, 2, 1, 0)


def test_wrong_belief_never_leaks_regime() -> None:
    belief = ExactJointBelief.from_theta_marginal((0.8, 0.15, 0.05))
    wrong = wrong_belief(belief)
    np.testing.assert_allclose(wrong.theta_marginal, (0.05, 0.15, 0.8))
    assert wrong.probability_regime_c == pytest.approx(0.5)


def test_controller_prefix_returns_only_requested_decisions() -> None:
    history = build_parameter_history(
        history_root=7_570_201,
        campaigns=2,
        persistence_mode="iid",
        scheduler=scheduler(),
    )
    from supply_chain.program_t_full_des_mpc import FullDEST0Config

    prefix, detail = controller_prefix(
        campaign=history[0],
        belief=ExactJointBelief.uniform(),
        scheduler=scheduler(),
        config=FullDEST0Config(1, "scenario", particles=2),
        decisions=2,
    )
    assert len(prefix) == 2
    assert len(detail["decisions"]) == 2
