"""Unit tests for the MPC* strengthened comparator (Q-R1 Gate 1)."""

from __future__ import annotations

import numpy as np
import pytest

from supply_chain.program_t_joint_belief import ExactJointBelief, THETA_GRID
from supply_chain.program_t_mpc_star import (
    MPCStarConfig,
    _realization_seed,
    _weighted_tail_mean,
    exact_stratified_skeletons,
)


def test_enumerate_states_is_exact_and_normalized():
    belief = ExactJointBelief.from_theta_marginal((0.7, 0.2, 0.1), probability_regime_c=0.6)
    states = belief.enumerate_states()
    assert len(states) == 2 * len(THETA_GRID)
    assert sum(weight for *_x, weight in states) == pytest.approx(1.0)
    # exact weights match the flattened posterior (renormalized)
    flat = belief.probability.reshape(-1)
    flat = flat / flat.sum()
    got = sorted(weight for *_x, weight in states)
    assert got == pytest.approx(sorted(flat.tolist()))


def test_enumerate_matches_sampler_distribution():
    belief = ExactJointBelief.from_theta_marginal((0.5, 0.3, 0.2), probability_regime_c=0.4)
    sampled = belief.sample_states(count=200_000, seed=11)
    from collections import Counter

    counts = Counter(sampled)
    enum = {(rho, share, regime): weight for rho, share, regime, weight in belief.enumerate_states()}
    for key, weight in enum.items():
        assert counts[key] / 200_000 == pytest.approx(weight, abs=0.01)


def test_oracle_collapses_to_two_states():
    belief = ExactJointBelief.oracle_parameters((0.90, 0.90))
    states = belief.enumerate_states()
    assert len(states) == 2
    assert all(weight == pytest.approx(0.5) for *_x, weight in states)


def test_weighted_tail_matches_unweighted_cvar():
    ret = np.arange(1, 21, dtype=float).reshape(1, 20) / 20.0
    weights = np.full(20, 0.05)
    # worst 10% weight-mass = worst 2 of 20 = mean(0.05, 0.10) = 0.075
    assert float(_weighted_tail_mean(ret, weights, 0.10)[0]) == pytest.approx(0.075)


def test_crn_seed_is_deterministic_and_arm_independent():
    a = _realization_seed("digest_abc", 2, 3)
    b = _realization_seed("digest_abc", 2, 3)
    c = _realization_seed("digest_abc", 2, 4)
    assert a == b  # same (obs, state, realization) -> identical seed (CRN)
    assert a != c  # different realization -> different seed


def _scheduler():
    from scripts.evaluate_program_q_replication import scheduler

    return scheduler()


def test_exact_stratified_skeletons_are_crn_reproducible():
    from supply_chain.retained_context_discovery import build_campaign_history
    from supply_chain.program_o_state_rich import StateRichConfiguration, state_rich_calendar

    sched = _scheduler()
    history = build_campaign_history(
        history_root=7_570_801, campaigns=2, kappa=0.9, scheduler=sched,
        regime_persistence=0.90, dominant_share=0.90,
    )
    campaign = history[1]
    _cal, rows = state_rich_calendar(
        skeleton=campaign.skeleton.as_dict(), scheduler=sched,
        config=StateRichConfiguration("belief_mpc", 1),
        regime_persistence=0.90, dominant_share=0.90,
        action_overrides=(0,) * campaign.skeleton.decision_weeks,
    )
    observation = rows[0].observation
    belief = ExactJointBelief.uniform()
    cfg = MPCStarConfig(horizon=3, realizations_per_state=6, mode="constraint_aware")
    first = exact_stratified_skeletons(campaign.skeleton, observation, belief, cfg)
    second = exact_stratified_skeletons(campaign.skeleton, observation, belief, cfg)
    assert len(first) == 6 * 6  # 6 states x 6 realizations
    assert sum(weight for _sk, weight in first) == pytest.approx(1.0)
    # byte-identical realizations across calls (frozen CRN)
    assert [sk.skeleton_sha256 for sk, _w in first] == [sk.skeleton_sha256 for sk, _w in second]
    assert [sk.order_products for sk, _w in first] == [sk.order_products for sk, _w in second]


def test_fail_closed_returns_safe_default_when_nothing_feasible():
    """With an unreachable worst-product floor, the planner must fall back to the
    safe (max weighted worst-fill) default and flag it, never pick on ReT."""
    from supply_chain.retained_context_discovery import build_campaign_history
    from supply_chain.program_o_state_rich import StateRichConfiguration, state_rich_calendar
    from supply_chain.program_t_mpc_star import choose_mpc_star_action

    sched = _scheduler()
    history = build_campaign_history(
        history_root=7_570_802, campaigns=2, kappa=0.9, scheduler=sched,
        regime_persistence=0.90, dominant_share=0.90,
    )
    campaign = history[1]
    _cal, rows = state_rich_calendar(
        skeleton=campaign.skeleton.as_dict(), scheduler=sched,
        config=StateRichConfiguration("belief_mpc", 1),
        regime_persistence=0.90, dominant_share=0.90,
        action_overrides=(0,) * campaign.skeleton.decision_weeks,
    )
    observation = rows[0].observation
    belief = ExactJointBelief.uniform()
    # floor 1.0 is unreachable -> feasible set empty -> fail-closed fallback
    cfg = MPCStarConfig(horizon=3, realizations_per_state=4, mode="constraint_aware", worst_product_floor=1.0)
    action, diag = choose_mpc_star_action(
        observation, base_skeleton=campaign.skeleton, prefix=[], scheduler=sched,
        config=cfg, belief=belief,
    )
    assert diag["fallback_used"] == 1.0
    assert diag["planning_feasible"] == 0.0
    assert action in (0, 1, 2, 3)
