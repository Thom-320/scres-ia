from __future__ import annotations

import pytest

from scripts.adjudicate_q_r1_successor import adjudicate
from scripts.run_q_r1_successor_abc import fixed_theta_belief, summarize
from supply_chain.q_r1_retained_learning import ESTIMANDS


def _row(estimand: str, kappa: float, arm: str, value: float) -> dict:
    return {
        "estimand": estimand,
        "kappa": kappa,
        "history_root": 1,
        "campaign_index": 1,
        "arm": arm,
        "early_ret_visible": value,
        "early_ret_complete_cohort": value,
        "ret_visible": value,
        "worst_product_fill": value,
        "unresolved_orders": 0.0,
        "unresolved_quantity": 0.0,
        "lost_orders": 0.0,
        "lost_quantity": 0.0,
        "service_loss_auc": 0.0,
        "gross_policy_batch_slots": 24.0,
        "gross_production_quantity": 120000.0,
        "charged_daily_dispatch_slots": 56.0,
        "charged_downstream_vehicle_hours": 2688.0,
    }


def test_successor_summary_keeps_estimands_separate() -> None:
    arms = (
        "oracle_initial_context",
        "retained_posterior",
        "reset_posterior_0p5",
        "shuffled_posterior",
        "wrong_posterior",
        "delayed_posterior",
    )
    rows = []
    for estimand_index, estimand in enumerate(ESTIMANDS):
        for kappa in (0.5, 0.75, 0.9):
            for arm in arms:
                value = 0.5
                if arm == "retained_posterior":
                    value += 0.01 * estimand_index
                rows.append(_row(estimand, kappa, arm, value))
    result = summarize(rows)
    assert set(result) == set(ESTIMANDS)
    assert (
        result["historical_splice"]["0.9"]["retained_posterior"]
        ["mean_early_ret_complete_cohort_delta"]
        == pytest.approx(0.02)
    )
    assert (
        result["prefix_natural_replanning"]["0.9"]["retained_posterior"]
        ["mean_early_ret_complete_cohort_delta"]
        == pytest.approx(0.0)
    )


def test_fixed_theta_belief_retains_only_context_probability() -> None:
    belief = fixed_theta_belief(0.8)
    assert belief.theta_marginal == (0.0, 0.0, 1.0)
    assert belief.probability_regime_c == 0.8


def _summary_arm(mean: float, lcb: float = 0.01) -> dict:
    return {
        "mean_early_ret_visible_delta": mean,
        "early_ret_visible_clustered_ci95": [lcb, mean + 0.01],
        "mean_early_ret_complete_cohort_delta": mean,
        "early_ret_complete_cohort_clustered_ci95": [lcb, mean + 0.01],
        "mean_lost_orders_delta": 0.0,
        "mean_lost_quantity_delta": 0.0,
        "worst_product_fill_clustered_ci95": [-0.01, 0.01],
        "max_scheduled_resource_error": 0.0,
    }


def test_adjudicator_never_selects_oracle_or_placebos() -> None:
    arms = (
        "oracle_initial_context",
        "retained_posterior",
        "reset_posterior_0p5",
        "shuffled_posterior",
        "wrong_posterior",
        "delayed_posterior",
    )
    summary = {}
    for estimand in ESTIMANDS:
        summary[estimand] = {}
        for kappa, retained in ((0.5, 0.0), (0.75, 0.012), (0.9, 0.02)):
            summary[estimand][str(kappa)] = {
                arm: _summary_arm(
                    retained
                    if arm == "retained_posterior"
                    else (-0.01 if arm in {"shuffled_posterior", "wrong_posterior", "delayed_posterior"} else 0.0),
                    lcb=(0.005 if arm == "retained_posterior" and kappa == 0.9 else 0.0),
                )
                for arm in arms
            }
    result = adjudicate({"schema_version": "test", "summary": summary})
    assert result["post_outcome_episode_selector_used"] is False
    assert result["eligible_controller_arms"] == [
        "retained_posterior",
        "reset_posterior_0p5",
    ]
    assert result["learner_training_authorized"] is False
