from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from scripts.evaluate_program_q_replication import scheduler
from scripts.audit_q_r1_comparator_power import audit, required_histories
from scripts.merge_q_r1_comparator_v2_shards import (
    merge_convergence,
    merge_pareto,
    merge_targeted,
)
from supply_chain.program_o_full_des_transducer import simulate_full_des_frontier
from supply_chain.program_o_state_rich import (
    StateRichConfiguration,
    state_rich_calendar,
)
from supply_chain.program_t_joint_belief import ExactJointBelief
from supply_chain.q_r1_comparator_v2 import (
    ComparatorV2Config,
    NoFeasibleStructuredAction,
    PlanningKey,
    choose_comparator_v2_action,
    conditional_scenario_bank,
)
from supply_chain.q_r1_retained_learning import (
    build_parameter_history,
    evaluate_calendar,
)


def _campaign_and_observation():
    campaign = build_parameter_history(
        history_root=7_570_801,
        campaigns=2,
        persistence_mode="persistent_0p90",
        scheduler=scheduler(),
    )[0]
    _calendar, rows = state_rich_calendar(
        skeleton=campaign.skeleton.as_dict(),
        scheduler=scheduler(),
        config=StateRichConfiguration("belief_mpc", 1),
        regime_persistence=0.90,
        dominant_share=0.90,
        action_overrides=(0,) * campaign.skeleton.decision_weeks,
    )
    return campaign, rows[0].observation


def test_exact_state_enumerator_preserves_all_six_weights() -> None:
    belief = ExactJointBelief.uniform()
    states = belief.enumerate_states()
    assert len(states) == 6
    assert sum(row[3] for row in states) == pytest.approx(1.0)
    assert {row[:3] for row in states} == {
        (0.75, 0.90, False),
        (0.75, 0.90, True),
        (0.90, 0.75, False),
        (0.90, 0.75, True),
        (0.90, 0.90, False),
        (0.90, 0.90, True),
    }


def test_vectorized_qr1_metric_matches_trace_metric() -> None:
    campaign, _observation = _campaign_and_observation()
    calendar = (0, 1, 2, 3, 0, 1, 2, 3)
    panel = simulate_full_des_frontier(
        skeleton=campaign.skeleton,
        scheduler=scheduler(),
        calendars=[calendar],
        include_q_r1_metrics=True,
    )
    direct = evaluate_calendar(
        campaign=campaign,
        calendar=calendar,
        scheduler=scheduler(),
    )
    for key in (
        "early_ret_visible",
        "early_ret_complete_cohort",
        "early_generated_orders",
        "early_visible_rows",
        "early_unresolved_orders",
        "early_fill_P_C",
        "early_fill_P_H",
        "early_worst_product_fill",
    ):
        assert panel[key][0] == pytest.approx(direct[key], abs=1e-12), key


def test_conditional_bank_is_arm_and_observation_hash_independent() -> None:
    campaign, observation = _campaign_and_observation()
    key = PlanningKey(campaign.history_root, campaign.campaign_index, 0)
    retained = ExactJointBelief.from_theta_marginal(
        (0.0, 0.0, 1.0), probability_regime_c=0.9
    )
    reset = ExactJointBelief.from_theta_marginal(
        (0.0, 0.0, 1.0), probability_regime_c=0.5
    )
    changed_hash = replace(observation, observation_sha256="different-policy-observation")
    retained_bank, retained_weights, retained_sha = conditional_scenario_bank(
        base=campaign.skeleton,
        observation=observation,
        belief=retained,
        key=key,
        conditional_paths=4,
    )
    reset_bank, reset_weights, reset_sha = conditional_scenario_bank(
        base=campaign.skeleton,
        observation=changed_hash,
        belief=reset,
        key=key,
        conditional_paths=4,
    )
    assert retained_sha == reset_sha
    assert [row.skeleton_sha256 for row in retained_bank] == [
        row.skeleton_sha256 for row in reset_bank
    ]
    assert not np.array_equal(retained_weights, reset_weights)
    assert len(retained_bank) == 24


def test_conditional_bank_is_nested_across_path_budgets() -> None:
    campaign, observation = _campaign_and_observation()
    key = PlanningKey(campaign.history_root, campaign.campaign_index, 0)
    belief = ExactJointBelief.uniform()
    bank4, _weights4, _sha4 = conditional_scenario_bank(
        base=campaign.skeleton,
        observation=observation,
        belief=belief,
        key=key,
        conditional_paths=4,
    )
    bank16, _weights16, _sha16 = conditional_scenario_bank(
        base=campaign.skeleton,
        observation=observation,
        belief=belief,
        key=key,
        conditional_paths=16,
    )
    assert {row.skeleton_sha256 for row in bank4} <= {
        row.skeleton_sha256 for row in bank16
    }


def test_no_feasible_sequence_abstains_instead_of_false_safe(monkeypatch) -> None:
    campaign, observation = _campaign_and_observation()

    def infeasible_panel(*, calendars, **_kwargs):
        size = len(calendars)
        return {
            "early_ret_complete_cohort": np.ones(size),
            "ret_visible": np.ones(size),
            "ret_full": np.ones(size),
            "worst_product_fill": np.full(size, 0.5),
            "unresolved_orders": np.zeros(size),
            "lost_orders": np.zeros(size),
        }

    monkeypatch.setattr(
        "supply_chain.q_r1_comparator_v2.simulate_full_des_frontier",
        infeasible_panel,
    )
    with pytest.raises(NoFeasibleStructuredAction):
        choose_comparator_v2_action(
            observation,
            base_skeleton=campaign.skeleton,
            prefix=(),
            scheduler=scheduler(),
            belief=ExactJointBelief.uniform(),
            planning_key=PlanningKey(
                campaign.history_root, campaign.campaign_index, 0
            ),
            config=ComparatorV2Config(
                horizon=1,
                conditional_paths=1,
                mode="constraint_aware",
                worst_product_floor=0.70,
            ),
        )


def test_merge_recomputes_convergence_from_raw_rows() -> None:
    signature = [4, "scenario", 0.0, "expected"]
    shards = []
    for root, actions, errors in (
        (7_570_801, ((0, 0), (1, 1)), (0.001, 0.002)),
        (7_570_807, ((2, 2), (3, 3)), (0.003, 0.004)),
    ):
        shards.append(
            {
                "convergence": [
                    {
                        "signature": signature,
                        "low_config": "low",
                        "high_config": "high",
                        "low_abstentions": 0,
                        "high_abstentions": 0,
                    }
                ],
                "convergence_pairs": [
                    {
                        "signature": signature,
                        "history_root": root,
                        "campaign_index": index + 1,
                        "persistence_mode": "binary_0.9",
                        "prior_arm": "retained",
                        "low_action": action[0],
                        "high_action": action[1],
                        "absolute_planning_value_error": error,
                    }
                    for index, (action, error) in enumerate(zip(actions, errors))
                ],
            }
        )
    merged = merge_convergence(shards)
    row = merged["convergence"][0]
    assert row["first_action_agreement"] == 1.0
    assert row["mean_abs_planning_value_error"] == pytest.approx(0.0025)
    assert row["convergence_pass"] is True


def test_merge_recomputes_pareto_from_raw_rows() -> None:
    def metrics(ret: float, fill: float, unresolved: float):
        return {
            "early_ret_complete_cohort": ret,
            "worst_product_fill": fill,
            "unresolved_orders": unresolved,
            "lost_orders": 0.0,
            "mass_residual": 0.0,
        }

    shards = [
        {
            "pareto_pairs": [
                {
                    "config_id": "h4_c64",
                    "history_root": 7_570_801 + index,
                    "campaign_index": 1,
                    "persistence_mode": "binary_0.9",
                    "retained": metrics(0.8 + 0.1 * index, 0.7, 1.0),
                    "reset": metrics(0.7, 0.72, 0.5),
                }
            ]
        }
        for index in range(2)
    ]
    merged = merge_pareto(shards)
    row = merged["pareto"][0]
    assert row["pairs"] == 2
    assert row["retained_minus_reset_early_ret_complete_cohort"] == pytest.approx(0.15)
    assert row["retained_minus_reset_worst_product_fill"] == pytest.approx(-0.02)
    assert row["retained_minus_reset_unresolved_orders"] == pytest.approx(0.5)


def test_power_audit_clusters_by_history_root() -> None:
    def metrics(ret: float):
        return {"early_ret_complete_cohort": ret}

    payload = {
        "pareto_pairs": [
            {
                "config_id": "h4_c64",
                "history_root": root,
                "persistence_mode": "binary_0.9",
                "retained": metrics(0.7 + delta),
                "reset": metrics(0.7),
            }
            for root, delta in ((1, 0.01), (2, 0.02), (3, 0.03))
        ]
    }
    result = audit(
        payload,
        config_id="h4_c64",
        persistence_mode="binary_0.9",
        sesoi=0.01,
        alpha=0.05,
        bootstrap_draws=100,
        bootstrap_seed=7,
    )
    assert result["n_history_roots"] == 3
    assert result["burned_mean_delta"] == pytest.approx(0.02)
    assert result["burned_sd_delta"] == pytest.approx(0.01)
    assert int(result["required_histories"]["0.9"]) >= 1
    assert required_histories(sd=0.0, sesoi=0.01, alpha=0.05, power=0.9) == 1


def test_merge_targeted_recomputes_high_budget_agreement() -> None:
    shards = [
        {
            "rows": [
                {
                    "history_root": 7_570_801 + index,
                    "campaign_index": 1,
                    "persistence_mode": "binary_0.9",
                    "prior_arm": "reset",
                    "c256_action": index,
                    "c1024_action": index,
                    "absolute_planning_value_error": 0.0001 * (index + 1),
                }
            ]
        }
        for index in range(2)
    ]
    merged = merge_targeted(shards)
    assert merged["target_count"] == 2
    assert merged["agreement"] == 1.0
    assert merged["mean_abs_planning_value_error"] == pytest.approx(0.00015)
