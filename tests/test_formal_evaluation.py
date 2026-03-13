from __future__ import annotations

from collections import Counter

import pytest

from scripts.formal_evaluation import (
    aggregate_policy_metrics,
    aggregate_seed_metrics,
    build_comparison_rows,
    finalize_episode_metrics,
)


def test_finalize_episode_metrics_computes_percentages() -> None:
    row = finalize_episode_metrics(
        policy="trained",
        seed=7,
        episode=1,
        eval_seed=50007,
        steps=4,
        reward_total=10.0,
        delivered_total=90.0,
        demanded_total=100.0,
        backorder_qty_total=10.0,
        disruption_hours_total=8.0,
        inventory_values=[10.0, 20.0, 30.0, 40.0],
        ret_values=[0.9, 1.0, 0.8, 0.7],
        step_fill_rates=[0.9, 0.95, 0.85, 0.8],
        shift_cost_values=[0.0, 0.06, 0.06, 0.12],
        shift_counts=Counter({1: 1, 2: 2, 3: 1}),
        ret_case_counts=Counter(
            {
                "fill_rate_only": 1,
                "autotomy": 1,
                "recovery": 1,
                "non_recovery": 1,
            }
        ),
    )
    assert row["fill_rate_episode"] == pytest.approx(0.9)
    assert row["service_loss_episode"] == pytest.approx(0.1)
    assert row["avg_inventory"] == pytest.approx(25.0)
    assert row["mean_ReT"] == pytest.approx(0.85)
    assert row["mean_step_fill_rate"] == pytest.approx(0.875)
    assert row["mean_shift_cost"] == pytest.approx(0.06)
    assert row["pct_steps_S2"] == pytest.approx(50.0)
    assert row["pct_non_recovery"] == pytest.approx(25.0)


def test_aggregate_seed_and_policy_metrics() -> None:
    rows = [
        {
            "policy": "trained",
            "seed": 7,
            "episode": 1,
            "eval_seed": 50007,
            "steps": 4,
            "reward_total": 10.0,
            "delivered_total": 90.0,
            "demanded_total": 100.0,
            "backorder_qty_total": 10.0,
            "fill_rate_episode": 0.9,
            "service_loss_episode": 0.1,
            "disruption_hours_total": 8.0,
            "avg_inventory": 25.0,
            "mean_ReT": 0.85,
            "mean_step_fill_rate": 0.88,
            "mean_shift_cost": 0.06,
            "pct_steps_S1": 25.0,
            "pct_steps_S2": 50.0,
            "pct_steps_S3": 25.0,
            "pct_fill_rate_only": 25.0,
            "pct_autotomy": 25.0,
            "pct_recovery": 25.0,
            "pct_non_recovery": 25.0,
            "pct_no_demand": 0.0,
        },
        {
            "policy": "trained",
            "seed": 7,
            "episode": 2,
            "eval_seed": 50008,
            "steps": 4,
            "reward_total": 12.0,
            "delivered_total": 95.0,
            "demanded_total": 100.0,
            "backorder_qty_total": 5.0,
            "fill_rate_episode": 0.95,
            "service_loss_episode": 0.05,
            "disruption_hours_total": 6.0,
            "avg_inventory": 20.0,
            "mean_ReT": 0.9,
            "mean_step_fill_rate": 0.92,
            "mean_shift_cost": 0.03,
            "pct_steps_S1": 50.0,
            "pct_steps_S2": 50.0,
            "pct_steps_S3": 0.0,
            "pct_fill_rate_only": 50.0,
            "pct_autotomy": 25.0,
            "pct_recovery": 25.0,
            "pct_non_recovery": 0.0,
            "pct_no_demand": 0.0,
        },
        {
            "policy": "random",
            "seed": 7,
            "episode": 1,
            "eval_seed": 50007,
            "steps": 4,
            "reward_total": 8.0,
            "delivered_total": 80.0,
            "demanded_total": 100.0,
            "backorder_qty_total": 20.0,
            "fill_rate_episode": 0.8,
            "service_loss_episode": 0.2,
            "disruption_hours_total": 9.0,
            "avg_inventory": 30.0,
            "mean_ReT": 0.75,
            "mean_step_fill_rate": 0.8,
            "mean_shift_cost": 0.05,
            "pct_steps_S1": 75.0,
            "pct_steps_S2": 25.0,
            "pct_steps_S3": 0.0,
            "pct_fill_rate_only": 25.0,
            "pct_autotomy": 25.0,
            "pct_recovery": 50.0,
            "pct_non_recovery": 0.0,
            "pct_no_demand": 0.0,
        },
    ]

    seed_rows = aggregate_seed_metrics(rows)
    assert len(seed_rows) == 2
    trained_seed = next(row for row in seed_rows if row["policy"] == "trained")
    assert trained_seed["reward_total_mean"] == pytest.approx(11.0)
    assert trained_seed["mean_ReT_mean"] == pytest.approx(0.875)

    policy_metrics = aggregate_policy_metrics(seed_rows)
    assert policy_metrics["trained"]["seed_count"] == 1
    assert policy_metrics["trained"]["reward_total_mean"]["mean"] == pytest.approx(11.0)

    comparison_rows = build_comparison_rows(seed_rows)
    trained_comparison = next(
        row for row in comparison_rows if row["policy"] == "trained"
    )
    assert trained_comparison["reward_mean"] == pytest.approx(11.0)
