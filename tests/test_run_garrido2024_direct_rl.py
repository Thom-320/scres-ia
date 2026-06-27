from __future__ import annotations

import argparse

import pytest

from scripts.run_garrido2024_direct_rl import (
    add_direct_metric_aliases,
    add_policy_direct_metric_aliases,
    build_direct_comparison_rows,
)


def test_add_direct_metric_aliases_adds_cd_and_excel_fields() -> None:
    rows = [
        {
            "order_level_ret_mean": 0.42,
            "policy": "x",
            "steps": 10,
            "ret_garrido2024_sigmoid_total": 7.5,
            "ret_garrido2024_raw_total": 3.0,
            "ret_garrido2024_train_total": 4.0,
        }
    ]

    out = add_direct_metric_aliases(rows)

    assert out[0]["mean_ret_excel_formula"] == 0.42
    assert out[0]["cd_sigmoid_mean"] == pytest.approx(0.75)
    assert out[0]["cd_raw_mean"] == pytest.approx(0.3)
    assert out[0]["cd_train_mean"] == pytest.approx(0.4)
    assert "mean_ret_excel_formula" not in rows[0]


def test_policy_aliases_copy_excel_summary_fields() -> None:
    row = {
        "order_level_ret_mean_mean": 0.5,
        "order_level_ret_mean_std": 0.1,
        "order_level_ret_mean_ci95_low": 0.4,
        "order_level_ret_mean_ci95_high": 0.6,
        "ret_garrido2024_sigmoid_total_mean": 0.75,
        "ret_garrido2024_sigmoid_total_std": 0.01,
        "ret_garrido2024_sigmoid_total_ci95_low": 0.73,
        "ret_garrido2024_sigmoid_total_ci95_high": 0.77,
    }

    out = add_policy_direct_metric_aliases([row])[0]

    assert out["mean_ret_excel_formula_mean"] == 0.5
    assert out["mean_ret_excel_formula_ci95_high"] == 0.6
    assert out["cd_sigmoid_mean_mean"] == 0.75
    assert out["cd_sigmoid_mean_ci95_low"] == 0.73


def test_direct_comparison_uses_cd_primary_and_excel_secondary() -> None:
    args = argparse.Namespace(
        reward_mode="ReT_garrido2024_train",
        algo="ppo",
        risk_level="increased",
    )
    rows = [
        {
            "phase": "ppo_eval",
            "policy": "ppo",
            "cd_sigmoid_mean_mean": 0.58,
            "mean_ret_excel_formula_mean": 0.55,
            "flow_fill_rate_mean": 0.90,
            "service_loss_total_mean": 1.2,
            "shift_cost_total_mean": 0.5,
        },
        {
            "phase": "static_screen",
            "policy": "static_s2",
            "cd_sigmoid_mean_mean": 0.50,
            "mean_ret_excel_formula_mean": 0.50,
            "flow_fill_rate_mean": 0.85,
            "service_loss_total_mean": 1.5,
            "shift_cost_total_mean": 0.7,
        },
        {
            "phase": "static_screen",
            "policy": "garrido_cf_s2",
            "cd_sigmoid_mean_mean": 0.51,
            "mean_ret_excel_formula_mean": 0.51,
            "flow_fill_rate_mean": 0.86,
            "service_loss_total_mean": 1.4,
            "shift_cost_total_mean": 0.7,
        },
        {
            "phase": "heuristic_eval",
            "policy": "heuristic_x",
            "cd_sigmoid_mean_mean": 0.60,
            "mean_ret_excel_formula_mean": 0.60,
            "flow_fill_rate_mean": 0.88,
            "service_loss_total_mean": 1.1,
            "shift_cost_total_mean": 0.6,
        },
    ]

    comparison = build_direct_comparison_rows(rows, args=args)[0]

    assert comparison["primary_metric"] == "cd_sigmoid_mean"
    assert comparison["best_baseline_cd_policy"] == "heuristic_x"
    assert comparison["best_baseline_excel_policy"] == "heuristic_x"
    assert comparison["delta_cd_vs_garrido_cf_s2"] == pytest.approx(0.07)
    assert comparison["delta_cd_vs_best_baseline"] == pytest.approx(-0.02)
    assert comparison["delta_excel_vs_best_baseline"] == pytest.approx(-0.05)
    assert comparison["learned_beats_static_s2_cd"] is True
    assert comparison["learned_beats_best_baseline_cd"] is False
