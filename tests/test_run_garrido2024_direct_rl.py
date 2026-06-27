from __future__ import annotations

import argparse

import pytest

from scripts.run_garrido2024_direct_rl import (
    add_excel_aliases,
    add_policy_excel_aliases,
    build_direct_comparison_rows,
)


def test_add_excel_aliases_uses_order_level_ret_mean() -> None:
    rows = [{"order_level_ret_mean": 0.42, "policy": "x"}]

    out = add_excel_aliases(rows)

    assert out[0]["mean_ret_excel_formula"] == 0.42
    assert "mean_ret_excel_formula" not in rows[0]


def test_policy_aliases_copy_excel_summary_fields() -> None:
    row = {
        "order_level_ret_mean_mean": 0.5,
        "order_level_ret_mean_std": 0.1,
        "order_level_ret_mean_ci95_low": 0.4,
        "order_level_ret_mean_ci95_high": 0.6,
    }

    out = add_policy_excel_aliases([row])[0]

    assert out["mean_ret_excel_formula_mean"] == 0.5
    assert out["mean_ret_excel_formula_ci95_high"] == 0.6


def test_direct_comparison_uses_excel_metric_not_reward() -> None:
    args = argparse.Namespace(
        reward_mode="ReT_garrido2024_train",
        algo="ppo",
        risk_level="increased",
    )
    rows = [
        {
            "phase": "ppo_eval",
            "policy": "ppo",
            "mean_ret_excel_formula_mean": 0.55,
        },
        {
            "phase": "static_screen",
            "policy": "static_s2",
            "mean_ret_excel_formula_mean": 0.50,
        },
        {
            "phase": "static_screen",
            "policy": "garrido_cf_s2",
            "mean_ret_excel_formula_mean": 0.51,
        },
        {
            "phase": "heuristic_eval",
            "policy": "heuristic_x",
            "mean_ret_excel_formula_mean": 0.60,
        },
    ]

    comparison = build_direct_comparison_rows(rows, args=args)[0]

    assert comparison["best_baseline_policy"] == "heuristic_x"
    assert comparison["delta_vs_garrido_cf_s2"] == pytest.approx(0.04)
    assert comparison["delta_vs_best_baseline"] == pytest.approx(-0.05)
    assert comparison["learned_beats_static_s2"] is True
    assert comparison["learned_beats_best_baseline"] is False
