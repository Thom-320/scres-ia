from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.benchmark_ret_ablation_static import (
    build_parser,
    build_transition_rows,
    compute_ret_ablation_components,
    run_ret_ablation,
)


def test_compute_ret_ablation_components_matches_expected_modes() -> None:
    base_kwargs = {
        "demanded": 100.0,
        "backorder_qty": 2.0,
        "disruption_fraction": 0.073,
        "autotomy_threshold": 0.95,
        "nonrecovery_disruption_threshold": 0.5,
        "nonrecovery_fill_rate_threshold": 0.5,
    }

    default = compute_ret_ablation_components(**base_kwargs, formula_mode="default")
    assert default["ret_case"] == "autotomy"
    assert default["ret_value"] == pytest.approx(0.927)

    unified = compute_ret_ablation_components(
        **base_kwargs,
        formula_mode="autotomy_equals_recovery",
    )
    assert unified["ret_case"] == "autotomy"
    assert unified["ret_value"] == pytest.approx(1.0 / 1.073)

    merged = compute_ret_ablation_components(
        **base_kwargs,
        formula_mode="merged_recovery_formula",
    )
    assert merged["ret_case"] == "autotomy"
    assert merged["ret_value"] == pytest.approx(1.0 / 1.073)


def test_build_transition_rows_reports_expected_gap_direction() -> None:
    policy_rows = [
        {
            "autotomy_threshold": 0.95,
            "formula_mode": "default",
            "rt_delta": 0.0,
            "policy": "static_s1",
            "reward_total_mean": 10.0,
            "ret_raw_total_mean": 10.0,
            "fill_rate_mean": 0.8,
            "backorder_rate_mean": 0.2,
            "mean_step_fill_rate_mean": 0.8,
            "mean_disruption_fraction_mean": 0.07,
            "avg_inventory_mean": 100.0,
            "pct_autotomy_mean": 20.0,
            "pct_recovery_mean": 80.0,
            "shift_cost_total_mean": 0.0,
        },
        {
            "autotomy_threshold": 0.95,
            "formula_mode": "default",
            "rt_delta": 0.0,
            "policy": "static_s2",
            "reward_total_mean": 9.5,
            "ret_raw_total_mean": 9.5,
            "fill_rate_mean": 0.95,
            "backorder_rate_mean": 0.05,
            "mean_step_fill_rate_mean": 0.95,
            "mean_disruption_fraction_mean": 0.06,
            "avg_inventory_mean": 130.0,
            "pct_autotomy_mean": 60.0,
            "pct_recovery_mean": 40.0,
            "shift_cost_total_mean": 0.0,
        },
        {
            "autotomy_threshold": 0.95,
            "formula_mode": "autotomy_equals_recovery",
            "rt_delta": 0.0,
            "policy": "static_s1",
            "reward_total_mean": 10.0,
            "ret_raw_total_mean": 10.0,
            "fill_rate_mean": 0.8,
            "backorder_rate_mean": 0.2,
            "mean_step_fill_rate_mean": 0.8,
            "mean_disruption_fraction_mean": 0.07,
            "avg_inventory_mean": 100.0,
            "pct_autotomy_mean": 20.0,
            "pct_recovery_mean": 80.0,
            "shift_cost_total_mean": 0.0,
        },
        {
            "autotomy_threshold": 0.95,
            "formula_mode": "autotomy_equals_recovery",
            "rt_delta": 0.0,
            "policy": "static_s2",
            "reward_total_mean": 10.4,
            "ret_raw_total_mean": 10.4,
            "fill_rate_mean": 0.95,
            "backorder_rate_mean": 0.05,
            "mean_step_fill_rate_mean": 0.95,
            "mean_disruption_fraction_mean": 0.06,
            "avg_inventory_mean": 130.0,
            "pct_autotomy_mean": 60.0,
            "pct_recovery_mean": 40.0,
            "shift_cost_total_mean": 0.0,
        },
    ]

    rows = build_transition_rows(policy_rows)
    default = next(
        row
        for row in rows
        if row["formula_mode"] == "default" and row["rt_delta"] == pytest.approx(0.0)
    )
    assert default["reward_gap_s2_minus_s1"] == pytest.approx(-0.5)
    assert default["preferred_policy_by_reward"] == "static_s1"

    unified = next(
        row
        for row in rows
        if row["formula_mode"] == "autotomy_equals_recovery"
        and row["rt_delta"] == pytest.approx(0.0)
    )
    assert unified["reward_gap_s2_minus_s1"] == pytest.approx(0.4)
    assert unified["preferred_policy_by_reward"] == "static_s2"


def test_run_ret_ablation_smoke_writes_expected_artifacts(tmp_path: Path) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--autotomy-thresholds",
            "0.95",
            "0.90",
            "--formula-modes",
            "default",
            "autotomy_equals_recovery",
            "--rt-deltas",
            "0.0",
            "--seeds",
            "1",
            "--eval-episodes",
            "1",
            "--step-size-hours",
            "24",
            "--max-steps",
            "4",
            "--output-dir",
            str(tmp_path),
        ]
    )
    summary = run_ret_ablation(args)

    assert (tmp_path / "episode_metrics.csv").exists()
    assert (tmp_path / "policy_summary.csv").exists()
    assert (tmp_path / "transition_summary.csv").exists()
    assert (tmp_path / "case_summary.csv").exists()
    assert (tmp_path / "fill_rate_only_buckets.csv").exists()
    assert (tmp_path / "summary.json").exists()
    assert summary["policies"] == ["static_s1", "static_s2"]

    payload = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert payload["config"]["autotomy_thresholds"] == [0.95, 0.9]
    assert payload["config"]["formula_modes"] == [
        "default",
        "autotomy_equals_recovery",
    ]
    assert "case_summary" in payload
    assert "fill_rate_only_buckets" in payload
