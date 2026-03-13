from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.benchmark_delta_sweep_static import (
    aggregate_policy_metrics,
    aggregate_seed_metrics,
    build_delta_transition_rows,
    build_parser,
    run_delta_sweep,
    static_policy_action,
)
from supply_chain.external_env_interface import make_shift_control_env


def test_static_policy_actions_select_expected_shifts() -> None:
    env = make_shift_control_env(
        step_size_hours=24, max_steps=1, risk_level="increased", rt_delta=0.06
    )
    env.reset(seed=7)
    _, _, _, _, s1_info = env.step(static_policy_action("static_s1"))
    assert s1_info["shifts_active"] == 1

    env.reset(seed=7)
    _, _, _, _, s2_info = env.step(static_policy_action("static_s2"))
    assert s2_info["shifts_active"] == 2


def test_aggregate_transition_metrics_compute_expected_values() -> None:
    episode_rows = [
        {
            "delta": 0.04,
            "policy": "static_s1",
            "seed": 11,
            "episode": 1,
            "eval_seed": 70011,
            "steps": 4,
            "reward_total": 5.0,
            "ret_raw_total": 5.0,
            "fill_rate": 0.8,
            "backorder_rate": 0.2,
            "shift_cost_total": 0.0,
            "disruption_hours_total": 1.0,
            "mean_step_fill_rate": 0.8,
            "mean_disruption_fraction": 0.1,
            "avg_inventory": 100.0,
            "pct_fill_rate_only": 75.0,
            "pct_autotomy": 0.0,
            "pct_recovery": 25.0,
            "pct_non_recovery": 0.0,
            "pct_no_demand": 0.0,
            "demanded_total": 100.0,
            "delivered_total": 80.0,
            "backorder_qty_total": 20.0,
        },
        {
            "delta": 0.04,
            "policy": "static_s2",
            "seed": 11,
            "episode": 1,
            "eval_seed": 70011,
            "steps": 4,
            "reward_total": 5.5,
            "ret_raw_total": 5.7,
            "fill_rate": 0.95,
            "backorder_rate": 0.05,
            "shift_cost_total": 0.16,
            "disruption_hours_total": 0.5,
            "mean_step_fill_rate": 0.95,
            "mean_disruption_fraction": 0.05,
            "avg_inventory": 130.0,
            "pct_fill_rate_only": 100.0,
            "pct_autotomy": 0.0,
            "pct_recovery": 0.0,
            "pct_non_recovery": 0.0,
            "pct_no_demand": 0.0,
            "demanded_total": 100.0,
            "delivered_total": 95.0,
            "backorder_qty_total": 5.0,
        },
        {
            "delta": 0.06,
            "policy": "static_s1",
            "seed": 11,
            "episode": 1,
            "eval_seed": 70011,
            "steps": 4,
            "reward_total": 5.0,
            "ret_raw_total": 5.0,
            "fill_rate": 0.8,
            "backorder_rate": 0.2,
            "shift_cost_total": 0.0,
            "disruption_hours_total": 1.0,
            "mean_step_fill_rate": 0.8,
            "mean_disruption_fraction": 0.1,
            "avg_inventory": 100.0,
            "pct_fill_rate_only": 75.0,
            "pct_autotomy": 0.0,
            "pct_recovery": 25.0,
            "pct_non_recovery": 0.0,
            "pct_no_demand": 0.0,
            "demanded_total": 100.0,
            "delivered_total": 80.0,
            "backorder_qty_total": 20.0,
        },
        {
            "delta": 0.06,
            "policy": "static_s2",
            "seed": 11,
            "episode": 1,
            "eval_seed": 70011,
            "steps": 4,
            "reward_total": 4.7,
            "ret_raw_total": 5.7,
            "fill_rate": 0.95,
            "backorder_rate": 0.05,
            "shift_cost_total": 0.24,
            "disruption_hours_total": 0.5,
            "mean_step_fill_rate": 0.95,
            "mean_disruption_fraction": 0.05,
            "avg_inventory": 130.0,
            "pct_fill_rate_only": 100.0,
            "pct_autotomy": 0.0,
            "pct_recovery": 0.0,
            "pct_non_recovery": 0.0,
            "pct_no_demand": 0.0,
            "demanded_total": 100.0,
            "delivered_total": 95.0,
            "backorder_qty_total": 5.0,
        },
    ]

    seed_rows = aggregate_seed_metrics(episode_rows)
    policy_rows = aggregate_policy_metrics(seed_rows)
    transition_rows = build_delta_transition_rows(policy_rows)

    delta_004 = next(
        row for row in transition_rows if row["delta"] == pytest.approx(0.04)
    )
    assert delta_004["reward_gap_s2_minus_s1"] == pytest.approx(0.5)
    assert delta_004["ret_raw_gap_s2_minus_s1"] == pytest.approx(0.7)
    assert delta_004["avg_inventory_gap_s2_minus_s1"] == pytest.approx(30.0)
    assert delta_004["preferred_policy_by_reward"] == "static_s2"

    delta_006 = next(
        row for row in transition_rows if row["delta"] == pytest.approx(0.06)
    )
    assert delta_006["reward_gap_s2_minus_s1"] == pytest.approx(-0.3)
    assert delta_006["fill_rate_gap_s2_minus_s1"] == pytest.approx(0.15)
    assert delta_006["disruption_hours_gap_s2_minus_s1"] == pytest.approx(-0.5)
    assert delta_006["preferred_policy_by_reward"] == "static_s1"


def test_run_delta_sweep_smoke_writes_expected_artifacts(tmp_path: Path) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--deltas",
            "0.04",
            "0.06",
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
    summary = run_delta_sweep(args)

    episode_csv = tmp_path / "episode_metrics.csv"
    policy_csv = tmp_path / "policy_summary.csv"
    transition_csv = tmp_path / "delta_transition.csv"
    summary_json = tmp_path / "summary.json"

    assert episode_csv.exists()
    assert policy_csv.exists()
    assert transition_csv.exists()
    assert summary_json.exists()
    assert summary["policies"] == ["static_s1", "static_s2"]

    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    assert payload["config"]["deltas"] == [0.04, 0.06]
    assert len(payload["delta_transition"]) == 2
