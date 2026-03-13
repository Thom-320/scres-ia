from __future__ import annotations

import json
from pathlib import Path

import pytest

from supply_chain.external_env_interface import make_shift_control_env
from scripts.benchmark_minimal_shift_control import (
    aggregate_policy_metrics,
    aggregate_seed_metrics,
    build_parser,
    run_benchmark,
    static_policy_action,
)


def test_static_policy_actions_select_expected_shifts() -> None:
    env = make_shift_control_env(
        step_size_hours=24, max_steps=1, risk_level="increased"
    )
    env.reset(seed=7)
    _, _, _, _, s1_info = env.step(static_policy_action("static_s1"))
    assert s1_info["shifts_active"] == 1

    env.reset(seed=7)
    _, _, _, _, s2_info = env.step(static_policy_action("static_s2"))
    assert s2_info["shifts_active"] == 2


def test_aggregate_metrics_compute_expected_values() -> None:
    episode_rows = [
        {
            "policy": "static_s1",
            "seed": 11,
            "episode": 1,
            "eval_seed": 50011,
            "steps": 4,
            "reward_total": 5.0,
            "fill_rate": 0.9,
            "backorder_rate": 0.1,
            "cost_total": 0.0,
            "cost_mean": 0.0,
            "demanded_total": 100.0,
            "delivered_total": 90.0,
            "backorder_qty_total": 10.0,
        },
        {
            "policy": "static_s1",
            "seed": 11,
            "episode": 2,
            "eval_seed": 50012,
            "steps": 4,
            "reward_total": 7.0,
            "fill_rate": 0.8,
            "backorder_rate": 0.2,
            "cost_total": 0.0,
            "cost_mean": 0.0,
            "demanded_total": 100.0,
            "delivered_total": 80.0,
            "backorder_qty_total": 20.0,
        },
        {
            "policy": "ppo",
            "seed": 11,
            "episode": 1,
            "eval_seed": 50011,
            "steps": 4,
            "reward_total": 10.0,
            "fill_rate": 0.95,
            "backorder_rate": 0.05,
            "cost_total": 0.24,
            "cost_mean": 0.06,
            "demanded_total": 100.0,
            "delivered_total": 95.0,
            "backorder_qty_total": 5.0,
        },
    ]

    seed_rows = aggregate_seed_metrics(episode_rows)
    s1_seed = next(row for row in seed_rows if row["policy"] == "static_s1")
    assert s1_seed["reward_total_mean"] == pytest.approx(6.0)
    assert s1_seed["fill_rate_mean"] == pytest.approx(0.85)
    assert s1_seed["backorder_rate_mean"] == pytest.approx(0.15)
    assert s1_seed["cost_total_mean"] == pytest.approx(0.0)

    policy_rows = aggregate_policy_metrics(seed_rows)
    ppo_policy = next(row for row in policy_rows if row["policy"] == "ppo")
    assert ppo_policy["reward_total_mean"] == pytest.approx(10.0)
    assert ppo_policy["fill_rate_mean"] == pytest.approx(0.95)
    assert ppo_policy["backorder_rate_mean"] == pytest.approx(0.05)
    assert ppo_policy["cost_mean_mean"] == pytest.approx(0.06)


def test_run_benchmark_smoke_writes_expected_artifacts(tmp_path: Path) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--seeds",
            "1",
            "--train-timesteps",
            "32",
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
    summary = run_benchmark(args)

    episode_csv = tmp_path / "episode_metrics.csv"
    policy_csv = tmp_path / "policy_summary.csv"
    summary_json = tmp_path / "summary.json"

    assert episode_csv.exists()
    assert policy_csv.exists()
    assert summary_json.exists()
    assert summary["policies"] == ["static_s1", "static_s2", "ppo"]

    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    assert payload["config"]["train_timesteps"] == 32
    assert len(payload["policy_summary"]) == 3
