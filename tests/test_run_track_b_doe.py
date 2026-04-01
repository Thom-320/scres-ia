from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import scripts.run_track_b_doe as track_b_doe


def test_build_policy_grid_and_direct_action_use_downstream_multiplier() -> None:
    grid = track_b_doe.build_policy_grid((2,), (0.5, 2.0))

    assert [policy.label for policy in grid] == ["s2_d0.50", "s2_d2.00"]
    action = track_b_doe.build_direct_policy_action(grid[1])
    assert action["assembly_shifts"] == 2
    assert action["op10_q_min"] == pytest.approx(4800.0)
    assert action["op10_q_max"] == pytest.approx(5200.0)
    assert action["op12_q_min"] == pytest.approx(4800.0)
    assert action["op12_q_max"] == pytest.approx(5200.0)


def test_build_decision_summary_uses_s2_neutral_as_baseline() -> None:
    summary_rows = [
        {
            "policy": "s2_d1.00",
            "assembly_shifts": 2,
            "downstream_multiplier": 1.0,
            "reward_total_mean": 100.0,
            "fill_rate_mean": 0.80,
        },
        {
            "policy": "s2_d2.00",
            "assembly_shifts": 2,
            "downstream_multiplier": 2.0,
            "reward_total_mean": 102.5,
            "fill_rate_mean": 0.815,
        },
        {
            "policy": "s3_d2.00",
            "assembly_shifts": 3,
            "downstream_multiplier": 2.0,
            "reward_total_mean": 101.0,
            "fill_rate_mean": 0.812,
        },
    ]

    decision = track_b_doe.build_decision_summary(summary_rows)

    assert decision["baseline_policy"] == "s2_d1.00"
    assert decision["best_by_fill"] == "s2_d2.00"
    assert decision["best_by_reward"] == "s2_d2.00"
    assert decision["delta_fill_pp_vs_s2_neutral"] == pytest.approx(1.5)
    assert decision["headroom_open_by_fill"] is True
    assert decision["headroom_open_by_reward"] is True


def test_main_writes_bundle_from_stubbed_episode_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "track_b_doe"

    def fake_run_static_policy_episode(
        policy: track_b_doe.PolicySpec, *, seed: int, env_kwargs: dict[str, object]
    ) -> dict[str, object]:
        del env_kwargs
        reward = (
            100.0
            + (10.0 * policy.downstream_multiplier)
            + float(policy.assembly_shifts)
        )
        fill = 0.75 + (0.01 * policy.downstream_multiplier)
        return {
            "policy": policy.label,
            "seed": seed,
            "assembly_shifts": policy.assembly_shifts,
            "downstream_multiplier": policy.downstream_multiplier,
            "reward_total": reward,
            "fill_rate": fill,
            "backorder_rate": 1.0 - fill,
            "order_level_ret_mean": fill,
            "flow_fill_rate": fill,
            "flow_backorder_rate": 1.0 - fill,
            "terminal_rolling_fill_rate_4w": fill,
            "terminal_rolling_backorder_rate_4w": 1.0 - fill,
        }

    monkeypatch.setattr(
        track_b_doe, "run_static_policy_episode", fake_run_static_policy_episode
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_track_b_doe.py",
            "--output-dir",
            str(output_dir),
            "--seeds",
            "11",
            "22",
            "--shift-levels",
            "2",
            "3",
            "--downstream-multipliers",
            "1.0",
            "2.0",
        ],
    )

    track_b_doe.main()

    payload = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    summary_rows = payload["summary_rows"]

    assert (output_dir / "seed_metrics.csv").exists()
    assert (output_dir / "policy_summary.csv").exists()
    assert (output_dir / "summary.md").exists()
    assert payload["decision"]["baseline_policy"] == "s2_d1.00"
    assert len(summary_rows) == 4
    assert payload["decision"]["best_by_reward"] == "s3_d2.00"
