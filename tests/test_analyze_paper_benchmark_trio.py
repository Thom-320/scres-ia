from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.analyze_paper_benchmark_trio import (
    build_comparable_row,
    build_status_row,
    choose_pragmatic_leader,
)


def _write_csv(
    path: Path, fieldnames: list[str], rows: list[dict[str, object]]
) -> None:
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_build_status_row_reports_pending_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "paper_ret_seq_k010_500k"
    run_dir.mkdir()
    (run_dir / "status.json").write_text(
        json.dumps(
            {
                "state": "running",
                "started_at_utc": "2026-03-26T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "heartbeat.json").write_text(
        json.dumps({"last_activity_utc": "2026-03-26T01:00:00+00:00"}),
        encoding="utf-8",
    )

    row = build_status_row(run_dir)

    assert row["label"] == "paper_ret_seq_k010_500k"
    assert row["state"] == "running"
    assert row["summary_exists"] is False
    assert row["reward_mode"] == "unknown"


def test_build_comparable_row_extracts_static_s2_comparison(tmp_path: Path) -> None:
    run_dir = tmp_path / "paper_control_v1_500k"
    run_dir.mkdir()
    (run_dir / "status.json").write_text(
        json.dumps({"state": "completed"}), encoding="utf-8"
    )
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "config": {
                    "algo": "ppo",
                    "reward_mode": "control_v1",
                },
                "reward_contract": {
                    "reward_family": "operational_penalty",
                },
            }
        ),
        encoding="utf-8",
    )
    rows = [
        {
            "phase": "static_screen",
            "policy": "static_s2",
            "fill_rate_mean": "0.840",
            "backorder_rate_mean": "0.160",
            "order_level_ret_mean_mean": "0.780",
            "pct_steps_S1_mean": "0.0",
            "pct_steps_S2_mean": "100.0",
            "pct_steps_S3_mean": "0.0",
        },
        {
            "phase": "ppo_eval",
            "policy": "ppo",
            "fill_rate_mean": "0.835",
            "backorder_rate_mean": "0.165",
            "order_level_ret_mean_mean": "0.790",
            "pct_steps_S1_mean": "10.0",
            "pct_steps_S2_mean": "30.0",
            "pct_steps_S3_mean": "60.0",
        },
    ]
    _write_csv(run_dir / "policy_summary.csv", list(rows[0].keys()), rows)

    row = build_comparable_row(run_dir)

    assert row is not None
    assert row["reward_mode"] == "control_v1"
    assert row["reward_family"] == "operational_penalty"
    assert row["delta_fill_rate_vs_static_s2"] == pytest.approx(-0.005)
    assert row["delta_backorder_rate_vs_static_s2"] == pytest.approx(0.005)
    assert row["delta_order_level_ret_mean_vs_static_s2"] == pytest.approx(0.01)


def test_choose_pragmatic_leader_prefers_ret_then_fill_then_backorder() -> None:
    leader = choose_pragmatic_leader(
        [
            {
                "label": "a",
                "order_level_ret_mean": 0.78,
                "fill_rate_mean": 0.84,
                "backorder_rate_mean": 0.16,
            },
            {
                "label": "b",
                "order_level_ret_mean": 0.79,
                "fill_rate_mean": 0.83,
                "backorder_rate_mean": 0.17,
            },
        ]
    )

    assert leader is not None
    assert leader["label"] == "b"
