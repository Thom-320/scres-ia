from __future__ import annotations

import json
import subprocess
import sys

from scripts.audit_thesis_reward_surface import build_parser
from supply_chain.config import TRACK_A_TRAINING_DOWNSTREAM_Q_SOURCE


def test_reward_surface_audit_defaults_to_frozen_training_downstream_source() -> None:
    args = build_parser().parse_args([])
    assert args.downstream_q_source == TRACK_A_TRAINING_DOWNSTREAM_Q_SOURCE


def test_reward_surface_audit_runs_without_dkana_torch(tmp_path) -> None:
    label = "pytest_reward_surface"
    cmd = [
        sys.executable,
        "scripts/audit_thesis_reward_surface.py",
        "--label",
        label,
        "--output-root",
        str(tmp_path),
        "--reward-modes",
        "ReT_thesis",
        "control_v1",
        "--replications",
        "1",
        "--max-steps",
        "1",
        "--risk-level",
        "current",
        "--downstream-q-source",
        "table_6_20",
        "--progress-every",
        "1000",
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)

    run_dir = tmp_path / label
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    policy_summary = (run_dir / "policy_summary.csv").read_text(encoding="utf-8")
    mode_summary = (run_dir / "reward_mode_summary.csv").read_text(encoding="utf-8")

    assert summary["episode_count"] == 36
    assert summary["policy_count"] == 36
    assert summary["config"]["downstream_q_source"] == "table_6_20"
    assert "downstream_q_source" in (run_dir / "episode_metrics.csv").read_text(
        encoding="utf-8"
    )
    assert "service_loss_area_mean" in policy_summary
    assert "garrido2024_step_cost_total_mean" in policy_summary
    assert "selection_gate" in mode_summary
    assert "spearman_reward_vs_negative_service_loss_area" in mode_summary


def test_downstream_q_comparator_shortlists_stable_rewards(tmp_path) -> None:
    header = [
        "reward_mode",
        "diagnostic_score",
        "reward_spread_ratio",
        "spearman_reward_vs_order_level_ret",
        "spearman_reward_vs_fill",
        "spearman_reward_vs_negative_service_loss_area",
        "spearman_reward_vs_negative_pending_backlog",
        "spearman_reward_vs_negative_step_cost",
        "selection_gate",
        "best_policy_by_reward",
        "best_policy_shifts",
        "best_policy_order_level_ret",
        "best_policy_fill_rate",
        "best_policy_service_loss_area",
        "best_policy_step_cost",
    ]
    figure_dir = tmp_path / "figure"
    table_dir = tmp_path / "table"
    figure_dir.mkdir()
    table_dir.mkdir()
    figure_rows = [
        ["control_v1", "3.0", "1.0", "0.9", "0.8", "0.7", "0.6", "0.5", "shortlist", "P1", "2", "0.5", "0.9", "1.0", "10.0"],
        ["ReT_seq_v1", "2.0", "1.0", "0.9", "0.8", "0.7", "0.6", "0.5", "shortlist", "P2", "2", "0.5", "0.9", "1.0", "10.0"],
    ]
    table_rows = [
        ["control_v1", "2.8", "1.0", "0.9", "0.8", "0.7", "0.6", "0.5", "shortlist", "P1", "2", "0.5", "0.9", "1.0", "10.0"],
        ["ReT_seq_v1", "2.1", "1.0", "0.9", "0.8", "0.7", "0.6", "0.5", "shortlist", "P3", "2", "0.5", "0.9", "1.0", "10.0"],
    ]
    for path, rows in (
        (figure_dir / "reward_mode_summary.csv", figure_rows),
        (table_dir / "reward_mode_summary.csv", table_rows),
    ):
        path.write_text(
            ",".join(header) + "\n" + "\n".join(",".join(row) for row in rows) + "\n",
            encoding="utf-8",
        )
    (figure_dir / "summary.json").write_text('{"config": {"downstream_q_source": "figure_6_2"}}', encoding="utf-8")
    (table_dir / "summary.json").write_text('{"config": {"downstream_q_source": "table_6_20"}}', encoding="utf-8")

    output_root = tmp_path / "out"
    cmd = [
        sys.executable,
        "scripts/compare_downstream_q_reward_surface.py",
        "--figure-dir",
        str(figure_dir),
        "--table-dir",
        str(table_dir),
        "--output-root",
        str(output_root),
        "--label",
        "pytest_compare",
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)

    summary = json.loads(
        (output_root / "pytest_compare" / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["shortlist"] == ["control_v1"]
