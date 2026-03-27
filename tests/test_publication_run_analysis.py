from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.publication_run_analysis import analyze_run


def _write_csv(
    path: Path, fieldnames: list[str], rows: list[dict[str, object]]
) -> None:
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_analyze_run_builds_baseline_and_comparison_rows(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    summary = {
        "config": {
            "algo": "ppo",
            "frame_stack": 1,
            "observation_version": "v1",
            "risk_level": "severe",
            "reward_mode": "ReT_seq_v1",
        },
        "reward_contract": {
            "reward_family": "resilience_index",
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    policy_summary_rows = [
        {
            "phase": "static_screen",
            "policy": "static_s1",
            "w_bo": "5.0",
            "w_cost": "0.03",
            "w_disr": "0.0",
            "reward_total_mean": "10.0",
            "fill_rate_mean": "0.70",
            "backorder_rate_mean": "0.30",
            "ret_thesis_corrected_total_mean": "200.0",
        },
        {
            "phase": "static_screen",
            "policy": "static_s2",
            "w_bo": "5.0",
            "w_cost": "0.03",
            "w_disr": "0.0",
            "reward_total_mean": "15.0",
            "fill_rate_mean": "0.82",
            "backorder_rate_mean": "0.18",
            "ret_thesis_corrected_total_mean": "220.0",
        },
        {
            "phase": "static_screen",
            "policy": "static_s3",
            "w_bo": "5.0",
            "w_cost": "0.03",
            "w_disr": "0.0",
            "reward_total_mean": "14.0",
            "fill_rate_mean": "0.80",
            "backorder_rate_mean": "0.20",
            "ret_thesis_corrected_total_mean": "215.0",
        },
        {
            "phase": "random_eval",
            "policy": "random",
            "w_bo": "5.0",
            "w_cost": "0.03",
            "w_disr": "0.0",
            "reward_total_mean": "8.0",
            "fill_rate_mean": "0.65",
            "backorder_rate_mean": "0.35",
            "ret_thesis_corrected_total_mean": "190.0",
        },
        {
            "phase": "heuristic_eval",
            "policy": "heuristic_tuned",
            "w_bo": "5.0",
            "w_cost": "0.03",
            "w_disr": "0.0",
            "reward_total_mean": "16.0",
            "fill_rate_mean": "0.81",
            "backorder_rate_mean": "0.19",
            "ret_thesis_corrected_total_mean": "225.0",
        },
        {
            "phase": "ppo_eval",
            "policy": "ppo",
            "w_bo": "5.0",
            "w_cost": "0.03",
            "w_disr": "0.0",
            "reward_total_mean": "18.0",
            "fill_rate_mean": "0.83",
            "backorder_rate_mean": "0.17",
            "ret_thesis_corrected_total_mean": "230.0",
        },
    ]
    _write_csv(
        run_dir / "policy_summary.csv",
        list(policy_summary_rows[0].keys()),
        policy_summary_rows,
    )

    comparison_rows = [
        {
            "algo": "ppo",
            "frame_stack": "1",
            "observation_version": "v1",
            "learned_policy": "ppo",
            "w_bo": "5.0",
            "w_cost": "0.03",
            "w_disr": "0.0",
            "best_static_policy": "static_s2",
            "best_heuristic_policy": "heuristic_tuned",
        }
    ]
    _write_csv(
        run_dir / "comparison_table.csv",
        list(comparison_rows[0].keys()),
        comparison_rows,
    )

    episode_rows = [
        {
            "phase": "ppo_eval",
            "policy": "ppo",
            "seed": "11",
            "w_bo": "5.0",
            "w_cost": "0.03",
            "w_disr": "0.0",
            "reward_total": "20.0",
        },
        {
            "phase": "ppo_eval",
            "policy": "ppo",
            "seed": "22",
            "w_bo": "5.0",
            "w_cost": "0.03",
            "w_disr": "0.0",
            "reward_total": "18.0",
        },
        {
            "phase": "static_screen",
            "policy": "static_s2",
            "seed": "11",
            "w_bo": "5.0",
            "w_cost": "0.03",
            "w_disr": "0.0",
            "reward_total": "15.0",
        },
        {
            "phase": "static_screen",
            "policy": "static_s2",
            "seed": "22",
            "w_bo": "5.0",
            "w_cost": "0.03",
            "w_disr": "0.0",
            "reward_total": "14.0",
        },
        {
            "phase": "heuristic_eval",
            "policy": "heuristic_tuned",
            "seed": "11",
            "w_bo": "5.0",
            "w_cost": "0.03",
            "w_disr": "0.0",
            "reward_total": "16.0",
        },
        {
            "phase": "heuristic_eval",
            "policy": "heuristic_tuned",
            "seed": "22",
            "w_bo": "5.0",
            "w_cost": "0.03",
            "w_disr": "0.0",
            "reward_total": "15.0",
        },
    ]
    _write_csv(
        run_dir / "episode_metrics.csv", list(episode_rows[0].keys()), episode_rows
    )

    baseline_rows, comparison_summary_rows, run_payload = analyze_run(
        run_dir,
        bootstrap_samples=1000,
        rng=__import__("numpy").random.default_rng(7),
    )

    assert run_payload["run_label"] == "ppo_fs1_v1"
    assert len(baseline_rows) == 6
    assert len(comparison_summary_rows) == 1
    assert comparison_summary_rows[0]["best_static_policy"] == "static_s2"
    assert comparison_summary_rows[0]["best_heuristic_policy"] == "heuristic_tuned"
    assert comparison_summary_rows[0]["reward_mode"] == "ReT_seq_v1"
    assert comparison_summary_rows[0]["reward_family"] == "resilience_index"
    assert comparison_summary_rows[0]["mean_diff_vs_best_static"] > 0.0
    assert comparison_summary_rows[0]["mean_diff_vs_best_heuristic"] > 0.0
