from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

import scripts.generate_track_b_results_package as package
import scripts.posthoc_track_b_resilience_audit as posthoc


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _episode_row(policy: str, seed: int) -> dict[str, object]:
    fill = 0.99 if policy == "ppo" else 0.96
    return {
        "policy": policy,
        "seed": seed,
        "episode": 1,
        "eval_seed": seed + 50_000,
        "steps": 10,
        "reward_total": 200.0 if policy == "ppo" else 150.0,
        "fill_rate": fill,
        "backorder_rate": 1.0 - fill,
        "order_level_ret_mean": 0.90 if policy == "ppo" else 0.50,
        "ret_thesis_corrected_total": 240.0 if policy == "ppo" else 180.0,
        "ret_unified_total": 250.0 if policy == "ppo" else 170.0,
        "ret_unified_fr_mean": fill,
        "ret_unified_rc_mean": fill - 0.01,
        "ret_unified_ce_mean": 0.90,
        "ret_unified_gate_mean": 0.80,
        "flow_fill_rate": fill,
        "flow_backorder_rate": 1.0 - fill,
        "fill_rate_state_terminal": fill,
        "backorder_rate_state_terminal": 1.0 - fill,
        "terminal_rolling_fill_rate_4w": fill - 0.01,
        "terminal_rolling_backorder_rate_4w": 1.0 - fill + 0.01,
        "order_count": 20.0,
        "completed_order_count": 20.0,
        "completed_order_fraction": 1.0,
        "order_case_fill_rate_share": 0.95 if policy == "ppo" else 0.70,
        "order_case_autotomy_share": 0.01,
        "order_case_recovery_share": 0.02,
        "order_case_non_recovery_share": 0.01,
        "order_case_unfulfilled_share": 0.01 if policy != "ppo" else 0.02,
        "pct_steps_S1": 70.0 if policy == "ppo" else 0.0,
        "pct_steps_S2": 20.0 if policy == "ppo" else 100.0,
        "pct_steps_S3": 10.0 if policy == "ppo" else 0.0,
    }


def test_posthoc_track_b_resilience_audit_builds_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "track_b_run"
    run_dir.mkdir()
    summary_path = run_dir / "summary.json"
    summary = {
        "config": {
            "reward_mode": "ReT_seq_v1",
            "ret_seq_kappa": 0.2,
            "risk_level": "adaptive_benchmark_v2",
            "step_size_hours": 168.0,
            "max_steps": 260,
            "eval_episodes": 1,
            "train_timesteps": 500_000,
            "seeds": [11],
            "action_contract": "track_b_v1",
            "observation_version": "v7",
        },
        "backbone": {"env_variant": "track_b_adaptive_control"},
        "env_spec": {"env_variant": "track_b_adaptive_control"},
        "metric_contract": {"fill_rate_primary": "terminal_order_level"},
        "artifacts": {"summary_json": str(summary_path.resolve())},
        "trained_models": [
            {
                "seed": 11,
                "model_path": str((run_dir / "seed11.zip").resolve()),
                "vec_normalize_path": str((run_dir / "seed11.pkl").resolve()),
            }
        ],
    }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    class FakeVecNorm:
        training = False
        norm_reward = False

        def close(self) -> None:
            return None

    monkeypatch.setattr(
        posthoc.PPO, "load", staticmethod(lambda *args, **kwargs: object())
    )
    monkeypatch.setattr(
        posthoc, "load_vec_normalize", lambda *args, **kwargs: FakeVecNorm()
    )
    monkeypatch.setattr(
        posthoc,
        "evaluate_static_policy",
        lambda policy, *, args, seed: [_episode_row(policy.label, seed)],
    )
    monkeypatch.setattr(
        posthoc,
        "evaluate_trained_policy",
        lambda *, args, seed, model, vec_norm: [_episode_row("ppo", seed)],
    )

    output_dir = run_dir / "posthoc_resilience_audit"
    source_summary = posthoc.load_summary(run_dir)
    episode_rows, seed_rows = posthoc.evaluate_run(source_summary)
    posthoc_summary = posthoc.build_summary(
        source_summary=source_summary,
        output_dir=output_dir,
        episode_rows=episode_rows,
        seed_rows=seed_rows,
    )

    assert (output_dir / "summary.json").exists()
    assert (output_dir / "policy_summary.csv").exists()
    assert posthoc_summary["decision"]["ppo_beats_s2_neutral_by_fill"] is True


def test_generate_track_b_results_package_writes_outputs(tmp_path: Path) -> None:
    run_dir = tmp_path / "track_b_run"
    audit_dir = run_dir / "posthoc_resilience_audit"
    track_a_dir = tmp_path / "track_a_run"
    output_dir = tmp_path / "package"
    audit_dir.mkdir(parents=True)
    track_a_dir.mkdir()

    summary = {
        "config": {
            "reward_mode": "ReT_seq_v1",
            "action_contract": "track_b_v1",
            "observation_version": "v7",
            "risk_level": "adaptive_benchmark_v2",
        },
        "artifacts": {"summary_json": str((audit_dir / "summary.json").resolve())},
    }
    (audit_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    _write_csv(
        audit_dir / "seed_metrics.csv",
        [
            {
                "policy": "ppo",
                "seed": 11,
                "fill_rate_mean": 0.999,
                "backorder_rate_mean": 0.001,
                "order_level_ret_mean_mean": 0.95,
                "terminal_rolling_fill_rate_4w_mean": 0.998,
                "ret_thesis_corrected_total_mean": 240.0,
                "ret_unified_total_mean": 250.0,
            },
            {
                "policy": "ppo",
                "seed": 22,
                "fill_rate_mean": 1.000,
                "backorder_rate_mean": 0.000,
                "order_level_ret_mean_mean": 0.96,
                "terminal_rolling_fill_rate_4w_mean": 0.999,
                "ret_thesis_corrected_total_mean": 242.0,
                "ret_unified_total_mean": 252.0,
            },
            {
                "policy": "s2_d1.00",
                "seed": 11,
                "fill_rate_mean": 0.966,
                "backorder_rate_mean": 0.034,
                "order_level_ret_mean_mean": 0.49,
                "terminal_rolling_fill_rate_4w_mean": 0.893,
                "ret_thesis_corrected_total_mean": 180.0,
                "ret_unified_total_mean": 175.0,
            },
            {
                "policy": "s2_d1.00",
                "seed": 22,
                "fill_rate_mean": 0.965,
                "backorder_rate_mean": 0.035,
                "order_level_ret_mean_mean": 0.48,
                "terminal_rolling_fill_rate_4w_mean": 0.892,
                "ret_thesis_corrected_total_mean": 181.0,
                "ret_unified_total_mean": 176.0,
            },
            {
                "policy": "s3_d2.00",
                "seed": 11,
                "fill_rate_mean": 0.988,
                "backorder_rate_mean": 0.012,
                "order_level_ret_mean_mean": 0.46,
                "terminal_rolling_fill_rate_4w_mean": 0.903,
                "ret_thesis_corrected_total_mean": 172.0,
                "ret_unified_total_mean": 168.0,
            },
            {
                "policy": "s3_d2.00",
                "seed": 22,
                "fill_rate_mean": 0.987,
                "backorder_rate_mean": 0.013,
                "order_level_ret_mean_mean": 0.45,
                "terminal_rolling_fill_rate_4w_mean": 0.902,
                "ret_thesis_corrected_total_mean": 171.0,
                "ret_unified_total_mean": 167.0,
            },
        ],
    )
    _write_csv(
        audit_dir / "policy_summary.csv",
        [
            {
                "policy": "ppo",
                "reward_total_mean": 254.2,
                "reward_total_ci95_low": 254.0,
                "reward_total_ci95_high": 254.4,
                "fill_rate_mean": 0.99996,
                "fill_rate_ci95_low": 0.99990,
                "fill_rate_ci95_high": 1.0,
                "backorder_rate_mean": 0.00004,
                "order_level_ret_mean_mean": 0.9503,
                "ret_thesis_corrected_total_mean": 241.0,
                "ret_unified_total_mean": 251.0,
                "terminal_rolling_fill_rate_4w_mean": 0.9980,
                "pct_steps_S1_mean": 77.8,
                "pct_steps_S2_mean": 15.7,
                "pct_steps_S3_mean": 6.5,
                "order_case_fill_rate_share_mean": 0.95,
                "order_case_autotomy_share_mean": 0.01,
                "order_case_recovery_share_mean": 0.02,
                "order_case_non_recovery_share_mean": 0.01,
                "order_case_unfulfilled_share_mean": 0.01,
            },
            {
                "policy": "s2_d1.00",
                "reward_total_mean": 178.2,
                "reward_total_ci95_low": 178.0,
                "reward_total_ci95_high": 178.4,
                "fill_rate_mean": 0.96592,
                "fill_rate_ci95_low": 0.9650,
                "fill_rate_ci95_high": 0.9668,
                "backorder_rate_mean": 0.03408,
                "order_level_ret_mean_mean": 0.4886,
                "ret_thesis_corrected_total_mean": 180.5,
                "ret_unified_total_mean": 175.5,
                "terminal_rolling_fill_rate_4w_mean": 0.8925,
                "pct_steps_S1_mean": 0.0,
                "pct_steps_S2_mean": 100.0,
                "pct_steps_S3_mean": 0.0,
                "order_case_fill_rate_share_mean": 0.70,
                "order_case_autotomy_share_mean": 0.05,
                "order_case_recovery_share_mean": 0.10,
                "order_case_non_recovery_share_mean": 0.10,
                "order_case_unfulfilled_share_mean": 0.05,
            },
            {
                "policy": "s3_d2.00",
                "reward_total_mean": 171.1,
                "reward_total_ci95_low": 170.9,
                "reward_total_ci95_high": 171.3,
                "fill_rate_mean": 0.98765,
                "fill_rate_ci95_low": 0.9870,
                "fill_rate_ci95_high": 0.9883,
                "backorder_rate_mean": 0.01235,
                "order_level_ret_mean_mean": 0.4584,
                "ret_thesis_corrected_total_mean": 171.5,
                "ret_unified_total_mean": 167.5,
                "terminal_rolling_fill_rate_4w_mean": 0.9025,
                "pct_steps_S1_mean": 0.0,
                "pct_steps_S2_mean": 0.0,
                "pct_steps_S3_mean": 100.0,
                "order_case_fill_rate_share_mean": 0.68,
                "order_case_autotomy_share_mean": 0.04,
                "order_case_recovery_share_mean": 0.12,
                "order_case_non_recovery_share_mean": 0.10,
                "order_case_unfulfilled_share_mean": 0.06,
            },
        ],
    )
    _write_csv(
        run_dir / "comparison_table.csv",
        [
            {
                "ppo_fill_rate_mean": 0.99996,
                "baseline_fill_rate_mean": 0.96592,
                "ppo_fill_gap_vs_baseline_pp": 3.40,
                "ppo_ret_thesis_corrected_mean": 241.0,
            }
        ],
    )
    _write_csv(
        track_a_dir / "comparison_table.csv",
        [
            {
                "ppo_fill_rate_mean": 0.78832,
                "static_s2_fill_rate_mean": 0.79223,
                "ppo_ret_thesis_corrected_total_mean": 133.0,
            }
        ],
    )

    sys.argv = [
        "generate_track_b_results_package.py",
        "--run-dir",
        str(run_dir),
        "--audit-dir",
        str(audit_dir),
        "--track-a-dir",
        str(track_a_dir),
        "--output-dir",
        str(output_dir),
        "--bootstrap-samples",
        "100",
        "--seed",
        "123",
    ]
    package.main()

    assert (output_dir / "policy_overview.csv").exists()
    assert (output_dir / "pairwise_statistics.csv").exists()
    assert (output_dir / "results_discussion_package.md").exists()
    md = (output_dir / "results_discussion_package.md").read_text(encoding="utf-8")
    assert "Track A vs Track B" in md
    assert "Op10/Op12" in md
