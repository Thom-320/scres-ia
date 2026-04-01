from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_benchmark_bundles import audit_bundle, compare_to_reference


def write_bundle(
    root: Path,
    *,
    summary_payload: dict,
    comparison_csv: str | None = None,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(
        json.dumps(summary_payload),
        encoding="utf-8",
    )
    if comparison_csv is not None:
        (root / "comparison_table.csv").write_text(comparison_csv, encoding="utf-8")
    return root


def test_audit_bundle_flags_legacy_historical_artifact(tmp_path: Path) -> None:
    root = write_bundle(
        tmp_path / "legacy_run",
        summary_payload={
            "config": {
                "risk_level": "increased",
                "stochastic_pt": True,
                "year_basis": "thesis",
                "step_size_hours": 168.0,
                "max_steps": 260,
            },
            "comparison_table": [
                {
                    "static_s2_fill_rate_mean": 0.837,
                    "ppo_pct_steps_S1_mean": 12.0,
                    "ppo_pct_steps_S2_mean": 25.0,
                    "ppo_pct_steps_S3_mean": 63.0,
                }
            ],
        },
    )

    row = audit_bundle(
        bundle=type(
            "BundleRefProxy",
            (),
            {
                "label": root.name,
                "root_dir": root,
                "summary_path": root / "summary.json",
                "comparison_csv_path": None,
            },
        )()
    )

    assert row["audit_status"] == "historical_artifact"
    assert "missing_backbone_metadata" in row["reasons"]
    assert "missing_metric_contract" in row["reasons"]
    assert "missing_benchmark_metadata" in row["reasons"]


def test_compare_to_reference_detects_non_comparable_runs(tmp_path: Path) -> None:
    modern_root = write_bundle(
        tmp_path / "paper_control",
        summary_payload={
            "backbone": {
                "env_variant": "shift_control",
                "observation_version": "v1",
                "frame_stack": 1,
                "year_basis": "thesis",
                "risk_level": "increased",
                "stochastic_pt": True,
                "step_size_hours": 168.0,
                "max_steps": 260,
                "reward_mode": "control_v1",
            },
            "metric_contract": {
                "protocol_version": "paper_facing_v1",
                "fill_rate_primary": "terminal_order_level",
            },
            "reward_contract": {
                "reward_mode": "control_v1",
                "reward_family": "operational_penalty",
            },
            "benchmark_metadata": {"git_commit": "abc"},
            "comparison_table": [
                {
                    "algo": "ppo",
                    "reward_mode": "control_v1",
                    "reward_family": "operational_penalty",
                    "frame_stack": 1,
                    "observation_version": "v1",
                    "learned_policy": "ppo",
                    "learned_reward_mean": -629.0,
                    "learned_fill_rate_mean": 0.782,
                    "static_s2_fill_rate_mean": 0.792,
                    "ppo_pct_steps_S1_mean": 45.5,
                    "ppo_pct_steps_S2_mean": 27.8,
                    "ppo_pct_steps_S3_mean": 26.7,
                }
            ],
        },
    )
    legacy_root = write_bundle(
        tmp_path / "legacy_stopt",
        summary_payload={
            "config": {
                "risk_level": "increased",
                "stochastic_pt": True,
                "year_basis": "thesis",
                "step_size_hours": 168.0,
                "max_steps": 260,
            },
            "comparison_table": [
                {
                    "static_s2_fill_rate_mean": 0.837,
                    "ppo_pct_steps_S1_mean": 12.0,
                    "ppo_pct_steps_S2_mean": 25.0,
                    "ppo_pct_steps_S3_mean": 63.0,
                }
            ],
        },
    )

    reference_row = audit_bundle(
        bundle=type(
            "BundleRefProxy",
            (),
            {
                "label": modern_root.name,
                "root_dir": modern_root,
                "summary_path": modern_root / "summary.json",
                "comparison_csv_path": None,
            },
        )()
    )
    legacy_row = audit_bundle(
        bundle=type(
            "BundleRefProxy",
            (),
            {
                "label": legacy_root.name,
                "root_dir": legacy_root,
                "summary_path": legacy_root / "summary.json",
                "comparison_csv_path": None,
            },
        )()
    )

    comparison = compare_to_reference(legacy_row, reference_row)

    assert comparison["service_metrics_comparable"] is False
    assert comparison["raw_reward_comparable"] is False
    assert "metric_contract_mismatch_or_missing" in comparison["reasons"]
    assert "large_static_s2_fill_delta" in comparison["reasons"]
