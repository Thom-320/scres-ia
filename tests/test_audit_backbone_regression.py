from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_backbone_regression import (
    build_markdown_report,
    load_legacy_reference,
)


def test_load_legacy_reference_extracts_static_policy_rows(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "policy_summary": [
                    {
                        "policy": "static_s1",
                        "fill_rate_mean": 0.65,
                        "backorder_rate_mean": 0.35,
                        "ret_thesis_corrected_total_mean": 242.9,
                        "reward_total_mean": -356.1,
                    },
                    {
                        "policy": "static_s2",
                        "fill_rate_mean": 0.84,
                        "backorder_rate_mean": 0.16,
                        "order_level_ret_mean_mean": 0.83,
                        "ret_thesis_corrected_total_mean": 243.0,
                        "reward_total_mean": -170.1,
                    },
                    {
                        "policy": "ppo",
                        "fill_rate_mean": 0.83,
                        "backorder_rate_mean": 0.17,
                        "ret_thesis_corrected_total_mean": 243.1,
                        "reward_total_mean": -172.0,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    legacy = load_legacy_reference(summary_path)

    assert set(legacy) == {"static_s1", "static_s2"}
    assert legacy["static_s2"]["fill_rate_mean"] == 0.84
    assert legacy["static_s2"]["order_level_ret_mean"] == 0.83


def test_build_markdown_report_includes_legacy_delta() -> None:
    class Args:
        reward_mode = "control_v1"
        observation_version = "v1"
        frame_stack = 1
        year_basis = "thesis"
        risk_level = "increased"
        stochastic_pt = True
        policies = ["static_s2"]

    report = build_markdown_report(
        args=Args(),
        rows=[
            {
                "ref": "HEAD",
                "commit": "abc123def456",
                "policy": "static_s2",
                "fill_rate_mean": 0.42,
                "backorder_rate_mean": 0.58,
                "order_level_ret_mean": 0.41,
                "ret_thesis_corrected_total_mean": 243.2,
            }
        ],
        legacy_reference={
            "static_s2": {
                "fill_rate_mean": 0.84,
                "backorder_rate_mean": 0.16,
                "order_level_ret_mean": 0.82,
                "ret_thesis_corrected_total_mean": 243.0,
                "reward_total_mean": -170.1,
            }
        },
    )

    assert "delta_fill_vs_legacy" in report
    assert "order-level-ReT" in report
    assert "HEAD" in report
    assert "-0.420" in report
