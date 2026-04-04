from __future__ import annotations

from scripts.eval_track_b_cross_scenario import (
    build_meeting_table_rows,
    build_overview_row,
)


def _row(policy: str, algo: str, fill: float, ret: float) -> dict[str, object]:
    return {
        "policy": policy,
        "algo": algo,
        "fill_rate_mean": fill,
        "backorder_rate_mean": 1.0 - fill,
        "order_level_ret_mean_mean": ret,
        "service_continuity_step_mean_mean": fill,
        "backlog_containment_step_mean_mean": 0.9,
        "adaptive_efficiency_step_mean_mean": 0.8,
        "pct_ret_case_fill_rate_only_mean": 1.0,
        "pct_ret_case_autotomy_mean": 80.0,
        "pct_ret_case_recovery_mean": 19.0,
        "pct_ret_case_non_recovery_mean": 0.0,
    }


def test_build_meeting_table_rows_filters_and_labels() -> None:
    rows = build_meeting_table_rows(
        [
            _row("s3_d2.00", "static", 0.98, 0.40),
            _row("ppo", "ppo", 1.0, 0.95),
            _row("s2_d1.50", "static", 0.99, 0.48),
            _row("recurrent_ppo", "recurrent_ppo", 1.0, 0.94),
        ]
    )

    assert [row["policy"] for row in rows] == [
        "PPO",
        "RecurrentPPO",
        "S2(d=1.5)",
        "S3(d=2.0)",
    ]
    assert rows[0]["order_level_ret"] == 0.95


def test_build_overview_row_selects_best_static() -> None:
    rows = [
        _row("ppo", "ppo", 1.0, 0.95),
        _row("recurrent_ppo", "recurrent_ppo", 1.0, 0.94),
        _row("s2_d1.50", "static", 0.99, 0.48),
        _row("s3_d2.00", "static", 0.98, 0.50),
    ]

    overview = build_overview_row("severe", rows)

    assert overview["best_static_policy"] == "s2_d1.50"
    assert overview["ppo_fill_gap_vs_best_static_pp"] > 0.0
    assert overview["ppo_ret_gap_vs_recurrent"] > 0.0
