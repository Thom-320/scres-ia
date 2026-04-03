from __future__ import annotations

import scripts.audit_track_b_all_rewards as audit_track_b


def test_rank_policy_rows_sorts_with_common_audit_contract() -> None:
    rows = [
        {
            "reward_mode": "ReT_seq_v1",
            "policy": "s2_d1.00",
            "algo": "static",
            "fill_rate_mean": 0.96,
            "backorder_rate_mean": 0.04,
            "order_level_ret_mean_mean": 0.48,
            "service_continuity_step_mean_mean": 0.62,
            "adaptive_efficiency_step_mean_mean": 0.90,
            "ret_garrido2024_sigmoid_total_mean": 90.0,
            "reward_total_mean": 100.0,
        },
        {
            "reward_mode": "ReT_seq_v1",
            "policy": "ppo",
            "algo": "ppo",
            "fill_rate_mean": 0.99,
            "backorder_rate_mean": 0.01,
            "order_level_ret_mean_mean": 0.95,
            "service_continuity_step_mean_mean": 0.99,
            "adaptive_efficiency_step_mean_mean": 0.97,
            "ret_garrido2024_sigmoid_total_mean": 140.0,
            "reward_total_mean": 250.0,
        },
        {
            "reward_mode": "ReT_seq_v1",
            "policy": "recurrent_ppo",
            "algo": "recurrent_ppo",
            "fill_rate_mean": 0.989,
            "backorder_rate_mean": 0.011,
            "order_level_ret_mean_mean": 0.949,
            "service_continuity_step_mean_mean": 0.989,
            "adaptive_efficiency_step_mean_mean": 0.969,
            "ret_garrido2024_sigmoid_total_mean": 139.0,
            "reward_total_mean": 249.0,
        },
    ]

    leaderboard = audit_track_b.rank_policy_rows(rows)

    assert leaderboard[0]["policy"] == "ppo"
    assert leaderboard[1]["policy"] == "recurrent_ppo"
    assert leaderboard[2]["policy"] == "s2_d1.00"


def test_parser_exposes_all_reward_modes() -> None:
    parser = audit_track_b.build_parser()

    args = parser.parse_args([])

    assert "ReT_seq_v1" in args.reward_modes
    assert "ReT_thesis" in args.reward_modes
    assert "ReT_garrido2024" in args.reward_modes


def test_build_pairwise_stats_anchors_on_ppo() -> None:
    episode_rows = [
        {
            "reward_mode": "ReT_seq_v1",
            "policy": "ppo",
            "seed": 11,
            "episode": 1,
            "eval_seed": 101,
            "fill_rate": 1.0,
            "backorder_rate": 0.0,
            "order_level_ret_mean": 0.95,
            "delivered_total": 100.0,
            "avg_annual_delivery": 20.0,
            "total_step_cost": 10.0,
            "avg_step_cost": 1.0,
            "service_continuity_step_mean": 0.99,
            "backlog_containment_step_mean": 0.98,
            "adaptive_efficiency_step_mean": 0.97,
            "ret_seq_total": 250.0,
            "ret_garrido2024_sigmoid_total": 180.0,
            "service_loss_area_below_095": 0.0,
            "mean_recovery_streak_hours": 0.0,
        },
        {
            "reward_mode": "ReT_seq_v1",
            "policy": "s2_d1.00",
            "seed": 11,
            "episode": 1,
            "eval_seed": 101,
            "fill_rate": 0.96,
            "backorder_rate": 0.04,
            "order_level_ret_mean": 0.47,
            "delivered_total": 90.0,
            "avg_annual_delivery": 18.0,
            "total_step_cost": 20.0,
            "avg_step_cost": 2.0,
            "service_continuity_step_mean": 0.62,
            "backlog_containment_step_mean": 0.90,
            "adaptive_efficiency_step_mean": 0.90,
            "ret_seq_total": 175.0,
            "ret_garrido2024_sigmoid_total": 147.0,
            "service_loss_area_below_095": 5.0,
            "mean_recovery_streak_hours": 100.0,
        },
    ]

    pairwise = audit_track_b.build_pairwise_stats(episode_rows)

    fill_row = next(row for row in pairwise if row["metric"] == "fill_rate")
    cost_row = next(row for row in pairwise if row["metric"] == "total_step_cost")

    assert fill_row["comparator_policy"] == "s2_d1.00"
    assert fill_row["anchor_better"] is True
    assert cost_row["anchor_better"] is True


def test_build_paper_table_uses_expected_policy_order() -> None:
    rows = [
        {
            "policy": "s2_d1.00",
            "fill_rate_mean": 0.96,
            "fill_rate_ci95_low": 0.95,
            "fill_rate_ci95_high": 0.97,
        },
        {
            "policy": "ppo",
            "fill_rate_mean": 1.0,
            "fill_rate_ci95_low": 1.0,
            "fill_rate_ci95_high": 1.0,
        },
    ]

    table = audit_track_b.build_paper_table(
        rows,
        spec=(("fill_rate", "fill_rate"),),
    )

    assert [row["policy"] for row in table] == ["ppo", "s2_d1.00"]


def test_build_manuscript_main_table_uses_curated_policy_labels() -> None:
    rows = [
        {
            "policy": "ppo",
            "fill_rate_mean": 1.0,
            "backorder_rate_mean": 0.0,
            "order_level_ret_mean_mean": 0.95,
            "ret_garrido2024_sigmoid_total_mean": 182.5,
            "avg_annual_delivery_mean": 920000.0,
            "pct_steps_S1_mean": 70.0,
            "pct_steps_S2_mean": 20.0,
            "pct_steps_S3_mean": 10.0,
            "pct_ret_case_autotomy_mean": 95.0,
            "pct_ret_case_recovery_mean": 5.0,
            "pct_ret_case_non_recovery_mean": 0.0,
        },
        {
            "policy": "s2_d2.00",
            "fill_rate_mean": 0.98,
            "backorder_rate_mean": 0.02,
            "order_level_ret_mean_mean": 0.45,
            "ret_garrido2024_sigmoid_total_mean": 150.0,
            "avg_annual_delivery_mean": 890000.0,
            "pct_steps_S1_mean": 0.0,
            "pct_steps_S2_mean": 100.0,
            "pct_steps_S3_mean": 0.0,
            "pct_ret_case_autotomy_mean": 20.0,
            "pct_ret_case_recovery_mean": 80.0,
            "pct_ret_case_non_recovery_mean": 0.0,
        },
    ]

    table = audit_track_b.build_manuscript_main_table(rows)

    assert [row["policy"] for row in table] == ["PPO", "S2(d=2.0)"]
