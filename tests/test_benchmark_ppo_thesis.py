from __future__ import annotations

from scripts.benchmark_ppo_thesis import (
    build_decision_table,
    build_parser,
    canonical_shift_deltas,
    default_output_root,
)


def test_benchmark_ppo_thesis_parser_accepts_ret_corrected_cost_alias() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--reward-mode",
            "ReT_corrected_cost",
            "--shift-delta",
            "0.02",
            "--shift-delta",
            "0.04",
            "--stochastic-pt",
        ]
    )
    assert args.reward_mode == "ReT_corrected_cost"
    assert canonical_shift_deltas(args) == [0.02, 0.04]
    assert args.stochastic_pt is True


def test_default_output_root_uses_ret_corrected_cost_lane_name() -> None:
    assert default_output_root("ReT_corrected_cost").as_posix().endswith(
        "outputs/benchmarks/ppo_shift_control_ret_corrected_cost"
    )


def test_build_decision_table_marks_collapse_to_s1() -> None:
    policy_summary = [
        {
            "policy": "ppo",
            "shift_delta": 0.04,
            "fill_rate_mean": 0.623,
            "backorder_rate_mean": 0.377,
            "ret_thesis_corrected_total_mean": 220.0,
            "pct_steps_S1_mean": 100.0,
            "pct_steps_S2_mean": 0.0,
            "pct_steps_S3_mean": 0.0,
        },
        {
            "policy": "static_s1",
            "shift_delta": 0.04,
            "fill_rate_mean": 0.630,
            "backorder_rate_mean": 0.370,
            "ret_thesis_corrected_total_mean": 221.0,
            "pct_steps_S1_mean": 100.0,
            "pct_steps_S2_mean": 0.0,
            "pct_steps_S3_mean": 0.0,
        },
        {
            "policy": "static_s2",
            "shift_delta": 0.04,
            "fill_rate_mean": 0.840,
            "backorder_rate_mean": 0.160,
            "ret_thesis_corrected_total_mean": 243.0,
            "pct_steps_S1_mean": 0.0,
            "pct_steps_S2_mean": 100.0,
            "pct_steps_S3_mean": 0.0,
        },
        {
            "policy": "static_s3",
            "shift_delta": 0.04,
            "fill_rate_mean": 0.800,
            "backorder_rate_mean": 0.200,
            "ret_thesis_corrected_total_mean": 240.0,
            "pct_steps_S1_mean": 0.0,
            "pct_steps_S2_mean": 0.0,
            "pct_steps_S3_mean": 100.0,
        },
    ]

    decision_table = build_decision_table(policy_summary)

    assert len(decision_table) == 1
    assert decision_table[0]["best_static_policy"] == "static_s2"
    assert decision_table[0]["collapse_s1"] is True
    assert decision_table[0]["verdict"] == "kill_collapse_s1"
