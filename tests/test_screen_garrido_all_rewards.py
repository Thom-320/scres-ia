from __future__ import annotations

from scripts.screen_garrido_all_rewards import _comparison_subset, _score_rows, build_parser
from supply_chain.env_experimental_shifts import REWARD_MODE_OPTIONS


def test_all_rewards_parser_defaults_to_registered_rewards() -> None:
    args = build_parser().parse_args([])

    assert args.reward_modes.split(",") == list(REWARD_MODE_OPTIONS)
    assert "faithful" in args.lanes
    assert "headroom_freq1_5" in args.lanes


def test_comparison_subset_keeps_only_ppo_dynamic_against_target() -> None:
    summary = {
        "comparison_table": [
            {
                "dynamic_policy": "ppo_dynamic",
                "static_policy": "static_S1_I168",
                "is_frozen_efficient_static": True,
                "delta_excel_ret": 0.1,
                "delta_cd_sigmoid_mean": 0.2,
                "delta_resource_composite_total": -1.0,
                "delta_flow_fill_rate": 0.0,
                "delta_lost_rate": 0.0,
                "delta_service_loss_cvar95": 0.0,
                "strict_service_resource_dominates": True,
                "resource_pareto_dominates": True,
            },
            {
                "dynamic_policy": "heuristic_threshold_buffer",
                "static_policy": "static_S1_I168",
                "is_frozen_efficient_static": True,
                "delta_excel_ret": 9.0,
                "delta_cd_sigmoid_mean": 9.0,
                "delta_resource_composite_total": -9.0,
                "delta_flow_fill_rate": 0.0,
                "delta_lost_rate": 0.0,
                "delta_service_loss_cvar95": 0.0,
                "strict_service_resource_dominates": True,
                "resource_pareto_dominates": True,
            },
        ]
    }

    rows = _comparison_subset(summary, target="frozen_efficient")
    score = _score_rows(rows)

    assert len(rows) == 1
    assert score["strict_wins"] == 1
    assert score["mean_delta_excel_ret"] == 0.1
