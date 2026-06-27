from __future__ import annotations

import pytest

from scripts.compare_garrido_dynamic_vs_static import (
    _env_kwargs_with_initial_policy,
    STATIC_BASELINES,
    build_parser,
    build_comparison,
    build_best_static_by_metric,
    heuristic_action,
    heuristic_initial_action,
    heuristic_is_stressed,
    static_action,
)
from supply_chain.config import THESIS_FAITHFUL_PROTOCOL


def test_static_action_encodes_i168_s1_and_i168_s2() -> None:
    assert static_action(1, 0) == 3
    assert static_action(1, 1) == 4


def test_static_baselines_cover_full_track_a_grid_once() -> None:
    assert len(STATIC_BASELINES) == 18
    assert STATIC_BASELINES["original_S1_I0"] == (0, 0)
    assert STATIC_BASELINES["static_S3_I1344"] == (5, 2)
    assert len(set(STATIC_BASELINES.values())) == 18


def test_dynamic_runner_defaults_to_garrido_fulfillment_delay() -> None:
    args = build_parser().parse_args([])

    assert args.demand_on_hand_fulfillment_delay == THESIS_FAITHFUL_PROTOCOL[
        "demand_on_hand_fulfillment_delay"
    ]
    assert args.stochastic_pt is False
    assert args.reward_mode == "ReT_garrido2024_raw"
    assert args.algo == "ppo"


def test_env_kwargs_can_seed_ppo_with_initial_static_policy() -> None:
    args = type(
        "Args",
        (),
        {
            "reward_mode": "ReT_garrido2024",
            "algo": "ppo",
            "observation_version": "v4",
            "stochastic_pt": True,
            "step_size_hours": 168.0,
            "max_steps": 52,
            "risk_occurrence_mode": "thesis_window",
            "risk_frequency_multiplier": 1.0,
            "risk_impact_multiplier": 1.0,
            "ret_g24_shift_cost": 0.5,
            "ret_g24_kappa_train_frac": 0.2,
            "w_bo": 4.0,
            "w_cost": 0.02,
            "w_disr": 0.0,
            "control_v2_w_fill": 1.0,
            "control_v2_w_service": 4.0,
            "control_v2_w_lost": 2.0,
            "control_v2_w_inventory": 0.05,
            "control_v2_w_shift": 0.08,
            "control_v2_w_switch": 0.02,
            "raw_material_flow_mode": "kit_equivalent_order_up_to",
            "raw_material_order_up_to_multiplier": 2.0,
            "demand_on_hand_fulfillment_delay": 0.0,
        },
    )()

    kwargs = _env_kwargs_with_initial_policy(args, "severe", "static_S1_I168")

    assert kwargs["initial_action"] == 3
    assert kwargs["risk_level"] == "severe"


def test_threshold_heuristic_switches_from_default_to_stress_action() -> None:
    args = type(
        "Args",
        (),
        {
            "heuristic_service_loss_threshold": 0.05,
            "heuristic_pending_backorder_threshold": 1000.0,
            "heuristic_new_backorder_threshold": 500.0,
            "heuristic_disruption_threshold": 0.10,
        },
    )()

    assert heuristic_initial_action("heuristic_threshold_lean") == static_action(0, 0)
    assert heuristic_action("heuristic_threshold_lean", stress_hold_steps=2) == static_action(1, 1)
    assert heuristic_is_stressed({"service_loss_step": 0.10}, args) is True
    assert heuristic_is_stressed({"service_loss_step": 0.01}, args) is False


def test_comparison_uses_cd_primary_and_excel_secondary() -> None:
    rows = [
        {
            "regime": "current",
            "policy": "ppo_dynamic",
            "cd_sigmoid_mean_mean": 0.75,
            "mean_ret_excel_formula_mean": 0.80,
            "fill_rate_order_level_mean": 0.82,
            "extra_shift_hours_total_mean": 10.0,
            "strategic_buffer_target_units_mean_mean": 100.0,
            "resource_composite_total_mean": 110.0,
            "service_loss_p95_mean": 0.2,
            "service_loss_cvar95_mean": 0.3,
            "pct_steps_S1_mean": 70.0,
            "pct_steps_S2_mean": 30.0,
            "pct_steps_S3_mean": 0.0,
        },
        {
            "regime": "current",
            "policy": "static_S1_I168",
            "cd_sigmoid_mean_mean": 0.70,
            "mean_ret_excel_formula_mean": 0.81,
            "fill_rate_order_level_mean": 0.81,
            "extra_shift_hours_total_mean": 20.0,
            "strategic_buffer_target_units_mean_mean": 200.0,
            "resource_composite_total_mean": 220.0,
            "service_loss_p95_mean": 0.1,
            "service_loss_cvar95_mean": 0.2,
            "pct_steps_S1_mean": 100.0,
            "pct_steps_S2_mean": 0.0,
            "pct_steps_S3_mean": 0.0,
        },
    ]

    comparison = build_comparison(rows, excel_noninferiority_tol=0.02)[0]

    assert comparison["primary_metric"] == "mean_ret_excel_formula"
    assert comparison["secondary_metric"] == "cd_sigmoid_mean"
    assert comparison["delta_cd_sigmoid_mean"] == pytest.approx(0.05)
    assert comparison["delta_excel_ret"] == pytest.approx(-0.01)
    assert comparison["ppo_beats_static_excel"] is False
    assert comparison["ppo_beats_static_cd"] is True
    assert comparison["excel_noninferior"] is True
    assert comparison["fewer_extra_shift_hours"] is True
    assert comparison["lower_buffer_target"] is True
    assert comparison["resource_pareto_dominates"] is True
    assert comparison["fill_noninferior"] is True
    assert comparison["p95_not_worse"] is False
    assert comparison["strict_service_resource_dominates"] is False


def test_best_static_by_metric_reports_metric_specific_winners() -> None:
    rows = [
        {
            "regime": "current",
            "policy": "original_S1_I0",
            "mean_ret_excel_formula_mean": 0.8,
            "resource_composite_total_mean": 1.0,
        },
        {
            "regime": "current",
            "policy": "static_S1_I168",
            "mean_ret_excel_formula_mean": 0.7,
            "resource_composite_total_mean": 2.0,
        },
    ]

    best = build_best_static_by_metric(rows)

    by_metric = {row["metric"]: row for row in best}
    assert by_metric["mean_ret_excel_formula"]["best_static_policy"] == "original_S1_I0"
    assert by_metric["resource_composite_total"]["best_static_policy"] == "original_S1_I0"
