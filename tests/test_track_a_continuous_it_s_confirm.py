from __future__ import annotations

from pathlib import Path

import pytest

import scripts.run_track_a_continuous_it_s_confirm as confirm


def test_build_smoke_args_uses_faithful_continuous_contract(tmp_path: Path) -> None:
    args = confirm.build_parser().parse_args(
        [
            "--output-root",
            str(tmp_path),
            "--reward-profile",
            "ret_ladder_steep",
            "--pt-profile",
            "stoch_pt_mean",
            "--risk-level",
            "severe",
        ]
    )

    smoke_args = confirm.build_smoke_args(args, seed=4242, run_root=tmp_path / "runs")

    assert smoke_args.action_space_mode == "continuous_it_s"
    assert smoke_args.inventory_period_mode == "thesis_strict"
    assert smoke_args.risk_occurrence_mode == "thesis_periodic"
    assert smoke_args.raw_material_flow_mode == "kit_equivalent_order_up_to"
    assert smoke_args.raw_material_order_up_to_multiplier == pytest.approx(2.0)
    assert smoke_args.reward_mode == "ReT_ladder_v1"
    assert smoke_args.ret_ladder_w_sc == pytest.approx(0.80)
    assert smoke_args.stochastic_pt is True
    assert smoke_args.stochastic_pt_mean_preserving is True


def test_continuous_static_action_maps_buffer_and_shift() -> None:
    action = confirm.continuous_static_action(0.4, 2)

    assert action.shape == (2,)
    assert action[0] == pytest.approx(0.4)
    assert action[1] == pytest.approx(0.0)


def test_build_overall_summary_counts_positive_seeds() -> None:
    summary = confirm.build_overall_summary(
        [
            {"status": "complete", "delta_fill": 0.02, "delta_ret": 0.01},
            {"status": "complete", "delta_fill": -0.03, "delta_ret": 0.04},
            {"status": "dry_run"},
        ]
    )

    assert summary["seed_count"] == 2
    assert summary["positive_fill_seeds"] == 1
    assert summary["positive_ret_seeds"] == 2
    assert summary["mean_delta_fill"] == pytest.approx(-0.005)
    assert summary["mean_delta_ret"] == pytest.approx(0.025)
