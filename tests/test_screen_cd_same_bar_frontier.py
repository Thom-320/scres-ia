from __future__ import annotations

import pytest

from scripts.screen_cd_same_bar_frontier import (
    STATIC_POLICIES,
    StaticPolicy,
    build_cells,
    build_parser,
    static_action,
    summarize_by_cell,
    summarize_by_policy,
)


def _episode_row(
    *,
    cell_id: str,
    regime: str,
    policy: StaticPolicy,
    cd: float,
    flow: float = 0.8,
) -> dict[str, object]:
    return {
        "cell_id": cell_id,
        "regime": regime,
        "seed": 1,
        "policy": policy.label,
        "inventory_level": policy.inventory_level,
        "inventory_label": policy.inventory_label,
        "shift_index": policy.shift_index,
        "shifts": policy.shifts,
        "is_max_corner": policy.is_max_corner,
        "is_interior_inventory": policy.is_interior_inventory,
        "phi": 1.0,
        "psi": 1.5,
        "stochastic_pt": False,
        "demand_multiplier": 1.0,
        "ret_g24_shift_cost": 0.5,
        "ret_g24_kappa_train_frac": 1.0,
        "steps": 4,
        "reward_total": cd,
        "cd_sigmoid_total": cd * 4,
        "cd_sigmoid_mean": cd,
        "cd_sigmoid_train_total": cd * 4,
        "cd_sigmoid_train_mean": cd,
        "cd_train_total": cd * 4,
        "cd_train_mean": cd,
        "cd_raw_total": cd * 4,
        "cd_raw_mean": cd,
        "mean_ret_excel_formula": 0.01,
        "mean_ret_text_formula": 0.01,
        "fill_rate_order_level": 0.0,
        "backorder_rate_order_level": 1.0,
        "flow_delivery_ratio": flow,
        "flow_fill_rate": flow,
        "demanded_total": 1000.0,
        "delivered_total": 800.0,
        "backorder_qty_total": 200.0,
        "pending_backorder_qty_terminal": 0.0,
        "unattended_orders_terminal": 0.0,
        "service_loss_mean": 0.2,
        "service_loss_p95": 0.5,
        "service_loss_cvar95": 0.6,
        "shift_hours_total": 168.0 * policy.shifts,
        "extra_shift_hours_total": 168.0 * max(0, policy.shifts - 1),
        "strategic_buffer_target_units_mean": 100.0 * policy.inventory_level,
        "strategic_buffer_target_unit_hours_total": 100.0 * policy.inventory_level,
        "resource_composite_total": 100.0 * policy.inventory_level,
        "zeta_avg": 1.0,
        "epsilon_avg": 1.0,
        "phi_avg": 1.0,
        "tau_avg": 1.0,
        "kappa_dot": 1.0,
    }


def _policy(label: str) -> StaticPolicy:
    return next(policy for policy in STATIC_POLICIES if policy.label == label)


def test_static_grid_matches_track_a_discrete18() -> None:
    assert len(STATIC_POLICIES) == 18
    assert static_action(_policy("static_S1_I168")) == 3
    assert static_action(_policy("static_S3_I1344")) == 17


def test_build_cells_crosses_environment_and_cost_pressure() -> None:
    args = build_parser().parse_args(
        [
            "--phis",
            "1,2",
            "--psis",
            "1.5",
            "--stochastic-pt-values",
            "False,True",
            "--demand-multipliers",
            "1.0,1.1",
            "--ret-g24-shift-costs",
            "0.5",
            "--ret-g24-kappa-train-fracs",
            "0.2,1.0",
        ]
    )

    cells = build_cells(args)

    assert len(cells) == 16
    assert any(cell["stochastic_pt"] is True for cell in cells)
    assert any(cell["demand_multiplier"] == pytest.approx(1.1) for cell in cells)


def test_cell_summary_selects_regime_dependent_interior_top() -> None:
    current_top = _policy("static_S1_I168")
    severe_top = _policy("static_S2_I504")
    corner = _policy("static_S3_I1344")
    rows = []
    for regime, winner in (("current", current_top), ("severe", severe_top)):
        for policy in (current_top, severe_top, corner):
            score = 0.8 if policy == winner else 0.7
            if policy == corner:
                score = 0.6
            rows.append(_episode_row(cell_id="cell_a", regime=regime, policy=policy, cd=score))

    policy_rows = summarize_by_policy(rows)
    cell_rows = summarize_by_cell(
        policy_rows,
        primary_metric="cd_sigmoid_mean",
        headroom_threshold=0.005,
    )

    assert cell_rows[0]["cell_id"] == "cell_a"
    assert cell_rows[0]["eligible_for_cd_training"] is True
    assert cell_rows[0]["all_regime_tops_non_corner"] is True
    assert cell_rows[0]["all_regime_tops_interior_inventory"] is True
    assert cell_rows[0]["regime_dependent_top"] is True


def test_cell_summary_rejects_max_corner_top() -> None:
    corner = _policy("static_S3_I1344")
    interior = _policy("static_S1_I168")
    rows = []
    for regime in ("current", "severe"):
        rows.append(_episode_row(cell_id="cell_b", regime=regime, policy=corner, cd=0.9))
        rows.append(_episode_row(cell_id="cell_b", regime=regime, policy=interior, cd=0.8))

    cell_rows = summarize_by_cell(
        summarize_by_policy(rows),
        primary_metric="cd_sigmoid_mean",
        headroom_threshold=0.005,
    )

    assert cell_rows[0]["eligible_for_cd_training"] is False
    assert cell_rows[0]["all_regime_tops_non_corner"] is False
