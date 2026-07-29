from __future__ import annotations

import pytest

from scripts.screen_garrido_cost_metrics import (
    ScreenConfig,
    build_summary,
    joint_6x3_descriptors,
    multiplicative_score,
    summarize_garrido2024_des,
    subtractive_score,
)


def _row(
    policy: str,
    *,
    ret: float,
    shifts: int,
    profile: str = "I0",
    period: str = "",
    family: str = "risk_r1",
) -> dict[str, str]:
    return {
        "policy": policy,
        "policy_kind": "test",
        "family": family,
        "mean_ret_excel_formula": str(ret),
        "shifts": str(shifts),
        "initial_buffer_profile": profile,
        "inventory_period": period,
    }


def test_multiplicative_score_discounts_excel_ret() -> None:
    row = _row("shift_S3", ret=0.8, shifts=3, profile="I1344", period="1344")

    result = multiplicative_score(row, k_s=0.4, k_i=0.25)

    assert 0.0 <= result["score"] <= 0.8
    assert result["cap_eff"] == pytest.approx(0.6)
    assert result["inv_eff"] < 1.0
    assert result["cost_eff"] == pytest.approx(
        (result["cap_eff"] * result["inv_eff"]) ** 0.5
    )


def test_subtractive_score_can_be_negative_net_benefit() -> None:
    row = _row("shift_S3", ret=0.01, shifts=3, profile="I1344", period="1344")

    result = subtractive_score(row, a_s=0.05, a_i=0.05)

    assert result["score"] < 0.01
    assert result["shift_penalty"] == pytest.approx(0.05)
    assert result["inventory_penalty"] > 0.0


def test_build_summary_reports_three_metric_families() -> None:
    rows = [
        _row("shift_S1", ret=0.50, shifts=1, family="risk_r1"),
        _row("shift_S2", ret=0.55, shifts=2, family="risk_r1"),
        _row("inventory_I168", ret=0.60, shifts=1, profile="I168", period="168", family="risk_r2"),
    ]
    config = ScreenConfig(
        k_s_values=(0.0, 0.05),
        k_i_values=(0.0, 0.05),
        a_s_values=(0.0, 0.01),
        a_i_values=(0.0, 0.01),
    )

    summary = build_summary(rows, config)

    assert summary["primary_ret_metric"] == "mean_ret_excel_formula"
    assert summary["excel_cost_multiplicative"]["selected"]["top_policy"]
    assert summary["excel_cost_subtractive"]["selected"]["top_policy"]
    assert summary["garrido2024_family"]["status"] == "not_run"


def test_garrido2024_des_summary_selects_regime_dependent_interior_candidate() -> None:
    rows = []
    for regime, winner in (("current", "shift_S2"), ("increased", "inventory_I168")):
        for policy in ("shift_S1", "shift_S2", "inventory_I168"):
            rows.append(
                {
                    "policy": policy,
                    "policy_kind": "test",
                    "risk_level": regime,
                    "seed": 7,
                    "ret_g24_kappa_train_frac": 0.2,
                    "ret_g24_shift_cost": 1.0,
                    "risk_frequency_multiplier": 1.25,
                    "risk_impact_multiplier": 1.0,
                    "ret_garrido2024_train_total": 2.0 if policy == winner else 1.0,
                }
            )

    summary = summarize_garrido2024_des(rows)

    assert summary["status"] == "computed"
    assert summary["selected"]["regime_dependent_top"] is True
    assert summary["selected"]["all_regime_tops_non_corner"] is True
    assert summary["selected"]["risk_frequency_multiplier"] == pytest.approx(1.25)
    assert summary["selected"]["robust_regime_dependent_top"] is True
    assert summary["first_eligible_interior_regime_dependent"] is not None


def test_joint_6x3_descriptors_cross_inventory_and_shifts() -> None:
    descriptors = joint_6x3_descriptors()

    assert len(descriptors) == 18
    assert {item["shifts"] for item in descriptors} == {1, 2, 3}
    assert "joint_I504_S3" in {item["policy"] for item in descriptors}
