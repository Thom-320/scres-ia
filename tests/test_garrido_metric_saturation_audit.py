from __future__ import annotations

import pytest
import numpy as np

from scripts.audit_garrido_metric_saturation import build_env_kwargs, build_parser, order_metric_distribution
from scripts.run_garrido_static_fidelity_stress import RISK_PROFILES
from supply_chain.config import RISKS_WAR_STRESS_V1
from supply_chain.external_env_interface import make_dkana_thesis_faithful_env
from supply_chain.supply_chain import OrderRecord
from supply_chain.thesis_design import design_spec_for_cfi


class FakeSim:
    def __init__(self) -> None:
        self.orders = [
            OrderRecord(j=1, OPTj=0, OATj=40, CTj=40, LTj=48),
            OrderRecord(j=2, OPTj=0, OATj=40, CTj=40, LTj=48, APj=24),
            OrderRecord(j=3, OPTj=0, OATj=None, CTj=None, LTj=48),
        ]
        self.pending_backorders = [self.orders[-1]]
        self.total_unattended_orders = 0
        self.env = type("FakeEnv", (), {"now": 48.0})()

    def _order_level_fill_rate(self) -> float:
        return 0.5


def test_order_metric_distribution_keeps_unfulfilled_orders_visible() -> None:
    metrics = order_metric_distribution(FakeSim())

    assert metrics["n_orders"] == 3
    assert metrics["n_completed"] == 2
    assert metrics["n_unfulfilled"] == 1
    assert metrics["pct_case_unfulfilled"] == pytest.approx(100.0 / 3.0)
    assert metrics["ret_mean_completed_orders"] == pytest.approx(0.5)
    assert metrics["ret_mean_all_orders_zero_unfulfilled"] == pytest.approx(1.0 / 3.0)
    assert metrics["re_fr_contribution_all"] == pytest.approx(1.0 / 6.0)
    assert metrics["re_ap_contribution_all"] == pytest.approx(1.0 / 6.0)
    assert metrics["re_rp_contribution_all"] == 0.0
    assert metrics["dynamic_ret_contribution_all"] == pytest.approx(1.0 / 6.0)
    assert metrics["static_ret_contribution_all"] == pytest.approx(1.0 / 6.0)
    assert metrics["cycle_time_weighted_ret_completed"] == pytest.approx(0.5)
    assert metrics["period_weighted_ret_proxy"] == pytest.approx(1.0 / 3.0)
    assert metrics["period_ap_exposure_pct"] == pytest.approx(100.0 / 6.0)
    assert metrics["period_dynamic_exposure_pct"] == pytest.approx(100.0 / 6.0)
    assert metrics["period_unfulfilled_exposure_pct"] == pytest.approx(100.0 / 3.0)
    assert metrics["bt_pending_orders"] == 1


def test_war_stress_profile_is_declared_and_env_accepts_it() -> None:
    assert "war_stress_v1" in RISK_PROFILES
    assert RISKS_WAR_STRESS_V1["R22"]["b"] == 168
    assert RISKS_WAR_STRESS_V1["R3"]["duration"] == 1_344

    env = make_dkana_thesis_faithful_env(
        risk_level="war_stress_v1",
        max_steps=1,
        stochastic_pt=True,
        raw_material_flow_mode="kit_equivalent_order_up_to",
        risk_occurrence_mode="thesis_periodic",
    )
    try:
        env.reset(seed=123)
        assert env.unwrapped.risk_level == "war_stress_v1"
    finally:
        env.close()


def test_metric_saturation_audit_forwards_ret_tail_knobs() -> None:
    args = build_parser().parse_args(
        [
            "--reward-mode",
            "ReT_tail_v1",
            "--ret-tail-w-sc",
            "0.2",
            "--ret-tail-w-rc",
            "0.65",
            "--ret-tail-w-ce",
            "0.15",
            "--ret-tail-cap-kappa",
            "0.35",
            "--ret-tail-inv-kappa",
            "0.8",
            "--ret-tail-boost",
            "3.0",
            "--ret-tail-transform",
            "exp_norm",
            "--ret-tail-beta",
            "4.0",
        ]
    )

    kwargs = build_env_kwargs(
        args=args,
        profile="increased",
        spec=design_spec_for_cfi(31),
        action=np.array([2, 1]),
        max_steps=10,
    )

    assert kwargs["reward_mode"] == "ReT_tail_v1"
    assert kwargs["ret_tail_w_sc"] == pytest.approx(0.2)
    assert kwargs["ret_tail_w_rc"] == pytest.approx(0.65)
    assert kwargs["ret_tail_w_ce"] == pytest.approx(0.15)
    assert kwargs["ret_tail_cap_kappa"] == pytest.approx(0.35)
    assert kwargs["ret_tail_inv_kappa"] == pytest.approx(0.8)
    assert kwargs["ret_tail_boost"] == pytest.approx(3.0)
    assert kwargs["ret_tail_transform"] == "exp_norm"
    assert kwargs["ret_tail_gamma"] == pytest.approx(1.0)
    assert kwargs["ret_tail_beta"] == pytest.approx(4.0)
