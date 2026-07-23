"""Fast tests for the product-coupled R24 carrier (Q-R1 risk door).

Contract: contracts/q_r1_product_coupled_execution_amendment_v1.json.
Only calibration probe seeds (7590901+) are used for DES runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from supply_chain.program_o_full_des import ProgramOFullDESSimulation
from supply_chain.q_r1_product_coupled import (
    ProductCoupledProgramODES,
    beta_bernoulli_sigma_posterior,
    demand_initial_regime_chain,
    direct_campaign_metrics,
    draw_surge_product,
    run_campaign,
    sigma_path,
)

ROOT = Path(__file__).resolve().parents[1]
PROBE_ROOT = 7_590_901  # calibration probe block 7590901-7590920


@pytest.fixture(scope="module")
def scheduler() -> dict:
    parent = json.loads(
        (ROOT / "contracts/program_o_full_des_hpi_translation_v1.json").read_text()
    )
    return parent["action"]["within_week_schedulers"][
        parent["action"]["primary_scheduler"]
    ]


def test_sigma_path_deterministic_and_valid() -> None:
    path_a = sigma_path(PROBE_ROOT, 12, 0.90)
    path_b = sigma_path(PROBE_ROOT, 12, 0.90)
    assert path_a == path_b
    assert len(path_a) == 12
    assert set(path_a) <= {"C", "H"}
    # kappa_r = 1.0 -> never flips
    frozen = sigma_path(PROBE_ROOT, 50, 1.0)
    assert len(set(frozen)) == 1
    # different roots decorrelate
    assert any(
        sigma_path(PROBE_ROOT, 12, 0.90) != sigma_path(PROBE_ROOT + k, 12, 0.90)
        for k in (1, 2, 3)
    )


def test_sigma_path_persistence_rate() -> None:
    kappa_r = 0.75
    flips = 0
    transitions = 0
    for offset in range(40):
        path = sigma_path(PROBE_ROOT + offset, 50, kappa_r)
        for previous, current in zip(path, path[1:]):
            transitions += 1
            flips += previous != current
    rate = flips / transitions
    assert abs(rate - (1.0 - kappa_r)) < 0.03  # 1960 transitions


def test_draw_surge_product_rate_and_determinism() -> None:
    s_r = 0.85
    draws = [
        draw_surge_product(
            campaign_seed=PROBE_ROOT * 100, event_index=index, sigma="C", s_r=s_r
        )
        for index in range(2000)
    ]
    assert draws == [
        draw_surge_product(
            campaign_seed=PROBE_ROOT * 100, event_index=index, sigma="C", s_r=s_r
        )
        for index in range(2000)
    ]
    favored_rate = float(np.mean([draw == "P_C" for draw in draws]))
    assert abs(favored_rate - s_r) < 0.03
    # symmetry: sigma H favors P_H at the same event indices
    draws_h = [
        draw_surge_product(
            campaign_seed=PROBE_ROOT * 100, event_index=index, sigma="H", s_r=s_r
        )
        for index in range(200)
    ]
    assert abs(float(np.mean([draw == "P_H" for draw in draws_h])) - s_r) < 0.08
    # s_r = 1.0 is deterministic on the favored product
    assert all(
        draw_surge_product(
            campaign_seed=PROBE_ROOT * 100, event_index=index, sigma="C", s_r=1.0
        )
        == "P_C"
        for index in range(50)
    )


def test_riskoff_couplingoff_reproduces_baseline(scheduler: dict) -> None:
    """Regression guard: the subclass with risks off and coupling off is
    bit-identical to the plain ProgramOFullDESSimulation for the same seed."""
    initial_regime = demand_initial_regime_chain(PROBE_ROOT, 1)[0]
    common = dict(
        seed=PROBE_ROOT * 100,
        calendar=(2,) * 8,
        scheduler=scheduler,
        regime_persistence=0.90,
        dominant_share=0.90,
        downstream_freight_physics_mode="fixed_clock_physical_v1",
        initial_regime=initial_regime,
    )
    sim_sub = ProductCoupledProgramODES(
        product_coupling_enabled=False, **common
    ).run_contract()
    sim_base = ProgramOFullDESSimulation(**common).run_contract()
    assert sim_sub.aggregate_state_hash() == sim_base.aggregate_state_hash()
    metrics_sub = direct_campaign_metrics(sim_sub)
    metrics_base = direct_campaign_metrics(sim_base)
    for key in (
        "early_ret_complete_cohort",
        "early_ret_visible",
        "ret_excel",
        "worst_product_fill",
        "unresolved_orders",
        "lost_orders",
    ):
        assert metrics_sub[key] == metrics_base[key]
    assert not sim_sub.pc_surge_log


def test_coupled_campaign_surge_assignment_and_crn(scheduler: dict) -> None:
    """One coupled probe campaign: s_r=1.0 assigns every surge to the favored
    product; the surge timeline is identical across postures (CRN); the same
    arm twice is identical (determinism)."""
    initial_regime = demand_initial_regime_chain(PROBE_ROOT, 1)[0]
    kwargs = dict(
        root=PROBE_ROOT,
        campaign_index=0,
        sigma="C",
        s_r=1.0,
        kappa=0.90,
        scheduler=scheduler,
        initial_regime=initial_regime,
    )
    row_a = run_campaign(calendar=[2] * 8, **kwargs)
    assert row_a["n_surges"] > 0
    assert row_a["n_surges_favored"] == row_a["n_surges"]
    assert all(
        item["assigned_product"] == "P_C" for item in row_a["surge_log"]
    )
    # conservation self-checks hold under the coupling
    assert row_a["max_abs_product_residual"] < 1e-6
    assert row_a["max_abs_partition_residual"] < 1e-6
    # surge quantity is actually demanded (merged into orders)
    assert row_a["merge_log_count"] > 0
    # CRN across arms: different posture, identical surge timeline
    row_b = run_campaign(calendar=[0] * 8, **kwargs)
    signature = lambda row: [  # noqa: E731
        (item["time"], item["surge"], item["assigned_product"])
        for item in row["surge_log"]
    ]
    assert signature(row_a) == signature(row_b)
    # determinism: same arm twice -> identical metrics
    row_a2 = run_campaign(calendar=[2] * 8, **kwargs)
    assert row_a2["early_ret_complete_cohort"] == row_a["early_ret_complete_cohort"]
    assert row_a2["aggregate_state_hash"] == row_a["aggregate_state_hash"]
    assert signature(row_a2) == signature(row_a)


def test_beta_bernoulli_posterior() -> None:
    # empty history -> uniform prior
    empty = beta_bernoulli_sigma_posterior([])
    assert empty["p_sigma_c"] == pytest.approx(0.5)
    # converges to the true sigma given synthetic logs
    c_heavy = [
        [{"assigned_product": "P_C"}] * 9 + [{"assigned_product": "P_H"}]
        for _ in range(5)
    ]
    posterior = beta_bernoulli_sigma_posterior(c_heavy)
    assert posterior["p_sigma_c"] > 0.85
    assert posterior["n_c"] == 45 and posterior["n_h"] == 5
    h_heavy = [[{"assigned_product": "P_H"}] * 20]
    assert beta_bernoulli_sigma_posterior(h_heavy)["p_sigma_c"] < 0.1
    with pytest.raises(ValueError):
        beta_bernoulli_sigma_posterior([[{"assigned_product": "P_X"}]])


def test_demand_initial_regime_chain_deterministic() -> None:
    chain_a = demand_initial_regime_chain(PROBE_ROOT, 6)
    chain_b = demand_initial_regime_chain(PROBE_ROOT, 6)
    assert chain_a == chain_b
    assert len(chain_a) == 6
    assert set(chain_a) <= {"P_C", "P_H"}


def test_coupling_requires_sigma(scheduler: dict) -> None:
    with pytest.raises(ValueError):
        ProductCoupledProgramODES(
            product_coupling_enabled=True,
            sigma=None,
            seed=PROBE_ROOT * 100,
            calendar=(2,) * 8,
            scheduler=scheduler,
            regime_persistence=0.90,
            dominant_share=0.90,
        )
