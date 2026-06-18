"""Cost-aware (Garrido 2024) evaluation aggregation.

The smoke evaluation now surfaces the Garrido 2024 cost-aware resilience index per
policy (`ret_garrido2024_sigmoid` + ζ/ε/φ/τ/κ components), so RL-vs-static can be
judged on cost-adjusted resilience (where the index has an INTERIOR optimum) and
not only on `ret_p10` (which plateaus and cannot show an adaptive win).

These tests lock in (a) the env emits the per-step cost-aware signals every step,
and (b) `aggregate()` carries them into the per-policy summary.
"""
from __future__ import annotations

import numpy as np

from scripts.run_thesis_decision_ppo_smoke import aggregate
from supply_chain.external_env_interface import make_dkana_thesis_faithful_env

COST_AWARE_STEP_KEYS = (
    "ret_garrido2024_sigmoid_step",
    "zeta_avg",
    "epsilon_avg",
    "phi_avg",
    "tau_avg",
    "kappa_dot",
)


def test_env_emits_cost_aware_signals_every_step() -> None:
    env = make_dkana_thesis_faithful_env(
        reward_mode="ReT_tail_v1",
        action_space_mode="continuous_it_s",
        risk_level="increased",
        risk_occurrence_mode="thesis_periodic",
        raw_material_flow_mode="kit_equivalent_order_up_to",
        max_steps=20,
    )
    _, info = env.reset(seed=3)
    seen = 0
    for _ in range(12):
        _, _, terminated, truncated, info = env.step(np.array([0.3, -1.0], dtype=np.float32))
        if info.get("action_phase") == "initial_decision":
            continue
        seen += 1
        for key in COST_AWARE_STEP_KEYS:
            assert key in info, f"missing {key} in step info"
        idx = float(info["ret_garrido2024_sigmoid_step"])
        assert 0.0 <= idx <= 1.0
        if terminated or truncated:
            break
    assert seen > 0


def test_aggregate_carries_cost_aware_columns() -> None:
    rows = []
    for policy in ("static_grid_b0.100_S1", "ppo_continuous"):
        for episode in range(2):
            row = {key: 0.0 for key in _AGG_REQUIRED_KEYS}
            row["policy"] = policy
            row["ret_garrido2024_sigmoid"] = 0.67 + 0.01 * episode
            row["zeta_avg"] = 140000.0
            row["epsilon_avg"] = 1000.0
            row["phi_avg"] = 50.0
            row["tau_avg"] = 2.0
            row["kappa_dot"] = 1.1
            rows.append(row)
    out = {r["policy"]: r for r in aggregate(rows)}
    for policy in ("static_grid_b0.100_S1", "ppo_continuous"):
        agg = out[policy]
        for col in (
            "ret_garrido2024_sigmoid_mean",
            "zeta_avg_mean",
            "epsilon_avg_mean",
            "phi_avg_mean",
            "tau_avg_mean",
            "kappa_dot_mean",
        ):
            assert col in agg, f"missing {col} in aggregate"
        assert agg["ret_garrido2024_sigmoid_mean"] == 0.675


# Keys aggregate() reads from each row (besides the cost-aware ones under test).
_AGG_REQUIRED_KEYS = (
    "reward_total",
    "fill_rate_order_level",
    "order_level_ret_mean",
    "ret_mean_all_orders_zero_unfulfilled",
    "flow_fill_rate",
    "stockout_week_pct",
    "p10_step_flow_fill",
    "re_fr_contribution_all",
    "re_ap_contribution_all",
    "re_rp_contribution_all",
    "re_dp_rp_contribution_all",
    "dynamic_ret_contribution_all",
    "dynamic_case_pct",
    "pct_case_fill_rate",
    "pct_ret_eq_1",
    "pct_ret_lt_05",
    "ret_p10_all",
    "ret_p50_all",
    "ret_p90_all",
    "pct_steps_S1",
    "pct_steps_S2",
    "pct_steps_S3",
    "assembly_shift_hours",
    "inventory_target_total_mean",
    "pending_backorders_count",
    "pending_backorder_qty",
    "unattended_orders_total",
)
