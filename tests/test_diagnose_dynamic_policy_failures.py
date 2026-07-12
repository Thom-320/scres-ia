from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.diagnose_dynamic_policy_failures import diagnose_run


def _summary_row(
    *,
    regime: str,
    policy: str,
    cd: float,
    excel: float,
    flow: float,
    cvar: float,
    resource: float,
    s1: float,
    s2: float,
    s3: float,
) -> dict[str, object]:
    row: dict[str, object] = {
        "regime": regime,
        "policy": policy,
        "cd_sigmoid_mean_mean": cd,
        "mean_ret_excel_formula_mean": excel,
        "flow_fill_rate_mean": flow,
        "fill_rate_order_level_mean": flow,
        "service_loss_mean_mean": 0.1,
        "service_loss_p95_mean": cvar,
        "service_loss_cvar95_mean": cvar,
        "backorder_qty_total_mean": 10.0,
        "pending_backorder_qty_terminal_mean": 5.0,
        "unattended_orders_terminal_mean": 0.0,
        "resource_composite_total_mean": resource,
        "pct_steps_S1_mean": s1,
        "pct_steps_S2_mean": s2,
        "pct_steps_S3_mean": s3,
    }
    for component in (
        "cd_zeta_avg",
        "cd_epsilon_avg",
        "cd_phi_avg",
        "cd_tau_avg",
        "cd_kappa_dot",
    ):
        row[f"{component}_mean"] = 1.0
    return row


def test_diagnose_run_flags_train_eval_mismatch_and_static_shortfall(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    path.write_text(
        json.dumps(
            {
                "config": {
                    "reward_mode": "ReT_garrido2024_train",
                    "ret_g24_kappa_train_frac": 0.2,
                    "risk_frequency_multiplier": 2.0,
                    "stochastic_pt": False,
                },
                "policy_summary": [
                    _summary_row(
                        regime="current",
                        policy="ppo_dynamic",
                        cd=0.60,
                        excel=0.004,
                        flow=0.80,
                        cvar=0.60,
                        resource=300.0,
                        s1=40.0,
                        s2=10.0,
                        s3=50.0,
                    ),
                    _summary_row(
                        regime="current",
                        policy="static_S1_I168",
                        cd=0.70,
                        excel=0.005,
                        flow=0.90,
                        cvar=0.30,
                        resource=100.0,
                        s1=100.0,
                        s2=0.0,
                        s3=0.0,
                    ),
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = diagnose_run(path)

    assert payload["aggregate"]["verdict"] == "no_defensible_dynamic_win_currently"
    assert payload["diagnostics"][0]["delta_cd_sigmoid_mean"] == pytest.approx(-0.10)
    assert payload["diagnostics"][0]["shift_mix_warning"] is True
    assert any("reward/eval mismatch" in blocker for blocker in payload["blockers"])
    assert any("below best static" in blocker for blocker in payload["blockers"])
