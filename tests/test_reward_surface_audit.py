from __future__ import annotations

import argparse

import scripts.reward_surface_audit as reward_surface


def test_reward_surface_audit_forwards_ret_tail_knobs(tmp_path) -> None:
    args = argparse.Namespace(
        profiles="increased",
        panel_cfis="31",
        policy_set="with_crossed",
        replications=1,
        multiplier=2.0,
        ret_tail_w_sc=0.2,
        ret_tail_w_rc=0.6,
        ret_tail_w_ce=0.2,
        ret_tail_cap_kappa=0.35,
        ret_tail_inv_kappa=0.75,
        ret_tail_boost=3.0,
        ret_tail_transform="power",
        ret_tail_gamma=1.5,
        ret_tail_beta=2.0,
        ret_tail_transform_grid="identity",
    )

    command = reward_surface.build_auditor_command(
        "ReT_tail_v1",
        tmp_path,
        args,
    )

    assert command[command.index("--ret-tail-w-sc") + 1] == "0.2"
    assert command[command.index("--panel-cfis") + 1] == "31"
    assert command[command.index("--ret-tail-w-rc") + 1] == "0.6"
    assert command[command.index("--ret-tail-w-ce") + 1] == "0.2"
    assert command[command.index("--ret-tail-cap-kappa") + 1] == "0.35"
    assert command[command.index("--ret-tail-inv-kappa") + 1] == "0.75"
    assert command[command.index("--ret-tail-boost") + 1] == "3.0"
    assert command[command.index("--ret-tail-transform") + 1] == "power"
    assert command[command.index("--ret-tail-gamma") + 1] == "1.5"


def test_reward_surface_audit_expands_ret_tail_transform_grid() -> None:
    args = argparse.Namespace(
        rewards="ReT_tail_v1,ReT_ladder_v1",
        ret_tail_transform_grid="identity,power:1.25,exp_norm:4",
    )

    specs = reward_surface.iter_reward_specs(args)

    assert [spec.label for spec in specs] == [
        "ReT_tail_v1__identity",
        "ReT_tail_v1__power_g1p25",
        "ReT_tail_v1__exp_norm_b4",
        "ReT_ladder_v1",
    ]
