from __future__ import annotations

import argparse

import scripts.reward_surface_audit as reward_surface


def test_reward_surface_audit_forwards_ret_tail_knobs(tmp_path) -> None:
    args = argparse.Namespace(
        profiles="increased",
        policy_set="with_crossed",
        replications=1,
        multiplier=2.0,
        ret_tail_w_sc=0.2,
        ret_tail_w_rc=0.6,
        ret_tail_w_ce=0.2,
        ret_tail_cap_kappa=0.35,
        ret_tail_inv_kappa=0.75,
        ret_tail_boost=3.0,
    )

    command = reward_surface.build_auditor_command(
        "ReT_tail_v1",
        tmp_path,
        args,
    )

    assert command[command.index("--ret-tail-w-sc") + 1] == "0.2"
    assert command[command.index("--ret-tail-w-rc") + 1] == "0.6"
    assert command[command.index("--ret-tail-w-ce") + 1] == "0.2"
    assert command[command.index("--ret-tail-cap-kappa") + 1] == "0.35"
    assert command[command.index("--ret-tail-inv-kappa") + 1] == "0.75"
    assert command[command.index("--ret-tail-boost") + 1] == "3.0"
