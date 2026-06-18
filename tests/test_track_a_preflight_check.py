from __future__ import annotations

import scripts.track_a_preflight_check as preflight


def test_track_a_preflight_builds_strict_thesis_factorized_command(tmp_path):
    args = preflight.build_parser().parse_args(
        [
            "--output-root",
            str(tmp_path),
            "--algo",
            "dmlpa_ppo",
            "--risk-level",
            "severe",
            "--ret-tail-transform",
            "power",
            "--ret-tail-gamma",
            "1.25",
        ]
    )

    command = preflight.build_probe_command(args)
    checks = preflight.validate_command(command, args)

    assert all(item["pass"] for item in checks)
    assert command[command.index("--algo") + 1] == "dmlpa_ppo"
    assert command[command.index("--action-space-mode") + 1] == "thesis_factorized"
    assert command[command.index("--reward-mode") + 1] == "ReT_tail_v1"
    assert command[command.index("--risk-occurrence-mode") + 1] == "thesis_periodic"
    assert command[command.index("--raw-material-flow-mode") + 1] == (
        "kit_equivalent_order_up_to"
    )
    assert command[command.index("--raw-material-order-up-to-multiplier") + 1] == "2.0"
    assert "--stochastic-pt" in command
    assert command[command.index("--stochastic-pt-spread") + 1] == "1.0"
    assert "--norm-reward" in command
    assert "--profile-eval-common-seed" in command
    assert "continuous_it_s" not in command
