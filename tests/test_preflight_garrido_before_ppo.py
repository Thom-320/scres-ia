from __future__ import annotations

import pytest

from scripts.preflight_garrido_before_ppo import _freeze_summary, build_parser, run


def test_preflight_freeze_summary_keeps_excel_as_primary_resilience() -> None:
    args = build_parser().parse_args(["--skip-reward-screen", "--skip-headroom-screen"])

    summary = _freeze_summary(args)

    assert summary["primary_resilience_metric"] == "mean_ret_excel_formula"
    assert "cd_sigmoid_mean" in summary["secondary_resilience_metrics"]
    assert summary["faithful_protocol"]["stochastic_pt"] is False
    assert summary["faithful_protocol"]["risk_frequency_multiplier"] == 1.0
    assert summary["faithful_protocol"]["risk_impact_multiplier"] == 1.0
    assert summary["gates_to_keep_green"]["forensic_replay_mae_required_max"] == 0.005
    assert summary["status"] == "blocked_reference_v2_not_promoted"


def test_preflight_blocks_new_ppo_when_reference_v2_is_not_promoted() -> None:
    args = build_parser().parse_args([])

    with pytest.raises(RuntimeError, match="reference_v2 failed"):
        run(args)
