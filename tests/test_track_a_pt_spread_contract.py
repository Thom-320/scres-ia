from __future__ import annotations

import argparse

import pytest

from scripts.run_unified_thesis_evaluation import base_kwargs


def test_track_a_base_kwargs_passes_stochastic_pt_spread() -> None:
    args = argparse.Namespace(
        reward_mode="ReT_thesis",
        risk_level="severe",
        observation_version="v5",
        observation_mode="env_sdm_history_reward",
        step_size_hours=168.0,
        max_steps=80,
        stochastic_pt=True,
        stochastic_pt_spread=1.75,
        stochastic_pt_mean_preserving=True,
        raw_material_flow_mode="kit_equivalent_order_up_to",
        raw_material_order_up_to_multiplier=2.0,
        risk_occurrence_mode="thesis_periodic",
    )

    kwargs = base_kwargs(args)

    assert kwargs["stochastic_pt"] is True
    assert kwargs["stochastic_pt_spread"] == pytest.approx(1.75)
    assert kwargs["stochastic_pt_mean_preserving"] is True
    assert kwargs["raw_material_flow_mode"] == "kit_equivalent_order_up_to"
    assert kwargs["risk_occurrence_mode"] == "thesis_periodic"
