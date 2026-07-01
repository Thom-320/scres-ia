from __future__ import annotations

from scripts.run_track_a_headroom_search import (
    FAMILY_RISKS,
    continuous_candidates,
    per_op_candidates,
    summarize_gate,
)
from supply_chain.continuous_its_env import make_per_op_buffer_multidiscrete_track_a_env


def test_family_risk_sets_include_required_lanes() -> None:
    assert {"R1", "R2", "R3", "R24", "mixed"} <= set(FAMILY_RISKS)
    assert FAMILY_RISKS["mixed"] is None
    assert FAMILY_RISKS["R24"] == ("R24",)


def test_continuous_candidates_encode_fraction_shift_and_resource() -> None:
    candidates = continuous_candidates([0.0, 0.1], [1, 3])
    by_label = {c.label: c for c in candidates}

    assert by_label["f0.1_S1"].action == (0.1, -1.0)
    assert by_label["f0.1_S1"].resource == 0.05
    assert by_label["f0.1_S3"].action == (0.1, 1.0)
    assert by_label["f0.1_S3"].resource == 0.55


def test_per_op_targeted_candidates_include_op9_only_policy() -> None:
    candidates = per_op_candidates([0.0, 0.1], [1], grid="targeted")
    by_label = {c.label: c for c in candidates}

    assert "op30_op50_op90.1_S1" in by_label
    assert by_label["op30_op50_op90.1_S1"].action == (0.0, 0.0, 0.1, -1.0)


def test_summarize_gate_detects_moving_oracle_headroom() -> None:
    candidates = continuous_candidates([0.0, 0.1], [1])
    rows = []
    # Regime A likes f0.0, regime B likes f0.1; no single constant matches oracle.
    for seed in (1, 2):
        rows.extend(
            [
                {
                    "regime": "A",
                    "candidate": "f0_S1",
                    "seed": seed,
                    "excel": 1.0,
                },
                {
                    "regime": "A",
                    "candidate": "f0.1_S1",
                    "seed": seed,
                    "excel": 0.0,
                },
                {
                    "regime": "B",
                    "candidate": "f0_S1",
                    "seed": seed,
                    "excel": 0.0,
                },
                {
                    "regime": "B",
                    "candidate": "f0.1_S1",
                    "seed": seed,
                    "excel": 1.0,
                },
            ]
        )

    summary = summarize_gate(rows, candidates, ["A", "B"], [1, 2])

    assert summary["best_action_changes_across_regimes"] is True
    assert summary["oracle_minus_best_static"] == 0.5
    assert summary["opening_real"] is True


def test_per_op_multidiscrete_decodes_and_applies_exact_buffers() -> None:
    env = make_per_op_buffer_multidiscrete_track_a_env(
        reward_mode="ReT_excel_delta",
        observation_version="v6",
        risk_level="current",
        max_steps=1,
        priming_enabled=False,
        risk_obs=False,
        frac_grid=[0.0, 0.05, 0.10, 0.25],
    )
    try:
        decoded = env.decode_action([2, 1, 0, 2])
        assert decoded.tolist() == [0.10000000149011612, 0.05000000074505806, 0.0, 1.0]

        env.reset(seed=123)
        _obs, _reward, _done, _truncated, info = env.step([2, 1, 0, 2])

        assert info["action_space_mode"] == "per_op_buffer_multidiscrete"
        assert info["per_op_multidiscrete_indices"] == [2, 1, 0, 2]
        assert info["per_op_op3_frac"] == 0.10000000149011612
        assert info["per_op_op5_frac"] == 0.05000000074505806
        assert info["per_op_op9_frac"] == 0.0
        assert info["assembly_shift_signal_level"] == 3
    finally:
        env.close()
