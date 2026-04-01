from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import scripts.run_track_b_smoke as track_b_smoke


def test_build_static_policy_action_scales_downstream_dispatch() -> None:
    policy = track_b_smoke.StaticPolicySpec(
        label="s3_d2.00", assembly_shifts=3, downstream_multiplier=2.0
    )

    action = track_b_smoke.build_static_policy_action(policy)

    assert action["assembly_shifts"] == 3
    assert action["op10_q_min"] == pytest.approx(4800.0)
    assert action["op10_q_max"] == pytest.approx(5200.0)
    assert action["op12_q_min"] == pytest.approx(4800.0)
    assert action["op12_q_max"] == pytest.approx(5200.0)


def test_build_decision_summary_flags_promotable_gap() -> None:
    policy_rows = [
        {
            "policy": "s2_d1.00",
            "reward_total_mean": 100.0,
            "fill_rate_mean": 0.95,
            "backorder_rate_mean": 0.05,
            "order_level_ret_mean": 0.48,
            "ret_thesis_corrected_total_mean": 180.0,
            "ret_unified_total_mean": 175.0,
        },
        {
            "policy": "s3_d1.00",
            "reward_total_mean": 101.0,
            "fill_rate_mean": 0.96,
            "backorder_rate_mean": 0.04,
            "order_level_ret_mean": 0.49,
            "ret_thesis_corrected_total_mean": 181.0,
            "ret_unified_total_mean": 174.0,
        },
        {
            "policy": "s3_d2.00",
            "reward_total_mean": 99.5,
            "fill_rate_mean": 0.97,
            "backorder_rate_mean": 0.03,
            "order_level_ret_mean": 0.50,
            "ret_thesis_corrected_total_mean": 179.0,
            "ret_unified_total_mean": 170.0,
        },
        {
            "policy": "ppo",
            "reward_total_mean": 100.5,
            "fill_rate_mean": 0.965,
            "backorder_rate_mean": 0.035,
            "order_level_ret_mean": 0.495,
            "ret_thesis_corrected_total_mean": 183.0,
            "ret_unified_total_mean": 185.0,
        },
    ]

    decision = track_b_smoke.build_decision_summary(policy_rows)

    assert decision["baseline_policy"] == "s2_d1.00"
    assert decision["best_static_policy"] == "s3_d2.00"
    assert decision["ppo_fill_gap_vs_s2_neutral_pp"] == pytest.approx(1.5)
    assert decision["ppo_fill_gap_vs_best_static_pp"] == pytest.approx(-0.5)
    assert decision["promote_to_long_run"] is True


def test_main_writes_bundle_from_stubbed_training_and_eval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "track_b_smoke"

    class FakeVecNorm:
        training = False

        def close(self) -> None:
            return None

    def fake_train_ppo(
        args: object, seed: int, run_dir: Path
    ) -> tuple[object, FakeVecNorm]:
        del args
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "ppo_model.zip").write_text("stub", encoding="utf-8")
        (run_dir / "vec_normalize.pkl").write_text("stub", encoding="utf-8")
        return object(), FakeVecNorm()

    def fake_evaluate_static_policy(
        policy: track_b_smoke.StaticPolicySpec, *, args: object, seed: int
    ) -> list[dict[str, object]]:
        del args
        fill = {
            "s2_d1.00": 0.95,
            "s3_d1.00": 0.965,
            "s3_d2.00": 0.975,
        }[policy.label]
        return [
            {
                "policy": policy.label,
                "seed": seed,
                "episode": 1,
                "eval_seed": seed + 50_000,
                "steps": 10,
                "reward_total": 100.0 + fill,
                "fill_rate": fill,
                "backorder_rate": 1.0 - fill,
                "order_level_ret_mean": fill / 2.0,
                "ret_thesis_corrected_total": 150.0 + fill,
                "ret_unified_total": 140.0 + fill,
                "ret_unified_fr_mean": fill,
                "ret_unified_rc_mean": fill - 0.01,
                "ret_unified_ce_mean": 0.85,
                "ret_unified_gate_mean": 0.75,
                "flow_fill_rate": fill,
                "flow_backorder_rate": 1.0 - fill,
                "fill_rate_state_terminal": fill,
                "backorder_rate_state_terminal": 1.0 - fill,
                "terminal_rolling_fill_rate_4w": fill,
                "terminal_rolling_backorder_rate_4w": 1.0 - fill,
                "order_count": 25.0,
                "completed_order_count": 24.0,
                "completed_order_fraction": 0.96,
                "order_case_fill_rate_share": fill,
                "order_case_autotomy_share": 0.0,
                "order_case_recovery_share": 0.02,
                "order_case_non_recovery_share": 0.01,
                "order_case_unfulfilled_share": 0.01,
                "pct_steps_S1": 0.0,
                "pct_steps_S2": 100.0 if policy.assembly_shifts == 2 else 0.0,
                "pct_steps_S3": 100.0 if policy.assembly_shifts == 3 else 0.0,
            }
        ]

    def fake_evaluate_trained_policy(
        *, args: object, seed: int, model: object, vec_norm: FakeVecNorm
    ) -> list[dict[str, object]]:
        del args, model, vec_norm
        fill = 0.971
        return [
            {
                "policy": "ppo",
                "seed": seed,
                "episode": 1,
                "eval_seed": seed + 50_000,
                "steps": 10,
                "reward_total": 101.25,
                "fill_rate": fill,
                "backorder_rate": 1.0 - fill,
                "order_level_ret_mean": 0.499,
                "ret_thesis_corrected_total": 190.0,
                "ret_unified_total": 210.0,
                "ret_unified_fr_mean": 0.98,
                "ret_unified_rc_mean": 0.97,
                "ret_unified_ce_mean": 0.92,
                "ret_unified_gate_mean": 0.88,
                "flow_fill_rate": fill,
                "flow_backorder_rate": 1.0 - fill,
                "fill_rate_state_terminal": fill,
                "backorder_rate_state_terminal": 1.0 - fill,
                "terminal_rolling_fill_rate_4w": fill,
                "terminal_rolling_backorder_rate_4w": 1.0 - fill,
                "order_count": 25.0,
                "completed_order_count": 25.0,
                "completed_order_fraction": 1.0,
                "order_case_fill_rate_share": 0.97,
                "order_case_autotomy_share": 0.01,
                "order_case_recovery_share": 0.01,
                "order_case_non_recovery_share": 0.01,
                "order_case_unfulfilled_share": 0.0,
                "pct_steps_S1": 5.0,
                "pct_steps_S2": 30.0,
                "pct_steps_S3": 65.0,
            }
        ]

    monkeypatch.setattr(track_b_smoke, "train_ppo", fake_train_ppo)
    monkeypatch.setattr(
        track_b_smoke, "evaluate_static_policy", fake_evaluate_static_policy
    )
    monkeypatch.setattr(
        track_b_smoke, "evaluate_trained_policy", fake_evaluate_trained_policy
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_track_b_smoke.py",
            "--output-dir",
            str(output_dir),
            "--seeds",
            "11",
            "22",
            "--eval-episodes",
            "1",
        ],
    )

    track_b_smoke.main()

    payload = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))

    assert (output_dir / "episode_metrics.csv").exists()
    assert (output_dir / "seed_metrics.csv").exists()
    assert (output_dir / "policy_summary.csv").exists()
    assert (output_dir / "summary.md").exists()
    assert payload["decision"]["best_static_policy"] == "s3_d2.00"
    assert payload["decision"]["ppo_beats_s2_neutral_by_fill"] is True
    assert payload["policy_summary"][0]["ret_thesis_corrected_total_mean"] > 0.0
    assert payload["comparison_table"][0]["ppo_ret_unified_mean"] > 0.0
    assert len(payload["policy_summary"]) == 4
