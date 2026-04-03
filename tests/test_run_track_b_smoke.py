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


def test_extract_downstream_multipliers_handles_learned_and_static_payloads() -> None:
    learned = {"clipped_action": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0]}
    static = {
        "raw_action": {
            "op10_q_min": 3600.0,
            "op12_q_min": 4800.0,
        }
    }

    learned_op10, learned_op12 = track_b_smoke.extract_downstream_multipliers(learned)
    static_op10, static_op12 = track_b_smoke.extract_downstream_multipliers(static)

    assert learned_op10 == pytest.approx(2.0)
    assert learned_op12 == pytest.approx(0.5)
    assert static_op10 == pytest.approx(1.5)
    assert static_op12 == pytest.approx(2.0)


def test_build_decision_summary_flags_promotable_gap() -> None:
    policy_rows = [
        {"policy": "s1_d1.00", "reward_total_mean": 96.0, "fill_rate_mean": 0.920, "backorder_rate_mean": 0.080, "order_level_ret_mean": 0.44},
        {"policy": "s1_d1.50", "reward_total_mean": 97.0, "fill_rate_mean": 0.925, "backorder_rate_mean": 0.075, "order_level_ret_mean": 0.45},
        {"policy": "s1_d2.00", "reward_total_mean": 98.0, "fill_rate_mean": 0.930, "backorder_rate_mean": 0.070, "order_level_ret_mean": 0.46},
        {"policy": "s2_d1.00", "reward_total_mean": 100.0, "fill_rate_mean": 0.950, "backorder_rate_mean": 0.050, "order_level_ret_mean": 0.48},
        {"policy": "s2_d1.50", "reward_total_mean": 100.3, "fill_rate_mean": 0.955, "backorder_rate_mean": 0.045, "order_level_ret_mean": 0.485},
        {"policy": "s2_d2.00", "reward_total_mean": 100.5, "fill_rate_mean": 0.960, "backorder_rate_mean": 0.040, "order_level_ret_mean": 0.49},
        {"policy": "s3_d1.00", "reward_total_mean": 101.0, "fill_rate_mean": 0.960, "backorder_rate_mean": 0.040, "order_level_ret_mean": 0.49},
        {"policy": "s3_d1.50", "reward_total_mean": 100.8, "fill_rate_mean": 0.965, "backorder_rate_mean": 0.035, "order_level_ret_mean": 0.495},
        {"policy": "s3_d2.00", "reward_total_mean": 99.5, "fill_rate_mean": 0.970, "backorder_rate_mean": 0.030, "order_level_ret_mean": 0.50},
        {"policy": "ppo", "reward_total_mean": 100.5, "fill_rate_mean": 0.965, "backorder_rate_mean": 0.035, "order_level_ret_mean": 0.495},
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
            "s1_d1.00": 0.930,
            "s1_d1.50": 0.935,
            "s1_d2.00": 0.940,
            "s2_d1.00": 0.950,
            "s2_d1.50": 0.955,
            "s2_d2.00": 0.960,
            "s3_d1.00": 0.965,
            "s3_d1.50": 0.970,
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
                "flow_fill_rate": fill,
                "flow_backorder_rate": 1.0 - fill,
                "terminal_rolling_fill_rate_4w": fill,
                "terminal_rolling_backorder_rate_4w": 1.0 - fill,
                "pct_steps_S1": 100.0 if policy.assembly_shifts == 1 else 0.0,
                "pct_steps_S2": 100.0 if policy.assembly_shifts == 2 else 0.0,
                "pct_steps_S3": 100.0 if policy.assembly_shifts == 3 else 0.0,
                "op10_multiplier_step_mean": policy.downstream_multiplier,
                "op12_multiplier_step_mean": policy.downstream_multiplier,
                "op10_multiplier_step_p95": policy.downstream_multiplier,
                "op12_multiplier_step_p95": policy.downstream_multiplier,
                "pct_steps_op10_multiplier_ge_190": 100.0 if policy.downstream_multiplier >= 1.9 else 0.0,
                "pct_steps_op12_multiplier_ge_190": 100.0 if policy.downstream_multiplier >= 1.9 else 0.0,
                "pct_steps_both_downstream_ge_190": 100.0 if policy.downstream_multiplier >= 1.9 else 0.0,
                "assembly_hours_total": policy.assembly_shifts * 8.0 * 7.0 * 10,
                "assembly_cost_index": policy.assembly_shifts / 3.0,
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
                "flow_fill_rate": fill,
                "flow_backorder_rate": 1.0 - fill,
                "terminal_rolling_fill_rate_4w": fill,
                "terminal_rolling_backorder_rate_4w": 1.0 - fill,
                "pct_steps_S1": 5.0,
                "pct_steps_S2": 30.0,
                "pct_steps_S3": 65.0,
                "op10_multiplier_step_mean": 1.85,
                "op12_multiplier_step_mean": 1.90,
                "op10_multiplier_step_p95": 2.0,
                "op12_multiplier_step_p95": 2.0,
                "pct_steps_op10_multiplier_ge_190": 60.0,
                "pct_steps_op12_multiplier_ge_190": 70.0,
                "pct_steps_both_downstream_ge_190": 55.0,
                "assembly_hours_total": 1120.0,
                "assembly_cost_index": 0.867,
            }
        ]

    def fake_evaluate_heuristic_policy(
        label: str, heuristic: object, *, args: object, seed: int
    ) -> list[dict[str, object]]:
        del heuristic, args
        fill = {"heur_hysteresis": 0.950, "heur_disruption_aware": 0.945,
                "heur_tuned": 0.948, "heur_downstream_reactive": 0.935,
                "heur_s1_max_downstream": 0.940}.get(label, 0.940)
        return [
            {
                "policy": label,
                "seed": seed,
                "episode": 1,
                "eval_seed": seed + 50_000,
                "steps": 10,
                "reward_total": 100.0 + fill,
                "fill_rate": fill,
                "backorder_rate": 1.0 - fill,
                "order_level_ret_mean": fill / 2.0,
                "flow_fill_rate": fill,
                "flow_backorder_rate": 1.0 - fill,
                "terminal_rolling_fill_rate_4w": fill,
                "terminal_rolling_backorder_rate_4w": 1.0 - fill,
                "pct_steps_S1": 50.0,
                "pct_steps_S2": 30.0,
                "pct_steps_S3": 20.0,
                "op10_multiplier_step_mean": 1.5,
                "op12_multiplier_step_mean": 1.5,
                "op10_multiplier_step_p95": 2.0,
                "op12_multiplier_step_p95": 2.0,
                "pct_steps_op10_multiplier_ge_190": 30.0,
                "pct_steps_op12_multiplier_ge_190": 30.0,
                "pct_steps_both_downstream_ge_190": 20.0,
                "assembly_hours_total": 840.0,
                "assembly_cost_index": 0.567,
            }
        ]

    monkeypatch.setattr(track_b_smoke, "train_ppo", fake_train_ppo)
    monkeypatch.setattr(
        track_b_smoke, "evaluate_static_policy", fake_evaluate_static_policy
    )
    monkeypatch.setattr(
        track_b_smoke, "evaluate_heuristic_policy", fake_evaluate_heuristic_policy
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
    assert len(payload["policy_summary"]) == 15  # 9 static + 5 heuristic + 1 ppo
