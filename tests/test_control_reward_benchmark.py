from __future__ import annotations

import json
from pathlib import Path

import pytest

from supply_chain.external_env_interface import make_shift_control_env
import numpy as np

from scripts.benchmark_control_reward import (
    HEURISTIC_DEFAULTS,
    HEURISTIC_POLICY_NAMES,
    RANDOM_POLICY_NAME,
    HeuristicDisruptionAware,
    HeuristicHysteresis,
    HeuristicTuned,
    build_env_kwargs,
    build_comparison_rows,
    build_parser,
    export_artifact_bundle,
    pick_survivors,
    resolve_output_dir,
    run_benchmark,
    static_policy_action,
    tune_heuristic_params,
)


def test_control_v1_step_exposes_reward_components_and_corrected_ret() -> None:
    env = make_shift_control_env(
        reward_mode="control_v1",
        step_size_hours=24,
        max_steps=2,
        w_bo=2.0,
        w_cost=0.06,
        w_disr=0.1,
    )
    env.reset(seed=7)
    _, reward, _, _, info = env.step(static_policy_action("static_s2"))

    components = info["control_components"]
    expected_reward = -(
        components["weighted_service_loss"]
        + components["weighted_shift_cost"]
        + components["weighted_disruption"]
    )
    assert reward == pytest.approx(expected_reward)
    assert info["reward_mode"] == "control_v1"
    assert "ret_thesis_corrected" in info
    assert info["ret_thesis_corrected"]["correction_mode"] == "autotomy_equals_recovery"
    assert info["shift_cost_step"] == pytest.approx(1.0)
    assert info["service_loss_step"] == pytest.approx(components["service_loss_step"])


def test_static_policy_actions_cover_all_shift_modes() -> None:
    env = make_shift_control_env(
        reward_mode="control_v1",
        step_size_hours=24,
        max_steps=1,
        risk_level="increased",
    )
    env.reset(seed=7)
    _, _, _, _, s1_info = env.step(static_policy_action("static_s1"))
    assert s1_info["shifts_active"] == 1

    env.reset(seed=7)
    _, _, _, _, s2_info = env.step(static_policy_action("static_s2"))
    assert s2_info["shifts_active"] == 2

    env.reset(seed=7)
    _, _, _, _, s3_info = env.step(static_policy_action("static_s3"))
    assert s3_info["shifts_active"] == 3


def test_pick_survivors_prefers_non_s1_fixed_baselines() -> None:
    args = build_parser().parse_args([])
    policy_rows = [
        {
            "phase": "static_screen",
            "policy": "static_s1",
            "algo": "ppo",
            "frame_stack": 1,
            "observation_version": "v1",
            "w_bo": 1.0,
            "w_cost": 0.02,
            "w_disr": 0.0,
            "reward_total_mean": 10.0,
            "fill_rate_mean": 0.60,
        },
        {
            "phase": "static_screen",
            "policy": "static_s2",
            "algo": "ppo",
            "frame_stack": 1,
            "observation_version": "v1",
            "w_bo": 1.0,
            "w_cost": 0.02,
            "w_disr": 0.0,
            "reward_total_mean": 12.5,
            "fill_rate_mean": 0.75,
        },
        {
            "phase": "static_screen",
            "policy": "static_s3",
            "algo": "ppo",
            "frame_stack": 1,
            "observation_version": "v1",
            "w_bo": 1.0,
            "w_cost": 0.02,
            "w_disr": 0.0,
            "reward_total_mean": 11.0,
            "fill_rate_mean": 0.72,
        },
        {
            "phase": "static_screen",
            "policy": "static_s1",
            "algo": "ppo",
            "frame_stack": 1,
            "observation_version": "v1",
            "w_bo": 1.0,
            "w_cost": 0.10,
            "w_disr": 0.0,
            "reward_total_mean": 10.0,
            "fill_rate_mean": 0.60,
        },
        {
            "phase": "static_screen",
            "policy": "static_s2",
            "algo": "ppo",
            "frame_stack": 1,
            "observation_version": "v1",
            "w_bo": 1.0,
            "w_cost": 0.10,
            "w_disr": 0.0,
            "reward_total_mean": 9.0,
            "fill_rate_mean": 0.75,
        },
        {
            "phase": "static_screen",
            "policy": "static_s3",
            "algo": "ppo",
            "frame_stack": 1,
            "observation_version": "v1",
            "w_bo": 1.0,
            "w_cost": 0.10,
            "w_disr": 0.0,
            "reward_total_mean": 8.0,
            "fill_rate_mean": 0.78,
        },
    ]

    survivors = pick_survivors(policy_rows, args)
    assert len(survivors) == 1
    assert survivors[0]["best_static_policy"] == "static_s2"
    assert survivors[0]["static_reward_gap_best_minus_s1"] == pytest.approx(2.5)


def test_build_env_kwargs_passes_stochastic_pt() -> None:
    args = build_parser().parse_args(["--stochastic-pt", "--observation-version", "v2"])
    env_kwargs = build_env_kwargs(
        args,
        {
            "w_bo": 4.0,
            "w_cost": 0.02,
            "w_disr": 0.0,
        },
    )
    assert env_kwargs["stochastic_pt"] is True
    assert env_kwargs["reward_mode"] == "control_v1"
    assert env_kwargs["observation_version"] == "v2"


def test_resolve_output_dir_disambiguates_algo_and_frame_stack() -> None:
    args = build_parser().parse_args(["--algo", "sac", "--frame-stack", "4"])
    output_dir = resolve_output_dir(args)
    assert output_dir.name == "control_reward_sac_fs4"


def test_resolve_output_dir_supports_recurrent_ppo() -> None:
    args = build_parser().parse_args(["--algo", "recurrent_ppo"])
    output_dir = resolve_output_dir(args)
    assert output_dir.name == "control_reward_recurrent_ppo_fs1"


def test_build_comparison_rows_marks_collapse_and_reward_wins() -> None:
    args = build_parser().parse_args(
        ["--algo", "ppo", "--frame-stack", "4", "--observation-version", "v2"]
    )
    survivors = [
        {
            "w_bo": 2.0,
            "w_cost": 0.02,
            "w_disr": 0.0,
            "best_static_policy": "static_s2",
            "static_reward_gap_best_minus_s1": 5.0,
        }
    ]
    policy_rows = [
        {
            "phase": "static_screen",
            "policy": "static_s2",
            "algo": "ppo",
            "frame_stack": 4,
            "observation_version": "v2",
            "w_bo": 2.0,
            "w_cost": 0.02,
            "w_disr": 0.0,
            "reward_total_mean": 15.0,
            "fill_rate_mean": 0.82,
            "backorder_rate_mean": 0.18,
            "ret_thesis_corrected_total_mean": 240.0,
        },
        {
            "phase": "random_eval",
            "policy": RANDOM_POLICY_NAME,
            "algo": "ppo",
            "frame_stack": 4,
            "observation_version": "v2",
            "w_bo": 2.0,
            "w_cost": 0.02,
            "w_disr": 0.0,
            "reward_total_mean": 8.0,
            "fill_rate_mean": 0.70,
            "backorder_rate_mean": 0.30,
            "ret_thesis_corrected_total_mean": 235.0,
        },
        {
            "phase": "ppo_eval",
            "policy": "ppo",
            "algo": "ppo",
            "frame_stack": 4,
            "observation_version": "v2",
            "w_bo": 2.0,
            "w_cost": 0.02,
            "w_disr": 0.0,
            "reward_total_mean": 16.0,
            "fill_rate_mean": 0.82,
            "backorder_rate_mean": 0.18,
            "ret_thesis_corrected_total_mean": 241.0,
            "pct_steps_S1_mean": 95.0,
            "pct_steps_S2_mean": 5.0,
            "pct_steps_S3_mean": 0.0,
        },
    ]

    comparison_rows = build_comparison_rows(policy_rows, survivors, args=args)
    assert len(comparison_rows) == 1
    assert comparison_rows[0]["ppo_beats_static_s2"] is True
    assert comparison_rows[0]["ppo_beats_best_static"] is True
    assert comparison_rows[0]["learned_beats_random"] is True
    assert comparison_rows[0]["learned_beats_best_static"] is True
    assert comparison_rows[0]["collapsed_to_S1"] is True
    assert comparison_rows[0]["collapsed_to_S2"] is False
    assert comparison_rows[0]["algo"] == "ppo"
    assert comparison_rows[0]["frame_stack"] == 4
    assert comparison_rows[0]["observation_version"] == "v2"


def test_run_benchmark_smoke_writes_expected_artifacts(tmp_path: Path) -> None:
    parser = build_parser()
    artifact_root = tmp_path / "artifacts"
    args = parser.parse_args(
        [
            "--seeds",
            "1",
            "--train-timesteps",
            "32",
            "--eval-episodes",
            "1",
            "--step-size-hours",
            "24",
            "--max-steps",
            "4",
            "--w-bo",
            "1.0",
            "--w-cost",
            "0.02",
            "--w-disr",
            "0.0",
            "--algo",
            "sac",
            "--observation-version",
            "v2",
            "--stochastic-pt",
            "--output-dir",
            str(tmp_path),
            "--artifact-root",
            str(artifact_root),
        ]
    )
    args.invocation = "python scripts/benchmark_control_reward.py --smoke"
    summary = run_benchmark(args)

    episode_csv = tmp_path / "episode_metrics.csv"
    policy_csv = tmp_path / "policy_summary.csv"
    comparison_csv = tmp_path / "comparison_table.csv"
    summary_json = tmp_path / "summary.json"
    manifest_json = artifact_root / tmp_path.name / "manifest.json"

    assert episode_csv.exists()
    assert policy_csv.exists()
    assert comparison_csv.exists()
    assert summary_json.exists()
    assert manifest_json.exists()
    assert "static_s3" in summary["policies"]
    assert "random" in summary["policies"]

    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    assert payload["config"]["train_timesteps"] == 32
    assert payload["config"]["w_disr"] == [0.0]
    assert payload["config"]["algo"] == "sac"
    assert payload["config"]["frame_stack"] == 1
    assert payload["config"]["observation_version"] == "v2"
    assert payload["config"]["reward_mode"] == "control_v1"
    assert payload["config"]["stochastic_pt"] is True
    assert payload["benchmark_metadata"]["command"] == args.invocation
    assert "artifact_bundle_dir" in payload["artifacts"]

    manifest = json.loads(manifest_json.read_text(encoding="utf-8"))
    assert manifest["command"] == "python scripts/benchmark_control_reward.py --smoke"
    assert manifest["source_benchmark_directory"] == str(tmp_path.resolve())
    assert Path(manifest["files"]["comparison_table.csv"]).exists()


def test_run_benchmark_smoke_supports_frame_stack(tmp_path: Path) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--seeds",
            "1",
            "--train-timesteps",
            "32",
            "--eval-episodes",
            "1",
            "--step-size-hours",
            "24",
            "--max-steps",
            "4",
            "--w-bo",
            "1.0",
            "--w-cost",
            "0.02",
            "--w-disr",
            "0.0",
            "--algo",
            "ppo",
            "--frame-stack",
            "4",
            "--observation-version",
            "v2",
            "--output-dir",
            str(tmp_path),
            "--skip-artifact-export",
        ]
    )
    args.invocation = "python scripts/benchmark_control_reward.py --ppo-stack-smoke"
    summary = run_benchmark(args)

    payload = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert payload["config"]["algo"] == "ppo"
    assert payload["config"]["frame_stack"] == 4
    assert payload["config"]["observation_version"] == "v2"
    assert summary["config"]["algo"] == "ppo"
    assert summary["config"]["frame_stack"] == 4
    assert summary["config"]["observation_version"] == "v2"
    assert (tmp_path / "comparison_table.csv").exists()


def test_run_benchmark_smoke_supports_recurrent_ppo(tmp_path: Path) -> None:
    pytest.importorskip("sb3_contrib")
    parser = build_parser()
    args = parser.parse_args(
        [
            "--seeds",
            "1",
            "--train-timesteps",
            "32",
            "--eval-episodes",
            "1",
            "--step-size-hours",
            "24",
            "--max-steps",
            "4",
            "--w-bo",
            "1.0",
            "--w-cost",
            "0.02",
            "--w-disr",
            "0.0",
            "--algo",
            "recurrent_ppo",
            "--observation-version",
            "v2",
            "--output-dir",
            str(tmp_path),
            "--skip-artifact-export",
        ]
    )
    args.invocation = "python scripts/benchmark_control_reward.py --recurrent-ppo-smoke"
    summary = run_benchmark(args)

    payload = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert payload["config"]["algo"] == "recurrent_ppo"
    assert payload["config"]["observation_version"] == "v2"
    assert summary["config"]["algo"] == "recurrent_ppo"
    assert summary["config"]["observation_version"] == "v2"
    assert (tmp_path / "comparison_table.csv").exists()


def test_export_artifact_bundle_copies_expected_files(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    for filename in ("comparison_table.csv", "policy_summary.csv", "summary.json"):
        (source_dir / filename).write_text(f"{filename}\n", encoding="utf-8")

    bundle_dir = export_artifact_bundle(
        source_dir=source_dir,
        artifact_root=tmp_path / "artifacts",
        label="control_reward_test",
        summary={"config": {"train_timesteps": 10}},
        command="python scripts/benchmark_control_reward.py --dummy",
    )

    assert (bundle_dir / "comparison_table.csv").exists()
    assert (bundle_dir / "policy_summary.csv").exists()
    assert (bundle_dir / "summary.json").exists()
    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["command"] == "python scripts/benchmark_control_reward.py --dummy"
    assert manifest["config"]["train_timesteps"] == 10
    assert manifest["source_benchmark_directory"] == str(source_dir.resolve())


# ---------------------------------------------------------------------------
# Heuristic policy unit tests
# ---------------------------------------------------------------------------


def _fake_obs(overrides: dict[int, float] | None = None) -> np.ndarray:
    """Build a 15-dim observation vector with sensible defaults and overrides.

    Index mapping (from env_experimental_shifts.py):
      6 = fill_rate, 7 = backorder_rate, 8 = assembly_down, 9 = any_loc_down
    """
    defaults = np.zeros(15, dtype=np.float32)
    defaults[6] = 0.95  # fill_rate
    if overrides:
        for idx, val in overrides.items():
            defaults[idx] = val
    return defaults


def test_heuristic_hysteresis_escalates_on_high_backorder() -> None:
    h = HeuristicHysteresis()
    h.reset()
    obs = _fake_obs({7: 0.20})  # above tau_high=0.15
    action = h(obs, {})
    assert action.shape == (5,)
    assert action[4] == pytest.approx(1.0)  # S3


def test_heuristic_hysteresis_maintains_in_deadband() -> None:
    h = HeuristicHysteresis()
    h.reset()
    # First escalate to S3
    h(_fake_obs({7: 0.20}), {})
    # Then give a value inside the deadband — should stay S3
    action = h(_fake_obs({7: 0.10}), {})
    assert action[4] == pytest.approx(1.0)  # still S3


def test_heuristic_hysteresis_deescalates_below_low() -> None:
    h = HeuristicHysteresis()
    h.reset()
    obs = _fake_obs({7: 0.03})  # below tau_low=0.05
    action = h(obs, {})
    assert action[4] == pytest.approx(-1.0)  # S1


def test_heuristic_disruption_responds_to_assembly_down() -> None:
    h = HeuristicDisruptionAware()
    h.reset()
    obs = _fake_obs({8: 1.0})  # assembly_down
    action = h(obs, {})
    assert action[4] == pytest.approx(1.0)  # S3
    assert action[0] == pytest.approx(1.0)  # max inventory boost


def test_heuristic_disruption_caution_on_low_fill_rate() -> None:
    h = HeuristicDisruptionAware()
    h.reset()
    obs = _fake_obs({6: 0.85})  # below fill_rate_caution=0.90
    action = h(obs, {})
    assert action[4] == pytest.approx(0.0)  # S2
    assert action[0] == pytest.approx(0.5)  # moderate boost


def test_heuristic_disruption_normal_mode() -> None:
    h = HeuristicDisruptionAware()
    h.reset()
    obs = _fake_obs({6: 0.95})  # above caution
    action = h(obs, {})
    assert action[4] == pytest.approx(-1.0)  # S1
    assert action[0] == pytest.approx(0.0)  # neutral


def test_heuristic_tuned_combines_strategies() -> None:
    h = HeuristicTuned()
    h.reset()
    obs = _fake_obs({7: 0.20, 8: 1.0})  # high backorder + assembly_down
    action = h(obs, {})
    assert action[4] == pytest.approx(1.0)  # S3 (hysteresis)
    assert action[0] == pytest.approx(1.0)  # crisis boost


def test_all_heuristics_within_action_bounds() -> None:
    env = make_shift_control_env(
        reward_mode="control_v1",
        step_size_hours=24,
        max_steps=5,
    )
    for name in HEURISTIC_POLICY_NAMES:
        heuristic = HEURISTIC_DEFAULTS[name]
        heuristic.reset()
        obs, info = env.reset(seed=42)
        for _ in range(5):
            action = heuristic(obs, info)
            assert action.shape == (5,), f"{name}: bad shape {action.shape}"
            assert np.all(action >= -1.0) and np.all(
                action <= 1.0
            ), f"{name}: action out of bounds {action}"
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break


def test_run_benchmark_smoke_with_heuristics(tmp_path: Path) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--seeds",
            "1",
            "--train-timesteps",
            "32",
            "--eval-episodes",
            "1",
            "--step-size-hours",
            "24",
            "--max-steps",
            "4",
            "--w-bo",
            "1.0",
            "--w-cost",
            "0.02",
            "--w-disr",
            "0.0",
            "--algo",
            "ppo",
            "--observation-version",
            "v2",
            "--output-dir",
            str(tmp_path),
            "--skip-artifact-export",
        ],
    )
    args.invocation = "python scripts/benchmark_control_reward.py --heuristic-smoke"
    summary = run_benchmark(args)

    assert "heuristic_hysteresis" in summary["policies"]
    assert "heuristic_disruption" in summary["policies"]
    assert "heuristic_tuned" in summary["policies"]
    assert "heuristic_eval" in summary["phases"]

    import csv

    with open(tmp_path / "episode_metrics.csv", newline="") as f:
        reader = csv.DictReader(f)
        heuristic_rows = [r for r in reader if r["phase"] == "heuristic_eval"]
    assert len(heuristic_rows) >= 3  # at least one episode per heuristic


def test_tune_heuristic_params_returns_best(tmp_path: Path) -> None:
    args = build_parser().parse_args(
        [
            "--seeds",
            "1",
            "--eval-episodes",
            "1",
            "--tune-episodes",
            "1",
            "--step-size-hours",
            "24",
            "--max-steps",
            "4",
            "--w-bo",
            "1.0",
            "--w-cost",
            "0.02",
            "--w-disr",
            "0.0",
            "--output-dir",
            str(tmp_path),
        ],
    )
    result = tune_heuristic_params(args)
    assert "best_params" in result
    assert "best_mean_reward" in result
    assert result["combos_evaluated"] > 0
    assert "tau_high" in result["best_params"]
    assert "boost_crisis" in result["best_params"]
    # Verify global default was updated
    tuned = HEURISTIC_DEFAULTS["heuristic_tuned"]
    assert tuned.tau_high == result["best_params"]["tau_high"]


def test_run_benchmark_with_tune_heuristic(tmp_path: Path) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--seeds",
            "1",
            "--train-timesteps",
            "32",
            "--eval-episodes",
            "1",
            "--tune-episodes",
            "1",
            "--step-size-hours",
            "24",
            "--max-steps",
            "4",
            "--w-bo",
            "1.0",
            "--w-cost",
            "0.02",
            "--w-disr",
            "0.0",
            "--algo",
            "ppo",
            "--tune-heuristic",
            "--output-dir",
            str(tmp_path),
            "--skip-artifact-export",
        ],
    )
    args.invocation = "python scripts/benchmark_control_reward.py --tune-smoke"
    summary = run_benchmark(args)

    assert "heuristic_tuning" in summary
    assert summary["heuristic_tuning"]["combos_evaluated"] > 0
    assert "tau_high" in summary["heuristic_tuning"]["best_params"]


def test_cross_scenario_evaluation(tmp_path: Path) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--seeds",
            "1",
            "--train-timesteps",
            "32",
            "--eval-episodes",
            "1",
            "--step-size-hours",
            "24",
            "--max-steps",
            "4",
            "--w-bo",
            "1.0",
            "--w-cost",
            "0.02",
            "--w-disr",
            "0.0",
            "--algo",
            "ppo",
            "--risk-level",
            "increased",
            "--eval-risk-levels",
            "current",
            "severe",
            "--output-dir",
            str(tmp_path),
            "--skip-artifact-export",
        ],
    )
    args.invocation = (
        "python scripts/benchmark_control_reward.py --cross-scenario-smoke"
    )
    run_benchmark(args)

    import csv

    with open(tmp_path / "episode_metrics.csv", newline="") as f:
        reader = csv.DictReader(f)
        phases = {r["phase"] for r in reader}
    assert "cross_eval_current" in phases
    assert "cross_eval_severe" in phases
    # The training risk level should NOT appear as a cross_eval phase
    assert "cross_eval_increased" not in phases
