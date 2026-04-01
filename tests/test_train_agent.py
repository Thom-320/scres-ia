from __future__ import annotations

import json
import numpy as np
import pytest

from supply_chain.config import RET_SHIFT_COST_DELTA_DEFAULT
from supply_chain.env_experimental_shifts import MFSCGymEnvShifts
from train_agent import (
    build_env_instance,
    build_parser,
    resolve_episode_max_steps,
    validate_args,
)


def test_shift_env_ret_thesis_emits_ret_metadata() -> None:
    env = MFSCGymEnvShifts(
        step_size_hours=24,
        max_steps=2,
        reward_mode="ReT_thesis",
    )
    env.reset(seed=42)
    _, reward, _, _, info = env.step(np.zeros(5, dtype=np.float32))
    assert isinstance(reward, float)
    assert info["reward_mode"] == "ReT_thesis"
    assert "ReT_raw" in info
    assert "shifts_active" in info


def test_shift_env_ret_corrected_cost_alias_preserves_lane_name() -> None:
    env = MFSCGymEnvShifts(
        step_size_hours=24,
        max_steps=2,
        reward_mode="ReT_corrected_cost",
        rt_delta=0.04,
    )
    env.reset(seed=42)
    _, reward, _, _, info = env.step(np.zeros(5, dtype=np.float32))
    assert isinstance(reward, float)
    assert env.reward_mode == "ReT_corrected_cost"
    assert info["reward_mode"] == "ReT_corrected_cost"
    assert "ReT_corrected_raw" in info
    assert info["shift_cost_linear"] >= 0.0


def test_shift_env_rt_v0_emits_shift_metadata() -> None:
    env = MFSCGymEnvShifts(
        step_size_hours=24,
        max_steps=2,
        reward_mode="rt_v0",
    )
    env.reset(seed=42)
    _, _, _, _, info = env.step(np.zeros(5, dtype=np.float32))
    assert info["reward_mode"] == "rt_v0"
    assert "shifts_active" in info
    assert "shift_cost_linear" in info


def test_shift_env_ret_seq_v1_emits_operational_resilience_metadata() -> None:
    env = MFSCGymEnvShifts(
        step_size_hours=24,
        max_steps=2,
        reward_mode="ReT_seq_v1",
        observation_version="v4",
        ret_seq_kappa=0.20,
    )
    env.reset(seed=42)
    _, reward, _, _, info = env.step(np.zeros(5, dtype=np.float32))
    assert isinstance(reward, float)
    assert info["reward_mode"] == "ReT_seq_v1"
    assert info["ret_seq_step"] == pytest.approx(reward)
    assert "ret_seq_components" in info
    assert "service_continuity_step" in info
    assert "backlog_containment_step" in info
    assert "adaptive_efficiency_step" in info
    assert info["ret_seq_kappa"] == pytest.approx(0.20)
    assert "ret_garrido2024_components" in info
    assert info["ret_garrido2024_sigmoid_step"] > 0.0
    assert info["ret_garrido2024_components"]["evaluation_index_recommendation"] == (
        "ReT_garrido2024"
    )
    assert info["cumulative_demanded_post_warmup"] >= 0.0
    assert info["cumulative_backorder_qty_post_warmup"] >= 0.0


def test_shift_env_ret_garrido2024_raw_emits_paper_faithful_metadata() -> None:
    env = MFSCGymEnvShifts(
        step_size_hours=24,
        max_steps=2,
        reward_mode="ReT_garrido2024_raw",
        observation_version="v4",
    )
    env.reset(seed=42)
    _, reward, _, _, info = env.step(np.zeros(5, dtype=np.float32))
    assert isinstance(reward, float)
    assert info["reward_mode"] == "ReT_garrido2024_raw"
    assert info["ret_garrido2024_step"] == pytest.approx(reward)
    assert info["ret_garrido2024_raw_step"] == pytest.approx(reward)
    assert "ret_garrido2024_components" in info
    assert info["ret_garrido2024_components"]["training_reward_recommendation"] == (
        "ReT_garrido2024_train"
    )
    assert info["ret_garrido2024_components"]["evaluation_index_recommendation"] == (
        "ReT_garrido2024"
    )
    assert info["ret_garrido2024_components"]["zeta_avg"] > 0.0
    assert info["ret_garrido2024_components"]["kappa_dot"] > 0.0


def test_shift_env_ret_garrido2024_sigmoid_is_bounded() -> None:
    env = MFSCGymEnvShifts(
        step_size_hours=24,
        max_steps=2,
        reward_mode="ReT_garrido2024",
        observation_version="v4",
    )
    env.reset(seed=42)
    _, reward, _, _, info = env.step(np.zeros(5, dtype=np.float32))
    assert isinstance(reward, float)
    assert 0.0 < reward < 1.0
    assert info["reward_mode"] == "ReT_garrido2024"
    assert info["ret_garrido2024_step"] == pytest.approx(reward)
    assert info["ret_garrido2024_sigmoid_step"] == pytest.approx(reward)


def test_shift_env_reset_primes_to_operational_state() -> None:
    env = MFSCGymEnvShifts(
        step_size_hours=168,
        max_steps=2,
        reward_mode="control_v1",
        risk_level="current",
        stochastic_pt=False,
    )
    _, info = env.reset(seed=42)
    state_context = info["state_constraint_context"]
    warmup_metadata = info["warmup_metadata"]

    assert info["time"] > env.warmup_hours
    assert warmup_metadata["primed_ready"] is True
    assert warmup_metadata["post_warmup_start_time"] == pytest.approx(info["time"])
    assert state_context["post_warmup_start_time"] == pytest.approx(info["time"])
    assert (
        warmup_metadata["reset_operational_context"]["fill_rate"]
        >= warmup_metadata["operational_fill_rate_threshold"]
    )
    assert state_context["inventory_detail"]["rations_theatre"] > 0.0
    assert state_context["cumulative_demanded_post_warmup"] == pytest.approx(0.0)
    assert state_context["cumulative_backorder_qty_post_warmup"] == pytest.approx(0.0)


def test_shift_env_ret_cd_v1_emits_continuous_resilience_metadata() -> None:
    env = MFSCGymEnvShifts(
        step_size_hours=24,
        max_steps=2,
        reward_mode="ReT_cd_v1",
        observation_version="v4",
    )
    env.reset(seed=42)
    _, reward, _, _, info = env.step(np.zeros(5, dtype=np.float32))
    assert isinstance(reward, float)
    assert info["reward_mode"] == "ReT_cd_v1"
    assert info["ret_cd_step"] == pytest.approx(reward)
    assert info["ret_cd_fill_rate_step"] >= 0.0
    assert info["ret_cd_availability_step"] >= 0.0
    assert "ret_cd_components" in info
    assert info["ret_cd_components"]["reward_mode"] == "ReT_cd_v1"


def test_shift_env_ret_cd_sigmoid_keeps_sigmoid_metadata_and_scale() -> None:
    env = MFSCGymEnvShifts(
        step_size_hours=24,
        max_steps=2,
        reward_mode="ReT_cd_sigmoid",
        observation_version="v4",
    )
    env.reset(seed=42)
    _, reward, _, _, info = env.step(np.zeros(5, dtype=np.float32))
    assert isinstance(reward, float)
    assert info["reward_mode"] == "ReT_cd_sigmoid"
    assert info["ret_cd_step"] == pytest.approx(reward)
    assert reward <= 0.5
    assert "sigmoid_bias_note" in info["ret_cd_components"]
    assert info["ret_cd_components"]["reward_mode"] == "ReT_cd_sigmoid"


def test_train_agent_defaults_to_shift_control_variant() -> None:
    parser = build_parser()
    args = parser.parse_args([])
    validate_args(parser, args)
    assert args.env_variant == "shift_control"
    assert args.reward_mode == "control_v1"
    assert args.observation_version == "v4"


def test_train_agent_build_env_instance_uses_shift_env() -> None:
    parser = build_parser()
    args = parser.parse_args([])
    validate_args(parser, args)
    env = build_env_instance(args)
    assert isinstance(env, MFSCGymEnvShifts)
    assert env.rt_delta == RET_SHIFT_COST_DELTA_DEFAULT
    assert env.reward_mode == "control_v1"
    assert env.observation_version == "v4"
    assert env.max_steps == 260


def test_train_agent_base_variant_gets_legacy_default_reward() -> None:
    parser = build_parser()
    args = parser.parse_args(["--env-variant", "base"])
    validate_args(parser, args)
    assert args.reward_mode == "rt_v0"


def test_train_agent_shift_delta_default_matches_config() -> None:
    parser = build_parser()
    args = parser.parse_args([])
    assert args.shift_delta == RET_SHIFT_COST_DELTA_DEFAULT
    assert args.w_bo == pytest.approx(4.0)
    assert args.w_cost == pytest.approx(0.02)
    assert args.w_disr == pytest.approx(0.0)


def test_train_agent_resolves_physical_horizon_for_faster_cadence() -> None:
    parser = build_parser()
    args = parser.parse_args(["--step-size-hours", "48", "--observation-version", "v5"])
    validate_args(parser, args)
    assert args.max_steps_per_episode == 910
    env = build_env_instance(args)
    assert env.max_steps == 910
    assert env.observation_version == "v5"


def test_train_agent_accepts_track_b_contract() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--step-size-hours",
            "48",
            "--observation-version",
            "v6",
            "--risk-level",
            "adaptive_benchmark_v1",
        ]
    )
    validate_args(parser, args)
    env = build_env_instance(args)
    assert env.observation_version == "v6"
    assert env.risk_level == "adaptive_benchmark_v1"


def test_train_agent_resolve_episode_max_steps_preserves_reference_horizon() -> None:
    assert resolve_episode_max_steps(168.0, None) == 260
    assert resolve_episode_max_steps(48.0, None) == 910
    assert resolve_episode_max_steps(24.0, None) == 1820
    assert resolve_episode_max_steps(24.0, 12) == 12


def test_train_agent_rejects_ret_thesis_on_base_env() -> None:
    parser = build_parser()
    args = parser.parse_args(["--env-variant", "base", "--reward-mode", "ReT_thesis"])
    with pytest.raises(SystemExit):
        validate_args(parser, args)


def test_train_agent_rejects_non_v1_observation_on_base_env() -> None:
    parser = build_parser()
    args = parser.parse_args(["--env-variant", "base", "--observation-version", "v2"])
    with pytest.raises(SystemExit):
        validate_args(parser, args)


def test_train_agent_accepts_ret_corrected_cost_on_shift_env() -> None:
    parser = build_parser()
    args = parser.parse_args(
        ["--env-variant", "shift_control", "--reward-mode", "ReT_corrected_cost"]
    )
    validate_args(parser, args)
    env = build_env_instance(args)
    assert isinstance(env, MFSCGymEnvShifts)
    assert env.reward_mode == "ReT_corrected_cost"


def test_train_agent_accepts_ret_seq_v1_and_kappa_on_shift_env() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--env-variant",
            "shift_control",
            "--reward-mode",
            "ReT_seq_v1",
            "--observation-version",
            "v4",
            "--ret-seq-kappa",
            "0.30",
        ]
    )
    validate_args(parser, args)
    env = build_env_instance(args)
    assert isinstance(env, MFSCGymEnvShifts)
    assert env.reward_mode == "ReT_seq_v1"
    assert env.observation_version == "v4"
    assert env.ret_seq_kappa == pytest.approx(0.30)


def test_train_agent_accepts_ret_cd_v1_on_shift_env() -> None:
    parser = build_parser()
    args = parser.parse_args(
        ["--env-variant", "shift_control", "--reward-mode", "ReT_cd_v1"]
    )
    validate_args(parser, args)
    env = build_env_instance(args)
    assert isinstance(env, MFSCGymEnvShifts)
    assert env.reward_mode == "ReT_cd_v1"


def test_train_agent_accepts_ret_cd_sigmoid_on_shift_env() -> None:
    parser = build_parser()
    args = parser.parse_args(
        ["--env-variant", "shift_control", "--reward-mode", "ReT_cd_sigmoid"]
    )
    validate_args(parser, args)
    env = build_env_instance(args)
    assert isinstance(env, MFSCGymEnvShifts)
    assert env.reward_mode == "ReT_cd_sigmoid"


def test_train_agent_accepts_ret_garrido2024_variants_on_shift_env() -> None:
    parser = build_parser()
    for reward_mode in (
        "ReT_garrido2024_raw",
        "ReT_garrido2024",
        "ReT_garrido2024_train",
    ):
        args = parser.parse_args(
            ["--env-variant", "shift_control", "--reward-mode", reward_mode]
        )
        validate_args(parser, args)
        env = build_env_instance(args)
        assert isinstance(env, MFSCGymEnvShifts)
        assert env.reward_mode == reward_mode


def test_train_agent_accepts_ret_unified_v1_and_calibration_path(tmp_path) -> None:
    calibration_path = tmp_path / "ret_unified.json"
    calibration_path.write_text(
        json.dumps(
            {
                "theta_sc": 0.76,
                "theta_bc": 0.75,
                "beta": 12.0,
                "kappa": 0.15,
                "w_fr": 0.60,
                "w_rc": 0.25,
                "w_ce": 0.15,
                "selection_rule": "unit_test",
                "source": "unit_test",
            }
        ),
        encoding="utf-8",
    )
    parser = build_parser()
    args = parser.parse_args(
        [
            "--env-variant",
            "shift_control",
            "--reward-mode",
            "ReT_unified_v1",
            "--ret-unified-calibration",
            str(calibration_path),
        ]
    )
    validate_args(parser, args)
    env = build_env_instance(args)
    assert env.reward_mode == "ReT_unified_v1"
    assert env.ret_unified_calibration_path == str(calibration_path.resolve())
    assert env.ret_unified_theta_sc == pytest.approx(0.76)
    assert env.ret_unified_theta_bc == pytest.approx(0.75)
    assert env.ret_unified_kappa == pytest.approx(0.15)


def test_ret_unified_explicit_overrides_beat_calibration_file(tmp_path) -> None:
    calibration_path = tmp_path / "ret_unified.json"
    calibration_path.write_text(
        json.dumps(
            {
                "theta_sc": 0.78,
                "theta_bc": 0.78,
                "beta": 12.0,
                "kappa": 0.20,
                "w_fr": 0.60,
                "w_rc": 0.25,
                "w_ce": 0.15,
                "selection_rule": "unit_test",
                "source": "unit_test",
            }
        ),
        encoding="utf-8",
    )

    env = MFSCGymEnvShifts(
        reward_mode="ReT_unified_v1",
        observation_version="v4",
        ret_unified_calibration_path=str(calibration_path),
        ret_unified_theta_sc=0.50,
        ret_unified_theta_bc=0.20,
        ret_unified_beta=20.0,
        ret_unified_kappa=0.60,
    )

    assert env.ret_unified_calibration_path == str(calibration_path.resolve())
    assert env.ret_unified_theta_sc == pytest.approx(0.50)
    assert env.ret_unified_theta_bc == pytest.approx(0.20)
    assert env.ret_unified_beta == pytest.approx(20.0)
    assert env.ret_unified_kappa == pytest.approx(0.60)


@pytest.mark.parametrize(
    "reward_mode",
    ["ReT_garrido2024_raw", "ReT_garrido2024", "ReT_garrido2024_train"],
)
def test_train_agent_passes_ret_g24_calibration_path(
    tmp_path, reward_mode: str
) -> None:
    calibration_path = tmp_path / "ret_g24.json"
    calibration_path.write_text(
        json.dumps(
            {
                "a_zeta": 0.1,
                "b_epsilon": 0.2,
                "c_phi": 0.3,
                "d_tau": 0.4,
                "n_kappa": 0.5,
                "kappa_ref": 123.0,
                "source": "unit_test",
            }
        ),
        encoding="utf-8",
    )
    parser = build_parser()
    args = parser.parse_args(
        [
            "--env-variant",
            "shift_control",
            "--reward-mode",
            reward_mode,
            "--ret-g24-calibration",
            str(calibration_path),
        ]
    )
    validate_args(parser, args)
    env = build_env_instance(args)
    assert env.ret_g24_calibration_path == str(calibration_path.resolve())
    assert env.ret_g24_a_zeta == pytest.approx(0.1)
    assert env.ret_g24_n_kappa == pytest.approx(0.5)
    assert env.ret_g24_kappa_ref == pytest.approx(123.0)


def test_ret_seq_v1_is_monotone_in_service_backlog_and_efficiency() -> None:
    env = MFSCGymEnvShifts(
        step_size_hours=24,
        max_steps=2,
        reward_mode="ReT_seq_v1",
        ret_seq_kappa=0.20,
    )
    env.reset(seed=42)
    env.sim.total_demanded = env._warmup_total_demanded + 100.0

    good_service = env._compute_ret_seq_v1(
        {
            "new_demanded": 100.0,
            "new_backorder_qty": 5.0,
            "pending_backorder_qty": 10.0,
        },
        shifts=2,
    )
    bad_service = env._compute_ret_seq_v1(
        {
            "new_demanded": 100.0,
            "new_backorder_qty": 40.0,
            "pending_backorder_qty": 10.0,
        },
        shifts=2,
    )
    bad_backlog = env._compute_ret_seq_v1(
        {
            "new_demanded": 100.0,
            "new_backorder_qty": 5.0,
            "pending_backorder_qty": 60.0,
        },
        shifts=2,
    )
    inefficient = env._compute_ret_seq_v1(
        {
            "new_demanded": 100.0,
            "new_backorder_qty": 5.0,
            "pending_backorder_qty": 10.0,
        },
        shifts=3,
    )

    assert good_service["ret_seq_step"] > bad_service["ret_seq_step"]
    assert good_service["ret_seq_step"] > bad_backlog["ret_seq_step"]
    assert good_service["ret_seq_step"] > inefficient["ret_seq_step"]


def test_ret_unified_v1_cost_gate_turns_off_when_service_is_poor() -> None:
    env = MFSCGymEnvShifts(
        step_size_hours=24,
        max_steps=2,
        reward_mode="ReT_unified_v1",
        observation_version="v4",
        ret_unified_theta_sc=0.78,
        ret_unified_theta_bc=0.78,
        ret_unified_beta=12.0,
        ret_unified_kappa=0.20,
    )
    env.reset(seed=42)
    env.sim.total_demanded = env._warmup_total_demanded + 100.0
    env.sim.pending_backorder_qty = 120.0

    s2 = env._compute_ret_unified_v1(
        {
            "new_demanded": 100.0,
            "new_backorder_qty": 70.0,
            "pending_backorder_qty": 120.0,
        },
        shifts=2,
    )
    s3 = env._compute_ret_unified_v1(
        {
            "new_demanded": 100.0,
            "new_backorder_qty": 70.0,
            "pending_backorder_qty": 120.0,
        },
        shifts=3,
    )

    assert s2["ret_unified_gate"] < 0.05
    assert s2["ret_unified_step"] == pytest.approx(s3["ret_unified_step"], rel=1e-3)


def test_ret_unified_v1_cost_gate_activates_when_service_is_good() -> None:
    env = MFSCGymEnvShifts(
        step_size_hours=24,
        max_steps=2,
        reward_mode="ReT_unified_v1",
        observation_version="v4",
        ret_unified_theta_sc=0.78,
        ret_unified_theta_bc=0.78,
        ret_unified_beta=12.0,
        ret_unified_kappa=0.20,
    )
    env.reset(seed=42)
    env.sim.total_demanded = env._warmup_total_demanded + 100.0
    env.sim.pending_backorder_qty = 1.0

    s2 = env._compute_ret_unified_v1(
        {
            "new_demanded": 100.0,
            "new_backorder_qty": 2.0,
            "pending_backorder_qty": 1.0,
        },
        shifts=2,
    )
    s3 = env._compute_ret_unified_v1(
        {
            "new_demanded": 100.0,
            "new_backorder_qty": 2.0,
            "pending_backorder_qty": 1.0,
        },
        shifts=3,
    )

    assert s2["ret_unified_gate"] > 0.5
    assert s2["ret_unified_step"] > s3["ret_unified_step"]
    assert s2["ret_unified_fr"] > 0.0
    assert s2["ret_unified_rc"] > 0.0
