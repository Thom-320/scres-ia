from __future__ import annotations

import json
import numpy as np
import pytest

from supply_chain.config import RET_SHIFT_COST_DELTA_DEFAULT
from supply_chain.env_experimental_shifts import MFSCGymEnvShifts
from train_agent import build_env_instance, build_parser, validate_args


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
        "ReT_garrido2024_raw"
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
    assert args.reward_mode == "ReT_seq_v1"
    assert args.ret_seq_kappa == pytest.approx(0.20)
    assert args.observation_version == "v1"


def test_train_agent_build_env_instance_uses_shift_env() -> None:
    parser = build_parser()
    args = parser.parse_args([])
    validate_args(parser, args)
    env = build_env_instance(args)
    assert isinstance(env, MFSCGymEnvShifts)
    assert env.rt_delta == RET_SHIFT_COST_DELTA_DEFAULT
    assert env.reward_mode == "ReT_seq_v1"
    assert env.ret_seq_kappa == pytest.approx(0.20)
    assert env.observation_version == "v1"


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
    for reward_mode in ("ReT_garrido2024_raw", "ReT_garrido2024"):
        args = parser.parse_args(
            ["--env-variant", "shift_control", "--reward-mode", reward_mode]
        )
        validate_args(parser, args)
        env = build_env_instance(args)
        assert isinstance(env, MFSCGymEnvShifts)
        assert env.reward_mode == reward_mode


def test_train_agent_passes_ret_g24_calibration_path(tmp_path) -> None:
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
            "ReT_garrido2024_raw",
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
