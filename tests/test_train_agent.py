from __future__ import annotations

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


def test_train_agent_defaults_to_shift_control_variant() -> None:
    parser = build_parser()
    args = parser.parse_args([])
    validate_args(parser, args)
    assert args.env_variant == "shift_control"
    assert args.reward_mode == "ReT_thesis"


def test_train_agent_build_env_instance_uses_shift_env() -> None:
    parser = build_parser()
    args = parser.parse_args([])
    validate_args(parser, args)
    env = build_env_instance(args)
    assert isinstance(env, MFSCGymEnvShifts)
    assert env.rt_delta == RET_SHIFT_COST_DELTA_DEFAULT


def test_train_agent_base_variant_gets_legacy_default_reward() -> None:
    parser = build_parser()
    args = parser.parse_args(["--env-variant", "base"])
    validate_args(parser, args)
    assert args.reward_mode == "rt_v0"


def test_train_agent_shift_delta_default_matches_config() -> None:
    parser = build_parser()
    args = parser.parse_args([])
    assert args.shift_delta == RET_SHIFT_COST_DELTA_DEFAULT


def test_train_agent_rejects_ret_thesis_on_base_env() -> None:
    parser = build_parser()
    args = parser.parse_args(["--env-variant", "base", "--reward-mode", "ReT_thesis"])
    with pytest.raises(SystemExit):
        validate_args(parser, args)
