#!/usr/bin/env python3
"""Smoke-check David's thesis-faithful DKANA environment contract."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.external_env_interface import (  # noqa: E402
    get_dkana_thesis_faithful_env_spec,
    make_dkana_thesis_faithful_env,
)


def thesis_action(period_index: int = 2, shift_index: int = 1) -> np.ndarray:
    """Return a one-hot 18D action: Op3/Op5/Op9 inventory period plus S."""
    action = np.zeros(18, dtype=np.float32)
    action[period_index] = 1.0
    action[5 + period_index] = 1.0
    action[10 + period_index] = 1.0
    action[15 + shift_index] = 1.0
    return action


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify the thesis-faithful 18D action DKANA env."
    )
    parser.add_argument(
        "--observation-mode",
        default="decision_reward",
        choices=[
            "decision_reward",
            "env_reward",
            "env_state_reward",
            "env_sdm_history_reward",
        ],
        help=(
            "decision_reward is the original 19D handoff. "
            "env_sdm_history_reward keeps 18D actions but exposes richer history."
        ),
    )
    parser.add_argument("--reward-mode", default="ReT_seq_v1")
    parser.add_argument("--risk-level", default="increased")
    parser.add_argument("--observation-version", default="v5")
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stochastic-pt", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = get_dkana_thesis_faithful_env_spec(
        reward_mode=args.reward_mode,
        observation_version=args.observation_version,
        observation_mode=args.observation_mode,
    )
    env = make_dkana_thesis_faithful_env(
        reward_mode=args.reward_mode,
        risk_level=args.risk_level,
        observation_version=args.observation_version,
        observation_mode=args.observation_mode,
        inventory_period_mode="thesis_strict",
        max_steps=args.max_steps,
        stochastic_pt=args.stochastic_pt,
    )

    obs, info = env.reset(seed=args.seed)
    action = thesis_action(period_index=2, shift_index=1)
    next_obs, reward, terminated, truncated, step_info = env.step(action)

    print("DKANA thesis-faithful smoke OK")
    print(f"  env_variant: {spec.env_variant}")
    print(f"  action_contract: {step_info['action_contract']}")
    print(f"  observation_contract: {info['observation_contract']}")
    print(f"  action_shape: {env.action_space.shape}")
    print(f"  observation_shape: {env.observation_space.shape}")
    print(f"  reset_obs_shape: {obs.shape}")
    print(f"  next_obs_shape: {next_obs.shape}")
    print(f"  reward: {reward:.6f}")
    print(f"  terminated: {terminated}, truncated: {truncated}")
    print(
        f"  inventory_period_hours: {step_info['thesis_decision']['inventory_period_hours']}"
    )
    print(f"  assembly_shifts: {step_info['thesis_decision']['assembly_shifts']}")
    print(f"  action_fields: {len(spec.action_fields)}")
    print(f"  observation_fields: {len(spec.observation_fields)}")
    env.close()


if __name__ == "__main__":
    main()
