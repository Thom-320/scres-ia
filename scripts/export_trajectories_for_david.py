#!/usr/bin/env python3
"""
Export MFSC trajectories for external model training (e.g., DKANA).

The exported bundle freezes the paper-facing online contract around
`ReT_unified_v1` + `observation_version=v4`, while still supporting historical
reward lanes, exact Garrido static baselines, and the research-only `v5`
cycle-precursor observation lane.

Outputs
-------
- observations.npy
- actions.npy
- direct_action_context.npy
- rewards.npy
- episode_ids.npy
- constraint_context.npy
- state_constraint_context.npy
- reward_terms.npy
- env_spec.json
- metadata.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from gymnasium.wrappers import FrameStackObservation
import numpy as np

try:
    from sb3_contrib import RecurrentPPO
except ImportError:  # pragma: no cover - exercised via runtime guard.
    RecurrentPPO = None
from stable_baselines3 import PPO, SAC

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.benchmark_control_reward import (
    FIXED_POLICY_ACTIONS,
    HeuristicCycleGuard,
    HeuristicDisruptionAware,
    HeuristicHysteresis,
    HeuristicTuned,
)
from supply_chain.config import BENCHMARK_EPISODE_HORIZON_HOURS
from supply_chain.external_env_interface import (
    ACTION_FIELDS,
    CONTROL_CONTEXT_FIELDS,
    REWARD_TERM_FIELDS,
    STATE_CONSTRAINT_FIELDS,
    build_reward_term_vector,
    build_shift_control_constraint_vector,
    build_shift_control_state_constraint_vector,
    get_shift_control_constraint_context,
    get_shift_control_env_spec,
    get_track_b_env_spec,
    make_shift_control_env,
    make_track_b_env,
    spec_to_dict,
)

STATIC_POLICY_CHOICES: tuple[str, ...] = (
    "static_s1",
    "static_s2",
    "static_s3",
    "garrido_cf_s1",
    "garrido_cf_s2",
    "garrido_cf_s3",
)
HEURISTIC_POLICY_CHOICES: tuple[str, ...] = (
    "heuristic_hysteresis",
    "heuristic_disruption",
    "heuristic_tuned",
    "heuristic_cycle_guard",
)
LEARNED_POLICY_CHOICES: tuple[str, ...] = ("ppo", "sac", "recurrent_ppo")
POLICY_CHOICES: tuple[str, ...] = (
    "random",
    *STATIC_POLICY_CHOICES,
    *HEURISTIC_POLICY_CHOICES,
    *LEARNED_POLICY_CHOICES,
)
REWARD_MODE_CHOICES: tuple[str, ...] = (
    "ReT_thesis",
    "control_v1",
    "ReT_seq_v1",
    "ReT_garrido2024_raw",
    "ReT_garrido2024",
    "ReT_garrido2024_train",
    "ReT_unified_v1",
    "ReT_cd_v1",
    "ReT_cd_sigmoid",
)
DIRECT_ACTION_CONTEXT_FIELDS: tuple[str, ...] = (
    "assembly_shifts",
    "op3_q",
    "op3_rop",
    "op9_q_min",
    "op9_q_max",
    "op9_rop",
    "batch_size",
)
SHIFT_SIGNAL_BY_COUNT: dict[int, float] = {1: -1.0, 2: 0.0, 3: 1.0}


def resolve_episode_max_steps(
    step_size_hours: float,
    explicit_max_steps: int | None,
) -> int:
    """Preserve the historical physical horizon when cadence changes."""
    if explicit_max_steps is not None:
        return int(explicit_max_steps)
    if step_size_hours <= 0:
        raise ValueError("step_size_hours must be > 0")
    return max(1, int(round(BENCHMARK_EPISODE_HORIZON_HOURS / step_size_hours)))


def reward_formula_description(reward_mode: str) -> str:
    if reward_mode == "control_v1":
        return (
            "reward_total = -(w_bo * service_loss_step + "
            "w_cost * shift_cost_step + w_disr * disruption_fraction_step)"
        )
    if reward_mode == "ReT_unified_v1":
        return (
            "ret_unified_step = FR_t^0.60 * RC_t^0.25 * CE_t^(0.15 * gate_t), "
            "with gate_t = sigmoid(beta*(FR_t-theta_sc)) * "
            "sigmoid(beta*(RC_t-theta_bc))"
        )
    if reward_mode == "ReT_seq_v1":
        return "ret_seq_step = SC_t^0.60 * BC_t^0.25 * AE_t^0.15"
    if reward_mode in (
        "ReT_garrido2024_raw",
        "ReT_garrido2024",
        "ReT_garrido2024_train",
    ):
        return "See env_experimental_shifts.py for Garrido-2024 Cobb-Douglas reward details."
    if reward_mode in ("ReT_cd_v1", "ReT_cd_sigmoid"):
        return "See env_experimental_shifts.py for continuous Cobb-Douglas bridge reward details."
    return "See env_experimental_shifts.py for reward details."


def direct_action_context_from_payload(
    action_payload: np.ndarray | dict[str, float | int],
) -> np.ndarray:
    if isinstance(action_payload, dict):
        return np.array(
            [
                float(action_payload.get("assembly_shifts", np.nan)),
                float(action_payload.get("op3_q", np.nan)),
                float(action_payload.get("op3_rop", np.nan)),
                float(action_payload.get("op9_q_min", np.nan)),
                float(action_payload.get("op9_q_max", np.nan)),
                float(action_payload.get("op9_rop", np.nan)),
                float(action_payload.get("batch_size", np.nan)),
            ],
            dtype=np.float32,
        )
    return np.full(len(DIRECT_ACTION_CONTEXT_FIELDS), np.nan, dtype=np.float32)


def action_vector_from_payload(
    action_payload: np.ndarray | dict[str, float | int],
    *,
    constraint_context: dict[str, Any],
) -> np.ndarray:
    if not isinstance(action_payload, dict):
        return np.asarray(action_payload, dtype=np.float32).copy()

    base = constraint_context["base_control_parameters"]
    op3_ratio = float(action_payload["op3_q"]) / float(base["op3_q"])
    op3_rop_ratio = float(action_payload["op3_rop"]) / float(base["op3_rop"])
    op9_min_ratio = float(action_payload["op9_q_min"]) / float(base["op9_q_min"])
    op9_max_ratio = float(action_payload["op9_q_max"]) / float(base["op9_q_max"])
    op9_rop_ratio = float(action_payload["op9_rop"]) / float(base["op9_rop"])
    op9_ratio = 0.5 * (op9_min_ratio + op9_max_ratio)

    def inverse_inventory_signal(multiplier: float) -> float:
        return float(np.clip((multiplier - 1.25) / 0.75, -1.0, 1.0))

    shift_signal = SHIFT_SIGNAL_BY_COUNT[int(action_payload["assembly_shifts"])]
    return np.array(
        [
            inverse_inventory_signal(op3_ratio),
            inverse_inventory_signal(op9_ratio),
            inverse_inventory_signal(op3_rop_ratio),
            inverse_inventory_signal(op9_rop_ratio),
            float(shift_signal),
        ],
        dtype=np.float32,
    )


class PolicyAdapter:
    """Minimal predict/reset interface for export-time policy execution."""

    def reset_episode(self) -> None:
        return None

    def act(
        self, obs: np.ndarray, info: dict[str, Any]
    ) -> np.ndarray | dict[str, float | int]:
        raise NotImplementedError


class RandomPolicy(PolicyAdapter):
    def __init__(self, env: Any) -> None:
        self._env = env

    def act(self, obs: np.ndarray, info: dict[str, Any]) -> np.ndarray:
        return np.asarray(self._env.action_space.sample(), dtype=np.float32)


class FixedPolicy(PolicyAdapter):
    def __init__(self, action_payload: np.ndarray | dict[str, float | int]) -> None:
        self._action_payload = (
            dict(action_payload) if isinstance(action_payload, dict) else action_payload
        )

    def act(
        self, obs: np.ndarray, info: dict[str, Any]
    ) -> np.ndarray | dict[str, float | int]:
        if isinstance(self._action_payload, dict):
            return dict(self._action_payload)
        return np.asarray(self._action_payload, dtype=np.float32)


class HeuristicPolicy(PolicyAdapter):
    def __init__(self, policy_impl: Any) -> None:
        self._policy_impl = policy_impl

    def reset_episode(self) -> None:
        if hasattr(self._policy_impl, "reset"):
            self._policy_impl.reset()

    def act(self, obs: np.ndarray, info: dict[str, Any]) -> np.ndarray:
        if hasattr(self._policy_impl, "predict"):
            prediction = self._policy_impl.predict(np.asarray(obs, dtype=np.float32))
            if isinstance(prediction, tuple):
                prediction = prediction[0]
            return np.asarray(prediction, dtype=np.float32)
        prediction = self._policy_impl(np.asarray(obs, dtype=np.float32), info)
        return np.asarray(prediction, dtype=np.float32)


class SB3Policy(PolicyAdapter):
    def __init__(self, policy_name: str, model_path: Path) -> None:
        if policy_name == "ppo":
            self._model = PPO.load(str(model_path))
            self._recurrent = False
        elif policy_name == "sac":
            self._model = SAC.load(str(model_path))
            self._recurrent = False
        elif policy_name == "recurrent_ppo":
            if RecurrentPPO is None:
                raise RuntimeError(
                    "recurrent_ppo export requested but sb3_contrib is not installed."
                )
            self._model = RecurrentPPO.load(str(model_path))
            self._recurrent = True
        else:
            raise ValueError(f"Unsupported learned policy {policy_name!r}.")
        self._lstm_states: Any = None
        self._episode_starts: np.ndarray | None = None

    def reset_episode(self) -> None:
        self._lstm_states = None
        self._episode_starts = np.ones((1,), dtype=bool)

    def act(self, obs: np.ndarray, info: dict[str, Any]) -> np.ndarray:
        obs_array = np.asarray(obs, dtype=np.float32)
        if self._recurrent:
            action, self._lstm_states = self._model.predict(
                obs_array,
                state=self._lstm_states,
                episode_start=self._episode_starts,
                deterministic=True,
            )
            self._episode_starts = np.zeros((1,), dtype=bool)
            return np.asarray(action, dtype=np.float32)
        action, _ = self._model.predict(obs_array, deterministic=True)
        return np.asarray(action, dtype=np.float32)


def build_policy_adapter(
    policy_name: str, env: Any, args: argparse.Namespace
) -> PolicyAdapter:
    if policy_name == "random":
        return RandomPolicy(env)
    if policy_name in STATIC_POLICY_CHOICES:
        return FixedPolicy(FIXED_POLICY_ACTIONS[policy_name])
    if policy_name == "heuristic_hysteresis":
        return HeuristicPolicy(HeuristicHysteresis())
    if policy_name == "heuristic_disruption":
        return HeuristicPolicy(HeuristicDisruptionAware())
    if policy_name == "heuristic_tuned":
        return HeuristicPolicy(HeuristicTuned())
    if policy_name == "heuristic_cycle_guard":
        return HeuristicPolicy(HeuristicCycleGuard())
    if policy_name in LEARNED_POLICY_CHOICES:
        if args.model_path is None:
            raise ValueError(f"--model-path is required for policy={policy_name}.")
        return SB3Policy(policy_name, args.model_path)
    raise ValueError(f"Unknown policy {policy_name!r}.")


def build_env(args: argparse.Namespace) -> Any:
    action_contract = getattr(args, "action_contract", None)
    if action_contract == "track_b_v1":
        env = make_track_b_env(
            risk_level=args.risk_level,
            reward_mode=args.reward_mode,
            step_size_hours=args.step_size_hours,
            max_steps=resolve_episode_max_steps(args.step_size_hours, args.max_steps),
            stochastic_pt=args.stochastic_pt,
        )
    else:
        env = make_shift_control_env(
            risk_level=args.risk_level,
            reward_mode=args.reward_mode,
            observation_version=args.observation_version,
            step_size_hours=args.step_size_hours,
            max_steps=resolve_episode_max_steps(args.step_size_hours, args.max_steps),
            stochastic_pt=args.stochastic_pt,
            ret_unified_calibration_path=(
                str(args.ret_unified_calibration)
                if args.ret_unified_calibration is not None
                else None
            ),
            ret_g24_calibration_path=(
                str(args.ret_g24_calibration)
                if args.ret_g24_calibration is not None
                else None
            ),
        )
    if args.frame_stack > 1:
        return FrameStackObservation(env, stack_size=args.frame_stack)
    return env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export MFSC trajectories for external models."
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/data_export"))
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument(
        "--risk-level", default="current", choices=["current", "increased", "severe", "adaptive_benchmark_v1", "adaptive_benchmark_v2"]
    )
    parser.add_argument(
        "--reward-mode",
        default="ReT_unified_v1",
        choices=list(REWARD_MODE_CHOICES),
        help="Reward lane used during collection.",
    )
    parser.add_argument(
        "--observation-version",
        default="v4",
        choices=["v1", "v2", "v3", "v4", "v5", "v6", "v7"],
        help="Observation contract used during collection.",
    )
    parser.add_argument(
        "--policy",
        default="random",
        choices=list(POLICY_CHOICES),
        help="Policy used to collect trajectories.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Required for learned policies (ppo, sac, recurrent_ppo).",
    )
    parser.add_argument(
        "--frame-stack",
        type=int,
        default=1,
        help="Optional observation frame stacking. Default is 1.",
    )
    parser.add_argument("--step-size-hours", type=float, default=168.0)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help=(
            "Episode length in control steps. Defaults to the historical "
            "260x168h physical horizon rescaled to the requested cadence."
        ),
    )
    parser.add_argument("--stochastic-pt", action="store_true")
    parser.add_argument(
        "--action-contract",
        default=None,
        choices=["track_b_v1"],
        help="Action contract. If set to track_b_v1, uses 7D actions with Op10/Op12 control.",
    )
    parser.add_argument(
        "--ret-unified-calibration",
        type=Path,
        default=None,
        help="Optional ReT_unified_v1 calibration JSON.",
    )
    parser.add_argument(
        "--ret-g24-calibration",
        type=Path,
        default=None,
        help="Optional ReT_garrido2024 calibration JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.max_steps = resolve_episode_max_steps(args.step_size_hours, args.max_steps)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_obs: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    all_direct_actions: list[np.ndarray] = []
    all_rewards: list[float] = []
    all_episode_ids: list[int] = []
    all_constraint_context: list[np.ndarray] = []
    all_state_constraint_context: list[np.ndarray] = []
    all_reward_terms: list[np.ndarray] = []
    episode_lengths: list[int] = []

    constraint_context = get_shift_control_constraint_context()
    constraint_vector = build_shift_control_constraint_vector(constraint_context)

    for ep in range(args.episodes):
        env = build_env(args)
        base_env = env.unwrapped
        policy = build_policy_adapter(args.policy, env, args)
        policy.reset_episode()
        obs, info = env.reset(seed=args.seed_start + ep)

        terminated = False
        truncated = False
        steps = 0

        while not (terminated or truncated):
            state_constraint_context = base_env.get_state_constraint_context()
            action_payload = policy.act(np.asarray(obs, dtype=np.float32), info)
            direct_action_context = direct_action_context_from_payload(action_payload)
            action_vector = action_vector_from_payload(
                action_payload,
                constraint_context=constraint_context,
            )
            env_action: np.ndarray | dict[str, float | int]
            if isinstance(action_payload, dict):
                env_action = dict(action_payload)
            else:
                env_action = np.asarray(action_payload, dtype=np.float32)

            obs_array = np.asarray(obs, dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(env_action)

            all_obs.append(obs_array.copy())
            all_actions.append(action_vector.copy())
            all_direct_actions.append(direct_action_context.copy())
            all_rewards.append(float(reward))
            all_episode_ids.append(ep)
            all_constraint_context.append(constraint_vector.copy())
            all_state_constraint_context.append(
                build_shift_control_state_constraint_vector(state_constraint_context)
            )
            all_reward_terms.append(build_reward_term_vector(info, float(reward)))
            steps += 1

        episode_lengths.append(steps)
        env.close()

        if (ep + 1) % 10 == 0 or ep == args.episodes - 1:
            print(f"  Episode {ep + 1}/{args.episodes} done (T={steps} steps)")

    observations = np.array(all_obs, dtype=np.float32)
    actions = np.array(all_actions, dtype=np.float32)
    direct_actions = np.array(all_direct_actions, dtype=np.float32)
    rewards = np.array(all_rewards, dtype=np.float32)
    episode_ids = np.array(all_episode_ids, dtype=np.int32)
    constraint_context_array = np.array(all_constraint_context, dtype=np.float32)
    state_constraint_context = np.array(all_state_constraint_context, dtype=np.float32)
    reward_terms = np.array(all_reward_terms, dtype=np.float32)

    np.save(args.output_dir / "observations.npy", observations)
    np.save(args.output_dir / "actions.npy", actions)
    np.save(args.output_dir / "direct_action_context.npy", direct_actions)
    np.save(args.output_dir / "rewards.npy", rewards)
    np.save(args.output_dir / "episode_ids.npy", episode_ids)
    np.save(args.output_dir / "constraint_context.npy", constraint_context_array)
    np.save(args.output_dir / "state_constraint_context.npy", state_constraint_context)
    np.save(args.output_dir / "reward_terms.npy", reward_terms)

    spec = get_shift_control_env_spec(
        reward_mode=args.reward_mode,
        observation_version=args.observation_version,
        step_size_hours=args.step_size_hours,
    )
    with (args.output_dir / "env_spec.json").open("w", encoding="utf-8") as file_obj:
        json.dump(spec_to_dict(spec), file_obj, indent=2)
    with (args.output_dir / "constraint_context.json").open(
        "w", encoding="utf-8"
    ) as file_obj:
        json.dump(constraint_context, file_obj, indent=2)
    with (args.output_dir / "constraint_context_fields.json").open(
        "w", encoding="utf-8"
    ) as file_obj:
        json.dump({"fields": list(CONTROL_CONTEXT_FIELDS)}, file_obj, indent=2)
    with (args.output_dir / "state_constraint_fields.json").open(
        "w", encoding="utf-8"
    ) as file_obj:
        json.dump({"fields": list(STATE_CONSTRAINT_FIELDS)}, file_obj, indent=2)
    with (args.output_dir / "reward_terms_fields.json").open(
        "w", encoding="utf-8"
    ) as file_obj:
        json.dump(
            {
                "reward_mode": args.reward_mode,
                "fields": list(REWARD_TERM_FIELDS),
                "formula": reward_formula_description(args.reward_mode),
            },
            file_obj,
            indent=2,
        )
    with (args.output_dir / "action_fields.json").open(
        "w", encoding="utf-8"
    ) as file_obj:
        json.dump({"fields": list(ACTION_FIELDS)}, file_obj, indent=2)
    with (args.output_dir / "direct_action_context_fields.json").open(
        "w", encoding="utf-8"
    ) as file_obj:
        json.dump({"fields": list(DIRECT_ACTION_CONTEXT_FIELDS)}, file_obj, indent=2)

    metadata = {
        "episodes": args.episodes,
        "total_steps": int(rewards.shape[0]),
        "episode_lengths": episode_lengths,
        "risk_level": args.risk_level,
        "reward_mode": args.reward_mode,
        "observation_version": args.observation_version,
        "frame_stack": int(args.frame_stack),
        "policy": args.policy,
        "model_path": str(args.model_path) if args.model_path is not None else None,
        "obs_shape": list(observations.shape),
        "action_shape": list(actions.shape),
        "direct_action_context_shape": list(direct_actions.shape),
        "constraint_context_shape": list(constraint_context_array.shape),
        "state_constraint_context_shape": list(state_constraint_context.shape),
        "reward_terms_shape": list(reward_terms.shape),
        "uses_direct_des_actions": bool(np.isfinite(direct_actions).any()),
        "note": (
            "The export preserves the online MFSC contract plus reward decomposition. "
            "actions.npy always follows the 5D RL action schema. "
            "direct_action_context.npy preserves exact DES control settings for "
            "policies that bypass the 5D action map (e.g., garrido_cf_* baselines)."
        ),
    }
    with (args.output_dir / "metadata.json").open("w", encoding="utf-8") as file_obj:
        json.dump(metadata, file_obj, indent=2)

    print(f"\nExported {rewards.shape[0]:,} steps from {args.episodes} episodes")
    print(f"  observations.npy             shape={observations.shape}")
    print(f"  actions.npy                  shape={actions.shape}")
    print(f"  direct_action_context.npy    shape={direct_actions.shape}")
    print(f"  rewards.npy                  shape={rewards.shape}")
    print(f"  episode_ids.npy              shape={episode_ids.shape}")
    print("  constraint_context.npy       " f"shape={constraint_context_array.shape}")
    print("  state_constraint_context.npy " f"shape={state_constraint_context.shape}")
    print(f"  reward_terms.npy             shape={reward_terms.shape}")
    print("  env_spec.json")
    print("  metadata.json")
    print(f"\nSaved to: {args.output_dir}")


if __name__ == "__main__":
    main()
