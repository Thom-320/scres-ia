#!/usr/bin/env python3
"""Fair same-run Track B architecture bakeoff.

Compares:
- PPO+MLP with one-step observation
- PPO+MLP with the same stacked state-action history as DMLPA
- PPO+DMLPA positional transformer-over-history

This is collaborator-facing architecture evidence. The paper headline remains
the canonical PPO+MLP result unless an architecture beats it under the same
protocol and primary Excel/order ReT metric.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.dmlpa_extractor import DMLPA
import scripts.run_track_b_smoke as smoke
from supply_chain.episode_metrics import compute_episode_metrics
from supply_chain.external_env_interface import get_episode_terminal_metrics, make_track_b_env


POLICIES = ("ppo_mlp", "ppo_mlp_history", "ppo_dmlpa_positional")


class PreviousActionObservationWrapper(gym.Wrapper):
    """Append the previous continuous action to each observation."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise TypeError("PreviousActionObservationWrapper requires Box observations.")
        if not isinstance(env.action_space, gym.spaces.Box):
            raise TypeError("PreviousActionObservationWrapper requires Box actions.")
        self._act_dim = int(np.prod(env.action_space.shape))
        low = np.concatenate(
            [env.observation_space.low.reshape(-1), np.full(self._act_dim, -1.0)]
        ).astype(np.float32)
        high = np.concatenate(
            [env.observation_space.high.reshape(-1), np.full(self._act_dim, 1.0)]
        ).astype(np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self._prev_action = np.zeros(self._act_dim, dtype=np.float32)

    def _augment(self, obs: np.ndarray) -> np.ndarray:
        return np.concatenate(
            [np.asarray(obs, dtype=np.float32).reshape(-1), self._prev_action]
        ).astype(np.float32)

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        self._prev_action = np.zeros(self._act_dim, dtype=np.float32)
        return self._augment(obs), info

    def step(self, action: np.ndarray):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._prev_action = np.asarray(action, dtype=np.float32).reshape(-1)[
            : self._act_dim
        ]
        return self._augment(obs), reward, terminated, truncated, info


def build_parser() -> argparse.ArgumentParser:
    parser = smoke.build_parser()
    parser.description = "Fair PPO+MLP vs PPO+MLP-history vs PPO+DMLPA Track B bakeoff."
    parser.set_defaults(
        output_dir=Path("outputs/experiments/track_b_architecture_fair_bakeoff_2026-07-03/full8d_v9_history_5seed_60k_h104"),
        seeds=[1, 2, 3, 4, 5],
        train_timesteps=60_000,
        eval_episodes=12,
        reward_mode="control_v1",
        risk_level="adaptive_benchmark_v2",
        observation_version="v9",
        max_steps=104,
        n_envs=4,
        n_steps=256,
        batch_size=256,
        learning_rate=3e-4,
        n_epochs=10,
        algo="ppo",
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        default=list(POLICIES),
        choices=POLICIES,
        help="Architecture arms to run.",
    )
    parser.add_argument("--history-factor", type=int, default=12)
    parser.add_argument("--dmlpa-features-dim", type=int, default=120)
    parser.add_argument("--dmlpa-nhead", type=int, default=12)
    parser.add_argument("--dmlpa-layers", type=int, default=4)
    parser.add_argument(
        "--skip-static",
        action="store_true",
        help="Skip static/heuristic baselines and run only architecture arms.",
    )
    return parser


def uses_history(policy: str, factor: int) -> bool:
    return factor > 1 and policy in {"ppo_mlp_history", "ppo_dmlpa_positional"}


def make_training_env(args: argparse.Namespace, seed: int, *, include_prev_action: bool):
    env_kwargs = smoke.build_env_kwargs(args)

    def _init():
        env = make_track_b_env(**env_kwargs)
        if include_prev_action:
            env = PreviousActionObservationWrapper(env)
        env.reset(seed=seed)
        return smoke.Monitor(env)

    return _init


def policy_kwargs_for(policy: str, args: argparse.Namespace) -> dict[str, Any]:
    if policy != "ppo_dmlpa_positional":
        return {"net_arch": {"pi": [64, 64], "vf": [64, 64]}}
    return {
        "features_extractor_class": DMLPA,
        "features_extractor_kwargs": {
            "factor": int(args.history_factor),
            "features_dim": int(args.dmlpa_features_dim),
            "nhead": int(args.dmlpa_nhead),
            "num_layers": int(args.dmlpa_layers),
        },
        "net_arch": {"pi": [], "vf": []},
    }


def train_policy(
    args: argparse.Namespace, *, policy: str, seed: int, run_dir: Path
) -> tuple[Any, VecNormalize]:
    include_prev_action = uses_history(policy, int(args.history_factor))
    vec_env = DummyVecEnv(
        [
            make_training_env(
                args, seed * 10_000 + i, include_prev_action=include_prev_action
            )
            for i in range(int(args.n_envs))
        ]
    )
    vec_norm = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    train_env = (
        VecFrameStack(vec_norm, n_stack=int(args.history_factor))
        if uses_history(policy, int(args.history_factor))
        else vec_norm
    )
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        policy_kwargs=policy_kwargs_for(policy, args),
        seed=seed,
        verbose=0,
        device="cpu",
    )
    model.learn(total_timesteps=int(args.train_timesteps))
    model.save(run_dir / f"{policy}_model.zip")
    vec_norm.save(str(run_dir / "vec_normalize.pkl"))
    return model, vec_norm


def stack_reset(obs_norm: np.ndarray, factor: int) -> np.ndarray:
    base = np.asarray(obs_norm, dtype=np.float32).reshape(-1)
    stacked = np.zeros((base.shape[0] * factor,), dtype=np.float32)
    stacked[-base.shape[0] :] = base
    return stacked


def stack_step(stack: np.ndarray, obs_norm: np.ndarray) -> np.ndarray:
    base = np.asarray(obs_norm, dtype=np.float32).reshape(-1)
    next_stack = np.roll(stack, -base.shape[0]).astype(np.float32, copy=False)
    next_stack[-base.shape[0] :] = base
    return next_stack


def evaluate_policy(
    args: argparse.Namespace,
    *,
    policy: str,
    seed: int,
    model: Any,
    vec_norm: VecNormalize,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    env_kwargs = smoke.build_env_kwargs(args)
    vec_norm.training = False
    vec_norm.norm_reward = False
    history = uses_history(policy, int(args.history_factor))
    include_prev_action = history

    for episode_idx in range(args.eval_episodes):
        eval_seed = seed + smoke.EVAL_EPISODE_SEED_OFFSET + episode_idx
        env = make_track_b_env(**env_kwargs)
        if include_prev_action:
            env = PreviousActionObservationWrapper(env)
        obs, info = env.reset(seed=eval_seed)
        obs_norm = vec_norm.normalize_obs(np.asarray(obs, dtype=np.float32)[None, :])[0]
        model_obs = (
            stack_reset(obs_norm, int(args.history_factor)) if history else obs_norm
        )
        terminated = False
        truncated = False
        reward_total = 0.0
        demanded_total = 0.0
        backorder_qty_total = 0.0
        steps = 0
        shift_counts = {1: 0, 2: 0, 3: 0}
        op10_multipliers: list[float] = []
        op12_multipliers: list[float] = []
        cd_totals = smoke.init_cd_totals()
        final_info = info

        while not (terminated or truncated):
            action, _ = model.predict(model_obs[None, :], deterministic=True)
            obs, reward, terminated, truncated, final_info = env.step(
                np.asarray(action[0], dtype=np.float32)
            )
            reward_total += float(reward)
            smoke.update_cd_totals(cd_totals, final_info)
            demanded_total += float(final_info.get("new_demanded", 0.0))
            backorder_qty_total += float(final_info.get("new_backorder_qty", 0.0))
            shift_counts[int(final_info.get("shifts_active", 1))] += 1
            op10_mult, op12_mult = smoke.extract_downstream_multipliers(final_info)
            op10_multipliers.append(op10_mult)
            op12_multipliers.append(op12_mult)
            steps += 1
            if not (terminated or truncated):
                obs_norm = vec_norm.normalize_obs(
                    np.asarray(obs, dtype=np.float32)[None, :]
                )[0]
                model_obs = (
                    stack_step(model_obs, obs_norm) if history else obs_norm
                )

        rows.append(
            smoke._finalize_episode_row(
                policy=policy,
                seed=seed,
                episode=episode_idx + 1,
                eval_seed=eval_seed,
                steps=steps,
                reward_total=reward_total,
                demanded_total=demanded_total,
                backorder_qty_total=backorder_qty_total,
                shift_counts=shift_counts,
                op10_multipliers=op10_multipliers,
                op12_multipliers=op12_multipliers,
                track_b_context=final_info["state_constraint_context"][
                    "track_b_context"
                ],
                terminal_metrics=get_episode_terminal_metrics(env),
                final_info=final_info,
                cd_totals=cd_totals,
                full_episode_metrics=compute_episode_metrics(env.unwrapped.sim),
            )
        )
        env.close()
    return rows


def paired_architecture_rows(episode_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    import pandas as pd

    df = pd.DataFrame(episode_rows)
    rows: list[dict[str, Any]] = []
    refs = [p for p in ("ppo_mlp_history", "ppo_mlp") if p in set(df["policy"])]
    for ref in refs:
        ref_df = df[df["policy"] == ref][["seed", "episode", "eval_seed", "order_level_ret_mean", "order_ret_excel"]]
        for candidate in sorted(set(df["policy"])):
            if candidate == ref or candidate.startswith("s") or candidate.startswith("heur_"):
                continue
            cand_df = df[df["policy"] == candidate][["seed", "episode", "eval_seed", "order_level_ret_mean", "order_ret_excel"]]
            paired = cand_df.merge(ref_df, on=["seed", "episode", "eval_seed"], suffixes=("_candidate", "_reference"))
            if paired.empty:
                continue
            for metric in ("order_level_ret_mean", "order_ret_excel"):
                delta = paired[f"{metric}_candidate"] - paired[f"{metric}_reference"]
                rows.append(
                    {
                        "candidate_policy": candidate,
                        "reference_policy": ref,
                        "metric": metric,
                        "delta_mean": float(delta.mean()),
                        "positive_pairs": int((delta > 0).sum()),
                        "n_pairs": int(len(delta)),
                    }
                )
    return rows


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "models").mkdir(exist_ok=True)
    args.invocation = "python scripts/run_track_b_architecture_fair_bakeoff.py " + " ".join(sys.argv[1:])

    rows: list[dict[str, Any]] = []
    trained: list[dict[str, Any]] = []

    for seed in args.seeds:
        if not args.skip_static:
            for static_policy in smoke.STATIC_POLICY_SPECS:
                rows.extend(smoke.evaluate_static_policy(static_policy, args=args, seed=int(seed)))
            for h_label, h_policy in smoke.make_heuristic_defaults().items():
                try:
                    rows.extend(smoke.evaluate_heuristic_policy(h_label, h_policy, args=args, seed=int(seed)))
                except ValueError as exc:
                    print(f"[warn] skipping heuristic {h_label}: {exc}", flush=True)
        for policy in args.policies:
            run_dir = output_dir / "models" / f"{policy}_seed{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            print(f"[train] policy={policy} seed={seed}", flush=True)
            model, vec_norm = train_policy(args, policy=policy, seed=int(seed), run_dir=run_dir)
            trained.append(
                {
                    "policy": policy,
                    "seed": int(seed),
                    "model_path": str((run_dir / f"{policy}_model.zip").resolve()),
                    "vec_normalize_path": str((run_dir / "vec_normalize.pkl").resolve()),
                }
            )
            rows.extend(
                evaluate_policy(
                    args, policy=policy, seed=int(seed), model=model, vec_norm=vec_norm
                )
            )
            vec_norm.close()

    seed_rows = smoke.aggregate_seed_metrics(rows)
    policy_rows = smoke.aggregate_policy_metrics(seed_rows, learned_policy="ppo_dmlpa_positional")
    paired_rows = paired_architecture_rows(rows)

    smoke.save_csv(output_dir / "episode_metrics.csv", rows)
    smoke.save_csv(output_dir / "seed_metrics.csv", seed_rows)
    smoke.save_csv(output_dir / "policy_summary.csv", policy_rows)
    smoke.save_csv(output_dir / "paired_architecture_comparison.csv", paired_rows)

    summary = {
        "config": {
            "seeds": [int(s) for s in args.seeds],
            "policies": list(args.policies),
            "train_timesteps": int(args.train_timesteps),
            "eval_episodes": int(args.eval_episodes),
            "reward_mode": str(args.reward_mode),
            "risk_level": str(args.risk_level),
            "observation_version": str(args.observation_version),
            "action_contract": "track_b_v1",
            "max_steps": int(args.max_steps),
            "n_envs": int(args.n_envs),
            "n_steps": int(args.n_steps),
            "history_factor": int(args.history_factor),
            "history_observation": {
                "ppo_mlp": "one-step v9 observation",
                "ppo_mlp_history": "12-frame stack of v9 plus previous 8D action",
                "ppo_dmlpa_positional": "12-frame stack of v9 plus previous 8D action",
            },
            "raw_material_flow_mode": "kit_equivalent_order_up_to",
            "primary_metric": "order_level_ret_mean",
            "invocation": args.invocation,
        },
        "trained_models": trained,
        "policy_summary": policy_rows,
        "paired_architecture_comparison": paired_rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (output_dir / "README.md").write_text(
        "# Track B Architecture Fair Bakeoff\n\n"
        "Compares PPO+MLP, PPO+MLP-history, and PPO+DMLPA-positional under the same corrected Track B protocol.\n"
    )
    print(f"Wrote fair bakeoff bundle to {output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
