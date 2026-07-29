#!/usr/bin/env python3
"""Run a Track B PPO+DMLPA architecture sidecar.

This runner mirrors ``run_track_b_kan_sidecar.py`` but uses David's
Transformer-over-history feature extractor. The careful bit is evaluation:
training uses VecNormalize followed by VecFrameStack, so evaluation must feed
the policy an identical stack of normalized frames instead of a single v7
observation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
import sys

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.dmlpa_extractor import DMLPA
import scripts.run_track_b_smoke as smoke
from supply_chain.episode_metrics import compute_episode_metrics
from supply_chain.external_env_interface import get_episode_terminal_metrics, make_track_b_env


def build_parser() -> argparse.ArgumentParser:
    parser = smoke.build_parser()
    parser.description = "PPO+DMLPA Track B architecture sidecar."
    parser.set_defaults(
        output_dir=Path("outputs/experiments/track_b_dmlpa_sidecar_2026-07-03/smoke_seed1"),
        seeds=[1],
        train_timesteps=10_000,
        eval_episodes=2,
        reward_mode="control_v1",
        risk_level="adaptive_benchmark_v2",
        observation_version="v7",
        max_steps=104,
        n_steps=1024,
        batch_size=256,
        learning_rate=3e-4,
        n_epochs=10,
        algo="ppo",
    )
    parser.add_argument("--dmlpa-factor", type=int, default=12)
    parser.add_argument("--dmlpa-features-dim", type=int, default=120)
    parser.add_argument("--dmlpa-nhead", type=int, default=12)
    parser.add_argument("--dmlpa-layers", type=int, default=4)
    return parser


def _stack_reset(obs_norm: np.ndarray, factor: int) -> np.ndarray:
    """Mimic SB3 VecFrameStack reset for flat observations."""
    base = np.asarray(obs_norm, dtype=np.float32).reshape(-1)
    stacked = np.zeros((base.shape[0] * factor,), dtype=np.float32)
    stacked[-base.shape[0] :] = base
    return stacked


def _stack_step(stack: np.ndarray, obs_norm: np.ndarray) -> np.ndarray:
    base = np.asarray(obs_norm, dtype=np.float32).reshape(-1)
    next_stack = np.roll(stack, -base.shape[0]).astype(np.float32, copy=False)
    next_stack[-base.shape[0] :] = base
    return next_stack


def train_ppo_dmlpa(
    args: argparse.Namespace, seed: int, run_dir: Path
) -> tuple[Any, VecNormalize]:
    n_envs = max(1, int(getattr(args, "n_envs", 1)))
    vec_env = DummyVecEnv(
        [smoke.make_monitored_training_env(args, seed + i) for i in range(n_envs)]
    )
    vec_norm = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
    )
    vec_stack = VecFrameStack(vec_norm, n_stack=int(args.dmlpa_factor))
    policy_kwargs = {
        "features_extractor_class": DMLPA,
        "features_extractor_kwargs": {
            "factor": int(args.dmlpa_factor),
            "features_dim": int(args.dmlpa_features_dim),
            "nhead": int(args.dmlpa_nhead),
            "num_layers": int(args.dmlpa_layers),
        },
        "net_arch": {"pi": [], "vf": []},
    }
    model: Any = PPO(
        "MlpPolicy",
        vec_stack,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        policy_kwargs=policy_kwargs,
        seed=seed,
        verbose=0,
        device="cpu",
    )
    model.learn(total_timesteps=int(args.train_timesteps))
    model.save(run_dir / "ppo_dmlpa_model.zip")
    vec_norm.save(str(run_dir / "vec_normalize.pkl"))
    return model, vec_norm


def evaluate_trained_policy_dmlpa(
    *,
    args: argparse.Namespace,
    seed: int,
    model: Any,
    vec_norm: VecNormalize,
    order_ledger_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    env_kwargs = smoke.build_env_kwargs(args)
    vec_norm.training = False
    vec_norm.norm_reward = False
    factor = int(args.dmlpa_factor)

    for episode_idx in range(args.eval_episodes):
        eval_seed = seed + smoke.EVAL_EPISODE_SEED_OFFSET + episode_idx
        env = smoke.apply_eval_wrappers(make_track_b_env(**env_kwargs), args)
        obs, info = env.reset(seed=eval_seed)
        obs_norm = vec_norm.normalize_obs(np.asarray(obs, dtype=np.float32)[None, :])[0]
        stacked_obs = _stack_reset(obs_norm, factor)
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
            action, _ = model.predict(stacked_obs[None, :], deterministic=True)
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
                stacked_obs = _stack_step(stacked_obs, obs_norm)

        rows.append(
            smoke._finalize_episode_row(
                policy="ppo_dmlpa",
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
        smoke.append_order_ledger_rows(
            order_ledger_rows,
            env,
            policy="ppo_dmlpa",
            seed=seed,
            episode=episode_idx + 1,
            eval_seed=eval_seed,
        )
        env.close()
    return rows


def main() -> None:
    args = build_parser().parse_args()
    args.invocation = "python scripts/run_track_b_dmlpa_sidecar.py " + " ".join(sys.argv[1:])
    original_train = smoke.train_ppo
    original_eval = smoke.evaluate_trained_policy
    original_learned_policy_name = smoke.learned_policy_name
    original_model_filename = smoke.model_filename
    try:
        smoke.train_ppo = train_ppo_dmlpa  # type: ignore[assignment]
        smoke.evaluate_trained_policy = evaluate_trained_policy_dmlpa  # type: ignore[assignment]
        smoke.learned_policy_name = lambda _args=None: "ppo_dmlpa"  # type: ignore[assignment]
        smoke.model_filename = lambda _args=None: "ppo_dmlpa_model.zip"  # type: ignore[assignment]
        summary = smoke.run_smoke(args)
    finally:
        smoke.train_ppo = original_train  # type: ignore[assignment]
        smoke.evaluate_trained_policy = original_eval  # type: ignore[assignment]
        smoke.learned_policy_name = original_learned_policy_name  # type: ignore[assignment]
        smoke.model_filename = original_model_filename  # type: ignore[assignment]

    print(f"Wrote PPO+DMLPA sidecar bundle to {summary['artifacts']['summary_json']}")
    for row in summary["policy_summary"]:
        print(
            f"{row['policy']}: fill={float(row['fill_rate_mean']):.3f}, "
            f"ret={float(row['order_level_ret_mean_mean']):.6f}, "
            f"cost={float(row['assembly_cost_index_mean']):.3f}"
        )
    decision = summary["decision"]
    print(
        "Decision: "
        f"best_static={decision['best_static_policy']}, "
        f"ppo_dmlpa_vs_best_ret={float(decision['learned_order_level_ret_gap_vs_best_static']):+.6f}, "
        f"raw_ret_win={decision['learned_raw_ret_win_vs_best_static']}"
    )


if __name__ == "__main__":
    main()
