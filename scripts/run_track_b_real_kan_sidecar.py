#!/usr/bin/env python3
"""Run a Track B PPO + real-KAN (pykan) architecture sidecar.

This is the literal version of Garrido's KAN suggestion: the official
``pykan`` package (learnable B-spline edge functions, Liu et al. 2024), not
the RBF-inspired sidecar in ``scripts/kan_extractor.py``. Same evaluation
protocol as the other Track B architecture sidecars (DMLPA, RBF-KAN) so the
result is directly, apples-to-apples comparable.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from scripts.real_kan_extractor import RealKANFeaturesExtractor
from scripts.run_track_b_ablation import PostCdcOnlyWrapper
import scripts.run_track_b_smoke as smoke


def build_parser() -> argparse.ArgumentParser:
    parser = smoke.build_parser()
    parser.description = "PPO + real pykan-KAN Track B architecture sidecar."
    parser.set_defaults(
        output_dir=Path("outputs/experiments/track_b_real_kan_sidecar_2026-07-03/smoke_seed1"),
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
    parser.add_argument("--kan-features-dim", type=int, default=32)
    parser.add_argument("--kan-hidden-width", type=int, default=32)
    parser.add_argument("--kan-grid", type=int, default=3)
    parser.add_argument("--kan-k", type=int, default=3)
    parser.add_argument(
        "--kan-head-width",
        type=int,
        default=0,
        help="Optional MLP head width after the KAN extractor. 0 uses a linear policy/value head.",
    )
    parser.add_argument(
        "--post-cdc-only",
        action="store_true",
        help="Freeze Op3/CDC quantity+ROP at Garrido baseline (PostCdcOnlyWrapper), matching the "
        "canonical PPO post_cdc_only ablation contract.",
    )
    parser.add_argument(
        "--obs-config",
        choices=("v7_full", "v7_no_forecast", "v7_no_regime_forecast", "v5_7d"),
        default=None,
        help=(
            "Optional observation-ablation contract to mirror "
            "run_track_b_observation_ablation.py. Use v7_no_forecast for the "
            "reviewer-safe no-forecast Real-KAN lane."
        ),
    )
    return parser


def train_ppo_real_kan(
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
    head_width = int(getattr(args, "kan_head_width", 0))
    net_arch: dict[str, list[int]] = (
        {"pi": [head_width], "vf": [head_width]} if head_width > 0 else {"pi": [], "vf": []}
    )
    policy_kwargs = {
        "features_extractor_class": RealKANFeaturesExtractor,
        "features_extractor_kwargs": {
            "features_dim": int(args.kan_features_dim),
            "hidden_width": int(args.kan_hidden_width),
            "grid": int(args.kan_grid),
            "k": int(args.kan_k),
            "seed": int(seed),
        },
        "net_arch": net_arch,
    }
    model: Any = PPO(
        "MlpPolicy",
        vec_norm,
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
    model.save(run_dir / "ppo_real_kan_model.zip")
    vec_norm.save(str(run_dir / "vec_normalize.pkl"))
    return model, vec_norm


def main() -> None:
    args = build_parser().parse_args()
    args.invocation = "python scripts/run_track_b_real_kan_sidecar.py " + " ".join(sys.argv[1:])
    if getattr(args, "obs_config", None):
        from scripts.run_track_b_observation_ablation import OBS_ABLATION_CONFIGS

        obs_config = OBS_ABLATION_CONFIGS[str(args.obs_config)]
        args.observation_version = obs_config.observation_version
        args._observation_wrapper = obs_config.wrapper
    if getattr(args, "post_cdc_only", False):
        args._ablation_wrapper = PostCdcOnlyWrapper  # type: ignore[attr-defined]
    original_train = smoke.train_ppo
    original_learned_policy_name = smoke.learned_policy_name
    original_model_filename = smoke.model_filename
    try:
        smoke.train_ppo = train_ppo_real_kan  # type: ignore[assignment]
        smoke.learned_policy_name = lambda _args=None: "ppo_real_kan"  # type: ignore[assignment]
        smoke.model_filename = lambda _args=None: "ppo_real_kan_model.zip"  # type: ignore[assignment]
        summary = smoke.run_smoke(args)
    finally:
        smoke.train_ppo = original_train  # type: ignore[assignment]
        smoke.learned_policy_name = original_learned_policy_name  # type: ignore[assignment]
        smoke.model_filename = original_model_filename  # type: ignore[assignment]

    print(f"Wrote PPO+real-KAN sidecar bundle to {summary['artifacts']['summary_json']}")
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
        f"ppo_real_kan_vs_best_ret={float(decision['learned_order_level_ret_gap_vs_best_static']):+.6f}, "
        f"raw_ret_win={decision['learned_raw_ret_win_vs_best_static']}"
    )


if __name__ == "__main__":
    main()
