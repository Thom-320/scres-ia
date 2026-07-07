#!/usr/bin/env python3
"""Run a Track B PPO-KAN sidecar smoke test.

The canonical paper result stays PPO+MLP. This runner reuses the Track B smoke
pipeline, but swaps the SB3 MLP feature extractor for a small KAN-style
univariate-basis extractor so we can test whether Garrido's KAN suggestion
changes the structural conclusion.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from scripts.kan_extractor import RBFKANFeaturesExtractor
import scripts.run_track_b_smoke as smoke


def build_parser() -> argparse.ArgumentParser:
    parser = smoke.build_parser()
    parser.description = "PPO-KAN Track B sidecar smoke test."
    parser.set_defaults(
        output_dir=Path("outputs/experiments/track_b_kan_sidecar_2026-07-02/smoke_seed1"),
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
    parser.add_argument("--kan-features-dim", type=int, default=64)
    parser.add_argument("--kan-centers", type=int, default=9)
    parser.add_argument("--kan-center-min", type=float, default=-2.5)
    parser.add_argument("--kan-center-max", type=float, default=2.5)
    parser.add_argument(
        "--kan-no-linear-skip",
        action="store_true",
        help="Disable the linear skip path so the sidecar tests the additive RBF/KAN basis alone.",
    )
    parser.add_argument(
        "--kan-head-width",
        type=int,
        default=0,
        help="Optional MLP head width after the KAN extractor. 0 uses a linear policy/value head.",
    )
    return parser


def train_ppo_kan(
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
        "features_extractor_class": RBFKANFeaturesExtractor,
        "features_extractor_kwargs": {
            "features_dim": int(args.kan_features_dim),
            "num_centers": int(args.kan_centers),
            "center_min": float(args.kan_center_min),
            "center_max": float(args.kan_center_max),
            "use_linear_skip": not bool(getattr(args, "kan_no_linear_skip", False)),
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
    model.save(run_dir / "ppo_kan_model.zip")
    vec_norm.save(str(run_dir / "vec_normalize.pkl"))
    return model, vec_norm


def main() -> None:
    args = build_parser().parse_args()
    args.invocation = "python scripts/run_track_b_kan_sidecar.py " + " ".join(sys.argv[1:])
    original_train = smoke.train_ppo
    original_learned_policy_name = smoke.learned_policy_name
    original_model_filename = smoke.model_filename
    try:
        smoke.train_ppo = train_ppo_kan  # type: ignore[assignment]
        smoke.learned_policy_name = lambda _args=None: "ppo_kan"  # type: ignore[assignment]
        smoke.model_filename = lambda _args=None: "ppo_kan_model.zip"  # type: ignore[assignment]
        summary = smoke.run_smoke(args)
    finally:
        smoke.train_ppo = original_train  # type: ignore[assignment]
        smoke.learned_policy_name = original_learned_policy_name  # type: ignore[assignment]
        smoke.model_filename = original_model_filename  # type: ignore[assignment]

    print(f"Wrote PPO-KAN sidecar bundle to {summary['artifacts']['summary_json']}")
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
        f"ppo_kan_vs_best_ret={float(decision['learned_order_level_ret_gap_vs_best_static']):+.6f}, "
        f"raw_ret_win={decision['learned_raw_ret_win_vs_best_static']}"
    )


if __name__ == "__main__":
    main()
