#!/usr/bin/env python3
"""Track B PPO with a pretrained belief-encoder trunk (Ruta A).

Loads a trunk pretrained by ``scripts/pretrain_risk_belief_encoder.py`` (BCE
loss only, on full v10 observations, predicting future R24 occurrence) as
PPO's ``features_extractor``, then fine-tunes normally with PPO's standard
loss. No auxiliary loss during RL; no observation wrapper appending scalars
(unlike ``scripts/run_track_b_risk_belief_sidecar.py``, which appends 2 frozen
probabilities instead of transplanting a shared representation and found no
benefit for PPO+MLP). PPO's reward and the final evaluation metric are
unchanged: Garrido Excel ReT (``order_ret_excel_mean``).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.belief_extractor import MLPBeliefExtractor, RealKANBeliefExtractor
import scripts.run_track_b_smoke as smoke


def build_parser() -> argparse.ArgumentParser:
    parser = smoke.build_parser()
    parser.description = "Track B PPO sidecar with a pretrained belief-encoder trunk (Ruta A)."
    parser.set_defaults(
        output_dir=Path("outputs/experiments/track_b_belief_encoder_sidecar_2026-07-04/smoke"),
        seeds=[1],
        train_timesteps=10_000,
        eval_episodes=2,
        reward_mode="control_v1",
        risk_level="adaptive_benchmark_v2",
        observation_version="v10",
        max_steps=104,
        n_steps=1024,
        batch_size=256,
        learning_rate=3e-4,
        n_epochs=10,
        algo="ppo",
    )
    parser.add_argument(
        "--architecture",
        choices=("ppo_mlp", "real_kan"),
        default="ppo_mlp",
    )
    parser.add_argument("--pretrained-encoder-path", type=Path, required=True)
    parser.add_argument("--features-dim", type=int, default=64)
    parser.add_argument("--hidden-width", type=int, default=64)
    parser.add_argument("--kan-features-dim", type=int, default=32)
    parser.add_argument("--kan-hidden-width", type=int, default=32)
    parser.add_argument("--kan-grid", type=int, default=3)
    parser.add_argument("--kan-k", type=int, default=3)
    return parser


def _load_encoder_weights(model: Any, encoder_path: Path) -> None:
    state = torch.load(encoder_path, map_location="cpu")
    missing, unexpected = model.policy.features_extractor.load_state_dict(state, strict=True)
    if missing or unexpected:  # pragma: no cover - defensive
        raise RuntimeError(f"Encoder state_dict mismatch: missing={missing}, unexpected={unexpected}")


def train_ppo_mlp_belief_encoder(
    args: argparse.Namespace, seed: int, run_dir: Path
) -> tuple[Any, VecNormalize]:
    n_envs = max(1, int(getattr(args, "n_envs", 1)))
    vec_env = DummyVecEnv(
        [smoke.make_monitored_training_env(args, seed + i) for i in range(n_envs)]
    )
    vec_norm = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    policy_kwargs = {
        "features_extractor_class": MLPBeliefExtractor,
        "features_extractor_kwargs": {
            "features_dim": int(args.features_dim),
            "hidden_width": int(args.hidden_width),
        },
        "net_arch": {"pi": [64, 64], "vf": [64, 64]},
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
    _load_encoder_weights(model, Path(args.pretrained_encoder_path))
    model.learn(total_timesteps=int(args.train_timesteps))
    model.save(run_dir / "ppo_mlp_belief_encoder_model.zip")
    vec_norm.save(str(run_dir / "vec_normalize.pkl"))
    return model, vec_norm


def train_ppo_real_kan_belief_encoder(
    args: argparse.Namespace, seed: int, run_dir: Path
) -> tuple[Any, VecNormalize]:
    n_envs = max(1, int(getattr(args, "n_envs", 1)))
    vec_env = DummyVecEnv(
        [smoke.make_monitored_training_env(args, seed + i) for i in range(n_envs)]
    )
    vec_norm = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    policy_kwargs = {
        "features_extractor_class": RealKANBeliefExtractor,
        "features_extractor_kwargs": {
            "features_dim": int(args.kan_features_dim),
            "hidden_width": int(args.kan_hidden_width),
            "grid": int(args.kan_grid),
            "k": int(args.kan_k),
            "seed": int(seed),
        },
        "net_arch": {"pi": [], "vf": []},
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
    _load_encoder_weights(model, Path(args.pretrained_encoder_path))
    model.learn(total_timesteps=int(args.train_timesteps))
    model.save(run_dir / "real_kan_belief_encoder_model.zip")
    vec_norm.save(str(run_dir / "vec_normalize.pkl"))
    return model, vec_norm


def main() -> None:
    args = build_parser().parse_args()
    if str(args.observation_version) != "v10":
        raise ValueError("Belief-encoder sidecar currently requires --observation-version v10.")

    original_train = smoke.train_ppo
    original_learned_policy_name = smoke.learned_policy_name
    original_model_filename = smoke.model_filename
    policy_name = "ppo_mlp_belief_encoder" if args.architecture == "ppo_mlp" else "real_kan_belief_encoder"
    try:
        if args.architecture == "ppo_mlp":
            smoke.train_ppo = train_ppo_mlp_belief_encoder  # type: ignore[assignment]
            smoke.model_filename = lambda _args=None: "ppo_mlp_belief_encoder_model.zip"  # type: ignore[assignment]
        else:
            smoke.train_ppo = train_ppo_real_kan_belief_encoder  # type: ignore[assignment]
            smoke.model_filename = lambda _args=None: "real_kan_belief_encoder_model.zip"  # type: ignore[assignment]
        smoke.learned_policy_name = lambda _args=None: policy_name  # type: ignore[assignment]
        summary = smoke.run_smoke(args)
    finally:
        smoke.train_ppo = original_train  # type: ignore[assignment]
        smoke.learned_policy_name = original_learned_policy_name  # type: ignore[assignment]
        smoke.model_filename = original_model_filename  # type: ignore[assignment]

    print(f"Wrote Track B belief-encoder sidecar bundle to {summary['artifacts']['summary_json']}")
    for row in summary["policy_summary"]:
        if row["policy"] == policy_name:
            print(
                f"{policy_name}: order_ret_excel={float(row['order_ret_excel_mean']):.6f}, "
                f"cost={float(row['assembly_cost_index_mean']):.3f}"
            )


if __name__ == "__main__":
    main()
