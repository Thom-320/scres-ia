#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import time
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from supply_chain.config import (
    BENCHMARK_OBSERVATION_VERSION,
    BENCHMARK_REWARD_MODE,
    BENCHMARK_W_BO,
    BENCHMARK_W_COST,
    BENCHMARK_W_DISR,
    DEFAULT_YEAR_BASIS,
    RET_SHIFT_COST_DELTA_DEFAULT,
    YEAR_BASIS_OPTIONS,
)
from supply_chain.env import MFSCGymEnv

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from supply_chain.env_experimental_shifts import MFSCGymEnvShifts, RET_SEQ_KAPPA

BASE_ENV_REWARD_MODES = {"proxy", "rt_v0"}
SHIFT_ENV_REWARD_MODES = {
    "rt_v0",
    "ReT_thesis",
    "ReT_corrected",
    "ReT_corrected_cost",
    "ReT_seq_v1",
    "control_v1",
    "control_v1_pbrs",
    "ReT_garrido2024_raw",
    "ReT_garrido2024",
    "ReT_garrido2024_train",
    "ReT_cd_v1",
    "ReT_cd_sigmoid",
}
DEFAULT_REWARD_MODE_BY_ENV = {
    "base": "rt_v0",
    "shift_control": BENCHMARK_REWARD_MODE,
}


def resolve_reward_mode(args: argparse.Namespace) -> str:
    reward_mode = args.reward_mode
    if reward_mode is None:
        reward_mode = DEFAULT_REWARD_MODE_BY_ENV[args.env_variant]
        args.reward_mode = reward_mode
    return reward_mode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train PPO on MFSC environments and report robust metrics."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode (defaults to 10k timesteps if --timesteps is not provided).",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Total training timesteps (defaults: quick=10k, full=100k).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Main RNG seed.")
    parser.add_argument(
        "--env-variant",
        choices=["base", "shift_control"],
        default="shift_control",
        help="Training environment: legacy 4-action env or recommended shift-control env.",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=1,
        help="Number of parallel envs for PPO collection.",
    )
    parser.add_argument("--step-size-hours", type=float, default=168.0)
    parser.add_argument("--max-steps-per-episode", type=int, default=260)
    parser.add_argument(
        "--observation-version",
        choices=["v1", "v2", "v3", "v4"],
        default=BENCHMARK_OBSERVATION_VERSION,
        help=(
            "Observation contract for shift-control env. The frozen paper "
            "benchmark uses v1; newer ablations may use v2+."
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--random-episodes", type=int, default=30)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for models, curves, json and csv metrics.",
    )
    parser.add_argument(
        "--year-basis",
        choices=YEAR_BASIS_OPTIONS,
        default=DEFAULT_YEAR_BASIS,
        help="Annualization basis passed to MFSCGymEnv.",
    )
    parser.add_argument(
        "--reward-mode",
        choices=[
            "proxy",
            "rt_v0",
            "ReT_thesis",
            "ReT_corrected",
            "ReT_corrected_cost",
            "ReT_seq_v1",
            "control_v1",
            "control_v1_pbrs",
            "ReT_garrido2024_raw",
            "ReT_garrido2024",
            "ReT_garrido2024_train",
            "ReT_cd_v1",
            "ReT_cd_sigmoid",
        ],
        default=None,
        help="Reward function for the selected environment variant.",
    )
    parser.add_argument(
        "--risk-level",
        choices=["current", "increased", "severe", "severe_training"],
        default="current",
        help="Risk parameter level from thesis Table 6.12.",
    )
    parser.add_argument(
        "--rt-alpha", type=float, default=8.0, help="R_t recovery weight."
    )
    parser.add_argument(
        "--rt-beta", type=float, default=1.0, help="R_t holding cost weight."
    )
    parser.add_argument(
        "--rt-gamma-reward", type=float, default=7.0, help="R_t service loss weight."
    )
    parser.add_argument("--rt-recovery-scale", type=float, default=46.0)
    parser.add_argument("--rt-inventory-scale", type=float, default=17_200_000.0)
    parser.add_argument(
        "--shift-delta",
        type=float,
        default=RET_SHIFT_COST_DELTA_DEFAULT,
        help="Linear shift-cost weight for the shift-control environment.",
    )
    parser.add_argument(
        "--stochastic-pt",
        action="store_true",
        help="Enable stochastic processing times in the shift-control environment.",
    )
    parser.add_argument(
        "--w-bo",
        type=float,
        default=BENCHMARK_W_BO,
        help="Service-loss weight for control_v1/control_v1_pbrs.",
    )
    parser.add_argument(
        "--w-cost",
        type=float,
        default=BENCHMARK_W_COST,
        help="Shift-cost weight for control_v1/control_v1_pbrs.",
    )
    parser.add_argument(
        "--w-disr",
        type=float,
        default=BENCHMARK_W_DISR,
        help="Disruption-fraction weight for control_v1/control_v1_pbrs.",
    )
    parser.add_argument(
        "--ret-seq-kappa",
        type=float,
        default=RET_SEQ_KAPPA,
        help="Adaptive-efficiency scaling for reward_mode=ReT_seq_v1.",
    )
    parser.add_argument(
        "--ret-g24-calibration",
        type=Path,
        default=None,
        help=(
            "Optional Garrido-2024 calibration JSON. "
            "Recommended when using ReT_garrido2024_raw, "
            "ReT_garrido2024, or ReT_garrido2024_train."
        ),
    )
    return parser


def ci95(values: list[float]) -> tuple[float, float]:
    if len(values) < 2:
        return (float("nan"), float("nan"))
    arr = np.asarray(values, dtype=np.float64)
    half = 1.96 * arr.std(ddof=1) / np.sqrt(len(arr))
    return float(arr.mean() - half), float(arr.mean() + half)


class RawRewardCallback(BaseCallback):
    """Collect raw Monitor episode rewards during training."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.raw_rewards: list[float] = []
        self.episode_lengths: list[int] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            ep = info.get("episode")
            if ep is None:
                continue
            self.raw_rewards.append(float(ep["r"]))
            self.episode_lengths.append(int(ep["l"]))
            if self.verbose > 0 and len(self.raw_rewards) % 5 == 0:
                tail = self.raw_rewards[-5:]
                print(
                    f"  Ep {len(self.raw_rewards):>4d} |"
                    f" Raw reward mean(last 5): {np.mean(tail):>12,.0f} |"
                    f" Timesteps: {self.num_timesteps:>8,}"
                )
        return True


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    resolve_reward_mode(args)
    if args.env_variant == "base":
        if args.reward_mode not in BASE_ENV_REWARD_MODES:
            parser.error("Base env supports only reward modes: proxy, rt_v0.")
        if args.risk_level not in ("current", "increased"):
            parser.error("Base env supports only current or increased risk levels.")
        if args.observation_version != "v1":
            parser.error("Base env supports only observation_version=v1.")
    elif args.reward_mode not in SHIFT_ENV_REWARD_MODES:
        parser.error(
            "Shift-control env supports only reward modes: "
            "rt_v0, ReT_thesis, ReT_corrected, ReT_corrected_cost, ReT_seq_v1, "
            "control_v1, control_v1_pbrs, ReT_garrido2024_raw, "
            "ReT_garrido2024, ReT_garrido2024_train, ReT_cd_v1, "
            "ReT_cd_sigmoid."
        )


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    return Path("outputs") / f"ppo_{args.env_variant}_{args.reward_mode}"


def build_env_instance(
    args: argparse.Namespace, seed: int | None = None
) -> MFSCGymEnv | MFSCGymEnvShifts:
    reward_mode = resolve_reward_mode(args)
    common_kwargs = {
        "step_size_hours": args.step_size_hours,
        "max_steps": args.max_steps_per_episode,
        "year_basis": args.year_basis,
        "risk_level": args.risk_level,
        "reward_mode": reward_mode,
        "rt_alpha": args.rt_alpha,
        "rt_beta": args.rt_beta,
        "rt_gamma": args.rt_gamma_reward,
        "rt_recovery_scale": args.rt_recovery_scale,
        "rt_inventory_scale": args.rt_inventory_scale,
    }
    if args.env_variant == "shift_control":
        env = MFSCGymEnvShifts(
            **common_kwargs,
            rt_delta=args.shift_delta,
            stochastic_pt=args.stochastic_pt,
            observation_version=args.observation_version,
            w_bo=args.w_bo,
            w_cost=args.w_cost,
            w_disr=args.w_disr,
            ret_seq_kappa=args.ret_seq_kappa,
            ret_g24_calibration_path=(
                str(args.ret_g24_calibration)
                if args.ret_g24_calibration is not None
                else None
            ),
        )
    else:
        env = MFSCGymEnv(**common_kwargs)
    if seed is not None:
        env.reset(seed=seed)
    return env


def make_env_factory(args: argparse.Namespace, seed: int) -> callable:
    def _init() -> Monitor:
        return Monitor(build_env_instance(args, seed=seed))

    return _init


def build_training_env(args: argparse.Namespace) -> VecNormalize:
    env_fns = [make_env_factory(args, seed=args.seed + i) for i in range(args.n_envs)]
    vec = DummyVecEnv(env_fns)
    norm_reward = args.reward_mode == "proxy"
    return VecNormalize(vec, norm_obs=True, norm_reward=norm_reward, clip_obs=10.0)


def _make_eval_env(args: argparse.Namespace) -> MFSCGymEnv | MFSCGymEnvShifts:
    return build_env_instance(args)


def random_baseline(args: argparse.Namespace) -> dict[str, Any]:
    rewards: list[float] = []
    rows: list[dict[str, Any]] = []
    for episode_idx in range(args.random_episodes):
        env = _make_eval_env(args)
        _, _ = env.reset(seed=args.seed + 10_000 + episode_idx)
        done, truncated, total, steps = False, False, 0.0, 0
        while not (done or truncated):
            _, reward, done, truncated, _ = env.step(env.action_space.sample())
            total += float(reward)
            steps += 1
        rewards.append(total)
        rows.append({"episode": episode_idx + 1, "raw_reward": total, "steps": steps})

    mean_reward = float(np.mean(rewards))
    std_reward = float(np.std(rewards, ddof=1)) if len(rewards) > 1 else 0.0
    lo, hi = ci95(rewards)
    return {
        "mean": mean_reward,
        "std": std_reward,
        "ci95_low": lo,
        "ci95_high": hi,
        "rewards": rewards,
        "rows": rows,
    }


def evaluate_trained(
    model: PPO, vec_norm: VecNormalize, args: argparse.Namespace
) -> dict[str, Any]:
    vec_norm.training = False
    rewards: list[float] = []
    rows: list[dict[str, Any]] = []

    for episode_idx in range(args.eval_episodes):
        env = _make_eval_env(args)
        obs_raw, _ = env.reset(seed=args.seed + 20_000 + episode_idx)
        done, truncated, total, steps = False, False, 0.0, 0

        while not (done or truncated):
            obs_norm = vec_norm.normalize_obs(
                np.asarray(obs_raw, dtype=np.float32)[None, :]
            )
            action, _ = model.predict(obs_norm, deterministic=True)
            obs_raw, reward, done, truncated, _ = env.step(action[0])
            total += float(reward)
            steps += 1

        rewards.append(total)
        rows.append({"episode": episode_idx + 1, "raw_reward": total, "steps": steps})

    mean_reward = float(np.mean(rewards))
    std_reward = float(np.std(rewards, ddof=1)) if len(rewards) > 1 else 0.0
    lo, hi = ci95(rewards)
    return {
        "mean": mean_reward,
        "std": std_reward,
        "ci95_low": lo,
        "ci95_high": hi,
        "rewards": rewards,
        "rows": rows,
    }


def save_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_learning_curve(
    output_path: Path,
    raw_rewards: list[float],
    random_mean: float,
    trained_mean: float,
    elapsed_s: float,
) -> None:
    if len(raw_rewards) < 3:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"MFSC PPO MLP Baseline ({len(raw_rewards)} episodes, {elapsed_s:.0f}s)",
        fontsize=13,
        fontweight="bold",
    )

    ax_left = axes[0]
    ax_left.plot(raw_rewards, alpha=0.35, color="#2196F3", lw=0.8, label="Per episode")
    window = min(15, max(5, len(raw_rewards) // 6))
    if len(raw_rewards) >= window:
        roll = np.convolve(raw_rewards, np.ones(window) / window, mode="valid")
        ax_left.plot(
            range(window - 1, len(raw_rewards)),
            roll,
            color="#E91E63",
            lw=2.2,
            label=f"Rolling avg ({window})",
        )
    ax_left.axhline(random_mean, color="gray", ls="--", lw=1.5, label="Random mean")
    ax_left.axhline(
        trained_mean, color="#2E7D32", ls=":", lw=1.5, label="Trained eval mean"
    )
    ax_left.set_xlabel("Episode")
    ax_left.set_ylabel("Raw reward")
    ax_left.set_title("Training raw rewards")
    ax_left.grid(True, alpha=0.3)
    ax_left.legend(fontsize=8)

    ax_right = axes[1]
    split_idx = max(1, len(raw_rewards) // 2)
    first_half = raw_rewards[:split_idx]
    second_half = raw_rewards[split_idx:]
    ax_right.hist(
        first_half,
        bins=15,
        alpha=0.5,
        color="#FF9800",
        label="First half",
        density=True,
    )
    ax_right.hist(
        second_half,
        bins=15,
        alpha=0.5,
        color="#4CAF50",
        label="Second half",
        density=True,
    )
    ax_right.axvline(random_mean, color="gray", ls="--", lw=1.2)
    ax_right.set_title("Distribution shift")
    ax_right.set_xlabel("Raw reward")
    ax_right.grid(True, alpha=0.3)
    ax_right.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def train_model(
    args: argparse.Namespace,
) -> tuple[PPO, VecNormalize, RawRewardCallback, float]:
    vec_env = build_training_env(args)

    tb_log = str(args.output_dir / "tensorboard")
    if importlib.util.find_spec("tensorboard") is None:
        tb_log = None

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        policy_kwargs={"net_arch": {"pi": [64, 64], "vf": [64, 64]}},
        verbose=1,
        seed=args.seed,
        device="cpu",
        tensorboard_log=tb_log,
    )

    callback = RawRewardCallback(verbose=1)
    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, callback=callback)
    elapsed = time.time() - t0
    return model, vec_env, callback, elapsed


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    if args.timesteps is None:
        args.timesteps = 10_000 if args.quick else 100_000

    args.output_dir = resolve_output_dir(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("MFSC PPO Training")
    print(
        f"mode={'quick' if args.quick else 'full'} | timesteps={args.timesteps:,} |"
        f" env={args.env_variant} | year_basis={args.year_basis}"
    )
    print(
        f"reward={args.reward_mode} | risk={args.risk_level}"
        + (
            f" | α={args.rt_alpha} β={args.rt_beta} γ={args.rt_gamma_reward}"
            if args.reward_mode == "rt_v0"
            else ""
        )
        + (
            f" | obs={args.observation_version}"
            if args.env_variant == "shift_control"
            else ""
        )
        + (
            f" | w_bo={args.w_bo} w_cost={args.w_cost} w_disr={args.w_disr}"
            if args.reward_mode in ("control_v1", "control_v1_pbrs")
            else ""
        )
        + (f" | κ={args.ret_seq_kappa}" if args.reward_mode == "ReT_seq_v1" else "")
        + (f" | δ={args.shift_delta}" if args.env_variant == "shift_control" else "")
    )
    print("=" * 70)

    random_stats = random_baseline(args)
    print(
        f"Random baseline mean={random_stats['mean']:,.0f} |"
        f" std={random_stats['std']:,.0f} |"
        f" 95% CI=[{random_stats['ci95_low']:,.0f}, {random_stats['ci95_high']:,.0f}]"
    )

    model, vec_env, callback, elapsed_s = train_model(args)
    model_path = args.output_dir / "ppo_mfsc_baseline"
    model.save(str(model_path))
    vec_env.save(str(args.output_dir / "vec_normalize.pkl"))

    trained_stats = evaluate_trained(model, vec_env, args)
    print(
        f"Trained mean={trained_stats['mean']:,.0f} |"
        f" std={trained_stats['std']:,.0f} |"
        f" 95% CI=[{trained_stats['ci95_low']:,.0f}, {trained_stats['ci95_high']:,.0f}]"
    )

    improvement_pct = (
        ((trained_stats["mean"] - random_stats["mean"]) / abs(random_stats["mean"]))
        * 100
        if random_stats["mean"] != 0
        else float("nan")
    )
    print(f"Improvement vs random baseline: {improvement_pct:+.2f}%")

    save_rows_csv(
        args.output_dir / "random_baseline_episodes.csv", random_stats["rows"]
    )
    save_rows_csv(args.output_dir / "trained_eval_episodes.csv", trained_stats["rows"])
    plot_learning_curve(
        output_path=args.output_dir / "learning_curve.png",
        raw_rewards=callback.raw_rewards,
        random_mean=random_stats["mean"],
        trained_mean=trained_stats["mean"],
        elapsed_s=elapsed_s,
    )

    training_log = {
        "timestamp_epoch": int(time.time()),
        "config": {
            "quick": args.quick,
            "timesteps": args.timesteps,
            "seed": args.seed,
            "n_envs": args.n_envs,
            "step_size_hours": args.step_size_hours,
            "max_steps_per_episode": args.max_steps_per_episode,
            "learning_rate": args.learning_rate,
            "n_steps": args.n_steps,
            "batch_size": args.batch_size,
            "n_epochs": args.n_epochs,
            "gamma": args.gamma,
            "gae_lambda": args.gae_lambda,
            "clip_range": args.clip_range,
            "ent_coef": args.ent_coef,
            "eval_episodes": args.eval_episodes,
            "random_episodes": args.random_episodes,
            "year_basis": args.year_basis,
            "env_variant": args.env_variant,
            "reward_mode": args.reward_mode,
            "observation_version": args.observation_version,
            "risk_level": args.risk_level,
            "rt_alpha": args.rt_alpha,
            "rt_beta": args.rt_beta,
            "rt_gamma_reward": args.rt_gamma_reward,
            "rt_recovery_scale": args.rt_recovery_scale,
            "rt_inventory_scale": args.rt_inventory_scale,
            "shift_delta": args.shift_delta,
            "stochastic_pt": args.stochastic_pt,
            "w_bo": args.w_bo,
            "w_cost": args.w_cost,
            "w_disr": args.w_disr,
            "ret_seq_kappa": args.ret_seq_kappa,
        },
        "random_baseline": random_stats,
        "trained_eval": trained_stats,
        "train_episode_rewards": callback.raw_rewards,
        "train_episode_lengths": callback.episode_lengths,
        "improvement_pct": improvement_pct,
        "elapsed_seconds": elapsed_s,
        "artifacts": {
            "model": str(model_path.with_suffix(".zip")),
            "vec_normalize": str(args.output_dir / "vec_normalize.pkl"),
            "learning_curve": str(args.output_dir / "learning_curve.png"),
            "random_csv": str(args.output_dir / "random_baseline_episodes.csv"),
            "trained_csv": str(args.output_dir / "trained_eval_episodes.csv"),
        },
    }
    with (args.output_dir / "training_log.json").open(
        "w", encoding="utf-8"
    ) as file_obj:
        json.dump(training_log, file_obj, indent=2)

    print(f"Artifacts saved under: {args.output_dir}")


if __name__ == "__main__":
    main()
