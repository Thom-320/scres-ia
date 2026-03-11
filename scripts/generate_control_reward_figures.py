#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from benchmark_control_reward import build_env_kwargs
from supply_chain.external_env_interface import make_shift_control_env

WINNING_COMBO = {"w_bo": 4.0, "w_cost": 0.02, "w_disr": 0.0}


class EpisodeRewardCallback(BaseCallback):
    """Collect per-episode raw rewards from Monitor during PPO training."""

    def __init__(self) -> None:
        super().__init__()
        self.timesteps: list[int] = []
        self.rewards: list[float] = []
        self.lengths: list[int] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            episode = info.get("episode")
            if episode is None:
                continue
            self.timesteps.append(int(self.num_timesteps))
            self.rewards.append(float(episode["r"]))
            self.lengths.append(int(episode["l"]))
        return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate paper figures for the control_v1 benchmark."
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    parser.add_argument("--train-timesteps", type=int, default=50_000)
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=Path("outputs/benchmarks/control_reward"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/figures/control_reward_paper"),
    )
    parser.add_argument("--step-size-hours", type=float, default=168.0)
    parser.add_argument("--max-steps", type=int, default=260)
    parser.add_argument("--risk-level", default="increased")
    parser.add_argument(
        "--stochastic-pt",
        action="store_true",
        help="Enable stochastic processing times while tracing PPO reward curves.",
    )
    parser.add_argument("--year-basis", default="thesis")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    return parser


def rolling_mean(values: list[float], window: int = 10) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return arr
    if len(arr) < window:
        return arr
    kernel = np.ones(window, dtype=np.float64) / window
    padded = np.convolve(arr, kernel, mode="valid")
    prefix = np.full(window - 1, np.nan)
    return np.concatenate([prefix, padded])


def train_with_logging(args: argparse.Namespace) -> list[dict[str, object]]:
    histories: list[dict[str, object]] = []
    env_kwargs = build_env_kwargs(args, WINNING_COMBO)

    for seed in args.seeds:

        def _init() -> Monitor:
            env = make_shift_control_env(**env_kwargs)
            env.reset(seed=seed)
            return Monitor(env)

        vec_env = DummyVecEnv([_init])
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
            policy_kwargs={"net_arch": {"pi": [64, 64], "vf": [64, 64]}},
            seed=seed,
            verbose=0,
            device="cpu",
        )
        callback = EpisodeRewardCallback()
        model.learn(total_timesteps=args.train_timesteps, callback=callback)
        histories.append(
            {
                "seed": seed,
                "timesteps": callback.timesteps,
                "rewards": callback.rewards,
                "lengths": callback.lengths,
            }
        )
        vec_env.close()

    return histories


def plot_training_curve(
    histories: list[dict[str, object]],
    policy_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))

    ppo_mask = (
        (policy_summary["phase"] == "ppo_eval")
        & (policy_summary["w_bo"] == WINNING_COMBO["w_bo"])
        & (policy_summary["w_cost"] == WINNING_COMBO["w_cost"])
        & (policy_summary["w_disr"] == WINNING_COMBO["w_disr"])
    )
    static_mask = (
        (policy_summary["phase"] == "static_screen")
        & (policy_summary["w_bo"] == WINNING_COMBO["w_bo"])
        & (policy_summary["w_cost"] == WINNING_COMBO["w_cost"])
        & (policy_summary["w_disr"] == WINNING_COMBO["w_disr"])
    )
    ppo_rows = policy_summary.loc[ppo_mask]
    static_rows = policy_summary.loc[static_mask]
    ppo_eval_mean = float(
        ppo_rows.loc[ppo_rows["policy"] == "ppo", "reward_total_mean"].iloc[0]
    )
    static_s2_mean = float(
        static_rows.loc[static_rows["policy"] == "static_s2", "reward_total_mean"].iloc[
            0
        ]
    )
    static_s1_mean = float(
        static_rows.loc[static_rows["policy"] == "static_s1", "reward_total_mean"].iloc[
            0
        ]
    )

    max_t = 0
    common_grid = np.linspace(0, max(h["timesteps"][-1] for h in histories), 200)
    interpolated: list[np.ndarray] = []
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for color, history in zip(colors, histories):
        timesteps = np.asarray(history["timesteps"], dtype=np.float64)
        rewards = np.asarray(history["rewards"], dtype=np.float64)
        max_t = max(max_t, int(timesteps[-1]))
        smoothed = rolling_mean(rewards.tolist(), window=10)
        ax.plot(
            timesteps,
            rewards,
            color=color,
            alpha=0.18,
            linewidth=1.0,
        )
        ax.plot(
            timesteps,
            smoothed,
            color=color,
            linewidth=2.0,
            label=f"Seed {history['seed']} (rolling mean)",
        )
        valid = ~np.isnan(smoothed)
        if valid.sum() >= 2:
            interpolated.append(
                np.interp(
                    common_grid,
                    timesteps[valid],
                    smoothed[valid],
                    left=np.nan,
                    right=smoothed[valid][-1],
                )
            )

    if interpolated:
        stacked = np.vstack(interpolated)
        mean_curve = np.nanmean(stacked, axis=0)
        ax.plot(
            common_grid,
            mean_curve,
            color="black",
            linewidth=2.8,
            linestyle="--",
            label="Mean rolling reward",
        )

    ax.axhline(
        static_s1_mean,
        color="#8c8c8c",
        linestyle=":",
        linewidth=1.8,
        label="Static S1 mean",
    )
    ax.axhline(
        static_s2_mean,
        color="#d62728",
        linestyle=":",
        linewidth=1.8,
        label="Static S2 mean",
    )
    ax.axhline(
        ppo_eval_mean,
        color="#9467bd",
        linestyle="-.",
        linewidth=1.8,
        label="PPO eval mean",
    )
    ax.set_title(
        "PPO training reward under control_v1 winner "
        "(diagnostic curve, not cross-validation loss)"
    )
    ax.set_xlabel("Training timesteps")
    ax.set_ylabel("Episode reward")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_policy_comparison(policy_summary: pd.DataFrame, output_path: Path) -> None:
    ppo_mask = (
        (policy_summary["phase"] == "ppo_eval")
        & (policy_summary["w_bo"] == WINNING_COMBO["w_bo"])
        & (policy_summary["w_cost"] == WINNING_COMBO["w_cost"])
        & (policy_summary["w_disr"] == WINNING_COMBO["w_disr"])
        & (policy_summary["policy"] == "ppo")
    )
    static_mask = (
        (policy_summary["phase"] == "static_screen")
        & (policy_summary["w_bo"] == WINNING_COMBO["w_bo"])
        & (policy_summary["w_cost"] == WINNING_COMBO["w_cost"])
        & (policy_summary["w_disr"] == WINNING_COMBO["w_disr"])
        & (policy_summary["policy"].isin(["static_s1", "static_s2", "static_s3"]))
    )
    df = pd.concat(
        [policy_summary.loc[static_mask], policy_summary.loc[ppo_mask]],
        ignore_index=True,
    ).copy()
    order = ["static_s1", "static_s2", "static_s3", "ppo"]
    df["policy"] = pd.Categorical(df["policy"], categories=order, ordered=True)
    df = df.sort_values("policy")

    x = np.arange(len(df))
    width = 0.24

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    axes[0].bar(x, df["reward_total_mean"], width=width, color="#4c78a8")
    axes[0].set_title("Control reward")
    axes[0].set_xticks(x, df["policy"], rotation=20)
    axes[0].set_ylabel("Episode mean")

    axes[1].bar(x, df["fill_rate_mean"], width=width, color="#59a14f")
    axes[1].set_title("Fill rate")
    axes[1].set_xticks(x, df["policy"], rotation=20)
    axes[1].set_ylim(0.0, 1.0)

    axes[2].bar(x, df["backorder_rate_mean"], width=width, color="#e15759")
    axes[2].set_title("Backorder rate")
    axes[2].set_xticks(x, df["policy"], rotation=20)
    axes[2].set_ylim(0.0, max(0.4, float(df["backorder_rate_mean"].max()) * 1.15))

    for ax in axes:
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Winning control_v1 regime: PPO vs fixed baselines")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_action_mix(
    policy_summary: pd.DataFrame, action_trace: dict[str, object], output_path: Path
) -> None:
    mask = (
        (policy_summary["phase"] == "ppo_eval")
        & (policy_summary["w_bo"] == WINNING_COMBO["w_bo"])
        & (policy_summary["w_cost"] == WINNING_COMBO["w_cost"])
        & (policy_summary["w_disr"] == WINNING_COMBO["w_disr"])
        & (
            policy_summary["policy"].isin(
                ["static_s1", "static_s2", "static_s3", "ppo"]
            )
        )
    )
    df = policy_summary.loc[mask].copy()
    order = ["static_s1", "static_s2", "static_s3", "ppo"]
    df["policy"] = pd.Categorical(df["policy"], categories=order, ordered=True)
    df = df.sort_values("policy")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(len(df))
    s1 = df["pct_steps_S1_mean"].to_numpy()
    s2 = df["pct_steps_S2_mean"].to_numpy()
    s3 = df["pct_steps_S3_mean"].to_numpy()
    axes[0].bar(x, s1, color="#4c78a8", label="S1")
    axes[0].bar(x, s2, bottom=s1, color="#f28e2b", label="S2")
    axes[0].bar(x, s3, bottom=s1 + s2, color="#59a14f", label="S3")
    axes[0].set_xticks(x, df["policy"], rotation=20)
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("% of decision epochs")
    axes[0].set_title("Shift mix by policy")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)

    per_seed = action_trace["per_seed"]
    seeds = [str(item["seed"]) for item in per_seed]
    seed_s1 = [item["shift_pct"]["S1"] for item in per_seed]
    seed_s2 = [item["shift_pct"]["S2"] for item in per_seed]
    seed_s3 = [item["shift_pct"]["S3"] for item in per_seed]
    seed_x = np.arange(len(seeds))
    axes[1].bar(seed_x, seed_s1, color="#4c78a8", label="S1")
    axes[1].bar(seed_x, seed_s2, bottom=seed_s1, color="#f28e2b", label="S2")
    axes[1].bar(
        seed_x,
        seed_s3,
        bottom=np.asarray(seed_s1) + np.asarray(seed_s2),
        color="#59a14f",
        label="S3",
    )
    axes[1].set_xticks(seed_x, [f"Seed {seed}" for seed in seeds])
    axes[1].set_ylim(0, 100)
    axes[1].set_title("Winning PPO policy by evaluation seed")
    axes[1].grid(axis="y", alpha=0.25)

    fig.suptitle("Adaptive shift allocation under the winning control_v1 regime")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    policy_summary = pd.read_csv(args.benchmark_dir / "policy_summary.csv")
    with (args.benchmark_dir / "action_trace_wbo4_cost002.json").open(
        "r", encoding="utf-8"
    ) as file_obj:
        action_trace = json.load(file_obj)

    histories = train_with_logging(args)
    plot_training_curve(
        histories=histories,
        policy_summary=policy_summary,
        output_path=args.output_dir / "figure_1_training_reward_curve.png",
    )
    plot_policy_comparison(
        policy_summary=policy_summary,
        output_path=args.output_dir / "figure_2_policy_comparison.png",
    )
    plot_action_mix(
        policy_summary=policy_summary,
        action_trace=action_trace,
        output_path=args.output_dir / "figure_3_action_mix.png",
    )

    manifest = {
        "winning_combo": WINNING_COMBO,
        "seeds": args.seeds,
        "train_timesteps": args.train_timesteps,
        "figures": {
            "training_curve": str(
                args.output_dir / "figure_1_training_reward_curve.png"
            ),
            "policy_comparison": str(
                args.output_dir / "figure_2_policy_comparison.png"
            ),
            "action_mix": str(args.output_dir / "figure_3_action_mix.png"),
        },
    }
    with (args.output_dir / "manifest.json").open("w", encoding="utf-8") as file_obj:
        json.dump(manifest, file_obj, indent=2)


if __name__ == "__main__":
    main()
