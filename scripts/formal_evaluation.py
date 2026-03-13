#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
import os
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_agent import build_env_instance, ci95
from scripts.benchmark_control_reward import (
    HEURISTIC_DEFAULTS,
    HEURISTIC_POLICY_NAMES,
)

DEFAULT_BENCHMARK_ROOT = Path("outputs/benchmarks/ppo_shift_control_ret_thesis")
DEFAULT_OUTPUT_DIR = Path("outputs/evaluations/ppo_shift_control_ret_thesis_formal")
EVAL_EPISODE_SEED_OFFSET = 50_000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run formal evaluation for PPO benchmark artifacts."
    )
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=DEFAULT_BENCHMARK_ROOT,
        help="Directory containing per-seed PPO benchmark artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where formal evaluation CSV/JSON outputs are written.",
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=["trained", "random", "default", *HEURISTIC_POLICY_NAMES],
        default=["trained", "random", "default", *HEURISTIC_POLICY_NAMES],
        help="Policies to evaluate under the common protocol.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=None,
        help="Override number of evaluation episodes. Defaults to benchmark config.",
    )
    return parser


def load_benchmark_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def load_training_log(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def build_args_namespace(config: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        quick=config.get("quick", False),
        timesteps=config["timesteps"],
        seed=config["seed"],
        env_variant=config["env_variant"],
        n_envs=config.get("n_envs", 1),
        step_size_hours=config["step_size_hours"],
        max_steps_per_episode=config["max_steps_per_episode"],
        learning_rate=config["learning_rate"],
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        n_epochs=config["n_epochs"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        clip_range=config["clip_range"],
        ent_coef=config["ent_coef"],
        eval_episodes=config["eval_episodes"],
        random_episodes=config["random_episodes"],
        output_dir=Path("."),
        year_basis=config["year_basis"],
        reward_mode=config["reward_mode"],
        risk_level=config["risk_level"],
        rt_alpha=config["rt_alpha"],
        rt_beta=config["rt_beta"],
        rt_gamma_reward=config["rt_gamma_reward"],
        rt_recovery_scale=config["rt_recovery_scale"],
        rt_inventory_scale=config["rt_inventory_scale"],
        shift_delta=config["shift_delta"],
        stochastic_pt=config["stochastic_pt"],
    )


def create_vec_norm(
    vec_norm_path: Path, args: SimpleNamespace
) -> tuple[VecNormalize, PPO]:
    def env_factory() -> Any:
        return build_env_instance(args)

    vec_norm = VecNormalize.load(str(vec_norm_path), DummyVecEnv([env_factory]))
    vec_norm.training = False
    vec_norm.norm_reward = False
    model = PPO.load(str(vec_norm_path.parent / "ppo_mfsc_baseline.zip"), device="cpu")
    return vec_norm, model


def finalize_episode_metrics(
    *,
    policy: str,
    seed: int,
    episode: int,
    eval_seed: int,
    steps: int,
    reward_total: float,
    delivered_total: float,
    demanded_total: float,
    backorder_qty_total: float,
    disruption_hours_total: float,
    inventory_values: list[float],
    ret_values: list[float],
    step_fill_rates: list[float],
    shift_cost_values: list[float],
    shift_counts: Counter[int],
    ret_case_counts: Counter[str],
) -> dict[str, Any]:
    fill_rate_episode = (
        1.0 - (backorder_qty_total / demanded_total) if demanded_total > 0 else 1.0
    )
    service_loss_episode = (
        backorder_qty_total / demanded_total if demanded_total > 0 else 0.0
    )
    mean_ret = float(np.mean(ret_values)) if ret_values else float("nan")
    mean_step_fill_rate = (
        float(np.mean(step_fill_rates)) if step_fill_rates else float("nan")
    )
    mean_shift_cost = (
        float(np.mean(shift_cost_values)) if shift_cost_values else float("nan")
    )
    avg_inventory = float(np.mean(inventory_values)) if inventory_values else 0.0

    def pct(count: int) -> float:
        return 100.0 * count / max(1, steps)

    return {
        "policy": policy,
        "seed": seed,
        "episode": episode,
        "eval_seed": eval_seed,
        "steps": steps,
        "reward_total": reward_total,
        "delivered_total": delivered_total,
        "demanded_total": demanded_total,
        "backorder_qty_total": backorder_qty_total,
        "fill_rate_episode": fill_rate_episode,
        "service_loss_episode": service_loss_episode,
        "disruption_hours_total": disruption_hours_total,
        "avg_inventory": avg_inventory,
        "mean_ReT": mean_ret,
        "mean_step_fill_rate": mean_step_fill_rate,
        "mean_shift_cost": mean_shift_cost,
        "pct_steps_S1": pct(shift_counts[1]),
        "pct_steps_S2": pct(shift_counts[2]),
        "pct_steps_S3": pct(shift_counts[3]),
        "pct_fill_rate_only": pct(ret_case_counts["fill_rate_only"]),
        "pct_autotomy": pct(ret_case_counts["autotomy"]),
        "pct_recovery": pct(ret_case_counts["recovery"]),
        "pct_non_recovery": pct(ret_case_counts["non_recovery"]),
        "pct_no_demand": pct(ret_case_counts["no_demand"]),
    }


def evaluate_policy(
    policy: str,
    *,
    args: SimpleNamespace,
    eval_episodes: int,
    vec_norm: VecNormalize | None,
    model: PPO | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for episode_idx in range(eval_episodes):
        eval_seed = args.seed + EVAL_EPISODE_SEED_OFFSET + episode_idx
        env = build_env_instance(args)
        obs_raw, _ = env.reset(seed=eval_seed)
        done, truncated = False, False
        reward_total = 0.0
        delivered_total = 0.0
        demanded_total = 0.0
        backorder_qty_total = 0.0
        disruption_hours_total = 0.0
        inventory_values: list[float] = []
        ret_values: list[float] = []
        step_fill_rates: list[float] = []
        shift_cost_values: list[float] = []
        shift_counts: Counter[int] = Counter()
        ret_case_counts: Counter[str] = Counter()
        steps = 0

        heuristic = HEURISTIC_DEFAULTS.get(policy)
        if heuristic is not None:
            heuristic.reset()
        prev_info: dict[str, Any] = {}

        while not (done or truncated):
            if policy == "trained":
                if vec_norm is None or model is None:
                    raise ValueError("trained policy requires model and vec_norm")
                obs_norm = vec_norm.normalize_obs(
                    np.asarray(obs_raw, dtype=np.float32)[None, :]
                )
                action, _ = model.predict(obs_norm, deterministic=True)
                action_to_apply = action[0]
            elif heuristic is not None:
                action_to_apply = heuristic(obs_raw, prev_info)
            elif policy == "default":
                action_to_apply = np.zeros(env.action_space.shape, dtype=np.float32)
            else:
                action_to_apply = env.action_space.sample()

            obs_raw, reward, done, truncated, info = env.step(action_to_apply)
            prev_info = info
            reward_total += float(reward)
            delivered_total += float(info.get("new_delivered", 0.0))
            demanded_total += float(info.get("new_demanded", 0.0))
            backorder_qty_total += float(info.get("new_backorder_qty", 0.0))
            disruption_hours_total += float(info.get("step_disruption_hours", 0.0))
            inventory_values.append(float(info.get("total_inventory", 0.0)))
            if "ReT_raw" in info:
                ret_values.append(float(info["ReT_raw"]))
            shift_cost_values.append(float(info.get("shift_cost_linear", 0.0)))
            shift_counts[int(info.get("shifts_active", 1))] += 1
            ret_case = (
                info.get("ret_components", {}).get("ret_case")
                if isinstance(info.get("ret_components"), dict)
                else None
            )
            step_fill_rate = (
                info.get("ret_components", {}).get("fill_rate")
                if isinstance(info.get("ret_components"), dict)
                else None
            )
            if step_fill_rate is not None:
                step_fill_rates.append(float(step_fill_rate))
            if ret_case is not None:
                ret_case_counts[str(ret_case)] += 1
            steps += 1

        rows.append(
            finalize_episode_metrics(
                policy=policy,
                seed=args.seed,
                episode=episode_idx + 1,
                eval_seed=eval_seed,
                steps=steps,
                reward_total=reward_total,
                delivered_total=delivered_total,
                demanded_total=demanded_total,
                backorder_qty_total=backorder_qty_total,
                disruption_hours_total=disruption_hours_total,
                inventory_values=inventory_values,
                ret_values=ret_values,
                step_fill_rates=step_fill_rates,
                shift_cost_values=shift_cost_values,
                shift_counts=shift_counts,
                ret_case_counts=ret_case_counts,
            )
        )
    return rows


def aggregate_seed_metrics(
    episode_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in episode_rows:
        grouped.setdefault((str(row["policy"]), int(row["seed"])), []).append(row)

    metrics = [
        "reward_total",
        "delivered_total",
        "demanded_total",
        "backorder_qty_total",
        "fill_rate_episode",
        "service_loss_episode",
        "disruption_hours_total",
        "avg_inventory",
        "mean_ReT",
        "mean_step_fill_rate",
        "mean_shift_cost",
        "pct_steps_S1",
        "pct_steps_S2",
        "pct_steps_S3",
        "pct_fill_rate_only",
        "pct_autotomy",
        "pct_recovery",
        "pct_non_recovery",
        "pct_no_demand",
    ]

    rows: list[dict[str, Any]] = []
    for (policy, seed), items in sorted(grouped.items()):
        row: dict[str, Any] = {
            "policy": policy,
            "seed": seed,
            "episodes": len(items),
        }
        for metric in metrics:
            values = [float(item[metric]) for item in items]
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            )
            ci_low, ci_high = ci95(values)
            row[f"{metric}_ci95_low"] = ci_low
            row[f"{metric}_ci95_high"] = ci_high
        rows.append(row)
    return rows


def aggregate_policy_metrics(
    seed_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in seed_rows:
        grouped.setdefault(str(row["policy"]), []).append(row)

    primary_metrics = [
        "reward_total_mean",
        "fill_rate_episode_mean",
        "service_loss_episode_mean",
        "backorder_qty_total_mean",
        "disruption_hours_total_mean",
        "avg_inventory_mean",
        "mean_ReT_mean",
        "mean_step_fill_rate_mean",
        "mean_shift_cost_mean",
        "pct_steps_S1_mean",
        "pct_steps_S2_mean",
        "pct_steps_S3_mean",
        "pct_fill_rate_only_mean",
        "pct_autotomy_mean",
        "pct_recovery_mean",
        "pct_non_recovery_mean",
    ]

    out: dict[str, Any] = {}
    for policy, items in sorted(grouped.items()):
        summary: dict[str, Any] = {"seed_count": len(items)}
        for metric in primary_metrics:
            values = [float(item[metric]) for item in items]
            ci_low, ci_high = ci95(values)
            summary[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
            }
        out[policy] = summary
    return out


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_comparison_rows(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    policies = sorted({str(row["policy"]) for row in seed_rows})
    rows: list[dict[str, Any]] = []
    for policy in policies:
        policy_rows = [row for row in seed_rows if row["policy"] == policy]
        reward_values = [float(row["reward_total_mean"]) for row in policy_rows]
        ret_values = [float(row["mean_ReT_mean"]) for row in policy_rows]
        fill_values = [float(row["fill_rate_episode_mean"]) for row in policy_rows]
        step_fill_values = [
            float(row["mean_step_fill_rate_mean"]) for row in policy_rows
        ]
        backorder_values = [
            float(row["backorder_qty_total_mean"]) for row in policy_rows
        ]
        shift_cost_values = [float(row["mean_shift_cost_mean"]) for row in policy_rows]
        s1_values = [float(row["pct_steps_S1_mean"]) for row in policy_rows]
        s2_values = [float(row["pct_steps_S2_mean"]) for row in policy_rows]
        s3_values = [float(row["pct_steps_S3_mean"]) for row in policy_rows]
        reward_ci_low, reward_ci_high = ci95(reward_values)
        rows.append(
            {
                "policy": policy,
                "seed_count": len(policy_rows),
                "reward_mean": float(np.mean(reward_values)),
                "reward_ci95_low": reward_ci_low,
                "reward_ci95_high": reward_ci_high,
                "mean_ReT": float(np.mean(ret_values)),
                "fill_rate": float(np.mean(fill_values)),
                "mean_step_fill_rate": float(np.mean(step_fill_values)),
                "backorder_qty": float(np.mean(backorder_values)),
                "mean_shift_cost": float(np.mean(shift_cost_values)),
                "pct_steps_S1": float(np.mean(s1_values)),
                "pct_steps_S2": float(np.mean(s2_values)),
                "pct_steps_S3": float(np.mean(s3_values)),
            }
        )
    return rows


def main() -> None:
    parser = build_parser()
    cli_args = parser.parse_args()
    summary = load_benchmark_summary(cli_args.benchmark_root / "benchmark_summary.json")
    eval_episodes = cli_args.eval_episodes or int(summary["config"]["eval_episodes"])
    cli_args.output_dir.mkdir(parents=True, exist_ok=True)

    episode_rows: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {
        "benchmark_root": str(cli_args.benchmark_root),
        "evaluated_policies": cli_args.policies,
        "eval_episodes": eval_episodes,
        "episode_seed_offset": EVAL_EPISODE_SEED_OFFSET,
        "benchmark_config": summary["config"],
        "seeds": [],
    }

    for run in summary["runs"]:
        seed_dir = Path(run["output_dir"])
        training_log = load_training_log(seed_dir / "training_log.json")
        args = build_args_namespace(training_log["config"])
        metadata["seeds"].append(
            {
                "seed": args.seed,
                "model": training_log["artifacts"]["model"],
                "vec_normalize": training_log["artifacts"]["vec_normalize"],
            }
        )
        vec_norm: VecNormalize | None = None
        model: PPO | None = None
        if "trained" in cli_args.policies:
            vec_norm, model = create_vec_norm(
                seed_dir / "vec_normalize.pkl",
                args,
            )
        for policy in cli_args.policies:
            rows = evaluate_policy(
                policy,
                args=args,
                eval_episodes=eval_episodes,
                vec_norm=vec_norm,
                model=model,
            )
            episode_rows.extend(rows)

    seed_rows = aggregate_seed_metrics(episode_rows)
    policy_summary = aggregate_policy_metrics(seed_rows)
    comparison_rows = build_comparison_rows(seed_rows)

    save_csv(cli_args.output_dir / "episode_metrics.csv", episode_rows)
    save_csv(cli_args.output_dir / "seed_metrics.csv", seed_rows)
    save_csv(cli_args.output_dir / "comparison_table.csv", comparison_rows)

    aggregate_payload = {
        "metadata": metadata,
        "policy_summary": policy_summary,
    }
    with (cli_args.output_dir / "aggregate_metrics.json").open(
        "w", encoding="utf-8"
    ) as file_obj:
        json.dump(aggregate_payload, file_obj, indent=2)

    print(f"Wrote formal evaluation artifacts to {cli_args.output_dir}")


if __name__ == "__main__":
    main()
