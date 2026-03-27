#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.external_env_interface import make_shift_control_env, run_episodes

STATIC_POLICIES = ("static_s1", "static_s2", "static_s3")
FIXED_POLICY_ACTIONS: dict[str, np.ndarray] = {
    "static_s1": np.array([0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32),
    "static_s2": np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    "static_s3": np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
}
POLICY_METRICS = (
    "reward_total",
    "fill_rate",
    "backorder_rate",
    "ret_thesis_corrected_total",
    "pct_steps_S1",
    "pct_steps_S2",
    "pct_steps_S3",
)
FILL_RATE_COMPETITIVE_TOL = 0.01
BACKORDER_COMPETITIVE_TOL = 0.01
RET_COMPETITIVE_TOL = 1.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a falsification-oriented PPO benchmark on thesis-aligned experimental reward lanes."
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[11, 22, 33],
        help="Random seeds to evaluate.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=10_000,
        help="Training timesteps per seed.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Directory where per-seed runs and benchmark artifacts are stored.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
        help="Evaluation episodes for PPO and static baselines.",
    )
    parser.add_argument(
        "--random-episodes",
        type=int,
        default=10,
        help="Random baseline episodes passed to train_agent.py.",
    )
    parser.add_argument(
        "--step-size-hours",
        type=float,
        default=168.0,
        help="Step size passed to train_agent.py.",
    )
    parser.add_argument(
        "--risk-level",
        choices=["current", "increased", "severe"],
        default="current",
        help="Risk setting passed to train_agent.py.",
    )
    parser.add_argument(
        "--reward-mode",
        choices=["ReT_thesis", "ReT_corrected", "ReT_corrected_cost"],
        default="ReT_thesis",
        help="Thesis-aligned reward family to benchmark.",
    )
    parser.add_argument(
        "--shift-delta",
        dest="shift_deltas",
        action="append",
        type=float,
        default=None,
        help="Shift-cost delta. Pass multiple times to benchmark multiple deltas.",
    )
    parser.add_argument(
        "--stochastic-pt",
        action="store_true",
        help="Enable stochastic processing times.",
    )
    return parser


def mean_std(values: list[float]) -> tuple[float, float]:
    mean = float(statistics.fmean(values))
    std = float(statistics.stdev(values)) if len(values) > 1 else 0.0
    return mean, std


def ci95(values: list[float]) -> tuple[float, float]:
    if len(values) < 2:
        value = float(values[0]) if values else float("nan")
        return value, value
    arr = np.asarray(values, dtype=np.float64)
    half = 1.96 * arr.std(ddof=1) / np.sqrt(len(arr))
    mean = float(arr.mean())
    return mean - float(half), mean + float(half)


def default_output_root(reward_mode: str) -> Path:
    if reward_mode == "ReT_corrected_cost":
        return Path("outputs/benchmarks/ppo_shift_control_ret_corrected_cost")
    if reward_mode == "ReT_corrected":
        return Path("outputs/benchmarks/ppo_shift_control_ret_corrected")
    return Path("outputs/benchmarks/ppo_shift_control_ret_thesis")


def canonical_shift_deltas(args: argparse.Namespace) -> list[float]:
    if args.shift_deltas:
        return [float(value) for value in args.shift_deltas]
    return [0.06]


def static_policy_action(policy: str) -> np.ndarray:
    if policy not in FIXED_POLICY_ACTIONS:
        raise ValueError(f"Unsupported fixed policy {policy!r}.")
    return FIXED_POLICY_ACTIONS[policy].copy()


def build_env_kwargs(args: argparse.Namespace, shift_delta: float) -> dict[str, Any]:
    return {
        "reward_mode": args.reward_mode,
        "step_size_hours": args.step_size_hours,
        "risk_level": args.risk_level,
        "stochastic_pt": args.stochastic_pt,
        "rt_delta": shift_delta,
    }


class LoadedPPOPolicy:
    """Adapt a PPO+VecNormalize pair to the run_episodes policy callable."""

    def __init__(self, model: PPO, vec_norm: VecNormalize) -> None:
        self.model = model
        self.vec_norm = vec_norm
        self.vec_norm.training = False

    def __call__(self, obs: np.ndarray, info: dict[str, Any]) -> np.ndarray:
        del info
        obs_norm = self.vec_norm.normalize_obs(
            np.asarray(obs, dtype=np.float32)[None, :]
        )
        action, _ = self.model.predict(obs_norm, deterministic=True)
        return np.asarray(action[0], dtype=np.float32)


def make_vec_normalize(run_dir: Path, env_kwargs: dict[str, Any]) -> VecNormalize:
    vec_env = DummyVecEnv([lambda: Monitor(make_shift_control_env(**env_kwargs))])
    vec_norm = VecNormalize.load(str(run_dir / "vec_normalize.pkl"), vec_env)
    vec_norm.training = False
    return vec_norm


def summarize_episode_rows(
    rows: list[dict[str, Any]],
    *,
    phase: str,
    policy: str,
    shift_delta: float,
    seed: int,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "phase": phase,
        "policy": policy,
        "shift_delta": shift_delta,
        "seed": seed,
        "episode_count": len(rows),
    }
    for metric in POLICY_METRICS:
        values = [float(row[metric]) for row in rows]
        summary[f"{metric}_mean"] = float(np.mean(values))
    return summary


def aggregate_policy_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (float(row["shift_delta"]), str(row["policy"]))
        grouped.setdefault(key, []).append(row)

    aggregated: list[dict[str, Any]] = []
    for (shift_delta, policy), bucket in sorted(grouped.items()):
        out: dict[str, Any] = {
            "policy": policy,
            "shift_delta": shift_delta,
            "seed_count": len(bucket),
        }
        for metric in POLICY_METRICS:
            values = [float(row[f"{metric}_mean"]) for row in bucket]
            mean, std = mean_std(values)
            ci_low, ci_high = ci95(values)
            out[f"{metric}_mean"] = mean
            out[f"{metric}_std"] = std
            out[f"{metric}_ci95_low"] = ci_low
            out[f"{metric}_ci95_high"] = ci_high
        aggregated.append(out)
    return aggregated


def best_static_row(policy_summary: list[dict[str, Any]], shift_delta: float) -> dict[str, Any]:
    static_rows = [
        row
        for row in policy_summary
        if float(row["shift_delta"]) == shift_delta and row["policy"] in STATIC_POLICIES
    ]
    if not static_rows:
        raise ValueError(f"No static rows found for shift_delta={shift_delta}.")
    return max(
        static_rows,
        key=lambda row: (
            float(row["ret_thesis_corrected_total_mean"]),
            float(row["fill_rate_mean"]),
            -float(row["backorder_rate_mean"]),
        ),
    )


def build_decision_table(policy_summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    decision_rows: list[dict[str, Any]] = []
    shift_deltas = sorted({float(row["shift_delta"]) for row in policy_summary})
    for shift_delta in shift_deltas:
        ppo_row = next(
            row
            for row in policy_summary
            if float(row["shift_delta"]) == shift_delta and row["policy"] == "ppo"
        )
        static_row = best_static_row(policy_summary, shift_delta)
        collapse_s1 = float(ppo_row["pct_steps_S1_mean"]) > 90.0
        collapse_s3 = float(ppo_row["pct_steps_S3_mean"]) > 90.0
        fill_rate_gap = float(ppo_row["fill_rate_mean"]) - float(static_row["fill_rate_mean"])
        backorder_gap = float(ppo_row["backorder_rate_mean"]) - float(
            static_row["backorder_rate_mean"]
        )
        ret_gap = float(ppo_row["ret_thesis_corrected_total_mean"]) - float(
            static_row["ret_thesis_corrected_total_mean"]
        )
        mixed_policy = max(
            float(ppo_row["pct_steps_S1_mean"]),
            float(ppo_row["pct_steps_S2_mean"]),
            float(ppo_row["pct_steps_S3_mean"]),
        ) < 90.0
        service_competitive = fill_rate_gap >= -FILL_RATE_COMPETITIVE_TOL
        backlog_competitive = backorder_gap <= BACKORDER_COMPETITIVE_TOL
        ret_competitive = ret_gap >= -RET_COMPETITIVE_TOL

        if collapse_s1:
            verdict = "kill_collapse_s1"
        elif collapse_s3:
            verdict = "kill_collapse_s3"
        elif not service_competitive:
            verdict = "kill_service_gap"
        elif not backlog_competitive:
            verdict = "kill_backorder_gap"
        elif not ret_competitive:
            verdict = "kill_ret_gap"
        elif not mixed_policy:
            verdict = "kill_no_mixed_policy"
        else:
            verdict = "survives"

        decision_rows.append(
            {
                "shift_delta": shift_delta,
                "best_static_policy": static_row["policy"],
                "ppo_fill_rate_mean": float(ppo_row["fill_rate_mean"]),
                "best_static_fill_rate_mean": float(static_row["fill_rate_mean"]),
                "fill_rate_gap": fill_rate_gap,
                "ppo_backorder_rate_mean": float(ppo_row["backorder_rate_mean"]),
                "best_static_backorder_rate_mean": float(
                    static_row["backorder_rate_mean"]
                ),
                "backorder_rate_gap": backorder_gap,
                "ppo_ret_thesis_corrected_total_mean": float(
                    ppo_row["ret_thesis_corrected_total_mean"]
                ),
                "best_static_ret_thesis_corrected_total_mean": float(
                    static_row["ret_thesis_corrected_total_mean"]
                ),
                "ret_thesis_corrected_gap": ret_gap,
                "ppo_pct_steps_S1_mean": float(ppo_row["pct_steps_S1_mean"]),
                "ppo_pct_steps_S2_mean": float(ppo_row["pct_steps_S2_mean"]),
                "ppo_pct_steps_S3_mean": float(ppo_row["pct_steps_S3_mean"]),
                "collapse_s1": collapse_s1,
                "collapse_s3": collapse_s3,
                "mixed_policy": mixed_policy,
                "service_competitive": service_competitive,
                "backlog_competitive": backlog_competitive,
                "ret_competitive": ret_competitive,
                "verdict": verdict,
            }
        )
    return decision_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def render_decision_memo(
    *,
    args: argparse.Namespace,
    decision_table: list[dict[str, Any]],
    policy_summary: list[dict[str, Any]],
) -> str:
    lines = [
        f"# {args.reward_mode} falsification memo",
        "",
        "## Benchmark config",
        f"- reward_mode: `{args.reward_mode}`",
        f"- risk_level: `{args.risk_level}`",
        f"- stochastic_pt: `{args.stochastic_pt}`",
        f"- timesteps: `{args.timesteps}`",
        f"- seeds: `{args.seeds}`",
        f"- eval_episodes: `{args.eval_episodes}`",
        "",
        "## Decision table",
        "",
        "| delta | best static | PPO fill rate | static fill rate | PPO ReTcorr | static ReTcorr | PPO shifts S1/S2/S3 | verdict |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in decision_table:
        lines.append(
            "| "
            f"{row['shift_delta']:.3f} | "
            f"`{row['best_static_policy']}` | "
            f"{row['ppo_fill_rate_mean']:.3f} | "
            f"{row['best_static_fill_rate_mean']:.3f} | "
            f"{row['ppo_ret_thesis_corrected_total_mean']:.3f} | "
            f"{row['best_static_ret_thesis_corrected_total_mean']:.3f} | "
            f"{row['ppo_pct_steps_S1_mean']:.1f}/{row['ppo_pct_steps_S2_mean']:.1f}/{row['ppo_pct_steps_S3_mean']:.1f} | "
            f"`{row['verdict']}` |"
        )
    surviving = [row for row in decision_table if row["verdict"] == "survives"]
    lines.extend(
        [
            "",
            "## Binary conclusion",
            (
                "- `seguir escalando`"
                if surviving
                else "- `matar la lane y mantener control_v1 como surrogate operacional`"
            ),
        ]
    )
    if surviving:
        lines.append(
            "- survived deltas: "
            + ", ".join(f"`{row['shift_delta']:.3f}`" for row in surviving)
        )
    else:
        lines.append(
            "- no tested delta met the non-collapse, service-competitive, and ReT-competitive criteria."
        )
    return "\n".join(lines) + "\n"


def evaluate_seed(
    args: argparse.Namespace,
    *,
    seed: int,
    shift_delta: float,
    run_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    env_kwargs = build_env_kwargs(args, shift_delta)
    model = PPO.load(str(run_dir / "ppo_mfsc_baseline.zip"))
    vec_norm = make_vec_normalize(run_dir, env_kwargs)
    ppo_policy = LoadedPPOPolicy(model, vec_norm)

    episode_rows = run_episodes(
        ppo_policy,
        n_episodes=args.eval_episodes,
        seed=seed + 50_000,
        env_kwargs=env_kwargs,
        policy_name="ppo",
    )
    for policy in STATIC_POLICIES:
        episode_rows.extend(
            run_episodes(
                lambda obs, info, policy=policy: static_policy_action(policy),
                n_episodes=args.eval_episodes,
                seed=seed + 60_000,
                env_kwargs=env_kwargs,
                policy_name=policy,
            )
        )

    seed_summary_rows = []
    for policy in ("ppo",) + STATIC_POLICIES:
        rows = [row for row in episode_rows if row["policy"] == policy]
        seed_summary_rows.append(
            summarize_episode_rows(
                rows,
                phase="eval",
                policy=policy,
                shift_delta=shift_delta,
                seed=seed,
            )
        )
    return seed_summary_rows, {
        "seed": seed,
        "shift_delta": shift_delta,
        "run_dir": str(run_dir),
    }


def train_seed(args: argparse.Namespace, *, seed: int, shift_delta: float, run_dir: Path) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "train_agent.py",
        "--timesteps",
        str(args.timesteps),
        "--seed",
        str(seed),
        "--env-variant",
        "shift_control",
        "--reward-mode",
        args.reward_mode,
        "--shift-delta",
        str(shift_delta),
        "--eval-episodes",
        str(args.eval_episodes),
        "--random-episodes",
        str(args.random_episodes),
        "--step-size-hours",
        str(args.step_size_hours),
        "--risk-level",
        args.risk_level,
        "--output-dir",
        str(run_dir),
    ]
    if args.stochastic_pt:
        command.append("--stochastic-pt")
    subprocess.run(command, check=True)
    with (run_dir / "training_log.json").open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    shift_deltas = canonical_shift_deltas(args)
    args.output_root = args.output_root or default_output_root(args.reward_mode)
    args.output_root.mkdir(parents=True, exist_ok=True)

    run_metadata: list[dict[str, Any]] = []
    seed_policy_rows: list[dict[str, Any]] = []
    training_logs: list[dict[str, Any]] = []

    for shift_delta in shift_deltas:
        for seed in args.seeds:
            run_dir = args.output_root / f"delta_{shift_delta:.3f}" / f"seed_{seed}"
            training_logs.append(
                train_seed(args, seed=seed, shift_delta=shift_delta, run_dir=run_dir)
            )
            rows, metadata = evaluate_seed(
                args,
                seed=seed,
                shift_delta=shift_delta,
                run_dir=run_dir,
            )
            seed_policy_rows.extend(rows)
            run_metadata.append(metadata)

    policy_summary = aggregate_policy_rows(seed_policy_rows)
    decision_table = build_decision_table(policy_summary)
    summary = {
        "benchmark": "ppo_shift_control_thesis_lane",
        "config": {
            "reward_mode": args.reward_mode,
            "shift_deltas": shift_deltas,
            "seeds": args.seeds,
            "timesteps": args.timesteps,
            "eval_episodes": args.eval_episodes,
            "random_episodes": args.random_episodes,
            "step_size_hours": args.step_size_hours,
            "risk_level": args.risk_level,
            "stochastic_pt": args.stochastic_pt,
        },
        "runs": run_metadata,
        "policy_summary": policy_summary,
        "decision_table": decision_table,
    }

    write_csv(args.output_root / "seed_policy_summary.csv", seed_policy_rows)
    write_csv(args.output_root / "policy_summary.csv", policy_summary)
    write_csv(args.output_root / "decision_table.csv", decision_table)
    with (args.output_root / "summary.json").open("w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, indent=2)
    with (args.output_root / "training_logs_index.json").open(
        "w", encoding="utf-8"
    ) as file_obj:
        json.dump(training_logs, file_obj, indent=2)
    (args.output_root / "decision_memo.md").write_text(
        render_decision_memo(
            args=args,
            decision_table=decision_table,
            policy_summary=policy_summary,
        ),
        encoding="utf-8",
    )
    print(f"Wrote thesis-lane benchmark artifacts to {args.output_root}")


if __name__ == "__main__":
    main()
