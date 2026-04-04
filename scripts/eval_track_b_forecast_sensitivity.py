#!/usr/bin/env python3
"""Evaluate whether frozen Track B PPO depends on explicit forecast channels.

This is a fast, no-retraining sensitivity check over the frozen PPO bundle:

1. `full`: original v7 observations
2. `zeroed`: explicit 48h/168h forecast channels forced to zero
3. `scrambled`: explicit 48h/168h forecast channels replaced by random draws
   from an empirical forecast bank collected under the same DES risk profile

If PPO degrades materially under `zeroed`/`scrambled`, that is evidence that
the explicit forecast channels are contributing meaningfully. If performance
holds, Track B is mostly adaptive to other observable state.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import gymnasium as gym
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.audit_track_b_all_rewards import (  # noqa: E402
    DEFAULT_EVAL_EPISODES,
    DEFAULT_MAX_STEPS,
    DEFAULT_PPO_BUNDLE,
    DEFAULT_SEEDS,
    DEFAULT_STEP_SIZE_HOURS,
    EVAL_EPISODE_SEED_OFFSET,
    PAPER_REFERENCE_REWARD_MODE,
    RET_CASES,
    LearnedBundle,
    aggregate_policy_rows,
    build_ret_seq_audit_terms,
    finalize_episode_row,
    load_model,
    load_vec_normalize,
    save_csv,
    validate_bundle,
)
from scripts.run_track_b_observation_ablation import (  # noqa: E402
    FORECAST_FIELD_NAMES,
)
from scripts.run_track_b_smoke import (  # noqa: E402
    StaticPolicySpec,
    build_static_policy_action,
    extract_downstream_multipliers,
)
from supply_chain.external_env_interface import (  # noqa: E402
    get_episode_terminal_metrics,
    get_observation_fields,
    make_track_b_env,
)


FORECAST_BANK_STATIC = StaticPolicySpec(
    label="s2_d1.00",
    assembly_shifts=2,
    downstream_multiplier=1.0,
)
SUMMARY_COLUMNS = (
    "forecast_condition",
    "fill_rate_mean",
    "backorder_rate_mean",
    "order_level_ret_mean_mean",
    "service_continuity_step_mean_mean",
    "backlog_containment_step_mean_mean",
    "adaptive_efficiency_step_mean_mean",
    "pct_ret_case_autotomy_mean",
    "pct_ret_case_recovery_mean",
    "pct_ret_case_non_recovery_mean",
    "pct_steps_S1_mean",
    "pct_steps_S2_mean",
    "pct_steps_S3_mean",
    "op10_multiplier_step_mean_mean",
    "op12_multiplier_step_mean_mean",
)


@dataclass(frozen=True)
class ForecastCondition:
    label: str
    wrapper: type[gym.ObservationWrapper] | None


class ForecastZeroWrapper(gym.ObservationWrapper):
    """Zero only the explicit forecast channels while preserving shape."""

    def __init__(self, env: gym.Env[np.ndarray, np.ndarray]) -> None:
        super().__init__(env)
        fields = tuple(get_observation_fields("v7"))
        self._forecast_indices = tuple(fields.index(name) for name in FORECAST_FIELD_NAMES)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        masked = np.array(observation, dtype=np.float32, copy=True)
        for idx in self._forecast_indices:
            masked[idx] = 0.0
        return masked


class ForecastScrambleWrapper(gym.Wrapper[np.ndarray, np.ndarray, np.ndarray, np.ndarray]):
    """Replace explicit forecasts with random draws from an empirical bank."""

    def __init__(
        self,
        env: gym.Env[np.ndarray, np.ndarray],
        *,
        forecast_bank: np.ndarray,
    ) -> None:
        super().__init__(env)
        if forecast_bank.ndim != 2 or forecast_bank.shape[1] != 2:
            raise ValueError("forecast_bank must have shape [N, 2].")
        if forecast_bank.shape[0] == 0:
            raise ValueError("forecast_bank must be non-empty.")
        fields = tuple(get_observation_fields("v7"))
        self._idx_48h = fields.index(FORECAST_FIELD_NAMES[0])
        self._idx_168h = fields.index(FORECAST_FIELD_NAMES[1])
        self._forecast_bank = np.asarray(forecast_bank, dtype=np.float32)
        self._rng = np.random.default_rng(0)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is None:
            self._rng = np.random.default_rng()
        else:
            self._rng = np.random.default_rng(int(seed) + 17_371)
        obs, info = self.env.reset(seed=seed, options=options)
        return self._scramble(obs), info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._scramble(obs), reward, terminated, truncated, info

    def _scramble(self, observation: np.ndarray) -> np.ndarray:
        scrambled = np.array(observation, dtype=np.float32, copy=True)
        replacement = self._forecast_bank[
            int(self._rng.integers(0, self._forecast_bank.shape[0]))
        ]
        scrambled[self._idx_48h] = float(replacement[0])
        scrambled[self._idx_168h] = float(replacement[1])
        return scrambled


CONDITIONS: tuple[ForecastCondition, ...] = (
    ForecastCondition(label="full", wrapper=None),
    ForecastCondition(label="zeroed", wrapper=ForecastZeroWrapper),
    ForecastCondition(label="scrambled", wrapper=ForecastScrambleWrapper),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fast sensitivity check for explicit Track B forecast channels using "
            "the frozen PPO bundle."
        )
    )
    parser.add_argument(
        "--ppo-bundle",
        type=Path,
        default=DEFAULT_PPO_BUNDLE,
        help="Frozen PPO bundle directory.",
    )
    parser.add_argument(
        "--reward-mode",
        default=PAPER_REFERENCE_REWARD_MODE,
        help="Evaluation reward lens.",
    )
    parser.add_argument(
        "--risk-level",
        default="adaptive_benchmark_v2",
        help="Risk profile for evaluation and empirical forecast bank generation.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help="Frozen model seeds to evaluate.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=DEFAULT_EVAL_EPISODES,
        help="Evaluation episodes per seed and condition.",
    )
    parser.add_argument(
        "--bank-episodes",
        type=int,
        default=DEFAULT_EVAL_EPISODES,
        help="Episodes per seed used to build the empirical forecast bank.",
    )
    parser.add_argument(
        "--step-size-hours",
        type=float,
        default=DEFAULT_STEP_SIZE_HOURS,
        help="Decision cadence in hours.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help="Episode horizon in decision steps.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write the forecast-sensitivity bundle.",
    )
    return parser


def default_output_dir() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("outputs/track_b_benchmarks") / f"track_b_forecast_sensitivity_{timestamp}"


def build_env_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "reward_mode": args.reward_mode,
        "risk_level": args.risk_level,
        "observation_version": "v7",
        "step_size_hours": args.step_size_hours,
        "max_steps": args.max_steps,
    }


def wrap_env(
    env: gym.Env[np.ndarray, np.ndarray],
    condition: ForecastCondition,
    *,
    forecast_bank: np.ndarray | None,
) -> gym.Env[np.ndarray, np.ndarray]:
    if condition.wrapper is None:
        return env
    if condition.wrapper is ForecastScrambleWrapper:
        if forecast_bank is None:
            raise ValueError("Scrambled evaluation requires a forecast bank.")
        return condition.wrapper(env, forecast_bank=forecast_bank)
    return condition.wrapper(env)


def collect_forecast_bank(args: argparse.Namespace) -> np.ndarray:
    env_kwargs = build_env_kwargs(args)
    action = build_static_policy_action(FORECAST_BANK_STATIC)
    pairs: list[tuple[float, float]] = []
    fields = tuple(get_observation_fields("v7"))
    idx_48h = fields.index(FORECAST_FIELD_NAMES[0])
    idx_168h = fields.index(FORECAST_FIELD_NAMES[1])

    for seed in args.seeds:
        for episode_idx in range(args.bank_episodes):
            eval_seed = int(seed) + EVAL_EPISODE_SEED_OFFSET + episode_idx
            env = make_track_b_env(**env_kwargs)
            obs, _ = env.reset(seed=eval_seed)
            terminated = False
            truncated = False
            while not (terminated or truncated):
                pairs.append((float(obs[idx_48h]), float(obs[idx_168h])))
                obs, _, terminated, truncated, _ = env.step(action)
            env.close()

    bank = np.asarray(pairs, dtype=np.float32)
    if bank.size == 0:
        raise RuntimeError("Failed to collect any forecast values for the scramble bank.")
    return bank.reshape(-1, 2)


def evaluate_condition(
    *,
    args: argparse.Namespace,
    bundle: LearnedBundle,
    condition: ForecastCondition,
    forecast_bank: np.ndarray | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    env_kwargs = build_env_kwargs(args)

    for seed in args.seeds:
        run_dir = bundle.bundle_dir / "models" / f"seed{seed}"
        model = load_model(bundle.algo, run_dir / "ppo_model.zip")
        vec_norm = load_vec_normalize(
            run_dir / "vec_normalize.pkl",
            args=argparse.Namespace(
                risk_level=args.risk_level,
                step_size_hours=args.step_size_hours,
                max_steps=args.max_steps,
            ),
            reward_mode=args.reward_mode,
        )
        for episode_idx in range(args.eval_episodes):
            eval_seed = int(seed) + EVAL_EPISODE_SEED_OFFSET + episode_idx
            env = wrap_env(
                make_track_b_env(**env_kwargs),
                condition,
                forecast_bank=forecast_bank,
            )
            obs, info = env.reset(seed=eval_seed)
            terminated = False
            truncated = False
            reward_total = 0.0
            ret_thesis_total = 0.0
            ret_thesis_corrected_total = 0.0
            ret_seq_total = 0.0
            ret_unified_total = 0.0
            ret_garrido2024_raw_total = 0.0
            ret_garrido2024_train_total = 0.0
            ret_garrido2024_sigmoid_total = 0.0
            delivered_total = 0.0
            produced_total = 0.0
            demanded_total = 0.0
            backorder_qty_total = 0.0
            disruption_hours_total = 0.0
            inventory_total_sum = 0.0
            step_cost_total = 0.0
            service_continuity_total = 0.0
            backlog_containment_total = 0.0
            adaptive_efficiency_total = 0.0
            service_loss_area = 0.0
            service_loss_area_below_095 = 0.0
            recovery_streaks: list[int] = []
            current_recovery_streak = 0
            shift_counts = {1: 0, 2: 0, 3: 0}
            ret_case_counts = {case: 0 for case in RET_CASES}
            op10_multipliers: list[float] = []
            op12_multipliers: list[float] = []
            steps = 0
            final_info = info

            while not (terminated or truncated):
                obs_norm = vec_norm.normalize_obs(
                    np.asarray(obs, dtype=np.float32)[None, :]
                )
                action, _ = model.predict(obs_norm, deterministic=True)
                obs, reward, terminated, truncated, final_info = env.step(
                    np.asarray(action[0], dtype=np.float32)
                )
                reward_total += float(reward)
                ret_thesis_total += float(final_info.get("ret_thesis_step", 0.0))
                ret_thesis_corrected_total += float(
                    final_info.get("ret_thesis_corrected_step", 0.0)
                )
                ret_unified_total += float(final_info.get("ret_unified_step", 0.0))
                ret_garrido2024_raw_total += float(
                    final_info.get("ret_garrido2024_raw_step", 0.0)
                )
                ret_garrido2024_train_total += float(
                    final_info.get("ret_garrido2024_train_step", 0.0)
                )
                ret_garrido2024_sigmoid_total += float(
                    final_info.get("ret_garrido2024_sigmoid_step", 0.0)
                )
                delivered_total += float(final_info.get("new_delivered", 0.0))
                produced_total += float(final_info.get("new_produced", 0.0))
                demanded_total += float(final_info.get("new_demanded", 0.0))
                backorder_qty_total += float(final_info.get("new_backorder_qty", 0.0))
                disruption_hours_total += float(final_info.get("step_disruption_hours", 0.0))
                inventory_total_sum += float(final_info.get("total_inventory", 0.0))
                step_cost_total += float(final_info.get("step_cost", 0.0))
                ret_seq_terms = build_ret_seq_audit_terms(final_info)
                service_continuity_total += ret_seq_terms["service_continuity_step"]
                backlog_containment_total += ret_seq_terms["backlog_containment_step"]
                adaptive_efficiency_total += ret_seq_terms["adaptive_efficiency_step"]
                ret_seq_total += ret_seq_terms["ret_seq_step"]
                service_loss_area += 1.0 - ret_seq_terms["service_continuity_step"]
                service_loss_area_below_095 += max(
                    0.0,
                    0.95 - ret_seq_terms["service_continuity_step"],
                )
                ret_case = final_info.get("ret_components", {}).get("ret_case")
                if isinstance(ret_case, str) and ret_case in ret_case_counts:
                    ret_case_counts[ret_case] += 1
                if ret_case == "recovery":
                    current_recovery_streak += 1
                elif current_recovery_streak > 0:
                    recovery_streaks.append(current_recovery_streak)
                    current_recovery_streak = 0
                shift_counts[int(final_info.get("shifts_active", 1))] += 1
                op10_mult, op12_mult = extract_downstream_multipliers(final_info)
                op10_multipliers.append(op10_mult)
                op12_multipliers.append(op12_mult)
                steps += 1

            if current_recovery_streak > 0:
                recovery_streaks.append(current_recovery_streak)

            rows.append(
                finalize_episode_row(
                    reward_mode=args.reward_mode,
                    policy=f"ppo_{condition.label}",
                    algo=bundle.algo,
                    seed=int(seed),
                    episode=episode_idx + 1,
                    eval_seed=eval_seed,
                    steps=steps,
                    reward_total=reward_total,
                    ret_thesis_total=ret_thesis_total,
                    ret_thesis_corrected_total=ret_thesis_corrected_total,
                    ret_seq_total=ret_seq_total,
                    ret_unified_total=ret_unified_total,
                    ret_garrido2024_raw_total=ret_garrido2024_raw_total,
                    ret_garrido2024_train_total=ret_garrido2024_train_total,
                    ret_garrido2024_sigmoid_total=ret_garrido2024_sigmoid_total,
                    delivered_total=delivered_total,
                    produced_total=produced_total,
                    demanded_total=demanded_total,
                    backorder_qty_total=backorder_qty_total,
                    disruption_hours_total=disruption_hours_total,
                    inventory_total_sum=inventory_total_sum,
                    step_cost_total=step_cost_total,
                    service_continuity_total=service_continuity_total,
                    backlog_containment_total=backlog_containment_total,
                    adaptive_efficiency_total=adaptive_efficiency_total,
                    service_loss_area=service_loss_area,
                    service_loss_area_below_095=service_loss_area_below_095,
                    recovery_streaks=recovery_streaks,
                    shift_counts=shift_counts,
                    ret_case_counts=ret_case_counts,
                    op10_multipliers=op10_multipliers,
                    op12_multipliers=op12_multipliers,
                    final_info=final_info,
                    terminal_metrics=get_episode_terminal_metrics(env),
                    step_size_hours=float(args.step_size_hours),
                )
            )
            env.close()
        vec_norm.close()

    return rows


def build_summary_rows(policy_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_policy = {str(row["policy"]): row for row in policy_rows}
    full_row = by_policy.get("ppo_full")
    if full_row is None:
        return []
    summary_rows: list[dict[str, Any]] = []
    full_fill = float(full_row["fill_rate_mean"])
    full_ret = float(full_row["order_level_ret_mean_mean"])
    for condition in CONDITIONS:
        row = by_policy.get(f"ppo_{condition.label}")
        if row is None:
            continue
        summary = {
            "forecast_condition": condition.label,
            "fill_rate_mean": float(row["fill_rate_mean"]),
            "backorder_rate_mean": float(row["backorder_rate_mean"]),
            "order_level_ret_mean_mean": float(row["order_level_ret_mean_mean"]),
            "service_continuity_step_mean_mean": float(
                row["service_continuity_step_mean_mean"]
            ),
            "backlog_containment_step_mean_mean": float(
                row["backlog_containment_step_mean_mean"]
            ),
            "adaptive_efficiency_step_mean_mean": float(
                row["adaptive_efficiency_step_mean_mean"]
            ),
            "pct_ret_case_autotomy_mean": float(row["pct_ret_case_autotomy_mean"]),
            "pct_ret_case_recovery_mean": float(row["pct_ret_case_recovery_mean"]),
            "pct_ret_case_non_recovery_mean": float(
                row["pct_ret_case_non_recovery_mean"]
            ),
            "pct_steps_S1_mean": float(row["pct_steps_S1_mean"]),
            "pct_steps_S2_mean": float(row["pct_steps_S2_mean"]),
            "pct_steps_S3_mean": float(row["pct_steps_S3_mean"]),
            "op10_multiplier_step_mean_mean": float(
                row["op10_multiplier_step_mean_mean"]
            ),
            "op12_multiplier_step_mean_mean": float(
                row["op12_multiplier_step_mean_mean"]
            ),
            "fill_gap_vs_full_pp": 100.0 * (float(row["fill_rate_mean"]) - full_fill),
            "ret_gap_vs_full": float(row["order_level_ret_mean_mean"]) - full_ret,
        }
        summary_rows.append(summary)
    return summary_rows


def render_markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    headers = [
        "forecast_condition",
        "fill_rate_mean",
        "order_level_ret_mean_mean",
        "service_continuity_step_mean_mean",
        "pct_ret_case_autotomy_mean",
        "pct_ret_case_recovery_mean",
        "pct_steps_S1_mean",
        "pct_steps_S2_mean",
        "pct_steps_S3_mean",
        "op10_multiplier_step_mean_mean",
        "op12_multiplier_step_mean_mean",
        "fill_gap_vs_full_pp",
        "ret_gap_vs_full",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values: list[str] = []
        for key in headers:
            value = row[key]
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def run_forecast_sensitivity(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir or default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = LearnedBundle(policy="ppo", algo="ppo", bundle_dir=args.ppo_bundle)
    validate_bundle(bundle, list(args.seeds))
    forecast_bank = collect_forecast_bank(args)

    episode_rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        episode_rows.extend(
            evaluate_condition(
                args=args,
                bundle=bundle,
                condition=condition,
                forecast_bank=forecast_bank,
            )
        )

    policy_rows = aggregate_policy_rows(episode_rows)
    summary_rows = build_summary_rows(policy_rows)
    summary = {
        "config": {
            "ppo_bundle": str(args.ppo_bundle.resolve()),
            "reward_mode": str(args.reward_mode),
            "risk_level": str(args.risk_level),
            "seeds": [int(seed) for seed in args.seeds],
            "eval_episodes": int(args.eval_episodes),
            "bank_episodes": int(args.bank_episodes),
            "step_size_hours": float(args.step_size_hours),
            "max_steps": int(args.max_steps),
        },
        "forecast_bank": {
            "n_pairs": int(forecast_bank.shape[0]),
            "forecast_48h_min": float(np.min(forecast_bank[:, 0])),
            "forecast_48h_max": float(np.max(forecast_bank[:, 0])),
            "forecast_48h_mean": float(np.mean(forecast_bank[:, 0])),
            "forecast_168h_min": float(np.min(forecast_bank[:, 1])),
            "forecast_168h_max": float(np.max(forecast_bank[:, 1])),
            "forecast_168h_mean": float(np.mean(forecast_bank[:, 1])),
            "pct_48h_below_020": 100.0 * float(np.mean(forecast_bank[:, 0] < 0.2)),
            "pct_48h_above_050": 100.0 * float(np.mean(forecast_bank[:, 0] >= 0.5)),
        },
        "episode_metrics": episode_rows,
        "policy_summary": policy_rows,
        "forecast_sensitivity_summary": summary_rows,
        "artifacts": {
            "output_dir": str(output_dir.resolve()),
            "episode_metrics_csv": str((output_dir / "episode_metrics.csv").resolve()),
            "policy_summary_csv": str((output_dir / "policy_summary.csv").resolve()),
            "forecast_sensitivity_csv": str(
                (output_dir / "forecast_sensitivity_summary.csv").resolve()
            ),
            "forecast_sensitivity_md": str(
                (output_dir / "forecast_sensitivity_summary.md").resolve()
            ),
            "summary_json": str((output_dir / "summary.json").resolve()),
        },
    }

    save_csv(output_dir / "episode_metrics.csv", episode_rows)
    save_csv(output_dir / "policy_summary.csv", policy_rows)
    save_csv(output_dir / "forecast_sensitivity_summary.csv", summary_rows)
    (output_dir / "forecast_sensitivity_summary.md").write_text(
        render_markdown_table(summary_rows),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = build_parser().parse_args()
    summary = run_forecast_sensitivity(args)
    print(f"Wrote forecast sensitivity bundle to {summary['artifacts']['output_dir']}")
    for row in summary["forecast_sensitivity_summary"]:
        print(
            f"{row['forecast_condition']}: fill={row['fill_rate_mean']:.4f}, "
            f"ret={row['order_level_ret_mean_mean']:.4f}, "
            f"delta_fill_pp={row['fill_gap_vs_full_pp']:.3f}, "
            f"delta_ret={row['ret_gap_vs_full']:.4f}"
        )


if __name__ == "__main__":
    main()
