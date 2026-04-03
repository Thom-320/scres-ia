#!/usr/bin/env python3
"""
Comprehensive Track B resilience evaluation under a single audit lens.

This script evaluates frozen PPO / RecurrentPPO bundles plus static policies on
the Track B backbone while capturing the rich resilience signals already
emitted by the environment. Unlike the multi-lens audit script, this runner
keeps one active ``reward_mode`` and expands the per-policy reporting.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import statistics
import sys
from typing import Any, Callable

import numpy as np
try:
    from sb3_contrib import RecurrentPPO
except ImportError:  # pragma: no cover - optional dependency at runtime.
    RecurrentPPO = None
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_track_b_smoke import (  # noqa: E402
    StaticPolicySpec,
    build_static_policy_action,
)
from supply_chain.env_experimental_shifts import REWARD_MODE_OPTIONS  # noqa: E402
from supply_chain.external_env_interface import (  # noqa: E402
    get_episode_terminal_metrics,
    make_track_b_env,
)

REPO = Path(__file__).resolve().parent.parent
DEFAULT_PPO_BUNDLE = REPO / "outputs/track_b_benchmarks/track_b_ret_seq_k020_500k_rerun1"
DEFAULT_RPPO_BUNDLE = (
    REPO / "outputs/track_b_benchmarks/track_b_ret_seq_k020_500k_recurrent_ppo_rerun1"
)
DEFAULT_OUTPUT_DIR = REPO / "outputs/comprehensive_resilience_eval"

DEFAULT_SEEDS = (11, 22, 33, 44, 55)
DEFAULT_EVAL_EPISODES = 10
DEFAULT_EVAL_SEED_OFFSET = 50_000
DEFAULT_STEP_SIZE_HOURS = 168.0
DEFAULT_MAX_STEPS = 260

STATIC_POLICIES: tuple[StaticPolicySpec, ...] = (
    StaticPolicySpec(label="s1_d1.00", assembly_shifts=1, downstream_multiplier=1.0),
    StaticPolicySpec(label="s1_d1.50", assembly_shifts=1, downstream_multiplier=1.5),
    StaticPolicySpec(label="s1_d2.00", assembly_shifts=1, downstream_multiplier=2.0),
    StaticPolicySpec(label="s2_d1.00", assembly_shifts=2, downstream_multiplier=1.0),
    StaticPolicySpec(label="s2_d1.50", assembly_shifts=2, downstream_multiplier=1.5),
    StaticPolicySpec(label="s2_d2.00", assembly_shifts=2, downstream_multiplier=2.0),
    StaticPolicySpec(label="s3_d1.00", assembly_shifts=3, downstream_multiplier=1.0),
    StaticPolicySpec(label="s3_d1.50", assembly_shifts=3, downstream_multiplier=1.5),
    StaticPolicySpec(label="s3_d2.00", assembly_shifts=3, downstream_multiplier=2.0),
)

STEP_METRICS: tuple[str, ...] = (
    "service_continuity_step",
    "backlog_containment_step",
    "adaptive_efficiency_step",
    "ret_seq_step",
    "ret_thesis_step",
    "ret_thesis_corrected_step",
    "ret_garrido2024_raw_step",
    "ret_garrido2024_sigmoid_step",
    "zeta_avg",
    "epsilon_avg",
    "phi_avg",
    "tau_avg",
    "kappa_dot",
    "ret_unified_step",
    "ret_unified_fr",
    "ret_unified_rc",
    "ret_unified_ce",
    "ret_unified_gate",
)
RET_CASES: tuple[str, ...] = (
    "autotomy",
    "recovery",
    "non_recovery",
    "fill_rate_only",
    "no_demand",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a rich single-lens resilience evaluation on Track B for "
            "learned and static policies."
        )
    )
    parser.add_argument(
        "--ppo-bundle",
        type=Path,
        default=DEFAULT_PPO_BUNDLE,
        help="Frozen PPO Track B bundle directory.",
    )
    parser.add_argument(
        "--recurrent-bundle",
        type=Path,
        default=DEFAULT_RPPO_BUNDLE,
        help="Frozen RecurrentPPO Track B bundle directory.",
    )
    parser.add_argument(
        "--reward-mode",
        choices=list(REWARD_MODE_OPTIONS),
        default="ReT_seq_v1",
        help="Single audit reward lens to activate during evaluation.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help="Training seeds to evaluate.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=DEFAULT_EVAL_EPISODES,
        help="Evaluation episodes per policy and seed.",
    )
    parser.add_argument(
        "--eval-seed-offset",
        type=int,
        default=DEFAULT_EVAL_SEED_OFFSET,
        help="Offset added to training seeds for deterministic evaluation seeds.",
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
        "--risk-level",
        default="adaptive_benchmark_v2",
        help="Track B risk profile.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where JSON/CSV outputs are written.",
    )
    parser.add_argument(
        "--skip-recurrent",
        action="store_true",
        help="Skip RecurrentPPO if you only want PPO + statics.",
    )
    return parser


def make_raw_env(args: argparse.Namespace) -> Any:
    return make_track_b_env(
        reward_mode=args.reward_mode,
        observation_version="v7",
        action_contract="track_b_v1",
        risk_level=args.risk_level,
        step_size_hours=args.step_size_hours,
        max_steps=args.max_steps,
        ret_seq_kappa=0.20,
        year_basis="thesis",
        stochastic_pt=True,
    )


def make_vec_env(args: argparse.Namespace) -> DummyVecEnv:
    return DummyVecEnv([lambda: make_raw_env(args)])


def load_model_and_env(
    *, bundle_dir: Path, seed: int, algo: str, args: argparse.Namespace
) -> tuple[Any, DummyVecEnv | VecNormalize]:
    model_dir = bundle_dir / "models" / f"seed{seed}"
    if algo == "ppo":
        model = PPO.load(str(model_dir / "ppo_model.zip"), device="cpu")
    else:
        if RecurrentPPO is None:
            raise RuntimeError(
                "RecurrentPPO requested but sb3_contrib is not installed."
            )
        model = RecurrentPPO.load(
            str(model_dir / "recurrent_ppo_model.zip"), device="cpu"
        )

    vec_norm_path = model_dir / "vec_normalize.pkl"
    venv = make_vec_env(args)
    if vec_norm_path.exists():
        venv = VecNormalize.load(str(vec_norm_path), venv)
        venv.training = False
        venv.norm_reward = False
    return model, venv


def run_episode_collect(
    env_raw: Any,
    action_fn: Callable[..., np.ndarray | dict[str, float | int]],
    *,
    eval_seed: int,
    is_recurrent: bool = False,
) -> dict[str, list[float] | list[str]]:
    obs, _ = env_raw.reset(seed=eval_seed)
    step_data: dict[str, list[float] | list[str]] = defaultdict(list)
    done = False
    total_reward = 0.0
    steps = 0
    lstm_states: Any = None
    episode_starts = np.ones((1,), dtype=bool)

    while not done:
        if is_recurrent:
            action, lstm_states = action_fn(obs, lstm_states, episode_starts)
            episode_starts = np.zeros((1,), dtype=bool)
        else:
            action = action_fn(obs)

        obs, reward, terminated, truncated, info = env_raw.step(action)
        done = terminated or truncated
        total_reward += float(reward)
        steps += 1

        for key in STEP_METRICS:
            val = info.get(key)
            if val is not None:
                step_data[key].append(float(val))

        ret_components = info.get("ret_components", {})
        if isinstance(ret_components, dict):
            ret_case = ret_components.get("ret_case")
            if isinstance(ret_case, str):
                step_data["_ret_cases"].append(ret_case)

        shift_value = info.get("shifts_active", info.get("assembly_shifts"))
        if shift_value is not None:
            step_data["shifts_active"].append(float(shift_value))

        fill_rate = info.get("fill_rate")
        if fill_rate is not None:
            step_data["fill_rate"].append(float(fill_rate))
        backorder_rate = info.get("backorder_rate")
        if backorder_rate is not None:
            step_data["backorder_rate"].append(float(backorder_rate))

    step_data["reward_total"] = [total_reward]
    step_data["steps"] = [float(steps)]

    terminal_metrics = get_episode_terminal_metrics(env_raw)
    step_data["order_level_ret_mean"] = [
        float(terminal_metrics["order_level_ret_mean"])
    ]
    step_data["order_level_fill_rate"] = [
        float(terminal_metrics["fill_rate_order_level"])
    ]
    step_data["order_level_backorder_rate"] = [
        float(terminal_metrics["backorder_rate_order_level"])
    ]
    return step_data


def aggregate(all_eps: list[dict[str, list[float] | list[str]]]) -> dict[str, Any]:
    agg: dict[str, Any] = {}
    all_keys = set()
    for ep in all_eps:
        all_keys.update(ep.keys())

    for key in sorted(all_keys):
        if key.startswith("_"):
            continue
        ep_means = [
            statistics.mean([float(v) for v in ep[key]])
            for ep in all_eps
            if key in ep and ep[key]
        ]
        if ep_means:
            agg[f"{key}_mean"] = round(statistics.mean(ep_means), 7)
            if len(ep_means) > 1:
                agg[f"{key}_std"] = round(statistics.stdev(ep_means), 7)

    all_cases: list[str] = []
    for ep in all_eps:
        all_cases.extend([str(case) for case in ep.get("_ret_cases", [])])
    if all_cases:
        total = len(all_cases)
        for case in RET_CASES:
            agg[f"pct_case_{case}"] = round(all_cases.count(case) / total * 100, 2)
    return agg


def eval_learned(
    *, bundle: Path, algo: str, policy_label: str, args: argparse.Namespace
) -> dict[str, Any]:
    print(f"\n>>> Evaluating {policy_label} ({args.reward_mode})...")
    is_recurrent = algo == "recurrent_ppo"
    all_eps: list[dict[str, list[float] | list[str]]] = []

    for seed in args.seeds:
        try:
            model, venv = load_model_and_env(
                bundle_dir=bundle, seed=seed, algo=algo, args=args
            )
        except Exception as exc:
            print(f"  SKIP seed {seed}: {exc}")
            continue

        raw_env = make_raw_env(args)
        for episode_idx in range(args.eval_episodes):
            eval_seed = seed + args.eval_seed_offset + episode_idx
            if is_recurrent:

                def action_fn(
                    obs: np.ndarray,
                    lstm_state: Any,
                    episode_start: np.ndarray,
                    *,
                    model_obj: Any = model,
                    vec_norm_obj: Any = venv,
                ) -> tuple[np.ndarray, Any]:
                    obs_norm = vec_norm_obj.normalize_obs(
                        np.asarray(obs, dtype=np.float32).reshape(1, -1)
                    )
                    action, next_state = model_obj.predict(
                        obs_norm,
                        state=lstm_state,
                        episode_start=episode_start,
                        deterministic=True,
                    )
                    return np.asarray(action).flatten(), next_state

                episode = run_episode_collect(
                    raw_env,
                    action_fn,
                    eval_seed=eval_seed,
                    is_recurrent=True,
                )
            else:

                def action_fn(
                    obs: np.ndarray,
                    *,
                    model_obj: Any = model,
                    vec_norm_obj: Any = venv,
                ) -> np.ndarray:
                    obs_norm = vec_norm_obj.normalize_obs(
                        np.asarray(obs, dtype=np.float32).reshape(1, -1)
                    )
                    action, _ = model_obj.predict(obs_norm, deterministic=True)
                    return np.asarray(action).flatten()

                episode = run_episode_collect(
                    raw_env,
                    action_fn,
                    eval_seed=eval_seed,
                )

            all_eps.append(episode)
        raw_env.close()
        if hasattr(venv, "close"):
            venv.close()
        print(f"  {policy_label} seed {seed}: {args.eval_episodes} eps done")

    agg = aggregate(all_eps)
    agg["policy"] = policy_label
    agg["n_episodes"] = len(all_eps)
    agg["reward_mode"] = args.reward_mode
    return agg


def eval_static(policy: StaticPolicySpec, args: argparse.Namespace) -> dict[str, Any]:
    print(f"\n>>> Evaluating static: {policy.label} ({args.reward_mode})")
    all_eps: list[dict[str, list[float] | list[str]]] = []
    action_payload = build_static_policy_action(policy)

    for seed in args.seeds:
        raw_env = make_raw_env(args)
        for episode_idx in range(args.eval_episodes):
            eval_seed = seed + args.eval_seed_offset + episode_idx
            episode = run_episode_collect(
                raw_env,
                lambda _obs, payload=action_payload: dict(payload),
                eval_seed=eval_seed,
            )
            all_eps.append(episode)
        raw_env.close()
        print(f"  {policy.label}: seed {seed} done")

    agg = aggregate(all_eps)
    agg["policy"] = policy.label
    agg["n_episodes"] = len(all_eps)
    agg["reward_mode"] = args.reward_mode
    return agg


def save_results(results: list[dict[str, Any]], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "results.json"
    csv_path = output_dir / "results.csv"
    json_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    all_keys = set()
    for result in results:
        all_keys.update(result.keys())
    cols = ["policy"] + sorted(k for k in all_keys if k != "policy")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    return json_path, csv_path


def print_summary_table(results: list[dict[str, Any]]) -> None:
    key_metrics = [
        "order_level_fill_rate_mean",
        "order_level_backorder_rate_mean",
        "reward_total_mean",
        "ret_seq_step_mean",
        "service_continuity_step_mean",
        "backlog_containment_step_mean",
        "adaptive_efficiency_step_mean",
        "ret_thesis_step_mean",
        "ret_thesis_corrected_step_mean",
        "ret_garrido2024_raw_step_mean",
        "ret_garrido2024_sigmoid_step_mean",
        "ret_unified_step_mean",
        "order_level_ret_mean_mean",
        "pct_case_autotomy",
        "pct_case_recovery",
        "pct_case_non_recovery",
        "pct_case_fill_rate_only",
        "pct_case_no_demand",
    ]

    print(f"\n{'=' * 72}")
    print("COMPREHENSIVE RESILIENCE COMPARISON")
    print(f"{'=' * 72}")
    header = f"{'Metric':<42}"
    for result in results:
        header += f" {result['policy']:>16}"
    print(header)
    print("-" * len(header))

    for metric in key_metrics:
        row = f"{metric:<42}"
        for result in results:
            value = result.get(metric)
            if value is None:
                row += f" {'N/A':>16}"
            elif "pct_case" in metric:
                row += f" {float(value):>15.1f}%"
            else:
                row += f" {float(value):>16.5f}"
        print(row)


def main() -> None:
    args = build_parser().parse_args()
    results: list[dict[str, Any]] = []

    results.append(
        eval_learned(
            bundle=args.ppo_bundle,
            algo="ppo",
            policy_label="PPO_7D",
            args=args,
        )
    )
    if not args.skip_recurrent:
        results.append(
            eval_learned(
                bundle=args.recurrent_bundle,
                algo="recurrent_ppo",
                policy_label="RecurrentPPO_7D",
                args=args,
            )
        )

    for policy in STATIC_POLICIES:
        results.append(eval_static(policy, args))

    json_path, csv_path = save_results(results, args.output_dir)
    print_summary_table(results)
    print(f"\nSaved: {json_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
