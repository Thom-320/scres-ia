#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

from gymnasium.wrappers import FrameStackObservation
import numpy as np

try:
    from sb3_contrib import RecurrentPPO
except ImportError:  # pragma: no cover - exercised via explicit runtime guard.
    RecurrentPPO = None
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.config import DEFAULT_YEAR_BASIS, YEAR_BASIS_OPTIONS
from supply_chain.external_env_interface import make_shift_control_env

STATIC_POLICY_ORDER = ("static_s1", "static_s2", "static_s3")
RANDOM_POLICY_NAME = "random"
FIXED_POLICY_ACTIONS: dict[str, np.ndarray] = {
    "static_s1": np.array([0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32),
    "static_s2": np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    "static_s3": np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
}

# ---------------------------------------------------------------------------
# Heuristic baseline policies
# ---------------------------------------------------------------------------

SHIFT_SIGNAL = {1: -1.0, 2: 0.0, 3: 1.0}


def _latest_frame(obs: np.ndarray) -> np.ndarray:
    """Extract the most recent frame from a potentially stacked observation."""
    return obs[-1] if obs.ndim == 2 else obs


class HeuristicHysteresis:
    """Shift control with hysteresis bands on backorder_rate (obs[7]).

    Avoids oscillation by only changing shifts when backorder_rate crosses
    the high or low threshold.  Inventory multipliers stay neutral.
    """

    def __init__(self, tau_high: float = 0.15, tau_low: float = 0.05) -> None:
        self.tau_high = tau_high
        self.tau_low = tau_low
        self._current_shift = 2

    def reset(self) -> None:
        self._current_shift = 2

    def __call__(self, obs: np.ndarray, info: dict[str, Any]) -> np.ndarray:
        frame = _latest_frame(obs)
        backorder_rate = float(frame[7])
        if backorder_rate > self.tau_high:
            self._current_shift = 3
        elif backorder_rate < self.tau_low:
            self._current_shift = 1
        return np.array(
            [0.0, 0.0, 0.0, 0.0, SHIFT_SIGNAL[self._current_shift]],
            dtype=np.float32,
        )


class HeuristicDisruptionAware:
    """Disruption-reactive shift + inventory control.

    Monitors downtime flags (obs[8], obs[9]) and fill_rate (obs[6]).
    Combines shift escalation with inventory order boosting.
    """

    def __init__(
        self,
        fill_rate_caution: float = 0.90,
        inventory_boost: float = 0.5,
        inventory_large_boost: float = 1.0,
    ) -> None:
        self.fill_rate_caution = fill_rate_caution
        self.inventory_boost = inventory_boost
        self.inventory_large_boost = inventory_large_boost

    def reset(self) -> None:
        pass

    def __call__(self, obs: np.ndarray, info: dict[str, Any]) -> np.ndarray:
        frame = _latest_frame(obs)
        assembly_down = float(frame[8]) > 0.5
        any_loc_down = float(frame[9]) > 0.5
        fill_rate = float(frame[6])
        if assembly_down or any_loc_down:
            shift_signal = 1.0  # S3
            inv_signal = self.inventory_large_boost
        elif fill_rate < self.fill_rate_caution:
            shift_signal = 0.0  # S2
            inv_signal = self.inventory_boost
        else:
            shift_signal = -1.0  # S1
            inv_signal = 0.0
        return np.array(
            [inv_signal, inv_signal, inv_signal, inv_signal, shift_signal],
            dtype=np.float32,
        )


class HeuristicTuned:
    """Combined hysteresis + disruption-aware with tunable parameters.

    Shift decision uses hysteresis on backorder_rate (like HeuristicHysteresis).
    Inventory decision uses disruption/fill-rate rules (like HeuristicDisruptionAware).
    Default parameters are reasonable fallbacks; use ``tune_heuristic_params()``
    to optimise them via grid search.
    """

    def __init__(
        self,
        tau_high: float = 0.12,
        tau_low: float = 0.05,
        fill_rate_caution: float = 0.90,
        boost_normal: float = 0.0,
        boost_caution: float = 0.5,
        boost_crisis: float = 1.0,
    ) -> None:
        self.tau_high = tau_high
        self.tau_low = tau_low
        self.fill_rate_caution = fill_rate_caution
        self.boost_normal = boost_normal
        self.boost_caution = boost_caution
        self.boost_crisis = boost_crisis
        self._current_shift = 2

    def reset(self) -> None:
        self._current_shift = 2

    def __call__(self, obs: np.ndarray, info: dict[str, Any]) -> np.ndarray:
        frame = _latest_frame(obs)
        backorder_rate = float(frame[7])
        if backorder_rate > self.tau_high:
            self._current_shift = 3
        elif backorder_rate < self.tau_low:
            self._current_shift = 1

        assembly_down = float(frame[8]) > 0.5
        any_loc_down = float(frame[9]) > 0.5
        fill_rate = float(frame[6])
        if assembly_down or any_loc_down:
            inv_signal = self.boost_crisis
        elif fill_rate < self.fill_rate_caution:
            inv_signal = self.boost_caution
        else:
            inv_signal = self.boost_normal

        return np.array(
            [
                inv_signal,
                inv_signal,
                inv_signal,
                inv_signal,
                SHIFT_SIGNAL[self._current_shift],
            ],
            dtype=np.float32,
        )


HeuristicPolicy = HeuristicHysteresis | HeuristicDisruptionAware | HeuristicTuned

HEURISTIC_POLICY_NAMES = (
    "heuristic_hysteresis",
    "heuristic_disruption",
    "heuristic_tuned",
)


def _make_heuristic_defaults() -> dict[str, HeuristicPolicy]:
    return {
        "heuristic_hysteresis": HeuristicHysteresis(),
        "heuristic_disruption": HeuristicDisruptionAware(),
        "heuristic_tuned": HeuristicTuned(),
    }


HEURISTIC_DEFAULTS: dict[str, HeuristicPolicy] = _make_heuristic_defaults()

EVAL_EPISODE_SEED_OFFSET = 80_000
SURVIVOR_REWARD_MARGIN = 1.0
SURVIVOR_FILL_RATE_MARGIN = 0.01
PPO_SERVICE_TOLERANCE = 0.01
PRIMARY_METRICS = (
    "reward_total",
    "service_loss_total",
    "shift_cost_total",
    "mean_disruption_fraction",
    "fill_rate",
    "backorder_rate",
    "ret_thesis_corrected_total",
    "pct_steps_S1",
    "pct_steps_S2",
    "pct_steps_S3",
)
EPISODE_FIELDNAMES = [
    "phase",
    "policy",
    "algo",
    "frame_stack",
    "observation_version",
    "seed",
    "episode",
    "eval_seed",
    "w_bo",
    "w_cost",
    "w_disr",
    "steps",
    "reward_total",
    "service_loss_total",
    "shift_cost_total",
    "mean_disruption_fraction",
    "fill_rate",
    "backorder_rate",
    "ret_thesis_corrected_total",
    "demanded_total",
    "delivered_total",
    "backorder_qty_total",
    "pct_steps_S1",
    "pct_steps_S2",
    "pct_steps_S3",
]
POLICY_SUMMARY_FIELDNAMES = [
    "phase",
    "policy",
    "algo",
    "frame_stack",
    "observation_version",
    "w_bo",
    "w_cost",
    "w_disr",
    "seed_count",
]
for _metric in PRIMARY_METRICS:
    POLICY_SUMMARY_FIELDNAMES.extend(
        [
            f"{_metric}_mean",
            f"{_metric}_std",
            f"{_metric}_ci95_low",
            f"{_metric}_ci95_high",
        ]
    )
COMPARISON_FIELDNAMES = [
    "algo",
    "frame_stack",
    "observation_version",
    "learned_policy",
    "w_bo",
    "w_cost",
    "w_disr",
    "best_static_policy",
    "static_reward_gap_best_minus_s1",
    "random_reward_mean",
    "random_fill_rate_mean",
    "random_backorder_rate_mean",
    "random_ret_thesis_corrected_total_mean",
    "learned_reward_mean",
    "learned_fill_rate_mean",
    "learned_backorder_rate_mean",
    "learned_ret_thesis_corrected_total_mean",
    "ppo_reward_mean",
    "static_s2_reward_mean",
    "best_static_reward_mean",
    "ppo_fill_rate_mean",
    "static_s2_fill_rate_mean",
    "best_static_fill_rate_mean",
    "ppo_backorder_rate_mean",
    "static_s2_backorder_rate_mean",
    "best_static_backorder_rate_mean",
    "ppo_ret_thesis_corrected_total_mean",
    "static_s2_ret_thesis_corrected_total_mean",
    "best_static_ret_thesis_corrected_total_mean",
    "ppo_pct_steps_S1_mean",
    "ppo_pct_steps_S2_mean",
    "ppo_pct_steps_S3_mean",
    "best_heuristic_policy",
    "best_heuristic_reward_mean",
    "best_heuristic_fill_rate_mean",
    "best_baseline_policy",
    "best_baseline_reward_mean",
    "learned_beats_random",
    "learned_beats_best_static",
    "learned_beats_best_heuristic",
    "learned_beats_best_baseline",
    "ppo_beats_static_s2",
    "ppo_beats_best_static",
    "collapsed_to_S1",
    "collapsed_to_S2",
    "collapsed_to_S3",
]

SAC_POLICY_KWARGS: dict[str, Any] = {"net_arch": [256, 256]}
PPO_POLICY_KWARGS: dict[str, Any] = {"net_arch": {"pi": [64, 64], "vf": [64, 64]}}
RECURRENT_PPO_POLICY_KWARGS: dict[str, Any] = {
    "net_arch": {"pi": [64], "vf": [64]},
    "lstm_hidden_size": 128,
    "n_lstm_layers": 1,
    "shared_lstm": False,
    "enable_critic_lstm": True,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark a control-oriented reward on the MFSC shift-control env."
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    parser.add_argument("--train-timesteps", type=int, default=50_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--step-size-hours", type=float, default=168.0)
    parser.add_argument("--max-steps", type=int, default=260)
    parser.add_argument(
        "--algo",
        choices=["ppo", "sac", "recurrent_ppo"],
        default="ppo",
        help="Learned policy algorithm for the adaptive lane.",
    )
    parser.add_argument(
        "--frame-stack",
        type=int,
        default=1,
        help="Number of observation frames to stack for temporal context.",
    )
    parser.add_argument(
        "--observation-version",
        choices=["v1", "v2"],
        default="v1",
        help="Observation contract version. v1 preserves historical 15-d runs; v2 adds previous-step diagnostics.",
    )
    parser.add_argument(
        "--risk-level",
        choices=["current", "increased", "severe"],
        default="increased",
    )
    parser.add_argument(
        "--stochastic-pt",
        action="store_true",
        help="Enable stochastic processing times in the shift-control environment.",
    )
    parser.add_argument(
        "--year-basis",
        choices=YEAR_BASIS_OPTIONS,
        default=DEFAULT_YEAR_BASIS,
    )
    parser.add_argument("--w-bo", type=float, nargs="+", default=[1.0, 2.0, 4.0])
    parser.add_argument("--w-cost", type=float, nargs="+", default=[0.02, 0.06, 0.10])
    parser.add_argument(
        "--w-disr",
        type=float,
        nargs="+",
        default=[0.0],
        help="Control reward disruption weights. Default keeps the first round at 0.0.",
    )
    parser.add_argument(
        "--max-survivors",
        type=int,
        default=4,
        help="Maximum number of weight combinations forwarded to PPO.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("docs/artifacts/control_reward"),
        help="Tracked destination for auditable benchmark bundles.",
    )
    parser.add_argument(
        "--artifact-label",
        type=str,
        default=None,
        help="Optional subdirectory name under --artifact-root for exported artifacts.",
    )
    parser.add_argument(
        "--skip-artifact-export",
        action="store_true",
        help="Skip copying the benchmark bundle into the tracked artifact directory.",
    )
    parser.add_argument(
        "--eval-risk-levels",
        nargs="*",
        choices=["current", "increased", "severe"],
        default=None,
        help="Risk levels to evaluate on after training. Defaults to same as --risk-level.",
    )
    parser.add_argument(
        "--tune-heuristic",
        action="store_true",
        help="Grid-search HeuristicTuned parameters before the main benchmark.",
    )
    parser.add_argument(
        "--tune-episodes",
        type=int,
        default=1,
        help="Episodes per parameter combo during heuristic tuning.",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--train-freq", type=int, default=1)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--learning-starts", type=int, default=5_000)
    return parser


def ci95(values: list[float]) -> tuple[float, float]:
    if len(values) < 2:
        value = float(values[0]) if values else float("nan")
        return value, value
    arr = np.asarray(values, dtype=np.float64)
    half = 1.96 * arr.std(ddof=1) / np.sqrt(len(arr))
    mean = arr.mean()
    return float(mean - half), float(mean + half)


def static_policy_action(policy: str) -> np.ndarray:
    if policy not in FIXED_POLICY_ACTIONS:
        raise ValueError(f"Unsupported fixed policy {policy!r}.")
    return FIXED_POLICY_ACTIONS[policy].copy()


def make_weight_combos(args: argparse.Namespace) -> list[dict[str, float]]:
    combos: list[dict[str, float]] = []
    for w_bo in args.w_bo:
        for w_cost in args.w_cost:
            for w_disr in args.w_disr:
                combos.append(
                    {
                        "w_bo": float(w_bo),
                        "w_cost": float(w_cost),
                        "w_disr": float(w_disr),
                    }
                )
    return combos


def build_env_kwargs(
    args: argparse.Namespace,
    weight_combo: dict[str, float],
    risk_level_override: str | None = None,
) -> dict[str, Any]:
    return {
        "reward_mode": "control_v1",
        "observation_version": args.observation_version,
        "step_size_hours": args.step_size_hours,
        "risk_level": risk_level_override or args.risk_level,
        "stochastic_pt": args.stochastic_pt,
        "max_steps": args.max_steps,
        "year_basis": args.year_basis,
        **weight_combo,
    }


def learned_policy_name(args: argparse.Namespace) -> str:
    return str(args.algo)


def learned_phase_name(args: argparse.Namespace) -> str:
    return f"{args.algo}_eval"


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    base_name = "control_reward"
    if args.algo != "ppo" or int(args.frame_stack) != 1:
        base_name = f"{base_name}_{args.algo}_fs{args.frame_stack}"
    return Path("outputs/benchmarks") / base_name


def wrap_env_for_benchmark(env: Any, args: argparse.Namespace) -> Any:
    if int(args.frame_stack) > 1:
        env = FrameStackObservation(env, stack_size=int(args.frame_stack))
    return env


def resolve_git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def resolve_invocation(args: argparse.Namespace) -> str:
    invocation = getattr(args, "invocation", None)
    if invocation:
        return str(invocation)
    return "python scripts/benchmark_control_reward.py (invocation unavailable)"


def export_artifact_bundle(
    *,
    source_dir: Path,
    artifact_root: Path,
    label: str,
    summary: dict[str, Any],
    command: str,
) -> Path:
    bundle_dir = artifact_root / label
    bundle_dir.mkdir(parents=True, exist_ok=True)

    copied_files: dict[str, str] = {}
    for filename in ("comparison_table.csv", "policy_summary.csv", "summary.json"):
        src = source_dir / filename
        if not src.exists():
            raise FileNotFoundError(f"Missing benchmark artifact: {src}")
        dst = bundle_dir / filename
        shutil.copy2(src, dst)
        copied_files[filename] = str(dst.resolve())

    manifest = {
        "artifact_type": "control_reward_benchmark",
        "benchmark_date_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": resolve_git_commit(),
        "command": command,
        "source_benchmark_directory": str(source_dir.resolve()),
        "bundle_directory": str(bundle_dir.resolve()),
        "config": summary.get("config", {}),
        "files": copied_files,
    }
    manifest_path = bundle_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as file_obj:
        json.dump(manifest, file_obj, indent=2)
    return bundle_dir


def make_monitored_training_env(
    args: argparse.Namespace, seed: int, weight_combo: dict[str, float]
) -> callable:
    env_kwargs = build_env_kwargs(args, weight_combo)

    def _init() -> Monitor:
        env = make_shift_control_env(**env_kwargs)
        env = wrap_env_for_benchmark(env, args)
        env.reset(seed=seed)
        return Monitor(env)

    return _init


def ensure_algo_dependencies(args: argparse.Namespace) -> None:
    if args.algo == "recurrent_ppo" and RecurrentPPO is None:
        raise RuntimeError(
            "recurrent_ppo requires sb3-contrib. Install dependencies from requirements.txt."
        )


def train_model(
    args: argparse.Namespace, seed: int, weight_combo: dict[str, float]
) -> tuple[Any, DummyVecEnv]:
    ensure_algo_dependencies(args)
    vec_env = DummyVecEnv([make_monitored_training_env(args, seed, weight_combo)])
    if args.algo == "ppo":
        model: Any = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            policy_kwargs=PPO_POLICY_KWARGS,
            seed=seed,
            verbose=0,
            device="cpu",
        )
    elif args.algo == "recurrent_ppo":
        model = RecurrentPPO(
            "MlpLstmPolicy",
            vec_env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            policy_kwargs=RECURRENT_PPO_POLICY_KWARGS,
            seed=seed,
            verbose=0,
            device="cpu",
        )
    else:
        model = SAC(
            "MlpPolicy",
            vec_env,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            gamma=args.gamma,
            tau=args.tau,
            train_freq=args.train_freq,
            gradient_steps=args.gradient_steps,
            learning_starts=args.learning_starts,
            policy_kwargs=SAC_POLICY_KWARGS,
            seed=seed,
            verbose=0,
            device="cpu",
        )
    model.learn(total_timesteps=args.train_timesteps)
    return model, vec_env


def finalize_episode_metrics(
    *,
    phase: str,
    policy: str,
    algo: str,
    frame_stack: int,
    observation_version: str,
    seed: int,
    episode: int,
    eval_seed: int,
    weight_combo: dict[str, float],
    steps: int,
    reward_total: float,
    service_loss_total: float,
    shift_cost_total: float,
    disruption_fraction_total: float,
    ret_thesis_corrected_total: float,
    demanded_total: float,
    delivered_total: float,
    backorder_qty_total: float,
    shift_counts: dict[int, int],
) -> dict[str, Any]:
    if demanded_total > 0:
        backorder_rate = backorder_qty_total / demanded_total
        fill_rate = 1.0 - backorder_rate
    else:
        backorder_rate = 0.0
        fill_rate = 1.0
    total_steps = max(1, steps)
    return {
        "phase": phase,
        "policy": policy,
        "algo": algo,
        "frame_stack": frame_stack,
        "observation_version": observation_version,
        "seed": seed,
        "episode": episode,
        "eval_seed": eval_seed,
        "w_bo": weight_combo["w_bo"],
        "w_cost": weight_combo["w_cost"],
        "w_disr": weight_combo["w_disr"],
        "steps": steps,
        "reward_total": reward_total,
        "service_loss_total": service_loss_total,
        "shift_cost_total": shift_cost_total,
        "mean_disruption_fraction": disruption_fraction_total / total_steps,
        "fill_rate": fill_rate,
        "backorder_rate": backorder_rate,
        "ret_thesis_corrected_total": ret_thesis_corrected_total,
        "demanded_total": demanded_total,
        "delivered_total": delivered_total,
        "backorder_qty_total": backorder_qty_total,
        "pct_steps_S1": 100.0 * shift_counts.get(1, 0) / total_steps,
        "pct_steps_S2": 100.0 * shift_counts.get(2, 0) / total_steps,
        "pct_steps_S3": 100.0 * shift_counts.get(3, 0) / total_steps,
    }


def evaluate_policy(
    phase: str,
    policy: str,
    *,
    args: argparse.Namespace,
    weight_combo: dict[str, float],
    seed: int,
    model: Any = None,
    risk_level_override: str | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    env_kwargs = build_env_kwargs(
        args, weight_combo, risk_level_override=risk_level_override
    )
    is_recurrent = args.algo == "recurrent_ppo" and policy == learned_policy_name(args)

    for episode_idx in range(args.eval_episodes):
        eval_seed = seed + EVAL_EPISODE_SEED_OFFSET + episode_idx
        episode_rng = np.random.default_rng(eval_seed)
        env = make_shift_control_env(**env_kwargs)
        env = wrap_env_for_benchmark(env, args)
        obs, reset_info = env.reset(seed=eval_seed)
        prev_info: dict[str, Any] = reset_info if isinstance(reset_info, dict) else {}
        terminated = False
        truncated = False
        reward_total = 0.0
        service_loss_total = 0.0
        shift_cost_total = 0.0
        disruption_fraction_total = 0.0
        ret_thesis_corrected_total = 0.0
        demanded_total = 0.0
        delivered_total = 0.0
        backorder_qty_total = 0.0
        steps = 0
        shift_counts = {1: 0, 2: 0, 3: 0}
        lstm_states: Any = None
        episode_start = np.ones((1,), dtype=bool)

        heuristic = HEURISTIC_DEFAULTS.get(policy)
        if heuristic is not None:
            heuristic.reset()

        while not (terminated or truncated):
            if policy == learned_policy_name(args):
                if model is None:
                    raise ValueError(
                        "Learned-policy evaluation requires a trained model."
                    )
                if is_recurrent:
                    action, lstm_states = model.predict(
                        obs,
                        state=lstm_states,
                        episode_start=episode_start,
                        deterministic=True,
                    )
                else:
                    action, _ = model.predict(obs, deterministic=True)
            elif policy == RANDOM_POLICY_NAME:
                action = episode_rng.uniform(-1.0, 1.0, size=5).astype(np.float32)
            elif heuristic is not None:
                action = heuristic(obs, prev_info)
            else:
                action = static_policy_action(policy)

            obs, reward, terminated, truncated, info = env.step(action)
            prev_info = info
            reward_total += float(reward)
            service_loss_total += float(info.get("service_loss_step", 0.0))
            shift_cost_total += float(info.get("shift_cost_step", 0.0))
            disruption_fraction_total += float(
                info.get("disruption_fraction_step", 0.0)
            )
            ret_thesis_corrected_total += float(
                info.get("ret_thesis_corrected_step", 0.0)
            )
            demanded_total += float(info.get("new_demanded", 0.0))
            delivered_total += float(info.get("new_delivered", 0.0))
            backorder_qty_total += float(info.get("new_backorder_qty", 0.0))
            shift_counts[int(info.get("shifts_active", 1))] += 1
            steps += 1
            if is_recurrent:
                episode_start = np.array([terminated or truncated], dtype=bool)

        rows.append(
            finalize_episode_metrics(
                phase=phase,
                policy=policy,
                algo=args.algo,
                frame_stack=int(args.frame_stack),
                observation_version=str(args.observation_version),
                seed=seed,
                episode=episode_idx + 1,
                eval_seed=eval_seed,
                weight_combo=weight_combo,
                steps=steps,
                reward_total=reward_total,
                service_loss_total=service_loss_total,
                shift_cost_total=shift_cost_total,
                disruption_fraction_total=disruption_fraction_total,
                ret_thesis_corrected_total=ret_thesis_corrected_total,
                demanded_total=demanded_total,
                delivered_total=delivered_total,
                backorder_qty_total=backorder_qty_total,
                shift_counts=shift_counts,
            )
        )
        env.close()

    return rows


def aggregate_seed_metrics(episode_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[
        tuple[str, str, str, int, str, float, float, float, int], list[dict[str, Any]]
    ] = {}
    for row in episode_rows:
        key = (
            str(row["phase"]),
            str(row["policy"]),
            str(row["algo"]),
            int(row["frame_stack"]),
            str(row["observation_version"]),
            float(row["w_bo"]),
            float(row["w_cost"]),
            float(row["w_disr"]),
            int(row["seed"]),
        )
        grouped.setdefault(key, []).append(row)

    seed_rows: list[dict[str, Any]] = []
    for (
        phase,
        policy,
        algo,
        frame_stack,
        observation_version,
        w_bo,
        w_cost,
        w_disr,
        seed,
    ), rows in sorted(grouped.items()):
        seed_row: dict[str, Any] = {
            "phase": phase,
            "policy": policy,
            "algo": algo,
            "frame_stack": frame_stack,
            "observation_version": observation_version,
            "w_bo": w_bo,
            "w_cost": w_cost,
            "w_disr": w_disr,
            "seed": seed,
            "episodes": len(rows),
        }
        for metric in PRIMARY_METRICS:
            values = [float(row[metric]) for row in rows]
            seed_row[f"{metric}_mean"] = float(np.mean(values))
            seed_row[f"{metric}_std"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            )
        seed_rows.append(seed_row)
    return seed_rows


def aggregate_policy_metrics(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[
        tuple[str, str, str, int, str, float, float, float], list[dict[str, Any]]
    ] = {}
    for row in seed_rows:
        key = (
            str(row["phase"]),
            str(row["policy"]),
            str(row["algo"]),
            int(row["frame_stack"]),
            str(row["observation_version"]),
            float(row["w_bo"]),
            float(row["w_cost"]),
            float(row["w_disr"]),
        )
        grouped.setdefault(key, []).append(row)

    policy_rows: list[dict[str, Any]] = []
    for (
        phase,
        policy,
        algo,
        frame_stack,
        observation_version,
        w_bo,
        w_cost,
        w_disr,
    ), rows in sorted(grouped.items()):
        out_row: dict[str, Any] = {
            "phase": phase,
            "policy": policy,
            "algo": algo,
            "frame_stack": frame_stack,
            "observation_version": observation_version,
            "w_bo": w_bo,
            "w_cost": w_cost,
            "w_disr": w_disr,
            "seed_count": len(rows),
        }
        for metric in PRIMARY_METRICS:
            values = [float(row[f"{metric}_mean"]) for row in rows]
            ci_low, ci_high = ci95(values)
            out_row[f"{metric}_mean"] = float(np.mean(values))
            out_row[f"{metric}_std"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            )
            out_row[f"{metric}_ci95_low"] = ci_low
            out_row[f"{metric}_ci95_high"] = ci_high
        policy_rows.append(out_row)
    return policy_rows


def save_csv(
    path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None
) -> None:
    resolved_fieldnames = fieldnames or (list(rows[0].keys()) if rows else None)
    if resolved_fieldnames is None:
        return
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=resolved_fieldnames)
        writer.writeheader()
        if rows:
            writer.writerows(rows)


def policy_lookup(
    policy_rows: list[dict[str, Any]],
    phase: str,
    policy: str,
    weight_combo: dict[str, float],
    *,
    algo: str,
    frame_stack: int,
    observation_version: str,
) -> dict[str, Any] | None:
    for row in policy_rows:
        if (
            row["phase"] == phase
            and row["policy"] == policy
            and row["algo"] == algo
            and int(row["frame_stack"]) == frame_stack
            and str(row["observation_version"]) == observation_version
            and float(row["w_bo"]) == float(weight_combo["w_bo"])
            and float(row["w_cost"]) == float(weight_combo["w_cost"])
            and float(row["w_disr"]) == float(weight_combo["w_disr"])
        ):
            return row
    return None


def pick_survivors(
    policy_rows: list[dict[str, Any]], args: argparse.Namespace
) -> list[dict[str, Any]]:
    survivors: list[dict[str, Any]] = []
    combos = {
        (float(row["w_bo"]), float(row["w_cost"]), float(row["w_disr"]))
        for row in policy_rows
        if row["phase"] == "static_screen"
    }
    for w_bo, w_cost, w_disr in sorted(combos):
        weight_combo = {"w_bo": w_bo, "w_cost": w_cost, "w_disr": w_disr}
        s1_row = policy_lookup(
            policy_rows,
            "static_screen",
            "static_s1",
            weight_combo,
            algo=args.algo,
            frame_stack=int(args.frame_stack),
            observation_version=str(args.observation_version),
        )
        s2_row = policy_lookup(
            policy_rows,
            "static_screen",
            "static_s2",
            weight_combo,
            algo=args.algo,
            frame_stack=int(args.frame_stack),
            observation_version=str(args.observation_version),
        )
        s3_row = policy_lookup(
            policy_rows,
            "static_screen",
            "static_s3",
            weight_combo,
            algo=args.algo,
            frame_stack=int(args.frame_stack),
            observation_version=str(args.observation_version),
        )
        if s1_row is None or s2_row is None or s3_row is None:
            continue

        # Gather all baseline candidates (static + heuristic).
        baseline_candidates: list[dict[str, Any]] = [s1_row, s2_row, s3_row]
        for h_name in HEURISTIC_POLICY_NAMES:
            h_row = policy_lookup(
                policy_rows,
                "heuristic_eval",
                h_name,
                weight_combo,
                algo=args.algo,
                frame_stack=int(args.frame_stack),
                observation_version=str(args.observation_version),
            )
            if h_row is not None:
                baseline_candidates.append(h_row)

        best_baseline = max(
            baseline_candidates, key=lambda row: float(row["reward_total_mean"])
        )
        reward_gap = float(best_baseline["reward_total_mean"]) - float(
            s1_row["reward_total_mean"]
        )
        fill_gap = float(best_baseline["fill_rate_mean"]) - float(
            s1_row["fill_rate_mean"]
        )
        if (
            best_baseline["policy"] != "static_s1"
            and reward_gap > SURVIVOR_REWARD_MARGIN
            and fill_gap > SURVIVOR_FILL_RATE_MARGIN
        ):
            survivors.append(
                {
                    **weight_combo,
                    "best_static_policy": best_baseline["policy"],
                    "static_reward_gap_best_minus_s1": reward_gap,
                    "static_fill_rate_gap_best_minus_s1": fill_gap,
                }
            )
    survivors.sort(
        key=lambda row: float(row["static_reward_gap_best_minus_s1"]), reverse=True
    )
    return survivors[: args.max_survivors]


def compare_policy_to_baseline(
    policy_row: dict[str, Any] | None, baseline_row: dict[str, Any] | None
) -> bool:
    if policy_row is None or baseline_row is None:
        return False
    return (
        float(policy_row["reward_total_mean"])
        > float(baseline_row["reward_total_mean"])
        and float(policy_row["fill_rate_mean"])
        >= float(baseline_row["fill_rate_mean"]) - PPO_SERVICE_TOLERANCE
        and float(policy_row["backorder_rate_mean"])
        <= float(baseline_row["backorder_rate_mean"]) + PPO_SERVICE_TOLERANCE
    )


def build_comparison_rows(
    policy_rows: list[dict[str, Any]],
    survivors: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    comparison_rows: list[dict[str, Any]] = []
    learned_policy = learned_policy_name(args)
    learned_phase = learned_phase_name(args)
    for survivor in survivors:
        weight_combo = {
            "w_bo": survivor["w_bo"],
            "w_cost": survivor["w_cost"],
            "w_disr": survivor["w_disr"],
        }
        learned_row = policy_lookup(
            policy_rows,
            learned_phase,
            learned_policy,
            weight_combo,
            algo=args.algo,
            frame_stack=int(args.frame_stack),
            observation_version=str(args.observation_version),
        )
        random_row = policy_lookup(
            policy_rows,
            "random_eval",
            RANDOM_POLICY_NAME,
            weight_combo,
            algo=args.algo,
            frame_stack=int(args.frame_stack),
            observation_version=str(args.observation_version),
        )
        s2_row = policy_lookup(
            policy_rows,
            "static_screen",
            "static_s2",
            weight_combo,
            algo=args.algo,
            frame_stack=int(args.frame_stack),
            observation_version=str(args.observation_version),
        )
        best_row = policy_lookup(
            policy_rows,
            "static_screen",
            survivor["best_static_policy"],
            weight_combo,
            algo=args.algo,
            frame_stack=int(args.frame_stack),
            observation_version=str(args.observation_version),
        )
        # If best_static_policy is a heuristic, look it up in heuristic_eval.
        if (
            best_row is None
            and survivor["best_static_policy"] in HEURISTIC_POLICY_NAMES
        ):
            best_row = policy_lookup(
                policy_rows,
                "heuristic_eval",
                survivor["best_static_policy"],
                weight_combo,
                algo=args.algo,
                frame_stack=int(args.frame_stack),
                observation_version=str(args.observation_version),
            )

        # Find the best heuristic policy for this weight combo.
        heuristic_rows_for_combo: list[dict[str, Any]] = []
        for h_name in HEURISTIC_POLICY_NAMES:
            h_row = policy_lookup(
                policy_rows,
                "heuristic_eval",
                h_name,
                weight_combo,
                algo=args.algo,
                frame_stack=int(args.frame_stack),
                observation_version=str(args.observation_version),
            )
            if h_row is not None:
                heuristic_rows_for_combo.append(h_row)
        best_heuristic_row = (
            max(heuristic_rows_for_combo, key=lambda r: float(r["reward_total_mean"]))
            if heuristic_rows_for_combo
            else None
        )

        # Best overall baseline = best of (best_static, best_heuristic).
        best_baseline_row = best_row
        if best_heuristic_row is not None:
            if best_baseline_row is None or float(
                best_heuristic_row["reward_total_mean"]
            ) > float(best_baseline_row["reward_total_mean"]):
                best_baseline_row = best_heuristic_row

        comparison_rows.append(
            {
                "algo": args.algo,
                "frame_stack": int(args.frame_stack),
                "observation_version": str(args.observation_version),
                "learned_policy": learned_policy,
                "w_bo": survivor["w_bo"],
                "w_cost": survivor["w_cost"],
                "w_disr": survivor["w_disr"],
                "best_static_policy": survivor["best_static_policy"],
                "static_reward_gap_best_minus_s1": survivor[
                    "static_reward_gap_best_minus_s1"
                ],
                "random_reward_mean": (
                    float(random_row["reward_total_mean"]) if random_row else None
                ),
                "random_fill_rate_mean": (
                    float(random_row["fill_rate_mean"]) if random_row else None
                ),
                "random_backorder_rate_mean": (
                    float(random_row["backorder_rate_mean"]) if random_row else None
                ),
                "random_ret_thesis_corrected_total_mean": (
                    float(random_row["ret_thesis_corrected_total_mean"])
                    if random_row
                    else None
                ),
                "learned_reward_mean": (
                    float(learned_row["reward_total_mean"]) if learned_row else None
                ),
                "learned_fill_rate_mean": (
                    float(learned_row["fill_rate_mean"]) if learned_row else None
                ),
                "learned_backorder_rate_mean": (
                    float(learned_row["backorder_rate_mean"]) if learned_row else None
                ),
                "learned_ret_thesis_corrected_total_mean": (
                    float(learned_row["ret_thesis_corrected_total_mean"])
                    if learned_row
                    else None
                ),
                "ppo_reward_mean": (
                    float(learned_row["reward_total_mean"]) if learned_row else None
                ),
                "static_s2_reward_mean": (
                    float(s2_row["reward_total_mean"]) if s2_row else None
                ),
                "best_static_reward_mean": (
                    float(best_row["reward_total_mean"]) if best_row else None
                ),
                "ppo_fill_rate_mean": (
                    float(learned_row["fill_rate_mean"]) if learned_row else None
                ),
                "static_s2_fill_rate_mean": (
                    float(s2_row["fill_rate_mean"]) if s2_row else None
                ),
                "best_static_fill_rate_mean": (
                    float(best_row["fill_rate_mean"]) if best_row else None
                ),
                "ppo_backorder_rate_mean": (
                    float(learned_row["backorder_rate_mean"]) if learned_row else None
                ),
                "static_s2_backorder_rate_mean": (
                    float(s2_row["backorder_rate_mean"]) if s2_row else None
                ),
                "best_static_backorder_rate_mean": (
                    float(best_row["backorder_rate_mean"]) if best_row else None
                ),
                "ppo_ret_thesis_corrected_total_mean": (
                    float(learned_row["ret_thesis_corrected_total_mean"])
                    if learned_row
                    else None
                ),
                "static_s2_ret_thesis_corrected_total_mean": (
                    float(s2_row["ret_thesis_corrected_total_mean"]) if s2_row else None
                ),
                "best_static_ret_thesis_corrected_total_mean": (
                    float(best_row["ret_thesis_corrected_total_mean"])
                    if best_row
                    else None
                ),
                "ppo_pct_steps_S1_mean": (
                    float(learned_row["pct_steps_S1_mean"]) if learned_row else None
                ),
                "ppo_pct_steps_S2_mean": (
                    float(learned_row["pct_steps_S2_mean"]) if learned_row else None
                ),
                "ppo_pct_steps_S3_mean": (
                    float(learned_row["pct_steps_S3_mean"]) if learned_row else None
                ),
                "best_heuristic_policy": (
                    str(best_heuristic_row["policy"]) if best_heuristic_row else None
                ),
                "best_heuristic_reward_mean": (
                    float(best_heuristic_row["reward_total_mean"])
                    if best_heuristic_row
                    else None
                ),
                "best_heuristic_fill_rate_mean": (
                    float(best_heuristic_row["fill_rate_mean"])
                    if best_heuristic_row
                    else None
                ),
                "best_baseline_policy": (
                    str(best_baseline_row["policy"]) if best_baseline_row else None
                ),
                "best_baseline_reward_mean": (
                    float(best_baseline_row["reward_total_mean"])
                    if best_baseline_row
                    else None
                ),
                "learned_beats_random": compare_policy_to_baseline(
                    learned_row, random_row
                ),
                "learned_beats_best_static": compare_policy_to_baseline(
                    learned_row, best_row
                ),
                "learned_beats_best_heuristic": compare_policy_to_baseline(
                    learned_row, best_heuristic_row
                ),
                "learned_beats_best_baseline": compare_policy_to_baseline(
                    learned_row, best_baseline_row
                ),
                "ppo_beats_static_s2": compare_policy_to_baseline(learned_row, s2_row),
                "ppo_beats_best_static": compare_policy_to_baseline(
                    learned_row, best_row
                ),
                "collapsed_to_S1": bool(
                    learned_row and float(learned_row["pct_steps_S1_mean"]) > 90.0
                ),
                "collapsed_to_S2": bool(
                    learned_row and float(learned_row["pct_steps_S2_mean"]) > 90.0
                ),
                "collapsed_to_S3": bool(
                    learned_row and float(learned_row["pct_steps_S3_mean"]) > 90.0
                ),
            }
        )
    return comparison_rows


# ---------------------------------------------------------------------------
# Heuristic tuning via grid search
# ---------------------------------------------------------------------------

TUNE_GRID: dict[str, list[float]] = {
    "tau_high": [0.08, 0.12, 0.20],
    "tau_low": [0.03, 0.05, 0.08],
    "fill_rate_caution": [0.85, 0.90, 0.95],
    "boost_normal": [-0.3, 0.0, 0.3],
    "boost_caution": [0.3, 0.5, 0.7],
    "boost_crisis": [0.7, 0.85, 1.0],
}


def tune_heuristic_params(
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Grid-search ``HeuristicTuned`` parameters on training seeds.

    Evaluates each parameter combo for ``args.tune_episodes`` episodes per
    seed, selects the combo with the highest mean cumulative reward, and
    returns a dict with the best parameters and tuning metadata.
    """
    import itertools

    weight_combo = {
        "w_bo": float(args.w_bo[0]),
        "w_cost": float(args.w_cost[0]),
        "w_disr": float(args.w_disr[0]),
    }
    param_names = list(TUNE_GRID.keys())
    param_combos = list(itertools.product(*TUNE_GRID.values()))

    best_reward = float("-inf")
    best_params: dict[str, float] = {}
    results: list[dict[str, Any]] = []

    for combo_vals in param_combos:
        params = dict(zip(param_names, combo_vals))
        if params["tau_low"] >= params["tau_high"]:
            continue  # invalid: deadband must have tau_low < tau_high
        heuristic = HeuristicTuned(**params)
        rewards: list[float] = []
        for seed in args.seeds:
            env_kwargs = build_env_kwargs(args, weight_combo)
            env = make_shift_control_env(**env_kwargs)
            env = wrap_env_for_benchmark(env, args)
            for ep in range(args.tune_episodes):
                eval_seed = seed + EVAL_EPISODE_SEED_OFFSET + ep
                heuristic.reset()
                obs, reset_info = env.reset(seed=eval_seed)
                prev_info: dict[str, Any] = (
                    reset_info if isinstance(reset_info, dict) else {}
                )
                terminated = False
                truncated = False
                reward_total = 0.0
                while not (terminated or truncated):
                    action = heuristic(obs, prev_info)
                    obs, reward, terminated, truncated, info = env.step(action)
                    prev_info = info
                    reward_total += float(reward)
                rewards.append(reward_total)
            env.close()

        mean_reward = float(np.mean(rewards))
        results.append(
            {**params, "mean_reward": mean_reward, "n_episodes": len(rewards)}
        )
        if mean_reward > best_reward:
            best_reward = mean_reward
            best_params = params

    # Inject best params into the global defaults for use in the benchmark.
    HEURISTIC_DEFAULTS["heuristic_tuned"] = HeuristicTuned(**best_params)

    return {
        "best_params": best_params,
        "best_mean_reward": best_reward,
        "combos_evaluated": len(results),
        "weight_combo_used": weight_combo,
        "seeds": args.seeds,
        "tune_episodes": args.tune_episodes,
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    ensure_algo_dependencies(args)
    args.output_dir = resolve_output_dir(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    heuristic_tuning_result: dict[str, Any] | None = None
    if getattr(args, "tune_heuristic", False):
        heuristic_tuning_result = tune_heuristic_params(args)

    weight_combos = make_weight_combos(args)
    episode_rows: list[dict[str, Any]] = []
    trained_models: list[dict[str, Any]] = []

    for weight_combo in weight_combos:
        for seed in args.seeds:
            for policy in STATIC_POLICY_ORDER:
                episode_rows.extend(
                    evaluate_policy(
                        "static_screen",
                        policy,
                        args=args,
                        weight_combo=weight_combo,
                        seed=seed,
                    )
                )
            episode_rows.extend(
                evaluate_policy(
                    "random_eval",
                    RANDOM_POLICY_NAME,
                    args=args,
                    weight_combo=weight_combo,
                    seed=seed,
                )
            )
            for policy in HEURISTIC_POLICY_NAMES:
                episode_rows.extend(
                    evaluate_policy(
                        "heuristic_eval",
                        policy,
                        args=args,
                        weight_combo=weight_combo,
                        seed=seed,
                    )
                )

    seed_rows = aggregate_seed_metrics(episode_rows)
    policy_rows = aggregate_policy_metrics(seed_rows)
    survivors = pick_survivors(policy_rows, args)

    for survivor in survivors:
        weight_combo = {
            "w_bo": survivor["w_bo"],
            "w_cost": survivor["w_cost"],
            "w_disr": survivor["w_disr"],
        }
        for seed in args.seeds:
            model, vec_env = train_model(args, seed, weight_combo)
            trained_models.append(
                {
                    "algo": args.algo,
                    "frame_stack": int(args.frame_stack),
                    "seed": seed,
                    "w_bo": survivor["w_bo"],
                    "w_cost": survivor["w_cost"],
                    "w_disr": survivor["w_disr"],
                    "train_timesteps": args.train_timesteps,
                }
            )
            episode_rows.extend(
                evaluate_policy(
                    learned_phase_name(args),
                    learned_policy_name(args),
                    args=args,
                    weight_combo=weight_combo,
                    seed=seed,
                    model=model,
                )
            )
            vec_env.close()

    # Cross-scenario evaluation: re-evaluate all policies on additional risk levels.
    eval_risk_levels = getattr(args, "eval_risk_levels", None) or []
    cross_eval_levels = [rl for rl in eval_risk_levels if rl != args.risk_level]
    for eval_rl in cross_eval_levels:
        cross_phase = f"cross_eval_{eval_rl}"
        for weight_combo_d in weight_combos:
            for seed in args.seeds:
                for pol in STATIC_POLICY_ORDER:
                    episode_rows.extend(
                        evaluate_policy(
                            cross_phase,
                            pol,
                            args=args,
                            weight_combo=weight_combo_d,
                            seed=seed,
                            risk_level_override=eval_rl,
                        )
                    )
                episode_rows.extend(
                    evaluate_policy(
                        cross_phase,
                        RANDOM_POLICY_NAME,
                        args=args,
                        weight_combo=weight_combo_d,
                        seed=seed,
                        risk_level_override=eval_rl,
                    )
                )
                for pol in HEURISTIC_POLICY_NAMES:
                    episode_rows.extend(
                        evaluate_policy(
                            cross_phase,
                            pol,
                            args=args,
                            weight_combo=weight_combo_d,
                            seed=seed,
                            risk_level_override=eval_rl,
                        )
                    )

    seed_rows = aggregate_seed_metrics(episode_rows)
    policy_rows = aggregate_policy_metrics(seed_rows)
    comparison_rows = build_comparison_rows(policy_rows, survivors, args=args)

    episode_csv = args.output_dir / "episode_metrics.csv"
    policy_csv = args.output_dir / "policy_summary.csv"
    comparison_csv = args.output_dir / "comparison_table.csv"
    summary_json = args.output_dir / "summary.json"

    save_csv(episode_csv, episode_rows, EPISODE_FIELDNAMES)
    save_csv(policy_csv, policy_rows, POLICY_SUMMARY_FIELDNAMES)
    save_csv(comparison_csv, comparison_rows, COMPARISON_FIELDNAMES)

    summary = {
        "config": {
            "algo": args.algo,
            "frame_stack": int(args.frame_stack),
            "observation_version": str(args.observation_version),
            "reward_mode": "control_v1",
            "seeds": args.seeds,
            "train_timesteps": args.train_timesteps,
            "eval_episodes": args.eval_episodes,
            "step_size_hours": args.step_size_hours,
            "max_steps": args.max_steps,
            "risk_level": args.risk_level,
            "stochastic_pt": args.stochastic_pt,
            "year_basis": args.year_basis,
            "w_bo": args.w_bo,
            "w_cost": args.w_cost,
            "w_disr": args.w_disr,
            "max_survivors": args.max_survivors,
            "learning_rate": args.learning_rate,
            "n_steps": args.n_steps,
            "batch_size": args.batch_size,
            "n_epochs": args.n_epochs,
            "gamma": args.gamma,
            "gae_lambda": args.gae_lambda,
            "clip_range": args.clip_range,
            "tau": args.tau,
            "train_freq": args.train_freq,
            "gradient_steps": args.gradient_steps,
            "learning_starts": args.learning_starts,
        },
        "benchmark_metadata": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": resolve_git_commit(),
            "command": resolve_invocation(args),
        },
        "phases": [
            "static_screen",
            "heuristic_eval",
            "random_eval",
            learned_phase_name(args),
        ],
        "policies": [
            *STATIC_POLICY_ORDER,
            *HEURISTIC_POLICY_NAMES,
            RANDOM_POLICY_NAME,
            learned_policy_name(args),
        ],
        "weight_combinations": weight_combos,
        "survivors": survivors,
        "trained_models": trained_models,
        "ppo_skipped": not survivors,
        "artifacts": {
            "episode_metrics_csv": str(episode_csv),
            "policy_summary_csv": str(policy_csv),
            "comparison_table_csv": str(comparison_csv),
            "summary_json": str(summary_json),
        },
        "policy_summary": policy_rows,
        "comparison_table": comparison_rows,
    }
    if heuristic_tuning_result is not None:
        summary["heuristic_tuning"] = heuristic_tuning_result
    with summary_json.open("w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, indent=2)

    if not args.skip_artifact_export:
        artifact_label = args.artifact_label or args.output_dir.name
        bundle_dir = export_artifact_bundle(
            source_dir=args.output_dir,
            artifact_root=args.artifact_root,
            label=artifact_label,
            summary=summary,
            command=resolve_invocation(args),
        )
        summary["artifacts"]["artifact_bundle_dir"] = str(bundle_dir)
        with summary_json.open("w", encoding="utf-8") as file_obj:
            json.dump(summary, file_obj, indent=2)
    return summary


def main() -> None:
    args = build_parser().parse_args()
    args.invocation = "python scripts/benchmark_control_reward.py " + " ".join(
        sys.argv[1:]
    )
    summary = run_benchmark(args)
    print(f"Wrote control reward benchmark artifacts to {args.output_dir}")
    if "artifact_bundle_dir" in summary["artifacts"]:
        print(
            f"Exported auditable bundle to {summary['artifacts']['artifact_bundle_dir']}"
        )
    print(f"Survivors forwarded to {args.algo.upper()}: {len(summary['survivors'])}")
    for row in summary["comparison_table"]:
        print(
            f"w_bo={row['w_bo']:.2f}, w_cost={row['w_cost']:.2f}, "
            f"obs={row['observation_version']}, "
            f"best_static={row['best_static_policy']}, "
            f"ppo_beats_best_static={row['ppo_beats_best_static']}, "
            f"collapsed_to_S1={row['collapsed_to_S1']}"
        )


if __name__ == "__main__":
    main()
