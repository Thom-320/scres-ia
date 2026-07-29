#!/usr/bin/env python3
"""E3 Track B frozen-policy cross-regime x horizon matrix.

Forces the canonical July-1 Track B v7 bundle and evaluates it at
current/increased/severe across h52 and h104. This wrapper intentionally
overrides legacy defaults from older audit scripts.
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import gymnasium as gym
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.eval_track_b_cross_scenario import build_parser, run_cross_scenario  # noqa: E402


CANONICAL_PPO_BUNDLE = Path(
    "outputs/experiments/track_b_gain_2026-06-30/"
    "top_tier_confirm_v3_output/track_b_top_tier_confirm_5seed_60k_h104"
)
DEFAULT_RISK_LEVELS = ("current", "increased", "severe")
DEFAULT_HORIZONS = (52, 104)
DEFAULT_SEEDS = (1, 2, 3, 4, 5)


class TruncateObservationWrapper(gym.ObservationWrapper):
    """Compatibility shim for frozen v7 checkpoints trained before appended fields."""

    target_dim = 48

    def __init__(self, env: gym.Env[Any, Any]) -> None:
        super().__init__(env)
        low = np.asarray(env.observation_space.low[: self.target_dim], dtype=np.float32)
        high = np.asarray(env.observation_space.high[: self.target_dim], dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return np.asarray(observation[: self.target_dim], dtype=np.float32)


def default_output_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    return Path("outputs/experiments") / f"track_b_e3_cross_regime_horizon_matrix_{stamp}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ppo-bundle", type=Path, default=CANONICAL_PPO_BUNDLE)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--risk-levels", nargs="+", default=list(DEFAULT_RISK_LEVELS))
    parser.add_argument("--horizons", nargs="+", type=int, default=list(DEFAULT_HORIZONS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--eval-episodes", type=int, default=12)
    parser.add_argument("--reward-mode", default="control_v1")
    parser.add_argument("--observation-version", default="v7")
    parser.add_argument("--step-size-hours", type=float, default=168.0)
    parser.add_argument("--include-heuristics", action="store_true", default=True)
    parser.add_argument("--skip-heuristics", action="store_true")
    return parser.parse_args()


def run_one_horizon(args: argparse.Namespace, *, horizon: int, output_dir: Path) -> dict[str, Any]:
    cross_parser = build_parser()
    cross_args = cross_parser.parse_args([])
    cross_args.ppo_bundle = args.ppo_bundle
    cross_args.recurrent_bundle = args.ppo_bundle
    cross_args.skip_recurrent = True
    cross_args.include_heuristics = bool(args.include_heuristics and not args.skip_heuristics)
    cross_args.reward_mode = args.reward_mode
    cross_args.observation_version = args.observation_version
    cross_args.risk_levels = list(args.risk_levels)
    cross_args.seeds = [int(seed) for seed in args.seeds]
    cross_args.eval_episodes = int(args.eval_episodes)
    cross_args.step_size_hours = float(args.step_size_hours)
    cross_args.max_steps = int(horizon)
    # eval_track_b_cross_scenario.build_parser() omits several reward/risk-shaping
    # flags that the shared build_env_kwargs() (run_track_b_smoke.py) reads
    # unconditionally in evaluate_heuristic_policy's code path. All are inert for
    # E3's control_v1 reward mode and non-multiplier risk levels; values mirror
    # run_track_b_smoke.py's own build_parser() defaults.
    cross_args.ret_seq_kappa = 0.20
    cross_args.ret_excel_cvar_alpha = 0.5
    cross_args.ret_excel_cvar_tail_level = 0.05
    cross_args.ret_excel_cvar_window = 50
    cross_args.enabled_risks = None
    cross_args.risk_frequency_multiplier = 1.0
    cross_args.risk_impact_multiplier = 1.0
    cross_args.demand_mean_multiplier = 1.0
    cross_args.output_dir = output_dir / f"h{horizon}"
    cross_args._observation_wrapper = infer_observation_wrapper(args.ppo_bundle, args.seeds[0])
    return run_cross_scenario(cross_args)


def saved_vec_observation_dim(bundle: Path, seed: int) -> int:
    vec_path = bundle / "models" / f"seed{seed}" / "vec_normalize.pkl"
    with vec_path.open("rb") as handle:
        vec_norm = pickle.load(handle)
    return int(vec_norm.observation_space.shape[0])


def infer_observation_wrapper(bundle: Path, seed: int) -> type[gym.ObservationWrapper] | None:
    saved_dim = saved_vec_observation_dim(bundle, int(seed))
    if saved_dim == 48:
        TruncateObservationWrapper.target_dim = 48
        return TruncateObservationWrapper
    if saved_dim == 52:
        return None
    raise ValueError(
        f"Unsupported frozen VecNormalize observation dimension {saved_dim}; "
        "add an explicit compatibility wrapper before evaluating this bundle."
    )


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: dict[str, Any] = {}
    overview_frames: list[pd.DataFrame] = []
    for horizon in args.horizons:
        summary = run_one_horizon(args, horizon=int(horizon), output_dir=output_dir)
        summaries[f"h{horizon}"] = summary
        overview_path = Path(summary["artifacts"]["risk_overview_csv"])
        frame = pd.read_csv(overview_path)
        frame.insert(0, "horizon", int(horizon))
        overview_frames.append(frame)

    matrix = pd.concat(overview_frames, ignore_index=True) if overview_frames else pd.DataFrame()
    matrix_path = output_dir / "cross_regime_horizon_matrix.csv"
    matrix.to_csv(matrix_path, index=False)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "ppo_bundle": str(args.ppo_bundle),
            "reward_mode": args.reward_mode,
            "observation_version": args.observation_version,
            "risk_levels": list(args.risk_levels),
            "horizons": [int(h) for h in args.horizons],
            "seeds": [int(seed) for seed in args.seeds],
            "eval_episodes": int(args.eval_episodes),
            "step_size_hours": float(args.step_size_hours),
            "include_heuristics": bool(args.include_heuristics and not args.skip_heuristics),
            "frozen_vec_observation_dim": saved_vec_observation_dim(args.ppo_bundle, int(args.seeds[0])),
        },
        "artifacts": {
            "cross_regime_horizon_matrix_csv": str(matrix_path.resolve()),
            "per_horizon": {
                horizon: summary["artifacts"] for horizon, summary in summaries.items()
            },
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    readme = output_dir / "README.md"
    readme.write_text(
        "# E3 Cross-Regime Horizon Matrix\n\n"
        f"- PPO bundle: `{args.ppo_bundle}`\n"
        f"- Reward mode: `{args.reward_mode}`\n"
        f"- Observation version: `{args.observation_version}`\n"
        f"- Horizons: `{', '.join('h' + str(h) for h in args.horizons)}`\n"
        f"- Matrix: `{matrix_path.name}`\n",
        encoding="utf-8",
    )
    latest = output_dir.parent / "track_b_e3_cross_regime_horizon_matrix_latest"
    if latest.exists() or latest.is_symlink():
        if latest.is_dir() and not latest.is_symlink():
            shutil.rmtree(latest)
        else:
            latest.unlink()
    latest.symlink_to(output_dir.resolve(), target_is_directory=True)
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
