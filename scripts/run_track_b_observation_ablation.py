#!/usr/bin/env python3
"""Track B observation ablation: full v7 vs no-forecast vs v5.

This isolates whether PPO's gains come from explicit forward-looking signals
(`risk_forecast_48h_norm`, `risk_forecast_168h_norm`) or mainly from adaptive
reaction to downstream state.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_track_b_smoke import build_parser as smoke_build_parser, run_smoke
from supply_chain.external_env_interface import get_observation_fields


FORECAST_FIELD_NAMES = ("risk_forecast_48h_norm", "risk_forecast_168h_norm")


class ForecastMaskWrapper(gym.ObservationWrapper):
    """Mask only the explicit risk-forecast channels while preserving shape."""

    def __init__(self, env: gym.Env[np.ndarray, np.ndarray]) -> None:
        super().__init__(env)
        fields = tuple(get_observation_fields("v7"))
        self._forecast_indices = tuple(fields.index(name) for name in FORECAST_FIELD_NAMES)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        masked = np.array(observation, dtype=np.float32, copy=True)
        for idx in self._forecast_indices:
            masked[idx] = 0.0
        return masked


@dataclass(frozen=True)
class ObservationAblationConfig:
    label: str
    observation_version: str
    wrapper: type[gym.ObservationWrapper] | None


OBS_ABLATION_CONFIGS: dict[str, ObservationAblationConfig] = {
    "v7_full": ObservationAblationConfig(
        label="v7_full",
        observation_version="v7",
        wrapper=None,
    ),
    "v7_no_forecast": ObservationAblationConfig(
        label="v7_no_forecast",
        observation_version="v7",
        wrapper=ForecastMaskWrapper,
    ),
    "v5_7d": ObservationAblationConfig(
        label="v5_7d",
        observation_version="v5",
        wrapper=None,
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = smoke_build_parser()
    parser.description = (
        "Track B observation ablation: compare v7 full, v7 with forecast inputs "
        "masked, and v5 with the same 7D action contract."
    )
    parser.add_argument(
        "--obs-configs",
        nargs="+",
        choices=list(OBS_ABLATION_CONFIGS.keys()),
        default=list(OBS_ABLATION_CONFIGS.keys()),
        help="Observation configurations to run.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    base_output = args.output_dir or Path("outputs/benchmarks/track_b_observation_ablation")
    base_output.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict[str, object]] = {}
    for config_name in args.obs_configs:
        config = OBS_ABLATION_CONFIGS[config_name]
        print(f"\n{'=' * 60}")
        print(f"OBSERVATION ABLATION: {config_name}")
        print(f"{'=' * 60}")
        config_args = argparse.Namespace(**vars(args))
        config_args.output_dir = base_output / config_name
        config_args.observation_version = config.observation_version
        config_args._observation_wrapper = config.wrapper
        config_args.invocation = (
            f"python scripts/run_track_b_observation_ablation.py --obs-configs {config_name}"
        )
        summary = run_smoke(config_args)
        results[config_name] = summary

    print(f"\n{'=' * 60}")
    print("OBSERVATION ABLATION COMPARISON")
    print(f"{'=' * 60}")
    print(
        f"{'Config':<18} {'Obs':<6} {'PPO Fill':>10} {'PPO ReT':>10} "
        f"{'S1%':>6} {'S2%':>6} {'S3%':>6}"
    )
    print("-" * 70)
    for config_name, summary in results.items():
        ppo_row = next(
            row for row in summary["policy_summary"] if row["policy"] in ("ppo", "recurrent_ppo")
        )
        print(
            f"{config_name:<18} "
            f"{summary['config']['observation_version']:<6} "
            f"{float(ppo_row['fill_rate_mean']):>10.6f} "
            f"{float(ppo_row['order_level_ret_mean_mean']):>10.4f} "
            f"{float(ppo_row['pct_steps_S1_mean']):>6.1f} "
            f"{float(ppo_row['pct_steps_S2_mean']):>6.1f} "
            f"{float(ppo_row['pct_steps_S3_mean']):>6.1f}"
        )


if __name__ == "__main__":
    main()
