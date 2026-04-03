#!/usr/bin/env python3
"""Track B ablation: shift-only vs downstream-only vs joint control.

Trains PPO under three action-space configurations to determine which
control dimension (assembly shifts vs downstream dispatch) matters most.

Configurations:
  joint           — full 7D action space (baseline, unchanged)
  shift_only      — dims 5-6 (op10/op12) frozen at neutral (signal=0 → mult=1.25)
  downstream_only — dim 4 (shift) frozen at S2 (signal=0)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_track_b_smoke import build_parser as smoke_build_parser, run_smoke


class ShiftOnlyWrapper(gym.ActionWrapper):
    """Freeze downstream multiplier signals to neutral (0.0 → 1.25x).

    The agent can only control dims 0-4 (inventory + shifts).
    Dims 5-6 are forced to 0.0 each step.
    """

    def action(self, action: np.ndarray) -> np.ndarray:
        modified = np.array(action, dtype=np.float32)
        modified[5] = 0.0
        modified[6] = 0.0
        return modified


class DownstreamOnlyWrapper(gym.ActionWrapper):
    """Freeze shift signal to S2 (0.0).

    The agent can only control dims 0-3 (inventory) and dims 5-6 (downstream).
    Dim 4 is forced to 0.0 (S2) each step.
    """

    def action(self, action: np.ndarray) -> np.ndarray:
        modified = np.array(action, dtype=np.float32)
        modified[4] = 0.0
        return modified


ABLATION_CONFIGS = {
    "joint": None,  # No wrapper needed
    "shift_only": ShiftOnlyWrapper,
    "downstream_only": DownstreamOnlyWrapper,
}


def build_parser() -> argparse.ArgumentParser:
    parser = smoke_build_parser()
    parser.description = (
        "Track B ablation study: compare joint vs shift-only vs downstream-only "
        "control under the same training protocol."
    )
    parser.add_argument(
        "--ablation-configs",
        nargs="+",
        choices=list(ABLATION_CONFIGS.keys()),
        default=list(ABLATION_CONFIGS.keys()),
        help="Which ablation configurations to run.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    base_output = args.output_dir or Path("outputs/benchmarks/track_b_ablation")
    base_output.mkdir(parents=True, exist_ok=True)

    results = {}
    for config_name in args.ablation_configs:
        print(f"\n{'='*60}")
        print(f"ABLATION: {config_name}")
        print(f"{'='*60}")
        config_args = argparse.Namespace(**vars(args))
        config_args.output_dir = base_output / config_name
        config_args.invocation = (
            f"python scripts/run_track_b_ablation.py --ablation-configs {config_name}"
        )
        # Store wrapper class for use by the training env factory
        config_args._ablation_wrapper = ABLATION_CONFIGS[config_name]
        summary = run_smoke(config_args)
        results[config_name] = summary

    # Print comparison
    print(f"\n{'='*60}")
    print("ABLATION COMPARISON")
    print(f"{'='*60}")
    print(f"{'Config':<20} {'PPO Fill':>10} {'PPO ReT':>10} {'S1%':>6} {'S2%':>6} {'S3%':>6}")
    print("-" * 60)
    for config_name, summary in results.items():
        ppo_row = None
        for row in summary["policy_summary"]:
            if row["policy"] in ("ppo", "recurrent_ppo"):
                ppo_row = row
                break
        if ppo_row:
            print(
                f"{config_name:<20} "
                f"{float(ppo_row['fill_rate_mean']):>10.6f} "
                f"{float(ppo_row['order_level_ret_mean_mean']):>10.4f} "
                f"{float(ppo_row['pct_steps_S1_mean']):>6.1f} "
                f"{float(ppo_row['pct_steps_S2_mean']):>6.1f} "
                f"{float(ppo_row['pct_steps_S3_mean']):>6.1f}"
            )


if __name__ == "__main__":
    main()
