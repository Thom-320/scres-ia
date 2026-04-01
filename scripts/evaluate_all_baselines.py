#!/usr/bin/env python3
"""
Evaluate all baselines under a given reward mode and produce paper table.

This script is reward-mode agnostic: works with any reward mode.
Uses run_episodes() from external_env_interface — no benchmark internals.

Usage:
    python scripts/evaluate_all_baselines.py --reward-mode ReT_unified_v1
    python scripts/evaluate_all_baselines.py --reward-mode ReT_unified_v1 --risk-level severe
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.benchmark_control_reward import HeuristicCycleGuard
from supply_chain.config import (
    BENCHMARK_OBSERVATION_VERSION,
    BENCHMARK_REWARD_MODE,
    CAPACITY_BY_SHIFTS,
    OPERATIONS,
)
from supply_chain.external_env_interface import run_episodes

# ---------------------------------------------------------------------------
# Policy definitions
# ---------------------------------------------------------------------------

SHIFT_ACTIONS = {"s1": -1.0, "s2": 0.0, "s3": 1.0}


def static_policy(shift_level: str):
    """Return a callable for a static shift policy."""
    action = np.array(
        [0.0, 0.0, 0.0, 0.0, SHIFT_ACTIONS[shift_level]], dtype=np.float32
    )
    return lambda obs, info: action


def garrido_cf_policy(shift_level: str):
    """
    Return a callable for Garrido's exact thesis configuration.

    Uses direct DES action bypass so the baseline matches the benchmark's
    thesis-exact capacity configuration rather than the approximate 5D action
    signal that routes through the 1.25 + 0.75 * signal multiplier mapping.
    """
    shift = {"s1": 1, "s2": 2, "s3": 3}[shift_level]
    action = {
        "assembly_shifts": shift,
        "op3_q": float(CAPACITY_BY_SHIFTS[shift]["op3_q"]),
        "op3_rop": float(OPERATIONS[3]["rop"]),
        "op9_q_min": float(OPERATIONS[9]["q"][0]),
        "op9_q_max": float(OPERATIONS[9]["q"][1]),
        "op9_rop": float(OPERATIONS[9]["rop"]),
        "batch_size": float(CAPACITY_BY_SHIFTS[shift]["op7_q"]),
    }
    return lambda obs, info: action


def random_policy(seed: int = 42):
    """Return a callable for random actions."""
    rng = np.random.default_rng(seed)
    return lambda obs, info: rng.uniform(-1.0, 1.0, size=5).astype(np.float32)


def heuristic_disruption_policy():
    """Simple heuristic: upshift when any disruption is active."""

    def policy(obs, info):
        assembly_down = obs[8] > 0.5 if len(obs) > 8 else False
        any_down = obs[9] > 0.5 if len(obs) > 9 else False
        if assembly_down or any_down:
            return np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)  # S3
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)  # S2

    return policy


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ALL_POLICIES = {
    "static_s1": static_policy("s1"),
    "static_s2": static_policy("s2"),
    "static_s3": static_policy("s3"),
    "garrido_cf_s1": garrido_cf_policy("s1"),
    "garrido_cf_s2": garrido_cf_policy("s2"),
    "garrido_cf_s3": garrido_cf_policy("s3"),
    "heuristic_disruption": heuristic_disruption_policy(),
    "heuristic_cycle_guard": HeuristicCycleGuard(),
    "random": random_policy(),
}

METRIC_FIELDS = [
    "policy",
    "reward_total_mean",
    "reward_total_std",
    "fill_rate_mean",
    "fill_rate_std",
    "backorder_rate_mean",
    "backorder_rate_std",
    "pct_steps_S1_mean",
    "pct_steps_S2_mean",
    "pct_steps_S3_mean",
    "ret_thesis_corrected_total_mean",
    "ret_unified_total_mean",
    "ret_garrido2024_sigmoid_total_mean",
]


def evaluate_policy(
    policy_name: str,
    policy_fn,
    env_kwargs: dict,
    n_episodes: int,
    seed: int,
) -> dict:
    """Evaluate a single policy and return aggregated metrics."""
    results = run_episodes(
        policy_fn,
        n_episodes=n_episodes,
        seed=seed,
        env_kwargs=env_kwargs,
        policy_name=policy_name,
    )

    rewards = [r["reward_total"] for r in results]
    fill_rates = [r["fill_rate"] for r in results]
    bo_rates = [r["backorder_rate"] for r in results]
    s1 = [r.get("pct_steps_S1", 0) for r in results]
    s2 = [r.get("pct_steps_S2", 0) for r in results]
    s3 = [r.get("pct_steps_S3", 0) for r in results]
    ret_corr = [r.get("ret_thesis_corrected_total", 0) for r in results]
    ret_unified = [r.get("ret_unified_total", 0) for r in results]
    ret_garrido = [r.get("ret_garrido2024_sigmoid_total", 0) for r in results]

    return {
        "policy": policy_name,
        "reward_total_mean": float(np.mean(rewards)),
        "reward_total_std": float(np.std(rewards)),
        "fill_rate_mean": float(np.mean(fill_rates)),
        "fill_rate_std": float(np.std(fill_rates)),
        "backorder_rate_mean": float(np.mean(bo_rates)),
        "backorder_rate_std": float(np.std(bo_rates)),
        "pct_steps_S1_mean": float(np.mean(s1)),
        "pct_steps_S2_mean": float(np.mean(s2)),
        "pct_steps_S3_mean": float(np.mean(s3)),
        "ret_thesis_corrected_total_mean": float(np.mean(ret_corr)),
        "ret_unified_total_mean": float(np.mean(ret_unified)),
        "ret_garrido2024_sigmoid_total_mean": float(np.mean(ret_garrido)),
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for baseline evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate all baselines under a reward mode"
    )
    parser.add_argument("--reward-mode", default=BENCHMARK_REWARD_MODE)
    parser.add_argument("--risk-level", default="increased")
    parser.add_argument("--stochastic-pt", action="store_true", default=True)
    parser.add_argument("--observation-version", default=BENCHMARK_OBSERVATION_VERSION)
    parser.add_argument("--ret-seq-kappa", type=float, default=0.20)
    parser.add_argument(
        "--ret-unified-calibration",
        type=Path,
        default=Path("supply_chain/data/ret_unified_v1_calibration.json"),
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/baseline_evaluation")
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    env_kwargs = {
        "reward_mode": args.reward_mode,
        "risk_level": args.risk_level,
        "stochastic_pt": args.stochastic_pt,
        "observation_version": args.observation_version,
        "step_size_hours": 168,
        "max_steps": 260,
        "year_basis": "thesis",
        "ret_seq_kappa": args.ret_seq_kappa,
        "ret_unified_calibration_path": str(args.ret_unified_calibration),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for name, policy_fn in ALL_POLICIES.items():
        print(f"Evaluating {name}...", end=" ", flush=True)
        row = evaluate_policy(name, policy_fn, env_kwargs, args.episodes, args.seed)
        rows.append(row)
        print(f"reward={row['reward_total_mean']:.2f}  FR={row['fill_rate_mean']:.3f}")

    # Save CSV
    csv_path = args.output_dir / "baseline_table.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    # Save markdown
    md_path = args.output_dir / "baseline_table.md"
    with open(md_path, "w") as f:
        f.write("# Baseline Evaluation\n\n")
        f.write(
            f"Reward mode: `{args.reward_mode}` | Risk: `{args.risk_level}` | "
            f"Observation: `{args.observation_version}` | Episodes: {args.episodes} | Seed: {args.seed}\n\n"
        )
        f.write(
            "| Policy | Reward | ReT_unified | ReT_garrido2024 | Fill Rate | Backorder | S1% | S2% | S3% |\n"
        )
        f.write(
            "|--------|--------|-------------|------------------|-----------|-----------|-----|-----|-----|\n"
        )
        for r in sorted(rows, key=lambda x: -x["reward_total_mean"]):
            f.write(
                f"| {r['policy']} | {r['reward_total_mean']:.1f}±{r['reward_total_std']:.1f} "
                f"| {r['ret_unified_total_mean']:.1f} "
                f"| {r['ret_garrido2024_sigmoid_total_mean']:.1f} "
                f"| {r['fill_rate_mean']:.3f} | {r['backorder_rate_mean']:.3f} "
                f"| {r['pct_steps_S1_mean']:.0f} | {r['pct_steps_S2_mean']:.0f} "
                f"| {r['pct_steps_S3_mean']:.0f} |\n"
            )

    # Save config
    with open(args.output_dir / "config.json", "w") as f:
        json.dump(
            {"env_kwargs": env_kwargs, "episodes": args.episodes, "seed": args.seed},
            f,
            indent=2,
        )

    print(f"\nSaved to {csv_path} and {md_path}")


if __name__ == "__main__":
    main()
