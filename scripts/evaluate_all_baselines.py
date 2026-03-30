#!/usr/bin/env python3
"""
Evaluate all baselines under a given reward mode and produce paper table.

This script is reward-mode agnostic: works with any reward mode.
Uses run_episodes() from external_env_interface — no benchmark internals.

Usage:
    python scripts/evaluate_all_baselines.py --reward-mode ReT_seq_v1
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

from supply_chain.config import OPERATIONS
from supply_chain.external_env_interface import run_episodes

# ---------------------------------------------------------------------------
# Policy definitions
# ---------------------------------------------------------------------------

SHIFT_ACTIONS = {"s1": -1.0, "s2": 0.0, "s3": 1.0}


def static_policy(shift_level: str):
    """Return a callable for a static shift policy."""
    action = np.array([0.0, 0.0, 0.0, 0.0, SHIFT_ACTIONS[shift_level]], dtype=np.float32)
    return lambda obs, info: action


def garrido_cf_policy(shift_level: str):
    """
    Return a callable for Garrido's exact thesis configuration.

    Uses thesis-exact dispatch quantities (no 1.25x multiplier).
    The action bypasses the multiplier mapping by setting the raw
    action to produce multiplier=1.0: signal = (1.0 - 1.25) / 0.75 = -0.333
    """
    unity_signal = (1.0 - 1.25) / 0.75  # ≈ -0.333, produces multiplier=1.0
    action = np.array(
        [unity_signal, unity_signal, unity_signal, unity_signal, SHIFT_ACTIONS[shift_level]],
        dtype=np.float32,
    )
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
    "random": random_policy(),
}

METRIC_FIELDS = [
    "policy", "reward_total_mean", "reward_total_std",
    "fill_rate_mean", "fill_rate_std",
    "backorder_rate_mean", "backorder_rate_std",
    "pct_steps_S1_mean", "pct_steps_S2_mean", "pct_steps_S3_mean",
    "ret_thesis_corrected_total_mean",
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
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate all baselines under a reward mode")
    parser.add_argument("--reward-mode", default="ReT_seq_v1")
    parser.add_argument("--risk-level", default="increased")
    parser.add_argument("--stochastic-pt", action="store_true", default=True)
    parser.add_argument("--observation-version", default="v1")
    parser.add_argument("--ret-seq-kappa", type=float, default=0.20)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/baseline_evaluation"))
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
        f.write(f"# Baseline Evaluation\n\n")
        f.write(f"Reward mode: `{args.reward_mode}` | Risk: `{args.risk_level}` | "
                f"Episodes: {args.episodes} | Seed: {args.seed}\n\n")
        f.write("| Policy | Reward | Fill Rate | Backorder | S1% | S2% | S3% |\n")
        f.write("|--------|--------|-----------|-----------|-----|-----|-----|\n")
        for r in sorted(rows, key=lambda x: -x["reward_total_mean"]):
            f.write(f"| {r['policy']} | {r['reward_total_mean']:.1f}±{r['reward_total_std']:.1f} "
                    f"| {r['fill_rate_mean']:.3f} | {r['backorder_rate_mean']:.3f} "
                    f"| {r['pct_steps_S1_mean']:.0f} | {r['pct_steps_S2_mean']:.0f} "
                    f"| {r['pct_steps_S3_mean']:.0f} |\n")

    # Save config
    with open(args.output_dir / "config.json", "w") as f:
        json.dump({"env_kwargs": env_kwargs, "episodes": args.episodes, "seed": args.seed}, f, indent=2)

    print(f"\nSaved to {csv_path} and {md_path}")


if __name__ == "__main__":
    main()
