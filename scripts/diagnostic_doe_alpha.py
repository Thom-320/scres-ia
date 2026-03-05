#!/usr/bin/env python3
"""
Mini-DOE: Sensitivity of reward spread to α weight.

Tests α ∈ {1, 3, 5, 8} × 5 seeds × {current, increased} risk levels.
Answers: does the spread plateau, and at what level?
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from supply_chain.env import MFSCGymEnv

POLICIES = {
    "all_min": np.array([-1, -1, -1, -1], dtype=np.float32),
    "default": np.array([0, 0, 0, 0], dtype=np.float32),
    "all_max": np.array([1, 1, 1, 1], dtype=np.float32),
    "random": None,
}

ALPHAS = [1, 3, 5, 8]
SEEDS = [7, 42, 123, 456, 2024]
RISK_LEVELS = ["current", "increased"]
BETA = 1.0
GAMMA = 7.0
RECOVERY_SCALE = 46.0
INVENTORY_SCALE = 17_200_000.0
MAX_STEPS = 260
STEP_SIZE = 168.0


def run_episode(policy_action, seed, risk_level, alpha):
    env = MFSCGymEnv(
        step_size_hours=STEP_SIZE,
        max_steps=MAX_STEPS,
        year_basis="thesis",
        risk_level=risk_level,
        reward_mode="rt_v0",
        rt_alpha=alpha,
        rt_beta=BETA,
        rt_gamma=GAMMA,
        rt_recovery_scale=RECOVERY_SCALE,
        rt_inventory_scale=INVENTORY_SCALE,
    )
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed + 99999)
    total_reward = 0.0
    done, truncated = False, False
    while not (done or truncated):
        action = rng.uniform(-1, 1, size=4).astype(np.float32) if policy_action is None else policy_action
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
    return total_reward


def main():
    output_dir = Path("outputs/doe_alpha")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("MINI-DOE: Reward Spread Sensitivity to α")
    print(f"  α ∈ {ALPHAS} | β={BETA} γ={GAMMA}")
    print(f"  {len(SEEDS)} seeds × {len(RISK_LEVELS)} risk levels × {len(POLICIES)} policies")
    print(f"  Total episodes: {len(ALPHAS) * len(SEEDS) * len(RISK_LEVELS) * len(POLICIES)}")
    print("=" * 80)

    t0 = time.time()
    results = []

    for risk_level in RISK_LEVELS:
        print(f"\n--- Risk: {risk_level} ---")
        print(f"  {'α':>4s} | {'Spread':>10s} | {'Spread%':>8s} | {'Best':>10s} | {'Worst':>10s} | {'Best Policy':>12s}")
        print(f"  {'-'*60}")

        for alpha in ALPHAS:
            policy_rewards = {}
            for pname, paction in POLICIES.items():
                episode_rewards = []
                for seed in SEEDS:
                    r = run_episode(paction, seed, risk_level, alpha)
                    episode_rewards.append(r)
                policy_rewards[pname] = {
                    "mean": float(np.mean(episode_rewards)),
                    "std": float(np.std(episode_rewards, ddof=1)),
                    "values": episode_rewards,
                }

            best = max(policy_rewards.items(), key=lambda x: x[1]["mean"])
            worst = min(policy_rewards.items(), key=lambda x: x[1]["mean"])
            spread = best[1]["mean"] - worst[1]["mean"]
            mid = np.mean([v["mean"] for v in policy_rewards.values()])
            pct = abs(spread / mid) * 100 if mid != 0 else 0

            print(
                f"  {alpha:>4d} | {spread:>10.1f} | {pct:>7.1f}% | "
                f"{best[1]['mean']:>10.1f} | {worst[1]['mean']:>10.1f} | {best[0]:>12s}"
            )

            results.append({
                "risk_level": risk_level,
                "alpha": alpha,
                "beta": BETA,
                "gamma": GAMMA,
                "spread": spread,
                "spread_pct": pct,
                "best_policy": best[0],
                "worst_policy": worst[0],
                "best_mean": best[1]["mean"],
                "worst_mean": worst[1]["mean"],
                "policy_means": {k: v["mean"] for k, v in policy_rewards.items()},
            })

    elapsed = time.time() - t0
    print(f"\n{'='*80}")
    print(f"Completed in {elapsed:.0f}s")

    # Summary table
    print(f"\n{'='*80}")
    print("SUMMARY: Spread % by α and Risk Level")
    print(f"{'='*80}")
    print(f"  {'α':>4s} | {'Current':>10s} | {'Increased':>10s}")
    print(f"  {'-'*30}")
    for alpha in ALPHAS:
        current = next(r for r in results if r["alpha"] == alpha and r["risk_level"] == "current")
        increased = next(r for r in results if r["alpha"] == alpha and r["risk_level"] == "increased")
        print(f"  {alpha:>4d} | {current['spread_pct']:>9.1f}% | {increased['spread_pct']:>9.1f}%")

    # Check plateau
    inc_spreads = [r["spread_pct"] for r in results if r["risk_level"] == "increased"]
    max_spread = max(inc_spreads)
    print(f"\n  Max spread (increased): {max_spread:.1f}%")
    if max_spread < 5.0:
        print("  VERDICT: Spread plateaus below 5% — inventory actions alone insufficient.")
        print("  → Capacity-level actions (shift control) needed for meaningful RL signal.")
    elif max_spread < 10.0:
        print("  VERDICT: Marginal signal exists. PPO may learn slowly.")
    else:
        print("  VERDICT: Sufficient signal for PPO learning.")

    # Save
    with (output_dir / "doe_results.json").open("w") as f:
        json.dump({"results": results, "elapsed_s": elapsed}, f, indent=2)
    print(f"\n  Results saved to {output_dir / 'doe_results.json'}")


if __name__ == "__main__":
    main()
