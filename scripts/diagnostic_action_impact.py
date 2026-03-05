#!/usr/bin/env python3
"""
Diagnostic: Do actions actually impact the simulation?

Runs full episodes with fixed extreme actions under BOTH risk levels
(current vs increased) and compares outcomes.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from supply_chain.config import OPERATIONS, SIMULATION_HORIZON, WARMUP
from supply_chain.supply_chain import MFSCSimulation

POLICIES = {
    "all_min (-1,-1,-1,-1)": np.array([-1, -1, -1, -1], dtype=np.float32),
    "low (-0.5)": np.array([-0.5, -0.5, -0.5, -0.5], dtype=np.float32),
    "default (0,0,0,0)": np.array([0, 0, 0, 0], dtype=np.float32),
    "high (+0.5)": np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32),
    "all_max (+1,+1,+1,+1)": np.array([1, 1, 1, 1], dtype=np.float32),
    "random": None,
}

N_SEEDS = 10
STEP_SIZE = 168.0
MAX_STEPS = 260
WARMUP_HOURS = float(WARMUP["estimated_deterministic_hrs"])


def action_to_dict(action_arr: np.ndarray) -> dict:
    """Convert [-1,1]^4 action to simulation parameter dict."""
    multipliers = 1.25 + 0.75 * action_arr
    base_op9_min = OPERATIONS[9]["q"][0]
    base_op9_max = OPERATIONS[9]["q"][1]
    return {
        "op3_q": OPERATIONS[3]["q"] * float(multipliers[0]),
        "op9_q_min": base_op9_min * float(multipliers[1]),
        "op9_q_max": base_op9_max * float(multipliers[1]),
        "op3_rop": OPERATIONS[3]["rop"] * float(multipliers[2]),
        "op9_rop": OPERATIONS[9]["rop"] * float(multipliers[3]),
    }


def run_episode(policy_action, seed: int, risk_level: str):
    """Run one full episode using MFSCSimulation directly."""
    sim = MFSCSimulation(
        shifts=1,
        risks_enabled=True,
        risk_level=risk_level,
        seed=seed,
        horizon=SIMULATION_HORIZON,
        year_basis="thesis",
    )
    sim._start_processes()
    sim.env.run(until=WARMUP_HOURS)

    total_reward = 0.0
    total_delivered = 0.0
    total_backorders = 0.0
    rng = np.random.default_rng(seed + 99999)

    for _ in range(MAX_STEPS):
        if policy_action is None:
            act = rng.uniform(-1, 1, size=4).astype(np.float32)
            action_dict = action_to_dict(act)
        else:
            action_dict = action_to_dict(policy_action)

        obs, reward, done, info = sim.step(action=action_dict, step_hours=STEP_SIZE)
        total_reward += reward
        total_delivered += info.get("new_delivered", 0)
        total_backorders += info.get("new_backorders", 0)
        if done:
            break

    fill_rate = sim._fill_rate()
    return {
        "reward": total_reward,
        "delivered": total_delivered,
        "backorders": total_backorders,
        "fill_rate": fill_rate,
    }


def run_diagnostic(risk_level: str):
    print(f"\n{'='*90}")
    print(f"  RISK LEVEL: {risk_level.upper()}")
    print(f"  {N_SEEDS} seeds × {len(POLICIES)} policies | {MAX_STEPS} steps | {STEP_SIZE}h/step")
    print(f"{'='*90}")

    if risk_level == "current":
        print("  (Thesis Table 6.12 base rates: R21~U(1,8064), R3~U(1,161280))")
    else:
        print("  (Thesis Table 6.12 increased: R21~U(1,4032), R3~U(1,80640) — 2x freq)")

    print(f"\n{'Policy':>30s} | {'Reward':>15s} | {'Delivered':>12s} | {'Backorders':>10s} | {'Fill Rate':>9s}")
    print("-" * 90)

    results = {}
    for policy_name, policy_action in POLICIES.items():
        rewards, delivered, backorders, fill_rates = [], [], [], []
        for s in range(N_SEEDS):
            r = run_episode(policy_action, seed=1000 + s, risk_level=risk_level)
            rewards.append(r["reward"])
            delivered.append(r["delivered"])
            backorders.append(r["backorders"])
            fill_rates.append(r["fill_rate"])

        results[policy_name] = {
            "reward_mean": np.mean(rewards),
            "reward_std": np.std(rewards, ddof=1),
            "delivered_mean": np.mean(delivered),
            "backorders_mean": np.mean(backorders),
            "fill_rate_mean": np.mean(fill_rates),
        }

        print(
            f"{policy_name:>30s} | "
            f"{np.mean(rewards):>12,.0f} ±{np.std(rewards, ddof=1):>6,.0f} | "
            f"{np.mean(delivered):>12,.0f} | "
            f"{np.mean(backorders):>10,.0f} | "
            f"{np.mean(fill_rates):>8.1%}"
        )

    # Analysis
    best = max(results.items(), key=lambda x: x[1]["reward_mean"])
    worst = min(results.items(), key=lambda x: x[1]["reward_mean"])
    spread = best[1]["reward_mean"] - worst[1]["reward_mean"]
    mid = np.mean([v["reward_mean"] for v in results.values()])

    bo_best = min(results.items(), key=lambda x: x[1]["backorders_mean"])
    bo_worst = max(results.items(), key=lambda x: x[1]["backorders_mean"])

    print(f"\n  Best reward:  {best[0]} → {best[1]['reward_mean']:>12,.0f}")
    print(f"  Worst reward: {worst[0]} → {worst[1]['reward_mean']:>12,.0f}")
    print(f"  Reward spread: {spread:>10,.0f} ({spread/mid*100:.2f}% of mean)")
    print(f"  Backorder range: {bo_best[1]['backorders_mean']:.0f} ({bo_best[0]}) "
          f"→ {bo_worst[1]['backorders_mean']:.0f} ({bo_worst[0]})")
    print(f"  Fill rate range: {worst[1]['fill_rate_mean']:.1%} → {best[1]['fill_rate_mean']:.1%}")

    return results, spread, mid


def main():
    print("=" * 90)
    print("DIAGNOSTIC: Action Impact — Current vs Increased Risk Levels")
    print("=" * 90)

    print("\nAction → Multiplier mapping (multiplier = 1.25 + 0.75 * action):")
    for name, act in POLICIES.items():
        if act is not None:
            mult = 1.25 + 0.75 * act
            print(f"  {name:>30s} → mult = [{mult[0]:.3f}, {mult[1]:.3f}, {mult[2]:.3f}, {mult[3]:.3f}]")
        else:
            print(f"  {name:>30s} → random U(-1,1) each step")

    res_current, spread_c, mid_c = run_diagnostic("current")
    res_increased, spread_i, mid_i = run_diagnostic("increased")

    # Comparison
    print(f"\n{'='*90}")
    print("COMPARISON: Current vs Increased Risk")
    print(f"{'='*90}")
    print(f"  {'Metric':<25s} | {'Current':>15s} | {'Increased':>15s} | {'Change':>10s}")
    print(f"  {'-'*25}-+-{'-'*15}-+-{'-'*15}-+-{'-'*10}")
    print(f"  {'Reward spread':.<25s} | {spread_c:>12,.0f}    | {spread_i:>12,.0f}    | {(spread_i/max(1,spread_c)-1)*100:>+.0f}%")
    print(f"  {'Spread % of mean':.<25s} | {spread_c/mid_c*100:>11.2f}%    | {spread_i/mid_i*100:>11.2f}%    |")

    # Backorder comparison
    bo_c = [v["backorders_mean"] for v in res_current.values()]
    bo_i = [v["backorders_mean"] for v in res_increased.values()]
    print(f"  {'Backorder range':.<25s} | {min(bo_c):.0f}-{max(bo_c):.0f}         | {min(bo_i):.0f}-{max(bo_i):.0f}         |")

    # Fill rate comparison
    fr_c = [v["fill_rate_mean"] for v in res_current.values()]
    fr_i = [v["fill_rate_mean"] for v in res_increased.values()]
    print(f"  {'Fill rate range':.<25s} | {min(fr_c):.1%}-{max(fr_c):.1%}     | {min(fr_i):.1%}-{max(fr_i):.1%}     |")

    print(f"\n{'='*90}")
    print("VERDICT:")
    if spread_i / mid_i > 0.05:
        print(f"  INCREASED risk: spread = {spread_i/mid_i*100:.1f}% — SIGNIFICANT signal for RL!")
        print(f"  vs CURRENT risk: spread = {spread_c/mid_c*100:.2f}% — too flat for learning.")
        print("  → Train PPO under increased risk levels for meaningful results.")
    elif spread_i / mid_i > 0.02:
        print(f"  INCREASED risk: spread = {spread_i/mid_i*100:.1f}% — marginal improvement.")
        print("  → May need even higher risk OR richer reward function.")
    else:
        print(f"  Even INCREASED risk shows only {spread_i/mid_i*100:.2f}% spread.")
        print("  → The bottleneck is structural (assembly line capacity), not risk level.")
        print("  → Need broader action space (shifts, capacity) or formal R_t metric.")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
