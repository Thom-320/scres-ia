#!/usr/bin/env python3
"""
Diagnostic: Compare action spread under proxy vs rt_v0 reward.

Uses correct units throughout:
  - service_loss = new_backorder_qty (rations) / new_demanded (rations)
  - step_disruption_hours from continuous tracker (not event-based)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from supply_chain.env import MFSCGymEnv

POLICIES = {
    "all_min (-1)": np.array([-1, -1, -1, -1], dtype=np.float32),
    "default (0)": np.array([0, 0, 0, 0], dtype=np.float32),
    "all_max (+1)": np.array([1, 1, 1, 1], dtype=np.float32),
    "random": None,
}

N_SEEDS = 10
MAX_STEPS = 260


def run_episode(policy_action, seed, reward_mode, risk_level, **kwargs):
    env = MFSCGymEnv(
        step_size_hours=168,
        max_steps=MAX_STEPS,
        year_basis="thesis",
        risk_level=risk_level,
        reward_mode=reward_mode,
        **kwargs,
    )
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    total_delivered = 0.0
    total_backorder_qty = 0.0
    total_demanded = 0.0
    total_disruption = 0.0
    total_inventory_sum = 0.0
    rng = np.random.default_rng(seed + 99999)

    done, truncated = False, False
    steps = 0
    while not (done or truncated):
        if policy_action is None:
            action = rng.uniform(-1, 1, size=4).astype(np.float32)
        else:
            action = policy_action
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        total_delivered += info.get("new_delivered", 0)
        total_backorder_qty += info.get("new_backorder_qty", 0)
        total_demanded += info.get("new_demanded", 0)
        total_disruption += info.get("step_disruption_hours", 0)
        total_inventory_sum += info.get("total_inventory", 0)
        steps += 1

    return {
        "reward": total_reward,
        "delivered": total_delivered,
        "backorder_qty": total_backorder_qty,
        "demanded": total_demanded,
        "disruption_hrs": total_disruption,
        "avg_inventory": total_inventory_sum / max(1, steps),
        "service_loss": total_backorder_qty / max(1, total_demanded),
    }


def run_test(reward_mode, risk_level, label, **kwargs):
    print(f"\n{'='*85}")
    print(f"  {label}")
    print(f"  reward_mode={reward_mode} | risk_level={risk_level} | {N_SEEDS} seeds")
    print(f"{'='*85}")
    print(f"{'Policy':>15s} | {'Reward':>14s} | {'Delivered':>12s} | {'BO Qty':>10s} | "
          f"{'Svc Loss':>8s} | {'Disrupt hrs':>11s} | {'Avg Inv':>12s}")
    print("-" * 85)

    results = {}
    for pname, paction in POLICIES.items():
        rewards, delivered, bo_qty, svc, disrupt, inv = [], [], [], [], [], []
        for s in range(N_SEEDS):
            r = run_episode(paction, 1000 + s, reward_mode, risk_level, **kwargs)
            rewards.append(r["reward"])
            delivered.append(r["delivered"])
            bo_qty.append(r["backorder_qty"])
            svc.append(r["service_loss"])
            disrupt.append(r["disruption_hrs"])
            inv.append(r["avg_inventory"])

        results[pname] = {"reward_mean": np.mean(rewards), "reward_std": np.std(rewards, ddof=1)}
        print(
            f"{pname:>15s} | "
            f"{np.mean(rewards):>11,.1f} ±{np.std(rewards, ddof=1):>5,.1f} | "
            f"{np.mean(delivered):>12,.0f} | "
            f"{np.mean(bo_qty):>10,.0f} | "
            f"{np.mean(svc):>7.2%} | "
            f"{np.mean(disrupt):>11,.0f} | "
            f"{np.mean(inv):>12,.0f}"
        )

    best = max(results.items(), key=lambda x: x[1]["reward_mean"])
    worst = min(results.items(), key=lambda x: x[1]["reward_mean"])
    spread = best[1]["reward_mean"] - worst[1]["reward_mean"]
    mid = np.mean([v["reward_mean"] for v in results.values()])
    pct = spread / abs(mid) * 100 if mid != 0 else 0

    print(f"\n  Spread: {spread:,.1f} ({pct:.1f}% of mean)")
    print(f"  Best: {best[0]} | Worst: {worst[0]}")
    return pct


def main():
    print("DIAGNOSTIC: Reward Spread Comparison (with correct units)")
    print("=" * 85)

    # 1. Proxy reward, current risk
    pct_proxy = run_test("proxy", "current", "PROXY REWARD — Current Risk")

    # 2. Rt_v0 reward, current risk (balanced: each component ~ equal weight)
    pct_rt = run_test(
        "rt_v0", "current", "Rt_v0 REWARD — Current Risk (balanced)",
        rt_alpha=1.0, rt_beta=1.0, rt_gamma=7.0,
        rt_recovery_scale=46.0, rt_inventory_scale=17_200_000.0,
    )

    # 3. Rt_v0 reward, increased risk
    pct_rt_inc = run_test(
        "rt_v0", "increased", "Rt_v0 REWARD — Increased Risk (balanced)",
        rt_alpha=1.0, rt_beta=1.0, rt_gamma=7.0,
        rt_recovery_scale=46.0, rt_inventory_scale=17_200_000.0,
    )

    print(f"\n{'='*85}")
    print("SUMMARY")
    print(f"{'='*85}")
    print(f"  Proxy (current risk):      {pct_proxy:>5.1f}% spread")
    print(f"  Rt_v0 (current risk):      {pct_rt:>5.1f}% spread")
    print(f"  Rt_v0 (increased risk):    {pct_rt_inc:>5.1f}% spread")
    threshold = 5.0
    print(f"\n  Threshold for viable RL: >{threshold}%")
    for name, pct in [("Proxy current", pct_proxy), ("Rt_v0 current", pct_rt), ("Rt_v0 increased", pct_rt_inc)]:
        status = "PASS" if pct > threshold else "FAIL"
        print(f"  {name:<25s}: {status}")


if __name__ == "__main__":
    main()
