#!/usr/bin/env python3
"""
DOE Decomposition: Break down reward spread by component.

For each (risk_level, alpha) config, shows how much of the spread
comes from recovery_time, holding_cost, and service_loss individually.
Also reports per-policy means with std across 10 seeds.
"""
from __future__ import annotations

import sys
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

ALPHAS = [1, 5, 8]
SEEDS = list(range(10))  # 10 seeds for CIs
RISK_LEVELS = ["current", "increased"]
BETA = 1.0
GAMMA = 7.0
RECOVERY_SCALE = 46.0
INVENTORY_SCALE = 17_200_000.0
MAX_STEPS = 260
STEP_SIZE = 168.0


def run_episode(policy_action, seed, risk_level, alpha):
    env = MFSCGymEnv(
        step_size_hours=STEP_SIZE, max_steps=MAX_STEPS, year_basis="thesis",
        risk_level=risk_level, reward_mode="rt_v0",
        rt_alpha=alpha, rt_beta=BETA, rt_gamma=GAMMA,
        rt_recovery_scale=RECOVERY_SCALE, rt_inventory_scale=INVENTORY_SCALE,
    )
    obs, _ = env.reset(seed=1000 + seed)
    rng = np.random.default_rng(seed + 99999)

    total_reward = 0.0
    total_recovery = 0.0
    total_holding = 0.0
    total_service = 0.0
    steps = 0
    done, truncated = False, False

    while not (done or truncated):
        action = rng.uniform(-1, 1, size=4).astype(np.float32) if policy_action is None else policy_action
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward

        # Decompose: recompute components manually
        recovery = info.get("step_disruption_hours", 0.0) / RECOVERY_SCALE
        holding = info.get("total_inventory", 0.0) / INVENTORY_SCALE
        demanded = info.get("new_demanded", 0.0)
        bo_qty = info.get("new_backorder_qty", 0.0)
        svc_loss = bo_qty / demanded if demanded > 0 else 0.0

        total_recovery += alpha * recovery
        total_holding += BETA * holding
        total_service += GAMMA * svc_loss
        steps += 1

    return {
        "reward": total_reward,
        "recovery_contrib": total_recovery,
        "holding_contrib": total_holding,
        "service_contrib": total_service,
    }


def ci95(values):
    arr = np.array(values)
    if len(arr) < 2:
        return arr.mean(), 0.0
    half = 1.96 * arr.std(ddof=1) / np.sqrt(len(arr))
    return float(arr.mean()), float(half)


def main():
    print("=" * 95)
    print("DOE DECOMPOSITION: Reward Spread by Component (10 seeds)")
    print("=" * 95)

    for risk_level in RISK_LEVELS:
        for alpha in ALPHAS:
            print(f"\n{'='*95}")
            print(f"  Risk: {risk_level} | α={alpha} β={BETA} γ={GAMMA}")
            print(f"{'='*95}")
            print(f"  {'Policy':>10s} | {'Reward':>16s} | {'Recovery':>14s} | {'Holding':>14s} | {'Service':>14s}")
            print(f"  {'-'*80}")

            all_rewards = {}
            all_components = {}

            for pname, paction in POLICIES.items():
                rewards, recoveries, holdings, services = [], [], [], []
                for seed in SEEDS:
                    r = run_episode(paction, seed, risk_level, alpha)
                    rewards.append(r["reward"])
                    recoveries.append(r["recovery_contrib"])
                    holdings.append(r["holding_contrib"])
                    services.append(r["service_contrib"])

                all_rewards[pname] = rewards
                all_components[pname] = {
                    "recovery": recoveries,
                    "holding": holdings,
                    "service": services,
                }

                r_mean, r_ci = ci95(rewards)
                rec_mean, _ = ci95(recoveries)
                hld_mean, _ = ci95(holdings)
                svc_mean, _ = ci95(services)

                print(
                    f"  {pname:>10s} | {r_mean:>8.1f} ±{r_ci:>5.1f} | "
                    f"{rec_mean:>8.1f} ({rec_mean/(rec_mean+hld_mean+svc_mean)*100:>4.1f}%) | "
                    f"{hld_mean:>8.1f} ({hld_mean/(rec_mean+hld_mean+svc_mean)*100:>4.1f}%) | "
                    f"{svc_mean:>8.1f} ({svc_mean/(rec_mean+hld_mean+svc_mean)*100:>4.1f}%)"
                )

            # Compute per-component spread
            print(f"\n  Component spreads:")
            for comp in ["recovery", "holding", "service"]:
                means = {p: np.mean(all_components[p][comp]) for p in POLICIES}
                comp_spread = max(means.values()) - min(means.values())
                best_p = max(means, key=means.get)
                worst_p = min(means, key=means.get)
                total_mean = np.mean(list(means.values()))
                pct = abs(comp_spread / total_mean) * 100 if total_mean != 0 else 0
                print(f"    {comp:>10s}: spread={comp_spread:>8.1f} ({pct:>5.1f}% of mean)  "
                      f"best={worst_p} worst={best_p}")  # Note: lower penalty = better

            # Total reward spread
            reward_means = {p: np.mean(all_rewards[p]) for p in POLICIES}
            total_spread = max(reward_means.values()) - min(reward_means.values())
            mid = np.mean(list(reward_means.values()))
            total_pct = abs(total_spread / mid) * 100 if mid != 0 else 0
            print(f"    {'TOTAL':>10s}: spread={total_spread:>8.1f} ({total_pct:>5.1f}% of mean)")


if __name__ == "__main__":
    main()
