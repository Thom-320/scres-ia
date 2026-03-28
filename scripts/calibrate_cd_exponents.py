#!/usr/bin/env python3
"""
Calibrate Cobb-Douglas exponents for ReT_cd following Garrido et al. (2024).

Methodology (Garrido 2024, Section 3.3):
1. Run N simulation episodes with random/static policies
2. Collect the four C-D input variables at each step
3. Find the max value of each variable across all episodes
4. Set exponents so each variable contributes 1/n at its maximum

For n=4 variables, each argument is equated to 1/4 = 0.25:
    a × ln(FR_max) = 0.25  →  a = 0.25 / ln(FR_max)
    b × ln(IB_max) = 0.25  →  b = 0.25 / ln(IB_max)
    etc.

Usage:
    python scripts/calibrate_cd_exponents.py --episodes 100 --risk-level increased
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.env_experimental_shifts import MFSCGymEnvShifts


def collect_variables(
    n_episodes: int,
    risk_level: str,
    stochastic_pt: bool,
    kappa: float,
    bo_norm: float,
    seed: int = 42,
) -> dict[str, list[float]]:
    """Run episodes and collect C-D variables at each step."""
    env = MFSCGymEnvShifts(
        reward_mode="ReT_seq_v1",  # use any mode, we just read info
        risk_level=risk_level,
        stochastic_pt=stochastic_pt,
        step_size_hours=168,
        max_steps=260,
        ret_seq_kappa=kappa,
    )

    all_vars: dict[str, list[float]] = {
        "fill_rate": [],
        "inverse_backlog": [],
        "spare_capacity": [],
        "inverse_cost": [],
    }

    policies = ["static_s1", "static_s2", "static_s3", "random"]

    for ep in range(n_episodes):
        policy = policies[ep % len(policies)]
        obs, info = env.reset(seed=seed + ep)

        for step in range(260):
            if policy == "static_s1":
                action = np.array([0, 0, 0, 0, -1.0], dtype=np.float32)
            elif policy == "static_s2":
                action = np.array([0, 0, 0, 0, 0.0], dtype=np.float32)
            elif policy == "static_s3":
                action = np.array([0, 0, 0, 0, 1.0], dtype=np.float32)
            else:
                action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)

            # Extract raw variables
            demanded = float(info.get("demanded_qty_step", 1.0))
            backorder = float(info.get("new_backorder_qty_step", 0.0))
            pending_bo = float(info.get("pending_backorder_qty", 0.0))
            shifts = int(info.get("shifts_active", 1))

            fr = max(1e-6, 1.0 - backorder / max(demanded, 1.0))
            ib = 1.0 / (1.0 + pending_bo / bo_norm)
            sc_cap = shifts / 3.0
            ic = 1.0 / (1.0 + kappa * (shifts - 1) / 2.0)

            all_vars["fill_rate"].append(fr)
            all_vars["inverse_backlog"].append(ib)
            all_vars["spare_capacity"].append(sc_cap)
            all_vars["inverse_cost"].append(ic)

            if terminated or truncated:
                break

    env.close()
    return all_vars


def calibrate_exponents(
    all_vars: dict[str, list[float]], n_factors: int = 4
) -> dict[str, float]:
    """
    Calibrate C-D exponents following Garrido 2024.

    For each variable, find the max and set:
        exponent × ln(max_value) = 1/n

    Returns dict with calibrated exponents.
    """
    target = 1.0 / n_factors  # Each factor contributes 1/n at max

    exponents = {}
    stats = {}
    for name, values in all_vars.items():
        arr = np.array(values)
        vmax = float(np.percentile(arr, 99))  # Use P99 to avoid outliers
        vmean = float(np.mean(arr))
        vmin = float(np.min(arr))

        if vmax > 0 and abs(math.log(vmax)) > 1e-10:
            exp_val = target / abs(math.log(vmax))
        else:
            exp_val = 1.0  # fallback

        exponents[name] = round(exp_val, 4)
        stats[name] = {"min": round(vmin, 4), "mean": round(vmean, 4), "max": round(vmax, 4)}

    # Normalize so exponents sum to 1 (proper C-D)
    total = sum(exponents.values())
    normalized = {k: round(v / total, 4) for k, v in exponents.items()}

    return {"raw_exponents": exponents, "normalized_exponents": normalized, "stats": stats}


def main():
    parser = argparse.ArgumentParser(description="Calibrate C-D exponents")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--risk-level", default="increased")
    parser.add_argument("--stochastic-pt", action="store_true", default=True)
    parser.add_argument("--kappa", type=float, default=0.20)
    parser.add_argument("--bo-norm", type=float, default=5000.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    print(f"Collecting C-D variables from {args.episodes} episodes...")
    print(f"  Risk level: {args.risk_level}")
    print(f"  Stochastic PT: {args.stochastic_pt}")
    print(f"  Kappa: {args.kappa}")
    print(f"  BO normalization: {args.bo_norm}")

    all_vars = collect_variables(
        n_episodes=args.episodes,
        risk_level=args.risk_level,
        stochastic_pt=args.stochastic_pt,
        kappa=args.kappa,
        bo_norm=args.bo_norm,
        seed=args.seed,
    )

    print(f"\nCollected {len(all_vars['fill_rate'])} step observations.")

    result = calibrate_exponents(all_vars)

    print("\n=== CALIBRATION RESULTS ===")
    print("\nVariable statistics:")
    for name, stat in result["stats"].items():
        print(f"  {name:20s}: min={stat['min']:.4f}, mean={stat['mean']:.4f}, max={stat['max']:.4f}")

    print("\nRaw C-D exponents (before normalization):")
    for name, exp in result["raw_exponents"].items():
        print(f"  {name:20s}: {exp:.4f}")

    print("\nNormalized C-D exponents (sum=1, proper weighted geometric mean):")
    for name, exp in result["normalized_exponents"].items():
        print(f"  {name:20s}: {exp:.4f}")
    print(f"  {'SUM':20s}: {sum(result['normalized_exponents'].values()):.4f}")

    print("\nComparison with ReT_seq_v1 weights:")
    print(f"  ReT_seq_v1: SC=0.60, BC=0.25, AE=0.15")
    ne = result["normalized_exponents"]
    print(f"  ReT_cd:     FR={ne['fill_rate']}, IB={ne['inverse_backlog']}, "
          f"SC={ne['spare_capacity']}, IC={ne['inverse_cost']}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved to {output_path}")

    return result


if __name__ == "__main__":
    main()
