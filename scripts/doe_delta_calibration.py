#!/usr/bin/env python
"""
DOE: Calibrate δ (shift cost weight) for ReT_thesis reward.

For each δ candidate, trains a short PPO agent and measures:
  - Mean ReT over evaluation episodes
  - Shift selection distribution (% S=1 / S=2 / S=3)
  - Mean reward

Target: δ that yields 20-40% S=2 selection under increased risks.

Usage:
  python scripts/doe_delta_calibration.py
  python scripts/doe_delta_calibration.py --risk-level current
  python scripts/doe_delta_calibration.py --deltas 0.01 0.02 0.05 0.1
"""

import argparse
import os
import sys
import time
from collections import Counter
from pathlib import Path
import json

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from supply_chain.config import RET_SHIFT_COST_DELTA_DEFAULT
from supply_chain.env_experimental_shifts import MFSCGymEnvShifts


def make_env(risk_level: str, delta: float, seed: int):
    def _init():
        env = MFSCGymEnvShifts(
            step_size_hours=168,
            reward_mode="ReT_thesis",
            rt_delta=delta,
            risk_level=risk_level,
            max_steps=52 * 2,  # 2 years per episode
        )
        env.reset(seed=seed)
        return env

    return _init


def evaluate(model, risk_level: str, delta: float, n_episodes: int = 5):
    """Run evaluation episodes and collect metrics."""
    all_rets = []
    all_rewards = []
    shift_counts = Counter()
    total_steps = 0

    for ep in range(n_episodes):
        env = MFSCGymEnvShifts(
            step_size_hours=168,
            reward_mode="ReT_thesis",
            rt_delta=delta,
            risk_level=risk_level,
            max_steps=52 * 2,
        )
        obs, _ = env.reset(seed=1000 + ep)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(action)
            all_rets.append(info["ReT_raw"])
            all_rewards.append(reward)
            shift_counts[info["shifts_active"]] += 1
            total_steps += 1
            done = term or trunc

    ret_arr = np.array(all_rets)
    rew_arr = np.array(all_rewards)
    total = sum(shift_counts.values())
    shift_pcts = {s: shift_counts[s] / total * 100 for s in [1, 2, 3]}

    return {
        "ReT_mean": float(ret_arr.mean()),
        "ReT_std": float(ret_arr.std()),
        "ReT_min": float(ret_arr.min()),
        "reward_mean": float(rew_arr.mean()),
        "shift_pcts": shift_pcts,
        "total_steps": total_steps,
    }


def main():
    parser = argparse.ArgumentParser(description="DOE: calibrate δ for ReT reward")
    parser.add_argument(
        "--deltas",
        type=float,
        nargs="+",
        default=[0.01, 0.02, 0.05, 0.08, 0.1, 0.15],
    )
    parser.add_argument(
        "--risk-level", default="increased", choices=["current", "increased", "severe"]
    )
    parser.add_argument("--train-steps", type=int, default=50_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save DOE results as JSON.",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  DOE: δ Calibration for ReT_thesis reward")
    print(f"  Risk level: {args.risk_level}")
    print(f"  Training steps per δ: {args.train_steps:,}")
    print(f"  Eval episodes: {args.eval_episodes}")
    print(f"  δ candidates: {args.deltas}")
    print("=" * 70)

    results = []

    for delta in args.deltas:
        print(f"\n{'─'*70}")
        print(f"  δ = {delta}")
        print(f"{'─'*70}")

        t0 = time.time()

        vec_env = DummyVecEnv([make_env(args.risk_level, delta, args.seed)])
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=0,
            seed=args.seed,
        )
        model.learn(total_timesteps=args.train_steps)
        elapsed = time.time() - t0

        metrics = evaluate(model, args.risk_level, delta, args.eval_episodes)
        metrics["delta"] = delta
        metrics["train_time_s"] = elapsed
        results.append(metrics)

        sp = metrics["shift_pcts"]
        print(f"  Train time:  {elapsed:.1f}s")
        print(
            f"  ReT:         {metrics['ReT_mean']:.4f} ± {metrics['ReT_std']:.4f}  "
            f"(min {metrics['ReT_min']:.4f})"
        )
        print(f"  Reward:      {metrics['reward_mean']:.4f}")
        print(
            f"  Shifts:      S=1: {sp.get(1,0):.1f}%  "
            f"S=2: {sp.get(2,0):.1f}%  S=3: {sp.get(3,0):.1f}%"
        )

        vec_env.close()

    # Summary table
    print(f"\n{'='*70}")
    print("  SUMMARY — δ Calibration Results")
    print(f"{'='*70}")
    print(
        f"  {'δ':>6}  {'ReT':>8}  {'Reward':>8}  {'S=1%':>6}  {'S=2%':>6}  {'S=3%':>6}  {'Target?':>8}"
    )
    print(f"  {'─'*56}")
    for r in results:
        sp = r["shift_pcts"]
        s2_pct = sp.get(2, 0)
        in_target = "✅" if 20 <= s2_pct <= 40 else "❌"
        print(
            f"  {r['delta']:>6.3f}  {r['ReT_mean']:>8.4f}  {r['reward_mean']:>8.4f}  "
            f"{sp.get(1,0):>5.1f}%  {s2_pct:>5.1f}%  {sp.get(3,0):>5.1f}%  {in_target:>8}"
        )

    # Find best δ
    best = None
    best_dist = float("inf")
    for r in results:
        s2 = r["shift_pcts"].get(2, 0)
        dist = abs(s2 - 30)  # Target 30% S=2
        if dist < best_dist:
            best_dist = dist
            best = r

    if best:
        print(
            f"\n  Recommended δ = {best['delta']:.3f}  "
            f"(S=2 at {best['shift_pcts'].get(2,0):.1f}%)"
        )
        print(f"  Current repo default δ = {RET_SHIFT_COST_DELTA_DEFAULT:.3f}")

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "risk_level": args.risk_level,
            "train_steps": args.train_steps,
            "eval_episodes": args.eval_episodes,
            "seed": args.seed,
            "deltas": args.deltas,
            "recommended_delta": best["delta"] if best else None,
            "repo_default_delta": RET_SHIFT_COST_DELTA_DEFAULT,
            "results": results,
        }
        with args.output_json.open("w", encoding="utf-8") as file_obj:
            json.dump(payload, file_obj, indent=2)
        print(f"  Saved JSON results to {args.output_json}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
