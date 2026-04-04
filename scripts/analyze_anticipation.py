#!/usr/bin/env python3
"""Analyze whether PPO uses anticipatory signals or is purely reactive.

Runs evaluation episodes with a trained PPO model and records per-step:
  - Operating regime (nominal/strained/pre_disruption/disrupted/recovery)
  - Actions taken (shifts, downstream multipliers)
  - Risk forecasts (48h, 168h)

Then computes conditional escalation rates:
  P(S>1 | regime) for each regime
  P(downstream > threshold | regime) for each regime
  P(S>1 | forecast_48h > 0.5) vs P(S>1 | forecast_48h < 0.2)

If P(escalate | pre_disruption) >> P(escalate | nominal), PPO anticipates.
If P(escalate | disrupted) >> P(escalate | pre_disruption), PPO only reacts.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.external_env_interface import make_track_b_env

# v7 observation field indices
REGIME_FIELDS = {
    "nominal": 30,
    "strained": 31,
    "pre_disruption": 32,
    "disrupted": 33,
    "recovery": 34,
}
FORECAST_48H_IDX = 35
FORECAST_168H_IDX = 36
FILL_RATE_IDX = 6
BACKORDER_RATE_IDX = 7
ASSEMBLY_DOWN_IDX = 8
ANY_LOC_DOWN_IDX = 9


def get_regime(obs: np.ndarray) -> str:
    best = "nominal"
    best_val = -1.0
    for name, idx in REGIME_FIELDS.items():
        if obs[idx] > best_val:
            best_val = obs[idx]
            best = name
    return best


def run_analysis(
    model_path: str,
    vec_path: str,
    n_episodes: int = 20,
    seed_base: int = 70000,
) -> list[dict]:
    """Run episodes and collect per-step regime + action data."""
    rows = []

    for ep in range(n_episodes):
        eval_seed = seed_base + ep
        env = make_track_b_env(
            reward_mode="ReT_seq_v1",
            risk_level="adaptive_benchmark_v2",
            step_size_hours=168.0,
            max_steps=260,
        )
        monitor_env = Monitor(env)
        vec_env = DummyVecEnv([lambda: monitor_env])
        vec_norm = VecNormalize.load(vec_path, vec_env)
        vec_norm.training = False
        model = PPO.load(model_path)

        obs_raw, info = env.reset(seed=eval_seed)
        obs_norm = vec_norm.normalize_obs(
            np.asarray(obs_raw, dtype=np.float32)[None, :]
        )
        terminated = False
        truncated = False
        step = 0

        while not (terminated or truncated):
            action, _ = model.predict(obs_norm, deterministic=True)
            regime = get_regime(obs_raw)
            forecast_48h = float(obs_raw[FORECAST_48H_IDX])
            forecast_168h = float(obs_raw[FORECAST_168H_IDX])
            fill_rate = float(obs_raw[FILL_RATE_IDX])

            obs_raw, reward, terminated, truncated, info = env.step(
                np.asarray(action[0], dtype=np.float32)
            )
            obs_norm = vec_norm.normalize_obs(
                np.asarray(obs_raw, dtype=np.float32)[None, :]
            )

            shifts_active = int(info.get("shifts_active", 1))
            clipped = info.get("clipped_action", action[0])
            if hasattr(clipped, "__len__") and len(clipped) >= 7:
                ds_op10 = 1.25 + 0.75 * float(clipped[5])
                ds_op12 = 1.25 + 0.75 * float(clipped[6])
            else:
                ds_op10 = 1.25
                ds_op12 = 1.25

            rows.append({
                "episode": ep,
                "step": step,
                "regime": regime,
                "forecast_48h": forecast_48h,
                "forecast_168h": forecast_168h,
                "fill_rate": fill_rate,
                "shifts_active": shifts_active,
                "escalated": 1 if shifts_active > 1 else 0,
                "ds_op10": ds_op10,
                "ds_op12": ds_op12,
                "ds_high": 1 if (ds_op10 >= 1.9 or ds_op12 >= 1.9) else 0,
            })
            step += 1

        vec_norm.close()

    return rows


def print_analysis(rows: list[dict]) -> None:
    print(f"\nTotal steps: {len(rows)}")
    print(f"Episodes: {max(r['episode'] for r in rows) + 1}")

    # 1. Escalation rate by regime
    by_regime = defaultdict(list)
    for r in rows:
        by_regime[r["regime"]].append(r)

    print("\n" + "=" * 70)
    print("ESCALATION RATE BY REGIME")
    print("=" * 70)
    print(f"{'Regime':<20} {'N steps':>8} {'P(S>1)':>8} {'P(S=3)':>8} "
          f"{'Avg Op10':>9} {'Avg Op12':>9} {'P(ds>=1.9)':>11}")
    print("-" * 70)

    regime_order = ["nominal", "strained", "pre_disruption", "disrupted", "recovery"]
    for regime in regime_order:
        steps = by_regime.get(regime, [])
        if not steps:
            print(f"{regime:<20} {'0':>8} {'n/a':>8}")
            continue
        n = len(steps)
        p_escalated = sum(s["escalated"] for s in steps) / n
        p_s3 = sum(1 for s in steps if s["shifts_active"] == 3) / n
        avg_op10 = sum(s["ds_op10"] for s in steps) / n
        avg_op12 = sum(s["ds_op12"] for s in steps) / n
        p_ds_high = sum(s["ds_high"] for s in steps) / n
        print(f"{regime:<20} {n:>8} {p_escalated:>8.3f} {p_s3:>8.3f} "
              f"{avg_op10:>9.3f} {avg_op12:>9.3f} {p_ds_high:>11.3f}")

    # 2. Escalation rate by forecast level
    print("\n" + "=" * 70)
    print("ESCALATION RATE BY FORECAST LEVEL (48h)")
    print("=" * 70)
    bins = [
        ("low (< 0.2)", lambda r: r["forecast_48h"] < 0.2),
        ("mid (0.2-0.5)", lambda r: 0.2 <= r["forecast_48h"] < 0.5),
        ("high (>= 0.5)", lambda r: r["forecast_48h"] >= 0.5),
    ]
    print(f"{'Forecast bin':<20} {'N steps':>8} {'P(S>1)':>8} {'Avg Op10':>9} {'Avg Op12':>9}")
    print("-" * 55)
    for label, pred in bins:
        steps = [r for r in rows if pred(r)]
        if not steps:
            continue
        n = len(steps)
        p_esc = sum(s["escalated"] for s in steps) / n
        avg10 = sum(s["ds_op10"] for s in steps) / n
        avg12 = sum(s["ds_op12"] for s in steps) / n
        print(f"{label:<20} {n:>8} {p_esc:>8.3f} {avg10:>9.3f} {avg12:>9.3f}")

    # 3. Key comparison
    nom = by_regime.get("nominal", [])
    pre = by_regime.get("pre_disruption", [])
    dis = by_regime.get("disrupted", [])
    if nom and pre:
        p_nom = sum(s["escalated"] for s in nom) / len(nom)
        p_pre = sum(s["escalated"] for s in pre) / len(pre)
        p_dis = sum(s["escalated"] for s in dis) / len(dis) if dis else 0
        print("\n" + "=" * 70)
        print("KEY COMPARISON: ANTICIPATORY vs REACTIVE")
        print("=" * 70)
        print(f"P(S>1 | nominal):         {p_nom:.3f}")
        print(f"P(S>1 | pre_disruption):  {p_pre:.3f}")
        print(f"P(S>1 | disrupted):       {p_dis:.3f}")
        if p_pre > p_nom * 1.5:
            print("\n>>> PPO ESCALATES BEFORE DISRUPTION — evidence of ANTICIPATION")
        elif p_dis > p_nom * 1.5 and p_pre <= p_nom * 1.5:
            print("\n>>> PPO ESCALATES ONLY DURING DISRUPTION — purely REACTIVE")
        else:
            print("\n>>> No clear escalation pattern — INCONCLUSIVE")


def main():
    parser = argparse.ArgumentParser(description="Analyze PPO anticipation vs reaction")
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="Directory containing ppo_model.zip and vec_normalize.pkl")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--output-csv", type=Path, default=None)
    args = parser.parse_args()

    model_path = str(args.model_dir / "ppo_model.zip")
    vec_path = str(args.model_dir / "vec_normalize.pkl")

    print(f"Loading model from {args.model_dir}")
    rows = run_analysis(model_path, vec_path, n_episodes=args.episodes)
    print_analysis(rows)

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nTrajectory data saved to {args.output_csv}")


if __name__ == "__main__":
    main()
