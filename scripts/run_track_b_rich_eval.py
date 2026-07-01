#!/usr/bin/env python3
"""Rich eval: reload Track B model and evaluate with full Garrido metrics."""
import sys, time, json, csv, re
from pathlib import Path
from statistics import fmean
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from supply_chain.env_experimental_shifts import MFSCGymEnvShifts
from supply_chain.episode_metrics import compute_episode_metrics
from stable_baselines3 import PPO

TRACK_B = Path("outputs/experiments/track_b_gain_2026-06-29/kaggle_joint_confirm_50k_v5_output/track_b_joint_confirm_50k_3seed_h104")
OUT = Path("outputs/experiments/track_b_rich_eval_2026-06-30")
OUT.mkdir(parents=True, exist_ok=True)
config = json.loads((TRACK_B / "summary.json").read_text())["config"]

all_metrics = []
for seed_dir in sorted((TRACK_B / "models").glob("seed*")):
    seed = int(seed_dir.name.replace("seed", ""))
    model = PPO.load(str(seed_dir / "ppo_model.zip"))
    print(f"Seed {seed}...", end=" ", flush=True)
    
    env = MFSCGymEnvShifts(
        reward_mode=config["reward_mode"], observation_version=config["observation_version"],
        risk_level=config["risk_level"], stochastic_pt=config["stochastic_pt"],
        max_steps=config["max_steps"], step_size_hours=config["step_size_hours"],
        year_basis=config["year_basis"], action_contract=config["action_contract"])
    
    for ep in range(4):
        obs, _ = env.reset(seed=9000 + seed*10 + ep)
        done = truncated = False
        while not (done or truncated):
            a, _ = model.predict(obs, deterministic=True)
            obs, r, done, truncated, info = env.step(a)
        metrics = compute_episode_metrics(env.unwrapped.sim)
        metrics["seed"] = seed; metrics["episode"] = ep
        all_metrics.append(metrics)
    env.close()

keys = sorted(set().union(*(m.keys() for m in all_metrics)))
with (OUT / "rich_metrics.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(all_metrics)

summary = {}
for k in keys:
    vals = [float(m[k]) for m in all_metrics if k in m and np.isfinite(float(m[k]))]
    if vals:
        summary[f"{k}_mean"] = float(np.mean(vals))
        summary[f"{k}_std"] = float(np.std(vals))

key_metrics = ['ret_excel','ret_thesis','ret_continuous','fill_rate','fill_rate_on_time',
    'flow_fill_rate','lost_rate','lost_orders','backorder_qty_final',
    'service_loss_auc_per_order','ttr_mean','ttr_p95',
    'ctj_p50','ctj_p90','ctj_p99','rpj_p90','rpj_p99','dpj_p99',
    'delivered_rations','demanded_rations','n_orders','n_served','n_lost','n_late']

print(f"\n{'='*60}")
print("TRACK B — RICH METRICS (3 seeds × 4 eps)")
print(f"{'='*60}")
for km in key_metrics:
    m = summary.get(f"{km}_mean")
    if m is not None:
        print(f"  {km:<30} {m:.4f}")

(OUT / "rich_summary.json").write_text(json.dumps(summary, indent=2))
(OUT / "rich_metrics.csv").touch()
print(f"\nWROTE {OUT}")
