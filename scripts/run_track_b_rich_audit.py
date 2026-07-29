#!/usr/bin/env python3
"""Rich metrics audit: PPO vs dense best static for Track B."""
import sys, time, json, csv, re
from pathlib import Path
from statistics import fmean
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from supply_chain.env_experimental_shifts import MFSCGymEnvShifts
from supply_chain.episode_metrics import compute_episode_metrics
from stable_baselines3 import PPO

MODEL_DIR = Path("outputs/experiments/track_b_gain_2026-06-29/kaggle_joint_confirm_50k_v5_output/track_b_joint_confirm_50k_3seed_h104/models")
CONFIG = dict(reward_mode="control_v1", observation_version="v7", risk_level="adaptive_benchmark_v2",
              stochastic_pt=True, max_steps=104, step_size_hours=168.0, year_basis="thesis",
              action_contract="track_b_v1")
OUT = Path("outputs/experiments/track_b_rich_audit_2026-07-01")
OUT.mkdir(parents=True, exist_ok=True)

# Best dense static action
BEST_STATIC = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1, 0.5, 0.5], dtype=np.float32)

all_metrics = []
for seed_dir in sorted(MODEL_DIR.glob("seed*")):
    seed = int(seed_dir.name.replace("seed", ""))
    model = PPO.load(str(seed_dir / "ppo_model.zip"))
    print(f"Seed {seed}...", end=" ", flush=True)
    for ep in range(4):
        for policy_name, act_fn in [
            ("ppo", lambda o, m=model: m.predict(o, deterministic=True)[0]),
            ("best_static", lambda o: BEST_STATIC),
        ]:
            env = MFSCGymEnvShifts(**CONFIG)
            obs, _ = env.reset(seed=9000 + seed*10 + ep)
            done = False
            while not done:
                obs, r, done, trunc, info = env.step(act_fn(obs))
            metrics = compute_episode_metrics(env.unwrapped.sim)
            metrics["seed"] = seed; metrics["episode"] = ep; metrics["policy"] = policy_name
            all_metrics.append(metrics)
            env.close()
print()

keys = sorted(set().union(*(m.keys() for m in all_metrics)))
with (OUT / "rich_audit.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=['policy','seed','episode']+keys)
    w.writeheader()
    for m in all_metrics:
        w.writerow({'policy':m.get('policy',''),'seed':m.get('seed',''),'episode':m.get('episode',''),**{k:m.get(k,'') for k in keys}})

# Summarize by policy
for policy in ['ppo', 'best_static']:
    pmetrics = [m for m in all_metrics if m['policy']==policy]
    key_mets = ['ret_excel','ret_thesis','fill_rate','flow_fill_rate','lost_rate','lost_orders',
                'backorder_qty_final','service_loss_auc_per_order','ttr_mean','ttr_p95',
                'ctj_p50','ctj_p90','ctj_p99','rpj_p90','rpj_p99','dpj_p99',
                'delivered_rations','n_served','n_lost','n_late']
    summary = {}
    for k in key_mets:
        vals = [float(m[k]) for m in pmetrics if k in m]
        if vals:
            summary[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    print(f"\n=== {policy.upper()} ({len(pmetrics)} eps) ===")
    for k, v in summary.items():
        print(f"  {k:<30} {v['mean']:.4f} ±{v['std']:.4f}")

# Comparison table
print(f"\n{'='*70}")
print(f"{'Metric':<30} {'PPO':>12} {'BestStatic':>12} {'Δ':>10} {'Win?'}")
print("-"*70)
wins = 0
total = 0
for policy in ['ppo', 'best_static']:
    pmetrics = [m for m in all_metrics if m['policy']==policy]
    key_mets = ['ret_excel','ret_thesis','fill_rate','flow_fill_rate','lost_rate','lost_orders',
                'backorder_qty_final','service_loss_auc_per_order','ttr_mean','ttr_p95',
                'ctj_p50','ctj_p90','ctj_p99','rpj_p90','rpj_p99','dpj_p99',
                'delivered_rations','n_served','n_lost']
    for k in key_mets:
        vals = [float(m[k]) for m in pmetrics if k in m]
        if vals:
            summary[k] = float(np.mean(vals))

ppo_s = {k: float(np.mean([float(m[k]) for m in [x for x in all_metrics if x['policy']=='ppo'] if k in m]))
         for k in key_mets}
bs_s = {k: float(np.mean([float(m[k]) for m in [x for x in all_metrics if x['policy']=='best_static'] if k in m]))
        for k in key_mets}

for k, direction in [('ret_excel','higher'),('flow_fill_rate','higher'),('fill_rate','higher'),
    ('lost_rate','lower'),('lost_orders','lower'),('backorder_qty_final','lower'),
    ('service_loss_auc_per_order','lower'),('ttr_mean','lower'),('ttr_p95','lower'),
    ('ctj_p50','lower'),('ctj_p90','lower'),('ctj_p99','lower'),
    ('rpj_p90','lower'),('rpj_p99','lower'),('dpj_p99','lower'),
    ('delivered_rations','higher'),('n_served','higher'),('n_lost','lower')]:
    if k in ppo_s and k in bs_s:
        pv = ppo_s[k]; bv = bs_s[k]
        delta = pv - bv
        if direction == 'higher': win = pv > bv
        else: win = pv < bv
        if win: wins += 1
        total += 1
        print(f"{k:<30} {pv:>12.4f} {bv:>12.4f} {delta:>+10.4f} {'✅' if win else '—'}")
print("-"*70)
print(f"PPO wins on {wins}/{total} metrics vs dense best static")
(OUT / "audit_summary.json").write_text(json.dumps({"ppo_summary": ppo_s, "best_static_summary": bs_s, "wins": wins, "total": total}, indent=2))
print(f"\nWROTE {OUT}")
