#!/usr/bin/env python3
"""Dense static frontier for Track B with CRN eval."""
import sys, time, json, csv
from pathlib import Path
from statistics import fmean
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from supply_chain.env_experimental_shifts import MFSCGymEnvShifts
from supply_chain.episode_metrics import compute_episode_metrics

SHIFTS = [1, 2, 3]
OP10_MULT = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5]
OP12_MULT = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5]
SEEDS = [7100, 7101, 7102]
CONFIG = dict(
    reward_mode="control_v1", observation_version="v7",
    risk_level="adaptive_benchmark_v2", stochastic_pt=True,
    max_steps=104, step_size_hours=168.0, year_basis="thesis",
    action_contract="track_b_v1")

OUT = Path("outputs/experiments/track_b_dense_frontier_2026-07-01")
OUT.mkdir(parents=True, exist_ok=True)

n_total = len(SHIFTS) * len(OP10_MULT) * len(OP12_MULT) * len(SEEDS)
print(f"DENSE TRACK B FRONTIER: {len(SHIFTS)} shifts × {len(OP10_MULT)} op10 × {len(OP12_MULT)} op12 × {len(SEEDS)} seeds = {n_total} cells", flush=True)

t0 = time.time()
rows = []
n = 0
for shift in SHIFTS:
    for op10m in OP10_MULT:
        for op12m in OP12_MULT:
            for seed in SEEDS:
                env = MFSCGymEnvShifts(**CONFIG)
                obs, _ = env.reset(seed=seed)
                done = truncated = False
                while not (done or truncated):
                    a = np.array([1.0, 1.0, 1.0, 1.0, 1.0, shift, op10m, op12m], dtype=np.float32)
                    obs, r, done, truncated, info = env.step(a)
                metrics = compute_episode_metrics(env.unwrapped.sim)
                rows.append({
                    'shift': shift, 'op10_mult': op10m, 'op12_mult': op12m, 'seed': seed,
                    'ret_excel': float(metrics.get('ret_excel', 0)),
                    'flow_fill_rate': float(metrics.get('flow_fill_rate', 0)),
                    'fill_rate': float(metrics.get('fill_rate', 0)),
                    'lost_rate': float(metrics.get('lost_rate', 0)),
                    'lost_orders': float(metrics.get('lost_orders', 0)),
                    'service_loss_auc': float(metrics.get('service_loss_auc_per_order', 0)),
                    'ttr_mean': float(metrics.get('ttr_mean', 0)),
                    'ctj_p99': float(metrics.get('ctj_p99', 0)),
                    'delivered_rations': float(metrics.get('delivered_rations', 0)),
                })
                env.close()
                n += 1
                if n % 50 == 0:
                    print(f"  {n}/{n_total} ({time.time()-t0:.0f}s)", flush=True)

elapsed = time.time() - t0
print(f"Done in {elapsed:.0f}s", flush=True)

# Aggregate across seeds
with (OUT / "cells.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

grid = {}
for r in rows:
    k = (r['shift'], r['op10_mult'], r['op12_mult'])
    grid.setdefault(k, []).append(r)

summary_rows = []
for (s, o10, o12), rlist in sorted(grid.items()):
    summary_rows.append({
        'shift': s, 'op10_mult': o10, 'op12_mult': o12,
        'ret_excel_mean': fmean(r['ret_excel'] for r in rlist),
        'flow_fill_mean': fmean(r['flow_fill_rate'] for r in rlist),
        'lost_rate_mean': fmean(r['lost_rate'] for r in rlist),
        'delivered_mean': fmean(r['delivered_rations'] for r in rlist),
        'n_seeds': len(rlist),
    })

with (OUT / "summary.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
    w.writeheader(); w.writerows(summary_rows)

# Find best static
best = max(summary_rows, key=lambda r: r['ret_excel_mean'])
print(f"\nBEST STATIC: S{best['shift']}_op10_{best['op10_mult']}_op12_{best['op12_mult']}")
print(f"  ret_excel={best['ret_excel_mean']:.6f} flow_fill={best['flow_fill_mean']:.4f} lost_rate={best['lost_rate_mean']:.4f}")
print(f"WROTE {OUT}")
