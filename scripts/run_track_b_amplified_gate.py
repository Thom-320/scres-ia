#!/usr/bin/env python3
"""Amplified regime gate for Track B — measure headroom with extreme contrast."""
import sys, time, json, csv
from pathlib import Path
from statistics import fmean
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from supply_chain.env_experimental_shifts import MFSCGymEnvShifts
from supply_chain.episode_metrics import compute_episode_metrics

# Create amplified regime params — MORE contrast between calm and crisis
AMPLIFIED_REGIMES = {
    "calm": {"risk_intensity": 0.3, "surge_scale": 0.5, "recovery_scale": 0.8, "demand_scale": 0.95},
    "crisis": {"risk_intensity": 3.5, "surge_scale": 3.0, "recovery_scale": 1.5, "demand_scale": 1.15},
}

SHIFTS = [1, 2, 3]
OP10_MULTS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
OP12_MULTS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
SEEDS = [7100, 7101]

OUT = Path("outputs/experiments/track_b_amplified_gate_2026-07-01")
OUT.mkdir(parents=True, exist_ok=True)

n_cells = len(AMPLIFIED_REGIMES) * len(SHIFTS) * len(OP10_MULTS) * len(OP12_MULTS) * len(SEEDS)
print(f"AMPLIFIED GATE: {len(AMPLIFIED_REGIMES)} regimes × {len(SHIFTS)} shifts × {len(OP10_MULTS)}×{len(OP12_MULTS)} dispatch × {len(SEEDS)} seeds = {n_cells} cells", flush=True)

t0 = time.time()
rows = []
n=0
for rname, rparams in AMPLIFIED_REGIMES.items():
    for shift in SHIFTS:
        for op10 in OP10_MULTS:
            for op12 in OP12_MULTS:
                for seed in SEEDS:
                    env = MFSCGymEnvShifts(
                        reward_mode="control_v1", observation_version="v7",
                        risk_level="adaptive_benchmark_v2",
                        risk_frequency_multiplier=rparams["risk_intensity"],
                        risk_impact_multiplier=rparams["surge_scale"],
                        stochastic_pt=True, max_steps=104, step_size_hours=168.0,
                        year_basis="thesis", action_contract="track_b_v1")
                    env.reset(seed=seed)
                    done = truncated = False
                    while not (done or truncated):
                        a = np.array([1.0,1.0,1.0,1.0,1.0,shift,op10,op12],dtype=np.float32)
                        obs, r, done, truncated, info = env.step(a)
                    metrics = compute_episode_metrics(env.unwrapped.sim)
                    rows.append({
                        'regime': rname, 'shift': shift, 'op10': op10, 'op12': op12, 'seed': seed,
                        'ret_excel': float(metrics.get('ret_excel',0)),
                        'flow_fill_rate': float(metrics.get('flow_fill_rate',0)),
                        'lost_rate': float(metrics.get('lost_rate',0)),
                        'ttr_mean': float(metrics.get('ttr_mean',0)),
                    })
                    env.close()
                    n+=1
                    if n%50==0: print(f"  {n}/{n_cells}", flush=True)

elapsed = time.time()-t0
print(f"Done in {elapsed:.0f}s", flush=True)

# Per-regime best
regime_best = {}
for rname in AMPLIFIED_REGIMES:
    rrows = [r for r in rows if r['regime']==rname]
    grid = {}
    for r in rrows:
        k = (r['shift'], r['op10'], r['op12'])
        grid.setdefault(k, []).append(r['ret_excel'])
    best_k = max(grid, key=lambda k: fmean(grid[k]))
    regime_best[rname] = {'shift': best_k[0], 'op10': best_k[1], 'op12': best_k[2], 'ret_excel': fmean(grid[best_k])}
    print(f"  {rname}: best=S{best_k[0]}_op10={best_k[1]}_op12={best_k[2]} ret={fmean(grid[best_k]):.6f}", flush=True)

# Oracle
oracle_ret = fmean(v['ret_excel'] for v in regime_best.values())

# Best constant
constant_grid = {}
for r in rows:
    k = (r['shift'], r['op10'], r['op12'])
    constant_grid.setdefault(k, []).append(r['ret_excel'])
best_const_k = max(constant_grid, key=lambda k: fmean(constant_grid[k]))
best_const_ret = fmean(constant_grid[best_const_k])

headroom = oracle_ret - best_const_ret
print(f"\nOracle (best per regime): {oracle_ret:.6f}", flush=True)
print(f"Best constant: S{best_const_k[0]}_op10={best_const_k[1]}_op12={best_const_k[2]} ret={best_const_ret:.6f}", flush=True)
print(f"HEADROOM: {headroom:+.6f} ({headroom/oracle_ret*100:+.1f}%)", flush=True)
actions_vary = len(set((r['shift'],r['op10'],r['op12']) for r in regime_best.values())) > 1
print(f"Actions vary across regimes: {actions_vary}", flush=True)
print(f"VERDICT: {'PROMOTE' if headroom > 0.0005 else 'MARGINAL' if headroom > 0.0001 else 'NULL'}", flush=True)

(OUT/"gate.json").write_text(json.dumps({
    "regimes": AMPLIFIED_REGIMES,
    "regime_best": {k: {"shift":v['shift'],"op10":v['op10'],"op12":v['op12'],"ret_excel":v['ret_excel']} for k,v in regime_best.items()},
    "oracle_ret": oracle_ret,
    "best_constant": {"shift":best_const_k[0],"op10":best_const_k[1],"op12":best_const_k[2],"ret_excel":best_const_ret},
    "headroom": headroom,
    "actions_vary": actions_vary,
}, indent=2))
print(f"WROTE {OUT}", flush=True)
