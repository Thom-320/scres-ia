#!/usr/bin/env python3
"""1.5x demand gate: does higher demand create headroom?"""
import sys, time, json, itertools
from pathlib import Path
from statistics import fmean
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from supply_chain.continuous_its_env import make_per_op_buffer_track_a_env
from supply_chain.episode_metrics import compute_episode_metrics

SHIFT_SIGS = {1: -1.0, 2: 0.0, 3: 1.0}
SEEDS = [7100, 7101]
HORIZON = 52
PHIS = [1, 2, 4, 8, 16]
OP9_FRACS = [0.0, 0.05, 0.10, 0.15, 0.25, 0.50]
FAMILIES = [("R24", ("R24",)), ("R13", ("R13",))]

print("DEMAND 1.5x GATE: R24+R13, phi={1,2,4,8,16}, 6 op9 fracs, 3 shifts, 2 seeds", flush=True)
print(f"Total cells: {len(FAMILIES)*len(PHIS)*len(OP9_FRACS)*3*len(SEEDS)}", flush=True)
print(flush=True)

rows = []
t0 = time.time()
for family_name, enabled in FAMILIES:
    for phi in PHIS:
        for op9 in OP9_FRACS:
            for shift in [1, 2, 3]:
                for seed in SEEDS:
                    env = make_per_op_buffer_track_a_env(
                        reward_mode='ReT_excel_delta', observation_version='v6',
                        risk_level='current', enabled_risks=enabled,
                        risk_frequency_multiplier=phi, risk_impact_multiplier=1.5,
                        demand_mean_multiplier=1.5, stochastic_pt=False,
                        max_steps=HORIZON, step_size_hours=168.0,
                        init_fracs=(0.0, 0.0, op9), risk_obs=True,
                        holding_cost=0.0, shift_cost=0.001,
                    )
                    obs, info = env.reset(seed=seed)
                    done = truncated = False
                    while not (done or truncated):
                        action = np.array([0.0, 0.0, op9, SHIFT_SIGS[shift]], dtype=np.float32)
                        obs, reward, done, truncated, info = env.step(action)
                    excel = float(compute_episode_metrics(env.unwrapped.sim).get('ret_excel', 0))
                    res = 0.5*(op9/3.0) + 0.5*((shift-1)/2.0)
                    rows.append({'family': family_name, 'phi': phi, 'op9': op9, 'shift': shift, 'seed': seed,
                                 'excel': excel, 'resource': res})
                    env.close()
                    n = len(rows)
                    if n % 20 == 0:
                        print(f"  {n} cells...", end=" ", flush=True)

print(f"\ndone ({time.time()-t0:.0f}s)", flush=True)

# Per-regime best
regime_best = {}
for r in rows:
    k = f"{r['family']}_phi{r['phi']}"
    regime_best.setdefault(k, []).append(r)

print("\nBEST PER REGIME:", flush=True)
oracle_cells = []
for regime, rlist in sorted(regime_best.items()):
    # Average across seeds
    grid = {}
    for r in rlist:
        k2 = (r['shift'], r['op9'])
        grid.setdefault(k2, []).append(r['excel'])
    best_k = max(grid, key=lambda k: fmean(grid[k]))
    best_r = [r for r in rlist if r['shift']==best_k[0] and r['op9']==best_k[1]][0]
    oracle_cells.append(best_r)
    print(f"  {regime:20} best=S{best_r['shift']}_op9{best_r['op9']} excel={fmean(grid[best_k]):.5f} res={best_r['resource']:.3f}", flush=True)

# Best single constant
best_constant = None
best_constant_score = -float('inf')
for op9 in OP9_FRACS:
    for shift in [1, 2, 3]:
        cells = [r for r in rows if r['op9']==op9 and r['shift']==shift]
        if cells:
            score = fmean(r['excel'] for r in cells)
            if score > best_constant_score:
                best_constant_score = score
                best_constant = (op9, shift, score)

oracle_excel = fmean(r['excel'] for r in oracle_cells)
headroom = oracle_excel - best_constant[2]
actions = set((r['family'], r['phi'], r['shift']) for r in oracle_cells)
unique_shifts = set(r['shift'] for r in oracle_cells)

print(f"\nORACLE: {oracle_excel:.5f}", flush=True)
print(f"BEST CONSTANT: S{best_constant[1]}_op9{best_constant[0]} excel={best_constant[2]:.5f}", flush=True)
print(f"HEADROOM: {headroom:+.5f}", flush=True)
print(f"Unique shifts across regimes: {unique_shifts}", flush=True)
verdict = "PROMOTE" if headroom > 0.001 else "MARGINAL" if headroom > 0.0001 else "NULL"
print(f"VERDICT: {verdict}", flush=True)
