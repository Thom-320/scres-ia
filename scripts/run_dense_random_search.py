#!/usr/bin/env python3
"""Random dense search around the per-op sweet spot to find a better policy.

Searches continuous [op3, op5, op9] + categorical S across all 9 campaign regimes.
If a better point than the best static (0.155254) exists, CEM/DE would find it.
This is the gate: if nothing beats 0.155254, PPO has no target.
"""
import sys, time, json, re, csv
from pathlib import Path
from statistics import fmean
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.run_track_a_headroom_search import FAMILY_RISKS
from supply_chain.continuous_its_env import make_per_op_buffer_track_a_env
from supply_chain.episode_metrics import compute_episode_metrics

GATE_DIR = Path("outputs/experiments/track_a_conflict_gate_per_op_full4_2026-06-29")
N_SAMPLES = 100
EVAL_SEED0 = 9000
HORIZON = 52
OUTPUT_DIR = Path("outputs/experiments/dense_random_search_2026-06-30")

summary = json.loads((GATE_DIR / "gate_summary.json").read_text())
regimes = list(summary["best_by_regime"].keys())

# Best known static
BEST_STATIC = 0.155254
BEST_ACTION = np.array([0.0, 0.1, 0.0, 0.0])  # op3,op5,op9,S2

def evaluate(action_tuple, seed0=EVAL_SEED0):
    """Run one episode per regime, return mean excel."""
    op3, op5, op9, shift_sig = action_tuple
    excels = []
    action = np.array([op3, op5, op9, shift_sig], dtype=np.float32)
    for i, regime in enumerate(regimes):
        family, phi, psi = re.fullmatch(r"(.+)_phi([0-9.]+)_psi([0-9.]+)", regime).groups()
        env = make_per_op_buffer_track_a_env(
            reward_mode="ReT_excel_plus_cvar", observation_version="v6", risk_level="current",
            enabled_risks=FAMILY_RISKS[family],
            risk_frequency_multiplier=float(phi), risk_impact_multiplier=float(psi),
            stochastic_pt=False, max_steps=HORIZON, step_size_hours=168.0,
            init_fracs=(op3, op5, op9), risk_obs=True, holding_cost=0.0,
            shift_cost=0.001, ret_excel_cvar_alpha=0.1)
        env.reset(seed=seed0 + i)
        obs, _ = env.reset(seed=seed0 + i)
        done = truncated = False
        while not (done or truncated):
            obs, r, done, truncated, info = env.step(action)
        excels.append(float(compute_episode_metrics(env.unwrapped.sim).get("ret_excel", 0)))
        env.close()
    return float(np.mean(excels))


def shift_label(shift_sig):
    return {-1.0: "S1", 0.0: "S2", 1.0: "S3"}[float(shift_sig)]

# Baselines
robust_action = (0.0, 0.0, 0.0, 0.0)  # S2
best_static_action = (0.0, 0.1, 0.0, 0.0)  # S2

print("Evaluating baselines...", flush=True)
robust_excel = evaluate(robust_action)
best_static_excel = evaluate(best_static_action)
print(f"  Robust [0,0,0,S2]:    {robust_excel:.6f}")
print(f"  BestStatic [0,0.1,0,S2]: {best_static_excel:.6f}")
print()

# Random search: sample around the sweet spot
rng = np.random.default_rng(42)
best_found = {"excel": best_static_excel, "action": best_static_action}
results = []

# Generate random candidates
# op3: mostly 0 (beta distribution skewed to 0)
# op5: around 0.10 (the known sweet spot dimension)
# op9: mostly 0 (beta skewed to 0)
# S: uniform {1,2,3}
candidates = []
for _ in range(N_SAMPLES):
    op3 = rng.beta(0.5, 5) * 0.3  # skewed to 0, max 0.3
    op5 = rng.beta(2, 5) * 0.3     # around 0.05-0.15
    op9 = rng.beta(0.5, 5) * 0.5   # skewed to 0, max 0.5
    S = rng.choice([1, 2, 3])
    shift_sig = {1: -1.0, 2: 0.0, 3: 1.0}[S]
    candidates.append((op3, op5, op9, shift_sig))

# Also include structured grid points (reduced)
for op9 in [0.0, 0.10, 0.20]:
    for op5 in [0.0, 0.05, 0.10, 0.15]:
        for op3 in [0.0]:
            for S in [1, 2, 3]:
                shift_sig = {1: -1.0, 2: 0.0, 3: 1.0}[S]
                candidates.append((op3, op5, op9, shift_sig))

print(f"Searching {len(candidates)} candidates...", flush=True)
t0 = time.time()
for i, action in enumerate(candidates):
    excel = evaluate(action)
    if excel > best_found["excel"]:
        best_found = {"excel": excel, "action": action}
        delta = excel - best_static_excel
        print(f"  NEW BEST: op3={action[0]:.3f} op5={action[1]:.3f} op9={action[2]:.3f} {shift_label(action[3])} excel={excel:.6f} (Δ{delta:+.6f})", flush=True)
    results.append(excel)
    if (i+1) % 10 == 0:
        print(f"  {i+1}/{len(candidates)}... best={best_found['excel']:.6f}", flush=True)

elapsed = time.time() - t0
print(f"\nDone in {elapsed:.0f}s ({len(candidates)} evals)", flush=True)
print(f"\n{'='*60}")
print(f"Best static known:    {BEST_STATIC:.6f}")
print(f"Best static eval'd:   {best_static_excel:.6f}")
print(f"Best random found:    {best_found['excel']:.6f}")
delta = best_found["excel"] - best_static_excel
print(f"Delta vs static:      {delta:+.6f}")
print(f"Best action:          op3={best_found['action'][0]:.4f} op5={best_found['action'][1]:.4f} op9={best_found['action'][2]:.4f} shift_sig={best_found['action'][3]:.1f}")
print(f"Verdict:              {'WIN FOUND!' if delta > 0 else 'NO BETTER POINT — static is optimal'}")
print(f"{'='*60}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
with (OUTPUT_DIR / "summary.json").open("w", encoding="utf-8") as fh:
    json.dump(
        {
            "horizon": HORIZON,
            "eval_seed0": EVAL_SEED0,
            "n_candidates": len(candidates),
            "best_static_known": BEST_STATIC,
            "best_static_evaluated": best_static_excel,
            "best_random_found": best_found["excel"],
            "delta_vs_evaluated_static": delta,
            "best_action": {
                "op3": best_found["action"][0],
                "op5": best_found["action"][1],
                "op9": best_found["action"][2],
                "shift_signal": best_found["action"][3],
                "shift": shift_label(best_found["action"][3]),
            },
            "raw_win_vs_evaluated_static": delta > 0,
            "raw_win_vs_known_static": best_found["excel"] > BEST_STATIC,
        },
        fh,
        indent=2,
    )
with (OUTPUT_DIR / "candidates.csv").open("w", newline="", encoding="utf-8") as fh:
    writer = csv.writer(fh)
    writer.writerow(["op3", "op5", "op9", "shift_signal", "shift", "excel"])
    for action, excel in zip(candidates, results):
        writer.writerow([action[0], action[1], action[2], action[3], shift_label(action[3]), excel])
print(f"WROTE {OUTPUT_DIR / 'summary.json'}", flush=True)
