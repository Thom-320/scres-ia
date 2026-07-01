#!/usr/bin/env python3
"""Track B campaign runner — PPO with regime-conditional BC warm-start.

Uses headroom matrix output to get per-regime best static actions,
trains PPO with BC warm-start on R2/R24 campaigns, and evaluates
against the static frontier.
"""
import sys, time, json, csv, re
from pathlib import Path
from statistics import fmean, median
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from supply_chain.env_experimental_shifts import MFSCGymEnvShifts
from supply_chain.episode_metrics import compute_episode_metrics
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

HEADROOM_DIR = Path("outputs/experiments/track_b_headroom_matrix_risklevel_2026-06-30")
OUT = Path("outputs/experiments/track_b_campaign_r2_r24_2026-07-01")
OUT.mkdir(parents=True, exist_ok=True)

# Load headroom matrix results
with (HEADROOM_DIR / "family_headroom_summary.csv").open() as f:
    promoted = [r for r in csv.DictReader(f) if r['verdict'] == 'PROMOTE']

# Load per-regime best actions
with (HEADROOM_DIR / "seed_metrics.csv").open() as f:
    all_cells = list(csv.DictReader(f))

# Build regime list: R2/R24 at promoted risk levels (exclude 'all')
regimes = []
regime_best_actions = {}
regime_names = []
for p in promoted:
    rl, fam = p['risk_level'], p['family']
    if fam == 'all':
        continue  # Skip all-risks (ReT scale too different)
    # Get best action for this regime
    regime_cells = [c for c in all_cells 
                  if c['risk_level']==rl and c['family']==fam 
                  and c['phi']=='1.0' and c['psi']=='1.0' and c['demand_mult']=='1.0']
    if regime_cells:
        regime_name = f"{rl}_{fam}"
        if regime_name not in regime_best_actions:
            best_cell = max(regime_cells, key=lambda x: float(x['ret_excel']))
            best_action = np.array([
                1.0, 1.0, 1.0, 1.0, 1.0,
                float(best_cell['shift']),
                float(best_cell['op10_mult']),
                float(best_cell['op12_mult']),
            ], dtype=np.float32)
            regime_best_actions[regime_name] = best_action
            regime_names.append(regime_name)
            regimes.append((rl, fam, 1.0, 1.0, 1.0))
            print(f"  {regime_name}: S{int(best_action[5])}_op10={best_action[6]}_op12={best_action[7]} ret={float(best_cell['ret_excel']):.4f}", flush=True)

if len(regime_names) < 2:
    print("Not enough regimes!", flush=True)
    sys.exit(1)

print(f"\nCampaign: {len(regime_names)} regimes", flush=True)

# Build env
def build_env(risk_level, family, seed):
    env = MFSCGymEnvShifts(
        reward_mode="ReT_garrido2024_train",
        observation_version="v7",
        risk_level=risk_level,
        stochastic_pt=True,
        max_steps=52,  # h52 for speed
        step_size_hours=168.0,
        year_basis="thesis",
        action_contract="track_b_v1",
    )
    env.reset(seed=int(seed))
    return env

# Evaluate a fixed action across all regimes
def eval_action(action, seed0):
    excels = []
    for i, (rl, fam, phi, psi, dm) in enumerate(regimes):
        env = build_env(rl, fam, seed0 + i)
        obs, _ = env.reset(seed=seed0 + i)
        done = False
        while not done:
            obs, r, done, trunc, info = env.step(action)
        excels.append(float(compute_episode_metrics(env.unwrapped.sim).get("ret_excel", 0)))
        env.close()
    return float(np.mean(excels))

# Evaluate: per-regime normalized score
def eval_action_normalized(action, seed0):
    """Return mean of per-regime normalized scores."""
    scores = []
    for i, (rl, fam, phi, psi, dm) in enumerate(regimes):
        env = build_env(rl, fam, seed0 + i)
        obs, _ = env.reset(seed=seed0 + i)
        done = False
        while not done:
            obs, r, done, trunc, info = env.step(action)
        ret = float(compute_episode_metrics(env.unwrapped.sim).get("ret_excel", 0))
        env.close()
        # Normalize by regime's best (oracle) for this regime
        rn = regime_names[i]
        oracle_ret = float([c for c in all_cells 
                          if c['risk_level']==rl and c['family']==fam 
                          and c['phi']=='1.0' and c['psi']=='1.0' and c['demand_mult']=='1.0'
                          ][0]['ret_excel'])  # approximate
        # Use the best cell for this regime as normalizer
        regime_cells = [c for c in all_cells 
                      if c['risk_level']==rl and c['family']==fam 
                      and c['phi']=='1.0' and c['psi']=='1.0' and c['demand_mult']=='1.0']
        best_regime_ret = max(float(c['ret_excel']) for c in regime_cells)
        scores.append(ret / max(best_regime_ret, 1e-9))
    return float(np.mean(scores))

# Robust eval (S3_1.5x_1.5x across all regimes, normalized)
robust_norm = eval_action_normalized(np.array([1.0,1.0,1.0,1.0,1.0,3,1.5,1.5], dtype=np.float32), 9000)
# Oracle = 1.0 (perfect per-regime)
print(f"Robust (normalized): {robust_norm:.4f}", flush=True)
print(f"Oracle (normalized): 1.0000", flush=True)
headroom = 1.0 - robust_norm
print(f"Headroom (normalized): {headroom:+.4f}", flush=True)

# BC: collect trajectories from oracle
bc_obs, bc_acts = [], []
for i, (rl, fam, phi, psi, dm) in enumerate(regimes):
    rn = regime_names[i]
    tgt = regime_best_actions[rn]
    env = build_env(rl, fam, 9000 + 1000 + i)
    obs, _ = env.reset(seed=9000 + 1000 + i)
    done = False
    while not done:
        bc_obs.append(np.asarray(obs, dtype=np.float32).copy())
        bc_acts.append(tgt.copy())
        obs, r, done, trunc, info = env.step(tgt)
    env.close()
bc_obs = np.vstack(bc_obs).astype(np.float32)
bc_acts = np.vstack(bc_acts).astype(np.float32)
print(f"BC data: {len(bc_obs)} samples", flush=True)

# Train PPO with BC warm-start
seeds = [1, 2]
learned = []
for seed in seeds:
    print(f"Seed {seed}...", end=" ", flush=True)
    t0 = time.time()
    
    env_fns = []
    for i in range(4):  # n_envs=4
        rl, fam, phi, psi, dm = regimes[i % len(regimes)]
        env_fns.append(lambda rl=rl, fam=fam, s=seed+i: build_env(rl, fam, s))
    
    venv = VecNormalize(DummyVecEnv(env_fns), norm_obs=True, norm_reward=True, clip_reward=10.0)
    
    def lr_sched(p): return 1e-4 * (0.1 + 0.9 * (1.0 - p))
    model = PPO("MlpPolicy", venv, seed=seed, verbose=0,
                n_steps=256, batch_size=64, learning_rate=lr_sched, n_epochs=10)
    
    # BC train
    obs_t = torch.as_tensor(bc_obs, device=model.policy.device)
    act_t = torch.as_tensor(bc_acts, device=model.policy.device)
    rng = np.random.default_rng(seed)
    for _ in range(100):
        order = rng.permutation(len(bc_obs))
        for start in range(0, len(bc_obs), 128):
            idx = torch.as_tensor(order[start:start+128], dtype=torch.long, device=obs_t.device)
            pred = model.policy.get_distribution(obs_t[idx]).mode()
            loss = torch.nn.functional.mse_loss(pred, act_t[idx])
            model.policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 0.5)
            model.policy.optimizer.step()
    
    model.learn(total_timesteps=20000)
    
    # Eval
    dyn = eval_policy_norm(model, seed)
    learned.append({"seed": seed, "excel": dyn})
    venv.close()
    print(f"score={dyn:.4f}", flush=True)

def eval_policy_norm(model, seed0):
    scores = []
    for i, (rl, fam, phi, psi, dm) in enumerate(regimes):
        env = build_env(rl, fam, seed0 + i)
        obs, _ = env.reset(seed=seed0 + i)
        done = False
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, done, trunc, info = env.step(a)
        ret = float(compute_episode_metrics(env.unwrapped.sim).get("ret_excel", 0))
        regime_cells = [c for c in all_cells 
                      if c['risk_level']==rl and c['family']==fam 
                      and c['phi']=='1.0' and c['psi']=='1.0' and c['demand_mult']=='1.0']
        best_regime_ret = max(float(c['ret_excel']) for c in regime_cells)
        scores.append(ret / max(best_regime_ret, 1e-9))
        env.close()
    return float(np.mean(scores))

dyn_mean = fmean(r["excel"] for r in learned)
dyn_med = median(r["excel"] for r in learned)

report = [
    f"# Track B Campaign — R2+R24",
    f"Regimes: {len(regime_names)} ({', '.join(regime_names)})",
    f"Robust (normalized): {robust_norm:.4f}",
    f"Oracle (normalized): 1.0000",
    f"Dynamic mean: {dyn_mean:.4f}",
    f"Dynamic median: {dyn_med:.4f}",
    f"Headroom: {headroom:+.4f}",
    f"Raw win vs robust: {dyn_mean > robust_norm}",
    f"Raw win vs oracle: {dyn_mean > 1.0}",
]
for r in learned:
    report.append(f"  seed {r['seed']}: excel={r['excel']:.4f}")
print("\n".join(report))
(OUT / "report.md").write_text("\n".join(report))
print(f"\nWROTE {OUT}")
