#!/usr/bin/env python3
"""V2: BC target = best static [0, 0.1, 0, 0] from full frontier re-evaluation."""
import sys, time, json, re, csv
from pathlib import Path
from statistics import fmean, median
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.run_track_a_headroom_search import FAMILY_RISKS
from supply_chain.continuous_its_env import make_per_op_buffer_track_a_env
from supply_chain.episode_metrics import compute_episode_metrics
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

OUT = Path("outputs/experiments/track_a_engineered_v2_2026-06-29")
GATE_DIR = Path("outputs/experiments/track_a_conflict_gate_per_op_full4_2026-06-29")
SEEDS = list(range(1, 21))
TIMESTEPS, BC_EPOCHS, N_ENVS, MAX_STEPS = 60000, 200, 4, 52
HOLDING_COST, CVAR_ALPHA, BC_REG, EVAL_SEED0 = 0.0, 0.1, 0.1, 9000
REWARD_MODE = "ReT_excel_plus_cvar"
OUT.mkdir(parents=True, exist_ok=True)

summary = json.loads((GATE_DIR/"gate_summary.json").read_text())
regimes = list(summary["best_by_regime"].keys())
BEST_STATIC_ACTION = np.array([0.0, 0.1, 0.0, 0.0], dtype=np.float32)
bc_actions = {r: BEST_STATIC_ACTION for r in regimes}

def build(regime, seed):
    family, phi, psi = re.fullmatch(r"(.+)_phi([0-9.]+)_psi([0-9.]+)", regime).groups()
    env = make_per_op_buffer_track_a_env(
        reward_mode=REWARD_MODE, observation_version="v6", risk_level="current",
        enabled_risks=FAMILY_RISKS[family], risk_frequency_multiplier=float(phi),
        risk_impact_multiplier=float(psi), stochastic_pt=False, max_steps=MAX_STEPS,
        step_size_hours=168.0, init_fracs=(0.0,0.0,0.0), risk_obs=True,
        holding_cost=HOLDING_COST, shift_cost=0.001, ret_excel_cvar_alpha=CVAR_ALPHA)
    env.reset(seed=int(seed)); return env

def eval_action(action, seed0):
    excels = []
    for i, regime in enumerate(regimes):
        env = build(regime, seed0+i); obs,_ = env.reset(seed=seed0+i); done=False
        while not done: obs, r, done, trunc, info = env.step(action)
        excels.append(float(compute_episode_metrics(env.unwrapped.sim).get("ret_excel",0)))
        env.close()
    return float(np.mean(excels))

robust_excel = eval_action(np.array([0.0,0.0,0.0,0.0],dtype=np.float32), EVAL_SEED0)
best_excel = eval_action(BEST_STATIC_ACTION, EVAL_SEED0)
print(f"Robust [0,0,0,0]: {robust_excel:.6f} | BestStatic [0,0.1,0,0]: {best_excel:.6f}", flush=True)

bc_obs,bc_acts=[],[]
for i, regime in enumerate(regimes):
    env=build(regime, EVAL_SEED0+1000+i); obs,_=env.reset(seed=EVAL_SEED0+1000+i); done=False
    while not done: bc_obs.append(np.asarray(obs,dtype=np.float32).copy()); bc_acts.append(BEST_STATIC_ACTION.copy()); obs,r,done,trunc,info=env.step(BEST_STATIC_ACTION)
    env.close()
bc_obs,bc_acts=np.vstack(bc_obs).astype(np.float32),np.vstack(bc_acts).astype(np.float32)
print(f"BC: {len(bc_obs)} samples", flush=True)

learned, t0 = [], time.time()
for si, seed in enumerate(SEEDS):
    env_fns = [lambda r=regimes[i%len(regimes)], s=seed+i: build(r,s) for i in range(N_ENVS)]
    venv = VecNormalize(DummyVecEnv(env_fns), norm_obs=True, norm_reward=True, clip_reward=10.0)
    model = PPO("MlpPolicy", venv, seed=seed, verbose=0, n_steps=min(512,MAX_STEPS*4),
                batch_size=64, learning_rate=lambda p: 1e-4*(0.1+0.9*(1.0-p)), n_epochs=10)
    obs_t = torch.as_tensor(bc_obs, device=model.policy.device)
    act_t = torch.as_tensor(bc_acts, device=model.policy.device)
    rng = np.random.default_rng(seed)
    for _ in range(BC_EPOCHS):
        order = rng.permutation(len(bc_obs))
        for start in range(0, len(bc_obs), 128):
            idx = torch.as_tensor(order[start:start+128], dtype=torch.long, device=obs_t.device)
            pred = model.policy.get_distribution(obs_t[idx]).mode()
            loss = torch.nn.functional.mse_loss(pred, act_t[idx])
            model.policy.optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 0.5); model.policy.optimizer.step()
    half = TIMESTEPS//2; done=0
    while done < TIMESTEPS:
        chunk = min(4096, TIMESTEPS-done)
        model.learn(total_timesteps=chunk, reset_num_timesteps=(done==0)); done+=chunk
        if done <= half and done % 2048 < chunk:
            for _ in range(3):
                o2 = rng.permutation(min(256, len(bc_obs)))
                idx2 = torch.as_tensor(o2, dtype=torch.long, device=obs_t.device)
                pred2 = model.policy.get_distribution(obs_t[idx2]).mode()
                (BC_REG * torch.nn.functional.mse_loss(pred2, act_t[idx2])).backward()
                model.policy.optimizer.step(); model.policy.optimizer.zero_grad()
    excels=[]
    for i, regime in enumerate(regimes):
        env=build(regime, EVAL_SEED0+i); obs,_=env.reset(seed=EVAL_SEED0+i); done=False
        while not done: a,_=model.predict(obs,deterministic=True); obs,r,done,trunc,info=env.step(a)
        excels.append(float(compute_episode_metrics(env.unwrapped.sim).get("ret_excel",0)))
        env.close()
    learned.append({"seed":seed,"excel":float(np.mean(excels))}); venv.close()
    print(f"  s{seed}: excel={learned[-1]['excel']:.6f}", flush=True)

excels = [r["excel"] for r in learned]
dyn_mean, dyn_med = fmean(excels), median(excels)
beats_robust = sum(1 for e in excels if e > robust_excel)
beats_best = sum(1 for e in excels if e > best_excel)
rng2 = np.random.default_rng(42)
boot = [float(np.mean(rng2.choice(excels,size=len(excels),replace=True))) for _ in range(10000)]
ci_lo, ci_hi = float(np.percentile(boot,2.5)), float(np.percentile(boot,97.5))

report = [
    f"ENGINEERED V2: BC target = best static [0,0.1,0,0]",
    f"Robust [0,0,0,0]: {robust_excel:.6f} | BestStatic: {best_excel:.6f}",
    f"Dynamic: mean={dyn_mean:.6f} median={dyn_med:.6f}",
    f"CI95: [{ci_lo:.6f}, {ci_hi:.6f}]",
    f"Beats robust: {beats_robust}/{len(SEEDS)} | Beats best_static: {beats_best}/{len(SEEDS)}",
    f"CI95 > robust: {ci_lo > robust_excel} | CI95 > best_static: {ci_lo > best_excel}",
]
for i, (e, s) in enumerate(zip(excels, SEEDS)):
    m = "✅" if e > best_excel else ("~" if e > robust_excel else "—")
    report.append(f"  seed {s:2d}: excel={e:.6f} {m}")
report.append(f"\nWROTE {OUT} in {time.time()-t0:.0f}s")
print("\n".join(report), flush=True)
json.dump({"learned": learned, "dynamic": {"mean":dyn_mean,"median":dyn_med},
           "robust_excel":robust_excel, "best_static_excel":best_excel,
           "ci95":[ci_lo,ci_hi], "beats_robust":beats_robust, "beats_best":beats_best,
           "n_seeds":len(SEEDS)}, (OUT/"summary.json").open("w"), indent=2)
(OUT/"report.md").write_text("\n".join(report))
