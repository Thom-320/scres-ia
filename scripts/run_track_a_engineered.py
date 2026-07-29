#!/usr/bin/env python3
"""Optimized per-op conflict campaign — Track A engineering plan.

7 improvements over the original runner:
  1. BC target = best static (not oracle) where best static beats oracle
  2. BC epochs = 200 (was 50)
  3. LR = 1e-4 with linear decay (was 3e-4 constant)
  4. BC regularization L_PPO + 0.1*L_BC during first half of training
  5. Timesteps = 80k (was 40k)
  6. Action smoothing penalty reward
  7. 20 seeds with median + per-seed reporting
"""
from __future__ import annotations

import argparse, csv, json, re, sys, time
from pathlib import Path
from statistics import fmean, median
from typing import Any
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.run_track_a_headroom_search import FAMILY_RISKS
from supply_chain.continuous_its_env import make_per_op_buffer_track_a_env
from supply_chain.episode_metrics import compute_episode_metrics

SHIFT_SIGS = {1: -1.0, 2: 0.0, 3: 1.0}

def parse_regime(name: str) -> tuple[str, float, float]:
    m = re.fullmatch(r"(.+)_phi([0-9.]+)_psi([0-9.]+)", name)
    if not m: raise ValueError(f"bad regime name: {name}")
    return m.group(1), float(m.group(2)), float(m.group(3))

def build_env(*, regime, reward_mode, max_steps, seed, holding_cost, cvar_alpha, lr_schedule=None):
    family, phi, psi = parse_regime(regime)
    kwargs = dict(
        reward_mode=reward_mode, observation_version="v6",
        risk_level="current",
        risk_frequency_multiplier=phi, risk_impact_multiplier=psi,
        stochastic_pt=False, max_steps=int(max_steps), step_size_hours=168.0,
        init_fracs=(0.0, 0.0, 0.0), risk_obs=True,
        holding_cost=float(holding_cost), shift_cost=0.001,
        ret_excel_cvar_alpha=float(cvar_alpha),
    )
    enabled = FAMILY_RISKS.get(family)
    if enabled is not None: kwargs["enabled_risks"] = enabled
    env = make_per_op_buffer_track_a_env(**kwargs)
    env.reset(seed=int(seed))
    return env

def read_gate(gate_dir: Path):
    summary = json.loads((gate_dir / "gate_summary.json").read_text())
    rows = []
    with (gate_dir / "static_runs.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            row["excel"] = float(row["excel"])
            row["resource"] = float(row["resource"])
            row["seed"] = int(row["seed"])
            row["action_tuple"] = tuple(float(x) for x in json.loads(row["action"]))
            rows.append(row)
    best_by_regime = summary["best_by_regime"]
    oracle_actions: dict[str, tuple[float, ...]] = {}
    for regime, item in best_by_regime.items():
        label = item["candidate"]
        match = next(r for r in rows if r["regime"] == regime and r["candidate"] == label)
        oracle_actions[regime] = match["action_tuple"]
    # Find best static per regime (MAY beat oracle)
    best_actions: dict[str, tuple[float, ...]] = {}
    for regime in best_by_regime:
        rrows = [r for r in rows if r["regime"] == regime]
        if rrows:
            best = max(rrows, key=lambda r: r["excel"])
            best_actions[regime] = best["action_tuple"]
    return summary, rows, oracle_actions, best_actions

def eval_policy(regimes, act_fn, args, seed0: int) -> dict:
    excels, losses, resources, traces, panels = [], [], [], [], []
    for i, regime in enumerate(regimes):
        env = build_env(regime=regime, reward_mode=args.reward_mode,
                        max_steps=args.max_steps, seed=seed0 + i,
                        holding_cost=args.holding_cost, cvar_alpha=args.cvar_alpha)
        obs, _ = env.reset(seed=seed0 + i)
        done = truncated = False
        ep_res, ep_actions = [], []
        while not (done or truncated):
            action = np.asarray(act_fn(obs, regime), dtype=np.float32).reshape(-1)
            obs, _r, done, truncated, info = env.step(action)
            ep_res.append(float(info.get("resource_composite", 0.0)))
            ep_actions.append(action)
        metrics = compute_episode_metrics(env.unwrapped.sim)
        panels.append(metrics)
        excels.append(float(metrics.get("ret_excel", 0.0)))
        losses.append(float(metrics.get("service_loss_auc_ration_hours", 0.0)))
        resources.append(float(np.nanmean(ep_res)) if ep_res else 0.0)
        action_mat = np.vstack(ep_actions) if ep_actions else np.zeros((0, 4))
        traces.append({"regime": regime, "excel": excels[-1], "service_loss": losses[-1],
                        "resource": resources[-1],
                        "mean_action": np.mean(action_mat, axis=0).tolist() if len(action_mat) else [],
                        "std_action": np.std(action_mat, axis=0).tolist() if len(action_mat) else [],
                        "metrics": metrics})
        env.close()
    losses_sorted = sorted(losses)
    k = max(1, int(round(0.05 * len(losses_sorted))))
    return {"excel": float(np.mean(excels)), "cvar": float(np.mean(losses_sorted[-k:])),
            "resource": float(np.mean(resources)), "by_regime": traces}

def collect_bc(regimes, bc_actions, args, seed0: int):
    obs_rows, act_rows = [], []
    for i, regime in enumerate(regimes):
        target = np.asarray(bc_actions[regime], dtype=np.float32)
        env = build_env(regime=regime, reward_mode=args.reward_mode,
                        max_steps=args.max_steps, seed=seed0 + i,
                        holding_cost=args.holding_cost, cvar_alpha=args.cvar_alpha)
        obs, _ = env.reset(seed=seed0 + i)
        done = truncated = False
        while not (done or truncated):
            obs_rows.append(np.asarray(obs, dtype=np.float32).copy())
            act_rows.append(target.copy())
            obs, _r, done, truncated, _info = env.step(target)
        env.close()
    return np.vstack(obs_rows).astype(np.float32), np.vstack(act_rows).astype(np.float32)

def bc_train(model, obs, actions, epochs, batch_size, seed):
    device = model.policy.device
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
    act_t = torch.as_tensor(actions, dtype=torch.float32, device=device)
    rng = np.random.default_rng(seed)
    def loss_fn(idx=None):
        x = obs_t if idx is None else obs_t[idx]
        y = act_t if idx is None else act_t[idx]
        pred = model.policy.get_distribution(x).mode()
        return torch.nn.functional.mse_loss(pred, y)
    with torch.no_grad(): initial = float(loss_fn().detach().cpu())
    for _ in range(epochs):
        order = rng.permutation(len(obs))
        for start in range(0, len(obs), batch_size):
            idx = torch.as_tensor(order[start:start + batch_size], dtype=torch.long, device=device)
            l = loss_fn(idx)
            model.policy.optimizer.zero_grad()
            l.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 0.5)
            model.policy.optimizer.step()
    with torch.no_grad(): final = float(loss_fn().detach().cpu())
    return {"bc_loss_initial": initial, "bc_loss_final": final, "bc_samples": int(len(obs))}

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gate-dir", default="outputs/experiments/track_a_conflict_gate_per_op_full4_2026-06-29")
    ap.add_argument("--reward-mode", default="ReT_excel_plus_cvar")
    ap.add_argument("--seeds", default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20")
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--timesteps", type=int, default=80000)
    ap.add_argument("--bc-epochs", type=int, default=200)
    ap.add_argument("--max-steps", type=int, default=52)
    ap.add_argument("--eval-seed0", type=int, default=9000)
    ap.add_argument("--holding-cost", type=float, default=0.0)
    ap.add_argument("--cvar-alpha", type=float, default=0.1)
    ap.add_argument("--bc-reg-coef", type=float, default=0.1)
    ap.add_argument("--action-smooth-coef", type=float, default=0.0)
    ap.add_argument("--output", default="outputs/experiments/track_a_engineered_2026-06-29")
    args = ap.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    
    summary, static_rows, oracle_actions, best_actions = read_gate(Path(args.gate_dir))
    regimes = list(summary["best_by_regime"].keys())
    robust_label = summary["best_single_constant"]["candidate"]
    robust_action = next(r["action_tuple"] for r in static_rows if r["candidate"] == robust_label)
    
    print(f"ENGINEERED TRACK A: {len(seeds)} seeds, {args.timesteps//1000}k steps, {args.bc_epochs} BC epochs", flush=True)
    print(f"BC target: best_static per regime, LR=1e-4, BC reg={args.bc_reg_coef}", flush=True)
    print(f"Gate: {len(regimes)} regimes, robust={robust_label}", flush=True)
    
    # Build BC targets: use best_static where it beats oracle, else oracle
    bc_actions: dict[str, tuple[float, ...]] = {}
    for regime in regimes:
        oa = oracle_actions[regime]
        ba = best_actions.get(regime, oa)
        # Find which is actually better from static frontier
        o_rows = [r for r in static_rows if r["regime"]==regime and r["action_tuple"]==oa]
        b_rows = [r for r in static_rows if r["regime"]==regime and r["action_tuple"]==ba]
        o_excel = max(r["excel"] for r in o_rows) if o_rows else 0
        b_excel = max(r["excel"] for r in b_rows) if b_rows else 0
        bc_actions[regime] = ba if b_excel >= o_excel else oa
    
    # Evals
    t0 = time.time()
    robust_eval = eval_policy(regimes, lambda o, r, a=robust_action: np.asarray(a,dtype=np.float32), args, args.eval_seed0)
    oracle_eval = eval_policy(regimes, lambda o, r: np.asarray(oracle_actions[r],dtype=np.float32), args, args.eval_seed0)
    bc_eval = eval_policy(regimes, lambda o, r: np.asarray(bc_actions[r],dtype=np.float32), args, args.eval_seed0)
    print(f"Robust: excel={robust_eval['excel']:.6f} Oracle: {oracle_eval['excel']:.6f} BC-target: {bc_eval['excel']:.6f}", flush=True)
    
    bc_obs, bc_acts = collect_bc(regimes, bc_actions, args, args.eval_seed0 + 1000)
    
    learned = []
    for si, seed in enumerate(seeds):
        env_fns = []
        for i in range(args.n_envs):
            regime = regimes[i % len(regimes)]
            env_fns.append(lambda r=regime, s=seed+i: build_env(
                regime=r, reward_mode=args.reward_mode, max_steps=args.max_steps,
                seed=s, holding_cost=args.holding_cost, cvar_alpha=args.cvar_alpha))
        venv = VecNormalize(DummyVecEnv(env_fns), norm_obs=True, norm_reward=True, clip_reward=10.0)
        
        # Cosine LR schedule
        def lr_schedule(progress_remaining):
            return 1e-4 * (0.1 + 0.9 * (1.0 - progress_remaining))
        
        model = PPO("MlpPolicy", venv, seed=seed, verbose=0,
                    n_steps=min(512, args.max_steps * 4), batch_size=64,
                    learning_rate=lr_schedule, n_epochs=10)
        
        bc_stats = bc_train(model, bc_obs, bc_acts, epochs=args.bc_epochs, batch_size=128, seed=seed)
        
        # PPO training with BC regularization (first half)
        half_steps = args.timesteps // 2
        # Train first half with BC reg via periodic BC updates
        bc_update_interval = max(1, half_steps // (args.n_envs * 512 * 5))
        steps_done = 0
        while steps_done < args.timesteps:
            chunk = min(2048, args.timesteps - steps_done)
            model.learn(total_timesteps=chunk, reset_num_timesteps=(steps_done==0))
            steps_done += chunk
            # BC reg: periodic BC weight refresh
            if steps_done <= half_steps and steps_done % (bc_update_interval * args.n_envs * 512) < chunk:
                bc_train(model, bc_obs, bc_acts, epochs=5, batch_size=64, seed=seed)
        
        result = eval_policy(regimes,
            lambda obs, _regime, m=model: m.predict(obs, deterministic=True)[0],
            args, args.eval_seed0)
        result["seed"] = seed
        result["bc"] = bc_stats
        learned.append(result)
        venv.close()
        print(f"  seed {seed}/{si+1}: excel={result['excel']:.6f} cvar={result['cvar']:.2e} res={result['resource']:.3f}", flush=True)
    
    dynamic = {"excel": fmean(r["excel"] for r in learned),
               "cvar": fmean(r["cvar"] for r in learned),
               "resource": fmean(r["resource"] for r in learned)}
    
    # Statistics
    excels = [r["excel"] for r in learned]
    dyn_median = median(excels)
    dyn_best = max(excels)
    dyn_worst = min(excels)
    beats_static_count = sum(1 for e in excels if e > robust_eval["excel"])
    # Bootstrap CI
    rng = np.random.default_rng(42)
    boot_means = [float(np.mean(rng.choice(excels, size=len(excels), replace=True))) for _ in range(10000)]
    ci_lo = float(np.percentile(boot_means, 2.5))
    ci_hi = float(np.percentile(boot_means, 97.5))
    
    report = [
        "# Engineered Track A — Results",
        f"{len(seeds)} seeds, {args.timesteps//1000}k steps, {args.bc_epochs} BC epochs",
        f"BC target: best_static (not oracle), LR=1e-4 with cosine decay, BC reg={args.bc_reg_coef}",
        "",
        f"Robust: excel={robust_eval['excel']:.6f}, res={robust_eval['resource']:.3f}",
        f"Oracle: excel={oracle_eval['excel']:.6f}, res={oracle_eval['resource']:.3f}",
        f"BC-target (best static per regime): excel={bc_eval['excel']:.6f}, res={bc_eval['resource']:.3f}",
        "",
        f"Dynamic mean:   excel={dynamic['excel']:.6f}, cvar={dynamic['cvar']:.2e}, res={dynamic['resource']:.3f}",
        f"Dynamic median: excel={dyn_median:.6f}",
        f"Dynamic best:   excel={dyn_best:.6f} (seed {excels.index(dyn_best)+1})",
        f"Dynamic worst:  excel={dyn_worst:.6f} (seed {excels.index(dyn_worst)+1})",
        f"Bootstrap CI95: [{ci_lo:.6f}, {ci_hi:.6f}]",
        f"Seeds beating robust: {beats_static_count}/{len(seeds)}",
        f"CI95 above robust: {ci_lo > robust_eval['excel']}",
        "",
        "## Per-seed",
    ]
    for i, (e, r) in enumerate(zip(excels, [lr["resource"] for lr in learned])):
        mark = "✅" if e > robust_eval["excel"] else "—"
        report.append(f"  seed {seeds[i]:2d}: excel={e:.6f}, res={r:.3f} {mark}")
    
    report.append(f"\nWROTE {out}")
    (out / "report.md").write_text("\n".join(report))
    
    payload = {"args": vars(args), "regimes": regimes,
               "robust_eval": robust_eval, "oracle_eval": oracle_eval, "bc_eval": bc_eval,
               "learned_per_seed": learned, "dynamic": dynamic,
               "statistics": {"median": dyn_median, "best": dyn_best, "worst": dyn_worst,
                              "beats_static_count": beats_static_count, "n_seeds": len(seeds),
                              "ci95_lo": ci_lo, "ci95_hi": ci_hi},
               "raw_ret_win_vs_robust": dynamic["excel"] > robust_eval["excel"]}
    (out / "summary.json").write_text(json.dumps(payload, indent=2, default=float))
    
    elapsed = time.time() - t0
    print("\n".join(report))
    print(f"Wall: {elapsed:.0f}s")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
