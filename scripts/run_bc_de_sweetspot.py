#!/usr/bin/env python3
"""BC+PPO and Direct Optimization for per-op buffer sweet spot.

Two complementary approaches to reach the static frontier sweet spot
(op3=0, op5=0, op9=0.10, S1, excel=0.00263):

1. BC+PPO: behavioral cloning from expert trajectories → PPO fine-tune
2. DE: differential evolution (black-box optimizer) over the 4D per-op space
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path
from statistics import fmean
from typing import Any
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.continuous_its_env import make_per_op_buffer_track_a_env
from supply_chain.episode_metrics import compute_episode_metrics
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from scipy.optimize import differential_evolution


def build_env(seed=None, **overrides):
    p = dict(
        reward_mode="ReT_excel_plus_cvar", observation_version="v6",
        risk_level="current", risk_frequency_multiplier=4.0, risk_impact_multiplier=1.5,
        stochastic_pt=False, max_steps=104, step_size_hours=168.0,
        init_fracs=(0.0, 0.0, 0.10), risk_obs=True,
        holding_cost=0.02, shift_cost=0.0, ret_excel_cvar_alpha=0.2,
    )
    p.update(overrides)
    p.pop("seed", None)
    env = make_per_op_buffer_track_a_env(**p)
    if seed is not None:
        env.reset(seed=int(seed))
    return env


def evaluate_policy(env_factory, act_fn, n_ep: int, seed0: int) -> dict:
    excels, cvar_pool, resources = [], [], []
    for ep in range(n_ep):
        env = env_factory()
        obs, _ = env.reset(seed=seed0 + ep)
        done = truncated = False
        ep_res = []
        while not (done or truncated):
            action = np.asarray(act_fn(obs), dtype=np.float32).reshape(-1)
            obs, _r, done, truncated, info = env.step(action)
            ep_res.append(float(info.get("resource_composite", 0.0)))
        metrics = compute_episode_metrics(env.unwrapped.sim)
        excels.append(float(metrics.get("ret_excel", 0.0)))
        cvar_pool.append(float(metrics.get("service_loss_auc_ration_hours", 0.0)))
        resources.append(float(np.nanmean(ep_res)) if ep_res else 0.0)
        env.close()
    sl = sorted(cvar_pool)
    cvar_idx = max(0, int(round(0.05 * len(sl))) - 1)
    return {
        "excel": float(np.mean(excels)),
        "cvar": float(np.mean(sl[cvar_idx:])),
        "resource": float(np.mean(resources)),
    }


def collect_expert_trajectories(env_factory, n_ep: int, seed0: int, params: tuple):
    """Run the static sweet spot policy and collect (obs, action) pairs."""
    X, Y = [], []
    op3, op5, op9, shift_sig = params
    def expert_act(_obs):
        return np.array([op3, op5, op9, shift_sig], dtype=np.float32)
    for ep in range(n_ep):
        env = env_factory()
        obs, _ = env.reset(seed=seed0 + ep)
        done = truncated = False
        while not (done or truncated):
            action = expert_act(obs)
            X.append(np.asarray(obs, dtype=np.float32).copy())
            Y.append(np.asarray(action, dtype=np.float32).copy())
            obs, _r, done, truncated, info = env.step(action)
        env.close()
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def train_bc_policy(X: np.ndarray, Y: np.ndarray, obs_dim: int, act_dim: int) -> dict:
    """Train a small MLP to imitate expert (obs→action mapping)."""
    import torch, torch.nn as nn, torch.optim as optim
    n = len(X)
    n_train = int(0.8 * n)
    idx = np.random.permutation(n)
    X_train = torch.tensor(X[idx[:n_train]], dtype=torch.float32)
    Y_train = torch.tensor(Y[idx[:n_train]], dtype=torch.float32)
    X_val = torch.tensor(X[idx[n_train:]], dtype=torch.float32)
    Y_val = torch.tensor(Y[idx[n_train:]], dtype=torch.float32)

    model = nn.Sequential(
        nn.Linear(obs_dim, 64), nn.ReLU(),
        nn.Linear(64, 64), nn.ReLU(),
        nn.Linear(64, act_dim), nn.Tanh(),
    )
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    best_loss = float("inf")
    best_state = None
    for epoch in range(200):
        model.train()
        opt.zero_grad()
        pred = model(X_train)
        loss = loss_fn(pred, Y_train)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val), Y_val).item()
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return {
        "state_dict": {k: v.cpu().numpy() for k, v in best_state.items()},
        "obs_dim": obs_dim, "act_dim": act_dim,
        "layers": [64, 64],
        "val_loss": best_loss,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--bc-episodes", type=int, default=30)
    ap.add_argument("--bc-timesteps", type=int, default=20_000)
    ap.add_argument("--de-population", type=int, default=20)
    ap.add_argument("--de-generations", type=int, default=30)
    ap.add_argument("--eval-episodes", type=int, default=6)
    ap.add_argument("--output", default="outputs/experiments/bc_de_sweetspot_2026-06-29")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # Sweet spot params: op3=0, op5=0, op9=0.10, S1 (shift_signal=-1.0)
    sweet_spot = (0.0, 0.0, 0.10, -1.0)

    # ============================
    # 1. BC + PPO
    # ============================
    print("=" * 60)
    print("1. BEHAVIORAL CLONING + PPO REFINE")
    print(f"   Expert: op3={sweet_spot[0]}, op5={sweet_spot[1]}, op9={sweet_spot[2]}, S1")
    print()

    # Collect expert trajectories
    expert_eval = evaluate_policy(lambda: build_env(seed=42), lambda o: np.array(sweet_spot, dtype=np.float32),
                                   args.eval_episodes * 2, 5000)
    print(f"   Expert eval: excel={expert_eval['excel']:.5f} cvar={expert_eval['cvar']:.2e} res={expert_eval['resource']:.3f}")

    # Collect data from multiple seeds
    all_X, all_Y = [], []
    for seed_val in range(args.bc_episodes):
        env = build_env(seed=seed_val)
        obs, _ = env.reset(seed=5000 + seed_val)
        done = truncated = False
        while not (done or truncated):
            action = np.array(sweet_spot, dtype=np.float32)
            all_X.append(np.asarray(obs, dtype=np.float32).copy())
            all_Y.append(action.copy())
            obs, _r, done, truncated, info = env.step(action)
        env.close()
    X = np.array(all_X, dtype=np.float32)
    Y = np.array(all_Y, dtype=np.float32)
    print(f"   Collected {len(X)} (obs,action) pairs from {args.bc_episodes} expert episodes")

    # Train BC policy
    obs_dim = X.shape[1]
    act_dim = Y.shape[1]
    bc_model = train_bc_policy(X, Y, obs_dim, act_dim)
    print(f"   BC trained: val_loss={bc_model['val_loss']:.6f}, act_dim={act_dim}")

    # Evaluate BC policy
    def bc_act_fn(obs_input):
        import torch, torch.nn as nn
        with torch.no_grad():
            x = torch.tensor(np.asarray(obs_input, dtype=np.float32).reshape(1, -1))
            m = nn.Sequential(
                nn.Linear(obs_dim, 64), nn.ReLU(),
                nn.Linear(64, 64), nn.ReLU(),
                nn.Linear(64, act_dim), nn.Tanh(),
            )
            for k, v in bc_model["state_dict"].items():
                m.state_dict()[k].copy_(torch.tensor(v))
            return m(x).numpy().flatten()

    bc_eval = evaluate_policy(lambda: build_env(seed=42), bc_act_fn, args.eval_episodes, 7000)
    print(f"   BC eval:    excel={bc_eval['excel']:.5f} cvar={bc_eval['cvar']:.2e} res={bc_eval['resource']:.3f}")

    # Now PPO fine-tune from BC
    bc_ppo_results = []
    for seed in seeds:
        print(f"   BC+PPO seed={seed}...", end=" ", flush=True)
        t0 = time.time()
        dummy = DummyVecEnv([lambda s=seed: build_env(seed=s)])
        venv = VecNormalize(dummy, norm_obs=True, norm_reward=True, clip_reward=10.0)
        model = PPO("MlpPolicy", venv, seed=seed, verbose=0,
                    n_steps=512, batch_size=64, learning_rate=1e-4, n_epochs=10,
                    policy_kwargs=dict(net_arch=[64, 64]))
        # Pre-load BC weights into the policy network
        import torch as _torch
        bc_weights = bc_model["state_dict"]
        ppo_state = model.policy.state_dict()
        ppo_layers = [k for k in ppo_state if "weight" in k or "bias" in k]
        bc_keys = list(bc_weights.keys())
        for i, pk in enumerate(ppo_layers[:len(bc_keys)]):
            if ppo_state[pk].shape == bc_weights[bc_keys[i]].shape:
                ppo_state[pk] = _torch.tensor(bc_weights[bc_keys[i]])
        model.policy.load_state_dict(ppo_state)
        model.learn(total_timesteps=args.bc_timesteps)
        train_s = time.time() - t0
        r = evaluate_policy(lambda: build_env(seed=42), lambda o: model.predict(o, deterministic=True)[0],
                           args.eval_episodes, 7000 + seed)
        r["seed"] = seed
        r["train_seconds"] = train_s
        bc_ppo_results.append(r)
        venv.close()
        print(f"excel={r['excel']:.5f} cvar={r['cvar']:.2e} res={r['resource']:.3f}")

    bc_ppo_excel = fmean(r["excel"] for r in bc_ppo_results)
    delta_bc = bc_ppo_excel - expert_eval["excel"]
    print(f"   BC+PPO mean: excel={bc_ppo_excel:.5f} (Δ{delta_bc:+.5f} vs expert)")

    # ============================
    # 2. Differential Evolution
    # ============================
    print()
    print("=" * 60)
    print("2. DIFFERENTIAL EVOLUTION (direct optimization)")
    print(f"   Population={args.de_population}, Generations={args.de_generations}")
    print()

    de_eval_count = [0]
    de_best = {"excel": float("-inf")}

    def de_objective(x):
        op3, op5, op9, shift_sig = float(x[0]), float(x[1]), float(x[2]), float(x[3])
        # Map shift_signal to discrete shift
        if shift_sig < -0.33:
            pass  # S1 already
        elif shift_sig < 0.33:
            shift_sig = 0.0  # S2
        else:
            shift_sig = 1.0  # S3
        def act_fn(_obs):
            return np.array([op3, op5, op9, shift_sig], dtype=np.float32)
        r = evaluate_policy(
            lambda: build_env(init_fracs=(op3, op5, op9)),
            act_fn, 3, int(de_eval_count[0] * 100 + 9000),
        )
        de_eval_count[0] += 1
        if r["excel"] > de_best["excel"]:
            de_best.update(r)
            de_best["params"] = [op3, op5, op9, shift_sig]
        if de_eval_count[0] % 20 == 0:
            print(f"   DE eval {de_eval_count[0]}: best excel={de_best['excel']:.5f} params={de_best.get('params','?')}")
        return -r["excel"]  # minimize negative excel

    bounds = [(0.0, 0.3), (0.0, 0.3), (0.0, 0.5), (-1.0, 1.0)]
    t0 = time.time()
    result = differential_evolution(
        de_objective, bounds, maxiter=args.de_generations,
        popsize=args.de_population, seed=42, polish=False,
        disp=False,
    )
    de_time = time.time() - t0

    de_opt = evaluate_policy(
        lambda: build_env(seed=42, init_fracs=(result.x[0], result.x[1], result.x[2])),
        lambda o, rx=result.x: np.array([rx[0], rx[1], rx[2], rx[3]], dtype=np.float32),
        args.eval_episodes * 2, 9500,
    )
    print(f"   DE done ({de_time:.0f}s, {de_eval_count[0]} evals)")
    print(f"   DE optimum: op3={result.x[0]:.3f} op5={result.x[1]:.3f} op9={result.x[2]:.3f} shift_sig={result.x[3]:.3f}")
    print(f"   DE eval:    excel={de_opt['excel']:.5f} cvar={de_opt['cvar']:.2e} res={de_opt['resource']:.3f}")
    delta_de = de_opt["excel"] - expert_eval["excel"]
    print(f"   Δ vs expert: {delta_de:+.5f}")

    # ============================
    # Report
    # ============================
    report = [
        "# BC+PPO & DE Optimization — Reaching the Per-Op Sweet Spot",
        f"Expert sweet spot: op3=0, op5=0, op9=0.10, S1",
        f"Expert eval: excel={expert_eval['excel']:.5f} cvar={expert_eval['cvar']:.2e} res={expert_eval['resource']:.3f}",
        "",
        "## Results",
        "| Method | Excel | CVaR | Resource | Δ vs Expert |",
        "|---:|---:|---:|---:|---:|",
        f"| Expert (static) | {expert_eval['excel']:.5f} | {expert_eval['cvar']:.2e} | {expert_eval['resource']:.3f} | — |",
        f"| BC (imitation only) | {bc_eval['excel']:.5f} | {bc_eval['cvar']:.2e} | {bc_eval['resource']:.3f} | {bc_eval['excel']-expert_eval['excel']:+.5f} |",
        f"| BC+PPO ({len(seeds)} seeds) | {bc_ppo_excel:.5f} | {fmean(r['cvar'] for r in bc_ppo_results):.2e} | {fmean(r['resource'] for r in bc_ppo_results):.3f} | {delta_bc:+.5f} |",
        f"| DE (direct opt) | {de_opt['excel']:.5f} | {de_opt['cvar']:.2e} | {de_opt['resource']:.3f} | {delta_de:+.5f} |",
        "",
        f"**DE optimum params:** op3={result.x[0]:.3f}, op5={result.x[1]:.3f}, op9={result.x[2]:.3f}, shift_sig={result.x[3]:.3f}",
        f"**DE evals:** {de_eval_count[0]} | **DE wall time:** {de_time:.0f}s",
        "",
        "## Verdict",
    ]
    winner = max([
        ("Expert", expert_eval["excel"]),
        ("BC", bc_eval["excel"]),
        ("BC+PPO", bc_ppo_excel),
        ("DE", de_opt["excel"]),
    ], key=lambda x: x[1])
    report.append(f"**Best method: {winner[0]} with excel={winner[1]:.5f}**")
    if de_opt["excel"] > expert_eval["excel"]:
        report.append(f"✅ DE found a BETTER policy than the expert sweet spot (Δ{delta_de:+.5f})")
    if bc_ppo_excel > expert_eval["excel"]:
        report.append(f"✅ BC+PPO improved over the expert (Δ{delta_bc:+.5f})")
    if winner[1] <= expert_eval["excel"]:
        report.append("❌ Neither method beat the expert sweet spot. The sweet spot IS the optimum.")

    (out / "report.md").write_text("\n".join(report))
    (out / "results.json").write_text(json.dumps({
        "expert": expert_eval,
        "bc": bc_eval, "bc_ppo": bc_ppo_results,
        "de_opt": {k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in de_opt.items()},
        "de_params": [float(x) for x in result.x],
        "de_evals": de_eval_count[0],
        "de_time": de_time,
        "winner": winner[0],
    }, indent=2, default=float))

    print(f"\n{'='*60}")
    print(f"WINNER: {winner[0]} excel={winner[1]:.5f}")
    print(f"WROTE {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
