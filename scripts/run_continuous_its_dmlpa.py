#!/usr/bin/env python3
"""DMLPA (Transformer-over-history) vs MLP on the continuous_its lane with a LIVE forecast.

Lane #19 (Exp 2, Garrido "learn the disruption"): give the agent HISTORY via VecFrameStack and a
Transformer-over-frames extractor (DMLPA), on adaptive_benchmark_v2 where risk_forecast_* is LIVE
(non-zero) — unlike the war/current lane where the forecast was dead (std 0). Compares DMLPA vs a
plain MLP baseline vs the best constant-fraction static, on Excel ReT + CVaR. Tests whether memory
over disruption history adds value once the forecast actually carries signal.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from supply_chain.continuous_its_env import make_continuous_its_track_a_env
from supply_chain.episode_metrics import compute_episode_metrics
from scripts.dmlpa_extractor import DMLPA

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

SHIFT_SIGS = {1: -1.0, 2: 0.0, 3: 1.0}


def build(reward, obs_v, regime, max_steps, init_frac, seed=None):
    env = make_continuous_its_track_a_env(
        reward_mode=reward, observation_version=obs_v, risk_level=regime,
        max_steps=int(max_steps), step_size_hours=168.0, init_frac=init_frac)
    if seed is not None:
        env.reset(seed=int(seed))
    return env


def eval_constant(build_fn, frac, sig, episodes, seed0):
    excels, sl = [], []
    for ep in range(episodes):
        env = build_fn()
        env.reset(seed=seed0 + ep)
        done = trunc = False
        a = np.array([frac, sig], dtype=np.float32)
        while not (done or trunc):
            _, _r, done, trunc, _i = env.step(a)
        m = compute_episode_metrics(env.unwrapped.sim)
        excels.append(float(m.get("ret_excel", np.nan)))
        sl.append(float(m.get("service_loss_auc_ration_hours", np.nan)))
    return float(np.nanmean(excels)), _cvar(sl)


def _cvar(sl):
    s = sorted(x for x in sl if x == x)
    k = max(1, int(round(0.05 * len(s))))
    return float(np.mean(s[-k:])) if s else float("nan")


def eval_framestacked(model, build_fn, factor, episodes, seed0):
    """Manual frame-stack eval (matches VecFrameStack: zeros then newest-last) so we can read
    env.unwrapped.sim at episode end (DummyVecEnv would auto-reset and lose it)."""
    excels, sl, fracs_all = [], [], []
    base = build_fn()
    dim = base.observation_space.shape[0]
    for ep in range(episodes):
        env = build_fn()
        obs, _ = env.reset(seed=seed0 + ep)
        stack = deque([np.zeros(dim, dtype=np.float32)] * (factor - 1) + [obs.astype(np.float32)],
                      maxlen=factor)
        done = trunc = False
        ep_frac = []
        while not (done or trunc):
            stacked = np.concatenate(list(stack)).astype(np.float32)
            a, _ = model.predict(stacked, deterministic=True)
            obs, _r, done, trunc, info = env.step(a)
            stack.append(obs.astype(np.float32))
            ep_frac.append(float(info.get("continuous_its_frac", np.nan)))
        m = compute_episode_metrics(env.unwrapped.sim)
        excels.append(float(m.get("ret_excel", np.nan)))
        sl.append(float(m.get("service_loss_auc_ration_hours", np.nan)))
        fracs_all += ep_frac
    fa = [f for f in fracs_all if f == f]
    return (float(np.nanmean(excels)), _cvar(sl),
            float(np.std(fa)) if fa else float("nan"))


def train_eval(arch, cfg, factor, seeds, timesteps, n_envs, episodes):
    rows = []
    for seed in seeds:
        venv = VecFrameStack(
            DummyVecEnv([lambda s=seed + i: build(**cfg, seed=s) for i in range(n_envs)]),
            n_stack=factor)
        if arch == "dmlpa":
            pk = dict(features_extractor_class=DMLPA, features_extractor_kwargs=dict(factor=factor))
        else:
            pk = None
        model = PPO("MlpPolicy", venv, seed=seed, verbose=0,
                    n_steps=min(1024, cfg["max_steps"] * 4), batch_size=64,
                    learning_rate=3e-4, n_epochs=10, policy_kwargs=pk)
        model.learn(total_timesteps=int(timesteps))
        ex, cv, fstd = eval_framestacked(model, lambda: build(**cfg), factor, episodes, seed * 100 + 9)
        rows.append({"seed": seed, "excel": ex, "cvar95": cv, "frac_std": fstd})
        print(f"  [{arch}] seed{seed} excel={ex:.5f} cvar={cv:.2e} frac_std={fstd:.3f}", flush=True)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reward-mode", default="ReT_excel_delta")
    ap.add_argument("--observation-version", default="v6")
    ap.add_argument("--regime", default="adaptive_benchmark_v2")
    ap.add_argument("--init-frac", type=float, default=1.0)
    ap.add_argument("--factor", type=int, default=8, help="VecFrameStack history length")
    ap.add_argument("--archs", default="mlp,dmlpa")
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--timesteps", type=int, default=40000)
    ap.add_argument("--eval-episodes", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=104)
    ap.add_argument("--output", default="outputs/experiments/continuous_its_dmlpa_2026-06-27")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    archs = [a.strip() for a in args.archs.split(",") if a.strip()]
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    cfg = dict(reward=args.reward_mode, obs_v=args.observation_version, regime=args.regime,
               max_steps=args.max_steps, init_frac=args.init_frac)

    # best constant static
    bc_ex, bc_cv = -1.0, float("inf")
    for f in (0.0, 0.25, 0.5, 0.75, 1.0):
        for sh, sig in SHIFT_SIGS.items():
            ex, cv = eval_constant(lambda: build(**cfg), f, sig, args.eval_episodes, 5000)
            bc_ex = max(bc_ex, ex); bc_cv = min(bc_cv, cv)

    results = {}
    for arch in archs:
        print(f"########## {arch.upper()} (factor={args.factor}, {args.regime}, live forecast) ##########")
        results[arch] = train_eval(arch, cfg, args.factor, seeds, args.timesteps, args.n_envs,
                                   args.eval_episodes)

    summary = {"args": vars(args), "best_const_excel": bc_ex, "best_const_cvar": bc_cv,
               "results": results}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"\n=== DMLPA vs MLP (continuous_its, {args.regime}, live forecast, h{args.max_steps}) ===")
    print(f"best constant: excel={bc_ex:.5f} cvar={bc_cv:.2e}")
    for arch in archs:
        ex = float(np.nanmean([r["excel"] for r in results[arch]]))
        cv = float(np.nanmean([r["cvar95"] for r in results[arch]]))
        fs = float(np.nanmean([r["frac_std"] for r in results[arch]]))
        print(f"  {arch:6}: excel={ex:.5f} ({'WIN' if ex>bc_ex else 'no'}) "
              f"cvar={cv:.2e} ({'WIN' if cv<bc_cv else 'no'}) frac_std={fs:.3f}")
    print(f"WROTE {out}/summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
