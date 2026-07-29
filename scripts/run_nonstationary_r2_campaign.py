#!/usr/bin/env python3
"""Non-stationary R2 campaign: can a per-op policy win with Garrido's SAME variables?

Structural basis (verified): in STATIONARY regimes the optimal buffer is ~constant base-stock, so a
dense static frontier dominates (the dense-CRN audit falsified the stationary Pareto win). But the
optimal Op9 buffer MOVES with R2 (distribution) intensity — non-monotonically (φ2->op9 0.10, φ4->0.15,
φ6->0.05). So under NON-STATIONARY R2 intensity, NO constant (however finely gridded) matches a policy
that reads the realized R2 rate (ewma/recent_R22) and adapts Op9. This is dense-frontier-proof BY
CONSTRUCTION, keeps Garrido's exact variables (per-op buffer x shift), and only edits the environment.

Test: train one per-op PPO across MIXED R2 intensities {φ2,φ4,φ6}; the agent infers intensity from the
realized-risk/hazard obs and sets Op9. Compare its mean ReT across the intensity mix to the BEST SINGLE
CONSTANT per-op static across the same mix (dense Op9 grid). Win = dynamic > best constant (a margin no
constant can reach). Oracle (per-intensity optimal) is reported as the ceiling.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from supply_chain.continuous_its_env import make_per_op_buffer_track_a_env
from supply_chain.episode_metrics import compute_episode_metrics

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

R2 = ["R21", "R22", "R23", "R24"]
SHIFT_SIGS = {1: -1.0, 2: 0.0, 3: 1.0}


def build(phi, reward, max_steps, seed=None):
    env = make_per_op_buffer_track_a_env(
        reward_mode=reward, observation_version="v6", risk_level="current",
        risk_frequency_multiplier=float(phi), risk_impact_multiplier=1.5, stochastic_pt=False,
        max_steps=int(max_steps), step_size_hours=168.0, risk_obs=True, enabled_risks=R2)
    if seed is not None:
        env.reset(seed=int(seed))
    return env


def eval_dynamic(model, phis, reward, max_steps, episodes, seed0):
    """Per-intensity ReT + the Op9 the policy chose (to verify adaptation)."""
    out = {}
    for phi in phis:
        rets, op9s = [], []
        for ep in range(episodes):
            env = build(phi, reward, max_steps)
            obs, _ = env.reset(seed=seed0 + ep)
            done = trunc = False
            ep_op9 = []
            while not (done or trunc):
                a, _ = model.predict(obs, deterministic=True)
                obs, _r, done, trunc, info = env.step(a)
                ep_op9.append(float(info.get("per_op_op9_frac", np.nan)))
            rets.append(float(compute_episode_metrics(env.unwrapped.sim)["ret_excel"]))
            op9s.append(float(np.nanmean(ep_op9)))
        out[phi] = {"ret": float(np.nanmean(rets)), "op9_mean": float(np.nanmean(op9s))}
    return out


def eval_constant(op9, phis, reward, max_steps, episodes, seed0, op3=0.0, op5=0.0, shift=1):
    sig = SHIFT_SIGS[shift]
    per_phi = {}
    for phi in phis:
        rets = []
        for ep in range(episodes):
            env = build(phi, reward, max_steps)
            env.reset(seed=seed0 + ep)
            a = np.array([op3, op5, op9, sig], dtype=np.float32)
            done = trunc = False
            while not (done or trunc):
                _, _r, done, trunc, _i = env.step(a)
            rets.append(float(compute_episode_metrics(env.unwrapped.sim)["ret_excel"]))
        per_phi[phi] = float(np.nanmean(rets))
    return per_phi


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reward-mode", default="ReT_excel_delta")
    ap.add_argument("--phis", default="2,4,6")
    ap.add_argument("--op9-grid", default="0.0,0.05,0.10,0.15,0.20,0.25,0.30,0.50")
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--n-envs", type=int, default=6, help="mixed across phis")
    ap.add_argument("--timesteps", type=int, default=30000)
    ap.add_argument("--eval-episodes", type=int, default=6)
    ap.add_argument("--max-steps", type=int, default=104)
    ap.add_argument("--output", default="outputs/experiments/nonstationary_r2_campaign_2026-06-28")
    args = ap.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    phis = [float(x) for x in args.phis.split(",")]
    op9_grid = [float(x) for x in args.op9_grid.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]

    # dense constant frontier across the intensity mix (best SINGLE constant = static deployment baseline)
    const = {}
    for op9 in op9_grid:
        per_phi = eval_constant(op9, phis, args.reward_mode, args.max_steps, args.eval_episodes, 7000)
        const[op9] = {"per_phi": per_phi, "mean": float(np.mean(list(per_phi.values())))}
    best_const_op9 = max(const, key=lambda o: const[o]["mean"])
    best_const_mean = const[best_const_op9]["mean"]
    oracle_mean = float(np.mean([max(const[o]["per_phi"][phi] for o in op9_grid) for phi in phis]))

    # dynamic: train one policy across MIXED intensities; it must infer intensity from obs and adapt Op9
    dyn_rows = []
    for seed in seeds:
        venv = DummyVecEnv([lambda i=i: build(phis[i % len(phis)], args.reward_mode, args.max_steps,
                                              seed=seed * 10 + i) for i in range(args.n_envs)])
        model = PPO("MlpPolicy", venv, seed=seed, verbose=0, n_steps=min(1024, args.max_steps * 4),
                    batch_size=64, learning_rate=3e-4, n_epochs=10)
        model.learn(total_timesteps=int(args.timesteps))
        per_phi = eval_dynamic(model, phis, args.reward_mode, args.max_steps, args.eval_episodes,
                               seed * 100 + 9)
        mean = float(np.mean([per_phi[phi]["ret"] for phi in phis]))
        dyn_rows.append({"seed": seed, "per_phi": per_phi, "mean": mean})
        print(f"  [seed {seed}] dyn mean ReT={mean:.4f}  op9 by φ: "
              + " ".join(f"φ{phi:.0f}={per_phi[phi]['op9_mean']:.2f}(ReT{per_phi[phi]['ret']:.3f})" for phi in phis),
              flush=True)

    dyn_mean = float(np.mean([r["mean"] for r in dyn_rows]))
    adapts = float(np.mean([np.std([r["per_phi"][phi]["op9_mean"] for phi in phis]) for r in dyn_rows]))
    win = dyn_mean > best_const_mean

    summary = {"args": vars(args), "constant_frontier": const, "best_const_op9": best_const_op9,
               "best_const_mean": best_const_mean, "oracle_mean": oracle_mean,
               "dynamic": dyn_rows, "dynamic_mean": dyn_mean, "op9_adaptation_std": adapts,
               "win_vs_best_constant": bool(win)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    print(f"\n=== NON-STATIONARY R2 CAMPAIGN (per-op, mixed φ{phis}, h{args.max_steps}) ===")
    print(f"best SINGLE constant: op9={best_const_op9} mean ReT={best_const_mean:.4f} "
          f"(per φ: {const[best_const_op9]['per_phi']})")
    print(f"oracle (per-φ optimal constant): mean ReT={oracle_mean:.4f}  (the adapt ceiling)")
    print(f"DYNAMIC (one adaptive policy): mean ReT={dyn_mean:.4f}  op9-adaptation std={adapts:.3f}")
    print(f"headroom captured: {(dyn_mean-best_const_mean)/(oracle_mean-best_const_mean+1e-9)*100:.0f}% of oracle gap")
    print(f"\n=> {'WIN: dynamic beats best constant (no static can adapt to intensity)' if win else 'no win: dynamic <= best constant'}")
    print(f"WROTE {out}/summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
