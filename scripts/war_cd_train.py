#!/usr/bin/env python3
"""Thin war-CD PPO trainer (does NOT touch Codex's run_garrido2024_direct_rl.py).

Builds an MFSCGymEnvShifts with the war-cell knobs (phi, psi, stochastic_pt,
demand_mean_multiplier) and the CD-family reward (ReT_garrido2024_raw,
ReT_garrido2024_train, ReT_cvar_cd, control_v1, ReT_seq_v1), trains PPO per
seed, and reports the eval under cd_sigmoid_mean + the secondary metrics.

Used to train the 4 CD cases (1-2 faithful, 3-4 war) under the SAME bar
(cd_sigmoid_mean) the static frontier is ranked on.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from supply_chain.episode_metrics import compute_episode_metrics, merge_resource_metrics
from supply_chain.external_env_interface import make_thesis_aligned_training_env
from supply_chain.ret_thesis import compute_order_level_ret_excel_formula

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    SB3 = True
except Exception as _e:  # pragma: no cover
    SB3 = False


def build_env(seed: int, *, reward_mode: str, phi: float, psi: float,
              stochastic_pt: bool, demand_multiplier: float,
              shift_cost: float, kappa_train_frac: float,
              cvar_lambda: float, cvar_alpha: float, max_steps: int):
    env = make_thesis_aligned_training_env(
        reward_mode=reward_mode,
        risk_level="current",  # env evaluates across regimes in metric
        risk_frequency_multiplier=float(phi),
        risk_impact_multiplier=float(psi),
        stochastic_pt=bool(stochastic_pt),
        demand_mean_multiplier=float(demand_multiplier),
        ret_g24_shift_cost=float(shift_cost),
        ret_g24_kappa_train_frac=float(kappa_train_frac),
        cvar_lambda=float(cvar_lambda),
        cvar_alpha=float(cvar_alpha),
        step_size_hours=168.0,
        max_steps=int(max_steps),
    )
    env.reset(seed=int(seed))
    return env


def _eval_one(env, episodes: int, seed: int):
    # Unused baseline (random policy) kept for reference; the real eval uses the
    # trained PPO below.
    return {
        "cd_sigmoid_mean": 0.0,
        "mean_ret_excel_formula": 0.0,
        "flow_fill_rate": 0.0,
        "n_lost_mean": 0.0,
        "service_loss_auc_per_order": 0.0,
    }


def evaluate_policy(env, episodes: int, seed_base: int):
    return _eval_one(env, episodes, seed_base)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phi", type=float, default=1.0)
    ap.add_argument("--psi", type=float, default=1.0)
    ap.add_argument("--stochastic-pt", action="store_true")
    ap.add_argument("--demand-multiplier", type=float, default=1.0)
    ap.add_argument("--shift-cost", type=float, default=1.0)
    ap.add_argument("--kappa-train-frac", type=float, default=1.0)
    ap.add_argument("--cvar-lambda", type=float, default=0.0)
    ap.add_argument("--cvar-alpha", type=float, default=0.05)
    ap.add_argument("--reward-mode", default="ReT_garrido2024_raw",
                    help="e.g. ReT_garrido2024_raw | ReT_garrido2024_train | ReT_cvar_cd | control_v1 | ReT_seq_v1")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--timesteps", type=int, default=20000)
    ap.add_argument("--eval-episodes", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=52)
    ap.add_argument("--output", default="outputs/experiments/war_cd_train")
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    rows = []
    for seed in seeds:
        if not SB3:
            print("stable_baselines3 not available; aborting")
            return 2

        def make_env_fn(s=seed):
            return build_env(
                seed=s, reward_mode=args.reward_mode,
                phi=args.phi, psi=args.psi, stochastic_pt=args.stochastic_pt,
                demand_multiplier=args.demand_multiplier,
                shift_cost=args.shift_cost, kappa_train_frac=args.kappa_train_frac,
                cvar_lambda=args.cvar_lambda, cvar_alpha=args.cvar_alpha,
                max_steps=args.max_steps)

        venv = DummyVecEnv([make_env_fn])
        model = PPO("MlpPolicy", venv, seed=seed, verbose=0,
                    n_steps=min(1024, args.max_steps),
                    batch_size=min(64, args.max_steps),
                    learning_rate=3e-4, n_epochs=10)
        model.learn(total_timesteps=int(args.timesteps))
        eval_env = build_env(seed=seed * 7 + 1, reward_mode=args.reward_mode,
                            phi=args.phi, psi=args.psi, stochastic_pt=args.stochastic_pt,
                            demand_multiplier=args.demand_multiplier,
                            shift_cost=args.shift_cost, kappa_train_frac=args.kappa_train_frac,
                            cvar_lambda=args.cvar_lambda, cvar_alpha=args.cvar_alpha,
                            max_steps=args.max_steps)
        cd_s, ret_excel, flow, lost, sloss = [], [], [], [], []
        for ep in range(args.eval_episodes):
            obs, _ = eval_env.reset(seed=seed * 1000 + ep)
            done = trunc = False
            while not (done or trunc):
                action, _ = model.predict(obs, deterministic=True)
                obs, r, done, trunc, info = eval_env.step(action)
            cd_s.append(float(info.get("ret_garrido2024_sigmoid_step", 0.0)))
            ret_excel.append(float(info.get("mean_ret_excel_formula", info.get("ret_excel_mean", 0.0))))
            flow.append(float(info.get("fill_rate_state_terminal", 0.0)))
            lost.append(int(info.get("n_lost", 0)))
            sloss.append(float(info.get("service_loss_auc_per_order", 0.0)))
        sl_sorted = sorted(sloss)
        k = max(1, int(round(0.05 * len(sl_sorted))))
        cvar95 = float(np.mean(sl_sorted[-k:])) if sl_sorted else 0.0
        row = {
            "seed": seed, "phi": args.phi, "psi": args.psi,
            "stochastic_pt": args.stochastic_pt, "demand_multiplier": args.demand_multiplier,
            "reward_mode": args.reward_mode,
            "cd_sigmoid_mean": float(np.mean(cd_s)),
            "mean_ret_excel_formula": float(np.mean(ret_excel)),
            "flow_fill_rate": float(np.mean(flow)),
            "n_lost_mean": float(np.mean(lost)),
            "service_loss_auc_per_order": float(np.mean(sloss)),
            "service_loss_cvar95": cvar95,
        }
        rows.append(row)
        print(f"seed={seed} cd_sigmoid={row['cd_sigmoid_mean']:.4f} "
              f"excel={row['mean_ret_excel_formula']:.5f} "
              f"flow={row['flow_fill_rate']:.3f} lost={row['n_lost_mean']:.0f} "
              f"cvar95={cvar95:.2f}", flush=True)

    csv_path = out / "summary.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (out / "summary.json").write_text(json.dumps({
        "args": vars(args), "rows": rows,
        "mean_cd_sigmoid": float(np.mean([r["cd_sigmoid_mean"] for r in rows])),
        "std_cd_sigmoid": float(np.std([r["cd_sigmoid_mean"] for r in rows])),
    }, indent=2))
    print(f"\nWROTE {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
