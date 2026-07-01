#!/usr/bin/env python3
"""Closing Track-A test: can PPO improve the dense-static sweet spot?

The prior sweeps showed alpha-only tuning loses because the learned policy spends too
much resource. This runner tests the genuinely untried region:

  * train with an explicit resource-budget penalty
  * warm-start the actor near f0.10_S1
  * evaluate against the dense continuous static frontier under CRN

If a warm-started, budget-constrained dynamic policy cannot improve the f0.10_S1
frontier, Track A is much closer to exhausted.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from scripts.run_preventive_pareto import SHIFT_SIGS, build, eval_pol, pareto_win


class BudgetPenaltyWrapper(gym.Wrapper):
    """Train-time reward penalty for exceeding a per-step resource budget."""

    def __init__(self, env: gym.Env, *, budget: float, penalty: float) -> None:
        super().__init__(env)
        self.budget = float(budget)
        self.penalty = float(penalty)
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.obs_field_names = getattr(env, "obs_field_names", [])

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        resource = float(info.get("resource_composite", 0.0) or 0.0)
        over_budget = max(0.0, resource - self.budget)
        penalty_value = self.penalty * over_budget
        info["resource_budget"] = self.budget
        info["resource_budget_overage"] = over_budget
        info["resource_budget_penalty"] = penalty_value
        return obs, float(reward) - penalty_value, terminated, truncated, info


def _shift_signal(label: int) -> float:
    return float(SHIFT_SIGS[int(label)])


def warm_start_policy(model: PPO, *, frac: float, shift: int, log_std: float) -> None:
    """Initialize PPO's Gaussian mean near a constant continuous action."""

    action = torch.as_tensor([float(frac), _shift_signal(int(shift))], dtype=torch.float32)
    with torch.no_grad():
        model.policy.action_net.weight.zero_()
        model.policy.action_net.bias.copy_(action)
        if hasattr(model.policy, "log_std"):
            model.policy.log_std.fill_(float(log_std))


def build_train_env(args: argparse.Namespace, *, seed: int | None = None):
    env = build(
        reward=args.reward_mode,
        obs_v=args.observation_version,
        regime=args.regime,
        phi=args.phi,
        psi=args.psi,
        max_steps=args.max_steps,
        init_frac=args.init_frac,
        holding_cost=args.holding_cost,
        shift_cost=args.shift_cost,
        risk_obs=True,
        cvar_alpha=args.cvar_alpha,
        step_size_hours=args.step_size_hours,
        seed=seed,
    )
    return BudgetPenaltyWrapper(env, budget=args.resource_budget, penalty=args.budget_penalty)


def build_eval_env(args: argparse.Namespace):
    return build(
        reward=args.reward_mode,
        obs_v=args.observation_version,
        regime=args.regime,
        phi=args.phi,
        psi=args.psi,
        max_steps=args.max_steps,
        init_frac=args.init_frac,
        holding_cost=args.holding_cost,
        shift_cost=args.shift_cost,
        risk_obs=True,
        cvar_alpha=args.cvar_alpha,
        step_size_hours=args.step_size_hours,
    )


def best_static_at_resource(summary: dict, key: str, higher_better: bool) -> dict | None:
    dyn_res = float(summary["dynamic"]["resource"])
    eligible = [s for s in summary["statics"] if float(s["resource"]) <= dyn_res + 1e-9]
    if not eligible:
        return None
    return max(eligible, key=lambda s: float(s[key])) if higher_better else min(
        eligible, key=lambda s: float(s[key])
    )


def run_case(args: argparse.Namespace) -> dict:
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    statics = []
    fracs = [round(i / max(1, args.n_fracs - 1), 4) for i in range(args.n_fracs)]
    for frac in fracs:
        for shift, sig in SHIFT_SIGS.items():
            r = eval_pol(
                lambda: build_eval_env(args),
                lambda obs, ff=frac, ss=sig: np.array([ff, ss], dtype=np.float32),
                args.eval_episodes,
                args.eval_seed0,
            )
            r["label"] = f"f{frac}_S{shift}"
            statics.append(r)

    learned = []
    for seed in args.seed_list:
        venv = DummyVecEnv(
            [lambda s=seed + i: build_train_env(args, seed=s) for i in range(args.n_envs)]
        )
        model = PPO(
            "MlpPolicy",
            venv,
            seed=seed,
            verbose=0,
            n_steps=min(1024, args.max_steps * 4),
            batch_size=64,
            learning_rate=args.learning_rate,
            n_epochs=10,
            ent_coef=args.ent_coef,
        )
        if args.warm_start:
            warm_start_policy(
                model,
                frac=args.warm_start_frac,
                shift=args.warm_start_shift,
                log_std=args.warm_start_log_std,
            )
        model.learn(total_timesteps=int(args.timesteps))
        r = eval_pol(
            lambda: build_eval_env(args),
            lambda obs: model.predict(obs, deterministic=True)[0],
            args.eval_episodes,
            args.eval_seed0 if args.crn_eval else seed * 100 + 9,
            trace_policy=f"budget_warm_seed{seed}",
        )
        r.pop("trace_rows", None)
        r.pop("field_names", None)
        r["seed"] = seed
        learned.append(r)

    dyn = {
        "excel": float(np.nanmean([x["excel"] for x in learned])),
        "cvar": float(np.nanmean([x["cvar"] for x in learned])),
        "resource": float(np.nanmean([x["resource"] for x in learned])),
        "frac_std": float(np.nanmean([x["frac_std"] for x in learned])),
    }
    summary = {
        "args": vars(args),
        "statics": statics,
        "learned_per_seed": learned,
        "dynamic": dyn,
        "excel_pareto": pareto_win(dyn, statics, "resource", "excel", higher_better=True),
        "cvar_pareto": pareto_win(dyn, statics, "resource", "cvar", higher_better=False),
    }
    summary["best_static_at_resource_excel"] = best_static_at_resource(
        summary, "excel", higher_better=True
    )
    summary["best_static_at_resource_cvar"] = best_static_at_resource(
        summary, "cvar", higher_better=False
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(
        f"\n=== BUDGET WARM-START Track A B={args.resource_budget} λ={args.budget_penalty} "
        f"α={args.cvar_alpha} warm={args.warm_start} ==="
    )
    print(
        f"DYNAMIC: excel={dyn['excel']:.6f} cvar={dyn['cvar']:.2e} "
        f"resource={dyn['resource']:.3f} frac_std={dyn['frac_std']:.3f}"
    )
    print(f"Excel Pareto: {summary['excel_pareto']}")
    print(f"CVaR Pareto:  {summary['cvar_pareto']}")
    print(f"WROTE {out / 'summary.json'}")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reward-mode", default="ReT_excel_plus_cvar")
    ap.add_argument("--cvar-alpha", type=float, default=0.2)
    ap.add_argument("--observation-version", default="v6")
    ap.add_argument("--regime", default="current")
    ap.add_argument("--phi", type=float, default=4.0)
    ap.add_argument("--psi", type=float, default=1.5)
    ap.add_argument("--init-frac", type=float, default=1.0)
    ap.add_argument("--holding-cost", type=float, default=0.0)
    ap.add_argument("--shift-cost", type=float, default=0.001)
    ap.add_argument("--resource-budget", type=float, default=0.05)
    ap.add_argument("--budget-penalty", type=float, default=10.0)
    ap.add_argument("--warm-start", action="store_true")
    ap.add_argument("--warm-start-frac", type=float, default=0.10)
    ap.add_argument("--warm-start-shift", type=int, default=1)
    ap.add_argument("--warm-start-log-std", type=float, default=-2.0)
    ap.add_argument("--seeds", default="1")
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--timesteps", type=int, default=20000)
    ap.add_argument("--eval-episodes", type=int, default=4)
    ap.add_argument("--eval-seed0", type=int, default=8600)
    ap.add_argument("--crn-eval", action="store_true")
    ap.add_argument("--n-fracs", type=int, default=21)
    ap.add_argument("--max-steps", type=int, default=104)
    ap.add_argument("--step-size-hours", type=float, default=168.0)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--ent-coef", type=float, default=0.0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    args.seed_list = [int(s) for s in args.seeds.split(",") if s.strip()]
    run_case(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
