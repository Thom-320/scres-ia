#!/usr/bin/env python3
"""Audit why PPO does not discover the CF20 low-buffer/S1 optimum.

This is not a new training lane. It is a microscope:

- verifies the fine-discrete action is applied to the DES as intended,
- logs PPO action probabilities at checkpoints,
- evaluates deterministic policy at checkpoints,
- compares learned actions with the known static optimum.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from scripts.run_controlled_risk_fine_discrete_probe import (
    FineDiscreteTrackAEnv,
    SHIFT_BY_INDEX,
    SIG_BY_SHIFT,
    build_env,
)
from scripts.run_garrido_controlled_risk_probe import (
    SHIFT_SIGS,
    eval_policy,
    parse_csv_floats,
    risk_cases,
)
from supply_chain.continuous_its_env import make_continuous_its_track_a_env


def label_for(action: int, fracs: list[float]) -> str:
    return f"f{fracs[action // 3]:g}_S{SHIFT_BY_INDEX[action % 3]}"


def apply_check(args, case, fracs: list[float], action: int) -> dict:
    env = build_env(args, case, fracs, seed=args.eval_seed0)
    _obs, _info = env.reset(seed=args.eval_seed0)
    _obs, reward, _done, _trunc, info = env.step(action)
    frac, shift = env.decode(action)
    env.close()
    return {
        "requested_action": int(action),
        "requested_label": label_for(action, fracs),
        "decoded_frac": float(frac),
        "decoded_shift": int(shift),
        "info_fine_discrete_frac": float(info.get("fine_discrete_frac", np.nan)),
        "info_fine_discrete_shift": int(info.get("fine_discrete_shift", -1)),
        "info_continuous_its_frac": float(info.get("continuous_its_frac", np.nan)),
        "info_continuous_its_shift": int(info.get("continuous_its_shift", -1)),
        "info_resource_composite": float(info.get("resource_composite", np.nan)),
        "step_reward": float(reward),
    }


def static_eval(args, case, fracs: list[float]) -> list[dict]:
    rows = []
    for frac in fracs:
        for shift, sig in SHIFT_SIGS.items():
            r = eval_policy(
                build_fn=lambda seed, f=frac, s=sig: make_continuous_its_track_a_env(
                    reward_mode=args.reward_mode,
                    observation_version=args.observation_version,
                    risk_level=args.risk_level,
                    risk_frequency_multiplier=float(args.phi),
                    risk_impact_multiplier=float(args.psi),
                    enabled_risks=case.enabled_risks,
                    risk_overrides=case.risk_overrides,
                    stochastic_pt=False,
                    max_steps=int(args.max_steps),
                    step_size_hours=float(args.step_size_hours),
                    risk_obs=True,
                    holding_cost=float(args.holding_cost),
                    shift_cost=float(args.shift_cost),
                    ret_excel_cvar_alpha=float(args.cvar_alpha),
                ),
                act_fn=lambda _obs, f=frac, s=sig: np.asarray([f, s], dtype=np.float32),
                episodes=args.eval_episodes,
                seed0=args.eval_seed0,
            )
            r["label"] = f"f{frac:g}_S{shift}"
            rows.append(r)
    return rows


def action_probs(model: PPO, obs: np.ndarray, fracs: list[float]) -> list[dict]:
    obs_tensor, _ = model.policy.obs_to_tensor(obs)
    with torch.no_grad():
        dist = model.policy.get_distribution(obs_tensor)
        probs = dist.distribution.probs.detach().cpu().numpy().reshape(-1)
    rows = [
        {"action": int(i), "label": label_for(i, fracs), "prob": float(p)}
        for i, p in enumerate(probs)
    ]
    rows.sort(key=lambda r: r["prob"], reverse=True)
    return rows


def eval_model(args, case, fracs: list[float], model: PPO, *, seed0: int) -> dict:
    counts: dict[str, int] = {}

    def act(obs):
        action = int(model.predict(obs, deterministic=True)[0])
        label = label_for(action, fracs)
        counts[label] = counts.get(label, 0) + 1
        return np.asarray([fracs[action // 3], SIG_BY_SHIFT[SHIFT_BY_INDEX[action % 3]]], dtype=np.float32)

    r = eval_policy(
        build_fn=lambda seed: make_continuous_its_track_a_env(
            reward_mode=args.reward_mode,
            observation_version=args.observation_version,
            risk_level=args.risk_level,
            risk_frequency_multiplier=float(args.phi),
            risk_impact_multiplier=float(args.psi),
            enabled_risks=case.enabled_risks,
            risk_overrides=case.risk_overrides,
            stochastic_pt=False,
            max_steps=int(args.max_steps),
            step_size_hours=float(args.step_size_hours),
            risk_obs=True,
            holding_cost=float(args.holding_cost),
            shift_cost=float(args.shift_cost),
            ret_excel_cvar_alpha=float(args.cvar_alpha),
        ),
        act_fn=act,
        episodes=args.eval_episodes,
        seed0=seed0,
    )
    r["action_counts"] = counts
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="CF20")
    ap.add_argument("--reward-mode", default="ReT_excel_plus_cvar")
    ap.add_argument("--cvar-alpha", type=float, default=0.2)
    ap.add_argument("--observation-version", default="v6")
    ap.add_argument("--risk-level", default="current")
    ap.add_argument("--phi", type=float, default=1.0)
    ap.add_argument("--psi", type=float, default=1.0)
    ap.add_argument("--holding-cost", type=float, default=0.0)
    ap.add_argument("--shift-cost", type=float, default=0.001)
    ap.add_argument("--max-steps", type=int, default=52)
    ap.add_argument("--step-size-hours", type=float, default=168.0)
    ap.add_argument("--fracs", default="0,0.01,0.025,0.05,0.075,0.10,0.125,0.15,0.20,0.25")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n-envs", type=int, default=2)
    ap.add_argument("--checkpoints", default="0,512,1024,2048,4096,8192")
    ap.add_argument("--eval-episodes", type=int, default=2)
    ap.add_argument("--eval-seed0", type=int, default=9100)
    ap.add_argument("--output", default="outputs/diagnostics/ppo_discovery_cf20_2026-06-29")
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    fracs = parse_csv_floats(args.fracs)
    case = risk_cases(args.case)[0]
    checkpoints = [int(x) for x in args.checkpoints.split(",") if x.strip()]
    if checkpoints[0] != 0:
        checkpoints = [0] + checkpoints

    venv = DummyVecEnv([lambda s=args.seed + i: build_env(args, case, fracs, seed=s) for i in range(args.n_envs)])
    model = PPO(
        "MlpPolicy",
        venv,
        seed=args.seed,
        verbose=0,
        n_steps=min(512, max(32, args.max_steps * 4)),
        batch_size=32,
        learning_rate=3e-4,
        n_epochs=5,
    )

    obs_probe, _ = build_env(args, case, fracs, seed=args.eval_seed0).reset(seed=args.eval_seed0)
    static_rows = static_eval(args, case, fracs)
    best_excel = max(static_rows, key=lambda r: r["excel"])
    best_tail = min(static_rows, key=lambda r: r["cvar95_service_loss"])
    target_action = fracs.index(0.075) * 3 + 0 if 0.075 in fracs else 0

    records = []
    prev = 0
    for ckpt in checkpoints:
        if ckpt > prev:
            model.learn(total_timesteps=ckpt - prev, reset_num_timesteps=False)
            prev = ckpt
        probs = action_probs(model, obs_probe, fracs)
        ev = eval_model(args, case, fracs, model, seed0=args.eval_seed0)
        records.append(
            {
                "timesteps": ckpt,
                "top_action_probs": probs[:8],
                "prob_target_f0.075_S1": next((r["prob"] for r in probs if r["label"] == "f0.075_S1"), None),
                "deterministic_eval": ev,
            }
        )
        top = probs[0]
        print(
            f"ckpt={ckpt:5d} top={top['label']} p={top['prob']:.3f} "
            f"p(f0.075_S1)={records[-1]['prob_target_f0.075_S1']:.3f} "
            f"eval_excel={ev['excel']:.6f} actions={ev['action_counts']}"
        )

    payload = {
        "args": vars(args),
        "case": {
            "name": case.name,
            "enabled_risks": case.enabled_risks,
            "risk_overrides": case.risk_overrides,
        },
        "static_best_excel": best_excel,
        "static_best_tail": best_tail,
        "apply_target_action_check": apply_check(args, case, fracs, target_action),
        "checkpoints": records,
    }
    (out / "summary.json").write_text(json.dumps(payload, indent=2, default=float))
    print(f"WROTE {out / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
