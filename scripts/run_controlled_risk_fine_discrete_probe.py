#!/usr/bin/env python3
"""Fine-discrete Track-A probe for controlled-risk CF cases.

Motivation: in the continuous_its wrapper, shift is represented as a continuous
signal and mapped by hard thresholds (< -0.33 -> S1, < 0.33 -> S2, else S3).
That is an optimization convenience, not the thesis mechanism. The thesis S
variable is categorical (S1/S2/S3). This probe exposes a fine discrete action:

    action in Discrete(len(fracs) * 3) -> (buffer_frac, shift)

It tests whether the CF20 failure was due to PPO being unable to cross the
continuous shift threshold and discover a low-buffer S1 needle.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from scripts.run_garrido_controlled_risk_probe import (
    SHIFT_SIGS,
    eval_policy,
    pareto_flags,
    parse_csv_floats,
    parse_csv_ints,
    risk_cases,
)
from supply_chain.continuous_its_env import make_continuous_its_track_a_env


SHIFT_BY_INDEX = (1, 2, 3)
SIG_BY_SHIFT = {1: -1.0, 2: 0.0, 3: 1.0}


class FineDiscreteTrackAEnv(gym.Wrapper):
    """Map a fine discrete (buffer, shift) action to continuous_its actions."""

    def __init__(self, env: gym.Env, *, fracs: list[float]) -> None:
        super().__init__(env)
        self.fracs = [float(np.clip(x, 0.0, 1.0)) for x in fracs]
        self.action_space = gym.spaces.Discrete(len(self.fracs) * 3)
        self.observation_space = env.observation_space

    def decode(self, action: int) -> tuple[float, int]:
        ai = int(action)
        if ai < 0 or ai >= self.action_space.n:
            raise ValueError(f"action {ai} outside [0,{self.action_space.n})")
        frac = self.fracs[ai // 3]
        shift = SHIFT_BY_INDEX[ai % 3]
        return frac, shift

    def step(self, action: Any):
        frac, shift = self.decode(int(action))
        obs, reward, terminated, truncated, info = self.env.step(
            np.asarray([frac, SIG_BY_SHIFT[shift]], dtype=np.float32)
        )
        info = dict(info)
        info["fine_discrete_action"] = int(action)
        info["fine_discrete_frac"] = float(frac)
        info["fine_discrete_shift"] = int(shift)
        return obs, reward, terminated, truncated, info


def build_env(args: argparse.Namespace, case, fracs: list[float], seed: int | None = None):
    env = make_continuous_its_track_a_env(
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
    )
    wrapped = FineDiscreteTrackAEnv(env, fracs=fracs)
    if seed is not None:
        wrapped.reset(seed=int(seed))
    return wrapped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="CF20")
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
    ap.add_argument("--seeds", default="1")
    ap.add_argument("--n-envs", type=int, default=2)
    ap.add_argument("--timesteps", type=int, default=8000)
    ap.add_argument("--eval-episodes", type=int, default=2)
    ap.add_argument("--eval-seed0", type=int, default=9100)
    ap.add_argument(
        "--init-policy-label",
        default="",
        help="Optional warm-start bias, e.g. f0.075_S1. Sets initial discrete logits toward that action.",
    )
    ap.add_argument("--output", default="outputs/experiments/controlled_risk_fine_discrete_probe")
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    fracs = parse_csv_floats(args.fracs)
    cases = risk_cases(args.cases)
    results = []
    for case in cases:
        print(f"\n=== FINE DISCRETE CASE {case.name}: {','.join(case.enabled_risks)} ===", flush=True)

        def builder(seed: int):
            return build_env(args, case, fracs, seed=seed)

        statics = []
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
                statics.append(r)

        learned = []
        action_counts_total: dict[str, int] = {}
        for seed in parse_csv_ints(args.seeds):
            venv = DummyVecEnv([lambda s=seed + i: builder(s) for i in range(args.n_envs)])
            model = PPO(
                "MlpPolicy",
                venv,
                seed=seed,
                verbose=0,
                n_steps=min(512, max(32, args.max_steps * 4)),
                batch_size=32,
                learning_rate=3e-4,
                n_epochs=5,
            )
            if args.init_policy_label:
                try:
                    target = args.init_policy_label.strip()
                    frac_text, shift_text = target.split("_S", maxsplit=1)
                    target_frac = float(frac_text.removeprefix("f"))
                    target_shift = int(shift_text)
                    frac_idx = min(range(len(fracs)), key=lambda i: abs(fracs[i] - target_frac))
                    target_action = frac_idx * 3 + (target_shift - 1)
                    with torch.no_grad():
                        model.policy.action_net.weight.zero_()
                        model.policy.action_net.bias.fill_(-5.0)
                        model.policy.action_net.bias[target_action] = 5.0
                    print(f"warm-started policy logits toward {target} (action {target_action})")
                except Exception as exc:
                    raise ValueError(f"Invalid --init-policy-label={args.init_policy_label!r}") from exc
            model.learn(total_timesteps=int(args.timesteps))
            action_counts: dict[str, int] = {}

            def act(obs, m=model):
                action = int(m.predict(obs, deterministic=True)[0])
                frac = fracs[action // 3]
                shift = SHIFT_BY_INDEX[action % 3]
                key = f"f{frac:g}_S{shift}"
                action_counts[key] = action_counts.get(key, 0) + 1
                action_counts_total[key] = action_counts_total.get(key, 0) + 1
                return np.asarray([frac, SIG_BY_SHIFT[shift]], dtype=np.float32)

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
                seed0=args.eval_seed0,
            )
            r["seed"] = seed
            r["action_counts"] = action_counts
            learned.append(r)

        dynamic = {
            "excel": float(np.nanmean([r["excel"] for r in learned])),
            "cvar95_service_loss": float(np.nanmean([r["cvar95_service_loss"] for r in learned])),
            "flow_fill": float(np.nanmean([r["flow_fill"] for r in learned])),
            "lost_rate": float(np.nanmean([r["lost_rate"] for r in learned])),
            "resource": float(np.nanmean([r["resource"] for r in learned])),
            "frac_std": float(np.nanmean([r["frac_std"] for r in learned])),
        }
        result = {
            "case": {"name": case.name, "enabled_risks": case.enabled_risks, "risk_overrides": case.risk_overrides},
            "fracs": fracs,
            "static_frontier": statics,
            "learned_per_seed": learned,
            "dynamic_mean": dynamic,
            "action_counts_total": action_counts_total,
            "excel_gate": pareto_flags(dynamic, statics, "excel", higher_better=True),
            "cvar_gate": pareto_flags(dynamic, statics, "cvar95_service_loss", higher_better=False),
        }
        results.append(result)
        best = result["excel_gate"]["best_static_any_resource"]
        print(
            f"dynamic excel={dynamic['excel']:.6f} resource={dynamic['resource']:.3f}; "
            f"best static={best['label']} excel={best['excel']:.6f}; actions={action_counts_total}"
        )

    payload = {"args": vars(args), "cases": results}
    (out / "summary.json").write_text(json.dumps(payload, indent=2, default=float))
    print(f"\nWROTE {out / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
