#!/usr/bin/env python3
"""CF20 learning-repair sprint.

Tests whether PPO can improve from a good static initialization, whether it
only maintains the optimum, and whether a persistent-policy bandit/CEM baseline
is a better tool for this narrow low-buffer/S1 needle than weekly PPO.

The script deliberately avoids changing the DES base. Extra reward variants are
implemented as train-time wrappers around the existing Excel reward.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from scripts.run_controlled_risk_fine_discrete_probe import (
    FineDiscreteTrackAEnv,
    SHIFT_BY_INDEX,
    SIG_BY_SHIFT,
)
from scripts.run_garrido_controlled_risk_probe import (
    SHIFT_SIGS,
    eval_policy,
    parse_csv_floats,
    risk_cases,
)
from supply_chain.config import INVENTORY_BUFFERS
from supply_chain.continuous_its_env import make_continuous_its_track_a_env


SUPPORTED_REWARDS = (
    "ReT_excel_plus_cvar",
    "ReT_excel_delta_bootstrap",
    "ReT_excel_terminal_shaped",
)


@dataclass(frozen=True)
class ActionSpec:
    action: int
    frac: float
    shift: int

    @property
    def label(self) -> str:
        return f"f{self.frac:g}_S{self.shift}"


class FutureCreditRewardWrapper(gym.Wrapper):
    """Train-time reward wrapper for future-credit sensitivities.

    ReT_excel_delta_bootstrap:
      existing completed-order ΔReT minus a small pending/lost proxy.

    ReT_excel_terminal_shaped:
      existing completed-order ΔReT plus potential-based shaping:
      r' = r + gamma * Phi(s') - Phi(s).
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        mode: str,
        pending_scale: float = 1_000_000.0,
        lost_scale: float = 100.0,
        pbrs_alpha: float = 1.0,
        pbrs_beta: float = 0.5,
        pbrs_eta: float = 0.05,
        pbrs_gamma: float = 0.99,
    ) -> None:
        super().__init__(env)
        self.mode = mode
        self.pending_scale = float(pending_scale)
        self.lost_scale = float(lost_scale)
        self.pbrs_alpha = float(pbrs_alpha)
        self.pbrs_beta = float(pbrs_beta)
        self.pbrs_eta = float(pbrs_eta)
        self.pbrs_gamma = float(pbrs_gamma)
        self._prev_phi = 0.0

    def _sim(self):
        return getattr(self.unwrapped, "sim", None)

    def _state_terms(self) -> dict[str, float]:
        sim = self._sim()
        if sim is None:
            return {"pending_norm": 0.0, "lost_norm": 0.0, "coverage": 0.0}
        pending_norm = float(getattr(sim, "pending_backorder_qty", 0.0) or 0.0) / self.pending_scale
        lost_norm = float(getattr(sim, "total_unattended_orders", 0.0) or 0.0) / self.lost_scale
        try:
            inventory = sim._inventory_detail()
            coverage = float(inventory.get("rations_sb", 0.0)) / max(
                float(INVENTORY_BUFFERS[1344]["op9_rations"]), 1.0
            )
        except Exception:
            coverage = 0.0
        return {
            "pending_norm": float(np.clip(pending_norm, 0.0, 10.0)),
            "lost_norm": float(np.clip(lost_norm, 0.0, 10.0)),
            "coverage": float(np.clip(coverage, 0.0, 1.0)),
        }

    def _phi(self) -> float:
        terms = self._state_terms()
        return (
            -self.pbrs_alpha * terms["pending_norm"]
            - self.pbrs_beta * terms["lost_norm"]
            + self.pbrs_eta * terms["coverage"]
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_phi = self._phi()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        base_reward = float(reward)
        terms = self._state_terms()
        shaped = base_reward
        if self.mode == "ReT_excel_delta_bootstrap":
            shaped = (
                base_reward
                - 0.05 * terms["pending_norm"]
                - 0.02 * terms["lost_norm"]
                + 0.005 * terms["coverage"]
            )
        elif self.mode == "ReT_excel_terminal_shaped":
            phi = self._phi()
            shaped = base_reward + self.pbrs_gamma * phi - self._prev_phi
            self._prev_phi = phi
        info = dict(info)
        info["future_credit_reward_mode"] = self.mode
        info["future_credit_base_reward"] = base_reward
        info["future_credit_shaped_reward"] = float(shaped)
        info["future_credit_terms"] = terms
        return obs, float(shaped), terminated, truncated, info


def action_from_label(label: str, fracs: list[float]) -> int:
    frac_text, shift_text = label.strip().split("_S", maxsplit=1)
    frac = float(frac_text.removeprefix("f"))
    shift = int(shift_text)
    frac_idx = min(range(len(fracs)), key=lambda i: abs(fracs[i] - frac))
    return frac_idx * 3 + (shift - 1)


def action_spec(action: int, fracs: list[float]) -> ActionSpec:
    return ActionSpec(
        action=int(action),
        frac=float(fracs[int(action) // 3]),
        shift=int(SHIFT_BY_INDEX[int(action) % 3]),
    )


def make_continuous_env(args: argparse.Namespace, case, *, train_reward: str, seed: int | None = None):
    if train_reward == "ReT_excel_plus_cvar":
        base_reward = "ReT_excel_plus_cvar"
    else:
        base_reward = "ReT_excel_delta"
    env = make_continuous_its_track_a_env(
        reward_mode=base_reward,
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
    if train_reward in ("ReT_excel_delta_bootstrap", "ReT_excel_terminal_shaped"):
        env = FutureCreditRewardWrapper(env, mode=train_reward)
    if seed is not None:
        env.reset(seed=int(seed))
    return env


def make_fine_env(args, case, fracs: list[float], *, train_reward: str, seed: int | None = None):
    env = make_continuous_env(args, case, train_reward=train_reward, seed=seed)
    return FineDiscreteTrackAEnv(env, fracs=fracs)


def static_frontier(args, case, fracs: list[float]) -> list[dict]:
    rows = []
    for frac in fracs:
        for shift, sig in SHIFT_SIGS.items():
            r = eval_policy(
                build_fn=lambda seed, f=frac, s=sig: make_continuous_env(
                    args, case, train_reward="ReT_excel_plus_cvar", seed=seed
                ),
                act_fn=lambda _obs, f=frac, s=sig: np.asarray([f, s], dtype=np.float32),
                episodes=args.eval_episodes,
                seed0=args.eval_seed0,
            )
            r["label"] = f"f{frac:g}_S{shift}"
            r["frac"] = frac
            r["shift"] = shift
            rows.append(r)
    return rows


def eval_action(args, case, fracs: list[float], action: int) -> dict:
    spec = action_spec(action, fracs)
    return eval_policy(
        build_fn=lambda seed: make_continuous_env(
            args, case, train_reward="ReT_excel_plus_cvar", seed=seed
        ),
        act_fn=lambda _obs, sp=spec: np.asarray([sp.frac, SIG_BY_SHIFT[sp.shift]], dtype=np.float32),
        episodes=args.eval_episodes,
        seed0=args.eval_seed0,
    )


def eval_model(args, case, fracs: list[float], model: PPO) -> dict:
    counts: dict[str, int] = {}

    def act(obs):
        action = int(model.predict(obs, deterministic=True)[0])
        spec = action_spec(action, fracs)
        counts[spec.label] = counts.get(spec.label, 0) + 1
        return np.asarray([spec.frac, SIG_BY_SHIFT[spec.shift]], dtype=np.float32)

    result = eval_policy(
        build_fn=lambda seed: make_continuous_env(
            args, case, train_reward="ReT_excel_plus_cvar", seed=seed
        ),
        act_fn=act,
        episodes=args.eval_episodes,
        seed0=args.eval_seed0,
    )
    result["action_counts"] = counts
    return result


def policy_probs(model: PPO, obs: np.ndarray, fracs: list[float]) -> list[dict]:
    obs_tensor, _ = model.policy.obs_to_tensor(obs)
    with torch.no_grad():
        dist = model.policy.get_distribution(obs_tensor)
        probs = dist.distribution.probs.detach().cpu().numpy().reshape(-1)
    rows = [
        {"action": int(i), "label": action_spec(i, fracs).label, "prob": float(p)}
        for i, p in enumerate(probs)
    ]
    rows.sort(key=lambda x: x["prob"], reverse=True)
    return rows


def train_ppo_arm(
    args,
    case,
    fracs: list[float],
    *,
    train_reward: str,
    init_label: str | None,
) -> dict:
    venv = DummyVecEnv([
        lambda s=args.seed + i: make_fine_env(args, case, fracs, train_reward=train_reward, seed=s)
        for i in range(args.n_envs)
    ])
    model = PPO(
        "MlpPolicy",
        venv,
        seed=args.seed,
        verbose=0,
        n_steps=min(512, max(32, args.max_steps * 4)),
        batch_size=32,
        learning_rate=args.learning_rate,
        n_epochs=5,
    )
    target_prob_before = None
    if init_label:
        target_action = action_from_label(init_label, fracs)
        with torch.no_grad():
            model.policy.action_net.weight.zero_()
            model.policy.action_net.bias.fill_(-5.0)
            model.policy.action_net.bias[target_action] = 5.0
        target_prob_before = 1.0
    obs_probe, _ = make_fine_env(
        args, case, fracs, train_reward=train_reward, seed=args.eval_seed0
    ).reset(seed=args.eval_seed0)
    probs_before = policy_probs(model, obs_probe, fracs)[:10]
    model.learn(total_timesteps=int(args.timesteps))
    probs_after = policy_probs(model, obs_probe, fracs)[:10]
    result = eval_model(args, case, fracs, model)
    return {
        "arm": f"ppo_{train_reward}_{init_label or 'scratch'}",
        "train_reward": train_reward,
        "init_label": init_label,
        "target_prob_before": target_prob_before,
        "top_probs_before": probs_before,
        "top_probs_after": probs_after,
        "eval": result,
    }


def run_cem(args, case, fracs: list[float]) -> dict:
    rng = random.Random(args.seed)
    n_actions = len(fracs) * 3
    probs = np.full(n_actions, 1.0 / n_actions, dtype=float)
    history = []
    best: dict[str, Any] | None = None
    for iteration in range(args.cem_iters):
        sampled = [rng.choices(range(n_actions), weights=probs, k=1)[0] for _ in range(args.cem_samples)]
        scored = []
        for action in sampled:
            score = eval_action(args, case, fracs, int(action))
            spec = action_spec(action, fracs)
            row = {
                "action": int(action),
                "label": spec.label,
                "excel": float(score["excel"]),
                "cvar95_service_loss": float(score["cvar95_service_loss"]),
                "resource": float(score["resource"]),
            }
            scored.append(row)
            if best is None or row["excel"] > best["excel"]:
                best = dict(row)
        scored.sort(key=lambda r: r["excel"], reverse=True)
        elite = scored[: max(1, int(math.ceil(args.cem_elite_frac * len(scored))))]
        probs = np.full(n_actions, args.cem_smoothing / n_actions, dtype=float)
        for row in elite:
            probs[int(row["action"])] += (1.0 - args.cem_smoothing) / len(elite)
        history.append(
            {
                "iteration": iteration,
                "best": scored[0],
                "elite_labels": [r["label"] for r in elite],
                "prob_best_known_f0.075_S1": float(probs[action_from_label("f0.075_S1", fracs)]),
            }
        )
    assert best is not None
    return {"arm": "cem_persistent_policy", "best": best, "history": history}


def apply_gate(args, case, fracs: list[float], target_label: str) -> dict:
    action = action_from_label(target_label, fracs)
    env = make_fine_env(args, case, fracs, train_reward="ReT_excel_plus_cvar", seed=args.eval_seed0)
    _obs, _info = env.reset(seed=args.eval_seed0)
    _obs, reward, _done, _trunc, info = env.step(action)
    spec = action_spec(action, fracs)
    env.close()
    return {
        "target_label": target_label,
        "action": int(action),
        "decoded_frac": spec.frac,
        "decoded_shift": spec.shift,
        "info_frac": float(info.get("continuous_its_frac", np.nan)),
        "info_shift": int(info.get("continuous_its_shift", -1)),
        "resource": float(info.get("resource_composite", np.nan)),
        "step_reward": float(reward),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="CF20")
    ap.add_argument("--fracs", default="0,0.01,0.025,0.05,0.075,0.10,0.125,0.15,0.20,0.25")
    ap.add_argument("--rewards", default="ReT_excel_plus_cvar,ReT_excel_delta_bootstrap,ReT_excel_terminal_shaped")
    ap.add_argument("--init-labels", default=",f0.075_S1,f0.05_S1")
    ap.add_argument("--cvar-alpha", type=float, default=0.2)
    ap.add_argument("--observation-version", default="v6")
    ap.add_argument("--risk-level", default="current")
    ap.add_argument("--phi", type=float, default=1.0)
    ap.add_argument("--psi", type=float, default=1.0)
    ap.add_argument("--holding-cost", type=float, default=0.0)
    ap.add_argument("--shift-cost", type=float, default=0.001)
    ap.add_argument("--max-steps", type=int, default=52)
    ap.add_argument("--step-size-hours", type=float, default=168.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n-envs", type=int, default=2)
    ap.add_argument("--timesteps", type=int, default=4000)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--eval-episodes", type=int, default=2)
    ap.add_argument("--eval-seed0", type=int, default=9100)
    ap.add_argument("--cem-iters", type=int, default=3)
    ap.add_argument("--cem-samples", type=int, default=6)
    ap.add_argument("--cem-elite-frac", type=float, default=0.33)
    ap.add_argument("--cem-smoothing", type=float, default=0.10)
    ap.add_argument("--skip-ppo", action="store_true")
    ap.add_argument("--output", default="outputs/experiments/cf20_learning_repair_2026-06-29")
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    fracs = parse_csv_floats(args.fracs)
    case = risk_cases(args.case)[0]
    rewards = [r.strip() for r in args.rewards.split(",") if r.strip()]
    for reward in rewards:
        if reward not in SUPPORTED_REWARDS:
            raise ValueError(f"Unsupported reward {reward!r}; expected {SUPPORTED_REWARDS}")
    init_labels = [x.strip() or None for x in args.init_labels.split(",")]

    print("static frontier...", flush=True)
    statics = static_frontier(args, case, fracs)
    best_excel = max(statics, key=lambda r: r["excel"])
    best_tail = min(statics, key=lambda r: r["cvar95_service_loss"])
    target_label = str(best_excel["label"])
    print(f"best_excel={target_label} excel={best_excel['excel']:.6f}")

    arms = []
    if not args.skip_ppo:
        for reward in rewards:
            for init_label in init_labels:
                print(f"training PPO arm reward={reward} init={init_label or 'scratch'}", flush=True)
                arm = train_ppo_arm(args, case, fracs, train_reward=reward, init_label=init_label)
                ev = arm["eval"]
                print(
                    f"  excel={ev['excel']:.6f} resource={ev['resource']:.3f} "
                    f"actions={ev['action_counts']}"
                )
                arms.append(arm)

    print("running CEM persistent-policy baseline...", flush=True)
    cem = run_cem(args, case, fracs)
    print(f"cem_best={cem['best']}")

    def classify(ev: dict) -> str:
        eps = 1e-9
        if ev["excel"] > best_excel["excel"] + eps and ev["resource"] <= best_excel["resource"] + eps:
            return "improves_over_best_static"
        if abs(ev["excel"] - best_excel["excel"]) <= 1e-9:
            return "maintains_best_static"
        if ev["excel"] < best_excel["excel"] - eps:
            return "degrades_below_best_static"
        return "inconclusive"

    decision = []
    for arm in arms:
        decision.append(
            {
                "arm": arm["arm"],
                "classification": classify(arm["eval"]),
                "excel_gap_vs_best_static": float(arm["eval"]["excel"] - best_excel["excel"]),
                "resource_gap_vs_best_static": float(arm["eval"]["resource"] - best_excel["resource"]),
            }
        )

    payload = {
        "args": vars(args),
        "case": {
            "name": case.name,
            "enabled_risks": case.enabled_risks,
            "risk_overrides": case.risk_overrides,
        },
        "apply_gate": apply_gate(args, case, fracs, target_label),
        "static_frontier": statics,
        "best_static_excel": best_excel,
        "best_static_tail": best_tail,
        "ppo_arms": arms,
        "cem": cem,
        "decision": decision,
        "promotion": {
            "any_improves": any(d["classification"] == "improves_over_best_static" for d in decision),
            "any_maintains": any(d["classification"] == "maintains_best_static" for d in decision),
            "recommended_claim": (
                "dynamic_headroom"
                if any(d["classification"] == "improves_over_best_static" for d in decision)
                else "static_optimum_no_dynamic_headroom"
            ),
        },
    }
    (out / "summary.json").write_text(json.dumps(payload, indent=2, default=float))
    print(f"WROTE {out / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
