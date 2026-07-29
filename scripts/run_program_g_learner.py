#!/usr/bin/env python3
"""Program G G3->G5 — freeze region, fit observable policies, train the learner.

Region = the 12 surge-1.50 cells that passed G2 (uniform mixture, per V1.2 discipline).
G3: fit a depth-3 observable tree + a logistic contextual bandit on the clairvoyant-action
    dataset from TRAIN tapes; freeze the best static (calibration) and the cover heuristic.
G4: evaluate frozen policies on HOLDOUT (1000001+), no tuning.
G5: train MaskablePPO on TRAIN tapes; evaluate rollout/MPC, bandit, tree, cover, PPO,
    best-static and the oracle on VIRGIN tapes (1010001+). PPO-once, no sweep.
Primary contrast: best-observable/learner minus best full-contract static on virgin.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gymnasium as gym
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from supply_chain.program_g import (
    ACTIONS, cover_signal_policy, enumerate_oracle, materialize_tape, mpc_policy,
    oracle_action_dataset, observe, periodic_calendars, rollout_policy, simulate,
)

REGION = [  # 12 surge-1.50 cells from G2
    {"cell_id": f"P{p}_Q{int(q*100)}_L{l}_S150", "signal_q": q, "lead_weeks": l,
     "surge_mult": 1.50, "persistence": p, "r22_weekly_prob": 0.05}
    for p in ("short", "long") for q in (0.65, 0.75, 0.85) for l in (1, 2)
]
WEEKS = 4
ARM = "TRS"


def region_tape(i, base):
    cell = REGION[i % len(REGION)]
    return materialize_tape(base + i, cell, WEEKS, persistent=True)


def boot_ci(x, n=2000, seed=7):
    rng = np.random.default_rng(seed); x = np.asarray(x, float)
    m = [rng.choice(x, len(x), True).mean() for _ in range(n)]
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


class GEnv(gym.Env):
    def __init__(self, tapes):
        self.tapes = tapes; self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(-5, 5, shape=(7,), dtype=np.float32)
        self._i = 0
    def action_masks(self):
        return np.array([True, True, True])
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = self.tapes[self._i % len(self.tapes)]; self._i += 1
        self.inv = np.zeros(2); self.sb = 10000.0; self.w = 0
        return observe(self.inv, self.sb, self.t, 0), {}
    def step(self, a):
        from supply_chain.program_g import _week_step, DEMAND_DAYS
        self.inv, self.sb, unmet = _week_step(self.inv, self.sb, ACTIONS[int(a)],
                                              self.t.demand[self.w], self.t.r22[self.w], True)
        self.w += 1; done = self.w >= self.t.weeks
        obs = (observe(self.inv, self.sb, self.t, self.w) if not done
               else np.zeros(7, dtype=np.float32))
        return obs, float(-unmet / 5000.0), done, False, {}


def eval_policy_fn(tapes, fn):
    return np.array([rollout_policy(t, fn, ARM)[0] for t in tapes])


def main() -> int:
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    import torch.nn as nn

    n = 240
    train = [region_tape(i, 990001) for i in range(n)]
    holdout = [region_tape(i, 1000001) for i in range(n)]
    virgin = [region_tape(i, 1010001) for i in range(n)]

    # best full-contract static frozen on TRAIN (over 120 periodic calendars)
    cals = periodic_calendars(WEEKS)
    cl = np.array([[simulate(t, c, arm=ARM).service_loss for t in train] for c in cals])
    best_cal = cals[int(cl.mean(axis=1).argmin())]

    # G3: fit tree + contextual bandit on clairvoyant-action dataset from TRAIN
    X, y = [], []
    for t in train:
        xs, ys = oracle_action_dataset(t, ARM); X += xs; y += ys
    X = np.array(X); y = np.array(y)
    tree = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, y)
    bandit = LogisticRegression(max_iter=2000).fit(X, y)
    tree_fn = lambda o: int(tree.predict(o.reshape(1, -1))[0])
    bandit_fn = lambda o: int(bandit.predict(o.reshape(1, -1))[0])
    cover_fn = None  # cover/mpc are tape-level closed-loop, evaluated directly

    # G5: train MaskablePPO once on TRAIN
    env = DummyVecEnv([lambda: GEnv(train)])
    ppo = MaskablePPO("MlpPolicy", env, learning_rate=3e-4, n_steps=512, batch_size=64,
                      gamma=0.99, ent_coef=0.01, policy_kwargs={"net_arch": [64, 64],
                      "activation_fn": nn.Tanh}, seed=9701, verbose=0, device="cpu")
    ppo.learn(total_timesteps=60000)
    ppo_fn = lambda o: int(ppo.predict(o, deterministic=True)[0])

    def eval_set(tapes, label):
        static = np.array([simulate(t, best_cal, arm=ARM).service_loss for t in tapes])
        oracle = np.array([enumerate_oracle(t, arm=ARM)[0] for t in tapes])
        cover = np.array([simulate(t, cover_signal_policy(t, ARM), arm=ARM).service_loss for t in tapes])
        mpc = np.array([simulate(t, mpc_policy(t, ARM), arm=ARM).service_loss for t in tapes])
        tree_l = eval_policy_fn(tapes, tree_fn)
        bandit_l = eval_policy_fn(tapes, bandit_fn)
        ppo_l = eval_policy_fn(tapes, ppo_fn)
        H = lambda arr: boot_ci(static - arr)   # >0 => beats the static (lower loss)
        return {
            "label": label,
            "mean_service_loss": {"oracle": float(oracle.mean()), "best_static": float(static.mean()),
                "cover": float(cover.mean()), "mpc": float(mpc.mean()), "tree": float(tree_l.mean()),
                "bandit": float(bandit_l.mean()), "ppo": float(ppo_l.mean())},
            "vs_static_ci95": {"cover": H(cover), "mpc": H(mpc), "tree": H(tree_l),
                "bandit": H(bandit_l), "ppo": H(ppo_l)},
            "eta_vs_oracle": {k: float((static - v).sum() / max((static - oracle).sum(), 1e-9))
                for k, v in [("cover", cover), ("mpc", mpc), ("tree", tree_l),
                             ("bandit", bandit_l), ("ppo", ppo_l)]},
            "ppo_minus_cover_ci95": boot_ci(cover - ppo_l),  # >0 => PPO beats cover heuristic
        }

    hold = eval_set(holdout, "holdout")
    virg = eval_set(virgin, "virgin")

    # verdict
    def beats(d, who):
        lo = d["vs_static_ci95"][who][1]
        return lo > 0
    v_best = max(["cover", "tree", "bandit", "mpc", "ppo"],
                 key=lambda k: virg["eta_vs_oracle"][k])
    ppo_incr = virg["ppo_minus_cover_ci95"][1] > 0
    interp = ("G5_LEARNER_BEATS_STATIC_OOS_NO_NEURAL_INCREMENT" if beats(virg, "cover") and not ppo_incr
              else "G5_PPO_ADDS_NEURAL_INCREMENT_OVER_HEURISTIC" if beats(virg, "ppo") and ppo_incr
              else "G5_NO_OOS_WIN")
    out = {"gate": "PROGRAM_G_G3_G5_LEARNER", "region_cells": len(REGION),
           "frozen_best_calendar": list(best_cal), "n_per_split": n,
           "tree_depth": 3, "ppo": {"algo": "MaskablePPO", "timesteps": 60000, "seed": 9701},
           "holdout": hold, "virgin": virg, "best_observable_on_virgin": v_best,
           "ppo_beats_cover_incrementally": bool(ppo_incr), "interpretation": interp,
           "note": ("Region = 12 surge-1.50 cells (uniform). Service-loss proxy, disclosed. "
                    "Tree/bandit fit on TRAIN clairvoyant actions; static frozen on TRAIN; "
                    "PPO trained once on TRAIN; ALL evaluated on disjoint holdout + virgin.")}
    output = Path("results/program_g/g5"); output.mkdir(parents=True, exist_ok=True)
    (output / "verdict.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"interpretation": interp, "best_observable_on_virgin": v_best,
        "ppo_beats_cover_incrementally": bool(ppo_incr),
        "virgin_vs_static_ci95": virg["vs_static_ci95"],
        "virgin_eta_vs_oracle": virg["eta_vs_oracle"],
        "virgin_mean_service_loss": virg["mean_service_loss"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
