#!/usr/bin/env python3
"""Pipeline-validation gate: can our transfer protocol detect retention where it
PROVABLY exists? (2026-06-24 audit recommendation #4.)

Minimal hidden-regime Markov bandit:
  - hidden regime z in {0,1}, persistence rho (stays with prob rho each step);
  - action a in {0,1}; reward = 1 if a == z else 0;
  - observation = previous (action one-hot, reward) -- z is NOT directly observable,
    but under high rho the last outcome predicts the current z (win-stay-lose-shift).

A trained (retained) policy learns win-stay-lose-shift and earns ~rho; a fresh policy
(theta_0) earns ~0.5. So the head-start retained - frozen MUST be large for high rho and
~0 for rho = 0.5. If our transfer harness does not recover Delta(0.9) >> Delta(0.5) ~ 0,
the harness is broken and no MFSC null is trustworthy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import DQN

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class MarkovBanditEnv(gym.Env):
    """Hidden two-regime bandit with persistence rho."""

    metadata: dict = {}

    def __init__(self, rho: float = 0.9, horizon: int = 30):
        super().__init__()
        self.rho = float(rho)
        self.horizon = int(horizon)
        self.observation_space = spaces.Box(0.0, 1.0, (3,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self._rng = np.random.default_rng()

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.z = int(self._rng.integers(2))
        self.t = 0
        self._last = np.zeros(3, dtype=np.float32)
        return self._last.copy(), {}

    def step(self, action):
        a = int(action)
        r = 1.0 if a == self.z else 0.0
        obs = np.array([1.0 if a == 0 else 0.0, 1.0 if a == 1 else 0.0, r], dtype=np.float32)
        if self._rng.random() > self.rho:   # flip with prob (1 - rho)
            self.z = 1 - self.z
        self.t += 1
        done = self.t >= self.horizon
        return obs.copy(), r, done, False, {}


def make_model(rho, horizon, seed, learning_starts=50):
    env = MarkovBanditEnv(rho=rho, horizon=horizon)
    return DQN("MlpPolicy", env, seed=seed, learning_rate=1e-3, buffer_size=10_000,
               learning_starts=learning_starts, batch_size=32, verbose=0)


def eval_episode(model, rho, horizon, seed) -> float:
    env = MarkovBanditEnv(rho=rho, horizon=horizon)
    obs, _ = env.reset(seed=seed)
    total, done = 0.0, False
    while not done:
        a, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, _ = env.step(int(np.asarray(a).item()))
        total += r
        done = term or trunc
    return total / horizon


def transfer_delta(rho, *, seeds, n_blocks, train_per_block, horizon) -> dict:
    deltas_late = []
    curves = []
    for s in seeds:
        with tempfile.TemporaryDirectory() as tmp:
            init = Path(tmp) / "m.zip"
            make_model(rho, horizon, seed=s).save(init)
            frozen = DQN.load(init)
            retained = DQN.load(init)
            retained.set_env(MarkovBanditEnv(rho=rho, horizon=horizon))
            curve = []
            for k in range(n_blocks):
                es = 70_000 + s * 1000 + k
                rf = eval_episode(frozen, rho, horizon, es)
                rr = eval_episode(retained, rho, horizon, es)
                curve.append(rr - rf)
                retained.learn(total_timesteps=train_per_block, reset_num_timesteps=False,
                               progress_bar=False)
            curves.append(curve)
            deltas_late.append(float(np.mean(curve[n_blocks // 2:])))
    arr = np.array(curves)
    return {
        "rho": rho,
        "delta_late_mean": float(np.mean(deltas_late)),
        "delta_late_sem": float(np.std(deltas_late, ddof=1) / np.sqrt(len(seeds))) if len(seeds) > 1 else float("nan"),
        "curve_first5": [round(float(x), 3) for x in arr.mean(0)[:5]],
        "curve_last5": [round(float(x), 3) for x in arr.mean(0)[-5:]],
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", default="1,2,3,4,5")
    p.add_argument("--n-blocks", type=int, default=30)
    p.add_argument("--train-per-block", type=int, default=200)
    p.add_argument("--horizon", type=int, default=30)
    p.add_argument("--output", type=Path, default=Path("outputs/benchmarks/sanity_markov/sanity.json"))
    a = p.parse_args()
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    res = {str(rho): transfer_delta(rho, seeds=seeds, n_blocks=a.n_blocks,
                                    train_per_block=a.train_per_block, horizon=a.horizon)
           for rho in (0.5, 0.9)}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("\nMARKOV RETENTION SANITY GATE (transfer delta = retained - frozen)")
    for rho in ("0.5", "0.9"):
        r = res[rho]
        print(f"  rho={rho}: Delta_late={r['delta_late_mean']:+.3f} +/-{r['delta_late_sem']:.3f}  "
              f"curve {r['curve_first5']} -> {r['curve_last5']}")
    d9 = res["0.9"]["delta_late_mean"]
    d5 = res["0.5"]["delta_late_mean"]
    ok = d9 > 0.10 and d9 > d5 + 0.08
    print(f"\n  GATE {'PASS' if ok else 'FAIL'}: Delta(0.9)={d9:+.3f} should be >>{0.10} and >> Delta(0.5)={d5:+.3f}")
    print("  If FAIL, the transfer harness cannot detect retention where it provably exists.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
