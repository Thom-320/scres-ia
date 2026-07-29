"""Paper 2b / Program K — the definitive RL test on the perishable-replenishment lane.

Central-cell screen already showed the FIRST observable state-feedback win of the project
(H_obs base-stock = +146 [52,240], eta~0.32) once the unrealistic no-holding-cost assumption is
removed. Here we (1) confirm robustness across lambda / shelf-life / signal quality, (2) decompose
signal-vs-inventory value, and (3) train PPO on a SEALED holdout to see whether a learner captures
(and matches/beats) the interpretable base-stock policy. All seeds reported; no metric tuned to win.
"""
from __future__ import annotations
import sys, json, os, argparse
sys.path.insert(0, ".")
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from supply_chain.perishable import (materialize_tape, simulate, enumerate_oracle,
    constant_and_periodic, basestock_policy, week_step, ORDER_LEVELS, D0)

CELL = {"cell_id": "k-central", "shelf_life": 2, "cap_mult": 2.0, "signal_q": 0.80,
        "surge_mult": 1.6, "lam": 0.5, "persist": 0.7}
WEEKS = 8
CAL = list(range(6300001, 6300001 + 160))
HOLD = list(range(6400001, 6400001 + 120))     # SEALED holdout


def _obs(tape, w, inv):
    L = tape.cell["shelf_life"]
    invn = np.asarray(inv, float) / D0
    sig = tape.signal[w] / D0 if w < tape.weeks else 0.0
    return np.concatenate([invn, [sig, w / WEEKS]]).astype(np.float32)


class PerishEnv(gym.Env):
    def __init__(self, seeds, cell):
        super().__init__()
        self.seeds = list(seeds); self.cell = cell; self._i = 0
        L = cell["shelf_life"]
        self.observation_space = spaces.Box(0.0, 5.0, shape=(L + 2,), dtype=np.float32)
        self.action_space = spaces.Discrete(len(ORDER_LEVELS))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        s = self.seeds[self._i % len(self.seeds)]; self._i += 1
        self.tape = materialize_tape(s, self.cell, WEEKS)
        self.w = 0; self.inv = np.zeros(self.cell["shelf_life"])
        return _obs(self.tape, 0, self.inv), {}

    def step(self, action):
        inv, wsl, ww = week_step(self.tape, self.w, ORDER_LEVELS[int(action)], self.inv)
        self.inv = inv; self.w += 1
        reward = -(wsl + self.cell["lam"] * ww) / D0
        done = self.w >= WEEKS
        obs = _obs(self.tape, self.w, self.inv) if not done else np.zeros(self.cell["shelf_life"] + 2, np.float32)
        return obs, float(reward), done, False, {}


def ppo_actions(model, tape):
    inv = np.zeros(tape.cell["shelf_life"]); acts = []
    for w in range(WEEKS):
        a, _ = model.predict(_obs(tape, w, inv), deterministic=True); a = int(a)
        acts.append(a); inv, _, _ = week_step(tape, w, ORDER_LEVELS[a], inv)
    return tuple(acts)


def basestock_nosignal(tape, S_units):
    """Ablation: order-up-to-S on observed INVENTORY only (no leading signal) -> isolates (s,S) value."""
    L = tape.cell["shelf_life"]; inv = np.zeros(L); acts = []
    for w in range(tape.weeks):
        q = float(np.clip(S_units - inv.sum() / D0, 0.0, ORDER_LEVELS[-1]))
        a = int(np.argmin([abs(q - x) for x in ORDER_LEVELS])); acts.append(a)
        inv, _, _ = week_step(tape, w, ORDER_LEVELS[a], inv)
    return tuple(acts)


def ci(x):
    x = np.asarray(x, float)
    r = np.random.default_rng(9); b = [r.choice(x, len(x), True).mean() for _ in range(4000)]
    return [round(float(x.mean()), 1), round(float(np.percentile(b, 2.5)), 1), round(float(np.percentile(b, 97.5)), 1)]


def best_static(cal_tapes):
    cals = constant_and_periodic(WEEKS)
    m = np.array([[simulate(t, c).J for t in cal_tapes] for c in cals])
    return cals[int(m.mean(axis=1).argmin())]


def best_S(cal_tapes, policy):
    Sg = np.round(np.arange(0.8, 2.6, 0.2), 2)
    m = np.array([[simulate(t, policy(t, S)).J for t in cal_tapes] for S in Sg])
    return float(Sg[int(m.mean(axis=1).argmin())])


def robustness():
    print("== robustness sweep (H_obs base-stock vs best static, held-out) ==")
    import itertools
    for lam, L, q in itertools.product([0.25, 0.5, 1.0], [2, 3], [0.65, 0.80]):
        cell = dict(CELL); cell.update(lam=lam, shelf_life=L, signal_q=q)
        cal = [materialize_tape(s, cell, WEEKS) for s in CAL[:100]]
        hold = [materialize_tape(s, cell, WEEKS) for s in HOLD[:80]]
        bc = best_static(cal); Ssig = best_S(cal, basestock_policy); Sno = best_S(cal, basestock_nosignal)
        st = np.array([simulate(t, bc).J for t in hold])
        bs = np.array([simulate(t, basestock_policy(t, Ssig)).J for t in hold])
        bn = np.array([simulate(t, basestock_nosignal(t, Sno)).J for t in hold])
        h = ci(st - bs); hn = ci(st - bn)
        tag = "WIN" if h[1] > 0 else "-"
        print(f"lam={lam} L={L} q={q}: H_obs(sig)={h} H_obs(no-sig)={hn}  {tag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--steps", type=int, default=150000)
    ap.add_argument("--skip-robust", action="store_true")
    args = ap.parse_args()
    from stable_baselines3 import PPO

    if not args.skip_robust:
        robustness()

    cal = [materialize_tape(s, CELL, WEEKS) for s in CAL]
    hold = [materialize_tape(s, CELL, WEEKS) for s in HOLD]
    bc = best_static(cal); Ssig = best_S(cal, basestock_policy)
    st = np.array([simulate(t, bc).J for t in hold])
    orc = np.array([enumerate_oracle(t)[0] for t in hold])
    bs = np.array([simulate(t, basestock_policy(t, Ssig)).J for t in hold])

    print("\n== PPO on sealed holdout (central cell) ==")
    rows = []
    for sd in range(args.seeds):
        env = PerishEnv(CAL, CELL)
        model = PPO("MlpPolicy", env, seed=sd, verbose=0, n_steps=2048, batch_size=256,
                    gamma=0.97, ent_coef=0.01, policy_kwargs=dict(net_arch=[64, 64]))
        model.learn(total_timesteps=args.steps)
        pj = np.array([simulate(t, ppo_actions(model, t)).J for t in hold])
        h_static = ci(st - pj); h_vs_bs = ci(bs - pj)
        rows.append({"seed": sd, "ppo_J": round(float(pj.mean()), 1),
                     "H_obs_ppo_vs_static": h_static, "ppo_minus_basestock": h_vs_bs,
                     "beats_static": h_static[1] > 0})
        print(f"seed {sd}: PPO J {pj.mean():.0f}  vs static {h_static} {'WIN' if h_static[1]>0 else 'lose'}"
              f"  vs base-stock {h_vs_bs}")

    out = {"cell": CELL, "weeks": WEEKS, "n_holdout": len(hold),
           "best_static": [ORDER_LEVELS[a] for a in bc], "best_basestock_S": Ssig,
           "static_J": round(float(st.mean()), 1), "oracle_J": round(float(orc.mean()), 1),
           "H_PI": ci(st - orc), "H_obs_basestock": ci(st - bs),
           "eta_basestock": round(float((st - bs).mean() / max((st - orc).mean(), 1e-9)), 3),
           "ppo_seeds": rows,
           "ppo_seeds_beating_static": sum(r["beats_static"] for r in rows)}
    os.makedirs("results/paper2b", exist_ok=True)
    with open("results/paper2b/rl_convertibility.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nH_PI:", out["H_PI"], " H_obs(base-stock):", out["H_obs_basestock"], " eta:", out["eta_basestock"])
    print("PPO seeds beating static:", out["ppo_seeds_beating_static"], "/", args.seeds)
    print("written results/paper2b/rl_convertibility.json")


if __name__ == "__main__":
    main()
