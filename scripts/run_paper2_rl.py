"""Paper 2 — the definitive RL convertibility test on the strongest-authority maintenance cell.

Phase diagram (run_paper2_rl sweep) established: this lane has real, non-trivial CLAIRVOYANT
headroom (H_PI ~ +2%, CI excludes 0) -- unlike every closed lane -- but two observable
heuristics (worst-condition, predictive-forecast) convert NEGATIVELY. RL over the full
observation->action map is the definitive convertibility test. We run it here, on the cell
with the most physical authority, on calibration tapes, evaluate on SEALED holdout tapes vs
the best static periodic calendar and the clairvoyant oracle. All seeds reported.
"""
from __future__ import annotations
import sys, json, argparse
sys.path.insert(0, ".")
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from supply_chain.maintenance import (materialize_tape, simulate, enumerate_oracle,
    periodic_calendars, worst_condition_policy, forecast_policy, week_step,
    condition_index, _threat_forecast, WIP_UNIT, ACTIONS)

CELL = {"cell_id": "paper2-authority", "sensor_q": 0.90, "pm_efficacy": 0.50,
        "wip_days": 2, "wear_hetero": "high", "r11_level": "increased"}
WEEKS = 8
CAL_SEEDS = list(range(5300001, 5300001 + 160))    # calibration pool (training)
HOLD_SEEDS = list(range(5400001, 5400001 + 120))   # SEALED holdout (evaluation only)


def _obs(tape, w, d_true, wip):
    """Fair, non-privileged observation: a noisy CBM sensor of the TRUE degradation, a noisy
    1-wk threat forecast, wip, and time. No privileged access to exact state or future."""
    ci = condition_index(tape, w, d_true)              # noisy TRUE degradation
    fc = _threat_forecast(tape, w).astype(float)
    wipn = wip / (CELL["wip_days"] * WIP_UNIT)
    return np.concatenate([ci, fc, wipn, [w / WEEKS]]).astype(np.float32)


class MaintenanceEnv(gym.Env):
    def __init__(self, seeds):
        super().__init__()
        self.seeds = list(seeds)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(9,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self._i = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        s = self.seeds[self._i % len(self.seeds)]; self._i += 1
        self.tape = materialize_tape(s, CELL, WEEKS)
        self.w = 0; self.d = np.zeros(3); self.wip = np.zeros(2)
        return _obs(self.tape, 0, self.d, self.wip), {}

    def step(self, action):
        d, wip, wsl, down, _ = week_step(self.tape, self.w, action, self.d, self.wip)
        self.d, self.wip = d, wip
        self.w += 1
        reward = -wsl / 2564.0                         # scale ~ weekly capacity units
        done = self.w >= WEEKS
        obs = (_obs(self.tape, self.w, self.d, self.wip) if not done
               else np.zeros(9, np.float32))
        return obs, float(reward), done, False, {}


def ppo_policy_actions(model, tape):
    """Roll the trained deterministic PPO policy on one tape -> action sequence (fair obs)."""
    d = np.zeros(3); wip = np.zeros(2); acts = []
    for w in range(WEEKS):
        obs = _obs(tape, w, d, wip)
        a, _ = model.predict(obs, deterministic=True); a = int(a)
        acts.append(ACTIONS[a])
        d, wip, _, _, _ = week_step(tape, w, a, d, wip)
    return tuple(acts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--steps", type=int, default=120000)
    args = ap.parse_args()
    from stable_baselines3 import PPO

    cal = [materialize_tape(s, CELL, WEEKS) for s in CAL_SEEDS]
    hold = [materialize_tape(s, CELL, WEEKS) for s in HOLD_SEEDS]
    cals = periodic_calendars(WEEKS)

    # best static periodic calendar chosen IN-SAMPLE on calibration (conservative for H_obs)
    sbc = np.array([[simulate(t, c).service_loss for t in cal] for c in cals])
    best_cal = cals[int(sbc.mean(axis=1).argmin())]
    static_h = np.array([simulate(t, best_cal).service_loss for t in hold])
    oracle_h = np.array([enumerate_oracle(t)[0] for t in hold])
    worst_h = np.array([simulate(t, worst_condition_policy(t)).service_loss for t in hold])
    fcast_h = np.array([simulate(t, forecast_policy(t)).service_loss for t in hold])

    def ci(x):
        r = np.random.default_rng(11); b = [r.choice(x, len(x), True).mean() for _ in range(4000)]
        return [round(float(x.mean()), 1), round(float(np.percentile(b, 2.5)), 1),
                round(float(np.percentile(b, 97.5)), 1)]

    rows = []
    for sd in range(args.seeds):
        env = MaintenanceEnv(CAL_SEEDS)
        model = PPO("MlpPolicy", env, seed=sd, verbose=0, n_steps=2048, batch_size=256,
                    gamma=0.98, ent_coef=0.01, policy_kwargs=dict(net_arch=[64, 64]))
        model.learn(total_timesteps=args.steps)
        ppo_h = np.array([simulate(t, ppo_policy_actions(model, t)).service_loss for t in hold])
        h_obs = static_h - ppo_h                       # positive = PPO beats best static
        rows.append({"seed": sd, "ppo_mean_sl": round(float(ppo_h.mean()), 1),
                     "H_obs_ppo": ci(h_obs), "beats_static": bool(ci(h_obs)[1] > 0)})
        print(f"seed {sd}: PPO sl {ppo_h.mean():.0f}  H_obs {ci(h_obs)}  "
              f"{'WIN' if ci(h_obs)[1] > 0 else 'lose'}")

    out = {
        "cell": CELL, "weeks": WEEKS, "n_holdout": len(hold), "best_static_calendar": list(best_cal),
        "static_mean_sl": round(float(static_h.mean()), 1),
        "oracle_mean_sl": round(float(oracle_h.mean()), 1),
        "H_PI_static_minus_oracle": ci(static_h - oracle_h),
        "H_obs_worst_condition": ci(static_h - worst_h),
        "H_obs_forecast_heuristic": ci(static_h - fcast_h),
        "ppo_seeds": rows,
        "ppo_best_H_obs": max((r["H_obs_ppo"][0] for r in rows), default=None),
        "any_seed_beats_static": any(r["beats_static"] for r in rows),
    }
    import os
    os.makedirs("results/paper2", exist_ok=True)
    with open("results/paper2/rl_convertibility.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nH_PI (clairvoyant ceiling):", out["H_PI_static_minus_oracle"])
    print("any PPO seed beats static:", out["any_seed_beats_static"])
    print("written results/paper2/rl_convertibility.json")


if __name__ == "__main__":
    main()
