#!/usr/bin/env python3
"""Is the preventive-Pareto winner ANTICIPATORY (prepares before risk) or just REACTIVE/efficient?

The resource-aware Pareto win is robust (5-seed + held-out). But "wins at lower resource" alone could
be pure efficiency (draw down when calm) without genuine anticipation. This audit tests the stronger
claim — that the agent LEARNS TO PREPARE FOR DISRUPTIONS — by interrogating the trained winner:

  A. Conditioning: corr(applied_frac, hazard features) — does the buffer rise with `weeks_since_last_R*`
     (overdue) and `ewma_risk_rate`, MORE than with already-realized `backorder_rate` (reactive)?
  B. Lead/lag: corr(frac[t], risk_onset[t+k]) for k in {-3..+3}. k>0 dominating = frac rises BEFORE the
     risk lands (anticipatory). k<0 dominating = frac only rises AFTER (reactive). risk_onset = number
     of risks that overlapped step t (n_active proxy from the realized-risk block).

Winner config: continuous_its + risk_obs/hazard + ReT_excel_delta + resource-charged + war φ4/ψ1.5, h104.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from supply_chain.continuous_its_env import make_continuous_its_track_a_env

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

HAZARD = ["weeks_since_last_R1", "weeks_since_last_R2", "weeks_since_last_R3", "ewma_risk_rate"]
REACTIVE = ["backorder_rate", "fill_rate"]


def build(regime, phi, psi, max_steps, holding_cost, shift_cost, seed=None):
    env = make_continuous_its_track_a_env(
        reward_mode="ReT_excel_delta", observation_version="v6", risk_level=regime,
        risk_frequency_multiplier=float(phi), risk_impact_multiplier=float(psi), stochastic_pt=False,
        max_steps=int(max_steps), step_size_hours=168.0, init_frac=1.0, risk_obs=True,
        holding_cost=float(holding_cost), shift_cost=float(shift_cost))
    if seed is not None:
        env.reset(seed=int(seed))
    return env


def collect(env, model, fields, episodes, seed0):
    """Per-episode aligned series of frac and obs columns (so lead/lag stays within an episode)."""
    idx = {n: i for i, n in enumerate(fields)}
    n_active_i = idx.get("n_active_risks_norm")
    per_ep = []
    FRAC, COLS = [], {n: [] for n in HAZARD + REACTIVE if n in idx}
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        done = trunc = False
        ef, eo = [], []
        while not (done or trunc):
            a, _ = model.predict(obs, deterministic=True)
            ob = np.asarray(obs, dtype=np.float64)
            obs, _r, done, trunc, info = env.step(a)
            fr = float(info.get("continuous_its_frac", np.nan))
            ef.append(fr)
            eo.append(float(ob[n_active_i]) if n_active_i is not None else np.nan)
            FRAC.append(fr)
            for n in COLS:
                COLS[n].append(float(ob[idx[n]]))
        if len(ef) > 7:
            per_ep.append((np.array(ef), np.array(eo)))
    return np.array(FRAC), {n: np.array(v) for n, v in COLS.items()}, per_ep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", default="current")
    ap.add_argument("--phi", type=float, default=4.0)
    ap.add_argument("--psi", type=float, default=1.5)
    ap.add_argument("--max-steps", type=int, default=104)
    ap.add_argument("--holding-cost", type=float, default=0.0)
    ap.add_argument("--shift-cost", type=float, default=0.001)
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--timesteps", type=int, default=40000)
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--eval-episodes", type=int, default=10)
    ap.add_argument("--output", default="outputs/audits/prevention_anticipation_2026-06-27")
    args = ap.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    fields = list(build(args.regime, args.phi, args.psi, args.max_steps,
                        args.holding_cost, args.shift_cost).obs_field_names)

    per_seed = []
    for seed in seeds:
        venv = DummyVecEnv([lambda s=seed + i: build(args.regime, args.phi, args.psi, args.max_steps,
                                                      args.holding_cost, args.shift_cost, seed=s)
                            for i in range(args.n_envs)])
        model = PPO("MlpPolicy", venv, seed=seed, verbose=0, n_steps=min(1024, args.max_steps * 4),
                    batch_size=64, learning_rate=3e-4, n_epochs=10)
        model.learn(total_timesteps=int(args.timesteps))
        FRAC, COLS, per_ep = collect(
            build(args.regime, args.phi, args.psi, args.max_steps, args.holding_cost, args.shift_cost),
            model, fields, args.eval_episodes, seed * 100 + 9)

        def corr(x, y):
            if np.std(x) < 1e-9 or np.std(y) < 1e-9:
                return 0.0
            return float(np.corrcoef(x, y)[0, 1])

        cond = {n: corr(COLS[n], FRAC) for n in COLS}
        # lead/lag: corr(frac[t], risk_onset[t+k])
        leadlag = {}
        for k in (-3, -2, -1, 0, 1, 2, 3):
            xs, ys = [], []
            for fr, on in per_ep:
                if k >= 0:
                    xs.append(fr[:len(fr) - k] if k else fr); ys.append(on[k:] if k else on)
                else:
                    xs.append(fr[-k:]); ys.append(on[:len(on) + k])
            xs = np.concatenate(xs); ys = np.concatenate(ys)
            leadlag[k] = corr(xs, ys)
        per_seed.append({"seed": seed, "conditioning": cond, "leadlag": leadlag,
                         "frac_std": float(np.std(FRAC))})

    def avg(d_key, sub):
        return {k: float(np.mean([ps[d_key][k] for ps in per_seed])) for k in per_seed[0][d_key]}
    cond_avg = avg("conditioning", None); ll_avg = avg("leadlag", None)
    lead = ll_avg[1] + ll_avg[2] + ll_avg[3]; lag = ll_avg[-1] + ll_avg[-2] + ll_avg[-3]
    haz_strength = float(np.mean([abs(cond_avg[n]) for n in HAZARD if n in cond_avg]))
    reac_strength = float(np.mean([abs(cond_avg[n]) for n in REACTIVE if n in cond_avg]))
    anticipatory = lead > lag + 0.03
    hazard_driven = haz_strength >= reac_strength

    summary = {"args": vars(args), "per_seed": per_seed, "conditioning_avg": cond_avg,
               "leadlag_avg": ll_avg, "lead_sum": lead, "lag_sum": lag,
               "hazard_strength": haz_strength, "reactive_strength": reac_strength,
               "anticipatory": bool(anticipatory), "hazard_driven": bool(hazard_driven)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    print(f"\n=== PREVENTION / ANTICIPATION AUDIT (winner: continuous_its+hazard, war φ{args.phi}/ψ{args.psi}, h{args.max_steps}) ===")
    print(f"frac_std={np.mean([ps['frac_std'] for ps in per_seed]):.3f} (adaptive if >0.05)")
    print("[A] conditioning corr(frac, feature):")
    for n in HAZARD + REACTIVE:
        if n in cond_avg:
            print(f"    {n:24} {cond_avg[n]:+.3f}")
    print(f"    -> hazard strength {haz_strength:.3f} vs reactive strength {reac_strength:.3f}  "
          f"({'HAZARD-DRIVEN' if hazard_driven else 'reactive-driven'})")
    print("[B] lead/lag corr(frac[t], risk_onset[t+k])  (k>0 = frac LEADS risk = anticipatory):")
    print("    " + "  ".join(f"k{k:+d}={ll_avg[k]:+.3f}" for k in (-3, -2, -1, 0, 1, 2, 3)))
    print(f"    -> lead(k>0)={lead:+.3f} vs lag(k<0)={lag:+.3f}  ({'ANTICIPATORY' if anticipatory else 'reactive/contemporaneous'})")
    verdict = anticipatory and hazard_driven
    print(f"\n=> {'PREVENTIVE: buffer leads risk AND keys on hazard (learns to prepare)' if verdict else 'EFFICIENT but not clearly anticipatory (draws down when calm; does not lead risk)'}")
    print(f"WROTE {out}/summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
