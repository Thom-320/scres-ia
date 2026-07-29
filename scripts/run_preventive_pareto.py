#!/usr/bin/env python3
"""Preventive continuous_its agent vs static, judged on the RESOURCE-AWARE PARETO frontier.

Fixes the fairness flaw: raw Excel ReT / CVaR let a profligate constant policy sit at S2/S3 + high
buffer EVERY week for free. Here BOTH static and dynamic are charged a per-step resource_composite
(buffer fraction + extra shifts). A preventive dynamic policy that spends only when the hazard is
high should Pareto-dominate: >= resilience at < resource.

Lane: continuous_its + v6 + realized-risk + history-derived HAZARD obs (weeks-since-last, EWMA rate)
+ a balanced HOLDING COST in the reward (so prevention is a real timing decision, not free max-buffer).

Win (resource-aware): dynamic point is not dominated by any static AND dominates >=1 static, i.e.
  no static has (resilience >= dyn AND resource <= dyn)   [strict Pareto win]
Reported on Excel ReT and on CVaR (lower=better), with the static Pareto frontier.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from supply_chain.continuous_its_env import make_continuous_its_track_a_env
from supply_chain.episode_metrics import compute_episode_metrics

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

SHIFT_SIGS = {1: -1.0, 2: 0.0, 3: 1.0}


def build(reward, obs_v, regime, phi, psi, max_steps, init_frac, holding_cost, shift_cost,
          risk_obs=True, cvar_alpha=0.2, step_size_hours=168.0, seed=None):
    env = make_continuous_its_track_a_env(
        reward_mode=reward, observation_version=obs_v, risk_level=regime,
        risk_frequency_multiplier=float(phi), risk_impact_multiplier=float(psi),
        stochastic_pt=False, max_steps=int(max_steps), step_size_hours=float(step_size_hours),
        init_frac=init_frac,
        risk_obs=bool(risk_obs), holding_cost=float(holding_cost), shift_cost=float(shift_cost),
        ret_excel_cvar_alpha=float(cvar_alpha))
    if seed is not None:
        env.reset(seed=int(seed))
    return env


def _cvar(sl):
    s = sorted(x for x in sl if x == x)
    k = max(1, int(round(0.05 * len(s))))
    return float(np.mean(s[-k:])) if s else float("nan")


def eval_pol(build_fn, act_fn, episodes, seed0, *, trace_policy=None):
    excels, sl, res, fr = [], [], [], []
    trace_rows = []
    field_names = []
    for ep in range(episodes):
        env = build_fn()
        field_names = list(getattr(env, "obs_field_names", []))
        obs, _ = env.reset(seed=seed0 + ep)
        done = trunc = False
        ep_res, ep_fr = [], []
        step = 0
        while not (done or trunc):
            obs_before = np.asarray(obs, dtype=np.float32)
            action = np.asarray(act_fn(obs), dtype=np.float32).reshape(-1)
            obs, _r, done, trunc, info = env.step(action)
            ep_res.append(float(info.get("resource_composite", np.nan)))
            ep_fr.append(float(info.get("continuous_its_frac", np.nan)))
            if trace_policy:
                row = {
                    "policy": trace_policy,
                    "episode": ep,
                    "step": step,
                    "action_frac": float(action[0]) if action.size else float("nan"),
                    "action_shift_signal": float(action[1]) if action.size > 1 else float("nan"),
                    "applied_frac": float(info.get("continuous_its_frac", np.nan)),
                    "applied_shift": float(info.get("continuous_its_shift", np.nan)),
                    "resource_composite": float(info.get("resource_composite", np.nan)),
                    "reward": float(_r),
                }
                for idx, value in enumerate(obs_before):
                    name = field_names[idx] if idx < len(field_names) else f"obs_{idx}"
                    row[f"obs.{name}"] = float(value)
                trace_rows.append(row)
            step += 1
        m = compute_episode_metrics(env.unwrapped.sim)
        excels.append(float(m.get("ret_excel", np.nan)))
        sl.append(float(m.get("service_loss_auc_ration_hours", np.nan)))
        res.append(float(np.nanmean(ep_res)))
        fr += [f for f in ep_fr if f == f]
    result = {"excel": float(np.nanmean(excels)), "cvar": _cvar(sl),
              "resource": float(np.nanmean(res)), "frac_std": float(np.std(fr)) if fr else 0.0}
    if trace_policy:
        result["trace_rows"] = trace_rows
        result["field_names"] = field_names
    return result


def action_correlations(trace_rows, *, target="applied_frac", top_k=20):
    vals = np.asarray([float(r.get(target, np.nan)) for r in trace_rows], dtype=float)
    rows = []
    if vals.size < 3 or np.nanstd(vals) <= 1e-12:
        return rows
    for key in sorted(k for k in trace_rows[0] if k.startswith("obs.")):
        xs = np.asarray([float(r.get(key, np.nan)) for r in trace_rows], dtype=float)
        mask = np.isfinite(xs) & np.isfinite(vals)
        if mask.sum() < 3 or np.nanstd(xs[mask]) <= 1e-12:
            continue
        corr = float(np.corrcoef(xs[mask], vals[mask])[0, 1])
        if np.isfinite(corr):
            rows.append({"feature": key[4:], "corr": corr, "abs_corr": abs(corr)})
    rows.sort(key=lambda r: r["abs_corr"], reverse=True)
    return rows[:top_k]


def pareto_win(dyn, statics, res_key, val_key, higher_better):
    """Strict resource-aware Pareto: no static dominates dyn, and dyn dominates >=1 static."""
    def better(a, b):
        return (a > b) if higher_better else (a < b)
    dominated = any(
        (better(s[val_key], dyn[val_key]) or s[val_key] == dyn[val_key])
        and s[res_key] <= dyn[res_key] + 1e-9
        and (better(s[val_key], dyn[val_key]) or s[res_key] < dyn[res_key] - 1e-9)
        for s in statics)
    dominates = any(
        (better(dyn[val_key], s[val_key]) or dyn[val_key] == s[val_key])
        and dyn[res_key] <= s[res_key] + 1e-9
        and (better(dyn[val_key], s[val_key]) or dyn[res_key] < s[res_key] - 1e-9)
        for s in statics)
    # best static achievable at <= dyn resource
    le = [s for s in statics if s[res_key] <= dyn[res_key] + 1e-9]
    best_le = (max(le, key=lambda s: s[val_key]) if higher_better
               else min(le, key=lambda s: s[val_key])) if le else None
    beats_at_equal_resource = best_le is not None and better(dyn[val_key], best_le[val_key])
    return {"pareto_win": bool((not dominated) and dominates),
            "beats_best_static_at_<=resource": bool(beats_at_equal_resource),
            "dominated_by_static": bool(dominated)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reward-mode", default="ReT_excel_delta")
    ap.add_argument("--observation-version", default="v6")
    ap.add_argument("--regime", default="current")
    ap.add_argument("--phi", type=float, default=4.0)
    ap.add_argument("--psi", type=float, default=1.5)
    ap.add_argument("--init-frac", type=float, default=1.0)
    ap.add_argument("--holding-cost", type=float, default=0.002)
    ap.add_argument("--shift-cost", type=float, default=0.001)
    ap.add_argument("--cvar-alpha", type=float, default=0.2)
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--timesteps", type=int, default=40000)
    ap.add_argument("--eval-episodes", type=int, default=8)
    ap.add_argument("--eval-seed0", type=int, default=5000,
                    help="first evaluation seed for static policies and, with --crn-eval, learned policies")
    ap.add_argument("--crn-eval", action="store_true",
                    help="evaluate learned policies on the same held-out seed block used by statics")
    ap.add_argument("--n-fracs", type=int, default=5,
                    help="static buffer grid size; 5 -> 0,.25,.5,.75,1; 21 -> dense .05 grid")
    ap.add_argument("--max-steps", type=int, default=104)
    ap.add_argument("--step-size-hours", type=float, default=168.0,
                    help="decision cadence in hours (168=weekly thesis; 24=daily)")
    ap.add_argument("--output", default="outputs/experiments/preventive_pareto_2026-06-27")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    cfg = dict(reward=args.reward_mode, obs_v=args.observation_version, regime=args.regime,
               phi=args.phi, psi=args.psi, max_steps=args.max_steps, init_frac=args.init_frac,
               holding_cost=args.holding_cost, shift_cost=args.shift_cost,
               cvar_alpha=args.cvar_alpha, step_size_hours=args.step_size_hours)

    # static grid: constant (frac, shift) — CHARGED its resource every week
    statics = []
    fracs = [round(i / max(1, args.n_fracs - 1), 4) for i in range(args.n_fracs)]
    for f in fracs:
        for sh, sig in SHIFT_SIGS.items():
            r = eval_pol(lambda: build(**cfg), lambda o, ff=f, ss=sig: np.array([ff, ss], np.float32),
                         args.eval_episodes, args.eval_seed0)
            r["label"] = f"f{f}_S{sh}"
            statics.append(r)

    # learned preventive policy
    learned = []
    trace_rows = []
    corr_by_seed = []
    for seed in seeds:
        venv = DummyVecEnv([lambda s=seed + i: build(**cfg, seed=s) for i in range(args.n_envs)])
        model = PPO("MlpPolicy", venv, seed=seed, verbose=0, n_steps=min(1024, args.max_steps * 4),
                    batch_size=64, learning_rate=3e-4, n_epochs=10)
        model.learn(total_timesteps=int(args.timesteps))
        eval_seed0 = args.eval_seed0 if args.crn_eval else seed * 100 + 9
        r = eval_pol(lambda: build(**cfg), lambda o: model.predict(o, deterministic=True)[0],
                     args.eval_episodes, eval_seed0, trace_policy=f"learned_seed{seed}")
        seed_trace = r.pop("trace_rows", [])
        r.pop("field_names", None)
        r["seed"] = seed
        corr_by_seed.append({"seed": seed, "top_frac_correlations": action_correlations(seed_trace)})
        trace_rows.extend({"seed": seed, **row} for row in seed_trace)
        learned.append(r)

    dyn = {"excel": float(np.nanmean([x["excel"] for x in learned])),
           "cvar": float(np.nanmean([x["cvar"] for x in learned])),
           "resource": float(np.nanmean([x["resource"] for x in learned])),
           "frac_std": float(np.nanmean([x["frac_std"] for x in learned]))}
    excel_verdict = pareto_win(dyn, statics, "resource", "excel", higher_better=True)
    cvar_verdict = pareto_win(dyn, statics, "resource", "cvar", higher_better=False)

    summary = {"args": vars(args), "static_fracs": fracs, "statics": statics, "learned_per_seed": learned,
               "dynamic": dyn, "excel_pareto": excel_verdict, "cvar_pareto": cvar_verdict,
               "action_correlations": corr_by_seed}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    if trace_rows:
        keys = sorted({k for row in trace_rows for k in row})
        with (out / "action_trace.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(trace_rows)

    print(f"\n=== RESOURCE-AWARE PARETO ({args.reward_mode}, {args.regime} φ{args.phi}/ψ{args.psi}, "
          f"h{args.max_steps}, hold={args.holding_cost}) ===")
    print(f"DYNAMIC: excel={dyn['excel']:.5f} cvar={dyn['cvar']:.2e} resource={dyn['resource']:.3f} "
          f"frac_std={dyn['frac_std']:.3f}")
    print("static frontier (resource | excel | cvar):")
    for s in sorted(statics, key=lambda s: s["resource"]):
        print(f"  {s['label']:10} res={s['resource']:.3f} excel={s['excel']:.5f} cvar={s['cvar']:.2e}")
    print(f"\nExcel  Pareto: {excel_verdict}")
    print(f"CVaR   Pareto: {cvar_verdict}")
    win = excel_verdict["pareto_win"] or cvar_verdict["pareto_win"]
    print(f"\n=> {'RESOURCE-AWARE PARETO WIN' if win else 'no Pareto win (dynamic does not dominate the charged static frontier)'}")
    print(f"WROTE {out}/summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
