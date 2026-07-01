#!/usr/bin/env python3
"""Mini-refinement around the raw ReT candidate (α=0.1, holding_cost=0.005).

Refines the best cell from the α×holding_cost grid sweep with:
  - α ∈ {0.05, 0.10, 0.15, 0.20}
  - holding_cost ∈ {0.003, 0.005, 0.007, 0.010}
  - seeds=1,2
  - timesteps=30000, n_envs=4
  - eval_episodes=8, CRN held-out eval_seed0=9000
  - n_fracs=21 (dense static frontier)
  - continuous_its action contract (same as the sweep)

Win rules:
  - raw_ret_win: dynamic Excel ReT > best static Excel ReT (any resource)
  - pareto_win: dynamic not dominated on (Excel, CVaR, resource)
  - resource_efficient: same/better ReT at lower resource
"""
from __future__ import annotations

import argparse, csv, json, itertools, sys, time
from pathlib import Path
from statistics import fmean
from typing import Any
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.continuous_its_env import make_continuous_its_track_a_env
from supply_chain.episode_metrics import compute_episode_metrics
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

SHIFT_SIGS = {1: -1.0, 2: 0.0, 3: 1.0}


def build_env(**overrides):
    p = dict(
        reward_mode="ReT_excel_plus_cvar",
        observation_version="v6",
        risk_level="current",
        risk_frequency_multiplier=4.0,
        risk_impact_multiplier=1.5,
        stochastic_pt=False,
        max_steps=104,
        step_size_hours=168.0,
        init_frac=0.0,
        risk_obs=True,
        holding_cost=0.005,
        shift_cost=0.0,
        ret_excel_cvar_alpha=0.1,
    )
    p.update(overrides)
    env = make_continuous_its_track_a_env(
        reward_mode=p["reward_mode"],
        observation_version=p["observation_version"],
        risk_level=p["risk_level"],
        risk_frequency_multiplier=float(p["risk_frequency_multiplier"]),
        risk_impact_multiplier=float(p["risk_impact_multiplier"]),
        stochastic_pt=p["stochastic_pt"],
        max_steps=int(p["max_steps"]),
        step_size_hours=float(p["step_size_hours"]),
        init_frac=p["init_frac"],
        risk_obs=p["risk_obs"],
        holding_cost=float(p["holding_cost"]),
        shift_cost=float(p["shift_cost"]),
        ret_excel_cvar_alpha=float(p["ret_excel_cvar_alpha"]),
    )
    if "seed" in p:
        env.reset(seed=int(p["seed"]))
    return env


def evaluate(env_factory, act_fn, n_ep: int, seed0: int) -> dict:
    excels, cvar_pool, resources = [], [], []
    for ep in range(n_ep):
        env = env_factory()
        obs, _ = env.reset(seed=seed0 + ep)
        done = truncated = False
        ep_res = []
        while not (done or truncated):
            action = np.asarray(act_fn(obs), dtype=np.float32).reshape(-1)
            obs, _r, done, truncated, info = env.step(action)
            ep_res.append(float(info.get("resource_composite", 0.0)))
        metrics = compute_episode_metrics(env.unwrapped.sim)
        excels.append(float(metrics.get("ret_excel", 0.0)))
        sl = float(metrics.get("service_loss_auc_ration_hours", 0.0))
        cvar_pool.append(sl)
        resources.append(float(np.nanmean(ep_res)) if ep_res else 0.0)
        env.close()
    sl_sorted = sorted(cvar_pool)
    cvar_idx = max(0, int(round(0.05 * len(sl_sorted))) - 1)
    return {
        "excel": float(np.mean(excels)),
        "cvar": float(np.mean(sl_sorted[cvar_idx:])),
        "resource": float(np.mean(resources)),
        "excel_per_ep": excels,
    }


def compute_static_frontier(n_fracs: int, n_ep: int, seed0: int, base_params: dict):
    fracs = [round(i / max(1, n_fracs - 1), 4) for i in range(n_fracs)]
    statics = []
    for frac in fracs:
        for shift, sig in SHIFT_SIGS.items():
            r = evaluate(
                lambda f=frac, s=sig: build_env(**base_params),
                lambda o, f=frac, s=sig: np.array([f, s], dtype=np.float32),
                n_ep, seed0,
            )
            r["label"] = f"f{frac}_S{shift}"
            r["frac"] = frac
            r["shift"] = shift
            statics.append(r)
    return statics


def win_verdicts(dynamic: dict, statics: list[dict]) -> dict:
    best_excel = max(statics, key=lambda s: s["excel"])
    best_cvar = min(statics, key=lambda s: s["cvar"])
    raw_ret_win = dynamic["excel"] > best_excel["excel"]
    raw_cvar_win = dynamic["cvar"] < best_cvar["cvar"]
    # Pareto: no static dominates dynamic
    dominated = any(
        s["excel"] >= dynamic["excel"] and s["cvar"] <= dynamic["cvar"] and s["resource"] <= dynamic["resource"]
        for s in statics
    )
    # resource efficient: same/better excel at <= resource
    le = [s for s in statics if s["resource"] <= dynamic["resource"] + 0.001]
    best_le = max(le, key=lambda s: s["excel"]) if le else None
    resource_efficient = best_le is not None and dynamic["excel"] > best_le["excel"]
    return {
        "raw_ret_win": raw_ret_win,
        "raw_cvar_win": raw_cvar_win,
        "pareto_win": (not dominated) and raw_ret_win,
        "resource_efficient": resource_efficient,
        "best_static_excel": best_excel["excel"],
        "best_static_label": best_excel["label"],
        "excel_delta": dynamic["excel"] - best_excel["excel"],
        "cvar_delta": dynamic["cvar"] - best_cvar["cvar"],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--timesteps", type=int, default=30_000)
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--eval-episodes", type=int, default=8)
    ap.add_argument("--eval-seed0", type=int, default=9000)
    ap.add_argument("--n-fracs", type=int, default=21)
    ap.add_argument("--max-steps", type=int, default=104)
    ap.add_argument("--output", default="outputs/experiments/refine_ret_candidate_2026-06-29")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    base_params = dict(
        reward_mode="ReT_excel_plus_cvar",
        observation_version="v6",
        risk_level="current",
        risk_frequency_multiplier=4.0,
        risk_impact_multiplier=1.5,
        max_steps=args.max_steps,
    )
    print("Computing dense static frontier (CRN held-out)...", flush=True)
    t0 = time.time()
    statics = compute_static_frontier(args.n_fracs, args.eval_episodes, args.eval_seed0, base_params)
    best_static = max(statics, key=lambda s: s["excel"])
    print(f"  done ({time.time()-t0:.0f}s). Best static: {best_static['label']} excel={best_static['excel']:.5f} cvar={best_static['cvar']:.2e} res={best_static['resource']:.3f}")
    print()

    grid = list(itertools.product(
        [0.05, 0.10, 0.15, 0.20],   # cvar_alpha
        [0.003, 0.005, 0.007, 0.010],  # holding_cost
    ))

    rows = []
    started = time.time()
    for ca, hc in grid:
        for seed in seeds:
            label = f"ca{ca}_hc{hc}_s{seed}"
            run_dir = out / label
            run_dir.mkdir(parents=True, exist_ok=True)
            cell_params = dict(base_params, cvar_alpha=ca, holding_cost=hc, seed=seed)
            print(f"  [{label}] train...", end=" ", flush=True)
            t_train = time.time()
            venv = DummyVecEnv(
                [lambda cp=cell_params, i=i: build_env(**(cp | {"seed": cp["seed"] + i}))
                 for i in range(args.n_envs)]
            )
            model = PPO("MlpPolicy", venv, seed=seed, verbose=0,
                        n_steps=512, batch_size=64, learning_rate=3e-4, n_epochs=10)
            model.learn(total_timesteps=args.timesteps)
            train_s = time.time() - t_train
            print(f"eval...", end=" ", flush=True)
            r = evaluate(
                lambda cp=cell_params: build_env(**cp),
                lambda o: model.predict(o, deterministic=True)[0],
                args.eval_episodes, args.eval_seed0,
            )
            verdicts = win_verdicts(r, statics)
            r.update({k: v for k, v in cell_params.items() if k not in r})
            r["train_seconds"] = train_s
            r.update(verdicts)
            rows.append(r)
            venv.close()
            print(f"excel={r['excel']:.5f} cvar={r['cvar']:.2e} res={r['resource']:.3f} raw_ret={r['raw_ret_win']}")

    elapsed = time.time() - started

    with (out / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Group by cell
    grouped = {}
    for r in rows:
        k = (r["cvar_alpha"], r["holding_cost"])
        grouped.setdefault(k, []).append(r)

    report = [
        "# Raw ReT Candidate Refinement",
        f"Grid: {len(grid)} cells × {len(seeds)} seeds × {args.timesteps} steps = {len(rows)} runs",
        f"Wall: {elapsed:.0f}s | Static frontier: {args.n_fracs} fracs × 3 shifts × {args.eval_episodes} eps, CRN seed0={args.eval_seed0}",
        "",
        f"**Best static (any resource):** {best_static['label']} excel={best_static['excel']:.5f} cvar={best_static['cvar']:.2e} res={best_static['resource']:.3f}",
        "",
        "## Per-Cell Results",
        "| α | hc | Excel mean | CVaR mean | Resource | raw ReT win | raw CVaR win | Pareto | res-eff |",
        "|---:|---:|---:|---:|---:|:---:|:---:|:---:|:---:|",
    ]
    winners = []
    for (ca, hc), grp in sorted(grouped.items()):
        excel_mean = fmean(r["excel"] for r in grp)
        cvar_mean = fmean(r["cvar"] for r in grp)
        res_mean = fmean(r["resource"] for r in grp)
        rw = any(r["raw_ret_win"] for r in grp)
        cw = any(r["raw_cvar_win"] for r in grp)
        pw = any(r["pareto_win"] for r in grp)
        re = any(r["resource_efficient"] for r in grp)
        if rw:
            winners.append((ca, hc, excel_mean, cvar_mean, res_mean))
        report.append(
            f"| {ca} | {hc} | {excel_mean:.5f} | {cvar_mean:.2e} | {res_mean:.3f} "
            f"| {'✅' if rw else '—'} | {'✅' if cw else '—'} | {'✅' if pw else '—'} | {'✅' if re else '—'} |"
        )

    report.extend([
        "",
        f"## Raw ReT Winners: {len(winners)} cells",
    ])
    for ca, hc, ex, cv, rs in sorted(winners, key=lambda x: -x[2]):
        report.append(f"- α={ca}, hc={hc}: excel={ex:.5f} cvar={cv:.2e} res={rs:.3f}")

    report.extend([
        "",
        "## Next Step",
        f"Top 2 cells by raw Excel ReT → promote to confirmatory (3 seeds, 60k steps, n_fracs=31, eval_eps=12).",
        f"Primary gate: dynamic ReT > best static ReT. Secondary: CVaR not catastrophically worse.",
    ])
    (out / "report.md").write_text("\n".join(report))
    (out / "static_frontier.json").write_text(json.dumps({
        "best_static": {"label": best_static["label"], "excel": best_static["excel"], "cvar": best_static["cvar"], "resource": best_static["resource"]},
        "n_fracs": args.n_fracs,
        "eval_seed0": args.eval_seed0,
        "statics": [{k: s[k] for k in ("label", "excel", "cvar", "resource", "frac", "shift")} for s in sorted(statics, key=lambda s: s["resource"])],
    }, indent=2))

    print(f"\nWROTE {out} ({len(rows)} runs, {len(winners)} raw ReT winners)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
