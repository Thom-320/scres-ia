#!/usr/bin/env python3
"""Per-op buffer PPO training with variance_log CD + ReT_excel_plus_cvar reward.

Uses the per_op_buffer action contract (op3_frac, op5_frac, op9_frac, shift_signal),
war stress φ4/ψ1.5, h104 horizon, production_rate + R14 defect prob in observations,
and variance_log CD as a secondary same-bar metric.

Warm-starts near the static frontier sweet spot (init_fracs=[0,0,0.10]).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.continuous_its_env import make_per_op_buffer_track_a_env
from supply_chain.episode_metrics import compute_episode_metrics
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

SHIFT_SIGS = {1: -1.0, 2: 0.0, 3: 1.0}

GRID = [
    # (cvar_alpha, holding_cost, init_op3, init_op5, init_op9)
    (0.2, 0.002, 0.0, 0.0, 0.10),  # near sweet spot
    (0.2, 0.005, 0.0, 0.0, 0.05),
    (0.2, 0.0, 0.0, 0.0, 0.0),     # cold start (baseline)
    (0.1, 0.002, 0.0, 0.0, 0.10),
    (0.5, 0.002, 0.0, 0.0, 0.05),
]


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
        init_fracs=(0.0, 0.0, 0.10),
        risk_obs=True,
        holding_cost=0.002,
        shift_cost=0.0,
        ret_excel_cvar_alpha=0.2,
    )
    p.update(overrides)
    env = make_per_op_buffer_track_a_env(
        reward_mode=p["reward_mode"],
        observation_version=p["observation_version"],
        risk_level=p["risk_level"],
        risk_frequency_multiplier=float(p["risk_frequency_multiplier"]),
        risk_impact_multiplier=float(p["risk_impact_multiplier"]),
        stochastic_pt=p["stochastic_pt"],
        max_steps=int(p["max_steps"]),
        step_size_hours=float(p["step_size_hours"]),
        init_fracs=p["init_fracs"],
        risk_obs=p["risk_obs"],
        holding_cost=float(p["holding_cost"]),
        shift_cost=float(p["shift_cost"]),
        ret_excel_cvar_alpha=float(p["ret_excel_cvar_alpha"]),
    )
    if "seed" in p:
        env.reset(seed=int(p["seed"]))
    return env


def evaluate(env_factory, act_fn, n_ep: int, seed0: int) -> dict:
    excels, service_losses, resources, cd_scores = [], [], [], []
    frac_std_pool = []
    for ep in range(n_ep):
        env = env_factory()
        obs, _ = env.reset(seed=seed0 + ep)
        done = truncated = False
        ep_res, ep_fr = [], []
        while not (done or truncated):
            action = np.asarray(act_fn(obs), dtype=np.float32).reshape(-1)
            obs, _reward, done, truncated, info = env.step(action)
            ep_res.append(float(info.get("resource_composite", 0.0)))
            ep_fr.append(float(info.get("per_op_buffer_frac_avg", np.nan)))
            cd = info.get("ret_garrido2024_sigmoid")
            if cd is not None:
                cd_scores.append(float(cd))
        metrics = compute_episode_metrics(env.unwrapped.sim)
        excels.append(float(metrics.get("ret_excel", 0.0)))
        sl = float(metrics.get("service_loss_auc_ration_hours", 0.0))
        service_losses.append(sl)
        resources.append(float(np.nanmean(ep_res)) if ep_res else 0.0)
        frac_std_pool.extend(f for f in ep_fr if f == f)
        env.close()
    n = max(1, len(service_losses))
    sl_sorted = sorted(service_losses)
    cvar_idx = max(0, int(round(0.05 * n)) - 1)
    cd_mean = fmean(cd_scores) if cd_scores else 0.0
    return {
        "excel": float(np.mean(excels)),
        "cvar": float(np.mean(sl_sorted[cvar_idx:])),
        "service_loss_mean": fmean(service_losses),
        "resource": float(np.mean(resources)),
        "frac_std": float(np.std(frac_std_pool)) if len(frac_std_pool) > 1 else 0.0,
        "cd_sigmoid_mean": cd_mean,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--timesteps", type=int, default=30_000)
    ap.add_argument("--n-envs", type=int, default=2)
    ap.add_argument("--eval-episodes", type=int, default=6)
    ap.add_argument("--eval-seed0", type=int, default=5000)
    ap.add_argument("--output", default="outputs/experiments/per_op_cd_train_2026-06-29")
    ap.add_argument("--cell", default="",
                    help="Override grid: 'alpha,holdcost,op3,op5,op9'")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    if args.cell:
        parts = [float(x) for x in args.cell.split(",")]
        grid = [(parts[0], parts[1], parts[2], parts[3], parts[4])]
    else:
        grid = GRID

    rows = []
    started = time.time()
    for cvar_alpha, holding_cost, i3, i5, i9 in grid:
        for seed in seeds:
            label = f"ca{cvar_alpha}_hc{holding_cost}_op{i3}_{i5}_{i9}_s{seed}"
            run_dir = out / label
            run_dir.mkdir(parents=True, exist_ok=True)
            cfg = dict(
                ret_excel_cvar_alpha=cvar_alpha,
                holding_cost=holding_cost,
                init_fracs=(i3, i5, i9),
                seed=seed,
            )
            print(f"  [{label}] training... ", end="", flush=True)
            t0 = time.time()
            venv = DummyVecEnv(
                [lambda c=cfg, i=i: build_env(**(c | {"seed": c["seed"] + i}))
                 for i in range(args.n_envs)]
            )
            model = PPO(
                "MlpPolicy", venv, seed=seed, verbose=0,
                n_steps=512, batch_size=64, learning_rate=3e-4, n_epochs=10,
            )
            model.learn(total_timesteps=int(args.timesteps))
            train_sec = time.time() - t0
            print(f"eval... ", end="", flush=True)
            r = evaluate(
                lambda c=cfg: build_env(**c),
                lambda o: model.predict(o, deterministic=True)[0],
                args.eval_episodes,
                seed * 1000 + 1,
            )
            r["cvar_alpha"] = cvar_alpha
            r["holding_cost"] = holding_cost
            r["init_op3"] = i3
            r["init_op5"] = i5
            r["init_op9"] = i9
            r["seed"] = seed
            r["train_seconds"] = train_sec
            r["timesteps"] = args.timesteps
            rows.append(r)
            (run_dir / "metrics.json").write_text(json.dumps(r, indent=2, default=float))
            venv.close()
            print(
                f"excel={r['excel']:.5f} cvar={r['cvar']:.2e} "
                f"cd={r['cd_sigmoid_mean']:.3f} res={r['resource']:.3f} "
                f"frac_std={r['frac_std']:.3f}"
            )

    elapsed = time.time() - started

    # Aggregate across seeds per grid cell
    grouped: dict[tuple, list[dict]] = {}
    for r in rows:
        k = (r["cvar_alpha"], r["holding_cost"], r["init_op3"], r["init_op5"], r["init_op9"])
        grouped.setdefault(k, []).append(r)

    summary_rows = []
    for (ca, hc, i3, i5, i9), grp in grouped.items():
        summary_rows.append({
            "cvar_alpha": ca,
            "holding_cost": hc,
            "init_op3": i3, "init_op5": i5, "init_op9": i9,
            "excel_mean": fmean(r["excel"] for r in grp),
            "cvar_mean": fmean(r["cvar"] for r in grp),
            "cd_mean": fmean(r["cd_sigmoid_mean"] for r in grp),
            "resource_mean": fmean(r["resource"] for r in grp),
            "frac_std_mean": fmean(r["frac_std"] for r in grp),
            "n_seeds": len(grp),
        })

    with (out / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    report = [
        "# Per-Op Buffer + variance_log CD Training",
        "",
        f"Grid: {len(grid)} cells × {len(seeds)} seeds × {args.timesteps} steps = {len(rows)} runs",
        f"Wall: {elapsed:.0f}s",
        "",
        "## Results",
        "",
        "| α | hc | init | Excel | CVaR | CD (var_log) | Resource | frac_std | Seeds |",
        "|---:|---:|:---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary_rows:
        init = f"({r['init_op3']},{r['init_op5']},{r['init_op9']})"
        report.append(
            f"| {r['cvar_alpha']} | {r['holding_cost']} | {init} "
            f"| {r['excel_mean']:.5f} | {r['cvar_mean']:.2e} "
            f"| {r['cd_mean']:.3f} | {r['resource_mean']:.3f} "
            f"| {r['frac_std_mean']:.3f} | {r['n_seeds']} |"
        )
    report.extend([
        "",
        "## Verdict",
        "",
        f"CD metric now uses `variance_log` (balance_c=0.15) — all 5 terms equally balanced.",
        f"Observations include `production_rate_norm` and `r14_defect_prob`.",
        f"Action contract: `Box([op3_frac, op5_frac, op9_frac, shift_signal])`.",
    ])
    (out / "report.md").write_text("\n".join(report))

    print(f"\nWROTE {out} ({len(rows)} runs)")
    for r in summary_rows:
        print(
            f"  α={r['cvar_alpha']} hc={r['holding_cost']} init=({r['init_op3']},{r['init_op5']},{r['init_op9']}) "
            f"excel={r['excel_mean']:.5f} cd={r['cd_mean']:.3f} res={r['resource_mean']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
