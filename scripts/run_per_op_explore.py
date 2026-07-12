#!/usr/bin/env python3
"""Per-op buffer PPO/SAC training combining VecNormalize + high holding_cost + warm-start.

Variants:
  A: VecNormalize + holding_cost=0.02 + warm-start sweet spot + PPO (ReT_excel_plus_cvar)
  B: Same as A + ReT_tail_v1 reward (tail/recovery focus)
  C: Same as A + entropy bonus ent_coef=0.01
  D: SAC instead of PPO (better for continuous action spaces)
  E: Same as A + holding_cost=0.05 (extreme economization)

Action: per_op_buffer [op3_frac, op5_frac, op9_frac, shift_signal]
Warm-start: init_fracs=[0.0, 0.0, 0.10] (the static sweet spot)
Training: 30k steps, n_envs=4, 2 seeds per variant
Evaluation: Excel ReT + CVaR + CD (variance_log), 8 episodes

Static frontier: op3=0,op5=0 with op9 ∈ {0,0.05,0.10,0.15,0.20,0.25} × S∈{1,2,3}
  (The sweet spot is op3=0,op5=0,op9=0.10,S1 with excel=0.00263)
"""
from __future__ import annotations

import argparse, csv, json, sys, time
from pathlib import Path
from statistics import fmean
from typing import Any
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.continuous_its_env import make_per_op_buffer_track_a_env
from supply_chain.episode_metrics import compute_episode_metrics
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

SHIFT_SIGS = {1: -1.0, 2: 0.0, 3: 1.0}
STATIC_OP9_FRACS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]

VARIANTS = [
    # (name, reward_mode, algo, holding_cost, ent_coef, init_fracs, timesteps)
    ("A_vecnorm_hold0.02_warmstart", "ReT_excel_plus_cvar", "PPO", 0.02, None, (0.0, 0.0, 0.10), 30_000),
    ("B_tailv1_hold0.02_warmstart", "ReT_tail_v1", "PPO", 0.02, None, (0.0, 0.0, 0.10), 30_000),
    ("C_entropy_hold0.02_warmstart", "ReT_excel_plus_cvar", "PPO", 0.02, 0.01, (0.0, 0.0, 0.10), 30_000),
    ("D_sac_hold0.02_warmstart", "ReT_excel_plus_cvar", "SAC", 0.02, None, (0.0, 0.0, 0.10), 20_000),
    ("E_vecnorm_hold0.05_warmstart", "ReT_excel_plus_cvar", "PPO", 0.05, None, (0.0, 0.0, 0.10), 30_000),
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
        holding_cost=0.02,
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
        cvar_pool.append(float(metrics.get("service_loss_auc_ration_hours", 0.0)))
        resources.append(float(np.nanmean(ep_res)) if ep_res else 0.0)
        env.close()
    sl_sorted = sorted(cvar_pool)
    cvar_idx = max(0, int(round(0.05 * len(sl_sorted))) - 1)
    return {
        "excel": float(np.mean(excels)),
        "cvar": float(np.mean(sl_sorted[cvar_idx:])),
        "resource": float(np.mean(resources)),
    }


def compute_static_frontier(n_ep: int, seed0: int, base_params: dict):
    statics = []
    for op9 in STATIC_OP9_FRACS:
        for shift, sig in SHIFT_SIGS.items():
            def act_fn(o, f=0.0, f9=op9, s=sig):
                return np.array([f, f, f9, s], dtype=np.float32)
            r = evaluate(
                lambda: build_env(**(base_params | {"init_fracs": (0.0, 0.0, op9)})),
                act_fn, n_ep, seed0,
            )
            r["label"] = f"op3=0_op5=0_op9={op9}_S{shift}"
            r["op9_frac"] = op9
            r["shift"] = shift
            statics.append(r)
    return statics


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--eval-episodes", type=int, default=8)
    ap.add_argument("--eval-seed0", type=int, default=5000)
    ap.add_argument("--max-steps", type=int, default=104)
    ap.add_argument("--output", default="outputs/experiments/per_op_explore_2026-06-29")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    base_params = dict(
        observation_version="v6",
        risk_level="current",
        risk_frequency_multiplier=4.0,
        risk_impact_multiplier=1.5,
        max_steps=args.max_steps,
    )

    print("Computing per-op static frontier (op9 × shift grid)...", flush=True)
    t0 = time.time()
    statics = compute_static_frontier(args.eval_episodes, args.eval_seed0, base_params)
    best_static = max(statics, key=lambda s: s["excel"])
    print(f"  done ({time.time()-t0:.0f}s). Best: {best_static['label']} excel={best_static['excel']:.5f} cvar={best_static['cvar']:.2e} res={best_static['resource']:.3f}")
    print()

    rows = []
    started = time.time()
    for vname, reward_mode, algo, hc, ent_coef, init_f, timesteps in VARIANTS:
        for seed in seeds:
            label = f"{vname}_s{seed}"
            run_dir = out / label
            run_dir.mkdir(parents=True, exist_ok=True)
            cell = dict(base_params, reward_mode=reward_mode, holding_cost=hc,
                       init_fracs=init_f, seed=seed)
            if reward_mode == "ReT_excel_plus_cvar":
                cell["ret_excel_cvar_alpha"] = 0.2
            print(f"  [{label}] train ({algo}, {reward_mode}, hc={hc})...", end=" ", flush=True)
            t_train = time.time()
            dummy = DummyVecEnv(
                [lambda c=cell, i=i: build_env(**(c | {"seed": c["seed"] + i}))
                 for i in range(4)]
            )
            venv = VecNormalize(dummy, norm_obs=True, norm_reward=True, clip_reward=10.0)
            if algo == "PPO":
                model = PPO("MlpPolicy", venv, seed=seed, verbose=0,
                           n_steps=512, batch_size=64, learning_rate=3e-4, n_epochs=10,
                           ent_coef=ent_coef if ent_coef else 0.0)
            else:
                model = SAC("MlpPolicy", venv, seed=seed, verbose=0,
                           learning_rate=3e-4, ent_coef="auto")
            model.learn(total_timesteps=int(timesteps))
            train_s = time.time() - t_train
            print(f"eval...", end=" ", flush=True)
            r = evaluate(
                lambda c=cell: build_env(**c),
                lambda o: model.predict(o, deterministic=True)[0],
                args.eval_episodes, args.eval_seed0,
            )
            r["variant"] = vname
            r["reward_mode"] = reward_mode
            r["algo"] = algo
            r["holding_cost"] = hc
            r["ent_coef"] = ent_coef if ent_coef else 0.0
            r["seed"] = seed
            r["train_seconds"] = train_s
            r["timesteps"] = timesteps
            rows.append(r)
            venv.close()
            delta = r["excel"] - best_static["excel"]
            print(f"excel={r['excel']:.5f} (Δ{delta:+.5f}) cvar={r['cvar']:.2e} res={r['resource']:.3f}")

    elapsed = time.time() - started

    with (out / "results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    grouped = {}
    for r in rows:
        grouped.setdefault(r["variant"], []).append(r)

    report = [
        "# Per-Op Buffer Exploration (VecNormalize + High Holding Cost + Warm-Start)",
        f"Variants: {len(VARIANTS)} × {len(seeds)} seeds = {len(rows)} runs | Wall: {elapsed:.0f}s",
        "",
        f"**Static sweet spot:** {best_static['label']} → excel={best_static['excel']:.5f} cvar={best_static['cvar']:.2e} res={best_static['resource']:.3f}",
        "",
        "## Results vs Sweet Spot",
        "| Variant | Reward | Algo | HC | Excel mean | Δ vs sweet | CVaR | Resource |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for vname, reward_mode, algo, hc, ent_coef, init_f, timesteps in VARIANTS:
        grp = grouped.get(vname, [])
        if not grp:
            continue
        ex = fmean(r["excel"] for r in grp)
        cv = fmean(r["cvar"] for r in grp)
        rs = fmean(r["resource"] for r in grp)
        delta = ex - best_static["excel"]
        report.append(
            f"| {vname} | {reward_mode} | {algo} | {hc} "
            f"| {ex:.5f} | {delta:+.5f} | {cv:.2e} | {rs:.3f} |"
        )

    winners = [r for r in rows if r["excel"] > best_static["excel"]]
    report.extend([
        "",
        f"## Raw Excel Wins vs Sweet Spot: {len(winners)}/{len(rows)}",
    ])
    if winners:
        for r in sorted(winners, key=lambda x: -x["excel"]):
            report.append(f"- {r['variant']} s{r['seed']}: excel={r['excel']:.5f} cvar={r['cvar']:.2e} res={r['resource']:.3f}")

    report.extend([
        "",
        "## Static Frontier (op3=0, op5=0, op9 × shift)",
        "| Label | Excel | CVaR | Resource |",
        "|---|---:|---:|---:|",
    ])
    for s in sorted(statics, key=lambda s: s["excel"], reverse=True):
        report.append(f"| {s['label']} | {s['excel']:.5f} | {s['cvar']:.2e} | {s['resource']:.3f} |")

    report.extend([
        "",
        "## Next",
        f"Winners: {len(winners)}. If VecNormalize + high HC reaches the sweet spot, promote to 5-seed confirmatory.",
        "If no variant reaches 0.00263, the 4D per-op space is too large for PPO at 104 steps/30k timesteps.",
        "Then try: longer training (60k+), SAC with auto entropy, or CMA-ES direct optimization.",
    ])
    (out / "report.md").write_text("\n".join(report))

    print(f"\nWROTE {out} ({len(rows)} runs, {len(winners)} winners)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
