#!/usr/bin/env python3
"""Joint α × holding_cost grid sweep — Lane A closeout.

Tests the last unexplored cell of the Track A continuous_its lane:
  cvar_alpha  ∈ {0.0, 0.1, 0.2, 0.5}
  holding_cost ∈ {0.0, 0.002, 0.005, 0.01, 0.02}

Each cell trains PPO (continuous_its + risk_obs + ReT_excel_plus_cvar, war φ4/ψ1.5,
h104) and evaluates against a dense 21-frac resource-charged static frontier.

Win rules:
  raw_ret_win: dynamic beats the best static on Excel ReT, regardless of resource
  cvar_win: dynamic beats the best static on CVaR, regardless of resource
  hard_resource_win: dynamic matches/exceeds f0.10_S1's Excel and CVaR at resource ≤ 0.05
  pareto_win: dynamic is not dominated by any static on Excel, CVaR, and resource

Phase 1: 1 seed × 20 cells × 30k steps (quick screen, ~15 min)
Phase 2: winner cells → 3 seeds × 60k steps (confirmatory, ~2h)
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
import time
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
STATIC_FRACS = [round(i / 20, 4) for i in range(21)]  # 0.00 .. 1.00 step 0.05
GRID = list(
    itertools.product(
        [0.0, 0.1, 0.2, 0.5],  # cvar_alpha
        [0.0, 0.002, 0.005, 0.01, 0.02],  # holding_cost
    )
)

DENSE_STATIC_FRACS = STATIC_FRACS


def build_env(**overrides):
    defaults = dict(
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
        holding_cost=0.0,
        shift_cost=0.0,
        ret_excel_cvar_alpha=0.2,
    )
    defaults.update(overrides)
    p = defaults
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


def evaluate_policy(env_factory, act_fn, n_episodes: int, seed0: int) -> dict[str, Any]:
    excels, service_losses, resources, fracs = [], [], [], []
    for ep in range(n_episodes):
        env = env_factory()
        obs, _ = env.reset(seed=seed0 + ep)
        done = truncated = False
        ep_res, ep_fr = [], []
        while not (done or truncated):
            action = np.asarray(act_fn(obs), dtype=np.float32).reshape(-1)
            obs, _reward, done, truncated, info = env.step(action)
            ep_res.append(float(info.get("resource_composite", np.nan)))
            ep_fr.append(float(info.get("continuous_its_frac", np.nan)))
        metrics = compute_episode_metrics(env.unwrapped.sim)
        excels.append(float(metrics.get("ret_excel", 0.0)))
        sl = float(metrics.get("service_loss_auc_ration_hours", 0.0))
        service_losses.append(sl)
        resources.append(float(np.nanmean(ep_res)) if ep_res else 0.0)
        fracs.extend(f for f in ep_fr if f == f)
        env.close()
    n = max(1, len(service_losses))
    sl_sorted = sorted(service_losses)
    cvar_idx = max(0, int(round(0.05 * n)) - 1)
    return {
        "excel": float(np.mean(excels)),
        "cvar": float(np.mean(sl_sorted[cvar_idx:])),
        "service_loss_mean": fmean(service_losses),
        "resource": float(np.mean(resources)),
        "frac_std": float(np.std(fracs)) if len(fracs) > 1 else 0.0,
        "excel_per_ep": excels,
    }


def compute_static_frontier(base_env_params: dict, n_episodes: int, seed0: int):
    statics = []
    for frac in DENSE_STATIC_FRACS:
        for shift, sig in SHIFT_SIGS.items():
            label = f"f{frac}_S{shift}"

            def act_fn(o, f=frac, s=sig):
                return np.array([f, s], dtype=np.float32)

            r = evaluate_policy(
                lambda sp=base_env_params: build_env(**sp),
                act_fn,
                n_episodes,
                seed0,
            )
            r["label"] = label
            r["frac"] = frac
            r["shift"] = shift
            statics.append(r)
    return statics


def f0_10_values(statics: list[dict]) -> dict:
    for s in statics:
        if s["label"] == "f0.1_S1":
            return {
                "excel": s["excel"],
                "cvar": s["cvar"],
                "resource": s["resource"],
            }
    return {"excel": 0.0, "cvar": float("inf"), "resource": 0.0}


def _best_static_values(statics: list[dict]) -> dict[str, dict]:
    return {
        "best_excel": max(statics, key=lambda s: s["excel"]),
        "best_cvar": min(statics, key=lambda s: s["cvar"]),
        "best_resource": min(statics, key=lambda s: s["resource"]),
    }


def _pareto_status(dynamic: dict, statics: list[dict]) -> dict:
    dominated_by = []
    dominates = []
    for static in statics:
        static_no_worse = (
            static["excel"] >= dynamic["excel"]
            and static["cvar"] <= dynamic["cvar"]
            and static["resource"] <= dynamic["resource"]
        )
        static_better = (
            static["excel"] > dynamic["excel"]
            or static["cvar"] < dynamic["cvar"]
            or static["resource"] < dynamic["resource"]
        )
        if static_no_worse and static_better:
            dominated_by.append(static["label"])

        dyn_no_worse = (
            dynamic["excel"] >= static["excel"]
            and dynamic["cvar"] <= static["cvar"]
            and dynamic["resource"] <= static["resource"]
        )
        dyn_better = (
            dynamic["excel"] > static["excel"]
            or dynamic["cvar"] < static["cvar"]
            or dynamic["resource"] < static["resource"]
        )
        if dyn_no_worse and dyn_better:
            dominates.append(static["label"])

    return {
        "pareto_win": bool((not dominated_by) and dominates),
        "pareto_dominated_by_static": bool(dominated_by),
        "pareto_dominated_by": dominated_by[:10],
        "pareto_dominates_static_count": len(dominates),
    }


def win_check(dynamic: dict, target: dict, statics: list[dict]) -> dict:
    best = _best_static_values(statics)
    excel_ok = dynamic["excel"] >= target["excel"]
    cvar_ok = dynamic["cvar"] <= target["cvar"]
    resource_ok = dynamic["resource"] <= 0.05
    hard_resource_win = excel_ok and cvar_ok and resource_ok
    raw_ret_delta = dynamic["excel"] - best["best_excel"]["excel"]
    cvar_delta_vs_best = dynamic["cvar"] - best["best_cvar"]["cvar"]
    resource_efficient_ret_win = (
        dynamic["excel"] >= best["best_excel"]["excel"]
        and dynamic["resource"] <= best["best_excel"]["resource"]
    )
    return {
        "win": hard_resource_win,
        "hard_resource_win": hard_resource_win,
        "raw_ret_win": raw_ret_delta > 0,
        "cvar_win": cvar_delta_vs_best < 0,
        "resource_efficient_win": resource_efficient_ret_win,
        "excel_ok": excel_ok,
        "cvar_ok": cvar_ok,
        "resource_ok": resource_ok,
        "excel_delta": dynamic["excel"] - target["excel"],
        "cvar_delta": dynamic["cvar"] - target["cvar"],
        "resource_delta": dynamic["resource"] - 0.05,
        "best_static_excel_label": best["best_excel"]["label"],
        "best_static_excel": best["best_excel"]["excel"],
        "raw_ret_delta_vs_best_static": raw_ret_delta,
        "best_static_cvar_label": best["best_cvar"]["label"],
        "best_static_cvar": best["best_cvar"]["cvar"],
        "cvar_delta_vs_best_static": cvar_delta_vs_best,
        "best_static_excel_resource": best["best_excel"]["resource"],
        "resource_delta_vs_best_excel_static": dynamic["resource"] - best["best_excel"]["resource"],
        **_pareto_status(dynamic, statics),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default="1")
    ap.add_argument("--timesteps", type=int, default=30_000)
    ap.add_argument("--n-envs", type=int, default=2)
    ap.add_argument("--eval-episodes", type=int, default=6)
    ap.add_argument("--eval-seed0", type=int, default=5000)
    ap.add_argument("--crn-eval", action="store_true",
                    help="evaluate every learned policy on the same seed block as the static frontier")
    ap.add_argument("--n-fracs", type=int, default=21)
    ap.add_argument("--max-steps", type=int, default=104)
    ap.add_argument("--output", default="outputs/experiments/alpha_holding_grid_2026-06-29")
    ap.add_argument("--phase", choices=["1", "2"], default="1",
                    help="1=quick screen, 2=confirmatory (more seeds+timesteps)")
    ap.add_argument("--cells", default="",
                    help="comma-separated 'alpha,hold' pairs override grid; e.g. '0.2,0.002 0.2,0.005'")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    if args.cells:
        cells = []
        for tok in args.cells.split():
            a, h = tok.split(",")
            cells.append((float(a), float(h)))
    else:
        cells = GRID

    if args.phase == "2":
        args.timesteps = 60_000
        if len(seeds) < 3:
            seeds = [1, 2, 3]

    global DENSE_STATIC_FRACS
    DENSE_STATIC_FRACS = [
        round(i / max(1, args.n_fracs - 1), 4) for i in range(args.n_fracs)
    ]

    base_params = dict(
        reward_mode="ReT_excel_plus_cvar",
        observation_version="v6",
        risk_level="current",
        risk_frequency_multiplier=4.0,
        risk_impact_multiplier=1.5,
        max_steps=args.max_steps,
    )

    print("Computing dense static frontier (shared across grid)... ", end="", flush=True)
    t0 = time.time()
    statics = compute_static_frontier(base_params, args.eval_episodes, args.eval_seed0)
    target = f0_10_values(statics)
    print(
        f"done ({(time.time()-t0):.0f}s, f0.10_S1: excel={target['excel']:.5f} "
        f"cvar={target['cvar']:.2e} resource={target['resource']:.3f})"
    )

    rows: list[dict] = []
    started = time.time()
    for cvar_alpha, holding_cost in cells:
        for seed in seeds:
            label = f"ca{cvar_alpha}_hc{holding_cost}_s{seed}"
            run_dir = out / label
            run_dir.mkdir(parents=True, exist_ok=True)
            cfg = {
                "cvar_alpha": cvar_alpha,
                "holding_cost": holding_cost,
                "seed": seed,
                **base_params,
            }
            print(f"  [{label}] training... ", end="", flush=True)
            t_train = time.time()
            venv = DummyVecEnv(
                [lambda c=cfg, i=i: build_env(**(c | {"seed": c["seed"] + i}))
                 for i in range(args.n_envs)]
            )
            model = PPO(
                "MlpPolicy",
                venv,
                seed=seed,
                verbose=0,
                n_steps=min(512, args.max_steps * 4),
                batch_size=64,
                learning_rate=3e-4,
                n_epochs=10,
            )
            model.learn(total_timesteps=int(args.timesteps))
            train_sec = time.time() - t_train
            print(f"eval... ", end="", flush=True)
            eval_seed0 = args.eval_seed0 if args.crn_eval else seed * 1000 + 1
            r = evaluate_policy(
                lambda c=cfg: build_env(**c),
                lambda o: model.predict(o, deterministic=True)[0],
                args.eval_episodes,
                eval_seed0,
            )
            verdict = win_check(r, target, statics)
            r["cvar_alpha"] = cvar_alpha
            r["holding_cost"] = holding_cost
            r["seed"] = seed
            r["train_seconds"] = train_sec
            r["timesteps"] = args.timesteps
            r.update(verdict)
            rows.append(r)
            (run_dir / "metrics.json").write_text(json.dumps(r, indent=2, default=float))
            venv.close()
            print(
                f"excel={r['excel']:.5f} cvar={r['cvar']:.2e} res={r['resource']:.3f} "
                f"frac_std={r['frac_std']:.3f} raw_ret_win={r['raw_ret_win']} "
                f"pareto_win={r['pareto_win']} hard_win={r['hard_resource_win']}"
            )

    elapsed = time.time() - started

    with (out / "grid_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    static_path = out / "static_frontier.json"
    static_payload = {
        "target_f0_10_S1": target,
        "statics": [
            {
                "label": s["label"],
                "frac": s["frac"],
                "shift": s["shift"],
                "excel": s["excel"],
                "cvar": s["cvar"],
                "resource": s["resource"],
            }
            for s in statics
        ],
    }
    static_path.write_text(json.dumps(static_payload, indent=2, default=float))

    winning = [r for r in rows if r["win"]]
    raw_ret_winning = [r for r in rows if r["raw_ret_win"]]
    cvar_winning = [r for r in rows if r["cvar_win"]]
    pareto_winning = [r for r in rows if r["pareto_win"]]
    candidates = [r for r in rows if r["resource"] <= 0.06 and r["excel"] >= target["excel"] * 0.95]
    best_static = _best_static_values(statics)
    best_dynamic = max(rows, key=lambda r: r["excel"])

    summary_lines = [
        "# α × holding_cost Grid Sweep",
        "",
        f"Phase: {args.phase} | Grid: {len(cells)} cells × {len(seeds)} seeds "
        f"× {args.timesteps} steps = {len(rows)} runs",
        f"Wall time: {elapsed:.0f}s",
        "",
        f"**Target:** f0.10_S1 → excel={target['excel']:.5f}, "
        f"cvar={target['cvar']:.2e}, resource={target['resource']:.3f}",
        "",
        f"**Best static ReT:** {best_static['best_excel']['label']} → "
        f"excel={best_static['best_excel']['excel']:.5f}, "
        f"cvar={best_static['best_excel']['cvar']:.2e}, "
        f"resource={best_static['best_excel']['resource']:.3f}",
        f"**Best dynamic ReT:** α={best_dynamic['cvar_alpha']}, hc={best_dynamic['holding_cost']} → "
        f"excel={best_dynamic['excel']:.5f}, "
        f"Δ vs best static={best_dynamic['raw_ret_delta_vs_best_static']:+.5f}, "
        f"resource={best_dynamic['resource']:.3f}",
        "",
        f"**Raw ReT winning cells:** {len(raw_ret_winning)} (dynamic beats best static ReT, resource ignored)",
        f"**CVaR winning cells:** {len(cvar_winning)} (dynamic beats best static CVaR, resource ignored)",
        f"**Pareto winning cells:** {len(pareto_winning)} (dynamic non-dominated on ReT/CVaR/resource)",
        f"**Hard resource winning cells:** {len(winning)} (dynamic matches/exceeds f0.10_S1 at resource ≤ 0.05)",
        f"**Candidate cells:** {len(candidates)} (resource ≤ 0.06 AND excel ≥ 95% of target)",
        "",
        "## Grid Results",
        "",
        "| cvar_α | holding_cost | Excel | CVaR | Resource | frac_std | Δ vs best ReT static | Δ vs target | Raw ReT | Pareto | Hard |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|:---:|",
    ]

    if winning:
        for r in sorted(winning, key=lambda x: x["excel"], reverse=True):
            summary_lines.append(
                f"| {r['cvar_alpha']} | {r['holding_cost']} | {r['excel']:.5f} "
                f"| {r['cvar']:.2e} | {r['resource']:.3f} | {r['frac_std']:.3f} "
                f"| {r['raw_ret_delta_vs_best_static']:+.5f} | {r['excel_delta']:+.5f} "
                f"| {'✅' if r['raw_ret_win'] else '—'} "
                f"| {'✅' if r['pareto_win'] else '—'} "
                f"| {'✅' if r['hard_resource_win'] else '—'} |"
            )
    else:
        # show all
        grouped: dict[tuple, list[dict]] = {}
        for r in rows:
            k = (r["cvar_alpha"], r["holding_cost"])
            grouped.setdefault(k, []).append(r)
        for (ca, hc), grp in sorted(grouped.items()):
            best = max(grp, key=lambda x: x["excel"])
            summary_lines.append(
                f"| {ca} | {hc} | {best['excel']:.5f} "
                f"| {best['cvar']:.2e} | {best['resource']:.3f} | {best['frac_std']:.3f} "
                f"| {best['raw_ret_delta_vs_best_static']:+.5f} | {best['excel_delta']:+.5f} "
                f"| {'✅' if best['raw_ret_win'] else '—'} "
                f"| {'✅' if best['pareto_win'] else '—'} "
                f"| {'✅' if best['hard_resource_win'] else '—'} |"
            )

    summary_lines.extend(
        [
            "",
            "## Static Frontier (resource-charged, dense)",
            "",
            "| Label | Excel | CVaR | Resource |",
            "|---|---:|---:|---:|",
        ]
    )
    for s in sorted(statics, key=lambda s: s["resource"]):
        summary_lines.append(
            f"| {s['label']} | {s['excel']:.5f} | {s['cvar']:.2e} | {s['resource']:.3f} |"
        )
    summary_lines.extend(
        [
            "",
            "## Verdict",
            "",
            f"Winning cells: {len(winning)}/{len(rows)}.",
            f"Raw ReT winning cells: {len(raw_ret_winning)}/{len(rows)}.",
            f"Pareto winning cells: {len(pareto_winning)}/{len(rows)}.",
            f"Hard resource winning cells: {len(winning)}/{len(rows)}.",
            f"Raw resilience verdict: {'SURVIVES' if raw_ret_winning else 'FAILS'} — "
            + (
                f"{len(raw_ret_winning)} cells beat the best static ReT, resource not used as a gate"
                if raw_ret_winning
                else "no dynamic cell beats the best static ReT"
            ),
            f"Hard resource verdict: {'FAILS' if not winning else 'SURVIVES'} — "
            + (
                "no dynamic cell matches f0.10_S1 at resource ≤ 0.05"
                if not winning
                else f"{len(winning)} cells survive the hard win rule"
            ),
        ]
    )
    (out / "grid_report.md").write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"\n=== GRID SWEEP COMPLETE ===")
    print(f"Target f0.10_S1: excel={target['excel']:.5f} cvar={target['cvar']:.2e} resource={target['resource']:.3f}")
    print(f"Winners: {len(winning)}/{len(rows)} | WROTE {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
