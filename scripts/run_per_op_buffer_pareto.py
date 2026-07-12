#!/usr/bin/env python3
"""Per-operation continuous buffer agent vs a same-space static frontier.

Action contract:
  Box([op3_frac, op5_frac, op9_frac, shift_signal])

This is the smallest thesis-family expansion after `continuous_it_s`: inventory
and shifts remain the only decision families, but strategic buffer can be placed
differently at Op3, Op5 and Op9.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from supply_chain.continuous_its_env import make_per_op_buffer_track_a_env
from supply_chain.episode_metrics import compute_episode_metrics

SHIFT_SIGS = {1: -1.0, 2: 0.0, 3: 1.0}


def build(
    *,
    reward: str,
    obs_v: str,
    regime: str,
    phi: float,
    psi: float,
    max_steps: int,
    init_fracs,
    holding_cost: float,
    shift_cost: float,
    risk_obs: bool,
    cvar_alpha: float,
    step_size_hours: float,
    seed: int | None = None,
):
    env = make_per_op_buffer_track_a_env(
        reward_mode=reward,
        observation_version=obs_v,
        risk_level=regime,
        risk_frequency_multiplier=float(phi),
        risk_impact_multiplier=float(psi),
        stochastic_pt=False,
        max_steps=int(max_steps),
        step_size_hours=float(step_size_hours),
        init_fracs=init_fracs,
        risk_obs=bool(risk_obs),
        holding_cost=float(holding_cost),
        shift_cost=float(shift_cost),
        ret_excel_cvar_alpha=float(cvar_alpha),
    )
    if seed is not None:
        env.reset(seed=int(seed))
    return env


def _cvar(values: Iterable[float], alpha: float = 0.05) -> float:
    clean = sorted(x for x in values if x == x)
    if not clean:
        return float("nan")
    k = max(1, int(round(alpha * len(clean))))
    return float(np.mean(clean[-k:]))


def eval_policy(build_fn, act_fn, episodes: int, seed0: int, *, trace_label: str | None = None):
    excels, service_loss, resources, flow, lost = [], [], [], [], []
    trace_rows: list[dict] = []
    field_names: list[str] = []
    for ep in range(int(episodes)):
        env = build_fn()
        field_names = list(getattr(env, "obs_field_names", []))
        obs, _info = env.reset(seed=int(seed0) + ep)
        done = truncated = False
        ep_resource = []
        step = 0
        while not (done or truncated):
            obs_before = np.asarray(obs, dtype=np.float32)
            action = np.asarray(act_fn(obs), dtype=np.float32).reshape(-1)
            obs, reward, done, truncated, info = env.step(action)
            ep_resource.append(float(info.get("resource_composite", np.nan)))
            if trace_label:
                row = {
                    "policy": trace_label,
                    "episode": ep,
                    "step": step,
                    "action_op3_frac": float(action[0]) if action.size > 0 else float("nan"),
                    "action_op5_frac": float(action[1]) if action.size > 1 else float("nan"),
                    "action_op9_frac": float(action[2]) if action.size > 2 else float("nan"),
                    "action_shift_signal": float(action[3]) if action.size > 3 else float("nan"),
                    "applied_op3_frac": float(info.get("per_op_op3_frac", np.nan)),
                    "applied_op5_frac": float(info.get("per_op_op5_frac", np.nan)),
                    "applied_op9_frac": float(info.get("per_op_op9_frac", np.nan)),
                    "applied_shift": float(info.get("continuous_its_shift", np.nan)),
                    "resource_composite": float(info.get("resource_composite", np.nan)),
                    "reward": float(reward),
                }
                for idx, value in enumerate(obs_before):
                    name = field_names[idx] if idx < len(field_names) else f"obs_{idx}"
                    row[f"obs.{name}"] = float(value)
                trace_rows.append(row)
            step += 1
        metrics = compute_episode_metrics(env.unwrapped.sim)
        excels.append(float(metrics.get("ret_excel", np.nan)))
        service_loss.append(float(metrics.get("service_loss_auc_ration_hours", np.nan)))
        flow.append(float(metrics.get("flow_fill_rate", np.nan)))
        lost.append(float(metrics.get("lost_rate", np.nan)))
        resources.append(float(np.nanmean(ep_resource)))
        env.close()
    result = {
        "excel": float(np.nanmean(excels)),
        "cvar": _cvar(service_loss),
        "flow_fill": float(np.nanmean(flow)),
        "lost_rate": float(np.nanmean(lost)),
        "resource": float(np.nanmean(resources)),
    }
    if trace_label:
        result["trace_rows"] = trace_rows
    return result


def parse_fracs(raw: str) -> list[float]:
    vals = sorted({round(float(x), 4) for x in raw.split(",") if x.strip()})
    if not vals:
        raise ValueError("--static-fracs must contain at least one value")
    return vals


def parse_init_fracs(raw: str) -> list[float]:
    vals = [float(x) for x in raw.split(",") if x.strip()]
    if len(vals) == 1:
        vals = vals * 3
    if len(vals) != 3:
        raise ValueError("--init-fracs must be one value or three comma-separated values")
    return [float(np.clip(x, 0.0, 1.0)) for x in vals]


def static_candidates(fracs: list[float], mode: str) -> list[tuple[float, float, float, int]]:
    triples: set[tuple[float, float, float]] = set()
    if mode == "full":
        triples.update(itertools.product(fracs, fracs, fracs))
    else:
        positives = [x for x in fracs if x > 0]
        low = min(positives) if positives else 0.0
        for f in fracs:
            triples.add((f, f, f))          # old common-frac surface
            triples.add((0.0, 0.0, f))      # downstream-only buffer
            triples.add((0.0, f, f))        # avoid Op3 overfeeding
            triples.add((low, low, f))      # small upstream, variable Op9
            triples.add((0.0, f, 0.0))      # isolate Op5
            triples.add((f, 0.0, 0.0))      # isolate Op3
    out = []
    for triple in sorted(triples):
        for shift in (1, 2, 3):
            out.append((*triple, shift))
    return out


def pareto(dynamic: dict, statics: list[dict], key: str, *, higher_better: bool) -> dict:
    def better(a, b):
        return a > b if higher_better else a < b

    dominated_by = [
        s for s in statics
        if (better(s[key], dynamic[key]) or s[key] == dynamic[key])
        and s["resource"] <= dynamic["resource"] + 1e-9
        and (better(s[key], dynamic[key]) or s["resource"] < dynamic["resource"] - 1e-9)
    ]
    dominated = [
        s for s in statics
        if (better(dynamic[key], s[key]) or dynamic[key] == s[key])
        and dynamic["resource"] <= s["resource"] + 1e-9
        and (better(dynamic[key], s[key]) or dynamic["resource"] < s["resource"] - 1e-9)
    ]
    eligible = [s for s in statics if s["resource"] <= dynamic["resource"] + 1e-9]
    best_equal = None
    if eligible:
        best_equal = max(eligible, key=lambda s: s[key]) if higher_better else min(eligible, key=lambda s: s[key])
    return {
        "pareto_win": bool(not dominated_by and dominated),
        "dominated_by_static": bool(dominated_by),
        "n_static_dominated": len(dominated),
        "best_static_at_le_dynamic_resource": best_equal,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reward-mode", default="ReT_excel_delta")
    ap.add_argument("--observation-version", default="v6")
    ap.add_argument("--regime", default="current")
    ap.add_argument("--phi", type=float, default=4.0)
    ap.add_argument("--psi", type=float, default=1.5)
    ap.add_argument("--init-fracs", default="1,1,1")
    ap.add_argument("--holding-cost", type=float, default=0.0)
    ap.add_argument("--shift-cost", type=float, default=0.001)
    ap.add_argument("--cvar-alpha", type=float, default=0.2)
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--timesteps", type=int, default=30000)
    ap.add_argument("--eval-episodes", type=int, default=4)
    ap.add_argument("--eval-seed0", type=int, default=9000)
    ap.add_argument("--crn-eval", action="store_true")
    ap.add_argument("--max-steps", type=int, default=104)
    ap.add_argument("--step-size-hours", type=float, default=168.0)
    ap.add_argument("--static-fracs", default="0,0.05,0.1,0.2,0.5,1.0")
    ap.add_argument("--static-grid", choices=("targeted", "full"), default="targeted")
    ap.add_argument("--static-init-mode", choices=("fixed", "match"), default="fixed",
                    help="fixed: every static gets --init-fracs; match: static preposition equals its constant action")
    ap.add_argument("--output", default="outputs/experiments/per_op_buffer_pareto_2026-06-28")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    init_fracs = parse_init_fracs(args.init_fracs)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    cfg = dict(
        reward=args.reward_mode,
        obs_v=args.observation_version,
        regime=args.regime,
        phi=args.phi,
        psi=args.psi,
        max_steps=args.max_steps,
        init_fracs=init_fracs,
        holding_cost=args.holding_cost,
        shift_cost=args.shift_cost,
        risk_obs=True,
        cvar_alpha=args.cvar_alpha,
        step_size_hours=args.step_size_hours,
    )

    fracs = parse_fracs(args.static_fracs)
    statics = []
    for op3, op5, op9, shift in static_candidates(fracs, args.static_grid):
        sig = SHIFT_SIGS[shift]
        label = f"op3_{op3:g}_op5_{op5:g}_op9_{op9:g}_S{shift}"
        static_cfg = dict(cfg)
        if args.static_init_mode == "match":
            static_cfg["init_fracs"] = [op3, op5, op9]
        result = eval_policy(
            lambda c=static_cfg: build(**c),
            lambda _obs, a=np.array([op3, op5, op9, sig], dtype=np.float32): a,
            args.eval_episodes,
            args.eval_seed0,
        )
        result.update({"label": label, "op3_frac": op3, "op5_frac": op5, "op9_frac": op9, "shift": shift})
        statics.append(result)

    learned = []
    traces = []
    for seed in seeds:
        venv = DummyVecEnv([lambda s=seed + i: build(**cfg, seed=s) for i in range(args.n_envs)])
        model = PPO(
            "MlpPolicy",
            venv,
            seed=seed,
            verbose=0,
            n_steps=min(1024, args.max_steps * 4),
            batch_size=64,
            learning_rate=3e-4,
            n_epochs=10,
        )
        model.learn(total_timesteps=int(args.timesteps))
        eval_seed0 = args.eval_seed0 if args.crn_eval else seed * 100 + 9
        result = eval_policy(
            lambda: build(**cfg),
            lambda obs: model.predict(obs, deterministic=True)[0],
            args.eval_episodes,
            eval_seed0,
            trace_label=f"learned_seed{seed}",
        )
        seed_trace = result.pop("trace_rows", [])
        traces.extend({"seed": seed, **row} for row in seed_trace)
        result["seed"] = seed
        learned.append(result)

    dynamic = {
        "excel": float(np.nanmean([x["excel"] for x in learned])),
        "cvar": float(np.nanmean([x["cvar"] for x in learned])),
        "flow_fill": float(np.nanmean([x["flow_fill"] for x in learned])),
        "lost_rate": float(np.nanmean([x["lost_rate"] for x in learned])),
        "resource": float(np.nanmean([x["resource"] for x in learned])),
    }
    summary = {
        "args": vars(args),
        "static_candidates": len(statics),
        "statics": statics,
        "learned_per_seed": learned,
        "dynamic": dynamic,
        "excel_pareto": pareto(dynamic, statics, "excel", higher_better=True),
        "cvar_pareto": pareto(dynamic, statics, "cvar", higher_better=False),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    if traces:
        keys = sorted({k for row in traces for k in row})
        with (out / "action_trace.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(traces)

    print(f"\n=== PER-OP BUFFER PARETO ({args.reward_mode}, {args.regime} phi={args.phi}, psi={args.psi}) ===")
    print(
        f"DYNAMIC excel={dynamic['excel']:.6f} cvar={dynamic['cvar']:.3e} "
        f"flow={dynamic['flow_fill']:.3f} lost={dynamic['lost_rate']:.3f} "
        f"resource={dynamic['resource']:.3f}"
    )
    best_excel = max(statics, key=lambda s: s["excel"])
    best_cvar = min(statics, key=lambda s: s["cvar"])
    print(f"best static Excel: {best_excel['label']} excel={best_excel['excel']:.6f} res={best_excel['resource']:.3f}")
    print(f"best static CVaR : {best_cvar['label']} cvar={best_cvar['cvar']:.3e} res={best_cvar['resource']:.3f}")
    print(f"Excel Pareto: {summary['excel_pareto']}")
    print(f"CVaR  Pareto: {summary['cvar_pareto']}")
    print(f"WROTE {out}/summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
