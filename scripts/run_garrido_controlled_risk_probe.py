#!/usr/bin/env python3
"""Cheap Garrido controlled-risk probe: R1/R2/R3 or CF-specific risks.

This is a signal-finding runner, not a confirmatory experiment. It answers:

1. In a thesis-like risk family (R1-only, R2-only, R3-only), does a learned
   continuous Track-A policy beat or tie the dense static buffer/shift frontier?
2. Does the learned policy actually vary its actions with observed risk/hazard,
   or does it collapse to a constant base-stock policy?

The design is intentionally cheap: 1-2 learner seeds, short h52/h104 horizons,
and CRN eval seeds. Scale only lanes that survive this screen.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from supply_chain.continuous_its_env import make_continuous_its_track_a_env
from supply_chain.episode_metrics import compute_episode_metrics
from supply_chain.thesis_design import (
    R1_RISKS,
    R2_RISKS,
    R3_RISKS,
    design_spec_for_cfi,
)

SHIFT_SIGS = {1: -1.0, 2: 0.0, 3: 1.0}
FAMILY_RISKS = {
    "R1": tuple(R1_RISKS),
    "R2": tuple(R2_RISKS),
    "R3": tuple(R3_RISKS),
}


@dataclass(frozen=True)
class RiskCase:
    name: str
    enabled_risks: tuple[str, ...]
    risk_overrides: dict[str, str]


def parse_csv_ints(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_csv_floats(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def risk_cases(raw: str) -> list[RiskCase]:
    cases: list[RiskCase] = []
    for token in [x.strip() for x in raw.split(",") if x.strip()]:
        up = token.upper()
        if up in FAMILY_RISKS:
            risks = FAMILY_RISKS[up]
            cases.append(
                RiskCase(
                    name=up,
                    enabled_risks=risks,
                    risk_overrides={risk_id: "current" for risk_id in risks},
                )
            )
            continue
        if up.startswith("CF"):
            cfi = int(up[2:])
        else:
            cfi = int(up)
        spec = design_spec_for_cfi(cfi)
        cases.append(
            RiskCase(
                name=f"CF{cfi}",
                enabled_risks=tuple(spec.enabled_risks),
                risk_overrides=dict(spec.risk_overrides),
            )
        )
    if not cases:
        raise ValueError("No risk cases parsed.")
    return cases


def build_env(
    *,
    case: RiskCase,
    reward_mode: str,
    observation_version: str,
    max_steps: int,
    step_size_hours: float,
    holding_cost: float,
    shift_cost: float,
    cvar_alpha: float,
    risk_level: str,
    phi: float,
    psi: float,
    seed: int | None = None,
):
    env = make_continuous_its_track_a_env(
        reward_mode=reward_mode,
        observation_version=observation_version,
        risk_level=risk_level,
        risk_frequency_multiplier=float(phi),
        risk_impact_multiplier=float(psi),
        enabled_risks=case.enabled_risks,
        risk_overrides=case.risk_overrides,
        stochastic_pt=False,
        max_steps=int(max_steps),
        step_size_hours=float(step_size_hours),
        risk_obs=True,
        holding_cost=float(holding_cost),
        shift_cost=float(shift_cost),
        ret_excel_cvar_alpha=float(cvar_alpha),
    )
    if seed is not None:
        env.reset(seed=int(seed))
    return env


def _cvar(values: Iterable[float], alpha: float = 0.05) -> float:
    clean = sorted(float(x) for x in values if np.isfinite(x))
    if not clean:
        return float("nan")
    k = max(1, int(round(alpha * len(clean))))
    return float(np.mean(clean[-k:]))


def eval_policy(*, build_fn, act_fn, episodes: int, seed0: int, trace_label: str | None = None) -> dict:
    excels, service_loss, flow_fill, lost_rate, resources, fracs = [], [], [], [], [], []
    traces: list[dict] = []
    for ep in range(int(episodes)):
        env = build_fn(seed0 + ep)
        field_names = list(getattr(env, "obs_field_names", []))
        obs, _ = env.reset(seed=seed0 + ep)
        done = truncated = False
        step = 0
        ep_resources = []
        while not (done or truncated):
            obs_before = np.asarray(obs, dtype=np.float32)
            action = np.asarray(act_fn(obs), dtype=np.float32).reshape(-1)
            obs, reward, done, truncated, info = env.step(action)
            frac = float(info.get("continuous_its_frac", np.nan))
            res = float(info.get("resource_composite", np.nan))
            fracs.append(frac)
            ep_resources.append(res)
            if trace_label:
                row = {
                    "policy": trace_label,
                    "episode": ep,
                    "step": step,
                    "action_frac": float(action[0]) if action.size > 0 else float("nan"),
                    "action_shift_signal": float(action[1]) if action.size > 1 else float("nan"),
                    "applied_frac": frac,
                    "applied_shift": float(info.get("continuous_its_shift", np.nan)),
                    "resource_composite": res,
                    "reward": float(reward),
                }
                for idx, value in enumerate(obs_before):
                    name = field_names[idx] if idx < len(field_names) else f"obs_{idx}"
                    row[f"obs.{name}"] = float(value)
                traces.append(row)
            step += 1
        metrics = compute_episode_metrics(env.unwrapped.sim)
        excels.append(float(metrics.get("ret_excel", np.nan)))
        service_loss.append(float(metrics.get("service_loss_auc_ration_hours", np.nan)))
        flow_fill.append(float(metrics.get("flow_fill_rate", np.nan)))
        lost_rate.append(float(metrics.get("lost_rate", np.nan)))
        resources.append(float(np.nanmean(ep_resources)))
        env.close()
    out = {
        "excel": float(np.nanmean(excels)),
        "cvar95_service_loss": _cvar(service_loss, alpha=0.05),
        "flow_fill": float(np.nanmean(flow_fill)),
        "lost_rate": float(np.nanmean(lost_rate)),
        "resource": float(np.nanmean(resources)),
        "frac_std": float(np.nanstd(fracs)) if fracs else 0.0,
    }
    if trace_label:
        out["trace_rows"] = traces
    return out


def action_correlations(trace_rows: list[dict], *, target: str = "applied_frac", top_k: int = 12) -> list[dict]:
    if not trace_rows:
        return []
    y = np.asarray([float(r.get(target, np.nan)) for r in trace_rows], dtype=float)
    if y.size < 4 or np.nanstd(y) <= 1e-12:
        return []
    rows = []
    for key in sorted(k for k in trace_rows[0] if k.startswith("obs.")):
        x = np.asarray([float(r.get(key, np.nan)) for r in trace_rows], dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 4 or np.nanstd(x[mask]) <= 1e-12:
            continue
        corr = float(np.corrcoef(x[mask], y[mask])[0, 1])
        if np.isfinite(corr):
            rows.append({"feature": key[4:], "corr": corr, "abs_corr": abs(corr)})
    rows.sort(key=lambda r: r["abs_corr"], reverse=True)
    return rows[:top_k]


def pareto_flags(dynamic: dict, statics: list[dict], metric: str, *, higher_better: bool) -> dict:
    def better(a: float, b: float) -> bool:
        return a > b if higher_better else a < b

    dominated_by = [
        s for s in statics
        if (better(s[metric], dynamic[metric]) or np.isclose(s[metric], dynamic[metric]))
        and s["resource"] <= dynamic["resource"] + 1e-9
        and (better(s[metric], dynamic[metric]) or s["resource"] < dynamic["resource"] - 1e-9)
    ]
    eligible = [s for s in statics if s["resource"] <= dynamic["resource"] + 1e-9]
    best_at_resource = None
    if eligible:
        best_at_resource = max(eligible, key=lambda s: s[metric]) if higher_better else min(eligible, key=lambda s: s[metric])
    best_any = max(statics, key=lambda s: s[metric]) if higher_better else min(statics, key=lambda s: s[metric])
    return {
        "dominated_by_static": bool(dominated_by),
        "beats_best_static_at_le_resource": (
            bool(best_at_resource and better(dynamic[metric], best_at_resource[metric]))
        ),
        "best_static_at_le_resource": best_at_resource,
        "best_static_any_resource": best_any,
    }


def learning_verdict(dynamic: dict, corr: list[dict]) -> dict:
    risk_corr = [
        r for r in corr
        if r["feature"].startswith(("active_", "recent_", "weeks_since_last_", "ewma_"))
    ]
    return {
        "varies_actions": bool(dynamic["frac_std"] > 0.03),
        "risk_conditioned": bool(risk_corr and risk_corr[0]["abs_corr"] >= 0.20),
        "top_risk_or_hazard_feature": risk_corr[0] if risk_corr else None,
        "interpretation": (
            "risk-conditioned adaptive policy"
            if dynamic["frac_std"] > 0.03 and risk_corr and risk_corr[0]["abs_corr"] >= 0.20
            else "mostly static or weakly risk-conditioned"
        ),
    }


def run_case(args: argparse.Namespace, case: RiskCase) -> dict:
    cfg = {
        "case": case,
        "reward_mode": args.reward_mode,
        "observation_version": args.observation_version,
        "max_steps": args.max_steps,
        "step_size_hours": args.step_size_hours,
        "holding_cost": args.holding_cost,
        "shift_cost": args.shift_cost,
        "cvar_alpha": args.cvar_alpha,
        "risk_level": args.risk_level,
        "phi": args.phi,
        "psi": args.psi,
    }

    def builder(seed: int):
        return build_env(**cfg, seed=seed)

    statics: list[dict] = []
    for frac in parse_csv_floats(args.fracs):
        for shift, sig in SHIFT_SIGS.items():
            result = eval_policy(
                build_fn=builder,
                act_fn=lambda _obs, f=frac, s=sig: np.array([f, s], dtype=np.float32),
                episodes=args.eval_episodes,
                seed0=args.eval_seed0,
            )
            result["label"] = f"f{frac:g}_S{shift}"
            result["frac"] = frac
            result["shift"] = shift
            statics.append(result)

    learned_per_seed: list[dict] = []
    trace_rows: list[dict] = []
    corr_by_seed: list[dict] = []
    for seed in parse_csv_ints(args.seeds):
        venv = DummyVecEnv([
            lambda s=seed + i: build_env(**cfg, seed=s)
            for i in range(args.n_envs)
        ])
        n_steps = min(512, max(32, args.max_steps * 4))
        batch_size = min(32, n_steps * max(1, args.n_envs))
        model = PPO(
            "MlpPolicy",
            venv,
            seed=seed,
            verbose=0,
            n_steps=n_steps,
            batch_size=batch_size,
            learning_rate=args.learning_rate,
            n_epochs=5,
        )
        model.learn(total_timesteps=int(args.timesteps))
        learned = eval_policy(
            build_fn=builder,
            act_fn=lambda obs, m=model: m.predict(obs, deterministic=True)[0],
            episodes=args.eval_episodes,
            seed0=args.eval_seed0,
            trace_label=f"{case.name}_seed{seed}",
        )
        rows = learned.pop("trace_rows", [])
        trace_rows.extend({"case": case.name, "seed": seed, **row} for row in rows)
        corr = action_correlations(rows)
        corr_by_seed.append({"seed": seed, "top_action_correlations": corr})
        learned["seed"] = seed
        learned["learning_verdict"] = learning_verdict(learned, corr)
        learned_per_seed.append(learned)

    dynamic = {
        "excel": float(np.nanmean([r["excel"] for r in learned_per_seed])),
        "cvar95_service_loss": float(np.nanmean([r["cvar95_service_loss"] for r in learned_per_seed])),
        "flow_fill": float(np.nanmean([r["flow_fill"] for r in learned_per_seed])),
        "lost_rate": float(np.nanmean([r["lost_rate"] for r in learned_per_seed])),
        "resource": float(np.nanmean([r["resource"] for r in learned_per_seed])),
        "frac_std": float(np.nanmean([r["frac_std"] for r in learned_per_seed])),
    }
    excel_flags = pareto_flags(dynamic, statics, "excel", higher_better=True)
    cvar_flags = pareto_flags(dynamic, statics, "cvar95_service_loss", higher_better=False)
    risk_conditioned = any(
        r["learning_verdict"]["risk_conditioned"] for r in learned_per_seed
    )
    case_summary = {
        "case": {
            "name": case.name,
            "enabled_risks": case.enabled_risks,
            "risk_overrides": case.risk_overrides,
        },
        "static_frontier": statics,
        "learned_per_seed": learned_per_seed,
        "dynamic_mean": dynamic,
        "excel_gate": excel_flags,
        "cvar_gate": cvar_flags,
        "action_correlations": corr_by_seed,
        "promote": bool(
            (
                excel_flags["beats_best_static_at_le_resource"]
                or cvar_flags["beats_best_static_at_le_resource"]
            )
            and dynamic["frac_std"] > 0.03
        ),
        "learning_signal": {
            "varies_actions": bool(dynamic["frac_std"] > 0.03),
            "any_seed_risk_conditioned": bool(risk_conditioned),
        },
        "trace_rows": trace_rows,
    }
    return case_summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="R1,R2,R3", help="Comma list: R1,R2,R3,CF1..CF30")
    ap.add_argument("--reward-mode", default="ReT_excel_plus_cvar")
    ap.add_argument("--cvar-alpha", type=float, default=0.2)
    ap.add_argument("--observation-version", default="v6")
    ap.add_argument("--risk-level", default="current")
    ap.add_argument("--phi", type=float, default=1.0)
    ap.add_argument("--psi", type=float, default=1.0)
    ap.add_argument("--holding-cost", type=float, default=0.0)
    ap.add_argument("--shift-cost", type=float, default=0.001)
    ap.add_argument("--max-steps", type=int, default=52)
    ap.add_argument("--step-size-hours", type=float, default=168.0)
    ap.add_argument("--fracs", default="0,0.05,0.10,0.125,0.15,0.25,0.50")
    ap.add_argument("--seeds", default="1")
    ap.add_argument("--n-envs", type=int, default=2)
    ap.add_argument("--timesteps", type=int, default=8000)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--eval-episodes", type=int, default=2)
    ap.add_argument("--eval-seed0", type=int, default=9100)
    ap.add_argument("--output", default="outputs/experiments/garrido_controlled_risk_probe")
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    cases = risk_cases(args.cases)
    summaries = []
    all_trace_rows: list[dict] = []
    for case in cases:
        print(f"\n=== CONTROLLED RISK CASE {case.name}: {','.join(case.enabled_risks)} ===", flush=True)
        summary = run_case(args, case)
        trace_rows = summary.pop("trace_rows", [])
        all_trace_rows.extend(trace_rows)
        summaries.append(summary)
        dyn = summary["dynamic_mean"]
        excel_best = summary["excel_gate"]["best_static_any_resource"]
        cvar_best = summary["cvar_gate"]["best_static_any_resource"]
        print(
            f"dynamic excel={dyn['excel']:.6f} cvar={dyn['cvar95_service_loss']:.3e} "
            f"resource={dyn['resource']:.3f} frac_std={dyn['frac_std']:.3f}"
        )
        print(
            f"best static Excel={excel_best['label']} {excel_best['excel']:.6f}; "
            f"best static CVaR={cvar_best['label']} {cvar_best['cvar95_service_loss']:.3e}; "
            f"promote={summary['promote']}"
        )

    result = {
        "args": vars(args),
        "cases": summaries,
        "promoted_cases": [s["case"]["name"] for s in summaries if s["promote"]],
    }
    (out / "summary.json").write_text(json.dumps(result, indent=2, default=float))
    if all_trace_rows:
        keys = sorted({k for row in all_trace_rows for k in row})
        with (out / "action_trace.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_trace_rows)
    print(f"\nWROTE {out / 'summary.json'}")
    if all_trace_rows:
        print(f"WROTE {out / 'action_trace.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
