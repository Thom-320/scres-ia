#!/usr/bin/env python3
"""Non-stationary R2 campaign runner for the per-op Track-A action contract.

The stationary war lane was falsified by a dense static frontier. This runner
tests the structural opening where RL can matter with the same Garrido decision
families: a persistent calm <-> R2-stress campaign where the optimal Op9 buffer
is expected to move over time.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from supply_chain.continuous_its_env import make_per_op_buffer_track_a_env
from supply_chain.episode_metrics import compute_episode_metrics

SHIFT_SIGS = {1: -1.0, 2: 0.0, 3: 1.0}
DEFAULT_R2_RISKS = ("R22", "R23", "R24")


@dataclass(frozen=True)
class CampaignPhase:
    name: str
    risk_level: str
    phi: float
    psi: float
    enabled_risks: tuple[str, ...]


def parse_float_list(raw: str) -> list[float]:
    vals = sorted({round(float(x), 4) for x in raw.split(",") if x.strip()})
    if not vals:
        raise ValueError("Expected at least one numeric value")
    return vals


def parse_str_tuple(raw: str) -> tuple[str, ...]:
    vals = tuple(x.strip() for x in raw.split(",") if x.strip())
    if not vals:
        raise ValueError("Expected at least one comma-separated value")
    return vals


def parse_init_fracs(raw: str) -> list[float]:
    vals = [float(x) for x in raw.split(",") if x.strip()]
    if len(vals) == 1:
        vals *= 3
    if len(vals) != 3:
        raise ValueError("--init-fracs must be one value or three comma-separated values")
    return [float(np.clip(x, 0.0, 1.0)) for x in vals]


def phase_from_name(name: str, *, enabled_risks: tuple[str, ...], psi: float) -> CampaignPhase:
    name = name.strip()
    if name == "calm":
        return CampaignPhase(name, "current", 1.0, 1.0, enabled_risks)
    match = re.fullmatch(r"r2_phi([0-9]+(?:\.[0-9]+)?)", name)
    if match:
        return CampaignPhase(name, "current", float(match.group(1)), float(psi), enabled_risks)
    raise ValueError(f"Unknown campaign phase {name!r}; use calm or r2_phi<multiplier>.")


def campaign_sequence(phases: list[CampaignPhase], *, n_blocks: int, rho: float, seed: int) -> list[CampaignPhase]:
    if n_blocks <= 0:
        return []
    if not phases:
        raise ValueError("phases must be non-empty")
    rho = float(rho)
    lo = 1.0 / len(phases)
    if not (lo - 1e-9 <= rho <= 1.0 + 1e-9):
        raise ValueError(f"rho={rho} outside [{lo}, 1.0] for {len(phases)} phases")
    rng = np.random.default_rng(int(seed))
    idx = int(rng.integers(len(phases)))
    seq = [phases[idx]]
    if len(phases) == 1:
        return seq * n_blocks
    off = (1.0 - rho) / (len(phases) - 1)
    for _ in range(1, n_blocks):
        probs = np.full(len(phases), off)
        probs[idx] = rho
        idx = int(rng.choice(len(phases), p=probs))
        seq.append(phases[idx])
    return seq


def static_candidates(
    op3_fracs: list[float], op5_fracs: list[float], op9_fracs: list[float]
) -> list[tuple[float, float, float, int]]:
    return [
        (op3, op5, op9, shift)
        for op3, op5, op9 in itertools.product(op3_fracs, op5_fracs, op9_fracs)
        for shift in (1, 2, 3)
    ]


def build_env(
    *,
    phase: CampaignPhase,
    reward: str,
    obs_v: str,
    max_steps: int,
    init_fracs: list[float],
    holding_cost: float,
    shift_cost: float,
    cvar_alpha: float,
    step_size_hours: float,
    seed: int | None = None,
):
    env = make_per_op_buffer_track_a_env(
        reward_mode=reward,
        observation_version=obs_v,
        risk_level=phase.risk_level,
        risk_frequency_multiplier=phase.phi,
        risk_impact_multiplier=phase.psi,
        enabled_risks=phase.enabled_risks,
        stochastic_pt=False,
        max_steps=int(max_steps),
        step_size_hours=float(step_size_hours),
        init_fracs=init_fracs,
        risk_obs=True,
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


def eval_on_sequence(
    *,
    sequence: list[CampaignPhase],
    build_fn: Callable[[CampaignPhase, int], object],
    act_fn: Callable[[np.ndarray], np.ndarray],
    seed0: int,
    trace_label: str | None = None,
) -> dict:
    excels, service_loss, resources, flow, lost = [], [], [], [], []
    by_phase: dict[str, list[dict]] = {}
    trace_rows: list[dict] = []
    for block_idx, phase in enumerate(sequence):
        env = build_fn(phase, seed0 + block_idx)
        field_names = list(getattr(env, "obs_field_names", []))
        obs, _info = env.reset(seed=seed0 + block_idx)
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
                    "block": block_idx,
                    "phase": phase.name,
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
        block_result = {
            "excel": float(metrics.get("ret_excel", np.nan)),
            "service_loss": float(metrics.get("service_loss_auc_ration_hours", np.nan)),
            "flow_fill": float(metrics.get("flow_fill_rate", np.nan)),
            "lost_rate": float(metrics.get("lost_rate", np.nan)),
            "resource": float(np.nanmean(ep_resource)),
        }
        by_phase.setdefault(phase.name, []).append(block_result)
        excels.append(block_result["excel"])
        service_loss.append(block_result["service_loss"])
        flow.append(block_result["flow_fill"])
        lost.append(block_result["lost_rate"])
        resources.append(block_result["resource"])
        env.close()
    out = {
        "excel": float(np.nanmean(excels)),
        "cvar": _cvar(service_loss),
        "flow_fill": float(np.nanmean(flow)),
        "lost_rate": float(np.nanmean(lost)),
        "resource": float(np.nanmean(resources)),
        "by_phase": {
            name: {
                "excel": float(np.nanmean([r["excel"] for r in rows])),
                "cvar": _cvar([r["service_loss"] for r in rows]),
                "flow_fill": float(np.nanmean([r["flow_fill"] for r in rows])),
                "lost_rate": float(np.nanmean([r["lost_rate"] for r in rows])),
                "resource": float(np.nanmean([r["resource"] for r in rows])),
                "n_blocks": len(rows),
            }
            for name, rows in by_phase.items()
        },
    }
    if trace_label:
        out["trace_rows"] = trace_rows
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


def summarize_oracle(statics: list[dict], sequence: list[CampaignPhase]) -> dict:
    by_label = {s["label"]: s for s in statics}
    labels = list(by_label)
    phase_names = sorted({p.name for p in sequence})
    oracle_by_phase = {}
    for phase in phase_names:
        best = max(labels, key=lambda label: by_label[label]["by_phase"].get(phase, {}).get("excel", float("-inf")))
        oracle_by_phase[phase] = {
            "label": best,
            "excel": by_label[best]["by_phase"].get(phase, {}).get("excel", float("nan")),
            "resource": by_label[best]["by_phase"].get(phase, {}).get("resource", float("nan")),
        }
    robust = max(statics, key=lambda row: row["excel"])
    return {
        "robust_static": {"label": robust["label"], "excel": robust["excel"], "resource": robust["resource"]},
        "oracle_by_phase": oracle_by_phase,
        "argmax_diversity": len({v["label"] for v in oracle_by_phase.values()}),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("static-gate", "train"), default="train")
    ap.add_argument("--reward-mode", default="ReT_excel_delta")
    ap.add_argument("--observation-version", default="v6")
    ap.add_argument("--campaign-phases", default="calm,r2_phi4")
    ap.add_argument(
        "--sequence-mode",
        choices=("markov", "listed"),
        default="markov",
        help="listed: replay --campaign-phases in order once/repeated; markov: persistent campaign tape",
    )
    ap.add_argument("--rho", type=float, default=0.85)
    ap.add_argument("--campaign-seed", type=int, default=909)
    ap.add_argument("--enabled-risks", default="R22,R23,R24")
    ap.add_argument("--psi", type=float, default=1.0)
    ap.add_argument("--init-fracs", default="1,1,1")
    ap.add_argument("--holding-cost", type=float, default=0.0003)
    ap.add_argument("--shift-cost", type=float, default=0.001)
    ap.add_argument("--cvar-alpha", type=float, default=0.2)
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--timesteps", type=int, default=30000)
    ap.add_argument("--eval-blocks", type=int, default=8)
    ap.add_argument("--eval-seed0", type=int, default=9000)
    ap.add_argument("--max-steps", type=int, default=104)
    ap.add_argument("--step-size-hours", type=float, default=168.0)
    ap.add_argument("--op3-fracs", default="0,0.05,0.1")
    ap.add_argument("--op5-fracs", default="0,0.05,0.1")
    ap.add_argument("--op9-fracs", default="0,0.05,0.1,0.15,0.2,0.25,0.3,0.5")
    ap.add_argument(
        "--static-init-mode",
        choices=("match", "fixed"),
        default="match",
        help="match: static preposition equals its constant action; fixed: every static gets --init-fracs",
    )
    ap.add_argument("--output", default="outputs/experiments/per_op_r2_campaign_2026-06-28")
    args = ap.parse_args()

    enabled = parse_str_tuple(args.enabled_risks)
    phases = [
        phase_from_name(name, enabled_risks=enabled, psi=args.psi)
        for name in parse_str_tuple(args.campaign_phases)
    ]
    if args.sequence_mode == "listed":
        eval_sequence = [phases[i % len(phases)] for i in range(args.eval_blocks)]
    else:
        eval_sequence = campaign_sequence(
            phases, n_blocks=args.eval_blocks, rho=args.rho, seed=args.campaign_seed
        )
    init_fracs = parse_init_fracs(args.init_fracs)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    def env_builder(phase: CampaignPhase, seed: int | None = None):
        return build_env(
            phase=phase,
            reward=args.reward_mode,
            obs_v=args.observation_version,
            max_steps=args.max_steps,
            init_fracs=init_fracs,
            holding_cost=args.holding_cost,
            shift_cost=args.shift_cost,
            cvar_alpha=args.cvar_alpha,
            step_size_hours=args.step_size_hours,
            seed=seed,
        )

    statics = []
    for op3, op5, op9, shift in static_candidates(
        parse_float_list(args.op3_fracs),
        parse_float_list(args.op5_fracs),
        parse_float_list(args.op9_fracs),
    ):
        sig = SHIFT_SIGS[shift]
        label = f"op3_{op3:g}_op5_{op5:g}_op9_{op9:g}_S{shift}"
        static_init_fracs = [op3, op5, op9] if args.static_init_mode == "match" else init_fracs

        def static_builder(
            phase: CampaignPhase,
            seed: int | None = None,
            fracs: list[float] = static_init_fracs,
        ):
            return build_env(
                phase=phase,
                reward=args.reward_mode,
                obs_v=args.observation_version,
                max_steps=args.max_steps,
                init_fracs=fracs,
                holding_cost=args.holding_cost,
                shift_cost=args.shift_cost,
                cvar_alpha=args.cvar_alpha,
                step_size_hours=args.step_size_hours,
                seed=seed,
            )

        result = eval_on_sequence(
            sequence=eval_sequence,
            build_fn=static_builder,
            act_fn=lambda _obs, a=np.array([op3, op5, op9, sig], dtype=np.float32): a,
            seed0=args.eval_seed0,
        )
        result.update({"label": label, "op3_frac": op3, "op5_frac": op5, "op9_frac": op9, "shift": shift})
        statics.append(result)

    learned = []
    traces = []
    if args.mode == "train":
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
        for seed in seeds:
            train_envs = []
            for i in range(args.n_envs):
                phase = phases[i % len(phases)]
                train_envs.append(lambda p=phase, s=seed + i: env_builder(p, s))
            venv = DummyVecEnv(train_envs)
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
            result = eval_on_sequence(
                sequence=eval_sequence,
                build_fn=env_builder,
                act_fn=lambda obs, m=model: m.predict(obs, deterministic=True)[0],
                seed0=args.eval_seed0,
                trace_label=f"learned_seed{seed}",
            )
            seed_trace = result.pop("trace_rows", [])
            traces.extend({"seed": seed, **row} for row in seed_trace)
            result["seed"] = seed
            learned.append(result)

    dynamic = None
    verdict = {}
    if learned:
        dynamic = {
            "excel": float(np.nanmean([x["excel"] for x in learned])),
            "cvar": float(np.nanmean([x["cvar"] for x in learned])),
            "flow_fill": float(np.nanmean([x["flow_fill"] for x in learned])),
            "lost_rate": float(np.nanmean([x["lost_rate"] for x in learned])),
            "resource": float(np.nanmean([x["resource"] for x in learned])),
        }
        verdict = {
            "excel_pareto": pareto(dynamic, statics, "excel", higher_better=True),
            "cvar_pareto": pareto(dynamic, statics, "cvar", higher_better=False),
        }

    summary = {
        "args": vars(args),
        "phases": [asdict(p) for p in phases],
        "eval_sequence": [p.name for p in eval_sequence],
        "static_candidates": len(statics),
        "statics": statics,
        "static_oracle": summarize_oracle(statics, eval_sequence),
        "learned_per_seed": learned,
        "dynamic": dynamic,
        **verdict,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    if traces:
        keys = sorted({k for row in traces for k in row})
        with (out / "action_trace.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(traces)

    best_excel = max(statics, key=lambda s: s["excel"])
    best_cvar = min(statics, key=lambda s: s["cvar"])
    print(f"\n=== PER-OP R2 CAMPAIGN ({args.mode}, {args.reward_mode}) ===")
    print(f"sequence={summary['eval_sequence']}")
    print(f"best static Excel: {best_excel['label']} excel={best_excel['excel']:.6f} res={best_excel['resource']:.3f}")
    print(f"best static CVaR : {best_cvar['label']} cvar={best_cvar['cvar']:.3e} res={best_cvar['resource']:.3f}")
    print(f"oracle: {summary['static_oracle']}")
    if dynamic:
        print(
            f"dynamic: excel={dynamic['excel']:.6f} cvar={dynamic['cvar']:.3e} "
            f"flow={dynamic['flow_fill']:.3f} lost={dynamic['lost_rate']:.3f} res={dynamic['resource']:.3f}"
        )
        print(f"Excel Pareto: {verdict['excel_pareto']}")
        print(f"CVaR  Pareto: {verdict['cvar_pareto']}")
    print(f"WROTE {out}/summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
