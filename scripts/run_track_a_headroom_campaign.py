#!/usr/bin/env python3
"""PPO smoke on Track-A regimes that passed the static headroom gate.

This runner is intentionally downstream of ``run_track_a_headroom_search.py``.
It should only be used after a static-only gate finds positive oracle-vs-static
headroom and action changes across regimes.
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

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_track_a_headroom_search import (  # noqa: E402
    FAMILY_RISKS,
    SHIFT_SIGS,
    continuous_candidates,
    parse_csv_floats,
)
from supply_chain.continuous_its_env import make_continuous_its_track_a_env  # noqa: E402
from supply_chain.episode_metrics import compute_episode_metrics  # noqa: E402


@dataclass(frozen=True)
class CampaignPhase:
    name: str
    family: str
    phi: float
    psi: float
    enabled_risks: tuple[str, ...] | None


def parse_phase(raw: str) -> CampaignPhase:
    match = re.fullmatch(
        r"(R1|R2|R3|R24|mixed)_phi([0-9]+(?:\.[0-9]+)?)_psi([0-9]+(?:\.[0-9]+)?)",
        raw.strip(),
    )
    if not match:
        raise ValueError(
            f"Invalid phase {raw!r}; expected e.g. R1_phi4_psi1.5 or mixed_phi8_psi2"
        )
    family = match.group(1)
    return CampaignPhase(
        name=raw.strip(),
        family=family,
        phi=float(match.group(2)),
        psi=float(match.group(3)),
        enabled_risks=FAMILY_RISKS[family],
    )


def parse_phases(raw: str) -> list[CampaignPhase]:
    phases = [parse_phase(x) for x in raw.split(",") if x.strip()]
    if not phases:
        raise ValueError("Expected at least one campaign phase")
    return phases


def listed_sequence(phases: list[CampaignPhase], n_blocks: int) -> list[CampaignPhase]:
    return [phases[i % len(phases)] for i in range(int(n_blocks))]


def markov_sequence(
    phases: list[CampaignPhase], *, n_blocks: int, rho: float, seed: int
) -> list[CampaignPhase]:
    if n_blocks <= 0:
        return []
    if len(phases) == 1:
        return phases * n_blocks
    rho = float(rho)
    lo = 1.0 / len(phases)
    if not (lo - 1e-9 <= rho <= 1.0 + 1e-9):
        raise ValueError(f"rho={rho} outside [{lo}, 1] for {len(phases)} phases")
    rng = np.random.default_rng(int(seed))
    idx = int(rng.integers(len(phases)))
    seq = [phases[idx]]
    off = (1.0 - rho) / (len(phases) - 1)
    for _ in range(1, n_blocks):
        probs = np.full(len(phases), off)
        probs[idx] = rho
        idx = int(rng.choice(len(phases), p=probs))
        seq.append(phases[idx])
    return seq


def build_env(
    *,
    phase: CampaignPhase,
    reward: str,
    obs_v: str,
    max_steps: int,
    init_frac: float,
    holding_cost: float,
    shift_cost: float,
    cvar_alpha: float,
    step_size_hours: float,
    seed: int | None = None,
):
    kwargs = dict(
        reward_mode=reward,
        observation_version=obs_v,
        risk_level="current",
        risk_frequency_multiplier=phase.phi,
        risk_impact_multiplier=phase.psi,
        stochastic_pt=False,
        max_steps=int(max_steps),
        step_size_hours=float(step_size_hours),
        init_frac=float(init_frac),
        risk_obs=True,
        holding_cost=float(holding_cost),
        shift_cost=float(shift_cost),
        ret_excel_cvar_alpha=float(cvar_alpha),
    )
    if phase.enabled_risks is not None:
        kwargs["enabled_risks"] = phase.enabled_risks
    env = make_continuous_its_track_a_env(**kwargs)
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
    excels, service_loss, resources, flow, lost, fracs = [], [], [], [], [], []
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
            frac = float(info.get("continuous_its_frac", np.nan))
            ep_resource.append(float(info.get("resource_composite", np.nan)))
            if frac == frac:
                fracs.append(frac)
            if trace_label:
                row = {
                    "policy": trace_label,
                    "block": block_idx,
                    "phase": phase.name,
                    "step": step,
                    "action_frac": float(action[0]) if action.size > 0 else float("nan"),
                    "action_shift_signal": float(action[1]) if action.size > 1 else float("nan"),
                    "applied_frac": frac,
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
        "frac_std": float(np.std(fracs)) if fracs else 0.0,
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
        s
        for s in statics
        if (better(s[key], dynamic[key]) or s[key] == dynamic[key])
        and s["resource"] <= dynamic["resource"] + 1e-9
        and (better(s[key], dynamic[key]) or s["resource"] < dynamic["resource"] - 1e-9)
    ]
    dominated = [
        s
        for s in statics
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
    phase_names = sorted({p.name for p in sequence})
    oracle_by_phase = {}
    for phase in phase_names:
        best = max(statics, key=lambda s, p=phase: s["by_phase"].get(p, {}).get("excel", float("-inf")))
        oracle_by_phase[phase] = {
            "label": best["label"],
            "excel": best["by_phase"].get(phase, {}).get("excel", float("nan")),
            "resource": best["by_phase"].get(phase, {}).get("resource", float("nan")),
        }
    robust = max(statics, key=lambda row: row["excel"])
    oracle_excel = float(np.mean([v["excel"] for v in oracle_by_phase.values()]))
    return {
        "robust_static": {
            "label": robust["label"],
            "excel": robust["excel"],
            "cvar": robust["cvar"],
            "resource": robust["resource"],
        },
        "oracle_by_phase": oracle_by_phase,
        "oracle_excel": oracle_excel,
        "oracle_minus_robust_static": oracle_excel - float(robust["excel"]),
        "argmax_diversity": len({v["label"] for v in oracle_by_phase.values()}),
    }


def oracle_action_map(statics: list[dict], sequence: list[CampaignPhase]) -> dict[str, tuple[float, ...]]:
    phase_names = sorted({p.name for p in sequence})
    out: dict[str, tuple[float, ...]] = {}
    for phase in phase_names:
        best = max(statics, key=lambda s, p=phase: s["by_phase"].get(p, {}).get("excel", float("-inf")))
        out[phase] = tuple(float(x) for x in best["action"])
    return out


def collect_behavior_cloning_data(
    *,
    sequence: list[CampaignPhase],
    build_fn: Callable[[CampaignPhase, int], object],
    oracle_actions: dict[str, tuple[float, ...]],
    seed0: int,
) -> tuple[np.ndarray, np.ndarray]:
    obs_rows: list[np.ndarray] = []
    action_rows: list[np.ndarray] = []
    for block_idx, phase in enumerate(sequence):
        target = np.asarray(oracle_actions[phase.name], dtype=np.float32)
        env = build_fn(phase, seed0 + block_idx)
        obs, _info = env.reset(seed=seed0 + block_idx)
        done = truncated = False
        while not (done or truncated):
            obs_rows.append(np.asarray(obs, dtype=np.float32).copy())
            action_rows.append(target.copy())
            obs, _reward, done, truncated, _info = env.step(target)
        env.close()
    if not obs_rows:
        raise RuntimeError("No behavior-cloning data collected")
    return np.vstack(obs_rows).astype(np.float32), np.vstack(action_rows).astype(np.float32)


def behavior_clone_policy(
    model: PPO,
    obs: np.ndarray,
    actions: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    seed: int,
) -> dict[str, float]:
    if epochs <= 0:
        return {"bc_epochs": 0, "bc_loss_initial": float("nan"), "bc_loss_final": float("nan")}
    rng = np.random.default_rng(int(seed))
    device = model.policy.device
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
    actions_t = torch.as_tensor(actions, dtype=torch.float32, device=device)

    def loss_for(indices: np.ndarray | None = None) -> torch.Tensor:
        x = obs_t if indices is None else obs_t[indices]
        y = actions_t if indices is None else actions_t[indices]
        pred = model.policy.get_distribution(x).mode()
        return torch.nn.functional.mse_loss(pred, y)

    with torch.no_grad():
        initial = float(loss_for().detach().cpu().item())

    n = int(obs_t.shape[0])
    bs = max(1, min(int(batch_size), n))
    model.policy.set_training_mode(True)
    for _epoch in range(int(epochs)):
        order = rng.permutation(n)
        for start in range(0, n, bs):
            idx_np = order[start : start + bs]
            idx = torch.as_tensor(idx_np, dtype=torch.long, device=device)
            loss = loss_for(idx)
            model.policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), max_norm=0.5)
            model.policy.optimizer.step()

    with torch.no_grad():
        final = float(loss_for().detach().cpu().item())
    return {
        "bc_epochs": int(epochs),
        "bc_batch_size": int(bs),
        "bc_n_samples": int(n),
        "bc_loss_initial": initial,
        "bc_loss_final": final,
    }


def action_correlations(trace_rows: list[dict], *, target: str = "applied_frac", top_k: int = 12):
    if not trace_rows:
        return []
    vals = np.asarray([float(r.get(target, np.nan)) for r in trace_rows], dtype=float)
    if vals.size < 3 or np.nanstd(vals) <= 1e-12:
        return []
    rows = []
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("static-gate", "train"), default="train")
    ap.add_argument("--reward-mode", default="ReT_excel_plus_cvar")
    ap.add_argument("--observation-version", default="v6")
    ap.add_argument(
        "--campaign-phases",
        default=(
            "R1_phi1_psi1,R1_phi4_psi1,R1_phi8_psi1,"
            "R2_phi4_psi1,R2_phi8_psi2,mixed_phi4_psi1,mixed_phi8_psi2"
        ),
    )
    ap.add_argument("--sequence-mode", choices=("listed", "markov"), default="listed")
    ap.add_argument("--rho", type=float, default=0.75)
    ap.add_argument("--campaign-seed", type=int, default=909)
    ap.add_argument("--init-frac", type=float, default=0.1)
    ap.add_argument("--holding-cost", type=float, default=0.0)
    ap.add_argument("--shift-cost", type=float, default=0.001)
    ap.add_argument("--cvar-alpha", type=float, default=0.2)
    ap.add_argument("--seeds", default="1")
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--timesteps", type=int, default=20000)
    ap.add_argument("--bc-epochs", type=int, default=0)
    ap.add_argument("--bc-batch-size", type=int, default=128)
    ap.add_argument("--bc-seed0", type=int, default=8000)
    ap.add_argument("--eval-blocks", type=int, default=14)
    ap.add_argument("--eval-seed0", type=int, default=9000)
    ap.add_argument("--max-steps", type=int, default=52)
    ap.add_argument("--step-size-hours", type=float, default=168.0)
    ap.add_argument("--fracs", default="0,0.05,0.10,0.15,0.25,0.50")
    ap.add_argument("--shifts", default="1,2,3")
    ap.add_argument("--output", default="outputs/experiments/track_a_headroom_campaign_2026-06-29")
    args = ap.parse_args()

    phases = parse_phases(args.campaign_phases)
    if args.sequence_mode == "listed":
        sequence = listed_sequence(phases, args.eval_blocks)
    else:
        sequence = markov_sequence(
            phases, n_blocks=args.eval_blocks, rho=args.rho, seed=args.campaign_seed
        )
    candidates = continuous_candidates(parse_csv_floats(args.fracs), [int(x) for x in parse_csv_floats(args.shifts)])
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    def env_builder(phase: CampaignPhase, seed: int | None = None):
        return build_env(
            phase=phase,
            reward=args.reward_mode,
            obs_v=args.observation_version,
            max_steps=args.max_steps,
            init_frac=args.init_frac,
            holding_cost=args.holding_cost,
            shift_cost=args.shift_cost,
            cvar_alpha=args.cvar_alpha,
            step_size_hours=args.step_size_hours,
            seed=seed,
        )

    statics = []
    for cand in candidates:
        result = eval_on_sequence(
            sequence=sequence,
            build_fn=lambda phase, seed, c=cand: build_env(
                phase=phase,
                reward=args.reward_mode,
                obs_v=args.observation_version,
                max_steps=args.max_steps,
                init_frac=float(c.action[0]),
                holding_cost=args.holding_cost,
                shift_cost=args.shift_cost,
                cvar_alpha=args.cvar_alpha,
                step_size_hours=args.step_size_hours,
                seed=seed,
            ),
            act_fn=lambda _obs, a=np.asarray(cand.action, dtype=np.float32): a,
            seed0=args.eval_seed0,
        )
        result.update({"label": cand.label, "action": cand.action, "static_resource": cand.resource})
        statics.append(result)

    learned = []
    traces = []
    corr_by_seed = []
    oracle_actions = oracle_action_map(statics, sequence)
    bc_dataset = None
    if args.mode == "train" and args.bc_epochs > 0:
        bc_dataset = collect_behavior_cloning_data(
            sequence=sequence,
            build_fn=env_builder,
            oracle_actions=oracle_actions,
            seed0=args.bc_seed0,
        )
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
                n_steps=min(512, args.max_steps * 4),
                batch_size=64,
                learning_rate=3e-4,
                n_epochs=10,
            )
            bc_stats = {"bc_epochs": 0}
            if bc_dataset is not None:
                bc_obs, bc_actions = bc_dataset
                bc_stats = behavior_clone_policy(
                    model,
                    bc_obs,
                    bc_actions,
                    epochs=args.bc_epochs,
                    batch_size=args.bc_batch_size,
                    seed=seed,
                )
            model.learn(total_timesteps=int(args.timesteps))
            result = eval_on_sequence(
                sequence=sequence,
                build_fn=env_builder,
                act_fn=lambda obs, m=model: m.predict(obs, deterministic=True)[0],
                seed0=args.eval_seed0,
                trace_label=f"learned_seed{seed}",
            )
            seed_trace = result.pop("trace_rows", [])
            corr_by_seed.append({"seed": seed, "top_frac_correlations": action_correlations(seed_trace)})
            traces.extend({"seed": seed, **row} for row in seed_trace)
            result["seed"] = seed
            result["behavior_cloning"] = bc_stats
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
            "frac_std": float(np.nanmean([x["frac_std"] for x in learned])),
        }
        verdict = {
            "excel_pareto": pareto(dynamic, statics, "excel", higher_better=True),
            "cvar_pareto": pareto(dynamic, statics, "cvar", higher_better=False),
        }

    summary = {
        "args": vars(args),
        "phases": [asdict(p) for p in phases],
        "eval_sequence": [p.name for p in sequence],
        "static_candidates": len(statics),
        "statics": statics,
        "static_oracle": summarize_oracle(statics, sequence),
        "oracle_actions": {k: list(v) for k, v in oracle_actions.items()},
        "learned_per_seed": learned,
        "dynamic": dynamic,
        "action_correlations": corr_by_seed,
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
    print(f"\n=== TRACK A HEADROOM CAMPAIGN ({args.mode}, {args.reward_mode}) ===")
    print(f"sequence={summary['eval_sequence']}")
    print(
        f"best static Excel: {best_excel['label']} excel={best_excel['excel']:.6f} "
        f"res={best_excel['resource']:.3f}"
    )
    print(
        f"best static CVaR : {best_cvar['label']} cvar={best_cvar['cvar']:.3e} "
        f"res={best_cvar['resource']:.3f}"
    )
    print(f"oracle: {summary['static_oracle']}")
    if dynamic:
        print(
            f"dynamic: excel={dynamic['excel']:.6f} cvar={dynamic['cvar']:.3e} "
            f"flow={dynamic['flow_fill']:.3f} lost={dynamic['lost_rate']:.3f} "
            f"res={dynamic['resource']:.3f} frac_std={dynamic['frac_std']:.3f}"
        )
        print(f"Excel Pareto: {verdict['excel_pareto']}")
        print(f"CVaR  Pareto: {verdict['cvar_pareto']}")
    print(f"WROTE {out}/summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
