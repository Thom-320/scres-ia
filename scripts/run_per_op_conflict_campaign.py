#!/usr/bin/env python3
"""Per-op conflict campaign: oracle-BC warm-start + PPO against static gate.

This runner consumes a `run_track_a_headroom_search.py` output directory. It
uses the gate's best per-regime static actions as behavior-cloning targets, then
fine-tunes PPO with VecNormalize and evaluates on the same campaign sequence.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_track_a_headroom_search import FAMILY_RISKS  # noqa: E402
from supply_chain.continuous_its_env import make_per_op_buffer_track_a_env  # noqa: E402
from supply_chain.episode_metrics import METRIC_KEYS, compute_episode_metrics  # noqa: E402


def parse_regime(name: str) -> tuple[str, float, float]:
    m = re.fullmatch(r"(.+)_phi([0-9.]+)_psi([0-9.]+)", name)
    if not m:
        raise ValueError(f"bad regime name: {name}")
    return m.group(1), float(m.group(2)), float(m.group(3))


def build_env(
    *,
    regime: str,
    reward_mode: str,
    max_steps: int,
    seed: int,
    holding_cost: float,
    cvar_alpha: float,
):
    family, phi, psi = parse_regime(regime)
    kwargs: dict[str, Any] = dict(
        reward_mode=reward_mode,
        observation_version="v6",
        risk_level="current",
        risk_frequency_multiplier=phi,
        risk_impact_multiplier=psi,
        stochastic_pt=False,
        max_steps=int(max_steps),
        step_size_hours=168.0,
        init_fracs=(0.0, 0.0, 0.0),
        risk_obs=True,
        holding_cost=float(holding_cost),
        shift_cost=0.001,
        ret_excel_cvar_alpha=float(cvar_alpha),
    )
    enabled = FAMILY_RISKS.get(family)
    if enabled is not None:
        kwargs["enabled_risks"] = enabled
    env = make_per_op_buffer_track_a_env(**kwargs)
    env.reset(seed=int(seed))
    return env


def read_gate(gate_dir: Path):
    summary = json.loads((gate_dir / "gate_summary.json").read_text())
    rows = []
    with (gate_dir / "static_runs.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            row["excel"] = float(row["excel"])
            row["resource"] = float(row["resource"])
            row["seed"] = int(row["seed"])
            row["action_tuple"] = tuple(float(x) for x in json.loads(row["action"]))
            rows.append(row)
    best_by_regime = summary["best_by_regime"]
    oracle_actions: dict[str, tuple[float, ...]] = {}
    for regime, item in best_by_regime.items():
        label = item["candidate"]
        match = next(r for r in rows if r["regime"] == regime and r["candidate"] == label)
        oracle_actions[regime] = match["action_tuple"]
    return summary, rows, oracle_actions


def unique_static_actions(static_rows: list[dict]) -> list[dict]:
    seen: dict[str, dict] = {}
    for row in static_rows:
        label = str(row["candidate"])
        if label not in seen:
            seen[label] = {
                "label": label,
                "action": tuple(float(x) for x in row["action_tuple"]),
                "gate_resource": float(row["resource"]),
            }
    return list(seen.values())


def eval_policy(regimes, act_fn, args, seed0: int) -> dict:
    excels, losses, resources, traces, panels = [], [], [], [], []
    for i, regime in enumerate(regimes):
        env = build_env(
            regime=regime,
            reward_mode=args.reward_mode,
            max_steps=args.max_steps,
            seed=seed0 + i,
            holding_cost=args.holding_cost,
            cvar_alpha=args.cvar_alpha,
        )
        obs, _ = env.reset(seed=seed0 + i)
        done = truncated = False
        ep_res, ep_actions = [], []
        while not (done or truncated):
            action = np.asarray(act_fn(obs, regime), dtype=np.float32).reshape(-1)
            obs, _r, done, truncated, info = env.step(action)
            ep_res.append(float(info.get("resource_composite", 0.0)))
            ep_actions.append(action)
        metrics = compute_episode_metrics(env.unwrapped.sim)
        panels.append(metrics)
        excels.append(float(metrics.get("ret_excel", 0.0)))
        losses.append(float(metrics.get("service_loss_auc_ration_hours", 0.0)))
        resources.append(float(np.nanmean(ep_res)) if ep_res else 0.0)
        action_mat = np.vstack(ep_actions) if ep_actions else np.zeros((0, 4), dtype=float)
        action_mean = np.mean(action_mat, axis=0).tolist() if len(action_mat) else []
        action_std = np.std(action_mat, axis=0).tolist() if len(action_mat) else []
        traces.append(
            {
                "regime": regime,
                "excel": excels[-1],
                "service_loss": losses[-1],
                "resource": resources[-1],
                "mean_action": action_mean,
                "std_action": action_std,
                "metrics": metrics,
            }
        )
        env.close()
    losses_sorted = sorted(losses)
    k = max(1, int(round(0.05 * len(losses_sorted))))
    metric_mean = aggregate_metric_panel(panels)
    return {
        "excel": float(np.mean(excels)),
        "cvar": float(np.mean(losses_sorted[-k:])),
        "resource": float(np.mean(resources)),
        "metrics_mean": metric_mean,
        "metrics_tail": tail_metric_panel(panels),
        "action_mean": aggregate_actions(traces, "mean_action", np.mean),
        "action_std_mean": aggregate_actions(traces, "std_action", np.mean),
        "by_regime": traces,
    }


def aggregate_metric_panel(panels: list[dict[str, float]]) -> dict[str, float]:
    out: dict[str, float] = {}
    if not panels:
        return out
    keys = sorted(set(METRIC_KEYS).union(*(p.keys() for p in panels)))
    for key in keys:
        vals = [float(p[key]) for p in panels if key in p and np.isfinite(float(p[key]))]
        if vals:
            out[key] = float(np.mean(vals))
    return out


def tail_metric_panel(panels: list[dict[str, float]]) -> dict[str, float]:
    """Episode-level tail summaries for metrics where tails matter."""
    out: dict[str, float] = {}
    if not panels:
        return out
    tail_specs = {
        "service_loss_auc_ration_hours": "high",
        "service_loss_auc_per_order": "high",
        "lost_rate": "high",
        "backorder_qty_final": "high",
        "ttr_p95": "high",
        "ctj_p99": "high",
        "rpj_p99": "high",
        "ret_excel": "low",
        "flow_fill_rate": "low",
        "fill_rate_on_time": "low",
    }
    for key, direction in tail_specs.items():
        vals = sorted(float(p[key]) for p in panels if key in p and np.isfinite(float(p[key])))
        if not vals:
            continue
        n = len(vals)
        if direction == "high":
            tail = vals[max(0, int(np.floor(0.95 * n)) - 1) :]
            out[f"{key}_p95"] = float(vals[min(n - 1, int(np.floor(0.95 * n)))])
            out[f"{key}_cvar95"] = float(np.mean(tail))
        else:
            tail = vals[: max(1, int(np.ceil(0.05 * n)))]
            out[f"{key}_p05"] = float(vals[max(0, int(np.floor(0.05 * n)) - 1)])
            out[f"{key}_cvar05"] = float(np.mean(tail))
    return out


def aggregate_actions(traces: list[dict], key: str, fn) -> list[float]:
    vals = [t.get(key, []) for t in traces if t.get(key)]
    if not vals:
        return []
    return np.asarray(fn(np.asarray(vals, dtype=float), axis=0), dtype=float).tolist()


def eval_static_frontier(regimes, static_rows, args, seed0: int) -> list[dict]:
    frontier = []
    for item in unique_static_actions(static_rows):
        action = np.asarray(item["action"], dtype=np.float32)
        result = eval_policy(regimes, lambda _obs, _regime, a=action: a, args, seed0)
        frontier.append(
            {
                "label": item["label"],
                "action": list(item["action"]),
                "excel": result["excel"],
                "cvar": result["cvar"],
                "resource": result["resource"],
                "metrics_mean": result.get("metrics_mean", {}),
                "metrics_tail": result.get("metrics_tail", {}),
                "action_mean": result.get("action_mean", []),
            }
        )
    return frontier


def pareto_verdict(dynamic: dict, static_frontier: list[dict]) -> dict:
    if not static_frontier:
        return {}
    best_excel = max(static_frontier, key=lambda r: r["excel"])
    eligible = [r for r in static_frontier if r["resource"] <= dynamic["resource"] + 1e-12]
    best_eligible = max(eligible, key=lambda r: r["excel"]) if eligible else None

    def dominates_static(row: dict, ref: dict) -> bool:
        return (
            row["excel"] >= ref["excel"] - 1e-12
            and row["cvar"] <= ref["cvar"] + 1e-12
            and row["resource"] <= ref["resource"] + 1e-12
            and (
                row["excel"] > ref["excel"] + 1e-12
                or row["cvar"] < ref["cvar"] - 1e-12
                or row["resource"] < ref["resource"] - 1e-12
            )
        )

    def dynamic_dominates(row: dict) -> bool:
        return (
            dynamic["excel"] >= row["excel"] - 1e-12
            and dynamic["cvar"] <= row["cvar"] + 1e-12
            and dynamic["resource"] <= row["resource"] + 1e-12
            and (
                dynamic["excel"] > row["excel"] + 1e-12
                or dynamic["cvar"] < row["cvar"] - 1e-12
                or dynamic["resource"] < row["resource"] - 1e-12
            )
        )

    dominated_by = [r for r in static_frontier if dominates_static(r, dynamic)]
    dominated = [r for r in static_frontier if dynamic_dominates(r)]
    return {
        "best_static_by_excel": best_excel,
        "best_static_at_or_below_dynamic_resource": best_eligible,
        "raw_ret_win": dynamic["excel"] > best_excel["excel"],
        "raw_ret_delta_vs_best_static": dynamic["excel"] - best_excel["excel"],
        "resource_constrained_ret_win": bool(
            best_eligible is not None and dynamic["excel"] > best_eligible["excel"]
        ),
        "resource_constrained_ret_delta": (
            dynamic["excel"] - best_eligible["excel"] if best_eligible is not None else None
        ),
        "pareto_non_dominated": len(dominated_by) == 0,
        "dominated_by_count": len(dominated_by),
        "dominates_count": len(dominated),
        "dominated_by": dominated_by[:10],
    }


def collect_bc(regimes, oracle_actions, args, seed0: int):
    obs_rows, act_rows = [], []
    for i, regime in enumerate(regimes):
        target = np.asarray(oracle_actions[regime], dtype=np.float32)
        env = build_env(
            regime=regime,
            reward_mode=args.reward_mode,
            max_steps=args.max_steps,
            seed=seed0 + i,
            holding_cost=args.holding_cost,
            cvar_alpha=args.cvar_alpha,
        )
        obs, _ = env.reset(seed=seed0 + i)
        done = truncated = False
        while not (done or truncated):
            obs_rows.append(np.asarray(obs, dtype=np.float32).copy())
            act_rows.append(target.copy())
            obs, _r, done, truncated, _info = env.step(target)
        env.close()
    return np.vstack(obs_rows).astype(np.float32), np.vstack(act_rows).astype(np.float32)


def bc_train(model: PPO, obs: np.ndarray, actions: np.ndarray, *, epochs: int, batch_size: int, seed: int):
    device = model.policy.device
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
    act_t = torch.as_tensor(actions, dtype=torch.float32, device=device)
    rng = np.random.default_rng(seed)

    def loss(idx=None):
        x = obs_t if idx is None else obs_t[idx]
        y = act_t if idx is None else act_t[idx]
        pred = model.policy.get_distribution(x).mode()
        return torch.nn.functional.mse_loss(pred, y)

    with torch.no_grad():
        initial = float(loss().detach().cpu())
    for _ in range(epochs):
        order = rng.permutation(len(obs))
        for start in range(0, len(obs), batch_size):
            idx = torch.as_tensor(order[start : start + batch_size], dtype=torch.long, device=device)
            l = loss(idx)
            model.policy.optimizer.zero_grad()
            l.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 0.5)
            model.policy.optimizer.step()
    with torch.no_grad():
        final = float(loss().detach().cpu())
    return {"bc_loss_initial": initial, "bc_loss_final": final, "bc_samples": int(len(obs))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gate-dir", required=True)
    ap.add_argument("--reward-mode", default="ReT_excel_delta")
    ap.add_argument("--seeds", default="1")
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--timesteps", type=int, default=20000)
    ap.add_argument("--bc-epochs", type=int, default=20)
    ap.add_argument("--bc-batch-size", type=int, default=128)
    ap.add_argument("--max-steps", type=int, default=52)
    ap.add_argument("--eval-seed0", type=int, default=9000)
    ap.add_argument("--holding-cost", type=float, default=0.02)
    ap.add_argument("--cvar-alpha", type=float, default=0.2)
    ap.add_argument("--eval-full-static-frontier", action="store_true")
    ap.add_argument("--output", default="outputs/experiments/per_op_conflict_campaign_2026-06-29")
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    summary, static_rows, oracle_actions = read_gate(Path(args.gate_dir))
    regimes = list(summary["best_by_regime"].keys())

    best_single = summary["best_single_constant"]
    oracle_excel = float(summary["oracle_excel"])
    robust_label = best_single["candidate"]
    robust_action = next(
        r["action_tuple"] for r in static_rows if r["candidate"] == robust_label
    )

    robust_eval = eval_policy(
        regimes,
        lambda _obs, _regime: np.asarray(robust_action, dtype=np.float32),
        args,
        args.eval_seed0,
    )
    oracle_eval = eval_policy(
        regimes,
        lambda _obs, regime: np.asarray(oracle_actions[regime], dtype=np.float32),
        args,
        args.eval_seed0,
    )

    learned = []
    bc_obs, bc_actions = collect_bc(regimes, oracle_actions, args, args.eval_seed0 + 1000)
    for seed in [int(s) for s in args.seeds.split(",") if s.strip()]:
        env_fns = []
        for i in range(args.n_envs):
            regime = regimes[i % len(regimes)]
            env_fns.append(lambda r=regime, s=seed + i: build_env(
                regime=r,
                reward_mode=args.reward_mode,
                max_steps=args.max_steps,
                seed=s,
                holding_cost=args.holding_cost,
                cvar_alpha=args.cvar_alpha,
            ))
        venv = VecNormalize(DummyVecEnv(env_fns), norm_obs=True, norm_reward=True, clip_reward=10.0)
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
        bc_stats = bc_train(
            model,
            bc_obs,
            bc_actions,
            epochs=args.bc_epochs,
            batch_size=args.bc_batch_size,
            seed=seed,
        )
        model.learn(total_timesteps=int(args.timesteps))
        result = eval_policy(
            regimes,
            lambda obs, _regime, m=model: m.predict(obs, deterministic=True)[0],
            args,
            args.eval_seed0,
        )
        result["seed"] = seed
        result["bc"] = bc_stats
        learned.append(result)
        venv.close()

    dynamic = {
        "excel": fmean(r["excel"] for r in learned),
        "cvar": fmean(r["cvar"] for r in learned),
        "resource": fmean(r["resource"] for r in learned),
        "metrics_mean": aggregate_metric_panel([r.get("metrics_mean", {}) for r in learned]),
        "metrics_tail": aggregate_metric_panel([r.get("metrics_tail", {}) for r in learned]),
        "action_mean": aggregate_actions(learned, "action_mean", np.mean),
        "action_std_mean": aggregate_actions(learned, "action_std_mean", np.mean),
    }
    static_frontier_eval = (
        eval_static_frontier(regimes, static_rows, args, args.eval_seed0)
        if args.eval_full_static_frontier
        else []
    )
    frontier_verdict = pareto_verdict(dynamic, static_frontier_eval)
    payload = {
        "args": vars(args),
        "gate_summary": summary,
        "regimes": regimes,
        "robust_static_from_gate": {"label": robust_label, "action": robust_action},
        "robust_eval": robust_eval,
        "oracle_excel_from_gate": oracle_excel,
        "oracle_eval": oracle_eval,
        "learned_per_seed": learned,
        "dynamic": dynamic,
        "static_frontier_eval": static_frontier_eval,
        "frontier_verdict": frontier_verdict,
        "raw_ret_win_vs_robust": dynamic["excel"] > robust_eval["excel"],
        "raw_ret_win_vs_oracle": dynamic["excel"] > oracle_eval["excel"],
    }
    (out / "summary.json").write_text(json.dumps(payload, indent=2, default=float))
    write_metrics_tables(out, payload)
    lines = [
        "# Per-Op Conflict Campaign",
        "",
        f"Robust static `{robust_label}` Excel={robust_eval['excel']:.6f}, resource={robust_eval['resource']:.3f}.",
        f"Oracle Excel={oracle_eval['excel']:.6f}, resource={oracle_eval['resource']:.3f}.",
        f"Dynamic Excel={dynamic['excel']:.6f}, CVaR={dynamic['cvar']:.3e}, resource={dynamic['resource']:.3f}.",
        "",
        f"Raw win vs robust: {payload['raw_ret_win_vs_robust']}",
        f"Raw win vs oracle: {payload['raw_ret_win_vs_oracle']}",
    ]
    if frontier_verdict:
        best = frontier_verdict["best_static_by_excel"]
        eligible = frontier_verdict["best_static_at_or_below_dynamic_resource"]
        lines.extend(
            [
                "",
                "## Full Static Frontier Eval",
                "",
                f"Best static `{best['label']}` Excel={best['excel']:.6f}, CVaR={best['cvar']:.3e}, resource={best['resource']:.3f}.",
                f"Raw ReT win vs full static: {frontier_verdict['raw_ret_win']} (delta={frontier_verdict['raw_ret_delta_vs_best_static']:+.6f}).",
                f"Pareto non-dominated: {frontier_verdict['pareto_non_dominated']} (dominated_by={frontier_verdict['dominated_by_count']}, dominates={frontier_verdict['dominates_count']}).",
            ]
        )
        if eligible is not None:
            lines.append(
                f"Best static at <= dynamic resource `{eligible['label']}` Excel={eligible['excel']:.6f}, resource={eligible['resource']:.3f}; "
                f"resource-constrained ReT win: {frontier_verdict['resource_constrained_ret_win']} "
                f"(delta={frontier_verdict['resource_constrained_ret_delta']:+.6f})."
            )
    (out / "report.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"WROTE {out}")
    return 0


def _flatten(prefix: str, d: dict) -> dict[str, float | str | bool]:
    out: dict[str, float | str | bool] = {}
    for key, value in d.items():
        if isinstance(value, dict):
            out.update(_flatten(f"{prefix}{key}.", value))
        elif isinstance(value, (int, float, str, bool)) or value is None:
            out[f"{prefix}{key}"] = value
    return out


def write_metrics_tables(out: Path, payload: dict) -> None:
    rows: list[dict[str, Any]] = []

    def add_row(policy: str, label: str, panel: dict) -> None:
        row: dict[str, Any] = {"policy": policy, "label": label}
        row.update(_flatten("", panel))
        rows.append(row)

    add_row("dynamic", "mean", payload["dynamic"])
    add_row("robust_static", payload["robust_static_from_gate"]["label"], payload["robust_eval"])
    add_row("oracle_static", "per_regime_oracle", payload["oracle_eval"])
    verdict = payload.get("frontier_verdict") or {}
    best = verdict.get("best_static_by_excel")
    if best:
        add_row("best_full_static", best.get("label", ""), best)
    eligible = verdict.get("best_static_at_or_below_dynamic_resource")
    if eligible:
        add_row("best_static_at_le_dynamic_resource", eligible.get("label", ""), eligible)

    keys = sorted({k for row in rows for k in row})
    with (out / "metrics_panel.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    frontier = payload.get("static_frontier_eval") or []
    if frontier:
        fkeys = sorted({k for row in frontier for k in _flatten("", row)})
        with (out / "static_frontier_metrics.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fkeys)
            writer.writeheader()
            for row in frontier:
                writer.writerow(_flatten("", row))


if __name__ == "__main__":
    raise SystemExit(main())
