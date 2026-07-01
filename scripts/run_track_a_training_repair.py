#!/usr/bin/env python3
"""Track-A training repair: teacher BC + checkpoint selection + optional MultiDiscrete.

This runner is intentionally conservative. It first re-scores the static
frontier under calibration CRN, uses the best static/oracle as a teacher, then
fine-tunes PPO while selecting the best checkpoint instead of trusting the final
model. The goal is to separate "PPO failed to find the plateau" from "there is
no dynamic headroom beyond the full static frontier."
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_per_op_conflict_campaign import (  # noqa: E402
    aggregate_actions,
    aggregate_metric_panel,
    parse_regime,
    read_gate,
    tail_metric_panel,
    unique_static_actions,
)
from scripts.run_track_a_headroom_search import FAMILY_RISKS  # noqa: E402
from supply_chain.continuous_its_env import (  # noqa: E402
    make_per_op_buffer_multidiscrete_track_a_env,
    make_per_op_buffer_track_a_env,
)
from supply_chain.episode_metrics import compute_episode_metrics  # noqa: E402


DEFAULT_FRAC_GRID = (0.0, 0.05, 0.10, 0.15, 0.25, 0.50)


def parse_csv_floats(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def build_env(
    *,
    regime: str,
    reward_mode: str,
    max_steps: int,
    seed: int,
    holding_cost: float,
    cvar_alpha: float,
    action_mode: str,
    frac_grid: list[float],
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
    if action_mode == "multidiscrete":
        kwargs["frac_grid"] = frac_grid
        env = make_per_op_buffer_multidiscrete_track_a_env(**kwargs)
    elif action_mode == "continuous":
        env = make_per_op_buffer_track_a_env(**kwargs)
    else:
        raise ValueError(f"unknown action_mode={action_mode}")
    env.reset(seed=int(seed))
    return env


def encode_action_for_env(env: Any, continuous_action: Any, action_mode: str) -> np.ndarray:
    action = np.asarray(continuous_action, dtype=np.float32).reshape(-1)
    if action_mode == "multidiscrete":
        if not hasattr(env, "encode_continuous_action"):
            raise TypeError("multidiscrete env is missing encode_continuous_action")
        return np.asarray(env.encode_continuous_action(action), dtype=np.int64)
    return action.astype(np.float32)


def action_to_continuous(env: Any, action: Any, action_mode: str) -> np.ndarray:
    if action_mode == "multidiscrete":
        return np.asarray(env.decode_action(action), dtype=np.float32)
    return np.asarray(action, dtype=np.float32).reshape(-1)


def eval_policy(regimes: list[str], act_fn, args, seed0: int, *, action_mode: str) -> dict:
    excels, losses, resources, traces, panels = [], [], [], [], []
    for i, regime in enumerate(regimes):
        env = build_env(
            regime=regime,
            reward_mode=args.reward_mode,
            max_steps=args.max_steps,
            seed=seed0 + i,
            holding_cost=args.holding_cost,
            cvar_alpha=args.cvar_alpha,
            action_mode=action_mode,
            frac_grid=args.frac_grid,
        )
        obs, _ = env.reset(seed=seed0 + i)
        done = truncated = False
        ep_res, ep_actions = [], []
        while not (done or truncated):
            action = act_fn(obs, regime, env)
            obs, _r, done, truncated, info = env.step(action)
            ep_res.append(float(info.get("resource_composite", 0.0)))
            ep_actions.append(action_to_continuous(env, action, action_mode))
        metrics = compute_episode_metrics(env.unwrapped.sim)
        panels.append(metrics)
        excels.append(float(metrics.get("ret_excel", 0.0)))
        losses.append(float(metrics.get("service_loss_auc_ration_hours", 0.0)))
        resources.append(float(np.nanmean(ep_res)) if ep_res else 0.0)
        action_mat = np.vstack(ep_actions) if ep_actions else np.zeros((0, 4), dtype=float)
        traces.append(
            {
                "regime": regime,
                "excel": excels[-1],
                "service_loss": losses[-1],
                "resource": resources[-1],
                "mean_action": np.mean(action_mat, axis=0).tolist() if len(action_mat) else [],
                "std_action": np.std(action_mat, axis=0).tolist() if len(action_mat) else [],
                "metrics": metrics,
            }
        )
        env.close()
    losses_sorted = sorted(losses)
    k = max(1, int(round(0.05 * len(losses_sorted))))
    return {
        "excel": float(np.mean(excels)),
        "cvar": float(np.mean(losses_sorted[-k:])),
        "resource": float(np.mean(resources)),
        "metrics_mean": aggregate_metric_panel(panels),
        "metrics_tail": tail_metric_panel(panels),
        "action_mean": aggregate_actions(traces, "mean_action", np.mean),
        "action_std_mean": aggregate_actions(traces, "std_action", np.mean),
        "by_regime": traces,
    }


def eval_static_frontier(regimes: list[str], static_rows: list[dict], args, seed0: int) -> list[dict]:
    frontier = []
    for item in unique_static_actions(static_rows):
        continuous = np.asarray(item["action"], dtype=np.float32)
        result = eval_policy(
            regimes,
            lambda _obs, _regime, env, a=continuous: encode_action_for_env(env, a, args.action_mode),
            args,
            seed0,
            action_mode=args.action_mode,
        )
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


def eval_static_oracle(regimes: list[str], static_rows: list[dict], args, seed0: int) -> tuple[dict[str, list[float]], dict]:
    action_by_regime: dict[str, list[float]] = {}
    rows = []
    for regime in regimes:
        local = eval_static_frontier([regime], static_rows, args, seed0)
        best = max(local, key=lambda r: r["excel"])
        action_by_regime[regime] = list(best["action"])
        rows.append({"regime": regime, **best})
    oracle = eval_policy(
        regimes,
        lambda _obs, regime, env: encode_action_for_env(env, action_by_regime[regime], args.action_mode),
        args,
        seed0,
        action_mode=args.action_mode,
    )
    oracle["by_regime_static_best"] = rows
    return action_by_regime, oracle


def pareto_verdict(dynamic: dict, static_frontier: list[dict]) -> dict:
    best = max(static_frontier, key=lambda r: r["excel"])
    eligible = [r for r in static_frontier if r["resource"] <= dynamic["resource"] + 1e-12]
    best_eligible = max(eligible, key=lambda r: r["excel"]) if eligible else None
    dominated_by = [
        r for r in static_frontier
        if r["excel"] >= dynamic["excel"] - 1e-12
        and r["cvar"] <= dynamic["cvar"] + 1e-12
        and r["resource"] <= dynamic["resource"] + 1e-12
        and (
            r["excel"] > dynamic["excel"] + 1e-12
            or r["cvar"] < dynamic["cvar"] - 1e-12
            or r["resource"] < dynamic["resource"] - 1e-12
        )
    ]
    return {
        "best_static_by_excel": best,
        "best_static_at_or_below_dynamic_resource": best_eligible,
        "raw_ret_win": dynamic["excel"] > best["excel"],
        "raw_ret_delta_vs_best_static": dynamic["excel"] - best["excel"],
        "resource_constrained_ret_win": bool(best_eligible and dynamic["excel"] > best_eligible["excel"]),
        "resource_constrained_ret_delta": (
            dynamic["excel"] - best_eligible["excel"] if best_eligible else None
        ),
        "pareto_non_dominated": len(dominated_by) == 0,
        "dominated_by_count": len(dominated_by),
        "dominated_by": dominated_by[:10],
    }


def collect_bc(regimes: list[str], teacher_actions: dict[str, list[float]], args, seed0: int):
    obs_rows, act_rows = [], []
    for i, regime in enumerate(regimes):
        env = build_env(
            regime=regime,
            reward_mode=args.reward_mode,
            max_steps=args.max_steps,
            seed=seed0 + i,
            holding_cost=args.holding_cost,
            cvar_alpha=args.cvar_alpha,
            action_mode=args.action_mode,
            frac_grid=args.frac_grid,
        )
        target = encode_action_for_env(env, teacher_actions[regime], args.action_mode)
        obs, _ = env.reset(seed=seed0 + i)
        done = truncated = False
        while not (done or truncated):
            obs_rows.append(np.asarray(obs, dtype=np.float32).copy())
            act_rows.append(np.asarray(target).copy())
            obs, _r, done, truncated, _info = env.step(target)
        env.close()
    return np.vstack(obs_rows).astype(np.float32), np.vstack(act_rows)


def bc_train(model: PPO, obs: np.ndarray, actions: np.ndarray, *, action_mode: str, epochs: int, batch_size: int, seed: int):
    device = model.policy.device
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
    rng = np.random.default_rng(seed)
    if action_mode == "multidiscrete":
        act_t = torch.as_tensor(actions, dtype=torch.long, device=device)

        def loss(idx=None):
            x = obs_t if idx is None else obs_t[idx]
            y = act_t if idx is None else act_t[idx]
            dist = model.policy.get_distribution(x)
            return -dist.log_prob(y).mean()
    else:
        act_t = torch.as_tensor(actions, dtype=torch.float32, device=device)

        def loss(idx=None):
            x = obs_t if idx is None else obs_t[idx]
            y = act_t if idx is None else act_t[idx]
            pred = model.policy.get_distribution(x).mode()
            return torch.nn.functional.mse_loss(pred, y)

    with torch.no_grad():
        initial = float(loss().detach().cpu())
    for _ in range(int(epochs)):
        order = rng.permutation(len(obs))
        for start in range(0, len(obs), int(batch_size)):
            idx = torch.as_tensor(order[start : start + int(batch_size)], dtype=torch.long, device=device)
            l = loss(idx)
            model.policy.optimizer.zero_grad()
            l.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 0.5)
            model.policy.optimizer.step()
    with torch.no_grad():
        final = float(loss().detach().cpu())
    return {"bc_loss_initial": initial, "bc_loss_final": final, "bc_samples": int(len(obs))}


def predict_with_normalizer(model: PPO, normalizer: VecNormalize, obs: np.ndarray):
    norm_obs = normalizer.normalize_obs(np.asarray(obs, dtype=np.float32).reshape(1, -1))
    action, _ = model.predict(norm_obs[0], deterministic=True)
    return action


def eval_learned(regimes: list[str], model: PPO, normalizer: VecNormalize, args, seed0: int) -> dict:
    return eval_policy(
        regimes,
        lambda obs, _regime, _env: predict_with_normalizer(model, normalizer, obs),
        args,
        seed0,
        action_mode=args.action_mode,
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = sorted({k for row in rows for k in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gate-dir", required=True)
    ap.add_argument("--reward-mode", default="ReT_excel_plus_cvar")
    ap.add_argument("--cvar-alpha", type=float, default=0.1)
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--timesteps", type=int, default=40000)
    ap.add_argument("--checkpoint-interval", type=int, default=5000)
    ap.add_argument("--bc-epochs", type=int, default=150)
    ap.add_argument("--bc-batch-size", type=int, default=128)
    ap.add_argument("--action-mode", choices=["multidiscrete", "continuous"], default="multidiscrete")
    ap.add_argument("--frac-grid", type=parse_csv_floats, default=list(DEFAULT_FRAC_GRID))
    ap.add_argument("--teacher", choices=["best_static", "oracle_if_better"], default="oracle_if_better")
    ap.add_argument("--max-steps", type=int, default=52)
    ap.add_argument("--selection-seed0", type=int, default=8000)
    ap.add_argument("--eval-seed0", type=int, default=9000)
    ap.add_argument("--holding-cost", type=float, default=0.0)
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--ent-coef", type=float, default=0.0)
    ap.add_argument("--clip-range", type=float, default=0.1)
    ap.add_argument("--target-kl", type=float, default=0.02)
    ap.add_argument("--collapse-resource", type=float, default=0.02)
    ap.add_argument("--collapse-ret-gap", type=float, default=0.001)
    ap.add_argument("--output", default="outputs/experiments/track_a_training_repair_2026-06-29")
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    summary, static_rows, _gate_oracle_actions = read_gate(Path(args.gate_dir))
    regimes = list(summary["best_by_regime"].keys())

    selection_frontier = eval_static_frontier(regimes, static_rows, args, args.selection_seed0)
    heldout_frontier = eval_static_frontier(regimes, static_rows, args, args.eval_seed0)
    best_selection = max(selection_frontier, key=lambda r: r["excel"])
    oracle_actions, oracle_selection = eval_static_oracle(regimes, static_rows, args, args.selection_seed0)
    if args.teacher == "oracle_if_better" and oracle_selection["excel"] > best_selection["excel"] + 1e-9:
        teacher_kind = "oracle_by_regime"
        teacher_actions = oracle_actions
    else:
        teacher_kind = "best_static"
        teacher_actions = {regime: list(best_selection["action"]) for regime in regimes}

    (out / "best_static_teacher.json").write_text(
        json.dumps(
            {
                "teacher_kind": teacher_kind,
                "best_selection": best_selection,
                "oracle_selection": oracle_selection,
                "teacher_actions": teacher_actions,
            },
            indent=2,
            default=float,
        )
    )
    write_csv(out / "static_frontier_selection.csv", selection_frontier)
    write_csv(out / "static_frontier_heldout.csv", heldout_frontier)

    bc_obs_raw, bc_actions = collect_bc(regimes, teacher_actions, args, args.selection_seed0 + 1000)
    learned, checkpoint_rows = [], []
    best_static_heldout = max(heldout_frontier, key=lambda r: r["excel"])
    for seed in [int(s) for s in args.seeds.split(",") if s.strip()]:
        env_fns = []
        for i in range(args.n_envs):
            regime = regimes[i % len(regimes)]
            env_fns.append(
                lambda r=regime, s=seed + i: build_env(
                    regime=r,
                    reward_mode=args.reward_mode,
                    max_steps=args.max_steps,
                    seed=s,
                    holding_cost=args.holding_cost,
                    cvar_alpha=args.cvar_alpha,
                    action_mode=args.action_mode,
                    frac_grid=args.frac_grid,
                )
            )
        venv = VecNormalize(DummyVecEnv(env_fns), norm_obs=True, norm_reward=True, clip_reward=10.0)
        venv.obs_rms.update(bc_obs_raw)
        bc_obs = venv.normalize_obs(bc_obs_raw.copy())
        model = PPO(
            "MlpPolicy",
            venv,
            seed=seed,
            verbose=0,
            n_steps=min(512, args.max_steps * 4),
            batch_size=64,
            learning_rate=args.learning_rate,
            n_epochs=10,
            ent_coef=args.ent_coef,
            clip_range=args.clip_range,
            target_kl=args.target_kl,
        )
        bc_stats = bc_train(
            model,
            bc_obs,
            bc_actions,
            action_mode=args.action_mode,
            epochs=args.bc_epochs,
            batch_size=args.bc_batch_size,
            seed=seed,
        )

        seed_dir = out / "checkpoints" / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        best_selection_result: dict | None = None
        best_step = 0
        best_score = -1e18
        elapsed = 0
        while elapsed < args.timesteps:
            chunk = min(args.checkpoint_interval, args.timesteps - elapsed)
            model.learn(total_timesteps=int(chunk), reset_num_timesteps=False)
            elapsed += int(chunk)
            selection_result = eval_learned(regimes, model, venv, args, args.selection_seed0)
            collapsed = (
                selection_result["resource"] < args.collapse_resource
                or selection_result["excel"] < best_selection["excel"] - args.collapse_ret_gap
            )
            score = selection_result["excel"] - (10.0 if collapsed else 0.0)
            row = {
                "seed": seed,
                "step": elapsed,
                "excel": selection_result["excel"],
                "cvar": selection_result["cvar"],
                "resource": selection_result["resource"],
                "collapsed": collapsed,
                "score": score,
            }
            checkpoint_rows.append(row)
            if score > best_score:
                best_score = score
                best_step = elapsed
                best_selection_result = selection_result
                model.save(seed_dir / "best_model.zip")
                venv.save(seed_dir / "best_vecnormalize.pkl")

        if best_selection_result is None:
            best_selection_result = eval_learned(regimes, model, venv, args, args.selection_seed0)
        heldout_result = eval_learned(regimes, model, venv, args, args.eval_seed0)
        heldout_result.update(
            {
                "seed": seed,
                "selected_step": best_step,
                "selection_excel": best_selection_result["excel"],
                "selection_resource": best_selection_result["resource"],
                "bc": bc_stats,
            }
        )
        learned.append(heldout_result)
        venv.close()

    dynamic = {
        "excel": float(np.mean([r["excel"] for r in learned])),
        "cvar": float(np.mean([r["cvar"] for r in learned])),
        "resource": float(np.mean([r["resource"] for r in learned])),
        "metrics_mean": aggregate_metric_panel([r.get("metrics_mean", {}) for r in learned]),
        "metrics_tail": aggregate_metric_panel([r.get("metrics_tail", {}) for r in learned]),
        "action_mean": aggregate_actions(learned, "action_mean", np.mean),
        "action_std_mean": aggregate_actions(learned, "action_std_mean", np.mean),
    }
    verdict = pareto_verdict(dynamic, heldout_frontier)
    payload = {
        "args": vars(args),
        "gate_summary": summary,
        "regimes": regimes,
        "teacher_kind": teacher_kind,
        "teacher_actions": teacher_actions,
        "best_selection_static": best_selection,
        "oracle_selection": oracle_selection,
        "best_heldout_static": best_static_heldout,
        "learned_per_seed": learned,
        "dynamic": dynamic,
        "frontier_verdict": verdict,
    }
    (out / "summary.json").write_text(json.dumps(payload, indent=2, default=float))
    write_csv(out / "checkpoint_metrics.csv", checkpoint_rows)
    write_csv(
        out / "seed_health.csv",
        [
            {
                "seed": r["seed"],
                "excel": r["excel"],
                "resource": r["resource"],
                "selected_step": r["selected_step"],
                "selection_excel": r["selection_excel"],
                "bc_loss_initial": r["bc"]["bc_loss_initial"],
                "bc_loss_final": r["bc"]["bc_loss_final"],
            }
            for r in learned
        ],
    )
    lines = [
        "# Track A Training Repair",
        "",
        f"Action mode: `{args.action_mode}`; teacher: `{teacher_kind}`.",
        f"Best held-out static `{best_static_heldout['label']}` Excel={best_static_heldout['excel']:.6f}, resource={best_static_heldout['resource']:.3f}.",
        f"Dynamic Excel={dynamic['excel']:.6f}, CVaR={dynamic['cvar']:.3e}, resource={dynamic['resource']:.3f}.",
        f"Raw ReT win: {verdict['raw_ret_win']} (delta={verdict['raw_ret_delta_vs_best_static']:+.6f}).",
        f"Pareto non-dominated: {verdict['pareto_non_dominated']} (dominated_by={verdict['dominated_by_count']}).",
    ]
    (out / "report.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"WROTE {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
