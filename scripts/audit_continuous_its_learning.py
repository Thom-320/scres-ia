#!/usr/bin/env python3
"""Audit what continuous I_t,S PPO learns.

This script is intentionally diagnostic, not confirmatory. It trains PPO in
small increments, evaluates every checkpoint with real ledger metrics, and
records action/observation traces so we can answer:

  * Does the learner improve over training?
  * Does it collapse to a constant or adapt weekly?
  * Which observed signals correlate with buffer/shift decisions?
  * Is it beating, tying, or merely imitating the best continuous static?
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from scripts.run_continuous_its import SHIFT_SIGS, build, parse_init_choice, parse_static_fracs
from supply_chain.episode_metrics import compute_episode_metrics
from supply_chain.external_env_interface import get_observation_fields


def _safe_corr(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(ys) < 3:
        return None
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return None
    x = x[mask]
    y = y[mask]
    if float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _cvar95(values: list[float]) -> float:
    clean = sorted(float(x) for x in values if np.isfinite(float(x)))
    if not clean:
        return float("nan")
    k = max(1, int(round(0.05 * len(clean))))
    return float(np.mean(clean[-k:]))


def _make_env(args: argparse.Namespace, *, init_frac: float, seed: int | None = None):
    return build(
        args.reward_mode,
        args.observation_version,
        args.phi,
        args.psi,
        args.regime,
        args.max_steps,
        init_frac,
        stochastic_pt=args.stochastic_pt,
        ret_tail_transform=args.ret_tail_transform,
        ret_tail_gamma=args.ret_tail_gamma,
        ret_tail_beta=args.ret_tail_beta,
        ret_tail_cap_kappa=args.ret_tail_cap_kappa,
        ret_tail_inv_kappa=args.ret_tail_inv_kappa,
        ret_cd_balanced_n_kappa=args.ret_cd_balanced_n_kappa,
        ret_excel_cvar_alpha=args.ret_excel_cvar_alpha,
        ret_excel_cvar_tail_level=args.ret_excel_cvar_tail_level,
        ret_excel_cvar_window=args.ret_excel_cvar_window,
        seed=seed,
    )


def eval_with_trace(
    args: argparse.Namespace,
    *,
    init_frac: float,
    action_fn: Callable[[np.ndarray], np.ndarray],
    episodes: int,
    seed0: int,
    checkpoint: int,
    policy_label: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    env = _make_env(args, init_frac=init_frac)
    fields = get_observation_fields(args.observation_version)
    episode_metrics: list[dict[str, float]] = []
    rows: list[dict[str, Any]] = []
    service_losses: list[float] = []

    for ep in range(int(episodes)):
        obs, _info = env.reset(seed=int(seed0 + ep))
        done = truncated = False
        step = 0
        rewards: list[float] = []
        while not (done or truncated):
            obs_before = np.asarray(obs, dtype=float)
            action = np.asarray(action_fn(obs_before), dtype=np.float32).reshape(-1)
            obs, reward, done, truncated, info = env.step(action)
            rewards.append(float(reward))
            row: dict[str, Any] = {
                "policy": policy_label,
                "checkpoint": int(checkpoint),
                "episode": int(ep),
                "step": int(step),
                "reward": float(reward),
                "action_frac_raw": float(action[0]) if action.size else float("nan"),
                "action_shift_raw": float(action[1]) if action.size > 1 else float("nan"),
                "frac": float(info.get("continuous_its_frac", float("nan"))),
                "shift": int(info.get("continuous_its_shift", 0) or 0),
            }
            for idx, name in enumerate(fields):
                if idx < obs_before.shape[0]:
                    row[f"obs_{name}"] = float(obs_before[idx])
            rows.append(row)
            step += 1

        metrics = compute_episode_metrics(env.unwrapped.sim)
        payload = {
            "ret_excel": float(metrics.get("ret_excel", metrics.get("mean_ret_excel_formula", float("nan")))),
            "flow_fill_rate": float(metrics.get("flow_fill_rate", float("nan"))),
            "lost_rate": float(metrics.get("lost_rate", float("nan"))),
            "service_loss_auc_ration_hours": float(
                metrics.get("service_loss_auc_ration_hours", float("nan"))
            ),
            "backorder_qty_final": float(metrics.get("backorder_qty_final", float("nan"))),
            "episode_reward_sum": float(np.sum(rewards)) if rewards else 0.0,
        }
        service_losses.append(payload["service_loss_auc_ration_hours"])
        episode_metrics.append(payload)

    frac_values = [float(r["frac"]) for r in rows if np.isfinite(float(r["frac"]))]
    shifts = [int(r["shift"]) for r in rows if int(r["shift"]) > 0]
    frac_by_step: dict[int, list[float]] = {}
    for row in rows:
        frac_by_step.setdefault(int(row["step"]), []).append(float(row["frac"]))
    early = [float(row["frac"]) for row in rows if int(row["step"]) < args.max_steps / 3]
    late = [float(row["frac"]) for row in rows if int(row["step"]) >= 2 * args.max_steps / 3]

    correlation_names = [
        "risk_forecast_48h_norm",
        "risk_forecast_168h_norm",
        "backlog_age_norm",
        "theatre_cover_days_norm",
        "regime_pre_disruption",
        "regime_disrupted",
        "regime_recovery",
        "prev_step_backorder_qty_norm",
        "prev_step_disruption_hours_norm",
    ]
    correlation_names.extend(
        name
        for name in fields
        if name.startswith(("active_", "recent_"))
    )
    correlations: dict[str, float | None] = {}
    for name in correlation_names:
        xs = [float(row.get(f"obs_{name}", float("nan"))) for row in rows]
        correlations[f"frac_vs_{name}"] = _safe_corr(frac_values, xs)

    summary = {
        "policy": policy_label,
        "checkpoint": int(checkpoint),
        "episodes": int(episodes),
        "ret_excel_mean": float(np.nanmean([m["ret_excel"] for m in episode_metrics])),
        "flow_fill_rate_mean": float(np.nanmean([m["flow_fill_rate"] for m in episode_metrics])),
        "lost_rate_mean": float(np.nanmean([m["lost_rate"] for m in episode_metrics])),
        "service_loss_auc_mean": float(
            np.nanmean([m["service_loss_auc_ration_hours"] for m in episode_metrics])
        ),
        "cvar95_service_loss": _cvar95(service_losses),
        "episode_reward_mean": float(
            np.nanmean([m["episode_reward_sum"] for m in episode_metrics])
        ),
        "frac_mean": float(np.nanmean(frac_values)) if frac_values else float("nan"),
        "frac_std": float(np.nanstd(frac_values)) if frac_values else float("nan"),
        "frac_early_mean": float(np.nanmean(early)) if early else float("nan"),
        "frac_late_mean": float(np.nanmean(late)) if late else float("nan"),
        "frac_step_mean_std": float(
            np.nanstd([np.nanmean(v) for _, v in sorted(frac_by_step.items())])
        )
        if frac_by_step
        else float("nan"),
        "shift_share_s1": float(np.mean([s == 1 for s in shifts])) if shifts else float("nan"),
        "shift_share_s2": float(np.mean([s == 2 for s in shifts])) if shifts else float("nan"),
        "shift_share_s3": float(np.mean([s == 3 for s in shifts])) if shifts else float("nan"),
        "adaptive_by_frac_std": bool(float(np.nanstd(frac_values)) > 0.05)
        if frac_values
        else False,
        "correlations": correlations,
    }
    return summary, rows


def static_surface(args: argparse.Namespace) -> dict[str, Any]:
    constants: dict[str, Any] = {}
    for init_f in args.static_init_fracs:
        for f in args.static_fracs:
            for sh, sig in SHIFT_SIGS.items():
                label = f"init{init_f}_f{f}_S{sh}"
                summary, _rows = eval_with_trace(
                    args,
                    init_frac=float(init_f),
                    action_fn=lambda _obs, ff=f, ss=sig: np.array([ff, ss], dtype=np.float32),
                    episodes=args.eval_episodes,
                    seed0=args.static_eval_seed,
                    checkpoint=0,
                    policy_label=label,
                )
                constants[label] = {
                    "init_frac": float(init_f),
                    "weekly_frac": float(f),
                    "shift": int(sh),
                    **summary,
                }
    best_excel = max(constants, key=lambda k: constants[k]["ret_excel_mean"])
    best_cvar = min(constants, key=lambda k: constants[k]["cvar95_service_loss"])
    best_reward = max(constants, key=lambda k: constants[k]["episode_reward_mean"])
    return {
        "constants": constants,
        "best_const_excel": best_excel,
        "best_const_cvar": best_cvar,
        "best_const_reward": best_reward,
    }


def parse_checkpoints(value: str) -> list[int]:
    checkpoints = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    if not checkpoints or checkpoints[0] < 0:
        raise argparse.ArgumentTypeError("Expected non-negative comma-separated checkpoints.")
    return checkpoints


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reward-mode", default="ReT_excel_delta")
    ap.add_argument("--observation-version", default="v6")
    ap.add_argument("--phi", type=float, default=4.0)
    ap.add_argument("--psi", type=float, default=1.5)
    ap.add_argument("--regime", default="current")
    ap.add_argument("--stochastic-pt", action="store_true")
    ap.add_argument("--ret-tail-transform", default="identity")
    ap.add_argument("--ret-tail-gamma", type=float, default=1.0)
    ap.add_argument("--ret-tail-beta", type=float, default=2.0)
    ap.add_argument("--ret-tail-cap-kappa", type=float, default=None)
    ap.add_argument("--ret-tail-inv-kappa", type=float, default=None)
    ap.add_argument("--ret-cd-balanced-n-kappa", type=float, default=0.05)
    ap.add_argument("--ret-excel-cvar-alpha", type=float, default=0.5)
    ap.add_argument("--ret-excel-cvar-tail-level", type=float, default=0.05)
    ap.add_argument("--ret-excel-cvar-window", type=int, default=50)
    ap.add_argument("--init-frac", type=parse_init_choice, default="best_excel")
    ap.add_argument("--seeds", default="8101,8102")
    ap.add_argument("--checkpoints", type=parse_checkpoints, default=[0, 4000, 20000, 60000])
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--ent-coef", type=float, default=0.0)
    ap.add_argument("--eval-episodes", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=104)
    ap.add_argument("--static-eval-seed", type=int, default=5000)
    ap.add_argument("--policy-eval-seed", type=int, default=9000)
    ap.add_argument("--static-fracs", type=parse_static_fracs, default="0,0.1,0.2,0.25,0.3,0.4,0.5,0.6,0.75,1")
    ap.add_argument("--static-init-fracs", type=parse_static_fracs, default="0,0.25,0.5,0.75,1")
    ap.add_argument("--output", default="outputs/diagnostics/continuous_its_learning_audit")
    args = ap.parse_args()

    seeds = [int(seed) for seed in args.seeds.split(",") if seed.strip()]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    surface = static_surface(args)
    constants = surface["constants"]
    if args.init_frac == "best_excel":
        learned_init_frac = float(constants[surface["best_const_excel"]]["init_frac"])
    elif args.init_frac == "best_cvar":
        learned_init_frac = float(constants[surface["best_const_cvar"]]["init_frac"])
    elif args.init_frac == "best_cd":
        # This diagnostic has no CD static summary; reward-best is the closest same-bar proxy.
        learned_init_frac = float(constants[surface["best_const_reward"]]["init_frac"])
    else:
        learned_init_frac = float(args.init_frac)

    checkpoint_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    max_checkpoint = max(args.checkpoints)
    for seed in seeds:
        n_envs = max(1, int(args.n_envs))
        venv = DummyVecEnv(
            [
                lambda s=seed + i: _make_env(args, init_frac=learned_init_frac, seed=s)
                for i in range(n_envs)
            ]
        )
        rollout_steps = max(8, min(1024, args.max_steps * 4))
        model = PPO(
            "MlpPolicy",
            venv,
            seed=seed,
            verbose=0,
            n_steps=rollout_steps,
            batch_size=min(64, rollout_steps * n_envs),
            learning_rate=float(args.learning_rate),
            ent_coef=float(args.ent_coef),
            n_epochs=10,
        )
        trained_to = 0
        for checkpoint in args.checkpoints:
            increment = int(checkpoint - trained_to)
            if increment > 0:
                model.learn(total_timesteps=increment, reset_num_timesteps=False)
                trained_to = int(checkpoint)
            summary, rows = eval_with_trace(
                args,
                init_frac=learned_init_frac,
                action_fn=lambda obs, m=model: m.predict(obs, deterministic=True)[0],
                episodes=args.eval_episodes,
                seed0=args.policy_eval_seed + seed,
                checkpoint=checkpoint,
                policy_label=f"ppo_seed{seed}",
            )
            summary["seed"] = int(seed)
            checkpoint_rows.append(summary)
            trace_rows.extend(rows)
        if trained_to < max_checkpoint:
            raise RuntimeError("Internal checkpoint accounting error.")

    best_excel = surface["best_const_excel"]
    best_cvar = surface["best_const_cvar"]
    best_reward = surface["best_const_reward"]
    final_rows = [r for r in checkpoint_rows if int(r["checkpoint"]) == max_checkpoint]
    final_excel = float(np.nanmean([r["ret_excel_mean"] for r in final_rows]))
    final_cvar = float(np.nanmean([r["cvar95_service_loss"] for r in final_rows]))
    final_frac_std = float(np.nanmean([r["frac_std"] for r in final_rows]))
    decision = {
        "args": {**vars(args), "init_frac": str(args.init_frac)},
        "learned_init_frac": learned_init_frac,
        "best_const_excel": best_excel,
        "best_const_cvar": best_cvar,
        "best_const_reward": best_reward,
        "best_const_excel_value": constants[best_excel]["ret_excel_mean"],
        "best_const_cvar_value": constants[best_cvar]["cvar95_service_loss"],
        "final_checkpoint": max_checkpoint,
        "final_learned_excel_mean": final_excel,
        "final_learned_cvar95_mean": final_cvar,
        "final_learned_frac_std_mean": final_frac_std,
        "win_excel": bool(final_excel > float(constants[best_excel]["ret_excel_mean"])),
        "win_cvar": bool(final_cvar < float(constants[best_cvar]["cvar95_service_loss"])),
        "adaptive": bool(final_frac_std > 0.05),
        "learning_curve": checkpoint_rows,
        "static_surface": surface,
    }

    (out / "learning_audit.json").write_text(json.dumps(decision, indent=2))
    with (out / "checkpoint_metrics.csv").open("w", newline="") as handle:
        fieldnames = sorted({key for row in checkpoint_rows for key in row if key != "correlations"})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{k: v for k, v in row.items() if k != "correlations"} for row in checkpoint_rows])
    if trace_rows:
        with (out / "action_trace.csv").open("w", newline="") as handle:
            fieldnames = sorted({key for row in trace_rows for key in row})
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(trace_rows)

    print(json.dumps({
        "output": str(out),
        "best_const_excel": best_excel,
        "best_const_excel_value": constants[best_excel]["ret_excel_mean"],
        "best_const_cvar": best_cvar,
        "best_const_cvar_value": constants[best_cvar]["cvar95_service_loss"],
        "learned_init_frac": learned_init_frac,
        "final_checkpoint": max_checkpoint,
        "final_learned_excel_mean": final_excel,
        "final_learned_cvar95_mean": final_cvar,
        "final_learned_frac_std_mean": final_frac_std,
        "win_excel": decision["win_excel"],
        "win_cvar": decision["win_cvar"],
        "adaptive": decision["adaptive"],
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
