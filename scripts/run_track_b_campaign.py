#!/usr/bin/env python3
"""Train Track B PPO on promoted R2/R24 campaign cells.

This runner is intentionally small and evidence-gated:

1. Read promoted cells from a static headroom gate.
2. Train PPO on a randomized episode distribution over those cells.
3. Evaluate PPO and dense static policies on the same campaign cells/CRN seeds.

It is the Track B analogue of the Track A "static gate before PPO" rule.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_track_b_independent_doe import Policy, action_for, parse_float_list, parse_int_list
from supply_chain.config import OPERATIONS
from supply_chain.episode_metrics import compute_episode_metrics, merge_resource_metrics
from supply_chain.external_env_interface import make_track_b_env


RISK_FAMILIES: dict[str, tuple[str, ...]] = {
    "R2": ("R21", "R22", "R23", "R24"),
    "R24": ("R24",),
}


@dataclass(frozen=True)
class CampaignCell:
    label: str
    risk_level: str
    family: str
    phi: float
    psi: float
    demand_mult: float

    @property
    def enabled_risks(self) -> tuple[str, ...]:
        return RISK_FAMILIES[self.family]


def mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def ci95(values: list[float]) -> tuple[float, float]:
    if len(values) < 2:
        value = float(values[0]) if values else float("nan")
        return value, value
    arr = np.asarray(values, dtype=np.float64)
    half = 1.96 * arr.std(ddof=1) / np.sqrt(len(arr))
    center = arr.mean()
    return float(center - half), float(center + half)


def extract_downstream_multipliers(final_info: dict[str, Any]) -> tuple[float, float]:
    """Return Op10/Op12 dispatch multipliers from Track B action info."""
    clipped_action = final_info.get("clipped_action")
    if isinstance(clipped_action, (list, tuple)) and len(clipped_action) >= 7:
        return (
            float(1.25 + 0.75 * float(clipped_action[5])),
            float(1.25 + 0.75 * float(clipped_action[6])),
        )

    raw_action = final_info.get("raw_action")
    if isinstance(raw_action, dict):
        op10_base = float(OPERATIONS[10]["q"][0])
        op12_base = float(OPERATIONS[12]["q"][0])
        return (
            float(raw_action.get("op10_q_min", op10_base)) / op10_base,
            float(raw_action.get("op12_q_min", op12_base)) / op12_base,
        )

    return 1.0, 1.0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def env_kwargs_for_cell(args: argparse.Namespace, cell: CampaignCell) -> dict[str, Any]:
    return {
        "reward_mode": args.reward_mode,
        "observation_version": args.observation_version,
        "action_contract": "track_b_v1",
        "risk_level": cell.risk_level,
        "enabled_risks": cell.enabled_risks,
        "risk_frequency_multiplier": cell.phi,
        "risk_impact_multiplier": cell.psi,
        "demand_mean_multiplier": cell.demand_mult,
        "step_size_hours": float(args.step_size_hours),
        "max_steps": int(args.max_steps),
        "ret_excel_cvar_alpha": float(args.ret_excel_cvar_alpha),
        "ret_excel_cvar_tail_level": float(args.ret_excel_cvar_tail_level),
        "ret_excel_cvar_window": int(args.ret_excel_cvar_window),
    }


class TrackBCampaignEnv(gym.Env):
    """Episode-randomized Track B environment.

    A new underlying DES is created at each reset so the policy trains on a
    campaign distribution rather than one stationary stress cell.
    """

    metadata = {"render_modes": []}

    def __init__(self, *, args: argparse.Namespace, cells: list[CampaignCell], seed: int):
        super().__init__()
        self.args = args
        self.cells = list(cells)
        self.rng = np.random.default_rng(seed)
        self._reset_count = 0
        self._env: Any | None = None
        probe = make_track_b_env(**env_kwargs_for_cell(args, self.cells[0]))
        self.observation_space = probe.observation_space
        self.action_space = probe.action_space
        probe.close()

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if self._env is not None:
            self._env.close()
        if options and "cell_index" in options:
            cell_idx = int(options["cell_index"])
        else:
            cell_idx = int(self.rng.integers(0, len(self.cells)))
        cell = self.cells[cell_idx]
        self._env = make_track_b_env(**env_kwargs_for_cell(self.args, cell))
        self._reset_count += 1
        reset_seed = None if seed is None else int(seed)
        obs, info = self._env.reset(seed=reset_seed)
        info = dict(info)
        info["campaign_cell"] = cell.label
        return obs, info

    def step(self, action):
        if self._env is None:
            raise RuntimeError("TrackBCampaignEnv.step called before reset.")
        return self._env.step(action)

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None


def load_campaign_cells(args: argparse.Namespace) -> list[CampaignCell]:
    rows = list(csv.DictReader((args.gate_dir / "cell_policy_summary.csv").open()))
    families = {f.strip() for f in args.families.split(",") if f.strip()}
    risk_levels = {r.strip() for r in args.risk_levels.split(",") if r.strip()}
    candidates: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row["family"] not in families:
            continue
        if row["risk_level"] not in risk_levels:
            continue
        cell = row["cell"]
        value = float(row["ret_excel_mean"])
        if cell not in candidates or value > float(candidates[cell]["ret_excel_mean"]):
            candidates[cell] = row

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in candidates.values():
        grouped.setdefault((row["risk_level"], row["family"]), []).append(row)

    selected: list[CampaignCell] = []
    for key in sorted(grouped):
        ranked = sorted(grouped[key], key=lambda r: float(r["ret_excel_mean"]), reverse=True)
        for row in ranked[: args.cells_per_group]:
            selected.append(
                CampaignCell(
                    label=str(row["cell"]),
                    risk_level=str(row["risk_level"]),
                    family=str(row["family"]),
                    phi=float(row["phi"]),
                    psi=float(row["psi"]),
                    demand_mult=float(row["demand_mult"]),
                )
            )
    if not selected:
        raise ValueError(f"No campaign cells selected from {args.gate_dir}.")
    return selected[: args.max_cells]


def static_policies(args: argparse.Namespace) -> list[Policy]:
    return [
        Policy(shift=s, op9_mult=o9, op10_mult=o10, op12_mult=o12)
        for s in parse_int_list(args.shifts)
        for o9 in parse_float_list(args.op9_mults)
        for o10 in parse_float_list(args.op10_mults)
        for o12 in parse_float_list(args.op12_mults)
    ]


def finalize_row(
    *,
    policy: str,
    seed: int,
    episode: int,
    eval_seed: int,
    cell: CampaignCell,
    reward_total: float,
    panel: dict[str, float],
    steps: int,
    shift_counts: dict[int, int],
    op10_values: list[float],
    op12_values: list[float],
    args: argparse.Namespace,
) -> dict[str, Any]:
    shift_hours = sum(
        shift_counts.get(s, 0) * s * float(args.step_size_hours) for s in (1, 2, 3)
    )
    merged = merge_resource_metrics(
        dict(panel),
        shift_hours=shift_hours,
        extra_shift_hours=sum(
            shift_counts.get(s, 0) * max(0, s - 1) * float(args.step_size_hours)
            for s in (1, 2, 3)
        ),
        strategic_buffer_units=0.0,
    )
    return {
        "policy": policy,
        "seed": seed,
        "episode": episode,
        "eval_seed": eval_seed,
        "cell": cell.label,
        "risk_level": cell.risk_level,
        "family": cell.family,
        "phi": cell.phi,
        "psi": cell.psi,
        "demand_mult": cell.demand_mult,
        "steps": steps,
        "reward_total": reward_total,
        "ret_excel": float(merged.get("ret_excel", 0.0)),
        "flow_fill_rate": float(merged.get("flow_fill_rate", 0.0)),
        "service_loss_auc_per_order": float(merged.get("service_loss_auc_per_order", 0.0)),
        "service_loss_auc_ration_hours": float(
            merged.get("service_loss_auc_ration_hours", 0.0)
        ),
        "lost_rate": float(merged.get("lost_rate", 0.0)),
        "ctj_p99": float(merged.get("ctj_p99", 0.0)),
        "rpj_p99": float(merged.get("rpj_p99", 0.0)),
        "dpj_p99": float(merged.get("dpj_p99", 0.0)),
        "shift_hours": float(merged.get("shift_hours", shift_hours)),
        "assembly_cost_index": sum(shift_counts.get(s, 0) * s for s in (1, 2, 3))
        / max(1.0, 3.0 * steps),
        "pct_steps_S1": 100.0 * shift_counts.get(1, 0) / max(1, steps),
        "pct_steps_S2": 100.0 * shift_counts.get(2, 0) / max(1, steps),
        "pct_steps_S3": 100.0 * shift_counts.get(3, 0) / max(1, steps),
        "op10_multiplier_mean": mean(op10_values),
        "op12_multiplier_mean": mean(op12_values),
    }


def evaluate_static(
    *,
    args: argparse.Namespace,
    policy: Policy,
    cells: list[CampaignCell],
    seed: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    action = action_for(policy)
    for cell_idx, cell in enumerate(cells):
        for episode in range(args.eval_episodes):
            eval_seed = int(args.eval_seed0 + seed * 100_000 + cell_idx * 1000 + episode)
            env = make_track_b_env(**env_kwargs_for_cell(args, cell))
            env.reset(seed=eval_seed)
            done = False
            reward_total = 0.0
            steps = 0
            while not done:
                _, reward, terminated, truncated, _ = env.step(action)
                reward_total += float(reward)
                done = bool(terminated or truncated)
                steps += 1
            panel = compute_episode_metrics(env.unwrapped.sim)
            out.append(
                finalize_row(
                    policy=policy.label,
                    seed=seed,
                    episode=episode + 1,
                    eval_seed=eval_seed,
                    cell=cell,
                    reward_total=reward_total,
                    panel=panel,
                    steps=steps,
                    shift_counts={policy.shift: steps},
                    op10_values=[policy.op10_mult] * steps,
                    op12_values=[policy.op12_mult] * steps,
                    args=args,
                )
            )
            env.close()
    return out


def make_vec_env(args: argparse.Namespace, cells: list[CampaignCell], seed: int):
    def _init() -> Monitor:
        return Monitor(TrackBCampaignEnv(args=args, cells=cells, seed=seed))

    return _init


def train_policy(args: argparse.Namespace, cells: list[CampaignCell], seed: int, run_dir: Path):
    vec = DummyVecEnv(
        [make_vec_env(args, cells, seed + i * 1000) for i in range(max(1, args.n_envs))]
    )
    vec_norm = VecNormalize(vec, norm_obs=True, norm_reward=False, clip_obs=10.0)
    model = PPO(
        "MlpPolicy",
        vec_norm,
        seed=seed,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        verbose=0,
    )
    model.learn(total_timesteps=args.timesteps, progress_bar=False)
    model_dir = run_dir / "models" / f"seed{seed}"
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(model_dir / "ppo_model.zip")
    vec_norm.save(model_dir / "vec_normalize.pkl")
    return model, vec_norm


def evaluate_ppo(
    *,
    args: argparse.Namespace,
    model: PPO,
    vec_norm: VecNormalize,
    cells: list[CampaignCell],
    seed: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    vec_norm.training = False
    for cell_idx, cell in enumerate(cells):
        for episode in range(args.eval_episodes):
            eval_seed = int(args.eval_seed0 + seed * 100_000 + cell_idx * 1000 + episode)
            env = make_track_b_env(**env_kwargs_for_cell(args, cell))
            obs, _ = env.reset(seed=eval_seed)
            done = False
            reward_total = 0.0
            steps = 0
            shift_counts = {1: 0, 2: 0, 3: 0}
            op10_values: list[float] = []
            op12_values: list[float] = []
            while not done:
                obs_norm = vec_norm.normalize_obs(np.asarray(obs, dtype=np.float32)[None, :])
                action, _ = model.predict(obs_norm, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(
                    np.asarray(action[0], dtype=np.float32)
                )
                reward_total += float(reward)
                shift = int(info.get("shifts_active", 1))
                shift_counts[shift] = shift_counts.get(shift, 0) + 1
                op10, op12 = extract_downstream_multipliers(info)
                op10_values.append(op10)
                op12_values.append(op12)
                steps += 1
                done = bool(terminated or truncated)
            panel = compute_episode_metrics(env.unwrapped.sim)
            out.append(
                finalize_row(
                    policy="ppo",
                    seed=seed,
                    episode=episode + 1,
                    eval_seed=eval_seed,
                    cell=cell,
                    reward_total=reward_total,
                    panel=panel,
                    steps=steps,
                    shift_counts=shift_counts,
                    op10_values=op10_values,
                    op12_values=op12_values,
                    args=args,
                )
            )
            env.close()
    return out


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["policy"]), []).append(row)
    metrics = [
        "ret_excel",
        "flow_fill_rate",
        "service_loss_auc_per_order",
        "service_loss_auc_ration_hours",
        "lost_rate",
        "ctj_p99",
        "rpj_p99",
        "dpj_p99",
        "shift_hours",
        "assembly_cost_index",
        "pct_steps_S1",
        "pct_steps_S2",
        "pct_steps_S3",
        "op10_multiplier_mean",
        "op12_multiplier_mean",
        "reward_total",
    ]
    out: list[dict[str, Any]] = []
    for policy, bucket in sorted(grouped.items()):
        row: dict[str, Any] = {"policy": policy, "episodes": len(bucket)}
        for metric in metrics:
            vals = [float(r.get(metric, 0.0)) for r in bucket]
            lo, hi = ci95(vals)
            row[f"{metric}_mean"] = mean(vals)
            row[f"{metric}_ci95_low"] = lo
            row[f"{metric}_ci95_high"] = hi
        out.append(row)
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gate-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--families", default="R2,R24")
    ap.add_argument("--risk-levels", default="current,increased,severe")
    ap.add_argument("--cells-per-group", type=int, default=1)
    ap.add_argument("--max-cells", type=int, default=8)
    ap.add_argument("--reward-mode", default="control_v1")
    ap.add_argument("--ret-excel-cvar-alpha", type=float, default=0.2)
    ap.add_argument("--ret-excel-cvar-tail-level", type=float, default=0.95)
    ap.add_argument("--ret-excel-cvar-window", type=int, default=8)
    ap.add_argument("--observation-version", default="v7")
    ap.add_argument("--timesteps", type=int, default=20_000)
    ap.add_argument("--seeds", default="1")
    ap.add_argument("--eval-episodes", type=int, default=1)
    ap.add_argument("--eval-seed0", type=int, default=9000)
    ap.add_argument("--max-steps", type=int, default=52)
    ap.add_argument("--step-size-hours", type=float, default=168.0)
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--n-steps", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--n-epochs", type=int, default=10)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--shifts", default="1,2,3")
    ap.add_argument("--op9-mults", default="1.0")
    ap.add_argument("--op10-mults", default="0.5,1.0,1.5,2.0")
    ap.add_argument("--op12-mults", default="0.5,1.0,1.5,2.0")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cells = load_campaign_cells(args)
    seeds = list(parse_int_list(args.seeds))
    policies = static_policies(args)
    print(
        f"Track B campaign: {len(cells)} cells × {len(policies)} statics × "
        f"{len(seeds)} seeds × {args.eval_episodes} eval episodes",
        flush=True,
    )
    for cell in cells:
        print(
            f"  cell={cell.label} risk={cell.risk_level}/{cell.family} "
            f"phi={cell.phi:g} psi={cell.psi:g} demand={cell.demand_mult:g}",
            flush=True,
        )

    static_rows: list[dict[str, Any]] = []
    for idx, policy in enumerate(policies, start=1):
        for seed in seeds:
            static_rows.extend(evaluate_static(args=args, policy=policy, cells=cells, seed=seed))
        if idx % max(1, args.__dict__.get("progress_every", 12)) == 0:
            print(f"  evaluated {idx}/{len(policies)} static policies", flush=True)

    learned_rows: list[dict[str, Any]] = []
    for seed in seeds:
        print(f"  training PPO seed={seed}", flush=True)
        model, vec_norm = train_policy(args, cells, seed, args.output_dir)
        learned_rows.extend(evaluate_ppo(args=args, model=model, vec_norm=vec_norm, cells=cells, seed=seed))
        vec_norm.close()

    all_rows = static_rows + learned_rows
    policy_summary = aggregate(all_rows)
    best_static = max(
        [r for r in policy_summary if r["policy"] != "ppo"],
        key=lambda r: float(r["ret_excel_mean"]),
    )
    best_reward_static = max(
        [r for r in policy_summary if r["policy"] != "ppo"],
        key=lambda r: float(r["reward_total_mean"]),
    )
    ppo = next(r for r in policy_summary if r["policy"] == "ppo")
    dominated = [
        r["policy"]
        for r in policy_summary
        if r["policy"] != "ppo"
        and float(r["ret_excel_mean"]) >= float(ppo["ret_excel_mean"])
        and float(r["assembly_cost_index_mean"]) <= float(ppo["assembly_cost_index_mean"])
    ]
    verdict = {
        "raw_ret_win": float(ppo["ret_excel_mean"]) > float(best_static["ret_excel_mean"]),
        "same_reward_win": float(ppo["reward_total_mean"])
        > float(best_reward_static["reward_total_mean"]),
        "tail_service_win": float(ppo["service_loss_auc_per_order_mean"])
        < float(best_static["service_loss_auc_per_order_mean"]),
        "pareto_ret_cost": len(dominated) == 0,
        "resource_efficient_win": float(ppo["ret_excel_mean"]) >= float(best_static["ret_excel_mean"])
        and float(ppo["assembly_cost_index_mean"]) <= float(best_static["assembly_cost_index_mean"]),
        "dominated_by_static": dominated,
    }
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "cells": [cell.__dict__ for cell in cells],
        "best_static": best_static,
        "best_reward_static": best_reward_static,
        "ppo": ppo,
        "deltas": {
            "ret_excel": float(ppo["ret_excel_mean"]) - float(best_static["ret_excel_mean"]),
            "same_reward": float(ppo["reward_total_mean"])
            - float(best_reward_static["reward_total_mean"]),
            "reward_vs_best_ret_static": float(ppo["reward_total_mean"])
            - float(best_static["reward_total_mean"]),
            "flow_fill_rate": float(ppo["flow_fill_rate_mean"])
            - float(best_static["flow_fill_rate_mean"]),
            "service_loss_auc_per_order_signed_win": float(
                best_static["service_loss_auc_per_order_mean"]
            )
            - float(ppo["service_loss_auc_per_order_mean"]),
            "assembly_cost_index": float(ppo["assembly_cost_index_mean"])
            - float(best_static["assembly_cost_index_mean"]),
        },
        "verdict": verdict,
    }
    write_csv(args.output_dir / "episode_metrics.csv", all_rows)
    write_csv(args.output_dir / "policy_summary.csv", policy_summary)
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (args.output_dir / "summary.md").write_text(
        "\n".join(
            [
                "# Track B Campaign PPO",
                "",
                f"Cells: {len(cells)}",
                f"Static policies: {len(policies)}",
                "",
                "## Best Static",
                "",
                json.dumps(best_static, indent=2),
                "",
                "## PPO",
                "",
                json.dumps(ppo, indent=2),
                "",
                "## Best Static By Same Reward",
                "",
                json.dumps(best_reward_static, indent=2),
                "",
                "## Verdict",
                "",
                json.dumps(verdict, indent=2),
            ]
        ),
        encoding="utf-8",
    )
    print(f"WROTE {args.output_dir}", flush=True)
    print(
        f"VERDICT raw_ret_win={verdict['raw_ret_win']} "
        f"pareto={verdict['pareto_ret_cost']} "
        f"delta_ret={payload['deltas']['ret_excel']:+.6f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
