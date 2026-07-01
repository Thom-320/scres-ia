#!/usr/bin/env python3
"""Matched-CRN dense static frontier for a frozen Track B PPO run.

This closes the exact audit gap that bit Track A: evaluate every dense static
under the same eval seeds/episodes as the learned policy, then compare raw ReT,
tail ReT, cost/resource, and CD on the same panel.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_track_b_smoke import (  # noqa: E402
    _finalize_episode_row,
    append_order_ledger_rows,
    extract_downstream_multipliers,
    get_episode_terminal_metrics,
    init_cd_totals,
    update_cd_totals,
)
from supply_chain.config import OPERATIONS  # noqa: E402
from supply_chain.episode_metrics import compute_episode_metrics  # noqa: E402
from supply_chain.external_env_interface import make_track_b_env  # noqa: E402


DEFAULT_RUN_DIR = Path(
    "outputs/experiments/track_b_gain_2026-06-30/"
    "top_tier_confirm_v3_output/track_b_top_tier_confirm_5seed_60k_h104"
)
DEFAULT_SHIFTS = "1,2,3"
DEFAULT_OP_MULTS = "0.5,0.75,1.0,1.25,1.5,2.0,2.5"


@dataclass(frozen=True)
class StaticPolicy:
    shift: int
    op10_mult: float
    op12_mult: float

    @property
    def label(self) -> str:
        return f"S{self.shift}_op10_{self.op10_mult:.2f}_op12_{self.op12_mult:.2f}"


def parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def ci95(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), float(values[0])
    half = 1.96 * statistics.stdev(values) / math.sqrt(len(values))
    mean = statistics.fmean(values)
    return float(mean - half), float(mean + half)


def cvar(values: list[float], *, frac: float = 0.05, lower_tail: bool = True) -> float:
    xs = sorted(float(v) for v in values if pd.notna(v))
    if not xs:
        return float("nan")
    n = max(1, math.ceil(len(xs) * frac))
    tail = xs[:n] if lower_tail else xs[-n:]
    return float(statistics.fmean(tail))


def infer_config(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return {}
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return dict(summary.get("config", {}))


def infer_eval_plan(run_dir: Path) -> list[dict[str, int]]:
    episode_path = run_dir / "episode_metrics.csv"
    if not episode_path.exists():
        return []
    df = pd.read_csv(episode_path)
    if "policy" in df.columns:
        df = df.loc[df["policy"].astype(str) == "ppo"].copy()
    keep = ["seed", "episode", "eval_seed"]
    if not all(col in df.columns for col in keep):
        return []
    plan = (
        df[keep]
        .drop_duplicates()
        .sort_values(["seed", "episode", "eval_seed"])
        .to_dict("records")
    )
    return [
        {
            "seed": int(row["seed"]),
            "episode": int(row["episode"]),
            "eval_seed": int(row["eval_seed"]),
        }
        for row in plan
    ]


def fallback_eval_plan(seeds: list[int], eval_episodes: int) -> list[dict[str, int]]:
    return [
        {"seed": seed, "episode": ep + 1, "eval_seed": seed + 50_000 + ep}
        for seed in seeds
        for ep in range(eval_episodes)
    ]


def action_for(policy: StaticPolicy) -> dict[str, float | int]:
    return {
        "op3_q": float(OPERATIONS[3]["q"]),
        "op3_rop": float(OPERATIONS[3]["rop"]),
        "op9_q_min": float(OPERATIONS[9]["q"][0]),
        "op9_q_max": float(OPERATIONS[9]["q"][1]),
        "op9_rop": float(OPERATIONS[9]["rop"]),
        "op10_q_min": float(OPERATIONS[10]["q"][0]) * float(policy.op10_mult),
        "op10_q_max": float(OPERATIONS[10]["q"][1]) * float(policy.op10_mult),
        "op12_q_min": float(OPERATIONS[12]["q"][0]) * float(policy.op12_mult),
        "op12_q_max": float(OPERATIONS[12]["q"][1]) * float(policy.op12_mult),
        "assembly_shifts": int(policy.shift),
    }


def build_env_kwargs(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    return {
        "reward_mode": str(args.reward_mode or config.get("reward_mode", "control_v1")),
        "observation_version": str(args.observation_version or config.get("observation_version", "v7")),
        "risk_level": str(args.risk_level or config.get("risk_level", "adaptive_benchmark_v2")),
        "step_size_hours": float(args.step_size_hours or config.get("step_size_hours", 168.0)),
        "max_steps": int(args.max_steps or config.get("max_steps", 104)),
    }


def run_static_episode(
    policy: StaticPolicy,
    *,
    eval_item: dict[str, int],
    args: argparse.Namespace,
    env_kwargs: dict[str, Any],
    order_ledger_rows: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    env = make_track_b_env(**env_kwargs)
    _, info = env.reset(seed=int(eval_item["eval_seed"]))
    terminated = False
    truncated = False
    reward_total = 0.0
    demanded_total = 0.0
    backorder_qty_total = 0.0
    steps = 0
    shift_counts = {1: 0, 2: 0, 3: 0}
    op10_multipliers: list[float] = []
    op12_multipliers: list[float] = []
    cd_totals = init_cd_totals()
    final_info = info
    action = action_for(policy)

    while not (terminated or truncated):
        _, reward, terminated, truncated, final_info = env.step(action)
        reward_total += float(reward)
        update_cd_totals(cd_totals, final_info)
        demanded_total += float(final_info.get("new_demanded", 0.0))
        backorder_qty_total += float(final_info.get("new_backorder_qty", 0.0))
        shift_counts[int(final_info.get("shifts_active", policy.shift))] += 1
        op10_mult, op12_mult = extract_downstream_multipliers(final_info)
        op10_multipliers.append(op10_mult)
        op12_multipliers.append(op12_mult)
        steps += 1

    row = _finalize_episode_row(
        policy=policy.label,
        seed=int(eval_item["seed"]),
        episode=int(eval_item["episode"]),
        eval_seed=int(eval_item["eval_seed"]),
        steps=steps,
        reward_total=reward_total,
        demanded_total=demanded_total,
        backorder_qty_total=backorder_qty_total,
        shift_counts=shift_counts,
        op10_multipliers=op10_multipliers,
        op12_multipliers=op12_multipliers,
        track_b_context=final_info["state_constraint_context"]["track_b_context"],
        terminal_metrics=get_episode_terminal_metrics(env),
        final_info=final_info,
        cd_totals=cd_totals,
        full_episode_metrics=compute_episode_metrics(env.unwrapped.sim),
    )
    row.update(
        {
            "static_shift": int(policy.shift),
            "static_op10_mult": float(policy.op10_mult),
            "static_op12_mult": float(policy.op12_mult),
        }
    )
    append_order_ledger_rows(
        order_ledger_rows,
        env,
        policy=policy.label,
        seed=int(eval_item["seed"]),
        episode=int(eval_item["episode"]),
        eval_seed=int(eval_item["eval_seed"]),
    )
    env.close()
    return row


def aggregate_seed(rows: list[dict[str, Any]], metrics: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["policy"]), int(row["seed"])), []).append(row)
    out: list[dict[str, Any]] = []
    for (policy, seed), bucket in sorted(grouped.items()):
        first = bucket[0]
        row: dict[str, Any] = {
            "policy": policy,
            "seed": seed,
            "episodes": len(bucket),
            "shift": int(first["static_shift"]),
            "op10_mult": float(first["static_op10_mult"]),
            "op12_mult": float(first["static_op12_mult"]),
        }
        for metric in metrics:
            vals = [float(item.get(metric, 0.0)) for item in bucket]
            row[f"{metric}_mean"] = float(statistics.fmean(vals))
            row[f"{metric}_std"] = float(statistics.stdev(vals)) if len(vals) > 1 else 0.0
        out.append(row)
    return out


def aggregate_policy(seed_rows: list[dict[str, Any]], metrics: list[str]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in seed_rows:
        grouped.setdefault(str(row["policy"]), []).append(row)
    out: list[dict[str, Any]] = []
    for policy, bucket in sorted(grouped.items()):
        first = bucket[0]
        row: dict[str, Any] = {
            "policy": policy,
            "seed_count": len(bucket),
            "shift": int(first["shift"]),
            "op10_mult": float(first["op10_mult"]),
            "op12_mult": float(first["op12_mult"]),
        }
        for metric in metrics:
            vals = [float(item[f"{metric}_mean"]) for item in bucket]
            low, high = ci95(vals)
            row[f"{metric}_mean"] = float(statistics.fmean(vals))
            row[f"{metric}_std"] = float(statistics.stdev(vals)) if len(vals) > 1 else 0.0
            row[f"{metric}_ci95_low"] = low
            row[f"{metric}_ci95_high"] = high
        out.append(row)
    out.sort(
        key=lambda row: (
            float(row["order_level_ret_mean_mean"]),
            -float(row["assembly_cost_index_mean"]),
        ),
        reverse=True,
    )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_verdict(
    *,
    dynamic_run_dir: Path,
    static_episode_rows: list[dict[str, Any]],
    static_seed_rows: list[dict[str, Any]],
    static_policy_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    dyn_policy = pd.read_csv(dynamic_run_dir / "policy_summary.csv")
    dyn_seed = pd.read_csv(dynamic_run_dir / "seed_metrics.csv")
    dyn_episode = pd.read_csv(dynamic_run_dir / "episode_metrics.csv")
    dyn = dyn_policy.loc[dyn_policy["policy"].astype(str) == "ppo"].iloc[0]
    best_static = max(static_policy_rows, key=lambda r: float(r["order_level_ret_mean_mean"]))
    static_episode = [r for r in static_episode_rows if r["policy"] == best_static["policy"]]
    static_seed = [r for r in static_seed_rows if r["policy"] == best_static["policy"]]

    dyn_ret = float(dyn["order_level_ret_mean_mean"])
    dyn_cost = float(dyn["assembly_cost_index_mean"])
    sta_ret = float(best_static["order_level_ret_mean_mean"])
    sta_cost = float(best_static["assembly_cost_index_mean"])

    dyn_ret_tail = cvar(
        dyn_episode.loc[dyn_episode["policy"].astype(str) == "ppo", "order_level_ret_mean"].tolist(),
        lower_tail=True,
    )
    sta_ret_tail = cvar([float(r["order_level_ret_mean"]) for r in static_episode], lower_tail=True)
    dyn_loss_tail = cvar(
        dyn_episode.loc[
            dyn_episode["policy"].astype(str) == "ppo", "order_service_loss_auc_per_order"
        ].tolist(),
        lower_tail=False,
    )
    sta_loss_tail = cvar(
        [float(r["order_service_loss_auc_per_order"]) for r in static_episode],
        lower_tail=False,
    )

    dominated_by = [
        row["policy"]
        for row in static_policy_rows
        if float(row["order_level_ret_mean_mean"]) >= dyn_ret
        and float(row["assembly_cost_index_mean"]) <= dyn_cost
    ]
    common_seeds = sorted(
        set(int(s) for s in dyn_seed.loc[dyn_seed["policy"].astype(str) == "ppo", "seed"])
        & set(int(r["seed"]) for r in static_seed)
    )
    seed_deltas: list[float] = []
    seed_flow_deltas: list[float] = []
    seed_cd_deltas: list[float] = []
    for seed in common_seeds:
        drow = dyn_seed.loc[
            (dyn_seed["policy"].astype(str) == "ppo") & (dyn_seed["seed"] == seed)
        ].iloc[0]
        srow = next(r for r in static_seed if int(r["seed"]) == seed)
        seed_deltas.append(
            float(drow["order_level_ret_mean_mean"]) - float(srow["order_level_ret_mean_mean"])
        )
        seed_flow_deltas.append(
            float(drow["flow_fill_rate_mean"]) - float(srow["flow_fill_rate_mean"])
        )
        seed_cd_deltas.append(
            float(drow["ret_garrido2024_sigmoid_mean_mean"])
            - float(srow["ret_garrido2024_sigmoid_mean_mean"])
        )
    ret_low, ret_high = ci95(seed_deltas)
    flow_low, flow_high = ci95(seed_flow_deltas)
    cd_low, cd_high = ci95(seed_cd_deltas)
    return {
        "dynamic_run_dir": str(dynamic_run_dir),
        "best_static_policy": best_static["policy"],
        "dynamic": {
            "order_level_ret": dyn_ret,
            "ret_tail_cvar05": dyn_ret_tail,
            "service_loss_cvar95": dyn_loss_tail,
            "assembly_cost_index": dyn_cost,
            "flow_fill_rate": float(dyn["flow_fill_rate_mean"]),
            "cd_sigmoid_mean": float(dyn["ret_garrido2024_sigmoid_mean_mean"]),
        },
        "best_static": {
            "order_level_ret": sta_ret,
            "ret_tail_cvar05": sta_ret_tail,
            "service_loss_cvar95": sta_loss_tail,
            "assembly_cost_index": sta_cost,
            "flow_fill_rate": float(best_static["flow_fill_rate_mean"]),
            "cd_sigmoid_mean": float(best_static["ret_garrido2024_sigmoid_mean_mean"]),
            "shift": int(best_static["shift"]),
            "op10_mult": float(best_static["op10_mult"]),
            "op12_mult": float(best_static["op12_mult"]),
        },
        "verdicts": {
            "raw_ret_win": bool(dyn_ret > sta_ret),
            "tail_ret_win": bool(dyn_ret_tail > sta_ret_tail),
            "service_loss_tail_win": bool(dyn_loss_tail < sta_loss_tail),
            "resource_efficient_win": bool(dyn_ret >= sta_ret and dyn_cost <= sta_cost),
            "pareto_ret_cost": bool(len(dominated_by) == 0 and dyn_ret >= sta_ret),
        },
        "deltas": {
            "order_level_ret": dyn_ret - sta_ret,
            "ret_tail_cvar05": dyn_ret_tail - sta_ret_tail,
            "service_loss_cvar95_signed_win": sta_loss_tail - dyn_loss_tail,
            "assembly_cost_index": dyn_cost - sta_cost,
            "flow_fill_rate": float(dyn["flow_fill_rate_mean"]) - float(best_static["flow_fill_rate_mean"]),
            "cd_sigmoid_mean": float(dyn["ret_garrido2024_sigmoid_mean_mean"])
            - float(best_static["ret_garrido2024_sigmoid_mean_mean"]),
        },
        "seed_paired_ci95": {
            "order_level_ret_delta": [ret_low, ret_high],
            "flow_fill_rate_delta": [flow_low, flow_high],
            "cd_sigmoid_mean_delta": [cd_low, cd_high],
            "n_common_seeds": len(common_seeds),
        },
        "dominated_by_static": dominated_by,
    }


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--shifts", default=DEFAULT_SHIFTS)
    ap.add_argument("--op10-mults", default=DEFAULT_OP_MULTS)
    ap.add_argument("--op12-mults", default=DEFAULT_OP_MULTS)
    ap.add_argument("--seeds", default=None, help="Optional comma list; defaults to PPO eval plan.")
    ap.add_argument("--eval-episodes", type=int, default=None)
    ap.add_argument("--reward-mode", default=None)
    ap.add_argument("--risk-level", default=None)
    ap.add_argument("--observation-version", default=None)
    ap.add_argument("--step-size-hours", type=float, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--export-order-ledger", action="store_true")
    ap.add_argument("--progress-every", type=int, default=100)
    return ap


def main() -> int:
    args = build_parser().parse_args()
    output_dir = args.output_dir or (
        Path("outputs/experiments")
        / f"track_b_dense_crn_static_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    config = infer_config(args.run_dir)
    eval_plan = infer_eval_plan(args.run_dir)
    if args.seeds:
        seeds = parse_int_list(args.seeds)
        eval_episodes = int(args.eval_episodes or config.get("eval_episodes", 12))
        eval_plan = fallback_eval_plan(seeds, eval_episodes)
    if not eval_plan:
        seeds = [1, 2, 3, 4, 5]
        eval_episodes = int(args.eval_episodes or 12)
        eval_plan = fallback_eval_plan(seeds, eval_episodes)

    env_kwargs = build_env_kwargs(args, config)
    policies = [
        StaticPolicy(shift=shift, op10_mult=op10, op12_mult=op12)
        for shift in parse_int_list(args.shifts)
        for op10 in parse_float_list(args.op10_mults)
        for op12 in parse_float_list(args.op12_mults)
    ]
    total = len(policies) * len(eval_plan)
    print(
        f"Track B dense CRN static frontier: {len(policies)} policies × "
        f"{len(eval_plan)} eval episodes = {total} episodes",
        flush=True,
    )
    print(f"Env kwargs: {env_kwargs}", flush=True)

    episode_rows: list[dict[str, Any]] = []
    ledger_rows: list[dict[str, Any]] | None = [] if args.export_order_ledger else None
    done = 0
    for policy in policies:
        for eval_item in eval_plan:
            episode_rows.append(
                run_static_episode(
                    policy,
                    eval_item=eval_item,
                    args=args,
                    env_kwargs=env_kwargs,
                    order_ledger_rows=ledger_rows,
                )
            )
            done += 1
            if args.progress_every > 0 and done % args.progress_every == 0:
                print(f"  {done}/{total}", flush=True)

    metrics = [
        "reward_total",
        "order_level_ret_mean",
        "flow_fill_rate",
        "flow_backorder_rate",
        "terminal_rolling_fill_rate_4w",
        "terminal_rolling_backorder_rate_4w",
        "assembly_cost_index",
        "assembly_hours_total",
        "ret_garrido2024_raw_total",
        "ret_garrido2024_train_total",
        "ret_garrido2024_sigmoid_total",
        "ret_garrido2024_sigmoid_mean",
        "terminal_kappa_dot",
        "terminal_epsilon_avg",
        "order_ret_excel",
        "order_lost_rate",
        "order_service_loss_auc_per_order",
        "order_ctj_p99",
        "order_rpj_p99",
        "order_dpj_p99",
        "order_delivered_rations",
        "order_demanded_rations",
    ]
    seed_rows = aggregate_seed(episode_rows, metrics)
    policy_rows = aggregate_policy(seed_rows, metrics)
    verdict = build_verdict(
        dynamic_run_dir=args.run_dir,
        static_episode_rows=episode_rows,
        static_seed_rows=seed_rows,
        static_policy_rows=policy_rows,
    )
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "run_dir": str(args.run_dir),
            "output_dir": str(output_dir),
            "env_kwargs": env_kwargs,
            "shifts": parse_int_list(args.shifts),
            "op10_mults": parse_float_list(args.op10_mults),
            "op12_mults": parse_float_list(args.op12_mults),
            "n_policies": len(policies),
            "n_eval_episodes": len(eval_plan),
            "export_order_ledger": bool(args.export_order_ledger),
        },
        "verdict": verdict,
        "best_static": policy_rows[0],
    }

    write_csv(output_dir / "episode_metrics.csv", episode_rows)
    write_csv(output_dir / "seed_metrics.csv", seed_rows)
    write_csv(output_dir / "policy_summary.csv", policy_rows)
    write_csv(output_dir / "summary.csv", policy_rows)
    if ledger_rows is not None:
        write_csv(output_dir / "order_ledger.csv", ledger_rows)
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output_dir / "verdict_vs_dynamic.json").write_text(
        json.dumps(verdict, indent=2), encoding="utf-8"
    )
    print(f"WROTE {output_dir}", flush=True)
    print(
        "BEST_STATIC {policy} ret={ret:.6f} cost={cost:.3f} flow={flow:.3f}".format(
            policy=verdict["best_static_policy"],
            ret=verdict["best_static"]["order_level_ret"],
            cost=verdict["best_static"]["assembly_cost_index"],
            flow=verdict["best_static"]["flow_fill_rate"],
        ),
        flush=True,
    )
    print(
        "DYNAMIC ret={ret:.6f} delta={delta:+.6f} raw_win={win} pareto={pareto}".format(
            ret=verdict["dynamic"]["order_level_ret"],
            delta=verdict["deltas"]["order_level_ret"],
            win=verdict["verdicts"]["raw_ret_win"],
            pareto=verdict["verdicts"]["pareto_ret_cost"],
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
