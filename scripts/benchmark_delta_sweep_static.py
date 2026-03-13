#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.config import DEFAULT_YEAR_BASIS, YEAR_BASIS_OPTIONS
from supply_chain.external_env_interface import make_shift_control_env

POLICY_ORDER = ("static_s1", "static_s2")
RET_CASES = (
    "fill_rate_only",
    "autotomy",
    "recovery",
    "non_recovery",
    "no_demand",
)
FIXED_POLICY_ACTIONS: dict[str, np.ndarray] = {
    "static_s1": np.array([0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32),
    "static_s2": np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
}
EVAL_EPISODE_SEED_OFFSET = 70_000
DEFAULT_DELTAS = (
    0.00,
    0.02,
    0.04,
    0.05,
    0.055,
    0.058,
    0.06,
    0.062,
    0.065,
    0.07,
    0.08,
    0.10,
)
PRIMARY_METRICS = (
    "reward_total",
    "ret_raw_total",
    "fill_rate",
    "backorder_rate",
    "shift_cost_total",
    "disruption_hours_total",
    "mean_step_fill_rate",
    "mean_disruption_fraction",
    "avg_inventory",
    "pct_fill_rate_only",
    "pct_autotomy",
    "pct_recovery",
    "pct_non_recovery",
    "pct_no_demand",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a static-policy delta sweep on the MFSC shift-control environment."
    )
    parser.add_argument(
        "--deltas",
        type=float,
        nargs="+",
        default=list(DEFAULT_DELTAS),
        help="Shift-cost delta values to evaluate.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[11, 22, 33],
        help="Benchmark seeds shared across both static policies.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
        help="Evaluation episodes per delta, seed, and policy.",
    )
    parser.add_argument(
        "--step-size-hours",
        type=float,
        default=168.0,
        help="Environment step size in hours.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=260,
        help="Maximum steps per episode.",
    )
    parser.add_argument(
        "--risk-level",
        choices=["current", "increased", "severe"],
        default="increased",
        help="Risk parameter level passed to make_shift_control_env().",
    )
    parser.add_argument(
        "--year-basis",
        choices=YEAR_BASIS_OPTIONS,
        default=DEFAULT_YEAR_BASIS,
        help="Annualization basis passed to make_shift_control_env().",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/benchmarks/delta_sweep_static"),
        help="Directory for CSV and JSON artifacts.",
    )
    return parser


def ci95(values: list[float]) -> tuple[float, float]:
    if len(values) < 2:
        value = float(values[0]) if values else float("nan")
        return value, value
    arr = np.asarray(values, dtype=np.float64)
    half = 1.96 * arr.std(ddof=1) / np.sqrt(len(arr))
    mean = arr.mean()
    return float(mean - half), float(mean + half)


def static_policy_action(policy: str) -> np.ndarray:
    if policy not in FIXED_POLICY_ACTIONS:
        raise ValueError(f"Unsupported fixed policy {policy!r}.")
    return FIXED_POLICY_ACTIONS[policy].copy()


def build_env_kwargs(args: argparse.Namespace, delta: float) -> dict[str, Any]:
    return {
        "step_size_hours": args.step_size_hours,
        "risk_level": args.risk_level,
        "max_steps": args.max_steps,
        "year_basis": args.year_basis,
        "rt_delta": float(delta),
    }


def finalize_episode_metrics(
    *,
    delta: float,
    policy: str,
    seed: int,
    episode: int,
    eval_seed: int,
    steps: int,
    reward_total: float,
    ret_raw_total: float,
    demanded_total: float,
    delivered_total: float,
    backorder_qty_total: float,
    shift_cost_total: float,
    disruption_hours_total: float,
    mean_step_fill_rate: float,
    mean_disruption_fraction: float,
    avg_inventory: float,
    ret_case_counts: dict[str, int],
) -> dict[str, Any]:
    if demanded_total > 0:
        backorder_rate = backorder_qty_total / demanded_total
        fill_rate = 1.0 - backorder_rate
    else:
        backorder_rate = 0.0
        fill_rate = 1.0
    row = {
        "delta": delta,
        "policy": policy,
        "seed": seed,
        "episode": episode,
        "eval_seed": eval_seed,
        "steps": steps,
        "reward_total": reward_total,
        "ret_raw_total": ret_raw_total,
        "fill_rate": fill_rate,
        "backorder_rate": backorder_rate,
        "shift_cost_total": shift_cost_total,
        "disruption_hours_total": disruption_hours_total,
        "mean_step_fill_rate": mean_step_fill_rate,
        "mean_disruption_fraction": mean_disruption_fraction,
        "avg_inventory": avg_inventory,
        "demanded_total": demanded_total,
        "delivered_total": delivered_total,
        "backorder_qty_total": backorder_qty_total,
    }
    for ret_case in RET_CASES:
        row[f"pct_{ret_case}"] = (
            100.0 * ret_case_counts.get(ret_case, 0) / max(1, steps)
        )
    return row


def evaluate_policy(
    policy: str,
    *,
    args: argparse.Namespace,
    seed: int,
    delta: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    env_kwargs = build_env_kwargs(args, delta)
    action = static_policy_action(policy)

    for episode_idx in range(args.eval_episodes):
        eval_seed = seed + EVAL_EPISODE_SEED_OFFSET + episode_idx
        env = make_shift_control_env(**env_kwargs)
        _, _ = env.reset(seed=eval_seed)
        terminated = False
        truncated = False
        reward_total = 0.0
        ret_raw_total = 0.0
        demanded_total = 0.0
        delivered_total = 0.0
        backorder_qty_total = 0.0
        shift_cost_total = 0.0
        disruption_hours_total = 0.0
        step_fill_rates: list[float] = []
        disruption_fractions: list[float] = []
        inventory_values: list[float] = []
        ret_case_counts = {ret_case: 0 for ret_case in RET_CASES}
        steps = 0

        while not (terminated or truncated):
            _, reward, terminated, truncated, info = env.step(action)
            reward_total += float(reward)
            ret_raw_total += float(info.get("ReT_raw", 0.0))
            demanded_total += float(info.get("new_demanded", 0.0))
            delivered_total += float(info.get("new_delivered", 0.0))
            backorder_qty_total += float(info.get("new_backorder_qty", 0.0))
            shift_cost_total += float(info.get("shift_cost_linear", 0.0))
            disruption_hours_total += float(info.get("step_disruption_hours", 0.0))
            inventory_values.append(float(info.get("total_inventory", 0.0)))
            ret_components = (
                info.get("ret_components")
                if isinstance(info.get("ret_components"), dict)
                else {}
            )
            if "fill_rate" in ret_components:
                step_fill_rates.append(float(ret_components["fill_rate"]))
            if "disruption_fraction" in ret_components:
                disruption_fractions.append(
                    float(ret_components["disruption_fraction"])
                )
            ret_case = ret_components.get("ret_case")
            if isinstance(ret_case, str) and ret_case in ret_case_counts:
                ret_case_counts[ret_case] += 1
            steps += 1

        rows.append(
            finalize_episode_metrics(
                delta=delta,
                policy=policy,
                seed=seed,
                episode=episode_idx + 1,
                eval_seed=eval_seed,
                steps=steps,
                reward_total=reward_total,
                ret_raw_total=ret_raw_total,
                demanded_total=demanded_total,
                delivered_total=delivered_total,
                backorder_qty_total=backorder_qty_total,
                shift_cost_total=shift_cost_total,
                disruption_hours_total=disruption_hours_total,
                mean_step_fill_rate=(
                    float(np.mean(step_fill_rates)) if step_fill_rates else float("nan")
                ),
                mean_disruption_fraction=(
                    float(np.mean(disruption_fractions))
                    if disruption_fractions
                    else float("nan")
                ),
                avg_inventory=(
                    float(np.mean(inventory_values)) if inventory_values else 0.0
                ),
                ret_case_counts=ret_case_counts,
            )
        )
        env.close()

    return rows


def aggregate_seed_metrics(episode_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, str, int], list[dict[str, Any]]] = {}
    for row in episode_rows:
        key = (float(row["delta"]), str(row["policy"]), int(row["seed"]))
        grouped.setdefault(key, []).append(row)

    seed_rows: list[dict[str, Any]] = []
    for (delta, policy, seed), rows in sorted(grouped.items()):
        seed_row: dict[str, Any] = {
            "delta": delta,
            "policy": policy,
            "seed": seed,
            "episodes": len(rows),
        }
        for metric in PRIMARY_METRICS:
            values = [float(row[metric]) for row in rows]
            seed_row[f"{metric}_mean"] = float(np.mean(values))
            seed_row[f"{metric}_std"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            )
        seed_rows.append(seed_row)
    return seed_rows


def aggregate_policy_metrics(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, str], list[dict[str, Any]]] = {}
    for row in seed_rows:
        key = (float(row["delta"]), str(row["policy"]))
        grouped.setdefault(key, []).append(row)

    policy_rows: list[dict[str, Any]] = []
    for delta in sorted({float(row["delta"]) for row in seed_rows}):
        for policy in POLICY_ORDER:
            rows = grouped.get((delta, policy), [])
            if not rows:
                continue
            out_row: dict[str, Any] = {
                "delta": delta,
                "policy": policy,
                "seed_count": len(rows),
            }
            for metric in PRIMARY_METRICS:
                values = [float(row[f"{metric}_mean"]) for row in rows]
                ci_low, ci_high = ci95(values)
                out_row[f"{metric}_mean"] = float(np.mean(values))
                out_row[f"{metric}_std"] = (
                    float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                )
                out_row[f"{metric}_ci95_low"] = ci_low
                out_row[f"{metric}_ci95_high"] = ci_high
            policy_rows.append(out_row)
    return policy_rows


def build_delta_transition_rows(
    policy_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    indexed = {(float(row["delta"]), str(row["policy"])): row for row in policy_rows}
    transition_rows: list[dict[str, Any]] = []
    deltas = sorted({float(row["delta"]) for row in policy_rows})

    for delta in deltas:
        s1 = indexed.get((delta, "static_s1"))
        s2 = indexed.get((delta, "static_s2"))
        if s1 is None or s2 is None:
            continue
        reward_gap = float(s2["reward_total_mean"]) - float(s1["reward_total_mean"])
        fill_rate_gap = float(s2["fill_rate_mean"]) - float(s1["fill_rate_mean"])
        backorder_gap = float(s2["backorder_rate_mean"]) - float(
            s1["backorder_rate_mean"]
        )
        transition_rows.append(
            {
                "delta": delta,
                "s1_reward_mean": float(s1["reward_total_mean"]),
                "s2_reward_mean": float(s2["reward_total_mean"]),
                "reward_gap_s2_minus_s1": reward_gap,
                "ret_raw_gap_s2_minus_s1": float(s2["ret_raw_total_mean"])
                - float(s1["ret_raw_total_mean"]),
                "fill_rate_gap_s2_minus_s1": fill_rate_gap,
                "backorder_rate_gap_s2_minus_s1": backorder_gap,
                "shift_cost_gap_s2_minus_s1": float(s2["shift_cost_total_mean"])
                - float(s1["shift_cost_total_mean"]),
                "disruption_hours_gap_s2_minus_s1": float(
                    s2["disruption_hours_total_mean"]
                )
                - float(s1["disruption_hours_total_mean"]),
                "mean_step_fill_rate_gap_s2_minus_s1": float(
                    s2["mean_step_fill_rate_mean"]
                )
                - float(s1["mean_step_fill_rate_mean"]),
                "mean_disruption_fraction_gap_s2_minus_s1": float(
                    s2["mean_disruption_fraction_mean"]
                )
                - float(s1["mean_disruption_fraction_mean"]),
                "avg_inventory_gap_s2_minus_s1": float(s2["avg_inventory_mean"])
                - float(s1["avg_inventory_mean"]),
                "pct_fill_rate_only_gap_s2_minus_s1": float(
                    s2["pct_fill_rate_only_mean"]
                )
                - float(s1["pct_fill_rate_only_mean"]),
                "pct_autotomy_gap_s2_minus_s1": float(s2["pct_autotomy_mean"])
                - float(s1["pct_autotomy_mean"]),
                "pct_recovery_gap_s2_minus_s1": float(s2["pct_recovery_mean"])
                - float(s1["pct_recovery_mean"]),
                "pct_non_recovery_gap_s2_minus_s1": float(s2["pct_non_recovery_mean"])
                - float(s1["pct_non_recovery_mean"]),
                "preferred_policy_by_reward": (
                    "static_s2" if reward_gap > 0 else "static_s1"
                ),
            }
        )
    return transition_rows


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_delta_sweep(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    episode_rows: list[dict[str, Any]] = []

    for delta in args.deltas:
        for seed in args.seeds:
            for policy in POLICY_ORDER:
                rows = evaluate_policy(policy, args=args, seed=seed, delta=float(delta))
                episode_rows.extend(rows)

    seed_rows = aggregate_seed_metrics(episode_rows)
    policy_rows = aggregate_policy_metrics(seed_rows)
    transition_rows = build_delta_transition_rows(policy_rows)

    episode_csv = args.output_dir / "episode_metrics.csv"
    policy_csv = args.output_dir / "policy_summary.csv"
    transition_csv = args.output_dir / "delta_transition.csv"
    summary_json = args.output_dir / "summary.json"

    save_csv(episode_csv, episode_rows)
    save_csv(policy_csv, policy_rows)
    save_csv(transition_csv, transition_rows)

    summary = {
        "config": {
            "deltas": [float(delta) for delta in args.deltas],
            "seeds": args.seeds,
            "eval_episodes": args.eval_episodes,
            "step_size_hours": args.step_size_hours,
            "max_steps": args.max_steps,
            "risk_level": args.risk_level,
            "year_basis": args.year_basis,
        },
        "policies": list(POLICY_ORDER),
        "artifacts": {
            "episode_metrics_csv": str(episode_csv),
            "policy_summary_csv": str(policy_csv),
            "delta_transition_csv": str(transition_csv),
            "summary_json": str(summary_json),
        },
        "seed_metrics": seed_rows,
        "policy_summary": policy_rows,
        "delta_transition": transition_rows,
    }
    with summary_json.open("w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, indent=2)

    return summary


def main() -> None:
    args = build_parser().parse_args()
    summary = run_delta_sweep(args)
    print(f"Wrote delta sweep artifacts to {args.output_dir}")
    for row in summary["delta_transition"]:
        print(
            f"delta={row['delta']:.3f} | preferred={row['preferred_policy_by_reward']} | "
            f"reward_gap_s2_minus_s1={row['reward_gap_s2_minus_s1']:.3f} | "
            f"fill_rate_gap_s2_minus_s1={row['fill_rate_gap_s2_minus_s1']:.3f}"
        )


if __name__ == "__main__":
    main()
