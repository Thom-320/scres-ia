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
FORMULA_MODES = (
    "default",
    "autotomy_equals_recovery",
    "merged_recovery_formula",
)
FIXED_POLICY_ACTIONS: dict[str, np.ndarray] = {
    "static_s1": np.array([0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32),
    "static_s2": np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
}
EVAL_EPISODE_SEED_OFFSET = 90_000
FILL_RATE_BUCKET_EDGES = tuple(i / 10 for i in range(11))
PRIMARY_METRICS = (
    "reward_total",
    "ret_raw_total",
    "fill_rate",
    "backorder_rate",
    "shift_cost_total",
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
        description="Run static-policy ablations of the ReT_thesis reward logic."
    )
    parser.add_argument(
        "--autotomy-thresholds",
        type=float,
        nargs="+",
        default=[0.95, 0.90],
        help="Autotomy fill-rate thresholds to evaluate.",
    )
    parser.add_argument(
        "--formula-modes",
        choices=FORMULA_MODES,
        nargs="+",
        default=["default", "autotomy_equals_recovery"],
        help="How to score disrupted steps once the case is classified.",
    )
    parser.add_argument(
        "--rt-deltas",
        type=float,
        nargs="+",
        default=[0.0, 0.06],
        help="Shift-cost weights applied after the ablated ReT_raw is computed.",
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
        help="Evaluation episodes per config, seed, and policy.",
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
        default=Path("outputs/benchmarks/ret_ablation_static"),
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


def build_env_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "step_size_hours": args.step_size_hours,
        "risk_level": args.risk_level,
        "max_steps": args.max_steps,
        "year_basis": args.year_basis,
        # Reward is recomputed offline for every ablation config.
        "rt_delta": 0.0,
    }


def make_case_stats() -> dict[str, dict[str, float]]:
    return {
        ret_case: {
            "count": 0.0,
            "ret_total": 0.0,
            "fill_rate_total": 0.0,
            "disruption_fraction_total": 0.0,
        }
        for ret_case in RET_CASES
    }


def bucket_label(fill_rate: float) -> str:
    fill_rate = min(1.0, max(0.0, fill_rate))
    lower = min(0.9, np.floor(fill_rate * 10.0) / 10.0)
    upper = min(1.0, lower + 0.1)
    return f"{lower:.1f}-{upper:.1f}"


def variant_key(variant: dict[str, Any]) -> tuple[float, str, float]:
    return (
        float(variant["autotomy_threshold"]),
        str(variant["formula_mode"]),
        float(variant["rt_delta"]),
    )


def build_variants(args: argparse.Namespace) -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    for threshold in args.autotomy_thresholds:
        for formula_mode in args.formula_modes:
            for rt_delta in args.rt_deltas:
                variants.append(
                    {
                        "autotomy_threshold": float(threshold),
                        "formula_mode": str(formula_mode),
                        "rt_delta": float(rt_delta),
                    }
                )
    return variants


def compute_ret_ablation_components(
    *,
    demanded: float,
    backorder_qty: float,
    disruption_fraction: float,
    autotomy_threshold: float,
    nonrecovery_disruption_threshold: float,
    nonrecovery_fill_rate_threshold: float,
    formula_mode: str,
) -> dict[str, float | str]:
    if demanded <= 0:
        return {
            "ret_case": "no_demand",
            "fill_rate": 1.0,
            "disruption_fraction": 0.0,
            "ret_value": 1.0,
        }

    fill_rate = max(0.0, 1.0 - backorder_qty / demanded)
    disruption_fraction = max(0.0, min(1.0, disruption_fraction))

    if (
        disruption_fraction > nonrecovery_disruption_threshold
        and fill_rate < nonrecovery_fill_rate_threshold
    ):
        return {
            "ret_case": "non_recovery",
            "fill_rate": fill_rate,
            "disruption_fraction": disruption_fraction,
            "ret_value": 0.0,
        }

    if disruption_fraction <= 0.0:
        return {
            "ret_case": "fill_rate_only",
            "fill_rate": fill_rate,
            "disruption_fraction": disruption_fraction,
            "ret_value": fill_rate,
        }

    if formula_mode == "merged_recovery_formula":
        return {
            "ret_case": ("autotomy" if fill_rate >= autotomy_threshold else "recovery"),
            "fill_rate": fill_rate,
            "disruption_fraction": disruption_fraction,
            "ret_value": 1.0 / (1.0 + disruption_fraction),
        }

    if fill_rate >= autotomy_threshold:
        ret_value = (
            1.0 / (1.0 + disruption_fraction)
            if formula_mode == "autotomy_equals_recovery"
            else 1.0 - disruption_fraction
        )
        return {
            "ret_case": "autotomy",
            "fill_rate": fill_rate,
            "disruption_fraction": disruption_fraction,
            "ret_value": ret_value,
        }

    return {
        "ret_case": "recovery",
        "fill_rate": fill_rate,
        "disruption_fraction": disruption_fraction,
        "ret_value": 1.0 / (1.0 + disruption_fraction),
    }


def finalize_episode_metrics(
    *,
    variant: dict[str, Any],
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
        "autotomy_threshold": float(variant["autotomy_threshold"]),
        "formula_mode": str(variant["formula_mode"]),
        "rt_delta": float(variant["rt_delta"]),
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


def build_case_episode_rows(
    *,
    variant: dict[str, Any],
    policy: str,
    seed: int,
    episode: int,
    eval_seed: int,
    steps: int,
    case_stats: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ret_case in RET_CASES:
        stats = case_stats[ret_case]
        count = int(stats["count"])
        if count > 0:
            mean_ret_value = stats["ret_total"] / count
            mean_fill_rate = stats["fill_rate_total"] / count
            mean_disruption_fraction = stats["disruption_fraction_total"] / count
        else:
            mean_ret_value = float("nan")
            mean_fill_rate = float("nan")
            mean_disruption_fraction = float("nan")
        rows.append(
            {
                "autotomy_threshold": float(variant["autotomy_threshold"]),
                "formula_mode": str(variant["formula_mode"]),
                "rt_delta": float(variant["rt_delta"]),
                "policy": policy,
                "seed": seed,
                "episode": episode,
                "eval_seed": eval_seed,
                "episode_steps": steps,
                "case": ret_case,
                "case_count": count,
                "case_pct": 100.0 * count / max(1, steps),
                "total_ret_contribution": float(stats["ret_total"]),
                "mean_ret_value": mean_ret_value,
                "mean_step_fill_rate": mean_fill_rate,
                "mean_disruption_fraction": mean_disruption_fraction,
            }
        )
    return rows


def build_fill_rate_bucket_rows(
    *,
    variant: dict[str, Any],
    policy: str,
    seed: int,
    episode: int,
    eval_seed: int,
    fill_rate_bucket_stats: dict[str, dict[str, float]],
    fill_rate_only_count: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bucket in sorted(fill_rate_bucket_stats.keys()):
        stats = fill_rate_bucket_stats[bucket]
        count = int(stats["count"])
        mean_ret_value = stats["ret_total"] / count if count > 0 else float("nan")
        rows.append(
            {
                "autotomy_threshold": float(variant["autotomy_threshold"]),
                "formula_mode": str(variant["formula_mode"]),
                "rt_delta": float(variant["rt_delta"]),
                "policy": policy,
                "seed": seed,
                "episode": episode,
                "eval_seed": eval_seed,
                "fill_rate_only_total": fill_rate_only_count,
                "fill_rate_bucket": bucket,
                "case_count": count,
                "case_pct_of_fill_rate_only": (
                    100.0 * count / max(1, fill_rate_only_count)
                ),
                "mean_ret_value": mean_ret_value,
            }
        )
    return rows


def evaluate_static_rollout(
    *,
    args: argparse.Namespace,
    policy: str,
    seed: int,
    episode_idx: int,
    variants: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    env = make_shift_control_env(**build_env_kwargs(args))
    eval_seed = seed + EVAL_EPISODE_SEED_OFFSET + episode_idx
    _, _ = env.reset(seed=eval_seed)
    action = static_policy_action(policy)

    counters: dict[tuple[float, str, float], dict[str, Any]] = {}
    for variant in variants:
        counters[variant_key(variant)] = {
            "reward_total": 0.0,
            "ret_raw_total": 0.0,
            "demanded_total": 0.0,
            "delivered_total": 0.0,
            "backorder_qty_total": 0.0,
            "shift_cost_total": 0.0,
            "step_fill_rates": [],
            "disruption_fractions": [],
            "inventory_values": [],
            "ret_case_counts": {ret_case: 0 for ret_case in RET_CASES},
            "case_stats": make_case_stats(),
            "fill_rate_bucket_stats": {},
            "steps": 0,
        }

    terminated = False
    truncated = False
    while not (terminated or truncated):
        _, _, terminated, truncated, info = env.step(action)
        demanded = float(info.get("new_demanded", 0.0))
        backorder_qty = float(info.get("new_backorder_qty", 0.0))
        delivered = float(info.get("new_delivered", 0.0))
        shifts_active = int(info.get("shifts_active", 1))
        inventory_total = float(info.get("total_inventory", 0.0))
        ret_components = (
            info.get("ret_components")
            if isinstance(info.get("ret_components"), dict)
            else {}
        )
        disruption_fraction = float(ret_components.get("disruption_fraction", 0.0))
        nonrecovery_disruption_threshold = float(
            ret_components.get("thresholds", {}).get(
                "nonrecovery_disruption_fraction_threshold", 0.5
            )
        )
        nonrecovery_fill_rate_threshold = float(
            ret_components.get("thresholds", {}).get(
                "nonrecovery_fill_rate_threshold", 0.5
            )
        )

        for variant in variants:
            key = variant_key(variant)
            ablated = compute_ret_ablation_components(
                demanded=demanded,
                backorder_qty=backorder_qty,
                disruption_fraction=disruption_fraction,
                autotomy_threshold=float(variant["autotomy_threshold"]),
                nonrecovery_disruption_threshold=nonrecovery_disruption_threshold,
                nonrecovery_fill_rate_threshold=nonrecovery_fill_rate_threshold,
                formula_mode=str(variant["formula_mode"]),
            )
            shift_cost = float(variant["rt_delta"]) * max(0, shifts_active - 1)
            ret_value = float(ablated["ret_value"])

            counters[key]["reward_total"] += ret_value - shift_cost
            counters[key]["ret_raw_total"] += ret_value
            counters[key]["demanded_total"] += demanded
            counters[key]["delivered_total"] += delivered
            counters[key]["backorder_qty_total"] += backorder_qty
            counters[key]["shift_cost_total"] += shift_cost
            counters[key]["step_fill_rates"].append(float(ablated["fill_rate"]))
            counters[key]["disruption_fractions"].append(
                float(ablated["disruption_fraction"])
            )
            counters[key]["inventory_values"].append(inventory_total)
            ret_case = str(ablated["ret_case"])
            counters[key]["ret_case_counts"][ret_case] += 1
            counters[key]["case_stats"][ret_case]["count"] += 1
            counters[key]["case_stats"][ret_case]["ret_total"] += ret_value
            counters[key]["case_stats"][ret_case]["fill_rate_total"] += float(
                ablated["fill_rate"]
            )
            counters[key]["case_stats"][ret_case]["disruption_fraction_total"] += float(
                ablated["disruption_fraction"]
            )
            if ret_case == "fill_rate_only":
                fill_bucket = bucket_label(float(ablated["fill_rate"]))
                bucket_stats = counters[key]["fill_rate_bucket_stats"].setdefault(
                    fill_bucket, {"count": 0.0, "ret_total": 0.0}
                )
                bucket_stats["count"] += 1
                bucket_stats["ret_total"] += ret_value
            counters[key]["steps"] += 1

    env.close()

    rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    bucket_rows: list[dict[str, Any]] = []
    for variant in variants:
        key = variant_key(variant)
        bucket = counters[key]
        episode_number = episode_idx + 1
        rows.append(
            finalize_episode_metrics(
                variant=variant,
                policy=policy,
                seed=seed,
                episode=episode_number,
                eval_seed=eval_seed,
                steps=int(bucket["steps"]),
                reward_total=float(bucket["reward_total"]),
                ret_raw_total=float(bucket["ret_raw_total"]),
                demanded_total=float(bucket["demanded_total"]),
                delivered_total=float(bucket["delivered_total"]),
                backorder_qty_total=float(bucket["backorder_qty_total"]),
                shift_cost_total=float(bucket["shift_cost_total"]),
                mean_step_fill_rate=float(np.mean(bucket["step_fill_rates"])),
                mean_disruption_fraction=float(np.mean(bucket["disruption_fractions"])),
                avg_inventory=float(np.mean(bucket["inventory_values"])),
                ret_case_counts=dict(bucket["ret_case_counts"]),
            )
        )
        case_rows.extend(
            build_case_episode_rows(
                variant=variant,
                policy=policy,
                seed=seed,
                episode=episode_number,
                eval_seed=eval_seed,
                steps=int(bucket["steps"]),
                case_stats=dict(bucket["case_stats"]),
            )
        )
        bucket_rows.extend(
            build_fill_rate_bucket_rows(
                variant=variant,
                policy=policy,
                seed=seed,
                episode=episode_number,
                eval_seed=eval_seed,
                fill_rate_bucket_stats=dict(bucket["fill_rate_bucket_stats"]),
                fill_rate_only_count=int(bucket["ret_case_counts"]["fill_rate_only"]),
            )
        )
    return rows, case_rows, bucket_rows


def aggregate_seed_metrics(episode_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, str, float, str, int], list[dict[str, Any]]] = {}
    for row in episode_rows:
        key = (
            float(row["autotomy_threshold"]),
            str(row["formula_mode"]),
            float(row["rt_delta"]),
            str(row["policy"]),
            int(row["seed"]),
        )
        grouped.setdefault(key, []).append(row)

    seed_rows: list[dict[str, Any]] = []
    for (threshold, mode, rt_delta, policy, seed), rows in sorted(grouped.items()):
        out_row: dict[str, Any] = {
            "autotomy_threshold": threshold,
            "formula_mode": mode,
            "rt_delta": rt_delta,
            "policy": policy,
            "seed": seed,
            "episodes": len(rows),
        }
        for metric in PRIMARY_METRICS:
            values = [float(row[metric]) for row in rows]
            out_row[f"{metric}_mean"] = float(np.mean(values))
            out_row[f"{metric}_std"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            )
        seed_rows.append(out_row)
    return seed_rows


def aggregate_policy_metrics(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, str, float, str], list[dict[str, Any]]] = {}
    for row in seed_rows:
        key = (
            float(row["autotomy_threshold"]),
            str(row["formula_mode"]),
            float(row["rt_delta"]),
            str(row["policy"]),
        )
        grouped.setdefault(key, []).append(row)

    configs = sorted(
        {
            (
                float(row["autotomy_threshold"]),
                str(row["formula_mode"]),
                float(row["rt_delta"]),
            )
            for row in seed_rows
        }
    )
    policy_rows: list[dict[str, Any]] = []
    for threshold, mode, rt_delta in configs:
        for policy in POLICY_ORDER:
            rows = grouped.get((threshold, mode, rt_delta, policy), [])
            if not rows:
                continue
            out_row: dict[str, Any] = {
                "autotomy_threshold": threshold,
                "formula_mode": mode,
                "rt_delta": rt_delta,
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


def build_transition_rows(policy_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = {
        (
            float(row["autotomy_threshold"]),
            str(row["formula_mode"]),
            float(row["rt_delta"]),
            str(row["policy"]),
        ): row
        for row in policy_rows
    }
    configs = sorted(
        {
            (
                float(row["autotomy_threshold"]),
                str(row["formula_mode"]),
                float(row["rt_delta"]),
            )
            for row in policy_rows
        }
    )

    transition_rows: list[dict[str, Any]] = []
    for threshold, mode, rt_delta in configs:
        s1 = indexed.get((threshold, mode, rt_delta, "static_s1"))
        s2 = indexed.get((threshold, mode, rt_delta, "static_s2"))
        if s1 is None or s2 is None:
            continue
        reward_gap = float(s2["reward_total_mean"]) - float(s1["reward_total_mean"])
        row = {
            "autotomy_threshold": threshold,
            "formula_mode": mode,
            "rt_delta": rt_delta,
            "reward_gap_s2_minus_s1": reward_gap,
            "ret_raw_gap_s2_minus_s1": float(s2["ret_raw_total_mean"])
            - float(s1["ret_raw_total_mean"]),
            "fill_rate_gap_s2_minus_s1": float(s2["fill_rate_mean"])
            - float(s1["fill_rate_mean"]),
            "backorder_rate_gap_s2_minus_s1": float(s2["backorder_rate_mean"])
            - float(s1["backorder_rate_mean"]),
            "mean_step_fill_rate_gap_s2_minus_s1": float(s2["mean_step_fill_rate_mean"])
            - float(s1["mean_step_fill_rate_mean"]),
            "mean_disruption_fraction_gap_s2_minus_s1": float(
                s2["mean_disruption_fraction_mean"]
            )
            - float(s1["mean_disruption_fraction_mean"]),
            "avg_inventory_gap_s2_minus_s1": float(s2["avg_inventory_mean"])
            - float(s1["avg_inventory_mean"]),
            "pct_autotomy_gap_s2_minus_s1": float(s2["pct_autotomy_mean"])
            - float(s1["pct_autotomy_mean"]),
            "pct_recovery_gap_s2_minus_s1": float(s2["pct_recovery_mean"])
            - float(s1["pct_recovery_mean"]),
            "shift_cost_gap_s2_minus_s1": float(s2["shift_cost_total_mean"])
            - float(s1["shift_cost_total_mean"]),
            "preferred_policy_by_reward": (
                "static_s2" if reward_gap > 0 else "static_s1"
            ),
        }
        transition_rows.append(row)
    return transition_rows


def aggregate_case_rows(
    case_episode_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, str, float, str, str], list[dict[str, Any]]] = {}
    for row in case_episode_rows:
        key = (
            float(row["autotomy_threshold"]),
            str(row["formula_mode"]),
            float(row["rt_delta"]),
            str(row["policy"]),
            str(row["case"]),
        )
        grouped.setdefault(key, []).append(row)

    out_rows: list[dict[str, Any]] = []
    for (threshold, mode, rt_delta, policy, ret_case), rows in sorted(grouped.items()):
        case_count_total = sum(int(row["case_count"]) for row in rows)
        ret_total = sum(float(row["total_ret_contribution"]) for row in rows)
        weighted_fill = sum(
            float(row["mean_step_fill_rate"]) * int(row["case_count"])
            for row in rows
            if not np.isnan(float(row["mean_step_fill_rate"]))
        )
        weighted_disruption = sum(
            float(row["mean_disruption_fraction"]) * int(row["case_count"])
            for row in rows
            if not np.isnan(float(row["mean_disruption_fraction"]))
        )
        episode_steps = sum(int(row["episode_steps"]) for row in rows)
        out_rows.append(
            {
                "autotomy_threshold": threshold,
                "formula_mode": mode,
                "rt_delta": rt_delta,
                "policy": policy,
                "case": ret_case,
                "case_count": case_count_total,
                "case_pct": (
                    100.0 * case_count_total / episode_steps
                    if episode_steps > 0
                    else 0.0
                ),
                "mean_ret_value": (
                    ret_total / case_count_total
                    if case_count_total > 0
                    else float("nan")
                ),
                "total_ret_contribution": ret_total,
                "mean_step_fill_rate": (
                    weighted_fill / case_count_total
                    if case_count_total > 0
                    else float("nan")
                ),
                "mean_disruption_fraction": (
                    weighted_disruption / case_count_total
                    if case_count_total > 0
                    else float("nan")
                ),
            }
        )
    return out_rows


def aggregate_fill_rate_bucket_rows(
    bucket_episode_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, str, float, str, str], list[dict[str, Any]]] = {}
    for row in bucket_episode_rows:
        key = (
            float(row["autotomy_threshold"]),
            str(row["formula_mode"]),
            float(row["rt_delta"]),
            str(row["policy"]),
            str(row["fill_rate_bucket"]),
        )
        grouped.setdefault(key, []).append(row)

    out_rows: list[dict[str, Any]] = []
    for (threshold, mode, rt_delta, policy, bucket), rows in sorted(grouped.items()):
        case_count_total = sum(int(row["case_count"]) for row in rows)
        weighted_ret = sum(
            float(row["mean_ret_value"]) * int(row["case_count"])
            for row in rows
            if not np.isnan(float(row["mean_ret_value"]))
        )
        fill_rate_only_total = sum(int(row["fill_rate_only_total"]) for row in rows)
        out_rows.append(
            {
                "autotomy_threshold": threshold,
                "formula_mode": mode,
                "rt_delta": rt_delta,
                "policy": policy,
                "fill_rate_bucket": bucket,
                "case_count": case_count_total,
                "case_pct_of_fill_rate_only": (
                    100.0 * case_count_total / fill_rate_only_total
                    if fill_rate_only_total > 0
                    else 0.0
                ),
                "mean_ret_value": (
                    weighted_ret / case_count_total
                    if case_count_total > 0
                    else float("nan")
                ),
            }
        )
    return out_rows


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_ret_ablation(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    variants = build_variants(args)
    episode_rows: list[dict[str, Any]] = []
    case_episode_rows: list[dict[str, Any]] = []
    bucket_episode_rows: list[dict[str, Any]] = []

    for seed in args.seeds:
        for policy in POLICY_ORDER:
            for episode_idx in range(args.eval_episodes):
                rows, case_rows, bucket_rows = evaluate_static_rollout(
                    args=args,
                    policy=policy,
                    seed=seed,
                    episode_idx=episode_idx,
                    variants=variants,
                )
                episode_rows.extend(rows)
                case_episode_rows.extend(case_rows)
                bucket_episode_rows.extend(bucket_rows)

    seed_rows = aggregate_seed_metrics(episode_rows)
    policy_rows = aggregate_policy_metrics(seed_rows)
    transition_rows = build_transition_rows(policy_rows)
    case_summary_rows = aggregate_case_rows(case_episode_rows)
    fill_rate_bucket_rows = aggregate_fill_rate_bucket_rows(bucket_episode_rows)

    episode_csv = args.output_dir / "episode_metrics.csv"
    policy_csv = args.output_dir / "policy_summary.csv"
    transition_csv = args.output_dir / "transition_summary.csv"
    case_csv = args.output_dir / "case_summary.csv"
    bucket_csv = args.output_dir / "fill_rate_only_buckets.csv"
    summary_json = args.output_dir / "summary.json"

    save_csv(episode_csv, episode_rows)
    save_csv(policy_csv, policy_rows)
    save_csv(transition_csv, transition_rows)
    save_csv(case_csv, case_summary_rows)
    save_csv(bucket_csv, fill_rate_bucket_rows)

    summary = {
        "config": {
            "autotomy_thresholds": [float(v) for v in args.autotomy_thresholds],
            "formula_modes": list(args.formula_modes),
            "rt_deltas": [float(v) for v in args.rt_deltas],
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
            "transition_summary_csv": str(transition_csv),
            "case_summary_csv": str(case_csv),
            "fill_rate_only_buckets_csv": str(bucket_csv),
            "summary_json": str(summary_json),
        },
        "policy_summary": policy_rows,
        "transition_summary": transition_rows,
        "case_summary": case_summary_rows,
        "fill_rate_only_buckets": fill_rate_bucket_rows,
    }
    with summary_json.open("w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, indent=2)
    return summary


def main() -> None:
    args = build_parser().parse_args()
    summary = run_ret_ablation(args)
    print(f"Wrote ReT ablation artifacts to {args.output_dir}")
    for row in summary["transition_summary"]:
        print(
            "threshold="
            f"{row['autotomy_threshold']:.3f} | mode={row['formula_mode']} | "
            f"delta={row['rt_delta']:.3f} | preferred={row['preferred_policy_by_reward']} | "
            f"reward_gap_s2_minus_s1={row['reward_gap_s2_minus_s1']:.3f}"
        )


if __name__ == "__main__":
    main()
