#!/usr/bin/env python3
"""Build a paper-facing Track B audit bundle from a Track B run directory.

The Track B training runner already emits episode/seed/policy CSVs.  This script
turns those raw outputs into the artifacts needed for a claim audit:

- raw ReT, Pareto/resource, and tail verdicts;
- seed-paired deltas with CI95;
- static frontier comparison;
- data-gap report for Garrido-style APj/RPj/DPj/CTj ledger coverage;
- Excel workbook generated with the bundled artifact-tool runtime.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import statistics
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_RUN_DIR = Path(
    "outputs/experiments/track_b_gain_2026-06-29/"
    "kaggle_joint_confirm_50k_v5_output/track_b_joint_confirm_50k_3seed_h104"
)

DEFAULT_NODE = Path(
    "/Users/thom/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node"
)
DEFAULT_NODE_MODULES = Path(
    "/Users/thom/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules"
)

STATIC_PREFIXES = ("s1_", "s2_", "s3_")


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def ci95(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), float(values[0])
    mean = statistics.fmean(values)
    half = 1.96 * statistics.stdev(values) / math.sqrt(len(values))
    return float(mean - half), float(mean + half)


def tail_mean(values: list[float], *, frac: float, high_bad: bool) -> float:
    xs = sorted(float(v) for v in values if pd.notna(v))
    if not xs:
        return float("nan")
    n = max(1, math.ceil(len(xs) * frac))
    tail = xs[-n:] if high_bad else xs[:n]
    return float(statistics.fmean(tail))


def quantile(values: pd.Series, q: float) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return float("nan")
    return float(clean.quantile(q))


def lower_tail_mean(values: pd.Series, frac: float = 0.05) -> float:
    clean = sorted(float(v) for v in pd.to_numeric(values, errors="coerce").dropna())
    if not clean:
        return float("nan")
    n = max(1, math.ceil(len(clean) * frac))
    return float(statistics.fmean(clean[:n]))


def qty_weighted_mean(values: pd.Series, quantities: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce")
    qty = pd.to_numeric(quantities, errors="coerce").fillna(0.0)
    mask = vals.notna() & qty.notna()
    if not mask.any():
        return float("nan")
    denom = float(qty[mask].sum())
    if denom <= 0.0:
        return float("nan")
    return float((vals[mask] * qty[mask]).sum() / denom)


def direction_win(dynamic: float, static: float, direction: str) -> bool:
    if direction == "higher":
        return dynamic > static
    if direction == "lower":
        return dynamic < static
    raise ValueError(direction)


def direction_delta(dynamic: float, static: float, direction: str) -> float:
    return float(dynamic - static) if direction == "higher" else float(static - dynamic)


def get_policy_row(policy_summary: pd.DataFrame, policy: str) -> pd.Series:
    rows = policy_summary.loc[policy_summary["policy"] == policy]
    if rows.empty:
        raise KeyError(f"No policy={policy!r} in policy_summary")
    return rows.iloc[0]


def best_static_by(policy_summary: pd.DataFrame, metric_col: str, direction: str) -> str:
    statics = policy_summary[
        policy_summary["policy"].astype(str).str.startswith(STATIC_PREFIXES)
    ].copy()
    if direction == "higher":
        idx = statics[metric_col].astype(float).idxmax()
    else:
        idx = statics[metric_col].astype(float).idxmin()
    return str(statics.loc[idx, "policy"])


def metric_available(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns and df[col].notna().any()


def build_metric_panel(
    policy_summary: pd.DataFrame,
    *,
    learned_policy: str,
    best_static: str,
) -> list[dict[str, Any]]:
    metric_specs = [
        ("Order-level Excel ReT", "order_level_ret_mean_mean", "higher"),
        ("Flow fill rate", "flow_fill_rate_mean", "higher"),
        ("Flow backorder rate", "flow_backorder_rate_mean", "lower"),
        ("Rolling fill 4w", "terminal_rolling_fill_rate_4w_mean", "higher"),
        ("Rolling backorder 4w", "terminal_rolling_backorder_rate_4w_mean", "lower"),
        ("Reward total", "reward_total_mean", "higher"),
        ("Balanced CD sigmoid mean", "ret_garrido2024_sigmoid_mean_mean", "higher"),
        ("CD raw total", "ret_garrido2024_raw_total_mean", "higher"),
        ("CD train total", "ret_garrido2024_train_total_mean", "higher"),
        ("Terminal kappa_dot", "terminal_kappa_dot_mean", "lower"),
        ("Terminal epsilon/backlog", "terminal_epsilon_avg_mean", "lower"),
        ("Assembly cost index", "assembly_cost_index_mean", "lower"),
        ("Assembly hours", "assembly_hours_total_mean", "lower"),
        ("S1 share", "pct_steps_S1_mean", "neutral"),
        ("S2 share", "pct_steps_S2_mean", "neutral"),
        ("S3 share", "pct_steps_S3_mean", "neutral"),
        ("Op10 multiplier mean", "op10_multiplier_step_mean_mean", "neutral"),
        ("Op12 multiplier mean", "op12_multiplier_step_mean_mean", "neutral"),
        ("Op10 multiplier p95", "op10_multiplier_step_p95_mean", "neutral"),
        ("Op12 multiplier p95", "op12_multiplier_step_p95_mean", "neutral"),
        ("Order ret_excel", "order_ret_excel_mean", "higher"),
        ("Order ret_excel CVaR05", "order_ret_excel_cvar05_mean", "higher"),
        ("Order ret_excel p10", "order_ret_excel_p10_mean", "higher"),
        ("Order ret_excel p25", "order_ret_excel_p25_mean", "higher"),
        ("Order ret_excel p50", "order_ret_excel_p50_mean", "higher"),
        ("Order ret_excel p75", "order_ret_excel_p75_mean", "higher"),
        ("Order ret_excel p90", "order_ret_excel_p90_mean", "higher"),
        ("Order ret_excel rolling 4w mean", "order_ret_excel_rolling_4w_mean_mean", "higher"),
        ("Order ret_excel rolling 4w min", "order_ret_excel_rolling_4w_min_mean", "higher"),
        ("Ration-weighted Excel ReT", "order_ration_ret_excel_mean", "higher"),
        ("Excel case % autotomy", "order_excel_case_pct_autotomy_mean", "higher"),
        ("Excel case % recovery", "order_excel_case_pct_recovery_mean", "neutral"),
        ("Excel case % risk no recovery", "order_excel_case_pct_risk_no_recovery_mean", "lower"),
        ("Excel case % unfulfilled", "order_excel_case_pct_unfulfilled_mean", "lower"),
        ("Order lost rate", "order_lost_rate_mean", "lower"),
        ("Order service loss AUC/order", "order_service_loss_auc_per_order_mean", "lower"),
        ("Order APj p50", "order_apj_p50_mean", "higher"),
        ("Order APj p99", "order_apj_p99_mean", "higher"),
        ("Order APj positive p50", "order_apj_positive_p50_mean", "higher"),
        ("Order APj positive p99", "order_apj_positive_p99_mean", "higher"),
        ("Order CTj p90", "order_ctj_p90_mean", "lower"),
        ("Order CTj p99", "order_ctj_p99_mean", "lower"),
        ("Order RPj p90", "order_rpj_p90_mean", "lower"),
        ("Order RPj p99", "order_rpj_p99_mean", "lower"),
        ("Order DPj p99", "order_dpj_p99_mean", "lower"),
        ("Delivered rations", "order_delivered_rations_mean", "higher"),
        ("Demanded rations", "order_demanded_rations_mean", "neutral"),
    ]

    dyn = get_policy_row(policy_summary, learned_policy)
    sta = get_policy_row(policy_summary, best_static)
    rows: list[dict[str, Any]] = []
    for label, col, direction in metric_specs:
        if col not in policy_summary.columns:
            continue
        dval = float(dyn[col])
        sval = float(sta[col])
        delta = dval - sval
        if direction == "neutral":
            verdict = "mechanism"
            signed_win_delta = delta
        else:
            verdict = "dynamic_win" if direction_win(dval, sval, direction) else "static_win_or_tie"
            signed_win_delta = direction_delta(dval, sval, direction)
        rows.append(
            {
                "metric": label,
                "column": col,
                "direction": direction,
                "dynamic_policy": learned_policy,
                "static_policy": best_static,
                "dynamic": dval,
                "static": sval,
                "delta_dynamic_minus_static": delta,
                "signed_win_delta": signed_win_delta,
                "verdict": verdict,
            }
        )
    return rows


def summarize_ledger_policy(bucket: pd.DataFrame) -> dict[str, float]:
    total = max(1, int(len(bucket)))
    out: dict[str, float] = {
        "rows": float(len(bucket)),
    }
    if "ReTj" in bucket:
        out.update(
            {
                "ret_excel_mean": float(pd.to_numeric(bucket["ReTj"], errors="coerce").mean()),
                "ret_excel_cvar05": lower_tail_mean(bucket["ReTj"], 0.05),
                "ret_excel_p10": quantile(bucket["ReTj"], 0.10),
                "ret_excel_p25": quantile(bucket["ReTj"], 0.25),
                "ret_excel_p50": quantile(bucket["ReTj"], 0.50),
                "ret_excel_p75": quantile(bucket["ReTj"], 0.75),
                "ret_excel_p90": quantile(bucket["ReTj"], 0.90),
            }
        )
        if "Q" in bucket:
            out["ration_ret_excel"] = qty_weighted_mean(bucket["ReTj"], bucket["Q"])
    for col in ("APj", "RPj", "DPj", "CTj"):
        if col in bucket:
            key = col.lower()
            out[f"{key}_p50"] = quantile(bucket[col], 0.50)
            out[f"{key}_p90"] = quantile(bucket[col], 0.90)
            out[f"{key}_p99"] = quantile(bucket[col], 0.99)
            pos = pd.to_numeric(bucket[col], errors="coerce")
            pos = pos[pos > 0.0]
            out[f"{key}_positive_p50"] = quantile(pos, 0.50)
            out[f"{key}_positive_p99"] = quantile(pos, 0.99)
    if "case" in bucket:
        counts = bucket["case"].astype(str).value_counts()
        for case in (
            "excel_fill_rate", "excel_autotomy", "excel_recovery",
            "excel_risk_no_recovery", "excel_unfulfilled"
        ):
            out[f"pct_{case}"] = 100.0 * float(counts.get(case, 0)) / total
    if "lost" in bucket:
        out["lost_rate"] = float(pd.to_numeric(bucket["lost"], errors="coerce").fillna(0.0).mean())
    if "backorder" in bucket:
        out["backorder_rate"] = float(
            pd.to_numeric(bucket["backorder"], errors="coerce").fillna(0.0).mean()
        )
    return out


def build_ledger_metric_panel(
    order_ledger: pd.DataFrame | None,
    *,
    learned_policy: str,
    best_static: str,
) -> list[dict[str, Any]]:
    if order_ledger is None or order_ledger.empty:
        return []
    if "policy" not in order_ledger:
        return []
    groups = {str(policy): summarize_ledger_policy(bucket)
              for policy, bucket in order_ledger.groupby("policy")}
    if learned_policy not in groups or best_static not in groups:
        return []

    specs = [
        ("Ledger mean Excel ReTj", "ret_excel_mean", "higher"),
        ("Ledger Excel ReTj CVaR05", "ret_excel_cvar05", "higher"),
        ("Ledger ration-weighted Excel ReT", "ration_ret_excel", "higher"),
        ("Ledger Excel ReTj p10", "ret_excel_p10", "higher"),
        ("Ledger Excel ReTj p25", "ret_excel_p25", "higher"),
        ("Ledger Excel ReTj p50", "ret_excel_p50", "higher"),
        ("Ledger Excel ReTj p75", "ret_excel_p75", "higher"),
        ("Ledger Excel ReTj p90", "ret_excel_p90", "higher"),
        ("APj p50", "apj_p50", "higher"),
        ("APj p99", "apj_p99", "higher"),
        ("APj positive p50", "apj_positive_p50", "higher"),
        ("APj positive p99", "apj_positive_p99", "higher"),
        ("RPj p99", "rpj_p99", "lower"),
        ("DPj p99", "dpj_p99", "lower"),
        ("CTj p99", "ctj_p99", "lower"),
        ("Excel case pct autotomy", "pct_excel_autotomy", "higher"),
        ("Excel case pct recovery", "pct_excel_recovery", "neutral"),
        ("Excel case pct risk no recovery", "pct_excel_risk_no_recovery", "lower"),
        ("Excel case pct unfulfilled", "pct_excel_unfulfilled", "lower"),
        ("Lost rate", "lost_rate", "lower"),
        ("Backorder rate", "backorder_rate", "lower"),
    ]

    dyn = groups[learned_policy]
    sta = groups[best_static]
    rows: list[dict[str, Any]] = []
    for label, col, direction in specs:
        if col not in dyn or col not in sta:
            continue
        dval = float(dyn[col])
        sval = float(sta[col])
        if math.isnan(dval) or math.isnan(sval):
            continue
        if direction == "neutral":
            verdict = "mechanism"
            signed_win_delta = dval - sval
        else:
            verdict = "dynamic_win" if direction_win(dval, sval, direction) else "static_win_or_tie"
            signed_win_delta = direction_delta(dval, sval, direction)
        rows.append(
            {
                "metric": label,
                "column": col,
                "direction": direction,
                "dynamic_policy": learned_policy,
                "static_policy": best_static,
                "dynamic": dval,
                "static": sval,
                "delta_dynamic_minus_static": dval - sval,
                "signed_win_delta": signed_win_delta,
                "verdict": verdict,
            }
        )
    return rows


def build_tail_panel(
    episode_metrics: pd.DataFrame,
    *,
    learned_policy: str,
    best_static: str,
) -> list[dict[str, Any]]:
    specs = [
        ("ReT CVaR05", "order_level_ret_mean", "higher", 0.05, False),
        ("Order Excel ReT CVaR05", "order_ret_excel", "higher", 0.05, False),
        ("Order Excel ReT rolling 4w CVaR05", "order_ret_excel_rolling_4w_mean", "higher", 0.05, False),
        ("Flow fill CVaR05", "flow_fill_rate", "higher", 0.05, False),
        ("Rolling fill CVaR05", "terminal_rolling_fill_rate_4w", "higher", 0.05, False),
        ("Flow backorder CVaR95", "flow_backorder_rate", "lower", 0.05, True),
        (
            "Rolling backorder CVaR95",
            "terminal_rolling_backorder_rate_4w",
            "lower",
            0.05,
            True,
        ),
        ("Cost index CVaR95", "assembly_cost_index", "lower", 0.05, True),
        ("Order lost-rate CVaR95", "order_lost_rate", "lower", 0.05, True),
        ("Balanced CD CVaR05", "ret_garrido2024_sigmoid_mean", "higher", 0.05, False),
        (
            "Order service-loss CVaR95",
            "order_service_loss_auc_per_order",
            "lower",
            0.05,
            True,
        ),
        ("Order CTj p99 CVaR95", "order_ctj_p99", "lower", 0.05, True),
        ("Order RPj p99 CVaR95", "order_rpj_p99", "lower", 0.05, True),
    ]
    rows: list[dict[str, Any]] = []
    for label, col, direction, frac, high_bad in specs:
        if col not in episode_metrics.columns:
            continue
        dyn_vals = episode_metrics.loc[episode_metrics["policy"] == learned_policy, col].tolist()
        sta_vals = episode_metrics.loc[episode_metrics["policy"] == best_static, col].tolist()
        dval = tail_mean(dyn_vals, frac=frac, high_bad=high_bad)
        sval = tail_mean(sta_vals, frac=frac, high_bad=high_bad)
        if math.isnan(dval) or math.isnan(sval):
            continue
        rows.append(
            {
                "metric": label,
                "column": col,
                "direction": direction,
                "dynamic": dval,
                "static": sval,
                "delta_dynamic_minus_static": dval - sval,
                "signed_win_delta": direction_delta(dval, sval, direction),
                "verdict": "dynamic_win"
                if direction_win(dval, sval, direction)
                else "static_win_or_tie",
            }
        )
    return rows


def build_seed_delta_panel(
    seed_metrics: pd.DataFrame,
    *,
    learned_policy: str,
    best_static: str,
) -> list[dict[str, Any]]:
    specs = [
        ("Order-level Excel ReT", "order_level_ret_mean_mean", "higher"),
        ("Order Excel ReT", "order_ret_excel_mean", "higher"),
        ("Order Excel ReT CVaR05", "order_ret_excel_cvar05_mean", "higher"),
        ("Flow fill rate", "flow_fill_rate_mean", "higher"),
        ("Rolling fill 4w", "terminal_rolling_fill_rate_4w_mean", "higher"),
        ("Flow backorder rate", "flow_backorder_rate_mean", "lower"),
        ("Assembly cost index", "assembly_cost_index_mean", "lower"),
        ("Balanced CD sigmoid mean", "ret_garrido2024_sigmoid_mean_mean", "higher"),
        ("Ration-weighted Excel ReT", "order_ration_ret_excel_mean", "higher"),
        ("Order lost rate", "order_lost_rate_mean", "lower"),
        ("Order service loss AUC/order", "order_service_loss_auc_per_order_mean", "lower"),
    ]
    rows: list[dict[str, Any]] = []
    dyn = seed_metrics.loc[seed_metrics["policy"] == learned_policy].copy()
    sta = seed_metrics.loc[seed_metrics["policy"] == best_static].copy()
    common_seeds = sorted(set(dyn["seed"]).intersection(set(sta["seed"])))
    for label, col, direction in specs:
        if col not in seed_metrics.columns:
            continue
        deltas: list[float] = []
        for seed in common_seeds:
            dval = float(dyn.loc[dyn["seed"] == seed, col].iloc[0])
            sval = float(sta.loc[sta["seed"] == seed, col].iloc[0])
            deltas.append(direction_delta(dval, sval, direction))
        low, high = ci95(deltas)
        rows.append(
            {
                "metric": label,
                "column": col,
                "direction": direction,
                "n_paired_seeds": len(deltas),
                "mean_signed_win_delta": statistics.fmean(deltas) if deltas else float("nan"),
                "ci95_low": low,
                "ci95_high": high,
                "ci95_clears_zero": bool(low > 0.0) if not math.isnan(low) else False,
            }
        )
    return rows


def build_verdicts(
    policy_summary: pd.DataFrame,
    tail_panel: list[dict[str, Any]],
    *,
    learned_policy: str,
    best_static: str,
) -> list[dict[str, Any]]:
    dyn = get_policy_row(policy_summary, learned_policy)
    sta = get_policy_row(policy_summary, best_static)
    ret_dyn = float(dyn["order_level_ret_mean_mean"])
    ret_sta = float(sta["order_level_ret_mean_mean"])
    cost_dyn = float(dyn["assembly_cost_index_mean"])
    cost_sta = float(sta["assembly_cost_index_mean"])
    ret_cvar = next((r for r in tail_panel if r["metric"] == "ReT CVaR05"), None)
    ret_cvar_win = bool(ret_cvar and ret_cvar["verdict"] == "dynamic_win")

    static_rows = policy_summary[
        policy_summary["policy"].astype(str).str.startswith(STATIC_PREFIXES)
    ]
    dominated_by: list[str] = []
    for _, row in static_rows.iterrows():
        if (
            float(row["order_level_ret_mean_mean"]) >= ret_dyn
            and float(row["assembly_cost_index_mean"]) <= cost_dyn
            and str(row["policy"]) != best_static
        ):
            dominated_by.append(str(row["policy"]))

    return [
        {
            "verdict": "raw_ret_win",
            "pass": ret_dyn > ret_sta,
            "dynamic": ret_dyn,
            "static": ret_sta,
            "delta": ret_dyn - ret_sta,
            "note": "Primary resilience bar; ignores resource.",
        },
        {
            "verdict": "ret_tail_win",
            "pass": ret_cvar_win,
            "dynamic": None if ret_cvar is None else ret_cvar["dynamic"],
            "static": None if ret_cvar is None else ret_cvar["static"],
            "delta": None if ret_cvar is None else ret_cvar["delta_dynamic_minus_static"],
            "note": "Lower-tail order-level ReT.",
        },
        {
            "verdict": "resource_efficient_win",
            "pass": ret_dyn >= ret_sta and cost_dyn <= cost_sta,
            "dynamic": cost_dyn,
            "static": cost_sta,
            "delta": cost_dyn - cost_sta,
            "note": "ReT at least as high and assembly cost index no higher.",
        },
        {
            "verdict": "pareto_ret_cost",
            "pass": len(dominated_by) == 0 and ret_dyn >= ret_sta,
            "dynamic": len(dominated_by),
            "static": 0,
            "delta": -len(dominated_by),
            "note": "No static has >= ReT and <= cost; dominated_by="
            + (",".join(dominated_by) if dominated_by else "none"),
        },
    ]


def build_static_frontier(policy_summary: pd.DataFrame) -> list[dict[str, Any]]:
    static_rows = policy_summary[
        policy_summary["policy"].astype(str).str.startswith(STATIC_PREFIXES)
    ].copy()
    keep = [
        "policy",
        "order_level_ret_mean_mean",
        "order_level_ret_mean_ci95_low",
        "order_level_ret_mean_ci95_high",
        "flow_fill_rate_mean",
        "order_ret_excel_mean",
        "order_ret_excel_cvar05_mean",
        "order_ret_excel_rolling_4w_mean_mean",
        "order_ration_ret_excel_mean",
        "order_apj_p99_mean",
        "terminal_rolling_fill_rate_4w_mean",
        "assembly_cost_index_mean",
        "ret_garrido2024_sigmoid_mean_mean",
        "reward_total_mean",
        "pct_steps_S1_mean",
        "pct_steps_S2_mean",
        "pct_steps_S3_mean",
        "op10_multiplier_step_mean_mean",
        "op12_multiplier_step_mean_mean",
    ]
    keep = [col for col in keep if col in static_rows.columns]
    static_rows = static_rows.sort_values(
        ["order_level_ret_mean_mean", "assembly_cost_index_mean"],
        ascending=[False, True],
    )
    return static_rows[keep].to_dict("records")


def build_external_static_frontier(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    df = pd.read_csv(path)
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        shift = row.get("shift")
        op10 = row.get("op10_mult")
        op12 = row.get("op12_mult")
        policy = row.get("policy") if "policy" in row.index else None
        if policy is None or pd.isna(policy):
            policy = f"S{int(shift)}_op10_{float(op10):.2f}_op12_{float(op12):.2f}"
        rows.append(
            {
                "policy": str(policy),
                "shift": None if pd.isna(shift) else int(shift),
                "op10_mult": None if pd.isna(op10) else float(op10),
                "op12_mult": None if pd.isna(op12) else float(op12),
                "order_level_ret_mean_mean": float(
                    row.get("order_level_ret_mean_mean", row.get("ret_excel_mean", float("nan")))
                ),
                "flow_fill_rate_mean": float(
                    row.get("flow_fill_rate_mean", row.get("flow_fill_mean", float("nan")))
                ),
                "order_lost_rate_mean": float(
                    row.get("order_lost_rate_mean", row.get("lost_rate_mean", float("nan")))
                ),
                "assembly_cost_index_mean": float(
                    row.get("assembly_cost_index_mean", float("nan"))
                ),
                "ret_garrido2024_sigmoid_mean_mean": float(
                    row.get("ret_garrido2024_sigmoid_mean_mean", float("nan"))
                ),
                "order_service_loss_auc_per_order_mean": float(
                    row.get("order_service_loss_auc_per_order_mean", float("nan"))
                ),
                "order_ctj_p99_mean": float(row.get("order_ctj_p99_mean", float("nan"))),
                "order_rpj_p99_mean": float(row.get("order_rpj_p99_mean", float("nan"))),
                "source": str(path),
            }
        )
    rows.sort(key=lambda r: (r["order_level_ret_mean_mean"], r.get("flow_fill_rate_mean", 0.0)), reverse=True)
    return rows


def build_external_dense_verdicts(
    policy_summary: pd.DataFrame,
    external_frontier: list[dict[str, Any]],
    *,
    learned_policy: str,
) -> list[dict[str, Any]]:
    if not external_frontier:
        return []
    dyn = get_policy_row(policy_summary, learned_policy)
    ret_dyn = float(dyn["order_level_ret_mean_mean"])
    cost_dyn = float(dyn.get("assembly_cost_index_mean", float("nan")))
    best = external_frontier[0]
    ret_sta = float(best["order_level_ret_mean_mean"])
    verdicts = [
        {
            "verdict": "raw_ret_win_vs_external_dense",
            "pass": ret_dyn > ret_sta,
            "dynamic": ret_dyn,
            "static": ret_sta,
            "delta": ret_dyn - ret_sta,
            "note": f"External dense frontier best={best['policy']}.",
        }
    ]
    if not math.isnan(float(best.get("assembly_cost_index_mean", float("nan")))) and not math.isnan(cost_dyn):
        cost_sta = float(best["assembly_cost_index_mean"])
        dominated_by = [
            str(row["policy"])
            for row in external_frontier
            if not math.isnan(float(row.get("assembly_cost_index_mean", float("nan"))))
            and float(row["order_level_ret_mean_mean"]) >= ret_dyn
            and float(row["assembly_cost_index_mean"]) <= cost_dyn
        ]
        verdicts.extend(
            [
                {
                    "verdict": "resource_efficient_win_vs_external_dense",
                    "pass": ret_dyn >= ret_sta and cost_dyn <= cost_sta,
                    "dynamic": cost_dyn,
                    "static": cost_sta,
                    "delta": cost_dyn - cost_sta,
                    "note": f"External dense best-by-ReT={best['policy']}.",
                },
                {
                    "verdict": "pareto_ret_cost_vs_external_dense",
                    "pass": len(dominated_by) == 0 and ret_dyn >= ret_sta,
                    "dynamic": len(dominated_by),
                    "static": 0,
                    "delta": -len(dominated_by),
                    "note": "No external static has >= ReT and <= cost; dominated_by="
                    + (",".join(dominated_by[:10]) if dominated_by else "none"),
                },
            ]
        )
    if not math.isnan(float(best.get("ret_garrido2024_sigmoid_mean_mean", float("nan")))):
        cd_dyn = float(dyn.get("ret_garrido2024_sigmoid_mean_mean", float("nan")))
        cd_sta = float(best["ret_garrido2024_sigmoid_mean_mean"])
        verdicts.append(
            {
                "verdict": "cd_sigmoid_win_vs_external_dense_best_ret",
                "pass": cd_dyn > cd_sta,
                "dynamic": cd_dyn,
                "static": cd_sta,
                "delta": cd_dyn - cd_sta,
                "note": f"CD comparison against external dense best-by-ReT={best['policy']}.",
            }
        )
    return verdicts


def build_data_gaps(episode_metrics: pd.DataFrame, order_ledger: pd.DataFrame | None) -> list[dict[str, Any]]:
    has_ledger = order_ledger is not None and not order_ledger.empty
    required = {
        "APj": ("APj", ("APj",)),
        "APj_p50_p99": ("order_apj_p99", ("APj",)),
        "RPj": ("RPj", ("RPj",)),
        "DPj": ("DPj", ("DPj",)),
        "CTj": ("CTj", ("CTj",)),
        "ret_excel_distribution": ("order_ret_excel_p10", ("ReTj",)),
        "ret_excel_cvar05": ("order_ret_excel_cvar05", ("ReTj",)),
        "ret_excel_case_breakdown": ("order_excel_case_pct_recovery", ("case",)),
        "ration_weighted_ret_excel": ("order_ration_ret_excel", ("ReTj", "Q")),
        "rolling_4w_ret_excel": ("order_ret_excel_rolling_4w_mean", ("ReTj", "OPTj")),
        "lost_rate": ("order_lost_rate", ("lost",)),
        "service_loss_auc": ("order_service_loss_auc_per_order", ()),
        "ctj_p99": ("order_ctj_p99", ("CTj",)),
        "rpj_p99": ("order_rpj_p99", ("RPj",)),
        "dpj_p99": ("order_dpj_p99", ("DPj",)),
        "variance_log_CD_sigmoid": ("ret_garrido2024_sigmoid_mean", ()),
        "CD_kappa_dot": ("terminal_kappa_dot", ()),
    }
    rows: list[dict[str, Any]] = []
    for label, (col, ledger_sources) in required.items():
        in_episode = col in episode_metrics.columns
        in_ledger = has_ledger and col in order_ledger.columns
        reconstructable = has_ledger and all(src in order_ledger.columns for src in ledger_sources)
        ok = bool(in_episode or in_ledger or reconstructable)
        rows.append(
            {
                "requested_metric": label,
                "available_in_episode_metrics": bool(in_episode),
                "available_in_order_ledger": bool(in_ledger),
                "reconstructable_from_order_ledger": bool(reconstructable),
                "column_or_status": (
                    col if ok else "missing in this run"
                ),
                "action": (
                    "OK"
                    if ok
                    else "rerun Track B with updated rich metrics / order-ledger export"
                ),
            }
        )
    return rows


def build_ledger_summary(order_ledger: pd.DataFrame | None, order_ledger_csv: Path) -> list[dict[str, Any]]:
    if order_ledger is None or order_ledger.empty:
        return [
            {
                "status": "missing",
                "rows": 0,
                "csv": str(order_ledger_csv),
                "note": "Run Track B with --export-order-ledger to populate APj/RPj/DPj/CTj rows.",
            }
        ]
    rows: list[dict[str, Any]] = []
    for policy, bucket in order_ledger.groupby("policy"):
        summary = summarize_ledger_policy(bucket)
        rows.append(
            {
                "status": "available",
                "policy": str(policy),
                "rows": int(len(bucket)),
                "mean_ReTj": summary.get("ret_excel_mean", float("nan")),
                "ret_excel_cvar05": summary.get("ret_excel_cvar05", float("nan")),
                "ret_excel_p10": summary.get("ret_excel_p10", float("nan")),
                "ret_excel_p90": summary.get("ret_excel_p90", float("nan")),
                "ration_ret_excel": summary.get("ration_ret_excel", float("nan")),
                "pct_excel_autotomy": summary.get("pct_excel_autotomy", float("nan")),
                "pct_excel_recovery": summary.get("pct_excel_recovery", float("nan")),
                "pct_excel_risk_no_recovery": summary.get("pct_excel_risk_no_recovery", float("nan")),
                "pct_excel_unfulfilled": summary.get("pct_excel_unfulfilled", float("nan")),
                "lost_rate": summary.get("lost_rate", float("nan")),
                "backorder_rate": summary.get("backorder_rate", float("nan")),
                "apj_p50": summary.get("apj_p50", float("nan")),
                "apj_p99": summary.get("apj_p99", float("nan")),
                "apj_positive_p50": summary.get("apj_positive_p50", float("nan")),
                "apj_positive_p99": summary.get("apj_positive_p99", float("nan")),
                "ctj_p99": summary.get("ctj_p99", float("nan")),
                "rpj_p99": summary.get("rpj_p99", float("nan")),
                "dpj_p99": summary.get("dpj_p99", float("nan")),
                "csv": str(order_ledger_csv),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def json_clean(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {str(k): json_clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_clean(v) for v in value]
    return value


def run_workbook_builder(payload_path: Path, workbook_path: Path, *, node: Path, node_modules: Path) -> None:
    builder_src = Path(__file__).with_name("build_track_b_top_tier_workbook.mjs")
    if not builder_src.exists():
        raise FileNotFoundError(builder_src)
    if not node.exists():
        raise FileNotFoundError(node)
    if not node_modules.exists():
        raise FileNotFoundError(node_modules)
    with tempfile.TemporaryDirectory(prefix="track_b_workbook_") as tmp_raw:
        tmp = Path(tmp_raw)
        (tmp / "node_modules").symlink_to(node_modules, target_is_directory=True)
        builder = tmp / builder_src.name
        shutil.copy2(builder_src, builder)
        subprocess.run(
            [str(node), str(builder), str(payload_path.resolve()), str(workbook_path.resolve())],
            check=True,
            cwd=tmp,
        )


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--learned-policy", default="ppo")
    ap.add_argument("--best-static-policy", default=None)
    ap.add_argument("--node", type=Path, default=DEFAULT_NODE)
    ap.add_argument("--node-modules", type=Path, default=DEFAULT_NODE_MODULES)
    ap.add_argument(
        "--ledger-max-rows",
        type=int,
        default=0,
        help="Rows to read from order_ledger.csv; <=0 reads all rows.",
    )
    ap.add_argument("--ledger-sample-rows", type=int, default=10_000)
    ap.add_argument(
        "--external-static-frontier",
        type=Path,
        default=None,
        help="Optional dense static frontier CSV to audit raw ReT against separately.",
    )
    ap.add_argument("--no-workbook", action="store_true")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    run_dir = args.run_dir
    output_dir = args.output_dir or (
        Path("outputs/audits")
        / f"track_b_top_tier_audit_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    episode = read_csv(run_dir / "episode_metrics.csv")
    seed = read_csv(run_dir / "seed_metrics.csv")
    policy = read_csv(run_dir / "policy_summary.csv")
    order_ledger_csv = run_dir / "order_ledger.csv"
    if order_ledger_csv.exists():
        nrows = None if int(args.ledger_max_rows) <= 0 else max(0, int(args.ledger_max_rows))
        order_ledger = pd.read_csv(order_ledger_csv, nrows=nrows)
    else:
        order_ledger = None
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

    learned = str(args.learned_policy)
    best_static = args.best_static_policy or best_static_by(
        policy, "order_level_ret_mean_mean", "higher"
    )

    metric_panel = build_metric_panel(policy, learned_policy=learned, best_static=best_static)
    ledger_metric_panel = build_ledger_metric_panel(
        order_ledger, learned_policy=learned, best_static=best_static
    )
    tail_panel = build_tail_panel(episode, learned_policy=learned, best_static=best_static)
    seed_delta_panel = build_seed_delta_panel(seed, learned_policy=learned, best_static=best_static)
    verdicts = build_verdicts(policy, tail_panel, learned_policy=learned, best_static=best_static)
    static_frontier = build_static_frontier(policy)
    external_static_frontier = build_external_static_frontier(args.external_static_frontier)
    verdicts.extend(
        build_external_dense_verdicts(
            policy,
            external_static_frontier,
            learned_policy=learned,
        )
    )
    data_gaps = build_data_gaps(episode, order_ledger)
    ledger_summary = build_ledger_summary(order_ledger, order_ledger_csv)
    ledger_sample = (
        order_ledger.head(max(0, int(args.ledger_sample_rows))).to_dict("records")
        if order_ledger is not None
        else []
    )

    payload = {
        "run_dir": str(run_dir.resolve()),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": summary.get("config", {}),
        "learned_policy": learned,
        "best_static_policy": best_static,
        "verdicts": verdicts,
        "metric_panel": metric_panel,
        "ledger_metric_panel": ledger_metric_panel,
        "tail_panel": tail_panel,
        "seed_delta_panel": seed_delta_panel,
        "static_frontier": static_frontier,
        "external_static_frontier": external_static_frontier,
        "data_gaps": data_gaps,
        "ledger_summary": ledger_summary,
        "order_ledger_sample": ledger_sample,
        "policy_summary": policy.to_dict("records"),
        "seed_metrics": seed.to_dict("records"),
    }

    write_csv(output_dir / "metric_panel.csv", metric_panel)
    write_csv(output_dir / "ledger_metric_panel.csv", ledger_metric_panel)
    write_csv(output_dir / "tail_panel.csv", tail_panel)
    write_csv(output_dir / "seed_delta_panel.csv", seed_delta_panel)
    write_csv(output_dir / "verdicts.csv", verdicts)
    write_csv(output_dir / "static_frontier.csv", static_frontier)
    write_csv(output_dir / "external_static_frontier.csv", external_static_frontier)
    write_csv(output_dir / "data_gaps.csv", data_gaps)
    write_csv(output_dir / "ledger_summary.csv", ledger_summary)
    write_csv(output_dir / "order_ledger_sample.csv", ledger_sample)
    payload_path = output_dir / "track_b_top_tier_audit.json"
    payload_path.write_text(json.dumps(json_clean(payload), indent=2), encoding="utf-8")

    workbook_path = output_dir / "track_b_top_tier_audit.xlsx"
    if not args.no_workbook:
        run_workbook_builder(payload_path, workbook_path, node=args.node, node_modules=args.node_modules)

    report = [
        "# Track B Top-Tier Audit",
        "",
        f"- Run dir: `{run_dir}`",
        f"- Learned policy: `{learned}`",
        f"- Best static by ReT: `{best_static}`",
        "",
        "## Verdicts",
        "",
        "| Verdict | Pass | Dynamic | Static | Delta | Note |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in verdicts:
        report.append(
            f"| {row['verdict']} | {row['pass']} | {row['dynamic']} | "
            f"{row['static']} | {row['delta']} | {row['note']} |"
        )
    report.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- JSON: `{payload_path}`",
            f"- Workbook: `{workbook_path if workbook_path.exists() else 'not built'}`",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(report), encoding="utf-8")
    print(f"WROTE {output_dir}")
    for row in verdicts:
        print(f"{row['verdict']}: {row['pass']} delta={row['delta']} note={row['note']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
