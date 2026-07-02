#!/usr/bin/env python3
"""Build Cohen's d tables and Pareto figures for the canonical Track B run.

The primary bar is Garrido/Excel ReT (`order_ret_excel`).  The script compares
the learned PPO policy against the best dense static policy under common eval
seeds, then exports paper-facing figures for the dense static frontier.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_RUN_DIR = Path(
    "outputs/experiments/track_b_gain_2026-06-30/"
    "top_tier_confirm_v3_output/track_b_top_tier_confirm_5seed_60k_h104"
)
DEFAULT_DENSE_DIR = Path("outputs/experiments/track_b_dense_crn_static_2026-06-30")
DEFAULT_OUT_DIR = Path("outputs/audits/track_b_q1_stats_2026-07-01")

PRIMARY_METRIC = "order_ret_excel"
STATIC_PREFIXES = ("s1_", "s2_", "s3_", "S1_", "S2_", "S3_")

LOWER_IS_BETTER = {
    "assembly_cost_index",
    "order_service_loss_auc_per_order",
    "order_backorder_qty_final",
    "order_lost_rate",
    "order_ctj_p99",
    "order_rpj_p99",
    "order_dpj_p99",
    "flow_backorder_rate",
    "terminal_rolling_backorder_rate_4w",
}

METRICS = [
    ("order_ret_excel", "higher", "Excel ReT"),
    ("order_level_ret_mean", "higher", "Order-level ReT"),
    ("flow_fill_rate", "higher", "Flow fill rate"),
    ("terminal_rolling_fill_rate_4w", "higher", "Rolling fill 4w"),
    ("ret_garrido2024_sigmoid_mean", "higher", "Balanced CD sigmoid"),
    ("assembly_cost_index", "lower", "Cost index"),
    ("order_service_loss_auc_per_order", "lower", "Service-loss AUC/order"),
    ("order_backorder_qty_final", "lower", "Final backlog qty"),
    ("order_lost_rate", "lower", "Lost rate"),
    ("order_ctj_p99", "lower", "CTj p99"),
    ("order_rpj_p99", "lower", "RPj p99"),
    ("order_dpj_p99", "lower", "DPj p99"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--dense-dir", type=Path, default=DEFAULT_DENSE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--learned-policy", default="ppo")
    parser.add_argument("--primary-metric", default=PRIMARY_METRIC)
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def metric_mean_col(metric: str) -> str:
    return f"{metric}_mean"


def best_static_policy(summary: pd.DataFrame, metric: str, direction: str) -> str:
    statics = summary[summary["policy"].astype(str).str.startswith(STATIC_PREFIXES)].copy()
    if statics.empty:
        raise ValueError("No static policies found in dense summary")
    col = metric_mean_col(metric)
    if col not in statics.columns:
        raise KeyError(f"Missing metric column {col}")
    values = pd.to_numeric(statics[col], errors="coerce")
    idx = values.idxmax() if direction == "higher" else values.idxmin()
    return str(statics.loc[idx, "policy"])


def paired_rows(
    learned: pd.DataFrame,
    static: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    keys = ["seed", "episode", "eval_seed"]
    lhs = learned[keys + [metric]].rename(columns={metric: "learned"})
    rhs = static[keys + [metric]].rename(columns={metric: "static"})
    pairs = lhs.merge(rhs, on=keys, how="inner")
    pairs["learned"] = pd.to_numeric(pairs["learned"], errors="coerce")
    pairs["static"] = pd.to_numeric(pairs["static"], errors="coerce")
    return pairs.dropna(subset=["learned", "static"])


def bootstrap_ci(values: np.ndarray, *, reps: int = 5000, seed: int = 20260701) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(reps, values.size))
    means = values[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def cohens_d_unpaired(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size < 2 or y.size < 2:
        return float("nan")
    sx = x.var(ddof=1)
    sy = y.var(ddof=1)
    pooled = ((x.size - 1) * sx + (y.size - 1) * sy) / (x.size + y.size - 2)
    if pooled <= 0:
        return float("nan")
    return float((x.mean() - y.mean()) / math.sqrt(pooled))


def cohens_d_paired(delta: np.ndarray) -> float:
    delta = np.asarray(delta, dtype=float)
    delta = delta[np.isfinite(delta)]
    if delta.size < 2:
        return float("nan")
    sd = delta.std(ddof=1)
    if sd <= 0:
        return float("nan")
    return float(delta.mean() / sd)


def directional_delta(metric: str, learned: np.ndarray, static: np.ndarray) -> np.ndarray:
    delta = learned - static
    return -delta if metric in LOWER_IS_BETTER else delta


def build_effect_sizes(
    learned_rows: pd.DataFrame,
    static_rows: pd.DataFrame,
    best_static: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metric, direction, label in METRICS:
        if metric not in learned_rows.columns or metric not in static_rows.columns:
            continue
        pairs = paired_rows(learned_rows, static_rows, metric)
        if pairs.empty:
            continue
        raw_delta = pairs["learned"].to_numpy(dtype=float) - pairs["static"].to_numpy(dtype=float)
        oriented = directional_delta(metric, pairs["learned"].to_numpy(dtype=float), pairs["static"].to_numpy(dtype=float))
        lo, hi = bootstrap_ci(raw_delta)
        oriented_lo, oriented_hi = bootstrap_ci(oriented)
        learned_mean = float(pairs["learned"].mean())
        static_mean = float(pairs["static"].mean())
        rows.append(
            {
                "metric": metric,
                "label": label,
                "direction": direction,
                "learned_policy": "ppo",
                "static_policy": best_static,
                "n_pairs": int(len(pairs)),
                "ppo_mean": learned_mean,
                "static_mean": static_mean,
                "raw_delta_ppo_minus_static": float(raw_delta.mean()),
                "raw_delta_ci95_low": lo,
                "raw_delta_ci95_high": hi,
                "directional_gain": float(oriented.mean()),
                "directional_gain_ci95_low": oriented_lo,
                "directional_gain_ci95_high": oriented_hi,
                "paired_cohens_d": cohens_d_paired(raw_delta),
                "oriented_paired_cohens_d": cohens_d_paired(oriented),
                "unpaired_cohens_d": cohens_d_unpaired(
                    pairs["learned"].to_numpy(dtype=float),
                    pairs["static"].to_numpy(dtype=float),
                ),
                "win": bool(oriented.mean() > 0.0),
                "ci95_directional_win": bool(oriented_lo > 0.0),
            }
        )
    return pd.DataFrame(rows)


def cvar_tail(values: np.ndarray, *, frac: float = 0.05, lower_tail: bool = True) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    values = np.sort(values)
    n = max(1, int(math.ceil(values.size * frac)))
    tail = values[:n] if lower_tail else values[-n:]
    return float(tail.mean())


def build_cvar05_effect_row(
    learned_rows: pd.DataFrame,
    static_rows: pd.DataFrame,
    best_static: str,
) -> dict[str, Any]:
    keys = ["seed", "episode", "eval_seed"]
    pairs = learned_rows[keys + [PRIMARY_METRIC]].rename(
        columns={PRIMARY_METRIC: "learned"}
    ).merge(
        static_rows[keys + [PRIMARY_METRIC]].rename(columns={PRIMARY_METRIC: "static"}),
        on=keys,
        how="inner",
    )
    learned = pairs["learned"].to_numpy(dtype=float)
    static = pairs["static"].to_numpy(dtype=float)
    learned_cvar = cvar_tail(learned, frac=0.05, lower_tail=True)
    static_cvar = cvar_tail(static, frac=0.05, lower_tail=True)
    rng = np.random.default_rng(20260701)
    reps = []
    if len(pairs) > 0:
        for _ in range(5000):
            idx = rng.integers(0, len(pairs), size=len(pairs))
            reps.append(
                cvar_tail(learned[idx], frac=0.05, lower_tail=True)
                - cvar_tail(static[idx], frac=0.05, lower_tail=True)
            )
    lo, hi = np.quantile(reps, [0.025, 0.975]) if reps else (float("nan"), float("nan"))
    return {
        "metric": "order_ret_excel_cvar05",
        "label": "Excel ReT CVaR05",
        "definition": "conditional mean of the lowest 5% of per-episode order_ret_excel",
        "learned_policy": "ppo",
        "static_policy": best_static,
        "n_pairs": int(len(pairs)),
        "ppo_mean": learned_cvar,
        "static_mean": static_cvar,
        "raw_delta_ppo_minus_static": float(learned_cvar - static_cvar),
        "raw_delta_ci95_low": float(lo),
        "raw_delta_ci95_high": float(hi),
        "direction": "higher",
        "win": bool(learned_cvar > static_cvar),
        "ci95_directional_win": bool(float(lo) > 0.0),
    }


def build_seed_level_inference(
    learned_rows: pd.DataFrame,
    static_rows: pd.DataFrame,
) -> pd.DataFrame:
    pairs = paired_rows(learned_rows, static_rows, PRIMARY_METRIC)
    rows = []
    for seed, bucket in pairs.groupby("seed", sort=True):
        delta = bucket["learned"].to_numpy(dtype=float) - bucket["static"].to_numpy(dtype=float)
        rows.append(
            {
                "seed": int(seed),
                "episode_count": int(len(bucket)),
                "mean_delta": float(delta.mean()),
                "min_delta": float(delta.min()),
                "positive_pairs": int((delta > 0.0).sum()),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        seed_deltas = out["mean_delta"].to_numpy(dtype=float)
        lo, hi = bootstrap_ci(seed_deltas, reps=5000, seed=20260702)
        out.attrs["summary"] = {
            "seed_count": int(len(out)),
            "all_seed_deltas_positive": bool((seed_deltas > 0.0).all()),
            "mean_seed_delta": float(seed_deltas.mean()),
            "seed_cluster_bootstrap_ci95_low": lo,
            "seed_cluster_bootstrap_ci95_high": hi,
        }
    return out


def build_top_static_robustness(
    learned_rows: pd.DataFrame,
    dense_episode: pd.DataFrame,
    dense_summary: pd.DataFrame,
    *,
    top_n: int = 12,
) -> pd.DataFrame:
    statics = dense_summary[dense_summary["policy"].astype(str).str.startswith(STATIC_PREFIXES)].copy()
    statics = statics.sort_values(f"{PRIMARY_METRIC}_mean", ascending=False).head(top_n)
    rows = []
    for rank, (_, static_summary) in enumerate(statics.iterrows(), start=1):
        label = str(static_summary["policy"])
        static_rows = dense_episode[dense_episode["policy"].astype(str) == label].copy()
        pairs = paired_rows(learned_rows, static_rows, PRIMARY_METRIC)
        delta = pairs["learned"].to_numpy(dtype=float) - pairs["static"].to_numpy(dtype=float)
        lo, hi = bootstrap_ci(delta, seed=20260701 + rank)
        rows.append(
            {
                "rank": rank,
                "static_policy": label,
                "static_order_ret_excel_mean": float(static_summary[f"{PRIMARY_METRIC}_mean"]),
                "n_pairs": int(len(pairs)),
                "ppo_delta_mean": float(delta.mean()) if len(delta) else float("nan"),
                "delta_ci95_low": lo,
                "delta_ci95_high": hi,
                "all_pairs_positive": bool((delta > 0.0).all()) if len(delta) else False,
            }
        )
    return pd.DataFrame(rows)


def build_dispatch_cost_sensitivity(
    learned_rows: pd.DataFrame,
    static_rows: pd.DataFrame,
    *,
    charges: list[float] | None = None,
) -> pd.DataFrame:
    charges = charges or [0.0, 0.01, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20]
    keys = ["seed", "episode", "eval_seed"]
    cols = keys + ["assembly_cost_index", "op10_multiplier_step_mean", "op12_multiplier_step_mean"]
    pairs = learned_rows[cols].rename(
        columns={
            "assembly_cost_index": "ppo_assembly_cost_index",
            "op10_multiplier_step_mean": "ppo_op10",
            "op12_multiplier_step_mean": "ppo_op12",
        }
    ).merge(
        static_rows[cols].rename(
            columns={
                "assembly_cost_index": "static_assembly_cost_index",
                "op10_multiplier_step_mean": "static_op10",
                "op12_multiplier_step_mean": "static_op12",
            }
        ),
        on=keys,
        how="inner",
    )
    rows = []
    for charge in charges:
        ppo_cost = (
            pairs["ppo_assembly_cost_index"]
            + charge * ((pairs["ppo_op10"] - 1.0).clip(lower=0.0) + (pairs["ppo_op12"] - 1.0).clip(lower=0.0))
        )
        static_cost = (
            pairs["static_assembly_cost_index"]
            + charge * ((pairs["static_op10"] - 1.0).clip(lower=0.0) + (pairs["static_op12"] - 1.0).clip(lower=0.0))
        )
        delta = ppo_cost.to_numpy(dtype=float) - static_cost.to_numpy(dtype=float)
        lo, hi = bootstrap_ci(delta, seed=20260801 + int(charge * 1000))
        rows.append(
            {
                "dispatch_charge_per_multiplier_step": float(charge),
                "ppo_dispatch_inclusive_cost_mean": float(ppo_cost.mean()),
                "static_dispatch_inclusive_cost_mean": float(static_cost.mean()),
                "ppo_minus_static_delta": float(delta.mean()),
                "delta_ci95_low": lo,
                "delta_ci95_high": hi,
                "ppo_cheaper": bool(delta.mean() < 0.0),
            }
        )
    return pd.DataFrame(rows)


def collect_pareto_points(
    learned_summary: pd.DataFrame,
    dense_summary: pd.DataFrame,
    learned_policy: str,
) -> pd.DataFrame:
    statics = dense_summary[dense_summary["policy"].astype(str).str.startswith(STATIC_PREFIXES)].copy()
    learned = learned_summary[learned_summary["policy"].astype(str) == learned_policy].copy()
    if learned.empty:
        raise ValueError(f"No learned policy {learned_policy!r} in canonical summary")
    rows: list[dict[str, Any]] = []
    for _, row in statics.iterrows():
        rows.append(
            {
                "policy": str(row["policy"]),
                "kind": "dense_static",
                "order_ret_excel": float(row["order_ret_excel_mean"]),
                "order_level_ret_mean": float(row["order_level_ret_mean_mean"]),
                "flow_fill_rate": float(row["flow_fill_rate_mean"]),
                "terminal_rolling_fill_rate_4w": float(row["terminal_rolling_fill_rate_4w_mean"]),
                "ret_garrido2024_sigmoid_mean": float(row["ret_garrido2024_sigmoid_mean_mean"]),
                "assembly_cost_index": float(row["assembly_cost_index_mean"]),
                "order_service_loss_auc_per_order": float(row["order_service_loss_auc_per_order_mean"]),
                "order_ctj_p99": float(row["order_ctj_p99_mean"]),
                "order_rpj_p99": float(row["order_rpj_p99_mean"]),
                "order_dpj_p99": float(row["order_dpj_p99_mean"]),
            }
        )
    ppo = learned.iloc[0]
    rows.append(
        {
            "policy": learned_policy,
            "kind": "ppo",
            "order_ret_excel": float(ppo["order_ret_excel_mean"]),
            "order_level_ret_mean": float(ppo["order_level_ret_mean_mean"]),
            "flow_fill_rate": float(ppo["flow_fill_rate_mean"]),
            "terminal_rolling_fill_rate_4w": float(ppo["terminal_rolling_fill_rate_4w_mean"]),
            "ret_garrido2024_sigmoid_mean": float(ppo["ret_garrido2024_sigmoid_mean_mean"]),
            "assembly_cost_index": float(ppo["assembly_cost_index_mean"]),
            "order_service_loss_auc_per_order": float(ppo["order_service_loss_auc_per_order_mean"]),
            "order_ctj_p99": float(ppo["order_ctj_p99_mean"]),
            "order_rpj_p99": float(ppo["order_rpj_p99_mean"]),
            "order_dpj_p99": float(ppo["order_dpj_p99_mean"]),
        }
    )
    return pd.DataFrame(rows)


def dominates(a: pd.Series, b: pd.Series, specs: list[tuple[str, str]]) -> bool:
    better_or_equal = True
    strictly_better = False
    for metric, direction in specs:
        av = float(a[metric])
        bv = float(b[metric])
        if direction == "higher":
            if av < bv:
                better_or_equal = False
            if av > bv:
                strictly_better = True
        else:
            if av > bv:
                better_or_equal = False
            if av < bv:
                strictly_better = True
    return better_or_equal and strictly_better


def add_pareto_flags(points: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("order_ret_excel", "higher"),
        ("assembly_cost_index", "lower"),
        ("order_ctj_p99", "lower"),
        ("flow_fill_rate", "higher"),
    ]
    flags = []
    for i, row in points.iterrows():
        dominated = False
        for j, other in points.iterrows():
            if i == j:
                continue
            if dominates(other, row, specs):
                dominated = True
                break
        flags.append(not dominated)
    out = points.copy()
    out["pareto_nondominated_ret_cost_tail_flow"] = flags
    return out


def plot_scatter(
    points: pd.DataFrame,
    *,
    x: str,
    y: str,
    xlabel: str,
    ylabel: str,
    title: str,
    path: Path,
    lower_x_better: bool,
) -> None:
    statics = points[points["kind"] == "dense_static"]
    ppo = points[points["kind"] == "ppo"]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    ax.scatter(statics[x], statics[y], s=22, alpha=0.55, label="Dense static frontier", color="#64748b")
    ax.scatter(ppo[x], ppo[y], s=160, marker="*", label="PPO Track B", color="#dc2626", edgecolor="black", linewidth=0.8)
    for _, row in ppo.iterrows():
        ax.annotate("PPO", (row[x], row[y]), xytext=(8, 8), textcoords="offset points", fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.margins(x=0.05, y=0.12)
    if lower_x_better:
        ax.annotate("better", xy=(0.02, 0.02), xycoords="axes fraction", fontsize=8, color="#475569")
    else:
        ax.annotate("better", xy=(0.84, 0.02), xycoords="axes fraction", fontsize=8, color="#475569")
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_markdown(
    path: Path,
    *,
    best_static: str,
    effects: pd.DataFrame,
    points: pd.DataFrame,
    seed_summary: dict[str, Any],
    cvar05: dict[str, Any],
) -> None:
    primary = effects.loc[effects["metric"] == PRIMARY_METRIC].iloc[0]
    ppo = points[points["kind"] == "ppo"].iloc[0]
    dominated_by_static = any(
        dominates(row, ppo, [
            ("order_ret_excel", "higher"),
            ("assembly_cost_index", "lower"),
            ("order_ctj_p99", "lower"),
            ("flow_fill_rate", "higher"),
        ])
        for _, row in points[points["kind"] == "dense_static"].iterrows()
    )
    table_cols = [
        "metric",
        "ppo_mean",
        "static_mean",
        "raw_delta_ppo_minus_static",
        "directional_gain",
        "directional_gain_ci95_low",
        "directional_gain_ci95_high",
        "oriented_paired_cohens_d",
        "ci95_directional_win",
    ]
    table_lines = [
        "| " + " | ".join(table_cols) + " |",
        "| " + " | ".join("---" for _ in table_cols) + " |",
    ]
    for _, row in effects[table_cols].iterrows():
        values = []
        for col in table_cols:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.6g}")
            else:
                values.append(str(value))
        table_lines.append("| " + " | ".join(values) + " |")

    lines = [
        "# Track B Q1 Stats Audit",
        "",
        f"Primary dense static comparator: `{best_static}` selected by `{PRIMARY_METRIC}`.",
        "",
        "## Primary Result",
        "",
        "| Metric | PPO | Best dense static | Delta | CI95 | Cohen d paired |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| Excel ReT | {primary['ppo_mean']:.6f} | {primary['static_mean']:.6f} | "
            f"{primary['raw_delta_ppo_minus_static']:+.6f} | "
            f"[{primary['raw_delta_ci95_low']:+.6f}, {primary['raw_delta_ci95_high']:+.6f}] | "
            f"{primary['paired_cohens_d']:.2f} |"
        ),
        "",
        "## Pareto Verdict",
        "",
        f"- PPO non-dominated vs dense static frontier on ReT/cost/tail/flow: `{not dominated_by_static}`.",
        "- Figures use `order_ret_excel` as the y-axis resilience metric.",
        "- PPO is not claimed as resource-efficient on assembly cost alone; dispatch-inclusive cost is reported as sensitivity.",
        "",
        "## Seed-Level Primary Inference",
        "",
        f"- Seed count: `{seed_summary.get('seed_count', 0)}`.",
        f"- All seed mean deltas positive: `{seed_summary.get('all_seed_deltas_positive', False)}`.",
        (
            f"- Seed-clustered bootstrap CI95 for mean delta: "
            f"[{seed_summary.get('seed_cluster_bootstrap_ci95_low', float('nan')):+.6f}, "
            f"{seed_summary.get('seed_cluster_bootstrap_ci95_high', float('nan')):+.6f}]."
        ),
        "",
        "## Unified Tail Metric",
        "",
        (
            f"- CVaR05 definition: {cvar05['definition']}; PPO `{cvar05['ppo_mean']:.6f}` "
            f"vs static `{cvar05['static_mean']:.6f}`, delta "
            f"`{cvar05['raw_delta_ppo_minus_static']:+.6f}`."
        ),
        "",
        "## Comparator Scope",
        "",
        "- The dense static frontier varies `shift x op10 x op12`; it is not an 8D static frontier.",
        "- Inventory-dimension constants require a separate bounded grid before any best-8D-static wording.",
        "",
        "## Effect Size Table",
        "",
        "\n".join(table_lines),
        "",
        "## Artifacts",
        "",
        "- `effect_sizes.csv`",
        "- `pareto_points.csv`",
        "- `pareto_ret_cost.png`",
        "- `pareto_ret_tail_ctj.png`",
        "- `pareto_ret_flow.png`",
        "- `pareto_summary.json`",
        "- `seed_level_inference.csv`",
        "- `top12_static_robustness.csv`",
        "- `cvar05_effect.csv`",
        "- `dispatch_cost_sensitivity.csv`",
        "- `comparator_scope.json`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    learned_episode = read_csv(args.run_dir / "episode_metrics.csv")
    learned_summary = read_csv(args.run_dir / "policy_summary.csv")
    dense_episode = read_csv(args.dense_dir / "episode_metrics.csv")
    dense_summary = read_csv(args.dense_dir / "policy_summary.csv")

    best_static = best_static_policy(dense_summary, args.primary_metric, "higher")
    learned_rows = learned_episode[learned_episode["policy"].astype(str) == args.learned_policy].copy()
    static_rows = dense_episode[dense_episode["policy"].astype(str) == best_static].copy()
    if learned_rows.empty:
        raise ValueError(f"No rows for learned policy {args.learned_policy!r}")
    if static_rows.empty:
        raise ValueError(f"No rows for best static policy {best_static!r}")

    effects = build_effect_sizes(learned_rows, static_rows, best_static)
    cvar05 = build_cvar05_effect_row(learned_rows, static_rows, best_static)
    effects.to_csv(args.output_dir / "effect_sizes.csv", index=False)
    pd.DataFrame([cvar05]).to_csv(args.output_dir / "cvar05_effect.csv", index=False)

    seed_level = build_seed_level_inference(learned_rows, static_rows)
    seed_summary = dict(seed_level.attrs.get("summary", {}))
    seed_level.to_csv(args.output_dir / "seed_level_inference.csv", index=False)

    top12 = build_top_static_robustness(learned_rows, dense_episode, dense_summary, top_n=12)
    top12.to_csv(args.output_dir / "top12_static_robustness.csv", index=False)

    dispatch_cost = build_dispatch_cost_sensitivity(learned_rows, static_rows)
    dispatch_cost.to_csv(args.output_dir / "dispatch_cost_sensitivity.csv", index=False)

    comparator_scope = {
        "dense_static_frontier_dims": ["assembly_shifts", "op10_multiplier", "op12_multiplier"],
        "not_varied_in_dense_static_frontier": [
            "op3_q",
            "op3_rop",
            "op9_q_min",
            "op9_q_max",
            "op9_rop",
        ],
        "paper_safe_wording": "best static over the shift x op10 x op12 family",
        "unsafe_wording": "best 8D static frontier",
        "inventory_constant_bound_status": "pending separate bounded grid",
    }
    (args.output_dir / "comparator_scope.json").write_text(
        json.dumps(comparator_scope, indent=2, sort_keys=True), encoding="utf-8"
    )

    points = collect_pareto_points(learned_summary, dense_summary, args.learned_policy)
    points = add_pareto_flags(points)
    points.to_csv(args.output_dir / "pareto_points.csv", index=False)

    plot_scatter(
        points,
        x="assembly_cost_index",
        y="order_ret_excel",
        xlabel="Cost index (lower is better)",
        ylabel="Excel ReT (higher is better)",
        title="Track B Pareto: Excel ReT vs Cost",
        path=args.output_dir / "pareto_ret_cost.png",
        lower_x_better=True,
    )
    plot_scatter(
        points,
        x="order_ctj_p99",
        y="order_ret_excel",
        xlabel="CTj p99 hours (lower is better)",
        ylabel="Excel ReT (higher is better)",
        title="Track B Pareto: Excel ReT vs Tail Cycle Time",
        path=args.output_dir / "pareto_ret_tail_ctj.png",
        lower_x_better=True,
    )
    plot_scatter(
        points,
        x="flow_fill_rate",
        y="order_ret_excel",
        xlabel="Flow fill rate (higher is better)",
        ylabel="Excel ReT (higher is better)",
        title="Track B Pareto: Excel ReT vs Flow Fill",
        path=args.output_dir / "pareto_ret_flow.png",
        lower_x_better=False,
    )

    ppo = points[points["kind"] == "ppo"].iloc[0]
    dominated_by_static = [
        str(row["policy"])
        for _, row in points[points["kind"] == "dense_static"].iterrows()
        if dominates(row, ppo, [
            ("order_ret_excel", "higher"),
            ("assembly_cost_index", "lower"),
            ("order_ctj_p99", "lower"),
            ("flow_fill_rate", "higher"),
        ])
    ]
    summary = {
        "run_dir": str(args.run_dir),
        "dense_dir": str(args.dense_dir),
        "learned_policy": args.learned_policy,
        "primary_metric": args.primary_metric,
        "best_dense_static_by_primary": best_static,
        "ppo_pareto_nondominated_ret_cost_tail_flow": len(dominated_by_static) == 0,
        "statics_dominating_ppo_ret_cost_tail_flow": dominated_by_static,
        "primary_effect": effects[effects["metric"] == args.primary_metric].to_dict(orient="records"),
        "cvar05_effect": cvar05,
        "seed_level_inference_summary": seed_summary,
        "top12_static_robustness": {
            "all_top12_ci95_positive": bool((top12["delta_ci95_low"] > 0.0).all()) if not top12.empty else False,
            "min_delta_ci95_low": float(top12["delta_ci95_low"].min()) if not top12.empty else float("nan"),
        },
        "comparator_scope": comparator_scope,
        "outputs": {
            "effect_sizes": str(args.output_dir / "effect_sizes.csv"),
            "cvar05_effect": str(args.output_dir / "cvar05_effect.csv"),
            "seed_level_inference": str(args.output_dir / "seed_level_inference.csv"),
            "top12_static_robustness": str(args.output_dir / "top12_static_robustness.csv"),
            "dispatch_cost_sensitivity": str(args.output_dir / "dispatch_cost_sensitivity.csv"),
            "comparator_scope": str(args.output_dir / "comparator_scope.json"),
            "pareto_points": str(args.output_dir / "pareto_points.csv"),
            "pareto_ret_cost": str(args.output_dir / "pareto_ret_cost.png"),
            "pareto_ret_tail_ctj": str(args.output_dir / "pareto_ret_tail_ctj.png"),
            "pareto_ret_flow": str(args.output_dir / "pareto_ret_flow.png"),
        },
    }
    (args.output_dir / "pareto_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_markdown(
        args.output_dir / "README.md",
        best_static=best_static,
        effects=effects,
        points=points,
        seed_summary=seed_summary,
        cvar05=cvar05,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
