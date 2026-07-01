#!/usr/bin/env python3
"""Mechanism audit for a completed Track B run.

The paper claim should be "adaptive recovery / backlog control" only when the
metrics support it. This script summarizes PPO vs best static across Excel ReT,
tail ReT, queue/recovery times, service loss, cost, and downstream dispatch.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any
import math

import pandas as pd


LOWER_IS_BETTER = {
    "order_CTj_p99",
    "order_RPj_p99",
    "order_DPj_p99",
    "order_service_loss_auc_per_order",
    "order_backorder_qty_final",
    "order_lost_rate",
    "assembly_cost_index",
}


MECHANISM_METRICS = (
    "order_ret_excel",
    "order_ret_excel_cvar05",
    "order_ret_excel_p05",
    "order_ret_excel_p50",
    "order_level_ret_mean",
    "order_CTj_p99",
    "order_RPj_p99",
    "order_DPj_p99",
    "order_service_loss_auc_per_order",
    "order_backorder_qty_final",
    "order_lost_rate",
    "flow_fill_rate",
    "terminal_rolling_fill_rate_4w",
    "ret_garrido2024_sigmoid_mean",
    "assembly_cost_index",
    "op10_multiplier_step_mean",
    "op12_multiplier_step_mean",
    "op10_multiplier_step_p95",
    "op12_multiplier_step_p95",
    "pct_steps_S1",
    "pct_steps_S2",
    "pct_steps_S3",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--learned-policy", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _mean_metric(row: pd.Series, metric: str) -> float:
    for key in (f"{metric}_mean", metric):
        if key in row.index:
            return float(row[key])
    return float("nan")


def _better_delta(metric: str, dynamic: float, static: float) -> float:
    delta = dynamic - static
    return -delta if metric in LOWER_IS_BETTER else delta


def _is_nan(value: float) -> bool:
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return True


def _choose_best_static(policy_summary: pd.DataFrame) -> str:
    statics = policy_summary[policy_summary["policy"].astype(str).str.startswith("s")]
    if statics.empty:
        return ""
    key = (
        "order_level_ret_mean"
        if "order_level_ret_mean" in statics.columns
        else "order_level_ret_mean_mean"
    )
    idx = statics[key].astype(float).idxmax()
    return str(statics.loc[idx, "policy"])


def _safe_quantile(series: pd.Series, q: float) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return float("nan")
    return float(values.quantile(q))


def build_mechanism_rows(
    policy_summary: pd.DataFrame,
    learned_policy: str,
    best_static: str,
) -> list[dict[str, Any]]:
    if policy_summary.empty:
        return []
    rows_by_policy = {
        str(row["policy"]): row for _, row in policy_summary.iterrows()
    }
    dyn = rows_by_policy.get(learned_policy)
    sta = rows_by_policy.get(best_static)
    if dyn is None or sta is None:
        return []
    out: list[dict[str, Any]] = []
    for metric in MECHANISM_METRICS:
        dynamic_value = _mean_metric(dyn, metric)
        static_value = _mean_metric(sta, metric)
        delta = dynamic_value - static_value
        out.append(
            {
                "metric": metric,
                "dynamic": dynamic_value,
                "best_static": static_value,
                "delta_dynamic_minus_static": delta,
                "directional_gain": _better_delta(metric, dynamic_value, static_value),
                "win": bool(_better_delta(metric, dynamic_value, static_value) > 0.0),
                "lower_is_better": metric in LOWER_IS_BETTER,
            }
        )
    return out


def build_ledger_tail_rows(
    ledger: pd.DataFrame,
    learned_policy: str,
    best_static: str,
) -> list[dict[str, Any]]:
    if ledger.empty or "policy" not in ledger.columns:
        return []
    metric_map = {
        "ReTj": "ReTj",
        "APj": "APj",
        "RPj": "RPj",
        "DPj": "DPj",
        "CTj": "CTj",
    }
    out: list[dict[str, Any]] = []
    for metric, column in metric_map.items():
        if column not in ledger.columns:
            continue
        for policy in (learned_policy, best_static):
            bucket = ledger[ledger["policy"].astype(str) == policy]
            out.append(
                {
                    "policy": policy,
                    "metric": metric,
                    "mean": float(pd.to_numeric(bucket[column], errors="coerce").mean()),
                    "p05": _safe_quantile(bucket[column], 0.05),
                    "p50": _safe_quantile(bucket[column], 0.50),
                    "p95": _safe_quantile(bucket[column], 0.95),
                    "p99": _safe_quantile(bucket[column], 0.99),
                    "rows": int(len(bucket)),
                }
            )
    return out


def augment_mechanism_rows_from_ledger(
    mechanism_rows: list[dict[str, Any]],
    ledger_rows: list[dict[str, Any]],
    learned_policy: str,
    best_static: str,
) -> list[dict[str, Any]]:
    """Fill order-level tail metrics in the main panel from the ledger.

    Older policy summaries do not always carry ReTj/CTj/RPj/DPj percentiles,
    while the Garrido-style order ledger does. Keep policy-summary values when
    present, but use ledger-derived values for the mechanism table otherwise.
    """
    by_metric_policy = {
        (str(row["metric"]), str(row["policy"])): row for row in ledger_rows
    }
    replacements = {
        "order_ret_excel_cvar05": ("ReTj", "p05"),
        "order_ret_excel_p05": ("ReTj", "p05"),
        "order_ret_excel_p50": ("ReTj", "p50"),
        "order_CTj_p99": ("CTj", "p99"),
        "order_RPj_p99": ("RPj", "p99"),
        "order_DPj_p99": ("DPj", "p99"),
    }
    for row in mechanism_rows:
        metric = str(row["metric"])
        if metric not in replacements or not _is_nan(row.get("dynamic")):
            continue
        ledger_metric, stat = replacements[metric]
        dyn = by_metric_policy.get((ledger_metric, learned_policy))
        sta = by_metric_policy.get((ledger_metric, best_static))
        if not dyn or not sta:
            continue
        dynamic_value = float(dyn[stat])
        static_value = float(sta[stat])
        delta = dynamic_value - static_value
        row.update(
            {
                "dynamic": dynamic_value,
                "best_static": static_value,
                "delta_dynamic_minus_static": delta,
                "directional_gain": _better_delta(metric, dynamic_value, static_value),
                "win": bool(_better_delta(metric, dynamic_value, static_value) > 0.0),
                "lower_is_better": metric in LOWER_IS_BETTER,
            }
        )
    return mechanism_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir
    output_dir = args.output_dir or (run_dir / "mechanism_audit")
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = run_dir / "summary.json"
    summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.exists()
        else {}
    )
    policy_summary = _read_csv(run_dir / "policy_summary.csv")
    ledger = _read_csv(run_dir / "order_ledger.csv")
    learned_policy = (
        args.learned_policy
        or str(summary.get("decision", {}).get("learned_policy", "ppo"))
    )
    best_static = str(
        summary.get("decision", {}).get("best_static_policy")
        or _choose_best_static(policy_summary)
    )

    ledger_rows = build_ledger_tail_rows(ledger, learned_policy, best_static)
    mechanism_rows = augment_mechanism_rows_from_ledger(
        build_mechanism_rows(policy_summary, learned_policy, best_static),
        ledger_rows,
        learned_policy,
        best_static,
    )
    ledger_by_metric_policy = {
        (str(row["metric"]), str(row["policy"])): row for row in ledger_rows
    }
    ledger_tail_win = False
    for metric in ("CTj", "RPj", "DPj"):
        dyn = ledger_by_metric_policy.get((metric, learned_policy))
        sta = ledger_by_metric_policy.get((metric, best_static))
        if dyn and sta and float(dyn["p99"]) < float(sta["p99"]):
            ledger_tail_win = True
            break
    panel_tail_win = any(
        row["metric"] in {"order_CTj_p99", "order_RPj_p99", "order_DPj_p99"}
        and row["win"]
        for row in mechanism_rows
    )
    payload = {
        "run_dir": str(run_dir.resolve()),
        "learned_policy": learned_policy,
        "best_static_policy": best_static,
        "mechanism_rows": mechanism_rows,
        "ledger_tail_rows": ledger_rows,
        "interpretation": {
            "primary_mechanism": (
                "adaptive recovery / backlog control"
                if panel_tail_win or ledger_tail_win
                else "not established"
            ),
            "anticipation_claim_allowed": False,
            "note": (
                "Use action-trace lead/lag correlations before upgrading this to "
                "anticipation. This audit only supports recovery/backlog mechanisms."
            ),
        },
    }
    write_csv(output_dir / "mechanism_metric_panel.csv", mechanism_rows)
    write_csv(output_dir / "ledger_tail_panel.csv", ledger_rows)
    (output_dir / "mechanism_audit.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(f"WROTE {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
