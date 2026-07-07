#!/usr/bin/env python3
"""Event-conditioned resilience purchase audit for Track B.

This audit answers a narrower question than global ReT:

    When a real risk event occurs, does a policy buy local resilience around
    that event, and how much resource posture does it spend to buy it?

The exact Garrido/Excel ReT remains the primary episode-level metric, but the
weekly ledgers do not contain per-order Excel ReT values inside each event
window. Therefore this script reports:

1. Event-window operational resilience proxies:
   - local service continuity = 1 - new_backorder / new_demand in the post window;
   - new backorder avoided;
   - pending backlog AUC above the pre-event baseline;
   - recovery weeks back to the pre-event pending-backlog level.
2. Purchase ratios:
   - improvement in those local resilience proxies per extra pre-event action
     intensity spent relative to a baseline policy.

Use this as a secondary diagnostic. Do not promote it over order_ret_excel_mean.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_OUTPUT_DIR = Path(
    "outputs/experiments/track_b_event_resilience_purchase_2026-07-04"
)
DEFAULT_STEP_LEDGER = Path(
    "outputs/experiments/track_b_belief_dataset_diverse_fixed_rng_2026-07-04/"
    "step_ledger_full.csv"
)
DEFAULT_RISK_LEDGER = Path(
    "outputs/experiments/track_b_belief_dataset_diverse_fixed_rng_2026-07-04/"
    "risk_event_ledger.csv"
)
# R22/R24 are the prevention-relevant downstream risks used in the oracle
# ceiling gate. R11/R13 are very frequent and can be passed explicitly, but
# they make the event-pair audit much heavier and are less diagnostic for
# "pre-positioning buys resilience".
DEFAULT_RISKS = ("R22", "R24")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--step-ledgers", nargs="+", type=Path, default=[DEFAULT_STEP_LEDGER])
    parser.add_argument("--risk-ledgers", nargs="+", type=Path, default=[DEFAULT_RISK_LEDGER])
    parser.add_argument("--baseline-policy", default="ppo_mlp")
    parser.add_argument("--risks", nargs="+", default=list(DEFAULT_RISKS))
    parser.add_argument("--pre-weeks", type=int, default=4)
    parser.add_argument("--post-weeks", type=int, default=8)
    parser.add_argument(
        "--min-pre-steps",
        type=int,
        default=1,
        help="Minimum observed pre-window steps required to score an event.",
    )
    parser.add_argument(
        "--min-post-steps",
        type=int,
        default=1,
        help="Minimum observed post-window steps required to score an event.",
    )
    return parser.parse_args()


def read_concat(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True)


def safe_mean(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return float(values.mean()) if len(values) else float("nan")


def safe_sum(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return float(values.sum()) if len(values) else 0.0


def event_key_columns() -> list[str]:
    return ["seed", "episode", "eval_seed", "risk_id", "event_index"]


def add_event_index(risks: pd.DataFrame) -> pd.DataFrame:
    risks = risks.sort_values(
        ["policy", "seed", "episode", "risk_id", "start_time_hours", "end_time_hours"]
    ).copy()
    risks["event_index"] = (
        risks.groupby(["policy", "seed", "episode", "risk_id"]).cumcount().astype(int)
    )
    return risks


def score_event(
    *,
    ev: pd.Series,
    steps: pd.DataFrame,
    pre_weeks: int,
    post_weeks: int,
    min_pre_steps: int,
    min_post_steps: int,
) -> dict[str, Any] | None:
    anchor = int(ev["start_step"])
    pre = steps[(steps["step"] >= anchor - pre_weeks) & (steps["step"] <= anchor - 1)]
    post = steps[(steps["step"] >= anchor) & (steps["step"] <= anchor + post_weeks)]
    if len(pre) < min_pre_steps or len(post) < min_post_steps:
        return None

    pre_pending_last = float(pd.to_numeric(pre["pending_backorder_qty"]).iloc[-1])
    pre_pending_median = safe_mean(pre["pending_backorder_qty"])
    post_pending = pd.to_numeric(post["pending_backorder_qty"], errors="coerce").fillna(0.0)
    post_new_demand = safe_sum(post["new_demanded"])
    post_new_backorder = safe_sum(post["new_backorder_qty"])
    local_service_continuity = (
        1.0 - (post_new_backorder / post_new_demand)
        if post_new_demand > 0.0
        else float("nan")
    )
    backlog_above_pre = np.maximum(post_pending.to_numpy(dtype=float) - pre_pending_last, 0.0)
    backlog_auc_above_pre = float(np.sum(backlog_above_pre))
    backlog_growth_peak = float(np.max(post_pending.to_numpy(dtype=float)) - pre_pending_last)

    recovered_week = float("nan")
    exceeded = False
    for _, row in post.iterrows():
        pending = float(row["pending_backorder_qty"])
        rel = int(row["step"]) - anchor
        if pending > pre_pending_last:
            exceeded = True
        if exceeded and pending <= pre_pending_last:
            recovered_week = float(rel)
            break
    if not exceeded:
        recovered_week = 0.0
    elif math.isnan(recovered_week):
        recovered_week = float(post_weeks + 1)

    return {
        "policy": ev["policy"],
        "seed": int(ev["seed"]),
        "episode": int(ev["episode"]),
        "eval_seed": int(ev["eval_seed"]),
        "risk_id": ev["risk_id"],
        "category": ev.get("category", ""),
        "event_index": int(ev["event_index"]),
        "start_step": anchor,
        "start_time_hours": float(ev["start_time_hours"]),
        "pre_steps": int(len(pre)),
        "post_steps": int(len(post)),
        "pre_action_intensity_mean": safe_mean(pre["action_intensity"]),
        "pre_shift_mean": safe_mean(pre["shift"]),
        "pre_op10_multiplier_mean": safe_mean(pre["op10_multiplier"]),
        "pre_op12_multiplier_mean": safe_mean(pre["op12_multiplier"]),
        "post_action_intensity_mean": safe_mean(post["action_intensity"]),
        "post_shift_mean": safe_mean(post["shift"]),
        "post_op10_multiplier_mean": safe_mean(post["op10_multiplier"]),
        "post_op12_multiplier_mean": safe_mean(post["op12_multiplier"]),
        "pre_pending_backlog_last": pre_pending_last,
        "pre_pending_backlog_mean": pre_pending_median,
        "post_pending_backlog_mean": safe_mean(post["pending_backorder_qty"]),
        "post_pending_backlog_max": float(np.max(post_pending.to_numpy(dtype=float))),
        "post_new_demanded": post_new_demand,
        "post_new_backorder_qty": post_new_backorder,
        "post_local_service_continuity": local_service_continuity,
        "post_backlog_auc_above_pre": backlog_auc_above_pre,
        "post_backlog_growth_peak": backlog_growth_peak,
        "recovery_weeks_to_pre_backlog": recovered_week,
        "post_rolling_fill_rate_4w_mean": safe_mean(post["rolling_fill_rate_4w"]),
        "post_rolling_fill_rate_4w_min": float(
            pd.to_numeric(post["rolling_fill_rate_4w"], errors="coerce").min()
        ),
    }


def build_event_metrics(
    steps: pd.DataFrame,
    risks: pd.DataFrame,
    *,
    risk_ids: set[str],
    pre_weeks: int,
    post_weeks: int,
    min_pre_steps: int,
    min_post_steps: int,
) -> pd.DataFrame:
    risks = add_event_index(risks[risks["risk_id"].isin(risk_ids)].copy())
    step_groups = {
        key: group.sort_values("step")
        for key, group in steps.groupby(["policy", "seed", "episode"], sort=False)
    }
    rows = []
    for _, ev in risks.iterrows():
        key = (ev["policy"], ev["seed"], ev["episode"])
        group = step_groups.get(key)
        if group is None:
            continue
        scored = score_event(
            ev=ev,
            steps=group,
            pre_weeks=pre_weeks,
            post_weeks=post_weeks,
            min_pre_steps=min_pre_steps,
            min_post_steps=min_post_steps,
        )
        if scored is not None:
            rows.append(scored)
    return pd.DataFrame(rows)


def summarize_event_metrics(events: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "pre_action_intensity_mean",
        "post_action_intensity_mean",
        "post_local_service_continuity",
        "post_new_backorder_qty",
        "post_pending_backlog_mean",
        "post_backlog_auc_above_pre",
        "post_backlog_growth_peak",
        "recovery_weeks_to_pre_backlog",
        "post_rolling_fill_rate_4w_mean",
    ]
    return (
        events.groupby(["policy", "risk_id"], dropna=False)
        .agg(n_events=("event_index", "size"), **{f"{m}_mean": (m, "mean") for m in metrics})
        .reset_index()
    )


def paired_purchase(events: pd.DataFrame, baseline_policy: str) -> pd.DataFrame:
    key_cols = event_key_columns()
    base = events[events["policy"] == baseline_policy].copy()
    if base.empty:
        raise ValueError(f"baseline policy {baseline_policy!r} not found in event metrics")
    base = base.drop(columns=["policy"]).add_prefix("baseline_")
    base = base.rename(columns={f"baseline_{c}": c for c in key_cols})
    others = events[events["policy"] != baseline_policy].copy()
    merged = others.merge(base, on=key_cols, how="inner")

    merged["delta_pre_action_intensity"] = (
        merged["pre_action_intensity_mean"] - merged["baseline_pre_action_intensity_mean"]
    )
    merged["delta_post_action_intensity"] = (
        merged["post_action_intensity_mean"] - merged["baseline_post_action_intensity_mean"]
    )
    merged["delta_local_service_continuity"] = (
        merged["post_local_service_continuity"]
        - merged["baseline_post_local_service_continuity"]
    )
    merged["new_backorder_avoided"] = (
        merged["baseline_post_new_backorder_qty"] - merged["post_new_backorder_qty"]
    )
    merged["pending_backlog_auc_reduced"] = (
        merged["baseline_post_backlog_auc_above_pre"] - merged["post_backlog_auc_above_pre"]
    )
    merged["recovery_weeks_reduced"] = (
        merged["baseline_recovery_weeks_to_pre_backlog"]
        - merged["recovery_weeks_to_pre_backlog"]
    )
    eps = 1e-9
    spend = merged["delta_pre_action_intensity"].where(
        merged["delta_pre_action_intensity"].abs() > eps
    )
    merged["service_continuity_purchase_per_pre_intensity"] = (
        merged["delta_local_service_continuity"] / spend
    )
    merged["backorder_avoided_per_pre_intensity"] = merged["new_backorder_avoided"] / spend
    merged["backlog_auc_reduced_per_pre_intensity"] = (
        merged["pending_backlog_auc_reduced"] / spend
    )
    merged["positive_local_service_purchase"] = (
        (merged["delta_local_service_continuity"] > 0.0)
        & (merged["delta_pre_action_intensity"] > 0.0)
    )
    return merged


def summarize_purchases(pairs: pd.DataFrame, baseline_policy: str) -> pd.DataFrame:
    agg = (
        pairs.groupby(["policy", "risk_id"], dropna=False)
        .agg(
            n_pairs=("event_index", "size"),
            mean_delta_pre_action_intensity=("delta_pre_action_intensity", "mean"),
            mean_delta_post_action_intensity=("delta_post_action_intensity", "mean"),
            mean_delta_local_service_continuity=("delta_local_service_continuity", "mean"),
            mean_new_backorder_avoided=("new_backorder_avoided", "mean"),
            mean_pending_backlog_auc_reduced=("pending_backlog_auc_reduced", "mean"),
            mean_recovery_weeks_reduced=("recovery_weeks_reduced", "mean"),
            purchase_positive_rate=("positive_local_service_purchase", "mean"),
            mean_service_continuity_purchase_per_pre_intensity=(
                "service_continuity_purchase_per_pre_intensity",
                "mean",
            ),
            mean_backorder_avoided_per_pre_intensity=(
                "backorder_avoided_per_pre_intensity",
                "mean",
            ),
            mean_backlog_auc_reduced_per_pre_intensity=(
                "backlog_auc_reduced_per_pre_intensity",
                "mean",
            ),
        )
        .reset_index()
    )
    agg.insert(1, "baseline_policy", baseline_policy)
    return agg


def write_verdict(
    *,
    out: Path,
    args: argparse.Namespace,
    event_summary: pd.DataFrame,
    purchase_summary: pd.DataFrame,
) -> None:
    top = purchase_summary.sort_values(
        ["mean_delta_local_service_continuity", "purchase_positive_rate"],
        ascending=[False, False],
    ).head(12)
    lines = [
        "# Event-conditioned resilience purchase audit — Track B",
        "",
        "## What This Measures",
        "",
        "Primary paper metric remains Garrido/Excel episode ReT (`order_ret_excel_mean`).",
        "This audit is secondary: it asks whether a policy buys local resilience around real",
        "risk events, and how much pre-event resource posture it spends to buy it.",
        "",
        "Because the weekly step ledger does not contain per-order Excel ReT values inside",
        "each event window, local resilience is measured with operational proxies:",
        "post-event service continuity, new backorders avoided, backlog AUC reduced, and",
        "recovery weeks back to the pre-event backlog level.",
        "",
        "## Protocol",
        "",
        f"- Baseline policy: `{args.baseline_policy}`",
        f"- Risks: `{', '.join(args.risks)}`",
        f"- Window: `t=-{args.pre_weeks}..+{args.post_weeks}` weeks around each real event onset.",
        f"- Step ledgers: `{', '.join(str(p) for p in args.step_ledgers)}`",
        f"- Risk ledgers: `{', '.join(str(p) for p in args.risk_ledgers)}`",
        "",
        "## Top Positive Purchase Rows",
        "",
    ]
    if top.empty:
        lines.append("No paired purchase rows were available.")
    else:
        cols = [
            "policy",
            "risk_id",
            "n_pairs",
            "mean_delta_pre_action_intensity",
            "mean_delta_local_service_continuity",
            "mean_new_backorder_avoided",
            "mean_pending_backlog_auc_reduced",
            "mean_recovery_weeks_reduced",
            "purchase_positive_rate",
        ]
        lines.extend(markdown_table(top[cols], floatfmt=".6f"))
    lines.extend(
        [
            "",
            "## Reading Rule",
            "",
            "A policy is buying resilience only if the paired row shows both:",
            "",
            "1. positive local resilience (`mean_delta_local_service_continuity > 0`,",
            "   `mean_new_backorder_avoided > 0`, or `mean_pending_backlog_auc_reduced > 0`), and",
            "2. non-trivial positive-purchase rate, not just one or two lucky events.",
            "",
            "If the policy spends more pre-event action intensity but does not improve these",
            "event-window outcomes, it is buying resource posture, not resilience.",
            "",
            "## Files",
            "",
            "- `event_window_metrics.csv`: one row per policy/event/window.",
            "- `event_window_summary.csv`: policy/risk means.",
            "- `paired_purchase.csv`: event-paired deltas versus baseline.",
            "- `purchase_summary.csv`: policy/risk purchase summary.",
        ]
    )
    (out / "verdict.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def markdown_table(df: pd.DataFrame, *, floatfmt: str = ".6f") -> list[str]:
    """Small dependency-free Markdown table writer.

    pandas.DataFrame.to_markdown requires the optional tabulate package, which is
    not guaranteed in the project venv.
    """
    if df.empty:
        return []
    headers = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in df.iterrows():
        cells: list[str] = []
        for col in df.columns:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                cells.append(format(float(value), floatfmt))
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def main() -> None:
    args = parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    steps = read_concat(args.step_ledgers)
    risks = read_concat(args.risk_ledgers)
    risk_ids = {str(r) for r in args.risks}
    events = build_event_metrics(
        steps,
        risks,
        risk_ids=risk_ids,
        pre_weeks=int(args.pre_weeks),
        post_weeks=int(args.post_weeks),
        min_pre_steps=int(args.min_pre_steps),
        min_post_steps=int(args.min_post_steps),
    )
    event_summary = summarize_event_metrics(events)
    pairs = paired_purchase(events, args.baseline_policy)
    purchase_summary = summarize_purchases(pairs, args.baseline_policy)

    events.to_csv(out / "event_window_metrics.csv", index=False)
    event_summary.to_csv(out / "event_window_summary.csv", index=False)
    pairs.to_csv(out / "paired_purchase.csv", index=False)
    purchase_summary.to_csv(out / "purchase_summary.csv", index=False)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "baseline_policy": args.baseline_policy,
        "risks": args.risks,
        "pre_weeks": args.pre_weeks,
        "post_weeks": args.post_weeks,
        "n_event_windows": int(len(events)),
        "n_paired_windows": int(len(pairs)),
        "step_ledgers": [str(p) for p in args.step_ledgers],
        "risk_ledgers": [str(p) for p in args.risk_ledgers],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_verdict(
        out=out,
        args=args,
        event_summary=event_summary,
        purchase_summary=purchase_summary,
    )
    print(f"Wrote {out / 'summary.json'}")
    print(purchase_summary.to_string(index=False))


if __name__ == "__main__":
    main()
