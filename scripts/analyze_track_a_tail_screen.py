#!/usr/bin/env python3
"""Apply the Track A stop/promote rule to a sweep_summary.csv.

This script is deliberately separate from training.  It judges trained policies
only on external resilience metrics against the best static policy under the
same scenario.  Reward totals are ignored.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RowDecision:
    label: str
    algo: str
    risk_level: str
    status: str
    promote: bool
    primary_ret_win: bool
    primary_flow_win: bool
    tail_win: bool
    delta_ret_all_orders: float
    delta_flow_fill: float
    delta_ret_p10_all: float
    improvement_stockout_week_pct: float
    delta_fill: float
    best_static_policy_by_ret_all_orders: str
    best_static_policy_by_flow_fill: str
    best_static_policy_by_ret_p10_all: str
    reason: str


def as_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if value in (None, ""):
        return default
    try:
        return float(value)
    except ValueError:
        return default


def choose_delta(row: dict[str, str], metric_specific: str, legacy: str) -> float:
    if row.get(metric_specific) not in (None, ""):
        return as_float(row, metric_specific)
    return as_float(row, legacy)


def decide_row(
    row: dict[str, str],
    *,
    primary_threshold: float,
    p10_threshold: float,
    max_fill_loss: float,
    min_stockout_improvement: float,
) -> RowDecision:
    status = str(row.get("status", ""))
    delta_ret = choose_delta(
        row,
        "delta_ret_all_orders_vs_best_metric",
        "delta_ret_all_orders",
    )
    delta_flow = choose_delta(row, "delta_flow_fill_vs_best_metric", "delta_flow_fill")
    delta_p10 = choose_delta(row, "delta_ret_p10_all_vs_best_metric", "delta_ret_p10_all")
    stockout_improvement = as_float(
        row,
        "improvement_stockout_week_pct_vs_best_metric",
        default=-as_float(row, "delta_stockout_week_pct"),
    )
    delta_fill = as_float(row, "delta_fill")

    primary_ret_win = delta_ret >= primary_threshold
    primary_flow_win = delta_flow >= primary_threshold
    tail_win = (
        delta_p10 >= p10_threshold
        and stockout_improvement > min_stockout_improvement
        and delta_fill >= -max_fill_loss
    )
    promote = status == "complete" and (primary_ret_win or primary_flow_win or tail_win)
    if status != "complete":
        reason = f"not complete: {status}"
    elif primary_ret_win:
        reason = f"all-order ReT improved by {delta_ret:+.4f}"
    elif primary_flow_win:
        reason = f"flow fill improved by {delta_flow:+.4f}"
    elif tail_win:
        reason = (
            f"tail p10 improved by {delta_p10:+.4f} with stockout improvement "
            f"{stockout_improvement:+.4f} and fill delta {delta_fill:+.4f}"
        )
    else:
        reason = "no stop-rule improvement"

    return RowDecision(
        label=str(row.get("label", "")),
        algo=str(row.get("algo", "")),
        risk_level=str(row.get("risk_level", "")),
        status=status,
        promote=promote,
        primary_ret_win=primary_ret_win,
        primary_flow_win=primary_flow_win,
        tail_win=tail_win,
        delta_ret_all_orders=delta_ret,
        delta_flow_fill=delta_flow,
        delta_ret_p10_all=delta_p10,
        improvement_stockout_week_pct=stockout_improvement,
        delta_fill=delta_fill,
        best_static_policy_by_ret_all_orders=str(
            row.get("best_static_policy_by_ret_all_orders", row.get("best_static_policy", ""))
        ),
        best_static_policy_by_flow_fill=str(
            row.get("best_static_policy_by_flow_fill", row.get("best_static_policy", ""))
        ),
        best_static_policy_by_ret_p10_all=str(
            row.get("best_static_policy_by_ret_p10_all", row.get("best_static_policy", ""))
        ),
        reason=reason,
    )


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def rank_promotions(decisions: list[RowDecision]) -> list[RowDecision]:
    return sorted(
        [item for item in decisions if item.promote],
        key=lambda item: (
            item.delta_ret_all_orders,
            item.delta_flow_fill,
            item.delta_ret_p10_all,
            item.improvement_stockout_week_pct,
        ),
        reverse=True,
    )


def summarize(decisions: list[RowDecision]) -> dict[str, Any]:
    complete = [item for item in decisions if item.status == "complete"]
    promotions = rank_promotions(decisions)
    return {
        "n_rows": len(decisions),
        "n_complete": len(complete),
        "n_promoted": len(promotions),
        "decision": (
            "PROMOTE_TOP_CONFIGS"
            if promotions
            else "STOP_TRACK_A_OR_RUN_EXTENSION"
            if len(complete) == len(decisions)
            else "INCONCLUSIVE_INCOMPLETE_RUNS"
        ),
        "top_configs": [asdict(item) for item in promotions[:2]],
        "rows": [asdict(item) for item in decisions],
    }


def write_markdown(path: Path, summary: dict[str, Any], *, source: Path) -> None:
    lines = [
        "# Track A Tail Screen Decision",
        "",
        f"- source: `{source}`",
        f"- decision: `{summary['decision']}`",
        f"- complete rows: {summary['n_complete']} / {summary['n_rows']}",
        f"- promoted rows: {summary['n_promoted']}",
        "",
        "## Stop Rule",
        "",
        "- Promote if all-order ReT or flow-fill improves by at least +0.02.",
        "- Also promote if p10 ReT improves with lower stockout and no material fill loss.",
        "- Ignore reward total as a victory criterion.",
        "",
        "## Rows",
        "",
        "| label | status | d_ReT_all | d_flow | d_p10 | stockout_impr | d_fill | promote | reason |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary["rows"]:
        lines.append(
            "| `{label}` | `{status}` | {delta_ret_all_orders:+.4f} | "
            "{delta_flow_fill:+.4f} | {delta_ret_p10_all:+.4f} | "
            "{improvement_stockout_week_pct:+.4f} | {delta_fill:+.4f} | "
            "`{promote}` | {reason} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep_summary", type=Path)
    parser.add_argument("--primary-threshold", type=float, default=0.02)
    parser.add_argument("--p10-threshold", type=float, default=0.01)
    parser.add_argument("--max-fill-loss", type=float, default=0.01)
    parser.add_argument("--min-stockout-improvement", type=float, default=0.0)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows = read_rows(args.sweep_summary)
    decisions = [
        decide_row(
            row,
            primary_threshold=args.primary_threshold,
            p10_threshold=args.p10_threshold,
            max_fill_loss=args.max_fill_loss,
            min_stockout_improvement=args.min_stockout_improvement,
        )
        for row in rows
    ]
    summary = summarize(decisions)
    output_json = args.output_json or args.sweep_summary.with_name(
        "track_a_tail_decision.json"
    )
    output_md = args.output_md or args.sweep_summary.with_name(
        "TRACK_A_TAIL_DECISION.md"
    )
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(output_md, summary, source=args.sweep_summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Saved {output_json}")
    print(f"Saved {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
