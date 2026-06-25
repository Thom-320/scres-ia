#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_ROOT = Path("outputs/benchmarks/downstream_q_sensitivity")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_mode_rows(run_dir: Path) -> dict[str, dict[str, Any]]:
    rows = read_csv_rows(run_dir / "reward_mode_summary.csv")
    sorted_rows = sorted(
        rows,
        key=lambda row: float(row["diagnostic_score"]),
        reverse=True,
    )
    out: dict[str, dict[str, Any]] = {}
    for rank, row in enumerate(sorted_rows, start=1):
        typed: dict[str, Any] = dict(row)
        typed["rank"] = rank
        for field in (
            "diagnostic_score",
            "reward_spread_ratio",
            "spearman_reward_vs_order_level_ret",
            "spearman_reward_vs_fill",
            "spearman_reward_vs_negative_service_loss_area",
            "spearman_reward_vs_negative_pending_backlog",
            "spearman_reward_vs_negative_step_cost",
            "best_policy_order_level_ret",
            "best_policy_fill_rate",
            "best_policy_service_loss_area",
            "best_policy_step_cost",
        ):
            typed[field] = float(typed[field])
        typed["best_policy_shifts"] = int(float(typed["best_policy_shifts"]))
        out[str(row["reward_mode"])] = typed
    return out


def load_summary(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return {}
    return json.loads(summary_path.read_text(encoding="utf-8"))


def compare_runs(
    *,
    figure_dir: Path,
    table_dir: Path,
    rank_tolerance: int,
) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    figure_rows = load_mode_rows(figure_dir)
    table_rows = load_mode_rows(table_dir)
    reward_modes = sorted(set(figure_rows) | set(table_rows))
    rows: list[dict[str, Any]] = []
    shortlist: list[str] = []
    for reward_mode in reward_modes:
        figure = figure_rows.get(reward_mode)
        table = table_rows.get(reward_mode)
        if figure is None or table is None:
            rows.append(
                {
                    "reward_mode": reward_mode,
                    "present_in_both": False,
                    "stable_for_smoke": False,
                    "reason": "missing_from_one_run",
                }
            )
            continue

        gate_match = figure["selection_gate"] == table["selection_gate"]
        best_policy_match = figure["best_policy_by_reward"] == table[
            "best_policy_by_reward"
        ]
        best_shift_match = figure["best_policy_shifts"] == table["best_policy_shifts"]
        rank_delta = int(table["rank"]) - int(figure["rank"])
        rank_stable = abs(rank_delta) <= rank_tolerance
        both_shortlist = (
            figure["selection_gate"] == "shortlist"
            and table["selection_gate"] == "shortlist"
        )
        negative_control = (
            figure["selection_gate"] == "negative_control"
            or table["selection_gate"] == "negative_control"
        )
        stable_for_smoke = bool(
            both_shortlist
            and gate_match
            and best_policy_match
            and rank_stable
            and not negative_control
        )
        if stable_for_smoke:
            shortlist.append(reward_mode)

        if not both_shortlist:
            reason = "not_shortlisted_in_both"
        elif not gate_match:
            reason = "gate_changed"
        elif not best_policy_match:
            reason = "best_policy_changed"
        elif not rank_stable:
            reason = "rank_unstable"
        elif negative_control:
            reason = "negative_control"
        else:
            reason = "stable"

        rows.append(
            {
                "reward_mode": reward_mode,
                "present_in_both": True,
                "figure_rank": figure["rank"],
                "table_rank": table["rank"],
                "rank_delta_table_minus_figure": rank_delta,
                "rank_stable": rank_stable,
                "figure_gate": figure["selection_gate"],
                "table_gate": table["selection_gate"],
                "gate_match": gate_match,
                "figure_best_policy": figure["best_policy_by_reward"],
                "table_best_policy": table["best_policy_by_reward"],
                "best_policy_match": best_policy_match,
                "figure_best_shift": figure["best_policy_shifts"],
                "table_best_shift": table["best_policy_shifts"],
                "best_shift_match": best_shift_match,
                "figure_diagnostic_score": figure["diagnostic_score"],
                "table_diagnostic_score": table["diagnostic_score"],
                "figure_rho_ret": figure["spearman_reward_vs_order_level_ret"],
                "table_rho_ret": table["spearman_reward_vs_order_level_ret"],
                "figure_rho_fill": figure["spearman_reward_vs_fill"],
                "table_rho_fill": table["spearman_reward_vs_fill"],
                "figure_rho_negative_loss": figure[
                    "spearman_reward_vs_negative_service_loss_area"
                ],
                "table_rho_negative_loss": table[
                    "spearman_reward_vs_negative_service_loss_area"
                ],
                "figure_best_policy_ret": figure["best_policy_order_level_ret"],
                "table_best_policy_ret": table["best_policy_order_level_ret"],
                "figure_best_policy_fill": figure["best_policy_fill_rate"],
                "table_best_policy_fill": table["best_policy_fill_rate"],
                "figure_best_policy_loss": figure["best_policy_service_loss_area"],
                "table_best_policy_loss": table["best_policy_service_loss_area"],
                "figure_best_policy_cost": figure["best_policy_step_cost"],
                "table_best_policy_cost": table["best_policy_step_cost"],
                "stable_for_smoke": stable_for_smoke,
                "reason": reason,
            }
        )

    rows.sort(
        key=lambda row: (
            not bool(row.get("stable_for_smoke", False)),
            int(row.get("figure_rank", 10_000)),
            str(row["reward_mode"]),
        )
    )
    metadata = {
        "figure_dir": str(figure_dir),
        "table_dir": str(table_dir),
        "figure_summary": load_summary(figure_dir).get("config", {}),
        "table_summary": load_summary(table_dir).get("config", {}),
        "rank_tolerance": rank_tolerance,
    }
    return rows, shortlist, metadata


def render_report(rows: list[dict[str, Any]], shortlist: list[str], metadata: dict[str, Any]) -> str:
    lines = [
        "# Downstream-Q Reward Surface Sensitivity",
        "",
        "This compares the thesis Figure 6.2 downstream quantity interpretation "
        "against the Table 6.20 interpretation. Rewards are only proposed for "
        "PPO/DQN smoke training when the selection gate, rank, and best static "
        "policy are stable across both runs.",
        "",
        "## Inputs",
        "",
        f"- figure_dir: `{metadata['figure_dir']}`",
        f"- table_dir: `{metadata['table_dir']}`",
        f"- rank_tolerance: `{metadata['rank_tolerance']}`",
        "",
        "## PPO/DQN Smoke Shortlist",
        "",
    ]
    if shortlist:
        for reward_mode in shortlist:
            lines.append(f"- `{reward_mode}`")
    else:
        lines.append("- No reward mode passed the downstream-Q stability gate.")

    lines.extend(
        [
            "",
            "## Comparison",
            "",
            "| reward_mode | stable | reason | figure rank | table rank | figure gate | table gate | figure best | table best |",
            "|---|---:|---|---:|---:|---|---|---|---|",
        ]
    )
    for row in rows:
        lines.append(
            f"| `{row['reward_mode']}` | {str(row.get('stable_for_smoke', False)).lower()} | "
            f"`{row.get('reason', '')}` | {row.get('figure_rank', '')} | "
            f"{row.get('table_rank', '')} | `{row.get('figure_gate', '')}` | "
            f"`{row.get('table_gate', '')}` | `{row.get('figure_best_policy', '')}` | "
            f"`{row.get('table_best_policy', '')}` |"
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure-dir", type=Path, required=True)
    parser.add_argument("--table-dir", type=Path, required=True)
    parser.add_argument("--label", default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--rank-tolerance", type=int, default=2)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_label = args.label or f"downstream_q_sensitivity_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir = args.output_root / run_label
    run_dir.mkdir(parents=True, exist_ok=False)
    rows, shortlist, metadata = compare_runs(
        figure_dir=args.figure_dir,
        table_dir=args.table_dir,
        rank_tolerance=args.rank_tolerance,
    )
    write_csv(run_dir / "downstream_q_reward_comparison.csv", rows)
    write_json(
        run_dir / "summary.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "label": run_label,
            "shortlist": shortlist,
            "metadata": metadata,
            "comparison": rows,
        },
    )
    report = render_report(rows, shortlist, metadata)
    (run_dir / "DOWNSTREAM_Q_SENSITIVITY.md").write_text(
        report + "\n",
        encoding="utf-8",
    )
    print(report, flush=True)
    print(f"Saved to: {run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
