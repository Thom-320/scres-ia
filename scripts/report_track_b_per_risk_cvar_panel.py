#!/usr/bin/env python3
"""Build a branch-aware ReT/CVaR panel for a Track B per-risk headroom grid."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-dir", type=Path, required=True)
    parser.add_argument("--obs-config", default="v7_no_forecast")
    return parser


def read_map_from_summary(payload: dict[str, Any], key: str) -> str:
    value = payload.get("config", {}).get(key) or payload.get("backbone", {}).get(key) or {}
    if not value:
        return ""
    return ",".join(f"{risk}={float(mult):g}" for risk, mult in sorted(value.items()))


def main() -> None:
    args = build_parser().parse_args()
    grid_dir: Path = args.grid_dir
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(grid_dir.glob(f"*/{args.obs_config}/summary.json")):
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        comparison = payload["comparison_table"][0]
        ppo = next(row for row in payload["policy_summary"] if row["policy"] == "ppo")
        best_static = next(
            row
            for row in payload["policy_summary"]
            if row["policy"] == comparison["best_static_policy"]
        )
        learned_ret = float(ppo["order_ret_excel_mean"])
        static_ret = float(best_static["order_ret_excel_mean"])
        learned_cvar = float(ppo["order_ret_excel_cvar05_mean"])
        static_cvar = float(best_static["order_ret_excel_cvar05_mean"])
        rows.append(
            {
                "cell": summary_path.parents[1].name,
                "risk_frequency_by_id": read_map_from_summary(payload, "risk_frequency_by_id"),
                "risk_impact_by_id": read_map_from_summary(payload, "risk_impact_by_id"),
                "order_ret_excel_mean": learned_ret,
                "order_ret_excel_cvar05_mean": learned_cvar,
                "order_ret_excel_p05_mean": float(ppo["order_ret_excel_p05_mean"]),
                "order_ret_excel_p10_mean": float(ppo["order_ret_excel_p10_mean"]),
                "best_static_policy": comparison["best_static_policy"],
                "best_static_order_ret_excel_mean": static_ret,
                "best_static_order_ret_excel_cvar05_mean": static_cvar,
                "delta_ret_excel_vs_static": learned_ret - static_ret,
                "relative_delta_ret_excel_vs_static": (
                    (learned_ret - static_ret) / abs(static_ret) if static_ret else float("nan")
                ),
                "delta_cvar05_vs_static": learned_cvar - static_cvar,
                "relative_delta_cvar05_vs_static": (
                    (learned_cvar - static_cvar) / abs(static_cvar)
                    if static_cvar else float("nan")
                ),
                "assembly_cost_index_mean": float(ppo["assembly_cost_index_mean"]),
                "order_excel_case_pct_fill_rate_mean": float(
                    ppo["order_excel_case_pct_fill_rate_mean"]
                ),
                "order_excel_case_pct_recovery_mean": float(
                    ppo["order_excel_case_pct_recovery_mean"]
                ),
                "order_excel_case_pct_risk_no_recovery_mean": float(
                    ppo["order_excel_case_pct_risk_no_recovery_mean"]
                ),
                "summary_json": str(summary_path),
            }
        )

    rows.sort(
        key=lambda row: (
            -float(row["relative_delta_ret_excel_vs_static"]),
            -float(row["delta_cvar05_vs_static"]),
        )
    )
    csv_path = grid_dir / "per_risk_cvar_panel.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    lines = [
        "# Track B Per-Risk Headroom CVaR Panel",
        "",
        "CVaR05 is the mean of the worst 5% per-order Garrido Excel ReT values. Higher is better.",
        "",
        "| rank | cell | freq by id | impact by id | ReT Excel | CVaR05 | delta ReT vs static | delta CVaR05 | cost | fill-rate branch % | recovery branch % |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(rows, start=1):
        lines.append(
            "| {rank} | `{cell}` | `{freq}` | `{impact}` | {ret:.9f} | {cvar:.9f} | "
            "{dret:+.9f} | {dcvar:+.9f} | {cost:.3f} | {fill:.2f} | {rec:.2f} |".format(
                rank=rank,
                cell=row["cell"],
                freq=row["risk_frequency_by_id"] or "{}",
                impact=row["risk_impact_by_id"] or "{}",
                ret=float(row["order_ret_excel_mean"]),
                cvar=float(row["order_ret_excel_cvar05_mean"]),
                dret=float(row["delta_ret_excel_vs_static"]),
                dcvar=float(row["delta_cvar05_vs_static"]),
                cost=float(row["assembly_cost_index_mean"]),
                fill=float(row["order_excel_case_pct_fill_rate_mean"]),
                rec=float(row["order_excel_case_pct_recovery_mean"]),
            )
        )
    (grid_dir / "per_risk_cvar_panel.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
