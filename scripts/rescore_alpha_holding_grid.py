#!/usr/bin/env python3
"""Re-score an α × holding-cost sweep under separate win definitions.

This is intentionally offline: it reads an existing grid_summary.csv and
static_frontier.json, then writes a report that separates raw Excel ReT wins
from Pareto/resource wins.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    numeric = {
        "excel",
        "cvar",
        "service_loss_mean",
        "resource",
        "frac_std",
        "cvar_alpha",
        "holding_cost",
        "seed",
        "train_seconds",
        "timesteps",
    }
    for row in rows:
        for key in numeric:
            if key in row and row[key] != "":
                row[key] = float(row[key])
    return rows


def _read_static(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    statics = payload["statics"] if isinstance(payload, dict) else payload
    return list(statics)


def _pareto(dynamic: dict[str, Any], statics: list[dict[str, Any]]) -> dict[str, Any]:
    dominated_by = []
    dominates = []
    for static in statics:
        static_no_worse = (
            static["excel"] >= dynamic["excel"]
            and static["cvar"] <= dynamic["cvar"]
            and static["resource"] <= dynamic["resource"]
        )
        static_better = (
            static["excel"] > dynamic["excel"]
            or static["cvar"] < dynamic["cvar"]
            or static["resource"] < dynamic["resource"]
        )
        if static_no_worse and static_better:
            dominated_by.append(static["label"])

        dyn_no_worse = (
            dynamic["excel"] >= static["excel"]
            and dynamic["cvar"] <= static["cvar"]
            and dynamic["resource"] <= static["resource"]
        )
        dyn_better = (
            dynamic["excel"] > static["excel"]
            or dynamic["cvar"] < static["cvar"]
            or dynamic["resource"] < static["resource"]
        )
        if dyn_no_worse and dyn_better:
            dominates.append(static["label"])
    return {
        "pareto_win": bool((not dominated_by) and dominates),
        "dominated_by_static": bool(dominated_by),
        "dominated_by": dominated_by[:10],
        "dominates_static_count": len(dominates),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sweep_dir", help="Directory containing grid_summary.csv and static_frontier.json")
    ap.add_argument("--out-name", default="raw_resilience_rescore")
    args = ap.parse_args()

    sweep_dir = Path(args.sweep_dir)
    rows = _read_rows(sweep_dir / "grid_summary.csv")
    statics = _read_static(sweep_dir / "static_frontier.json")
    if not rows or not statics:
        raise SystemExit("No rows/statics found")

    best_static_excel = max(statics, key=lambda s: s["excel"])
    best_static_cvar = min(statics, key=lambda s: s["cvar"])
    target = next((s for s in statics if s["label"] == "f0.1_S1"), best_static_excel)

    rescored = []
    for row in rows:
        row = dict(row)
        row["best_static_excel_label"] = best_static_excel["label"]
        row["best_static_excel"] = best_static_excel["excel"]
        row["raw_ret_delta_vs_best_static"] = row["excel"] - best_static_excel["excel"]
        row["raw_ret_win"] = row["raw_ret_delta_vs_best_static"] > 0
        row["best_static_cvar_label"] = best_static_cvar["label"]
        row["best_static_cvar"] = best_static_cvar["cvar"]
        row["cvar_delta_vs_best_static"] = row["cvar"] - best_static_cvar["cvar"]
        row["cvar_win"] = row["cvar_delta_vs_best_static"] < 0
        row["target_f0_10_excel"] = target["excel"]
        row["target_f0_10_cvar"] = target["cvar"]
        row["target_f0_10_resource"] = target["resource"]
        row["hard_resource_win"] = (
            row["excel"] >= target["excel"]
            and row["cvar"] <= target["cvar"]
            and row["resource"] <= 0.05
        )
        row.update(_pareto(row, statics))
        rescored.append(row)

    best_dynamic = max(rescored, key=lambda r: r["excel"])
    raw_winners = [r for r in rescored if r["raw_ret_win"]]
    cvar_winners = [r for r in rescored if r["cvar_win"]]
    pareto_winners = [r for r in rescored if r["pareto_win"]]
    hard_winners = [r for r in rescored if r["hard_resource_win"]]

    csv_path = sweep_dir / f"{args.out_name}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rescored[0].keys()))
        writer.writeheader()
        writer.writerows(rescored)

    report = [
        "# Raw Resilience Re-score",
        "",
        "Exploratory re-score of an existing sweep. This does not retrain and does not fix any non-CRN evaluation in the original run.",
        "",
        f"Best static ReT: `{best_static_excel['label']}` excel={best_static_excel['excel']:.6f}, "
        f"cvar={best_static_excel['cvar']:.2e}, resource={best_static_excel['resource']:.3f}.",
        f"Best dynamic ReT: alpha={best_dynamic['cvar_alpha']}, hc={best_dynamic['holding_cost']} "
        f"excel={best_dynamic['excel']:.6f}, "
        f"delta={best_dynamic['raw_ret_delta_vs_best_static']:+.6f}, "
        f"cvar={best_dynamic['cvar']:.2e}, resource={best_dynamic['resource']:.3f}.",
        "",
        f"Raw ReT winners: {len(raw_winners)}/{len(rescored)}.",
        f"CVaR winners: {len(cvar_winners)}/{len(rescored)}.",
        f"Pareto winners: {len(pareto_winners)}/{len(rescored)}.",
        f"Hard resource winners: {len(hard_winners)}/{len(rescored)}.",
        "",
        "## Top Dynamic Cells By Raw ReT",
        "",
        "| alpha | holding_cost | Excel | Δ vs best static | CVaR | Resource | Raw ReT | Pareto |",
        "|---:|---:|---:|---:|---:|---:|:---:|:---:|",
    ]
    for row in sorted(rescored, key=lambda r: r["excel"], reverse=True)[:10]:
        report.append(
            f"| {row['cvar_alpha']} | {row['holding_cost']} | {row['excel']:.6f} "
            f"| {row['raw_ret_delta_vs_best_static']:+.6f} | {row['cvar']:.2e} "
            f"| {row['resource']:.3f} | {'✅' if row['raw_ret_win'] else '—'} "
            f"| {'✅' if row['pareto_win'] else '—'} |"
        )
    (sweep_dir / f"{args.out_name}.md").write_text("\n".join(report), encoding="utf-8")

    print(f"Wrote {csv_path}")
    print(f"Raw ReT winners: {len(raw_winners)}/{len(rescored)}")
    print(
        "Best dynamic: "
        f"alpha={best_dynamic['cvar_alpha']} hc={best_dynamic['holding_cost']} "
        f"excel={best_dynamic['excel']:.6f} "
        f"delta={best_dynamic['raw_ret_delta_vs_best_static']:+.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
