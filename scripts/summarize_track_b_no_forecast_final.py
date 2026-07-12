#!/usr/bin/env python3
"""Merge fixed-RNG Track B no-forecast bundles against the full v7 spine.

The observation-ablation runner writes one bundle per seed block and nests the
actual smoke output below ``v7_no_forecast/``.  This helper reads those bundles,
compares them seed-by-seed against the fixed-RNG full-v7 control bundles, and
writes a compact reviewer-facing summary using Garrido Excel ReT as the primary
metric.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scipy import stats


METRICS: tuple[tuple[str, str, str], ...] = (
    ("order_ret_excel_mean", "ReT Excel", "higher"),
    ("order_ret_excel_cvar05_mean", "ReT Excel CVaR05", "higher"),
    ("order_ret_excel_rolling_4w_min_mean", "Worst 4w ReT", "higher"),
    ("order_service_loss_auc_per_order_mean", "Service-loss AUC/order", "lower"),
    ("order_ttr_mean_mean", "TTR mean", "lower"),
    ("order_ttr_p95_mean", "TTR p95", "lower"),
    ("assembly_cost_index_mean", "Cost index", "lower"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--full-v7-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--no-forecast-dirs", nargs="+", type=Path, required=True)
    return parser.parse_args()


def _seed_metrics_path(bundle_dir: Path) -> Path:
    direct = bundle_dir / "seed_metrics.csv"
    if direct.exists():
        return direct
    nested = bundle_dir / "v7_no_forecast" / "seed_metrics.csv"
    if nested.exists():
        return nested
    raise FileNotFoundError(f"No seed_metrics.csv found under {bundle_dir}")


def read_rows(bundle_dirs: list[Path], *, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bundle_dir in bundle_dirs:
        path = _seed_metrics_path(bundle_dir)
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("policy") != "ppo":
                    continue
                parsed: dict[str, Any] = {
                    "bundle_dir": str(bundle_dir),
                    "label": label,
                    "seed": int(row["seed"]),
                    "policy": row["policy"],
                }
                for key, _metric_label, _direction in METRICS:
                    parsed[key] = float(row[key])
                rows.append(parsed)
    rows.sort(key=lambda row: int(row["seed"]))
    seen: set[int] = set()
    for row in rows:
        seed = int(row["seed"])
        if seed in seen:
            raise ValueError(f"Duplicate seed {seed} for {label}")
        seen.add(seed)
    return rows


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def ci95(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), float(values[0])
    mu = mean(values)
    var = sum((value - mu) ** 2 for value in values) / (len(values) - 1)
    tcrit = float(stats.t.ppf(0.975, df=len(values) - 1))
    half = tcrit * (var ** 0.5) / (len(values) ** 0.5)
    return float(mu - half), float(mu + half)


def aggregate(label: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "label": label,
        "n_seeds": len(rows),
        "seeds": [int(row["seed"]) for row in rows],
    }
    for key, _metric_label, _direction in METRICS:
        values = [float(row[key]) for row in rows]
        lo, hi = ci95(values)
        out[f"{key}_mean"] = mean(values)
        out[f"{key}_ci95_low"] = lo
        out[f"{key}_ci95_high"] = hi
    return out


def paired_rows(
    full_rows: list[dict[str, Any]], no_forecast_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_seed_full = {int(row["seed"]): row for row in full_rows}
    by_seed_no_forecast = {int(row["seed"]): row for row in no_forecast_rows}
    seeds = sorted(set(by_seed_full) & set(by_seed_no_forecast))
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        full = by_seed_full[seed]
        no_forecast = by_seed_no_forecast[seed]
        row: dict[str, Any] = {"seed": seed}
        for key, _metric_label, direction in METRICS:
            delta = float(no_forecast[key]) - float(full[key])
            row[f"{key}_full_v7"] = float(full[key])
            row[f"{key}_no_forecast"] = float(no_forecast[key])
            row[f"{key}_delta_no_forecast_minus_full"] = delta
            row[f"{key}_favorable_no_forecast"] = (
                delta > 0 if direction == "higher" else delta < 0
            )
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    full_rows = read_rows(args.full_v7_dirs, label="full_v7")
    no_forecast_rows = read_rows(args.no_forecast_dirs, label="no_forecast")
    paired = paired_rows(full_rows, no_forecast_rows)

    paired_summary: list[dict[str, Any]] = []
    for key, metric_label, direction in METRICS:
        delta_key = f"{key}_delta_no_forecast_minus_full"
        deltas = [float(row[delta_key]) for row in paired]
        favorable_key = f"{key}_favorable_no_forecast"
        favorable = sum(bool(row[favorable_key]) for row in paired)
        lo, hi = ci95(deltas)
        paired_summary.append(
            {
                "metric": key,
                "metric_label": metric_label,
                "direction": direction,
                "n_pairs": len(paired),
                "mean_delta_no_forecast_minus_full": mean(deltas),
                "ci95_low": lo,
                "ci95_high": hi,
                "favorable_no_forecast_pairs": favorable,
                "favorable_no_forecast_rate": favorable / len(paired)
                if paired
                else float("nan"),
            }
        )

    aggregates = [
        aggregate("full_v7", full_rows),
        aggregate("no_forecast", no_forecast_rows),
    ]

    write_csv(out / "merged_seed_metrics.csv", full_rows + no_forecast_rows)
    write_csv(out / "paired_seed_deltas.csv", paired)
    write_csv(out / "paired_summary.csv", paired_summary)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "full_v7_dirs": [str(path) for path in args.full_v7_dirs],
        "no_forecast_dirs": [str(path) for path in args.no_forecast_dirs],
        "aggregates": aggregates,
        "paired_summary": paired_summary,
    }
    (out / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Track B no-forecast fixed-RNG final summary",
        "",
        f"Generated: {payload['generated_at']}",
        "",
        "| Metric | No-forecast minus full-v7 | CI95 | No-forecast favorable | Direction |",
        "|---|---:|---:|---:|---|",
    ]
    for row in paired_summary:
        lines.append(
            "| {label} | {delta:+.8f} | [{lo:+.8f}, {hi:+.8f}] | {fav}/{n} | {direction} |".format(
                label=row["metric_label"],
                delta=float(row["mean_delta_no_forecast_minus_full"]),
                lo=float(row["ci95_low"]),
                hi=float(row["ci95_high"]),
                fav=int(row["favorable_no_forecast_pairs"]),
                n=int(row["n_pairs"]),
                direction=row["direction"],
            )
        )
    (out / "verdict.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out / 'summary.json'}")
    print(f"Wrote {out / 'verdict.md'}")


if __name__ == "__main__":
    main()
