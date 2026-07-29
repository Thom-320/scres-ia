#!/usr/bin/env python3
"""Merge Track B reward bundles and compare tail/recovery metrics.

The smoke runner writes one ``seed_metrics.csv`` per bundle.  This helper merges
multiple bundles for a baseline reward and a candidate reward, then reports the
same external metrics used in the tail/recovery audit.  It is intentionally
read-only: original experiment bundles remain untouched.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
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
    parser.add_argument("--baseline-label", default="control_v1")
    parser.add_argument("--candidate-label", default="ret_tail_v2")
    parser.add_argument("--baseline-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--candidate-dirs", nargs="+", type=Path, required=True)
    return parser.parse_args()


def read_seed_rows(dirs: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bundle_dir in dirs:
        path = bundle_dir / "seed_metrics.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("policy") != "ppo":
                    continue
                parsed: dict[str, Any] = {
                    "bundle_dir": str(bundle_dir),
                    "seed": int(row["seed"]),
                    "policy": row["policy"],
                }
                for key, _label, _direction in METRICS:
                    parsed[key] = float(row[key])
                rows.append(parsed)
    rows.sort(key=lambda r: int(r["seed"]))
    seen: set[int] = set()
    for row in rows:
        seed = int(row["seed"])
        if seed in seen:
            raise ValueError(f"Duplicate seed {seed} across bundles")
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
    var = sum((x - mu) ** 2 for x in values) / (len(values) - 1)
    tcrit = float(stats.t.ppf(0.975, df=len(values) - 1))
    half = tcrit * (var ** 0.5) / (len(values) ** 0.5)
    return float(mu - half), float(mu + half)


def aggregate(label: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "label": label,
        "n_seeds": len(rows),
        "seeds": [int(r["seed"]) for r in rows],
    }
    for key, _metric_label, _direction in METRICS:
        values = [float(r[key]) for r in rows]
        lo, hi = ci95(values)
        out[f"{key}_mean"] = mean(values)
        out[f"{key}_ci95_low"] = lo
        out[f"{key}_ci95_high"] = hi
    return out


def paired_rows(
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_seed_base = {int(r["seed"]): r for r in baseline_rows}
    by_seed_cand = {int(r["seed"]): r for r in candidate_rows}
    seeds = sorted(set(by_seed_base) & set(by_seed_cand))
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        b = by_seed_base[seed]
        c = by_seed_cand[seed]
        row: dict[str, Any] = {"seed": seed}
        for key, _label, direction in METRICS:
            delta = float(c[key]) - float(b[key])
            row[f"{key}_baseline"] = float(b[key])
            row[f"{key}_candidate"] = float(c[key])
            row[f"{key}_delta"] = delta
            row[f"{key}_favorable"] = delta > 0 if direction == "higher" else delta < 0
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

    baseline_rows = read_seed_rows(args.baseline_dirs)
    candidate_rows = read_seed_rows(args.candidate_dirs)
    paired = paired_rows(baseline_rows, candidate_rows)

    aggregates = [
        aggregate(args.baseline_label, baseline_rows),
        aggregate(args.candidate_label, candidate_rows),
    ]

    paired_summary: list[dict[str, Any]] = []
    for key, label, direction in METRICS:
        deltas = [float(row[f"{key}_delta"]) for row in paired]
        favorable = sum(bool(row[f"{key}_favorable"]) for row in paired)
        lo, hi = ci95(deltas)
        paired_summary.append(
            {
                "metric": key,
                "metric_label": label,
                "direction": direction,
                "n_pairs": len(paired),
                "mean_delta": mean(deltas),
                "ci95_low": lo,
                "ci95_high": hi,
                "favorable_pairs": favorable,
                "favorable_rate": favorable / len(paired) if paired else float("nan"),
            }
        )

    write_csv(out / "merged_seed_metrics.csv", baseline_rows + candidate_rows)
    write_csv(out / "paired_seed_deltas.csv", paired)
    write_csv(out / "paired_summary.csv", paired_summary)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "baseline_dirs": [str(p) for p in args.baseline_dirs],
        "candidate_dirs": [str(p) for p in args.candidate_dirs],
        "aggregates": aggregates,
        "paired_summary": paired_summary,
    }
    (out / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Track B tail/recovery merged extension summary",
        "",
        f"Generated: {payload['generated_at']}",
        "",
        "| Metric | Mean delta | CI95 | Favorable pairs | Direction |",
        "|---|---:|---:|---:|---|",
    ]
    for row in paired_summary:
        lines.append(
            "| {label} | {delta:+.8f} | [{lo:+.8f}, {hi:+.8f}] | {fav}/{n} | {direction} |".format(
                label=row["metric_label"],
                delta=float(row["mean_delta"]),
                lo=float(row["ci95_low"]),
                hi=float(row["ci95_high"]),
                fav=int(row["favorable_pairs"]),
                n=int(row["n_pairs"]),
                direction=row["direction"],
            )
        )
    (out / "verdict.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out / 'summary.json'}")
    print(f"Wrote {out / 'verdict.md'}")


if __name__ == "__main__":
    main()
