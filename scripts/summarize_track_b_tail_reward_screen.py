#!/usr/bin/env python3
"""Summarize Track B tail/recovery reward screens.

This is an eval-only summarizer: it reads completed ``run_track_b_smoke.py``
bundles and compares the learned PPO row against a baseline bundle under the
same external metrics.  The primary bar remains Garrido Excel ReT; tail and
recovery metrics are reported to answer whether a reward is "buying"
resilience in the part of the distribution the mean can hide.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = Path(
    "outputs/experiments/track_b_tail_reward_screen_summary_2026-07-04"
)
DEFAULT_BASELINE_DIR = Path(
    "outputs/experiments/track_b_v7_calibration_3seed_30k_2026-07-04"
)

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
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument(
        "--run",
        action="append",
        nargs=2,
        metavar=("LABEL", "DIR"),
        required=True,
        help="Candidate label and completed run directory. Repeatable.",
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def learned_row(run_dir: Path) -> dict[str, str]:
    rows = read_csv_rows(run_dir / "policy_summary.csv")
    candidates = [
        row for row in rows
        if row.get("policy") in {"ppo", "recurrent_ppo", "ppo_mlp"}
    ]
    if not candidates:
        candidates = [
            row for row in rows
            if not str(row.get("policy", "")).startswith("s")
            and "heur" not in str(row.get("policy", ""))
        ]
    if not candidates:
        raise RuntimeError(f"No learned policy row found in {run_dir / 'policy_summary.csv'}")
    return candidates[0]


def value(row: dict[str, str], key: str) -> float:
    raw = row.get(key)
    if raw in (None, ""):
        return float("nan")
    return float(raw)


def pct_delta(candidate: float, baseline: float, direction: str) -> float:
    if baseline == 0.0:
        return float("nan")
    raw = (candidate - baseline) / abs(baseline)
    return -raw if direction == "lower" else raw


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

    baseline_dir = args.baseline_dir
    baseline = learned_row(baseline_dir)
    baseline_config = read_json(baseline_dir / "summary.json").get("config", {})

    rows: list[dict[str, Any]] = []
    for label, run_dir_raw in args.run:
        run_dir = Path(run_dir_raw)
        candidate = learned_row(run_dir)
        config = read_json(run_dir / "summary.json").get("config", {})
        for key, display, direction in METRICS:
            cand = value(candidate, key)
            base = value(baseline, key)
            rows.append(
                {
                    "label": label,
                    "reward_mode": config.get("reward_mode", ""),
                    "metric": key,
                    "metric_label": display,
                    "direction": direction,
                    "candidate": cand,
                    "baseline": base,
                    "delta": cand - base,
                    "relative_improvement": pct_delta(cand, base, direction),
                    "candidate_policy": candidate.get("policy", ""),
                    "baseline_policy": baseline.get("policy", ""),
                    "run_dir": str(run_dir),
                }
            )

    write_csv(out / "tail_reward_screen_summary.csv", rows)

    summary_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "baseline_dir": str(baseline_dir),
        "baseline_config": baseline_config,
        "runs": [
            {"label": label, "dir": str(Path(run_dir_raw))}
            for label, run_dir_raw in args.run
        ],
        "metrics": [
            {"key": key, "label": display, "direction": direction}
            for key, display, direction in METRICS
        ],
    }
    (out / "summary.json").write_text(
        json.dumps(summary_payload, indent=2, default=str),
        encoding="utf-8",
    )

    md_lines = [
        "# Track B tail/recovery reward screen",
        "",
        f"Generated: {summary_payload['generated_at']}",
        "",
        "Baseline: "
        f"`{baseline_dir}` (`reward_mode={baseline_config.get('reward_mode')}`, "
        f"seeds={baseline_config.get('seeds')}, "
        f"train_timesteps={baseline_config.get('train_timesteps')}).",
        "",
        "| Candidate | Reward | ReT Excel | CVaR05 | Worst 4w | Service-loss AUC/order | TTR mean | TTR p95 | Cost |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def metric_for(label: str, metric: str) -> dict[str, Any]:
        return next(row for row in rows if row["label"] == label and row["metric"] == metric)

    for label, _run_dir_raw in args.run:
        reward = metric_for(label, "order_ret_excel_mean")["reward_mode"]
        vals = {
            metric: metric_for(label, metric)["candidate"]
            for metric, _display, _direction in METRICS
        }
        md_lines.append(
            "| {label} | `{reward}` | {ret:.6f} | {cvar:.6f} | {worst:.6f} | "
            "{auc:.1f} | {ttr:.2f} | {ttr95:.2f} | {cost:.3f} |".format(
                label=label,
                reward=reward,
                ret=vals["order_ret_excel_mean"],
                cvar=vals["order_ret_excel_cvar05_mean"],
                worst=vals["order_ret_excel_rolling_4w_min_mean"],
                auc=vals["order_service_loss_auc_per_order_mean"],
                ttr=vals["order_ttr_mean_mean"],
                ttr95=vals["order_ttr_p95_mean"],
                cost=vals["assembly_cost_index_mean"],
            )
        )

    md_lines.extend(
        [
            "",
            "Positive tail/recovery evidence requires improving at least one tail/recovery metric "
            "without materially damaging Garrido Excel ReT.",
            "",
        ]
    )
    (out / "verdict.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Wrote {out / 'tail_reward_screen_summary.csv'}")
    print(f"Wrote {out / 'verdict.md'}")


if __name__ == "__main__":
    main()
