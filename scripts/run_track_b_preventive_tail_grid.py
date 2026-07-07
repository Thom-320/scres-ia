#!/usr/bin/env python3
"""Mini-grid for a preventive Track B policy candidate.

This lane is deliberately narrow and reviewer-safe:

- Case C selected stress environment (R22/R23/R24, R24 frequency x3,
  R22/R23 impact x1.5).
- ``v10_no_forecast`` observation: v10 operational memory is available, but
  explicit forecast fields are masked for both the agent and the belief head.
- PPO+MLP only for the optimizer screen. Real-KAN should follow only if this
  produces causal preventive signal.
- Reward shaping uses a pre-trained R22 belief head and adds risk-conditioned
  readiness/exposure/tail terms. Final judgment is still unshaped Garrido
  Excel ReT and the R_full - R_reset(pre-risk) counterfactual.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Variant:
    label: str
    alpha: float
    beta: float
    kappa: float
    rho: float
    exposure: float
    backlog_age: float
    tail: float


VARIANTS = (
    Variant("belief_v2_control", 0.10, 0.05, 0.20, 0.05, 0.00, 0.00, 0.00),
    Variant("preventive_conservative", 0.05, 0.02, 0.30, 0.10, 0.20, 0.10, 0.10),
    Variant("preventive_balanced", 0.10, 0.05, 0.45, 0.12, 0.30, 0.15, 0.15),
    Variant("preventive_aggressive", 0.10, 0.05, 0.70, 0.18, 0.45, 0.25, 0.25),
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--belief-encoder-path", type=Path, required=True)
    ap.add_argument("--belief-head-path", type=Path, required=True)
    ap.add_argument("--belief-base-rate", type=float, default=0.1705)
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--train-timesteps", type=int, default=30_000)
    ap.add_argument("--eval-episodes", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=104)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--n-steps", type=int, default=1024)
    ap.add_argument("--n-epochs", type=int, default=10)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def csv_values(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def build_cmd(args: argparse.Namespace, variant: Variant, cell_dir: Path) -> list[str]:
    return [
        sys.executable,
        "scripts/run_track_b_future_credit_sidecar.py",
        "--output-dir",
        str(cell_dir),
        "--obs-config",
        "v10_no_forecast",
        "--credit-mode",
        "belief_conditioned_tail_pbrs",
        "--belief-encoder-path",
        str(args.belief_encoder_path),
        "--belief-head-path",
        str(args.belief_head_path),
        "--belief-target-index",
        "0",
        "--belief-base-rate",
        str(args.belief_base_rate),
        "--belief-mask-forecast",
        "--pbrs-alpha",
        str(variant.alpha),
        "--pbrs-beta",
        str(variant.beta),
        "--pbrs-kappa",
        str(variant.kappa),
        "--pbrs-rho",
        str(variant.rho),
        "--pbrs-exposure",
        str(variant.exposure),
        "--pbrs-backlog-age",
        str(variant.backlog_age),
        "--pbrs-tail",
        str(variant.tail),
        "--pbrs-gamma",
        "0.99",
        "--seeds",
        *csv_values(args.seeds),
        "--train-timesteps",
        str(args.train_timesteps),
        "--eval-episodes",
        str(args.eval_episodes),
        "--max-steps",
        str(args.max_steps),
        "--reward-mode",
        "control_v1",
        "--risk-level",
        "current",
        "--enabled-risks",
        "R22,R23,R24",
        "--risk-frequency-by-id",
        "R24=3",
        "--risk-impact-by-id",
        "R22=1.5,R23=1.5",
        "--faithful",
        "--learning-rate",
        str(args.learning_rate),
        "--n-steps",
        str(args.n_steps),
        "--batch-size",
        str(args.batch_size),
        "--n-epochs",
        str(args.n_epochs),
        "--gamma",
        "0.99",
        "--gae-lambda",
        "0.95",
        "--clip-range",
        "0.2",
        "--ent-coef",
        "0.0",
    ]


def policy_row(summary: dict[str, Any]) -> dict[str, Any]:
    return next(row for row in summary["policy_summary"] if row["policy"] == "ppo")


def result_row(args: argparse.Namespace, variant: Variant, cell_dir: Path) -> dict[str, Any]:
    summary_path = cell_dir / "summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    row = policy_row(payload)
    comparison = payload["comparison_table"][0]
    best_static = next(
        item
        for item in payload["policy_summary"]
        if item["policy"] == comparison["best_static_policy"]
    )
    ret = float(row["order_ret_excel_mean"])
    best = float(best_static["order_ret_excel_mean"])
    return {
        "variant": variant.label,
        "order_ret_excel_mean": ret,
        "order_ret_excel_cvar05_mean": float(row.get("order_ret_excel_cvar05_mean", 0.0)),
        "order_ret_excel_risk_conditional_mean_mean": float(
            row.get("order_ret_excel_risk_conditional_mean_mean", 0.0)
        ),
        "assembly_cost_index_mean": float(row.get("assembly_cost_index_mean", 0.0)),
        "best_static_policy": comparison["best_static_policy"],
        "best_static_order_ret_excel_mean": best,
        "delta_vs_best_static": ret - best,
        "relative_delta_vs_best_static_pct": 100.0 * (ret / best - 1.0) if best else 0.0,
        "pbrs_alpha": variant.alpha,
        "pbrs_beta": variant.beta,
        "pbrs_kappa": variant.kappa,
        "pbrs_rho": variant.rho,
        "pbrs_exposure": variant.exposure,
        "pbrs_backlog_age": variant.backlog_age,
        "pbrs_tail": variant.tail,
        "summary_json": str(summary_path),
    }


def write_rows(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    csv_path = output_dir / "preventive_tail_grid_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    md_path = output_dir / "preventive_tail_grid_results.md"
    headers = [
        "variant",
        "order_ret_excel_mean",
        "order_ret_excel_cvar05_mean",
        "relative_delta_vs_best_static_pct",
        "assembly_cost_index_mean",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["variant"]),
                    f"{float(row['order_ret_excel_mean']):.9f}",
                    f"{float(row['order_ret_excel_cvar05_mean']):.9f}",
                    f"{float(row['relative_delta_vs_best_static_pct']):.3f}",
                    f"{float(row['assembly_cost_index_mean']):.3f}",
                ]
            )
            + " |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        cell_dir = args.output_dir / variant.label
        summary_path = cell_dir / "summary.json"
        cmd = build_cmd(args, variant, cell_dir)
        print(f"\n=== {variant.label} ===", flush=True)
        print(" ".join(cmd), flush=True)
        if args.dry_run:
            continue
        if not summary_path.exists():
            subprocess.run(cmd, check=True)
        rows.append(result_row(args, variant, cell_dir))
        write_rows(args.output_dir, rows)
    if rows:
        best = max(rows, key=lambda row: float(row["order_ret_excel_mean"]))
        print(
            "BEST_BY_RET "
            f"{best['variant']} ret={float(best['order_ret_excel_mean']):.9f} "
            f"cvar05={float(best['order_ret_excel_cvar05_mean']):.9f} "
            f"rel_delta={float(best['relative_delta_vs_best_static_pct']):.3f}% "
            f"cost={float(best['assembly_cost_index_mean']):.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
