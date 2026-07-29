#!/usr/bin/env python3
"""Mini-grid for Ruta B live auxiliary PPO on Track B Case C.

This is the high-cost but direct test of the preventive-learning hypothesis:
keep the future-risk prediction task alive during PPO updates via
``RutaBAuxPPO`` instead of pretraining once and letting PPO repurpose the
representation.
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
class RutaBVariant:
    label: str
    aux_coef: float
    lead_weeks: int
    target_risks: tuple[str, ...]


VARIANTS = (
    RutaBVariant("r22_l4_c025", 0.25, 4, ("R22",)),
    RutaBVariant("r22_l4_c050", 0.50, 4, ("R22",)),
    RutaBVariant("r22r24_l2_c025", 0.25, 2, ("R22", "R24")),
    RutaBVariant("r22r24_l4_c025", 0.25, 4, ("R22", "R24")),
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--train-timesteps", type=int, default=30_000)
    ap.add_argument("--eval-episodes", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=104)
    ap.add_argument("--n-steps", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--n-epochs", type=int, default=10)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--obs-config", default="v10_no_forecast")
    ap.add_argument("--max-events-per-risk-episode", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def csv_values(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def build_cmd(args: argparse.Namespace, variant: RutaBVariant, cell_dir: Path) -> list[str]:
    return [
        sys.executable,
        "scripts/run_track_b_ruta_b_sidecar.py",
        "--output-dir",
        str(cell_dir),
        "--seeds",
        *csv_values(args.seeds),
        "--train-timesteps",
        str(args.train_timesteps),
        "--eval-episodes",
        str(args.eval_episodes),
        "--max-steps",
        str(args.max_steps),
        "--n-steps",
        str(args.n_steps),
        "--batch-size",
        str(args.batch_size),
        "--n-epochs",
        str(args.n_epochs),
        "--learning-rate",
        str(args.learning_rate),
        "--aux-coef",
        str(variant.aux_coef),
        "--aux-lead-weeks",
        str(variant.lead_weeks),
        "--aux-target-risks",
        *variant.target_risks,
        "--obs-config",
        str(args.obs_config),
        "--max-events-per-risk-episode",
        str(args.max_events_per_risk_episode),
    ]


def row_from_summary(variant: RutaBVariant, summary_path: Path) -> dict[str, Any]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        "variant": variant.label,
        "aux_coef": variant.aux_coef,
        "aux_lead_weeks": variant.lead_weeks,
        "aux_target_risks": ",".join(variant.target_risks),
        "ruta_b_ret_excel_mean": float(payload["ruta_b_ret_excel_mean"]),
        "ruta_b_cost_index_mean": float(payload["ruta_b_cost_index_mean"]),
        "best_static_policy": payload["best_static_policy"],
        "best_static_ret_excel_mean": float(payload["best_static_ret_excel_mean"]),
        "delta_vs_best_static": float(payload["delta_vs_best_static"]),
        "relative_delta_vs_best_static_pct": float(payload["relative_delta_vs_best_static_pct"]),
        "counterfactual_n_pairs": int(payload["counterfactual_n_pairs"]),
        "counterfactual_n_positive": int(payload["counterfactual_n_positive"]),
        "counterfactual_positive_rate": float(payload["counterfactual_positive_rate"]),
        "counterfactual_mean_delta_ret_excel": float(payload["counterfactual_mean_delta_ret_excel"]),
        "summary_json": str(summary_path),
    }


def write_rows(out: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    csv_path = out / "ruta_b_grid_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    headers = [
        "variant",
        "ruta_b_ret_excel_mean",
        "relative_delta_vs_best_static_pct",
        "ruta_b_cost_index_mean",
        "counterfactual_positive_rate",
        "counterfactual_mean_delta_ret_excel",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["variant"]),
                    f"{float(row['ruta_b_ret_excel_mean']):.9f}",
                    f"{float(row['relative_delta_vs_best_static_pct']):.3f}",
                    f"{float(row['ruta_b_cost_index_mean']):.3f}",
                    f"{float(row['counterfactual_positive_rate']):.3f}",
                    f"{float(row['counterfactual_mean_delta_ret_excel']):+.8f}",
                ]
            )
            + " |"
        )
    (out / "ruta_b_grid_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
        rows.append(row_from_summary(variant, summary_path))
        write_rows(args.output_dir, rows)
    if rows:
        best = max(
            rows,
            key=lambda row: (
                float(row["counterfactual_positive_rate"]),
                float(row["counterfactual_mean_delta_ret_excel"]),
                float(row["ruta_b_ret_excel_mean"]),
            ),
        )
        print(
            "BEST_BY_PREVENTION "
            f"{best['variant']} ret={float(best['ruta_b_ret_excel_mean']):.9f} "
            f"pos_rate={float(best['counterfactual_positive_rate']):.3f} "
            f"delta={float(best['counterfactual_mean_delta_ret_excel']):+.8f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
