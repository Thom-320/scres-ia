#!/usr/bin/env python3
"""Small Track B reward/observation sweep for adaptive_benchmark_v2.

This orchestrates the canonical Track B smoke runner rather than duplicating
training logic. It is intentionally a screening harness: short runs, explicit
Excel-ReT gates, and no headline claim without a later confirmatory run.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_ROOT = Path("outputs/experiments/track_b_adaptive_sweep")
BASELINE_RET_DELTA = 0.000415
BASELINE_EXCEL_RET_DELTA = 0.000470
BASELINE_CVAR05_DELTA = 0.000506
DEFAULT_COST_CAP = 0.70


@dataclass(frozen=True)
class SweepCell:
    reward_mode: str
    observation_version: str
    ret_excel_cvar_alpha: float | None = None

    @property
    def label(self) -> str:
        alpha = (
            ""
            if self.ret_excel_cvar_alpha is None
            else f"_a{self.ret_excel_cvar_alpha:g}"
        )
        return f"{self.reward_mode}{alpha}_{self.observation_version}".replace(".", "p")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--train-timesteps", type=int, default=40_000)
    parser.add_argument("--eval-episodes", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=104)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--reward-modes", default="control_v1,ReT_excel_plus_cvar,ReT_tail_v2,ReT_garrido2024_train")
    parser.add_argument("--observation-versions", default="v7,v8,v9")
    parser.add_argument("--cvar-alphas", default="0.05,0.1,0.2")
    parser.add_argument("--cost-cap", type=float, default=DEFAULT_COST_CAP)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Rebuild sweep_summary.* from existing cell summary.json files without training.",
    )
    return parser.parse_args()


def _split_csv(text: str) -> list[str]:
    return [item.strip() for item in str(text).split(",") if item.strip()]


def build_cells(args: argparse.Namespace) -> list[SweepCell]:
    reward_modes = _split_csv(args.reward_modes)
    obs_versions = _split_csv(args.observation_versions)
    cvar_alphas = [float(item) for item in _split_csv(args.cvar_alphas)]
    cells: list[SweepCell] = []
    for obs_version in obs_versions:
        for reward_mode in reward_modes:
            if reward_mode == "ReT_excel_plus_cvar":
                for alpha in cvar_alphas:
                    cells.append(SweepCell(reward_mode, obs_version, alpha))
            else:
                cells.append(SweepCell(reward_mode, obs_version, None))
    return cells


def _policy_rows(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["policy"]): row for row in summary.get("policy_summary", [])}


def _metric(row: dict[str, Any], key: str) -> float:
    for candidate in (
        f"{key}_mean",
        f"{key}_mean_mean",
        key,
    ):
        if candidate in row and row[candidate] not in (None, ""):
            return float(row[candidate])
    return 0.0


RICH_METRICS: tuple[str, ...] = (
    # Primary Garrido/Excel bar and tail.
    "order_ret_excel",
    "order_ret_excel_cvar05",
    "order_ret_excel_p05",
    "order_ret_excel_p50",
    "order_ret_excel_p95",
    "order_ration_ret_excel",
    "order_ret_excel_rolling_4w_mean",
    "order_ret_excel_rolling_4w_min",
    "order_ret_excel_rolling_4w_final",
    # Thesis/continuous alternatives for audit only.
    "order_ret_thesis",
    "order_ret_continuous",
    "order_level_ret_mean",
    # Garrido-style operational time metrics.
    "order_apj_p99",
    "order_ctj_p99",
    "order_rpj_p99",
    "order_dpj_p99",
    "order_ttr_mean",
    "order_ttr_p95",
    # Service and backlog.
    "flow_fill_rate",
    "terminal_rolling_fill_rate_4w",
    "terminal_rolling_backorder_rate_4w",
    "order_fill_rate",
    "order_fill_rate_on_time",
    "order_lost_rate",
    "order_backorder_qty_final",
    "order_service_loss_auc_per_order",
    # Cobb-Douglas / Garrido 2024 indices.
    "ret_garrido2024_sigmoid_mean",
    "ret_garrido2024_sigmoid_total",
    "ret_garrido2024_train_total",
    "ret_garrido2024_raw_total",
    "terminal_zeta_avg",
    "terminal_epsilon_avg",
    "terminal_phi_avg",
    "terminal_tau_avg",
    "terminal_kappa_dot",
    # Resource/action mechanism.
    "assembly_cost_index",
    "pct_steps_S1",
    "pct_steps_S2",
    "pct_steps_S3",
    "op10_multiplier_step_mean",
    "op12_multiplier_step_mean",
    "op10_multiplier_step_p95",
    "op12_multiplier_step_p95",
)


LOWER_IS_BETTER: set[str] = {
    "order_ctj_p99",
    "order_rpj_p99",
    "order_dpj_p99",
    "order_ttr_mean",
    "order_ttr_p95",
    "terminal_rolling_backorder_rate_4w",
    "order_lost_rate",
    "order_backorder_qty_final",
    "order_service_loss_auc_per_order",
    "assembly_cost_index",
}


def summarize_cell(cell: SweepCell, run_dir: Path, cost_cap: float) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return {
            "cell": cell.label,
            "reward_mode": cell.reward_mode,
            "observation_version": cell.observation_version,
            "ret_excel_cvar_alpha": cell.ret_excel_cvar_alpha,
            "status": "missing_summary",
            "run_dir": str(run_dir),
        }
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    policies = _policy_rows(summary)
    learned_name = str(summary.get("decision", {}).get("learned_policy", "ppo"))
    learned = policies.get(learned_name)
    statics = [row for name, row in policies.items() if str(name).startswith("s")]
    best_static = max(
        statics,
        key=lambda row: _metric(row, "order_ret_excel"),
        default=None,
    )
    if learned is None or best_static is None:
        status = "missing_policy_rows"
        excel_ret_delta = order_level_ret_delta = cvar_delta = cost = float("nan")
        best_static_name = ""
        rich_values: dict[str, float] = {}
    else:
        excel_ret_delta = _metric(learned, "order_ret_excel") - _metric(
            best_static, "order_ret_excel"
        )
        order_level_ret_delta = _metric(learned, "order_level_ret_mean") - _metric(
            best_static, "order_level_ret_mean"
        )
        cvar_delta = _metric(learned, "order_ret_excel_cvar05") - _metric(
            best_static, "order_ret_excel_cvar05"
        )
        cost = _metric(learned, "assembly_cost_index")
        best_static_name = str(best_static["policy"])
        status = "ok"
        rich_values = {}
        for metric in RICH_METRICS:
            learned_value = _metric(learned, metric)
            static_value = _metric(best_static, metric)
            delta = learned_value - static_value
            if metric in LOWER_IS_BETTER:
                win = delta < 0.0
            else:
                win = delta > 0.0
            rich_values[f"learned_{metric}"] = learned_value
            rich_values[f"best_static_{metric}"] = static_value
            rich_values[f"delta_{metric}"] = delta
            rich_values[f"win_{metric}"] = win
    promote = bool(
        status == "ok"
        and (
            excel_ret_delta > BASELINE_EXCEL_RET_DELTA
            or cvar_delta > BASELINE_CVAR05_DELTA
        )
        and (cost <= cost_cap or cvar_delta > 2.0 * BASELINE_CVAR05_DELTA)
    )
    return {
        "cell": cell.label,
        "reward_mode": cell.reward_mode,
        "observation_version": cell.observation_version,
        "ret_excel_cvar_alpha": cell.ret_excel_cvar_alpha,
        "status": status,
        "learned_policy": learned_name,
        "best_static_policy": best_static_name,
        "excel_ret_delta_vs_best_static": excel_ret_delta,
        "order_level_ret_delta_vs_best_static": order_level_ret_delta,
        "cvar05_delta_vs_best_static": cvar_delta,
        "learned_cost_index": cost,
        "promote": promote,
        "run_dir": str(run_dir),
        **rich_values,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or (DEFAULT_OUTPUT_ROOT / timestamp)
    output_dir.mkdir(parents=True, exist_ok=True)
    cells = build_cells(args)
    rows: list[dict[str, Any]] = []

    for idx, cell in enumerate(cells, start=1):
        run_dir = output_dir / cell.label
        cmd = [
            sys.executable,
            "scripts/run_track_b_smoke.py",
            "--output-dir",
            str(run_dir),
            "--reward-mode",
            cell.reward_mode,
            "--observation-version",
            cell.observation_version,
            "--risk-level",
            "adaptive_benchmark_v2",
            "--train-timesteps",
            str(args.train_timesteps),
            "--eval-episodes",
            str(args.eval_episodes),
            "--max-steps",
            str(args.max_steps),
            "--n-envs",
            str(args.n_envs),
            "--n-steps",
            str(args.n_steps),
            "--batch-size",
            str(args.batch_size),
            "--learning-rate",
            str(args.learning_rate),
            "--seeds",
            *(str(seed) for seed in args.seeds),
        ]
        if cell.ret_excel_cvar_alpha is not None:
            cmd.extend(["--ret-excel-cvar-alpha", str(cell.ret_excel_cvar_alpha)])
        print(f"[{idx}/{len(cells)}] {' '.join(cmd)}", flush=True)
        if not args.dry_run and not args.summarize_only:
            subprocess.run(cmd, check=True)
        rows.append(summarize_cell(cell, run_dir, args.cost_cap))
        write_csv(output_dir / "sweep_summary.csv", rows)
        (output_dir / "sweep_summary.json").write_text(
            json.dumps({"cells": rows}, indent=2), encoding="utf-8"
        )

    promoted = [row for row in rows if row.get("promote")]
    (output_dir / "promotion_decision.json").write_text(
        json.dumps(
            {
                "promotion_rule": {
                    "excel_ret_delta_gt": BASELINE_EXCEL_RET_DELTA,
                    "order_level_ret_delta_reference": BASELINE_RET_DELTA,
                    "cvar05_delta_gt": BASELINE_CVAR05_DELTA,
                    "cost_cap": args.cost_cap,
                },
                "promoted": promoted,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"WROTE {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
