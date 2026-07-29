#!/usr/bin/env python3
"""Track B Case C per-risk headroom grid.

This screen keeps the downstream Garrido-native risk set fixed (R22/R23/R24)
but changes frequency/impact by individual risk ID. The purpose is to create
interpretable adaptive/preventive headroom:

- R24 frequency tests learnable, recurring demand-surge pressure.
- R22/R23 impact tests whether downstream disruption severity makes early
  preparation valuable.
- Mixed cells test whether a frequent warning-like risk plus severe downstream
  disruptions creates enough margin for a preventive policy.
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


@dataclass(frozen=True)
class PerRiskCell:
    label: str
    description: str
    frequency_by_id: dict[str, float]
    impact_by_id: dict[str, float]


DEFAULT_CELLS: tuple[PerRiskCell, ...] = (
    PerRiskCell(
        label="base",
        description="R22/R23/R24 at Garrido current baseline.",
        frequency_by_id={},
        impact_by_id={},
    ),
    PerRiskCell(
        label="r24_freq2_impact1p25",
        description="Make R24 more learnable/frequent and mildly stronger.",
        frequency_by_id={"R24": 2.0},
        impact_by_id={"R24": 1.25},
    ),
    PerRiskCell(
        label="r24_freq3_impact1p0",
        description="High-frequency R24 headroom, matching the broad frequency screen.",
        frequency_by_id={"R24": 3.0},
        impact_by_id={},
    ),
    PerRiskCell(
        label="r22_impact1p5",
        description="Increase LOC disruption severity only.",
        frequency_by_id={},
        impact_by_id={"R22": 1.5},
    ),
    PerRiskCell(
        label="r23_impact1p5",
        description="Increase advanced-unit disruption severity only.",
        frequency_by_id={},
        impact_by_id={"R23": 1.5},
    ),
    PerRiskCell(
        label="r22r23_impact1p5",
        description="Increase downstream disruption severity without changing R24.",
        frequency_by_id={},
        impact_by_id={"R22": 1.5, "R23": 1.5},
    ),
    PerRiskCell(
        label="r24_freq2_r22r23_impact1p5",
        description="Frequent R24 plus severe downstream disruptions.",
        frequency_by_id={"R24": 2.0},
        impact_by_id={"R22": 1.5, "R23": 1.5},
    ),
    PerRiskCell(
        label="r24_freq3_r22r23_impact1p5",
        description="Max headroom cell: high R24 frequency plus severe R22/R23.",
        frequency_by_id={"R24": 3.0},
        impact_by_id={"R22": 1.5, "R23": 1.5},
    ),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--obs-config", default="v7_no_forecast")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--train-timesteps", type=int, default=30_000)
    parser.add_argument("--eval-episodes", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=104)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--reward-mode", default="control_v1")
    parser.add_argument("--enabled-risks", default="R22,R23,R24")
    parser.add_argument("--cell-limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def parse_csv_numbers(raw: str, cast: type = int) -> list[Any]:
    return [cast(item.strip()) for item in raw.split(",") if item.strip()]


def map_arg(values: dict[str, float]) -> str:
    return ",".join(f"{key}={value:g}" for key, value in sorted(values.items()))


def read_summary(
    summary_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    ppo = next(row for row in payload["policy_summary"] if row["policy"] == "ppo")
    comparison = payload["comparison_table"][0]
    best_static = next(
        row
        for row in payload["policy_summary"]
        if row["policy"] == comparison["best_static_policy"]
    )
    return payload, ppo, comparison, best_static


def build_command(args: argparse.Namespace, *, cell: PerRiskCell, cell_dir: Path) -> list[str]:
    seeds = parse_csv_numbers(args.seeds, int)
    cmd = [
        sys.executable,
        "scripts/run_track_b_observation_ablation.py",
        "--obs-configs",
        str(args.obs_config),
        "--output-dir",
        str(cell_dir),
        "--seeds",
        *(str(seed) for seed in seeds),
        "--train-timesteps",
        str(args.train_timesteps),
        "--eval-episodes",
        str(args.eval_episodes),
        "--max-steps",
        str(args.max_steps),
        "--reward-mode",
        str(args.reward_mode),
        "--risk-level",
        "current",
        "--risk-frequency-multiplier",
        "1.0",
        "--risk-impact-multiplier",
        "1.0",
        "--learning-rate",
        str(args.learning_rate),
        "--n-steps",
        str(args.n_steps),
        "--batch-size",
        str(args.batch_size),
        "--n-epochs",
        str(args.n_epochs),
        "--gamma",
        str(args.gamma),
        "--gae-lambda",
        str(args.gae_lambda),
        "--clip-range",
        str(args.clip_range),
        "--ent-coef",
        str(args.ent_coef),
        "--faithful",
        "--enabled-risks",
        str(args.enabled_risks),
    ]
    if cell.frequency_by_id:
        cmd.extend(["--risk-frequency-by-id", map_arg(cell.frequency_by_id)])
    if cell.impact_by_id:
        cmd.extend(["--risk-impact-by-id", map_arg(cell.impact_by_id)])
    return cmd


def write_results(output_dir: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "scenario": "garrido_downstream_cherry_per_risk",
        "enabled_risks": args.enabled_risks,
        "obs_config": args.obs_config,
        "seeds": args.seeds,
        "train_timesteps": int(args.train_timesteps),
        "eval_episodes": int(args.eval_episodes),
        "max_steps": int(args.max_steps),
        "batch_size": int(args.batch_size),
        "primary_metric": "order_ret_excel_mean",
        "note": (
            "Per-risk multipliers are continuous headroom knobs. Absolute ReT "
            "values are not comparable to all-risk Garrido because branch "
            "composition changes when only R22/R23/R24 are enabled."
        ),
    }
    (output_dir / "per_risk_headroom_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    if rows:
        with (output_dir / "per_risk_headroom_results.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    ordered = sorted(
        rows,
        key=lambda row: (
            -float(row["relative_delta_vs_static"]),
            -float(row["order_ret_excel_mean"]),
        ),
    )
    lines = [
        "# Track B Case C Per-Risk Headroom Grid",
        "",
        f"Updated UTC: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "| rank | cell | freq by id | impact by id | ReT Excel | CVaR05 ReT | rel delta vs static | delta CVaR05 | cost | fill-rate branch % | recovery branch % | no-recovery branch % |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(ordered, start=1):
        lines.append(
            "| {rank} | `{cell}` | `{freq}` | `{impact}` | {ret:.9f} | "
            "{cvar:.9f} | {rel:+.2%} | {delta_cvar:+.9f} | {cost:.3f} | "
            "{fill:.2f} | {recovery:.2f} | {no_rec:.2f} |".format(
                rank=rank,
                cell=row["cell"],
                freq=row["risk_frequency_by_id"] or "{}",
                impact=row["risk_impact_by_id"] or "{}",
                ret=float(row["order_ret_excel_mean"]),
                cvar=float(row["order_ret_excel_cvar05_mean"]),
                rel=float(row["relative_delta_vs_static"]),
                delta_cvar=float(row["delta_cvar05_vs_best_static"]),
                cost=float(row["assembly_cost_index_mean"]),
                fill=float(row["order_excel_case_pct_fill_rate_mean"]),
                recovery=float(row["order_excel_case_pct_recovery_mean"]),
                no_rec=float(row["order_excel_case_pct_risk_no_recovery_mean"]),
            )
        )
    (output_dir / "per_risk_headroom_results.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    args = build_parser().parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for index, cell in enumerate(DEFAULT_CELLS, start=1):
        if args.cell_limit is not None and index > int(args.cell_limit):
            break
        cell_dir = output_dir / cell.label
        summary_path = cell_dir / args.obs_config / "summary.json"
        cmd = build_command(args, cell=cell, cell_dir=cell_dir)
        print(f"\n=== per-risk headroom cell {index}: {cell.label} ===", flush=True)
        print(cell.description, flush=True)
        print(" ".join(cmd), flush=True)
        if args.dry_run:
            continue
        if not summary_path.exists():
            log_path = output_dir / f"{cell.label}.run.log"
            with log_path.open("w", encoding="utf-8") as log:
                proc = subprocess.run(
                    cmd,
                    cwd=Path.cwd(),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
            if proc.returncode != 0:
                raise SystemExit(
                    f"Per-risk cell failed with exit {proc.returncode}: "
                    f"{cell.label}; see {log_path}"
                )
        _, ppo, comparison, best_static_row = read_summary(summary_path)
        learned = float(ppo["order_ret_excel_mean"])
        best_static = float(best_static_row["order_ret_excel_mean"])
        row = {
            "cell": cell.label,
            "description": cell.description,
            "enabled_risks": args.enabled_risks,
            "risk_frequency_by_id": map_arg(cell.frequency_by_id),
            "risk_impact_by_id": map_arg(cell.impact_by_id),
            "order_ret_excel_mean": learned,
            "order_ret_excel_cvar05_mean": float(ppo["order_ret_excel_cvar05_mean"]),
            "order_ret_excel_p05_mean": float(ppo["order_ret_excel_p05_mean"]),
            "order_ret_excel_p10_mean": float(ppo["order_ret_excel_p10_mean"]),
            "order_level_ret_mean_mean": float(ppo["order_level_ret_mean_mean"]),
            "assembly_cost_index_mean": float(ppo["assembly_cost_index_mean"]),
            "best_static_policy": comparison["best_static_policy"],
            "best_static_order_ret_excel_mean": best_static,
            "best_static_order_ret_excel_cvar05_mean": float(
                best_static_row["order_ret_excel_cvar05_mean"]
            ),
            "delta_vs_best_static_excel": learned - best_static,
            "delta_cvar05_vs_best_static": float(ppo["order_ret_excel_cvar05_mean"])
            - float(best_static_row["order_ret_excel_cvar05_mean"]),
            "relative_delta_vs_static": (
                (learned - best_static) / abs(best_static) if best_static else float("nan")
            ),
            "order_excel_case_pct_fill_rate_mean": float(
                ppo["order_excel_case_pct_fill_rate_mean"]
            ),
            "order_excel_case_pct_recovery_mean": float(
                ppo["order_excel_case_pct_recovery_mean"]
            ),
            "order_excel_case_pct_risk_no_recovery_mean": float(
                ppo["order_excel_case_pct_risk_no_recovery_mean"]
            ),
            "order_excel_case_pct_unfulfilled_mean": float(
                ppo["order_excel_case_pct_unfulfilled_mean"]
            ),
            "summary_json": str(summary_path),
        }
        rows.append(row)
        write_results(output_dir, rows, args)


if __name__ == "__main__":
    main()
