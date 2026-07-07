#!/usr/bin/env python3
"""Screen Track B horizons before final confirmation.

The goal is not to crown a final result; it is to choose a defensible episode
horizon for the expensive PPO/Real-KAN confirmations.  We keep compute modest
and compare horizons within the same scenario.
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
class Scenario:
    label: str
    risk_level: str
    faithful: bool
    enabled_risks: str | None = None


SCENARIOS: dict[str, Scenario] = {
    "case_a_all_risks": Scenario("case_a_all_risks", "current", True, None),
    "case_b_downstream": Scenario(
        "case_b_downstream", "current", True, "R22,R23,R24"
    ),
}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--scenarios", default="case_a_all_risks,case_b_downstream")
    ap.add_argument("--horizons", default="52,104,156,260")
    ap.add_argument("--obs-config", default="v7_no_forecast")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--train-timesteps", type=int, default=20_000)
    ap.add_argument("--eval-episodes", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--n-steps", type=int, default=1024)
    ap.add_argument("--n-epochs", type=int, default=10)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--reward-mode", default="control_v1")
    ap.add_argument("--cell-limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    return ap


def parse_csv(raw: str, cast: type = str) -> list[Any]:
    return [cast(item.strip()) for item in raw.split(",") if item.strip()]


def slug(label: str) -> str:
    return label.replace(".", "p").replace("-", "m")


def command_for(
    *,
    scenario: Scenario,
    horizon: int,
    cell_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    seeds = parse_csv(args.seeds, int)
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
        str(horizon),
        "--reward-mode",
        str(args.reward_mode),
        "--risk-level",
        scenario.risk_level,
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
        "0.99",
        "--gae-lambda",
        "0.95",
        "--clip-range",
        "0.2",
        "--ent-coef",
        "0.0",
    ]
    if scenario.faithful:
        cmd.append("--faithful")
    if scenario.enabled_risks:
        cmd.extend(["--enabled-risks", scenario.enabled_risks])
    return cmd


def ppo_row(summary_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    ppo = next(row for row in payload["policy_summary"] if row["policy"] == "ppo")
    comparison = payload["comparison_table"][0]
    best_static = next(
        row
        for row in payload["policy_summary"]
        if row["policy"] == comparison["best_static_policy"]
    )
    return ppo, comparison, best_static


def write_results(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    if rows:
        with (output_dir / "horizon_screen_results.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    lines = [
        "# Track B Horizon Screen",
        "",
        f"Updated UTC: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "Purpose: choose a defensible episode horizon before expensive PPO/Real-KAN final confirmations.",
        "",
        "| scenario | horizon_weeks | horizon_years | ReT Excel | CVaR05 | risk-cond ReT | best static | delta ReT | delta % | cost |",
        "|---|---:|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda r: (r["scenario"], int(r["horizon_weeks"]))):
        lines.append(
            "| {scenario} | {horizon_weeks} | {horizon_years:.2f} | {ret:.9f} | "
            "{cvar:.9f} | {risk_cond:.9f} | `{best_static}` | {delta:+.9f} | "
            "{delta_pct:+.2f}% | {cost:.3f} |".format(
                scenario=row["scenario"],
                horizon_weeks=int(row["horizon_weeks"]),
                horizon_years=float(row["horizon_years"]),
                ret=float(row["order_ret_excel_mean"]),
                cvar=float(row["order_ret_excel_cvar05_mean"]),
                risk_cond=float(row["order_ret_excel_risk_conditional_mean_mean"]),
                best_static=row["best_static_policy"],
                delta=float(row["delta_vs_best_static"]),
                delta_pct=float(row["relative_delta_vs_best_static_pct"]),
                cost=float(row["assembly_cost_index_mean"]),
            )
        )
    (output_dir / "horizon_screen_results.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scenarios = [SCENARIOS[item] for item in parse_csv(args.scenarios, str)]
    horizons = parse_csv(args.horizons, int)
    cells: list[tuple[Scenario, int]] = [
        (scenario, horizon) for scenario in scenarios for horizon in horizons
    ]
    if args.cell_limit is not None:
        cells = cells[: int(args.cell_limit)]

    rows: list[dict[str, Any]] = []
    for scenario, horizon in cells:
        cell = f"{scenario.label}_h{horizon}"
        cell_dir = args.output_dir / cell
        cmd = command_for(scenario=scenario, horizon=horizon, cell_dir=cell_dir, args=args)
        print(f"\n=== horizon cell: {cell} ===", flush=True)
        print(" ".join(cmd), flush=True)
        if args.dry_run:
            continue
        log_path = args.output_dir / f"{cell}.run.log"
        with log_path.open("w", encoding="utf-8") as log:
            subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
        summary_path = cell_dir / args.obs_config / "summary.json"
        ppo, comparison, best_static = ppo_row(summary_path)
        ret = float(ppo["order_ret_excel_mean"])
        best = float(best_static["order_ret_excel_mean"])
        row = {
            "scenario": scenario.label,
            "horizon_weeks": int(horizon),
            "horizon_years": float(horizon) / 52.0,
            "cell": cell,
            "order_ret_excel_mean": ret,
            "order_ret_excel_cvar05_mean": float(
                ppo.get("order_ret_excel_cvar05_mean", 0.0)
            ),
            "order_ret_excel_risk_conditional_mean_mean": float(
                ppo.get("order_ret_excel_risk_conditional_mean_mean", 0.0)
            ),
            "assembly_cost_index_mean": float(ppo.get("assembly_cost_index_mean", 0.0)),
            "best_static_policy": comparison["best_static_policy"],
            "best_static_order_ret_excel_mean": best,
            "delta_vs_best_static": ret - best,
            "relative_delta_vs_best_static_pct": 100.0 * (ret / best - 1.0)
            if best
            else 0.0,
            "summary_json": str(summary_path),
        }
        rows.append(row)
        write_results(args.output_dir, rows)


if __name__ == "__main__":
    main()
