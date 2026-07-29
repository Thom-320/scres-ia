#!/usr/bin/env python3
"""Final Track B scenario confirmation for PPO+MLP and Real-KAN.

This runner is intentionally scenario-first: Case A, Case B, and an optional
Case C cell are evaluated with the same horizon, seeds, reward, observation
contract, and action contract.  It delegates training/evaluation to the existing
PPO and Real-KAN runners so each artifact remains independently auditable.
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
    description: str
    risk_level: str = "current"
    faithful: bool = True
    enabled_risks: str | None = None
    risk_frequency_by_id: dict[str, float] | None = None
    risk_impact_by_id: dict[str, float] | None = None


BASE_SCENARIOS: dict[str, Scenario] = {
    "case_a_all_risks": Scenario(
        label="case_a_all_risks",
        description="Garrido current risk level with all thesis risks active.",
    ),
    "case_b_downstream": Scenario(
        label="case_b_downstream",
        description="Garrido current risk level with R22/R23/R24 only.",
        enabled_risks="R22,R23,R24",
    ),
}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--scenarios", default="case_a_all_risks,case_b_downstream")
    ap.add_argument("--architectures", default="ppo,real_kan")
    ap.add_argument("--obs-config", default="v7_no_forecast")
    ap.add_argument("--seeds", default="1,2,3,4,5")
    ap.add_argument("--train-timesteps", type=int, default=60_000)
    ap.add_argument("--eval-episodes", type=int, default=12)
    ap.add_argument("--max-steps", type=int, default=104)
    ap.add_argument("--batch-size-ppo", type=int, default=64)
    ap.add_argument("--batch-size-kan", type=int, default=256)
    ap.add_argument("--n-steps", type=int, default=1024)
    ap.add_argument("--n-epochs", type=int, default=10)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--reward-mode", default="control_v1")
    ap.add_argument(
        "--case-c-cell-json",
        type=Path,
        default=None,
        help=(
            "Optional JSON file describing a Case C scenario with keys "
            "label, enabled_risks, risk_frequency_by_id, risk_impact_by_id."
        ),
    )
    ap.add_argument("--cell-limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    return ap


def parse_csv(raw: str, cast: type = str) -> list[Any]:
    return [cast(item.strip()) for item in raw.split(",") if item.strip()]


def map_arg(values: dict[str, float] | None) -> str | None:
    if not values:
        return None
    return ",".join(f"{key}={value:g}" for key, value in sorted(values.items()))


def load_case_c(path: Path | None) -> Scenario | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return Scenario(
        label=str(payload.get("label", "case_c_selected")),
        description=str(payload.get("description", "Selected Case C per-risk cell.")),
        risk_level=str(payload.get("risk_level", "current")),
        faithful=bool(payload.get("faithful", True)),
        enabled_risks=str(payload.get("enabled_risks", "R22,R23,R24")),
        risk_frequency_by_id={
            str(k): float(v) for k, v in dict(payload.get("risk_frequency_by_id", {})).items()
        },
        risk_impact_by_id={
            str(k): float(v) for k, v in dict(payload.get("risk_impact_by_id", {})).items()
        },
    )


def scenario_list(args: argparse.Namespace) -> list[Scenario]:
    scenarios = dict(BASE_SCENARIOS)
    case_c = load_case_c(args.case_c_cell_json)
    if case_c is not None:
        scenarios[case_c.label] = case_c
    return [scenarios[item] for item in parse_csv(args.scenarios, str)]


def common_args(
    *,
    scenario: Scenario,
    output_dir: Path,
    args: argparse.Namespace,
    batch_size: int,
) -> list[str]:
    seeds = parse_csv(args.seeds, int)
    cmd = [
        "--output-dir",
        str(output_dir),
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
        str(batch_size),
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
    freq = map_arg(scenario.risk_frequency_by_id)
    impact = map_arg(scenario.risk_impact_by_id)
    if freq:
        cmd.extend(["--risk-frequency-by-id", freq])
    if impact:
        cmd.extend(["--risk-impact-by-id", impact])
    return cmd


def build_command(
    *,
    architecture: str,
    scenario: Scenario,
    cell_dir: Path,
    args: argparse.Namespace,
) -> tuple[list[str], Path]:
    if architecture == "ppo":
        cmd = [
            sys.executable,
            "scripts/run_track_b_observation_ablation.py",
            "--obs-configs",
            str(args.obs_config),
            *common_args(
                scenario=scenario,
                output_dir=cell_dir,
                args=args,
                batch_size=int(args.batch_size_ppo),
            ),
        ]
        summary = cell_dir / str(args.obs_config) / "summary.json"
        return cmd, summary
    if architecture == "real_kan":
        cmd = [
            sys.executable,
            "scripts/run_track_b_real_kan_sidecar.py",
            "--obs-config",
            str(args.obs_config),
            "--kan-features-dim",
            "32",
            "--kan-hidden-width",
            "32",
            "--kan-grid",
            "3",
            "--kan-k",
            "3",
            *common_args(
                scenario=scenario,
                output_dir=cell_dir,
                args=args,
                batch_size=int(args.batch_size_kan),
            ),
        ]
        summary = cell_dir / "summary.json"
        return cmd, summary
    raise ValueError(f"Unknown architecture: {architecture}")


def learned_policy(summary: dict[str, Any], architecture: str) -> dict[str, Any]:
    policy_name = "ppo_real_kan" if architecture == "real_kan" else "ppo"
    return next(row for row in summary["policy_summary"] if row["policy"] == policy_name)


def result_row(
    *,
    scenario: Scenario,
    architecture: str,
    summary_path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    row = learned_policy(payload, architecture)
    comparison = payload["comparison_table"][0]
    best_static = next(
        item
        for item in payload["policy_summary"]
        if item["policy"] == comparison["best_static_policy"]
    )
    ret = float(row["order_ret_excel_mean"])
    best = float(best_static["order_ret_excel_mean"])
    return {
        "scenario": scenario.label,
        "architecture": architecture,
        "horizon_weeks": int(args.max_steps),
        "horizon_years": float(args.max_steps) / 52.0,
        "order_ret_excel_mean": ret,
        "order_ret_excel_cvar05_mean": float(row.get("order_ret_excel_cvar05_mean", 0.0)),
        "order_ret_excel_risk_conditional_mean_mean": float(
            row.get("order_ret_excel_risk_conditional_mean_mean", 0.0)
        ),
        "assembly_cost_index_mean": float(row.get("assembly_cost_index_mean", 0.0)),
        "best_static_policy": comparison["best_static_policy"],
        "best_static_order_ret_excel_mean": best,
        "delta_vs_best_static": ret - best,
        "relative_delta_vs_best_static_pct": 100.0 * (ret / best - 1.0)
        if best
        else 0.0,
        "summary_json": str(summary_path),
    }


def write_results(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    if rows:
        with (output_dir / "final_scenario_confirm_results.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    lines = [
        "# Track B final scenario confirmation",
        "",
        f"Updated UTC: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "| scenario | architecture | horizon | ReT Excel | CVaR05 | risk-cond ReT | best static | delta ReT | delta % | cost |",
        "|---|---|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {scenario} | {architecture} | {horizon_weeks} | {ret:.9f} | "
            "{cvar:.9f} | {risk_cond:.9f} | `{best_static}` | {delta:+.9f} | "
            "{delta_pct:+.2f}% | {cost:.3f} |".format(
                scenario=row["scenario"],
                architecture=row["architecture"],
                horizon_weeks=int(row["horizon_weeks"]),
                ret=float(row["order_ret_excel_mean"]),
                cvar=float(row["order_ret_excel_cvar05_mean"]),
                risk_cond=float(row["order_ret_excel_risk_conditional_mean_mean"]),
                best_static=row["best_static_policy"],
                delta=float(row["delta_vs_best_static"]),
                delta_pct=float(row["relative_delta_vs_best_static_pct"]),
                cost=float(row["assembly_cost_index_mean"]),
            )
        )
    (output_dir / "final_scenario_confirm_results.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scenarios = scenario_list(args)
    architectures = parse_csv(args.architectures, str)
    cells = [(scenario, arch) for scenario in scenarios for arch in architectures]
    if args.cell_limit is not None:
        cells = cells[: int(args.cell_limit)]

    rows: list[dict[str, Any]] = []
    for scenario, architecture in cells:
        cell = f"{scenario.label}_{architecture}_h{args.max_steps}"
        cell_dir = args.output_dir / cell
        cmd, summary_path = build_command(
            architecture=architecture, scenario=scenario, cell_dir=cell_dir, args=args
        )
        print(f"\n=== final confirm cell: {cell} ===", flush=True)
        print(" ".join(cmd), flush=True)
        if args.dry_run:
            continue
        log_path = args.output_dir / f"{cell}.run.log"
        with log_path.open("w", encoding="utf-8") as log:
            subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
        rows.append(
            result_row(
                scenario=scenario,
                architecture=architecture,
                summary_path=summary_path,
                args=args,
            )
        )
        write_results(args.output_dir, rows)


if __name__ == "__main__":
    main()
