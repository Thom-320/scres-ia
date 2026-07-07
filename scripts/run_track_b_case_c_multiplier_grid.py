#!/usr/bin/env python3
"""Track B Case C: risk frequency/impact multiplier screen.

Case C starts after the environment/batch-size screen. It keeps the winner's
risk scenario and PPO batch size fixed, then tunes only two environment knobs:

- phi: risk_frequency_multiplier
- psi: risk_impact_multiplier

The goal is not to manufacture a bigger win. It is to find a defensible stress
level where PPO has useful learning headroom without moving away from Garrido's
risk families.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_track_b_env_hparam_grid import (  # noqa: E402
    ENV_SCENARIOS,
    EnvScenario,
    build_command,
)


def parse_csv_numbers(raw: str, cast: type = float) -> list[Any]:
    return [cast(item.strip()) for item in raw.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--parent-grid",
        type=Path,
        default=None,
        help="Optional grid_results.csv from Stage A/B. If present, choose the best row.",
    )
    parser.add_argument(
        "--wait-parent",
        action="store_true",
        help="Wait until --parent-grid has at least --expected-parent-rows rows.",
    )
    parser.add_argument("--expected-parent-rows", type=int, default=9)
    parser.add_argument("--parent-poll-seconds", type=int, default=120)
    parser.add_argument(
        "--allowed-parent-scenarios",
        default="garrido_all_current,garrido_downstream_cherry",
        help=(
            "Comma-separated parent scenarios eligible for Case C. Default excludes "
            "adaptive_v2_current because phi/psi are meant for Garrido-native risks."
        ),
    )
    parser.add_argument(
        "--scenario",
        choices=sorted(ENV_SCENARIOS),
        default=None,
        help="Manual scenario override. If omitted, use --parent-grid winner.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--phis", default="0.75,1.0,1.25,1.5")
    parser.add_argument("--psis", default="0.75,1.0,1.25,1.5")
    parser.add_argument("--obs-config", default="v7_no_forecast")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--train-timesteps", type=int, default=30_000)
    parser.add_argument("--eval-episodes", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=104)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--norm-reward", action="store_true")
    parser.add_argument("--clip-reward", type=float, default=10.0)
    parser.add_argument("--reward-mode", default="control_v1")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def wait_for_parent(path: Path, *, expected_rows: int, poll_seconds: int) -> list[dict[str, str]]:
    while True:
        rows = read_rows(path)
        if len(rows) >= expected_rows:
            return rows
        print(
            f"[wait] parent grid has {len(rows)}/{expected_rows} rows: {path}",
            flush=True,
        )
        time.sleep(max(5, int(poll_seconds)))


def select_parent_row(
    rows: list[dict[str, str]], *, allowed_scenarios: set[str]
) -> dict[str, str]:
    candidates = [
        row for row in rows if str(row.get("scenario", "")) in allowed_scenarios
    ]
    if not candidates:
        raise ValueError(
            f"No parent rows match allowed scenarios {sorted(allowed_scenarios)}"
        )
    return sorted(
        candidates,
        key=lambda row: (
            -float(row["order_ret_excel_mean"]),
            float(row.get("assembly_cost_index_mean", "inf")),
        ),
    )[0]


def slug_float(value: float) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def cell_slug(scenario: str, batch_size: int, phi: float, psi: float) -> str:
    return f"{scenario}_bs{batch_size}_phi{slug_float(phi)}_psi{slug_float(psi)}"


def read_ppo_and_comparison(summary_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    ppo = next(row for row in payload["policy_summary"] if row["policy"] == "ppo")
    comparison = payload["comparison_table"][0]
    return payload, ppo, comparison


def write_results(output_dir: Path, rows: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "case_c_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )
    if rows:
        with (output_dir / "case_c_results.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    ordered = sorted(
        rows,
        key=lambda row: (
            -float(row["order_ret_excel_mean"]),
            float(row["assembly_cost_index_mean"]),
        ),
    )
    lines = [
        "# Track B Case C Risk Multiplier Grid",
        "",
        f"Updated UTC: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "## Metadata",
        "",
        f"- parent scenario: `{metadata.get('parent_scenario')}`",
        f"- parent batch size: `{metadata.get('parent_batch_size')}`",
        f"- parent ReT Excel: `{metadata.get('parent_order_ret_excel_mean')}`",
        "",
        "| rank | cell | phi | psi | ReT Excel | cost | best static | delta vs static |",
        "|---:|---|---:|---:|---:|---:|---|---:|",
    ]
    for rank, row in enumerate(ordered, start=1):
        lines.append(
            "| {rank} | `{cell}` | {phi:.3g} | {psi:.3g} | {ret:.9f} | "
            "{cost:.3f} | `{best_static}` | {delta:+.9f} |".format(
                rank=rank,
                cell=row["cell"],
                phi=float(row["risk_frequency_multiplier"]),
                psi=float(row["risk_impact_multiplier"]),
                ret=float(row["order_ret_excel_mean"]),
                cost=float(row["assembly_cost_index_mean"]),
                best_static=row["best_static_policy"],
                delta=float(row["delta_vs_best_static_excel"]),
            )
        )
    (output_dir / "case_c_results.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    parent_row: dict[str, str] | None = None
    if args.parent_grid:
        parent_rows = (
            wait_for_parent(
                args.parent_grid,
                expected_rows=int(args.expected_parent_rows),
                poll_seconds=int(args.parent_poll_seconds),
            )
            if args.wait_parent
            else read_rows(args.parent_grid)
        )
        allowed = {
            item.strip()
            for item in str(args.allowed_parent_scenarios).split(",")
            if item.strip()
        }
        parent_row = select_parent_row(parent_rows, allowed_scenarios=allowed)

    scenario_name = args.scenario or (parent_row["scenario"] if parent_row else None)
    if scenario_name is None:
        raise SystemExit("Provide --scenario or --parent-grid.")
    batch_size = args.batch_size or (
        int(parent_row["batch_size"]) if parent_row else None
    )
    if batch_size is None:
        raise SystemExit("Provide --batch-size or --parent-grid.")

    base_scenario = ENV_SCENARIOS[scenario_name]
    seeds = parse_csv_numbers(args.seeds, int)
    phis = parse_csv_numbers(args.phis, float)
    psis = parse_csv_numbers(args.psis, float)
    rows: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "parent_grid": str(args.parent_grid) if args.parent_grid else None,
        "parent_row": parent_row,
        "parent_scenario": scenario_name,
        "parent_batch_size": batch_size,
        "parent_order_ret_excel_mean": (
            float(parent_row["order_ret_excel_mean"]) if parent_row else None
        ),
        "obs_config": args.obs_config,
        "seeds": seeds,
        "train_timesteps": int(args.train_timesteps),
        "eval_episodes": int(args.eval_episodes),
        "max_steps": int(args.max_steps),
        "note": (
            "Case C changes only risk_frequency_multiplier and "
            "risk_impact_multiplier around the selected Garrido-native scenario."
        ),
    }

    for phi in phis:
        for psi in psis:
            scenario = replace(
                base_scenario,
                risk_frequency_multiplier=float(phi),
                risk_impact_multiplier=float(psi),
            )
            slug = cell_slug(scenario_name, int(batch_size), float(phi), float(psi))
            cell_dir = output_dir / slug
            summary_path = cell_dir / args.obs_config / "summary.json"
            cmd = build_command(
                scenario=scenario,
                cell_dir=cell_dir,
                obs_config=args.obs_config,
                seeds=seeds,
                train_timesteps=int(args.train_timesteps),
                eval_episodes=int(args.eval_episodes),
                max_steps=int(args.max_steps),
                n_steps=int(args.n_steps),
                batch_size=int(batch_size),
                n_epochs=int(args.n_epochs),
                learning_rate=float(args.learning_rate),
                gamma=float(args.gamma),
                gae_lambda=float(args.gae_lambda),
                clip_range=float(args.clip_range),
                ent_coef=float(args.ent_coef),
                norm_reward=bool(args.norm_reward),
                clip_reward=float(args.clip_reward),
                reward_mode=str(args.reward_mode),
            )
            print(f"\n=== Case C cell: {slug} ===", flush=True)
            print(" ".join(cmd), flush=True)
            if not args.dry_run and not summary_path.exists():
                log_path = output_dir / f"{slug}.run.log"
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
                        f"Case C cell failed with exit {proc.returncode}: {slug}; see {log_path}"
                    )
            if args.dry_run:
                continue
            payload, ppo, comparison = read_ppo_and_comparison(summary_path)
            config = payload.get("config", {})
            row = {
                "cell": slug,
                "scenario": scenario_name,
                "risk_level": scenario.risk_level,
                "faithful": scenario.faithful,
                "enabled_risks": scenario.enabled_risks or "",
                "risk_frequency_multiplier": float(phi),
                "risk_impact_multiplier": float(psi),
                "obs_config": args.obs_config,
                "batch_size": int(config.get("batch_size", batch_size)),
                "train_timesteps": int(config.get("train_timesteps", args.train_timesteps)),
                "eval_episodes": int(config.get("eval_episodes", args.eval_episodes)),
                "order_ret_excel_mean": float(ppo["order_ret_excel_mean"]),
                "order_level_ret_mean_mean": float(ppo["order_level_ret_mean_mean"]),
                "fill_rate_mean": float(ppo["fill_rate_mean"]),
                "assembly_cost_index_mean": float(ppo["assembly_cost_index_mean"]),
                "pct_steps_S1_mean": float(ppo["pct_steps_S1_mean"]),
                "pct_steps_S2_mean": float(ppo["pct_steps_S2_mean"]),
                "pct_steps_S3_mean": float(ppo["pct_steps_S3_mean"]),
                "best_static_policy": comparison["best_static_policy"],
                "delta_vs_best_static_excel": float(
                    comparison["learned_order_level_ret_mean"]
                )
                - float(comparison["best_static_order_level_ret_mean"]),
                "summary_json": str(summary_path),
            }
            rows.append(row)
            write_results(output_dir, rows, metadata)


if __name__ == "__main__":
    main()
