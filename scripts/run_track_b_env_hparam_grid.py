#!/usr/bin/env python3
"""Track B environment and PPO hyperparameter screen.

This runner separates three questions that were previously mixed:

1. Which risk environment is most useful/defensible?
2. Does the historical manuscript batch size (256) still win under fixed-RNG?
3. Is a small entropy bonus worth carrying into confirmatory runs?

It intentionally calls the existing Track B runners as subprocesses so each cell
gets its own clean Python process, wrappers, and artifact bundle.
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
class EnvScenario:
    label: str
    description: str
    risk_level: str
    faithful: bool = False
    enabled_risks: str | None = None
    risk_frequency_multiplier: float = 1.0
    risk_impact_multiplier: float = 1.0


ENV_SCENARIOS: dict[str, EnvScenario] = {
    "garrido_all_current": EnvScenario(
        label="garrido_all_current",
        description=(
            "Garrido-native current risk level, thesis-faithful timing, all thesis "
            "risks active."
        ),
        risk_level="current",
        faithful=True,
    ),
    "garrido_downstream_cherry": EnvScenario(
        label="garrido_downstream_cherry",
        description=(
            "Garrido-native current risk level with only downstream/contingent risks "
            "R22/R23/R24 active; chosen because Track B controls Op10/Op12 dispatch."
        ),
        risk_level="current",
        faithful=True,
        enabled_risks="R22,R23,R24",
    ),
    "adaptive_v2_current": EnvScenario(
        label="adaptive_v2_current",
        description=(
            "Current Track B adaptive benchmark v2: Markov risk regimes plus "
            "downstream-risk uplift."
        ),
        risk_level="adaptive_benchmark_v2",
        faithful=False,
    ),
}


def parse_csv_numbers(raw: str, cast: type = float) -> list[Any]:
    return [cast(item.strip()) for item in raw.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--scenarios",
        default="garrido_all_current,garrido_downstream_cherry,adaptive_v2_current",
        help=f"Comma-separated scenario labels. Available: {','.join(ENV_SCENARIOS)}",
    )
    parser.add_argument("--obs-config", default="v7_no_forecast")
    parser.add_argument("--batch-sizes", default="64,128,256")
    parser.add_argument("--learning-rates", default="0.0003")
    parser.add_argument("--ent-coefs", default="0.0")
    parser.add_argument("--gammas", default="0.99")
    parser.add_argument("--gae-lambdas", default="0.95")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--train-timesteps", type=int, default=30_000)
    parser.add_argument("--eval-episodes", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=104)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--norm-reward-options", default="0")
    parser.add_argument("--clip-reward", type=float, default=10.0)
    parser.add_argument("--reward-mode", default="control_v1")
    parser.add_argument(
        "--cell-limit",
        type=int,
        default=None,
        help="Optional limit for smoke-testing the first N cells.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned cells without executing them.",
    )
    return parser


def read_ppo_row(summary_path: Path) -> dict[str, Any]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = payload["policy_summary"]
    return next(row for row in rows if row["policy"] == "ppo")


def cell_slug(
    scenario: str,
    *,
    batch_size: int,
    learning_rate: float,
    ent_coef: float,
    gamma: float,
    gae_lambda: float,
    norm_reward: bool,
) -> str:
    def clean(value: float) -> str:
        return str(value).replace(".", "p").replace("-", "m")

    return (
        f"{scenario}_bs{batch_size}_lr{clean(learning_rate)}_"
        f"ent{clean(ent_coef)}_g{clean(gamma)}_gae{clean(gae_lambda)}_"
        f"{'norm' if norm_reward else 'raw'}"
    )


def build_command(
    *,
    scenario: EnvScenario,
    cell_dir: Path,
    obs_config: str,
    seeds: list[int],
    train_timesteps: int,
    eval_episodes: int,
    max_steps: int,
    n_steps: int,
    batch_size: int,
    n_epochs: int,
    learning_rate: float,
    gamma: float,
    gae_lambda: float,
    clip_range: float,
    ent_coef: float,
    norm_reward: bool,
    clip_reward: float,
    reward_mode: str,
) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/run_track_b_observation_ablation.py",
        "--obs-configs",
        obs_config,
        "--output-dir",
        str(cell_dir),
        "--seeds",
        *(str(seed) for seed in seeds),
        "--train-timesteps",
        str(train_timesteps),
        "--eval-episodes",
        str(eval_episodes),
        "--max-steps",
        str(max_steps),
        "--reward-mode",
        reward_mode,
        "--risk-level",
        scenario.risk_level,
        "--risk-frequency-multiplier",
        str(scenario.risk_frequency_multiplier),
        "--risk-impact-multiplier",
        str(scenario.risk_impact_multiplier),
        "--learning-rate",
        str(learning_rate),
        "--n-steps",
        str(n_steps),
        "--batch-size",
        str(batch_size),
        "--n-epochs",
        str(n_epochs),
        "--gamma",
        str(gamma),
        "--gae-lambda",
        str(gae_lambda),
        "--clip-range",
        str(clip_range),
        "--ent-coef",
        str(ent_coef),
        "--clip-reward",
        str(clip_reward),
    ]
    if norm_reward:
        cmd.append("--norm-reward")
    if scenario.faithful:
        cmd.append("--faithful")
    if scenario.enabled_risks:
        cmd.extend(["--enabled-risks", scenario.enabled_risks])
    return cmd


def write_results(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        with (output_dir / "grid_results.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    ordered = sorted(
        rows,
        key=lambda row: (
            -float(row.get("order_ret_excel_mean", float("-inf"))),
            float(row.get("assembly_cost_index_mean", float("inf"))),
        ),
    )
    lines = [
        "# Track B Environment + Hyperparameter Grid",
        "",
        f"Updated UTC: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "| rank | cell | scenario | batch | lr | ent | gamma | gae | norm_reward | ReT Excel | cost | best static | delta vs static |",
        "|---:|---|---|---:|---:|---:|---:|---:|---|---:|---:|---|---:|",
    ]
    for rank, row in enumerate(ordered, start=1):
        lines.append(
            "| {rank} | `{cell}` | `{scenario}` | {batch_size} | {learning_rate:.4g} | "
            "{ent_coef:.4g} | {gamma:.4g} | {gae_lambda:.4g} | {norm_reward} | "
            "{ret:.9f} | {cost:.3f} | `{best_static}` | {delta:+.9f} |".format(
                rank=rank,
                cell=row["cell"],
                scenario=row["scenario"],
                batch_size=int(row["batch_size"]),
                learning_rate=float(row["learning_rate"]),
                ent_coef=float(row["ent_coef"]),
                gamma=float(row["gamma"]),
                gae_lambda=float(row["gae_lambda"]),
                norm_reward=str(row["norm_reward"]),
                ret=float(row["order_ret_excel_mean"]),
                cost=float(row["assembly_cost_index_mean"]),
                best_static=row.get("best_static_policy", ""),
                delta=float(row.get("delta_vs_best_static_excel", 0.0)),
            )
        )
    lines.append("")
    lines.append("## Scenario Definitions")
    lines.append("")
    for scenario in ENV_SCENARIOS.values():
        lines.append(f"- `{scenario.label}`: {scenario.description}")
    (output_dir / "grid_results.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    scenario_names = [name.strip() for name in args.scenarios.split(",") if name.strip()]
    unknown = [name for name in scenario_names if name not in ENV_SCENARIOS]
    if unknown:
        raise SystemExit(f"Unknown scenarios: {unknown}")
    batch_sizes = parse_csv_numbers(args.batch_sizes, int)
    learning_rates = parse_csv_numbers(args.learning_rates, float)
    ent_coefs = parse_csv_numbers(args.ent_coefs, float)
    gammas = parse_csv_numbers(args.gammas, float)
    gae_lambdas = parse_csv_numbers(args.gae_lambdas, float)
    norm_reward_options = [bool(int(v)) for v in parse_csv_numbers(args.norm_reward_options, int)]
    seeds = parse_csv_numbers(args.seeds, int)

    results: list[dict[str, Any]] = []
    cell_index = 0
    for scenario_name in scenario_names:
        scenario = ENV_SCENARIOS[scenario_name]
        for batch_size in batch_sizes:
            for learning_rate in learning_rates:
                for ent_coef in ent_coefs:
                    for gamma in gammas:
                        for gae_lambda in gae_lambdas:
                            for norm_reward in norm_reward_options:
                                cell_index += 1
                                if args.cell_limit is not None and cell_index > args.cell_limit:
                                    write_results(output_dir, results)
                                    return
                                slug = cell_slug(
                                    scenario_name,
                                    batch_size=batch_size,
                                    learning_rate=learning_rate,
                                    ent_coef=ent_coef,
                                    gamma=gamma,
                                    gae_lambda=gae_lambda,
                                    norm_reward=norm_reward,
                                )
                                cell_dir = output_dir / slug
                                summary_path = cell_dir / args.obs_config / "summary.json"
                                cmd = build_command(
                                    scenario=scenario,
                                    cell_dir=cell_dir,
                                    obs_config=args.obs_config,
                                    seeds=seeds,
                                    train_timesteps=args.train_timesteps,
                                    eval_episodes=args.eval_episodes,
                                    max_steps=args.max_steps,
                                    n_steps=args.n_steps,
                                    batch_size=batch_size,
                                    n_epochs=args.n_epochs,
                                    learning_rate=learning_rate,
                                    gamma=gamma,
                                    gae_lambda=gae_lambda,
                                    clip_range=args.clip_range,
                                    ent_coef=ent_coef,
                                    norm_reward=norm_reward,
                                    clip_reward=args.clip_reward,
                                    reward_mode=args.reward_mode,
                                )
                                print(f"\n=== cell {cell_index}: {slug} ===", flush=True)
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
                                            f"Cell failed with exit {proc.returncode}: {slug}; see {log_path}"
                                        )
                                if args.dry_run:
                                    continue
                                ppo = read_ppo_row(summary_path)
                                payload = json.loads(summary_path.read_text(encoding="utf-8"))
                                comparison = payload["comparison_table"][0]
                                row = {
                                    "cell": slug,
                                    "scenario": scenario_name,
                                    "scenario_description": scenario.description,
                                    "obs_config": args.obs_config,
                                    "risk_level": scenario.risk_level,
                                    "faithful": scenario.faithful,
                                    "enabled_risks": scenario.enabled_risks or "",
                                    "batch_size": batch_size,
                                    "learning_rate": learning_rate,
                                    "ent_coef": ent_coef,
                                    "gamma": gamma,
                                    "gae_lambda": gae_lambda,
                                    "norm_reward": norm_reward,
                                    "train_timesteps": args.train_timesteps,
                                    "eval_episodes": args.eval_episodes,
                                    "seeds": ",".join(str(seed) for seed in seeds),
                                    "order_ret_excel_mean": float(ppo["order_ret_excel_mean"]),
                                    "order_level_ret_mean_mean": float(
                                        ppo["order_level_ret_mean_mean"]
                                    ),
                                    "fill_rate_mean": float(ppo["fill_rate_mean"]),
                                    "assembly_cost_index_mean": float(
                                        ppo["assembly_cost_index_mean"]
                                    ),
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
                                results.append(row)
                                write_results(output_dir, results)


if __name__ == "__main__":
    main()
