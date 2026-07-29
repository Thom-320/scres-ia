"""Mini grid for Track B PPO discount horizon and reward normalization.

This is a diagnostic runner, not a confirmatory paper lane.  It calls the
existing Track B observation-ablation runner so the no-forecast wrapper can be
used without duplicating environment code.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def parse_floats(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def parse_ints(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a small Track B gamma/GAE/reward-normalization grid."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/experiments/track_b_gamma_rewardnorm_grid_2026-07-05"),
    )
    parser.add_argument("--obs-config", default="v7_no_forecast")
    parser.add_argument("--gammas", default="0.99,0.995,0.999")
    parser.add_argument("--gae-lambdas", default="0.95,0.98")
    parser.add_argument(
        "--norm-reward-options",
        default="0,1",
        help="Comma-separated 0/1 flags. 1 enables VecNormalize norm_reward.",
    )
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--train-timesteps", type=int, default=20_000)
    parser.add_argument("--eval-episodes", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=104)
    parser.add_argument("--reward-mode", default="control_v1")
    parser.add_argument("--risk-level", default="adaptive_benchmark_v2")
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--clip-reward", type=float, default=10.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def ppo_row(summary_path: Path) -> dict[str, Any]:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    for row in data.get("policy_summary", []):
        if row.get("policy") in {"ppo", "recurrent_ppo"}:
            return row
    raise RuntimeError(f"No PPO row in {summary_path}")


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    gammas = parse_floats(args.gammas)
    gae_lambdas = parse_floats(args.gae_lambdas)
    norm_options = [bool(int(x)) for x in parse_ints(args.norm_reward_options)]
    seeds = parse_ints(args.seeds)

    rows: list[dict[str, Any]] = []
    for gamma in gammas:
        for gae_lambda in gae_lambdas:
            for norm_reward in norm_options:
                label = (
                    f"g{gamma:.3f}".replace(".", "p")
                    + f"_gae{gae_lambda:.2f}".replace(".", "p")
                    + ("_norm" if norm_reward else "_raw")
                )
                cell_dir = args.output_dir / label
                cmd = [
                    sys.executable,
                    "scripts/run_track_b_observation_ablation.py",
                    "--obs-configs",
                    args.obs_config,
                    "--output-dir",
                    str(cell_dir),
                    "--seeds",
                    *[str(seed) for seed in seeds],
                    "--train-timesteps",
                    str(args.train_timesteps),
                    "--eval-episodes",
                    str(args.eval_episodes),
                    "--max-steps",
                    str(args.max_steps),
                    "--reward-mode",
                    args.reward_mode,
                    "--risk-level",
                    args.risk_level,
                    "--learning-rate",
                    str(args.learning_rate),
                    "--n-steps",
                    str(args.n_steps),
                    "--batch-size",
                    str(args.batch_size),
                    "--n-epochs",
                    str(args.n_epochs),
                    "--gamma",
                    str(gamma),
                    "--gae-lambda",
                    str(gae_lambda),
                    "--clip-range",
                    str(args.clip_range),
                    "--clip-reward",
                    str(args.clip_reward),
                ]
                if norm_reward:
                    cmd.append("--norm-reward")

                log_path = args.output_dir / f"{label}.run.log"
                print(f"=== {label} ===", flush=True)
                print(" ".join(cmd), flush=True)
                if args.dry_run:
                    continue
                with log_path.open("w", encoding="utf-8") as log:
                    proc = subprocess.run(
                        cmd,
                        cwd=Path.cwd(),
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        text=True,
                        check=False,
                    )
                summary_path = cell_dir / args.obs_config / "summary.json"
                row: dict[str, Any] = {
                    "label": label,
                    "gamma": gamma,
                    "gae_lambda": gae_lambda,
                    "norm_reward": norm_reward,
                    "returncode": proc.returncode,
                    "summary_json": str(summary_path),
                    "run_log": str(log_path),
                }
                if proc.returncode == 0 and summary_path.exists():
                    ppo = ppo_row(summary_path)
                    for key in [
                        "order_ret_excel_mean",
                        "order_level_ret_mean_mean",
                        "order_ret_excel_cvar05_mean",
                        "order_ret_excel_rolling_4w_min_mean",
                        "order_ttr_p95_mean",
                        "assembly_cost_index_mean",
                        "pct_steps_S1_mean",
                        "pct_steps_S2_mean",
                        "pct_steps_S3_mean",
                    ]:
                        row[key] = ppo.get(key)
                rows.append(row)
                write_outputs(args.output_dir, rows, args)


def write_outputs(out_dir: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    csv_path = out_dir / "grid_results.csv"
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    completed = [row for row in rows if row.get("returncode") == 0 and row.get("order_ret_excel_mean") is not None]
    completed.sort(key=lambda row: float(row.get("order_ret_excel_mean") or -1.0), reverse=True)
    md = [
        "# Track B gamma / GAE / reward-normalization mini-grid",
        "",
        "Diagnostic run. Primary metric is Garrido/Excel `order_ret_excel_mean`.",
        "",
        "## Protocol",
        "",
        f"- observation config: `{args.obs_config}`",
        f"- reward mode: `{args.reward_mode}`",
        f"- risk level: `{args.risk_level}`",
        f"- train timesteps: `{args.train_timesteps}`",
        f"- eval episodes: `{args.eval_episodes}`",
        f"- seeds: `{args.seeds}`",
        f"- n_steps: `{args.n_steps}`",
        f"- batch_size: `{args.batch_size}`",
        "",
        "## Current ranking",
        "",
        "| rank | label | ReT Excel | cost | CVaR05 | 4w min | S1/S2/S3 |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for idx, row in enumerate(completed, start=1):
        s_mix = (
            f"{float(row.get('pct_steps_S1_mean') or 0):.1f}/"
            f"{float(row.get('pct_steps_S2_mean') or 0):.1f}/"
            f"{float(row.get('pct_steps_S3_mean') or 0):.1f}"
        )
        md.append(
            "| "
            f"{idx} | `{row['label']}` | "
            f"{float(row.get('order_ret_excel_mean') or 0):.9f} | "
            f"{float(row.get('assembly_cost_index_mean') or 0):.3f} | "
            f"{float(row.get('order_ret_excel_cvar05_mean') or 0):.9f} | "
            f"{float(row.get('order_ret_excel_rolling_4w_min_mean') or 0):.9f} | "
            f"{s_mix} |"
        )
    (out_dir / "grid_results.md").write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
