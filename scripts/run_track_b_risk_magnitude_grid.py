#!/usr/bin/env python3
"""Case C: risk-magnitude fine-tuning on top of Codex's env/hyperparameter grid.

Reuses ``scripts/run_track_b_env_hparam_grid.py`` (Codex's runner, which
already covers Case A `garrido_all_current` and Case B
`garrido_downstream_cherry` crossed with batch size) unmodified -- this adds
the one axis that grid does not sweep: ``risk_frequency_multiplier`` and
``risk_impact_multiplier`` on top of a chosen base scenario, to search for
the risk-magnitude configuration with the best PPO-vs-static/heuristic gap.

Does not touch ``run_track_b_env_hparam_grid.py`` on disk (it may still be
mid-run) -- imports its dataclass/helpers and builds new scenario variants
in memory.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_track_b_env_hparam_grid import (  # noqa: E402
    ENV_SCENARIOS,
    build_command,
    read_ppo_row,
    write_results,
)
import json
import subprocess


def parse_csv_floats(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--base-scenarios",
        default="garrido_all_current,garrido_downstream_cherry",
        help=f"Comma-separated base scenarios from {list(ENV_SCENARIOS)}.",
    )
    parser.add_argument("--risk-frequency-multipliers", default="0.75,1.0,1.5,2.0")
    parser.add_argument("--risk-impact-multipliers", default="1.0,1.5")
    parser.add_argument("--obs-config", default="v7_no_forecast")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--norm-reward", action="store_true")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--train-timesteps", type=int, default=30_000)
    parser.add_argument("--eval-episodes", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=104)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--clip-reward", type=float, default=10.0)
    parser.add_argument("--reward-mode", default="control_v1")
    return parser


def cell_slug(scenario_name: str, freq: float, impact: float) -> str:
    def clean(value: float) -> str:
        return str(value).replace(".", "p")

    return f"{scenario_name}_freq{clean(freq)}_impact{clean(impact)}"


def main() -> None:
    args = build_parser().parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    base_names = [name.strip() for name in args.base_scenarios.split(",") if name.strip()]
    unknown = [name for name in base_names if name not in ENV_SCENARIOS]
    if unknown:
        raise SystemExit(f"Unknown base scenarios: {unknown}")
    freqs = parse_csv_floats(args.risk_frequency_multipliers)
    impacts = parse_csv_floats(args.risk_impact_multipliers)
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    results: list[dict[str, Any]] = []
    for base_name in base_names:
        base = ENV_SCENARIOS[base_name]
        for freq in freqs:
            for impact in impacts:
                if freq == 1.0 and impact == 1.0:
                    continue  # already covered by the base env/hparam grid
                scenario = dataclasses.replace(
                    base,
                    label=f"{base_name}_freq{freq}_impact{impact}",
                    risk_frequency_multiplier=freq,
                    risk_impact_multiplier=impact,
                )
                slug = cell_slug(base_name, freq, impact)
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
                    batch_size=args.batch_size,
                    n_epochs=args.n_epochs,
                    learning_rate=args.learning_rate,
                    gamma=args.gamma,
                    gae_lambda=args.gae_lambda,
                    clip_range=args.clip_range,
                    ent_coef=args.ent_coef,
                    norm_reward=args.norm_reward,
                    clip_reward=args.clip_reward,
                    reward_mode=args.reward_mode,
                )
                print(f"\n=== {slug} ===", flush=True)
                print(" ".join(cmd), flush=True)
                if not summary_path.exists():
                    log_path = output_dir / f"{slug}.run.log"
                    with log_path.open("w", encoding="utf-8") as log:
                        proc = subprocess.run(
                            cmd, cwd=Path.cwd(), stdout=log, stderr=subprocess.STDOUT,
                            text=True, check=False,
                        )
                    if proc.returncode != 0:
                        raise SystemExit(f"Cell failed: {slug}; see {log_path}")
                ppo = read_ppo_row(summary_path)
                payload = json.loads(summary_path.read_text(encoding="utf-8"))
                comparison = payload["comparison_table"][0]
                results.append(
                    {
                        "cell": slug,
                        "scenario": base_name,
                        "risk_frequency_multiplier": freq,
                        "risk_impact_multiplier": impact,
                        "batch_size": args.batch_size,
                        "learning_rate": args.learning_rate,
                        "ent_coef": args.ent_coef,
                        "gamma": args.gamma,
                        "gae_lambda": args.gae_lambda,
                        "norm_reward": args.norm_reward,
                        "order_ret_excel_mean": float(ppo["order_ret_excel_mean"]),
                        "assembly_cost_index_mean": float(ppo["assembly_cost_index_mean"]),
                        "best_static_policy": comparison["best_static_policy"],
                        "delta_vs_best_static_excel": float(
                            comparison["learned_order_level_ret_mean"]
                        )
                        - float(comparison["best_static_order_level_ret_mean"]),
                        "summary_json": str(summary_path),
                    }
                )
                write_results(output_dir, results)


if __name__ == "__main__":
    main()
