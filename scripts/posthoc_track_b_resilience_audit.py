#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import sys
from typing import Any

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_track_b_smoke import (
    STATIC_POLICY_SPECS,
    aggregate_policy_metrics,
    aggregate_seed_metrics,
    build_comparison_rows,
    build_decision_summary,
    build_reward_contract,
    evaluate_static_policy,
    evaluate_trained_policy,
    make_monitored_training_env,
    render_markdown,
    save_csv,
)

REPO = Path(__file__).resolve().parent.parent
DEFAULT_RUN_DIR = REPO / "outputs/track_b_benchmarks/track_b_ret_seq_k020_500k_rerun1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Re-evaluate a completed Track B run to export resilience audit metrics "
            "without retraining the PPO models."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help="Completed Track B benchmark directory containing summary.json and model checkpoints.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for the posthoc resilience audit bundle. Defaults to <run-dir>/posthoc_resilience_audit.",
    )
    return parser


def load_summary(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))


def build_args_namespace(summary: dict[str, Any]) -> SimpleNamespace:
    config = summary["config"]
    return SimpleNamespace(
        reward_mode=config["reward_mode"],
        ret_seq_kappa=float(config.get("ret_seq_kappa", 0.20)),
        risk_level=config["risk_level"],
        step_size_hours=float(config["step_size_hours"]),
        max_steps=int(config["max_steps"]),
        eval_episodes=int(config["eval_episodes"]),
    )


def load_vec_normalize(args: SimpleNamespace, seed: int, path: Path) -> VecNormalize:
    vec_env = DummyVecEnv([make_monitored_training_env(args, seed)])
    vec_norm = VecNormalize.load(str(path), vec_env)
    vec_norm.training = False
    vec_norm.norm_reward = False
    return vec_norm


def evaluate_run(
    summary: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    args = build_args_namespace(summary)
    episode_rows: list[dict[str, Any]] = []
    trained_models = summary.get("trained_models", [])
    if not trained_models:
        raise ValueError("summary.json does not include trained_models")

    for model_meta in trained_models:
        seed = int(model_meta["seed"])
        model = PPO.load(str(model_meta["model_path"]), device="cpu")
        vec_norm = load_vec_normalize(
            args, seed, Path(model_meta["vec_normalize_path"])
        )

        for policy in STATIC_POLICY_SPECS:
            episode_rows.extend(evaluate_static_policy(policy, args=args, seed=seed))
        episode_rows.extend(
            evaluate_trained_policy(
                args=args, seed=seed, model=model, vec_norm=vec_norm
            )
        )
        vec_norm.close()

    seed_rows = aggregate_seed_metrics(episode_rows)
    return episode_rows, seed_rows


def build_summary(
    *,
    source_summary: dict[str, Any],
    output_dir: Path,
    episode_rows: list[dict[str, Any]],
    seed_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    args = build_args_namespace(source_summary)
    policy_rows = aggregate_policy_metrics(seed_rows)
    decision = build_decision_summary(policy_rows)
    comparison_rows = build_comparison_rows(policy_rows, args=args)
    reward_contract = build_reward_contract(str(args.reward_mode))

    episode_csv = output_dir / "episode_metrics.csv"
    seed_csv = output_dir / "seed_metrics.csv"
    policy_csv = output_dir / "policy_summary.csv"
    comparison_csv = output_dir / "comparison_table.csv"
    summary_json = output_dir / "summary.json"
    summary_md = output_dir / "summary.md"

    save_csv(episode_csv, episode_rows)
    save_csv(seed_csv, seed_rows)
    save_csv(policy_csv, policy_rows)
    save_csv(comparison_csv, comparison_rows)

    summary = {
        "artifact_type": "track_b_posthoc_resilience_audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_run_dir": str(Path(source_summary["artifacts"]["summary_json"]).parent),
        "source_summary_json": source_summary["artifacts"]["summary_json"],
        "config": dict(source_summary["config"]),
        "backbone": dict(source_summary["backbone"]),
        "env_spec": dict(source_summary["env_spec"]),
        "metric_contract": dict(source_summary["metric_contract"]),
        "reward_contract": reward_contract,
        "decision": decision,
        "policies": [policy.label for policy in STATIC_POLICY_SPECS] + ["ppo"],
        "seed_metrics": seed_rows,
        "policy_summary": policy_rows,
        "comparison_table": comparison_rows,
        "artifacts": {
            "episode_metrics_csv": str(episode_csv.resolve()),
            "seed_metrics_csv": str(seed_csv.resolve()),
            "policy_summary_csv": str(policy_csv.resolve()),
            "comparison_table_csv": str(comparison_csv.resolve()),
            "summary_json": str(summary_json.resolve()),
            "summary_md": str(summary_md.resolve()),
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary_md.write_text(render_markdown(summary), encoding="utf-8")
    return summary


def main() -> None:
    args = build_parser().parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (run_dir / "posthoc_resilience_audit").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    source_summary = load_summary(run_dir)
    episode_rows, seed_rows = evaluate_run(source_summary)
    summary = build_summary(
        source_summary=source_summary,
        output_dir=output_dir,
        episode_rows=episode_rows,
        seed_rows=seed_rows,
    )
    print(f"Wrote Track B posthoc resilience audit to {output_dir}")
    print(
        "Policies: "
        + ", ".join(
            f"{row['policy']} fill={float(row['fill_rate_mean']):.4f}"
            for row in summary["policy_summary"]
        )
    )


if __name__ == "__main__":
    main()
