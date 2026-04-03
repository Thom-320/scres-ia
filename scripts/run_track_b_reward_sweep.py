#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_track_b_benchmark import build_parser as build_benchmark_parser
from scripts.run_track_b_benchmark import run_launcher

DEFAULT_REWARD_MODES: tuple[str, ...] = (
    "ReT_thesis",
    "ReT_corrected",
    "ReT_seq_v1",
    "ReT_unified_v1",
    "ReT_garrido2024_train",
    "ReT_cd_v1",
    "control_v1",
)


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def default_sweep_root() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("outputs/track_b_benchmarks/reward_sweeps") / timestamp


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Launch a fair Track B reward sweep: same algorithm, same seeds, "
            "same backbone, different training rewards."
        )
    )
    parser.add_argument(
        "--algo",
        choices=["ppo", "recurrent_ppo"],
        default="ppo",
        help="Algorithm to sweep across reward modes.",
    )
    parser.add_argument(
        "--reward-modes",
        nargs="+",
        default=list(DEFAULT_REWARD_MODES),
        help="Reward modes to train and benchmark.",
    )
    parser.add_argument(
        "--train-timesteps",
        type=int,
        default=500_000,
        help="Training timesteps per reward mode.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=20,
        help="Evaluation episodes per reward-mode run.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[11, 22, 33, 44, 55],
        help="Training seeds for each reward-mode run.",
    )
    parser.add_argument(
        "--sweep-root",
        type=Path,
        default=None,
        help="Directory for manifest and benchmark subdirectories.",
    )
    parser.add_argument(
        "--label-prefix",
        default="track_b_reward_sweep",
        help="Prefix for generated benchmark labels.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip runs whose summary.json already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the planned labels and commands.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop the sweep as soon as one reward-mode run fails.",
    )
    parser.add_argument(
        "--heartbeat-interval-seconds",
        type=float,
        default=30.0,
        help="Heartbeat cadence forwarded to each benchmark run.",
    )
    return parser


def resolve_sweep_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    sweep_root = args.sweep_root or default_sweep_root()
    runs_root = sweep_root / "runs"
    manifest_path = sweep_root / "sweep_manifest.json"
    return sweep_root, runs_root, manifest_path


def build_label(
    *, label_prefix: str, algo: str, reward_mode: str, train_timesteps: int
) -> str:
    return (
        f"{label_prefix}_{algo}_{slugify(reward_mode)}_{int(train_timesteps/1000)}k"
    )


def build_run_args(
    *,
    label: str,
    runs_root: Path,
    args: argparse.Namespace,
    reward_mode: str,
) -> argparse.Namespace:
    benchmark_parser = build_benchmark_parser()
    cli_args = [
        "--label",
        label,
        "--output-root",
        str(runs_root),
        "--algo",
        str(args.algo),
        "--reward-mode",
        str(reward_mode),
        "--train-timesteps",
        str(args.train_timesteps),
        "--eval-episodes",
        str(args.eval_episodes),
        "--heartbeat-interval-seconds",
        str(args.heartbeat_interval_seconds),
        "--seeds",
        *[str(seed) for seed in args.seeds],
    ]
    return benchmark_parser.parse_args(cli_args)


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    sweep_root, runs_root, manifest_path = resolve_sweep_paths(args)
    sweep_root.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "algo": args.algo,
        "reward_modes": list(args.reward_modes),
        "train_timesteps": int(args.train_timesteps),
        "eval_episodes": int(args.eval_episodes),
        "seeds": [int(seed) for seed in args.seeds],
        "runs_root": str(runs_root.resolve()),
        "runs": [],
    }

    for reward_mode in args.reward_modes:
        label = build_label(
            label_prefix=str(args.label_prefix),
            algo=str(args.algo),
            reward_mode=str(reward_mode),
            train_timesteps=int(args.train_timesteps),
        )
        run_dir = runs_root / label
        summary_path = run_dir / "summary.json"
        run_record: dict[str, Any] = {
            "reward_mode": reward_mode,
            "label": label,
            "run_dir": str(run_dir.resolve()),
            "summary_json": str(summary_path.resolve()),
            "status": "planned",
        }

        if args.skip_existing and summary_path.exists():
            run_record["status"] = "skipped_existing"
            manifest["runs"].append(run_record)
            continue

        run_args = build_run_args(
            label=label, runs_root=runs_root, args=args, reward_mode=str(reward_mode)
        )
        run_record["launcher_args"] = {
            "label": run_args.label,
            "algo": run_args.algo,
            "reward_mode": run_args.reward_mode,
            "train_timesteps": run_args.train_timesteps,
            "eval_episodes": run_args.eval_episodes,
            "seeds": list(run_args.seeds),
            "output_root": str(run_args.output_root),
        }

        if args.dry_run:
            run_record["status"] = "dry_run"
            manifest["runs"].append(run_record)
            continue

        exit_code = run_launcher(run_args)
        run_record["exit_code"] = int(exit_code)
        run_record["status"] = "completed" if exit_code == 0 else "failed"
        manifest["runs"].append(run_record)
        write_manifest(manifest_path, manifest)
        if exit_code != 0 and args.stop_on_error:
            break

    write_manifest(manifest_path, manifest)
    print(f"Wrote reward sweep manifest to {manifest_path.resolve()}")
    for run in manifest["runs"]:
        print(
            f"{run['reward_mode']}: {run['status']} -> "
            f"{Path(run['run_dir']).name}"
        )


if __name__ == "__main__":
    main()
