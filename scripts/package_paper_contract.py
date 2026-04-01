#!/usr/bin/env python3
"""Run the frozen MFSC benchmark contract and package DKANA-ready artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.generate_proof_of_learning_artifacts import (
    generate_proof_of_learning_artifacts,
)
from supply_chain.config import BENCHMARK_EPISODE_HORIZON_HOURS

EXPORT_POLICIES: tuple[str, ...] = (
    "random",
    "garrido_cf_s1",
    "garrido_cf_s2",
    "garrido_cf_s3",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package the frozen MFSC paper contract for downstream DKANA work."
    )
    parser.add_argument("--label", default="paper_contract_track_a_control_v1")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/paper_contract"),
    )
    parser.add_argument(
        "--reward-mode",
        default="control_v1",
        choices=["ReT_unified_v1", "ReT_seq_v1", "control_v1"],
    )
    parser.add_argument(
        "--observation-version",
        default="v4",
        choices=["v1", "v2", "v3", "v4", "v5"],
    )
    parser.add_argument("--frame-stack", type=int, default=1)
    parser.add_argument(
        "--algo", default="ppo", choices=["ppo", "sac", "recurrent_ppo"]
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    parser.add_argument("--train-timesteps", type=int, default=100000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--step-size-hours", type=float, default=168.0)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help=(
            "Episode length in control steps. Defaults to the historical "
            "260x168h physical horizon rescaled to the requested cadence."
        ),
    )
    parser.add_argument(
        "--risk-level",
        default="increased",
        choices=["current", "increased", "severe"],
    )
    parser.add_argument(
        "--eval-risk-levels",
        nargs="+",
        default=["current", "increased", "severe"],
        choices=["current", "increased", "severe"],
    )
    parser.add_argument("--stochastic-pt", action="store_true")
    parser.add_argument("--window-size", type=int, default=12)
    parser.add_argument(
        "--ret-unified-calibration",
        type=Path,
        default=Path("supply_chain/data/ret_unified_v1_calibration.json"),
    )
    parser.add_argument(
        "--existing-benchmark-dir",
        type=Path,
        default=None,
        help="Reuse an existing benchmark dir instead of re-running the benchmark.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_episode_max_steps(
    step_size_hours: float,
    explicit_max_steps: int | None,
) -> int:
    """Preserve the historical physical horizon when cadence changes."""
    if explicit_max_steps is not None:
        return int(explicit_max_steps)
    if step_size_hours <= 0:
        raise ValueError("step_size_hours must be > 0")
    return max(1, int(round(BENCHMARK_EPISODE_HORIZON_HOURS / step_size_hours)))


def run_command(command: list[str], *, cwd: Path) -> None:
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        check=False,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}: {' '.join(command)}"
        )


def resolve_benchmark_dir(args: argparse.Namespace) -> Path:
    if args.existing_benchmark_dir is not None:
        return args.existing_benchmark_dir.resolve()
    return (args.output_root / args.label / "benchmark").resolve()


def invoke_benchmark(args: argparse.Namespace, benchmark_dir: Path, cwd: Path) -> None:
    command = [
        sys.executable,
        "scripts/benchmark_control_reward.py",
        "--algo",
        args.algo,
        "--reward-mode",
        args.reward_mode,
        "--observation-version",
        args.observation_version,
        "--frame-stack",
        str(args.frame_stack),
        "--seeds",
        *[str(seed) for seed in args.seeds],
        "--train-timesteps",
        str(args.train_timesteps),
        "--eval-episodes",
        str(args.eval_episodes),
        "--step-size-hours",
        str(args.step_size_hours),
        "--max-steps",
        str(resolve_episode_max_steps(args.step_size_hours, args.max_steps)),
        "--risk-level",
        args.risk_level,
        "--eval-risk-levels",
        *args.eval_risk_levels,
        "--output-dir",
        str(benchmark_dir),
    ]
    if args.stochastic_pt:
        command.append("--stochastic-pt")
    if (
        args.reward_mode == "ReT_unified_v1"
        and args.ret_unified_calibration is not None
    ):
        command.extend(["--ret-unified-calibration", str(args.ret_unified_calibration)])
    run_command(command, cwd=cwd)


def select_learned_model(summary: dict[str, Any]) -> dict[str, Any] | None:
    trained_models = summary.get("trained_models", [])
    return trained_models[0] if trained_models else None


def export_policy_bundle(
    *,
    policy: str,
    export_dir: Path,
    benchmark_summary: dict[str, Any],
    benchmark_dir: Path,
    args: argparse.Namespace,
    cwd: Path,
) -> None:
    command = [
        sys.executable,
        "scripts/export_trajectories_for_david.py",
        "--episodes",
        str(args.eval_episodes),
        "--seed-start",
        str(args.seeds[0]),
        "--risk-level",
        args.risk_level,
        "--reward-mode",
        args.reward_mode,
        "--observation-version",
        args.observation_version,
        "--policy",
        policy,
        "--frame-stack",
        str(args.frame_stack),
        "--step-size-hours",
        str(args.step_size_hours),
        "--max-steps",
        str(resolve_episode_max_steps(args.step_size_hours, args.max_steps)),
        "--output-dir",
        str(export_dir),
    ]
    if args.stochastic_pt:
        command.append("--stochastic-pt")
    if (
        args.reward_mode == "ReT_unified_v1"
        and args.ret_unified_calibration is not None
    ):
        command.extend(["--ret-unified-calibration", str(args.ret_unified_calibration)])

    learned_model = select_learned_model(benchmark_summary)
    if policy in ("ppo", "sac", "recurrent_ppo"):
        if learned_model is None or not learned_model.get("model_path"):
            raise ValueError(
                "No trained model found in benchmark summary for learned export."
            )
        command.extend(["--model-path", str(learned_model["model_path"])])

    run_command(command, cwd=cwd)

    dkana_dir = export_dir.parent / f"{export_dir.name}_dkana"
    run_command(
        [
            sys.executable,
            "scripts/build_dkana_dataset.py",
            "--input-dir",
            str(export_dir),
            "--output-dir",
            str(dkana_dir),
            "--window-size",
            str(args.window_size),
        ],
        cwd=cwd,
    )


def main() -> None:
    args = parse_args()
    cwd = Path(__file__).resolve().parent.parent
    package_dir = (args.output_root / args.label).resolve()
    package_dir.mkdir(parents=True, exist_ok=True)

    benchmark_dir = resolve_benchmark_dir(args)
    if args.existing_benchmark_dir is None:
        benchmark_dir.mkdir(parents=True, exist_ok=True)
        invoke_benchmark(args, benchmark_dir, cwd)

    summary = read_json(benchmark_dir / "summary.json")
    proof_manifest: dict[str, Any] | None
    try:
        proof_manifest = generate_proof_of_learning_artifacts(benchmark_dir)
    except ValueError:
        proof_manifest = None

    export_root = package_dir / "trajectory_exports"
    export_root.mkdir(parents=True, exist_ok=True)
    comparator_set = list(EXPORT_POLICIES)
    learned_model = select_learned_model(summary)
    if learned_model is not None:
        comparator_set.append(str(summary["config"]["algo"]))

    export_manifests: dict[str, Any] = {}
    for policy in comparator_set:
        export_dir = export_root / policy
        export_policy_bundle(
            policy=policy,
            export_dir=export_dir,
            benchmark_summary=summary,
            benchmark_dir=benchmark_dir,
            args=args,
            cwd=cwd,
        )
        export_manifests[policy] = {
            "export_dir": str(export_dir.resolve()),
            "dkana_dir": str((export_root / f"{policy}_dkana").resolve()),
        }

    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=str(cwd),
        text=True,
    ).strip()
    comparison_row = summary.get("comparison_table", [{}])
    acceptance = comparison_row[0] if comparison_row else {}
    calibration_payload = (
        read_json(args.ret_unified_calibration)
        if args.reward_mode == "ReT_unified_v1"
        and args.ret_unified_calibration is not None
        and args.ret_unified_calibration.exists()
        else None
    )
    manifest = {
        "artifact_type": "paper_contract_package",
        "git_commit": git_commit,
        "reward_mode": args.reward_mode,
        "observation_version": args.observation_version,
        "frame_stack": args.frame_stack,
        "seeds": args.seeds,
        "selected_hyperparameters": calibration_payload,
        "comparator_set": comparator_set,
        "acceptance_results": {
            "ppo_beats_garrido_cf_s2": acceptance.get("ppo_beats_garrido_cf_s2"),
            "ppo_beats_best_garrido": acceptance.get("ppo_beats_best_garrido"),
            "ppo_meets_garrido_cf_s2_fill_rate": acceptance.get(
                "ppo_meets_garrido_cf_s2_fill_rate"
            ),
            "ppo_meets_unified_acceptance_gate": acceptance.get(
                "ppo_meets_unified_acceptance_gate"
            ),
            "shift_collapse_flag": acceptance.get("shift_collapse_flag"),
            "ret_unified_total": acceptance.get("ppo_ret_unified_total_mean"),
            "ret_garrido2024_sigmoid_total": acceptance.get(
                "ppo_ret_garrido2024_sigmoid_total_mean"
            ),
        },
        "benchmark_dir": str(benchmark_dir.resolve()),
        "proof_of_learning_dir": (
            proof_manifest["output_dir"] if proof_manifest is not None else None
        ),
        "proof_of_learning_status": (
            "generated"
            if proof_manifest is not None
            else "skipped_insufficient_cross_eval"
        ),
        "trajectory_exports": export_manifests,
    }
    manifest_path = package_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Packaged frozen paper contract at {package_dir}")
    print(f"  manifest: {manifest_path}")


if __name__ == "__main__":
    main()
