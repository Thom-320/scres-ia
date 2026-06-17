#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

DEFAULT_OUTPUT_ROOT = Path("outputs/benchmarks/track_a_exhaustion_sweep")

REWARD_PROFILES: dict[str, list[str]] = {
    "ret_thesis": ["--reward-mode", "ReT_thesis"],
    "ret_ladder": ["--reward-mode", "ReT_ladder_v1"],
    "ret_ladder_steep": [
        "--reward-mode",
        "ReT_ladder_v1",
        "--ret-ladder-w-sc",
        "0.80",
        "--ret-ladder-w-rc",
        "0.20",
        "--ret-ladder-w-ef",
        "0.00",
        "--ret-ladder-gate-beta",
        "20.0",
    ],
    "ret_seq": ["--reward-mode", "ReT_seq_v1"],
    "ret_cd_sigmoid": ["--reward-mode", "ReT_cd_sigmoid"],
    "control": ["--reward-mode", "control_v1"],
    "control_steep": [
        "--reward-mode",
        "control_v1",
        "--w-bo",
        "5.0",
        "--w-cost",
        "0.03",
    ],
    "control_pbrs_steep": [
        "--reward-mode",
        "control_v1_pbrs",
        "--w-bo",
        "5.0",
        "--w-cost",
        "0.03",
    ],
}

PT_PROFILES: dict[str, list[str]] = {
    "det_pt": [],
    "stoch_pt_hist": ["--stochastic-pt", "--stochastic-pt-spread", "1.0"],
    "stoch_pt_mean": [
        "--stochastic-pt",
        "--stochastic-pt-spread",
        "1.0",
        "--stochastic-pt-mean-preserving",
    ],
    "stoch_pt_mean_hi": [
        "--stochastic-pt",
        "--stochastic-pt-spread",
        "2.0",
        "--stochastic-pt-mean-preserving",
    ],
}


def utc_now_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def slug(value: str) -> str:
    return (
        str(value)
        .replace("/", "_")
        .replace(" ", "_")
        .replace(".", "p")
        .replace("-", "_")
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a reproducible Track A exhaustion sweep over action surfaces, "
            "reward profiles, risk levels, and stochastic-PT profiles."
        )
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--label-prefix", default="track_a_exhaust")
    parser.add_argument("--train-timesteps", type=int, default=30_000)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument(
        "--eval-seed-base",
        type=int,
        default=None,
        help=(
            "Optional held-out evaluation seed base forwarded to "
            "run_thesis_decision_ppo_smoke.py. Defaults to --seed."
        ),
    )
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument(
        "--n-envs",
        type=int,
        default=1,
        help=(
            "Parallel training envs forwarded to the PPO smoke runner. "
            "Use >1 for lower-variance PPO gradients in serious convergence runs."
        ),
    )
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument(
        "--algos",
        nargs="+",
        default=["ppo_mlp"],
        choices=["ppo_mlp", "recurrent_ppo", "dmlpa_ppo"],
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--history-window", type=int, default=30)
    parser.add_argument(
        "--action-space-modes",
        nargs="+",
        default=["thesis_factorized", "continuous_it_s"],
        choices=["thesis_factorized", "continuous_it_s", "factorized", "onehot_18d"],
    )
    parser.add_argument(
        "--reward-profiles",
        nargs="+",
        default=["ret_ladder", "ret_ladder_steep", "control_steep", "ret_seq"],
        choices=sorted(REWARD_PROFILES),
    )
    parser.add_argument(
        "--risk-levels",
        nargs="+",
        default=["severe_extended"],
        choices=[
            "current",
            "increased",
            "severe",
            "severe_extended",
            "severe_training",
            "war_stress_v1",
            "adaptive_benchmark_v1",
            "adaptive_benchmark_v2",
        ],
    )
    parser.add_argument(
        "--pt-profiles",
        nargs="+",
        default=["det_pt", "stoch_pt_mean"],
        choices=sorted(PT_PROFILES),
    )
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--panel-cfis", default="31-90")
    parser.add_argument(
        "--use-cf-risk-profile",
        action="store_true",
        help=(
            "Train/evaluate over --panel-cfis using the selected risk level as "
            "a static-fidelity profile for each Cf row."
        ),
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument(
        "--norm-reward",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Forward VecNormalize reward normalization to the PPO smoke runner. "
            "This is a training-scale transform only; summaries must still be "
            "judged on external resilience metrics, not reward_total."
        ),
    )
    return parser


def build_label(
    *,
    prefix: str,
    algo: str,
    action_space_mode: str,
    reward_profile: str,
    risk_level: str,
    pt_profile: str,
    train_timesteps: int,
) -> str:
    return "_".join(
        [
            prefix,
            slug(algo),
            slug(action_space_mode),
            slug(reward_profile),
            slug(risk_level),
            slug(pt_profile),
            f"{int(train_timesteps / 1000)}k",
        ]
    )


def build_command(
    *,
    args: argparse.Namespace,
    run_root: Path,
    label: str,
    algo: str,
    action_space_mode: str,
    reward_profile: str,
    risk_level: str,
    pt_profile: str,
) -> list[str]:
    command = [
        sys.executable,
        "scripts/run_thesis_decision_ppo_smoke.py",
        "--label",
        label,
        "--output-root",
        str(run_root),
        "--train-timesteps",
        str(args.train_timesteps),
        "--eval-episodes",
        str(args.eval_episodes),
        "--seed",
        str(args.seed),
        "--algo",
        algo,
        "--device",
        args.device,
        "--history-window",
        str(args.history_window),
        "--risk-level",
        risk_level,
        "--risk-occurrence-mode",
        "thesis_periodic",
        "--raw-material-flow-mode",
        "kit_equivalent_order_up_to",
        "--raw-material-order-up-to-multiplier",
        "2.0",
        "--action-space-mode",
        action_space_mode,
        "--inventory-period-mode",
        "thesis_strict",
        "--max-steps",
        str(args.max_steps),
        "--n-envs",
        str(args.n_envs),
        "--include-static-grid",
        "--no-eval-ai-on-garrido-cfis",
        "--n-steps",
        str(args.n_steps),
        "--batch-size",
        str(args.batch_size),
        "--n-epochs",
        str(args.n_epochs),
        *REWARD_PROFILES[reward_profile],
        *PT_PROFILES[pt_profile],
    ]
    command.append("--norm-reward" if args.norm_reward else "--no-norm-reward")
    if args.eval_seed_base is not None:
        command.extend(["--eval-seed-base", str(args.eval_seed_base)])
    if args.use_cf_risk_profile:
        command.extend(
            [
                "--train-cfis",
                args.panel_cfis,
                "--garrido-cfis",
                args.panel_cfis,
                "--train-risk-profile",
                risk_level,
                "--eval-risk-profile",
                risk_level,
            ]
        )
    return command


def read_policy_summary(run_dir: Path, *, algo: str = "ppo_mlp") -> dict[str, Any]:
    path = run_dir / "policy_summary.csv"
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}

    def as_float(row: dict[str, str], key: str) -> float:
        return float(row.get(key) or 0.0)

    ppo_rows = [row for row in rows if row.get("policy") == algo]
    static_rows = [
        row for row in rows if str(row.get("policy", "")).startswith("static_grid_")
    ]
    if not ppo_rows or not static_rows:
        return {}
    ppo = ppo_rows[0]
    best_static = max(
        static_rows,
        key=lambda row: (
            as_float(row, "fill_rate_order_level_mean"),
            as_float(row, "order_level_ret_mean"),
        ),
    )
    ppo_fill = as_float(ppo, "fill_rate_order_level_mean")
    static_fill = as_float(best_static, "fill_rate_order_level_mean")
    ppo_ret = as_float(ppo, "order_level_ret_mean")
    static_ret = as_float(best_static, "order_level_ret_mean")
    ppo_ret_all = as_float(ppo, "ret_mean_all_orders_zero_unfulfilled_mean")
    static_ret_all = as_float(best_static, "ret_mean_all_orders_zero_unfulfilled_mean")
    ppo_flow_fill = as_float(ppo, "flow_fill_rate_mean")
    static_flow_fill = as_float(best_static, "flow_fill_rate_mean")
    ppo_stockout_week = as_float(ppo, "stockout_week_pct_mean")
    static_stockout_week = as_float(best_static, "stockout_week_pct_mean")
    return {
        "ppo_fill": ppo_fill,
        "best_static_policy": best_static.get("policy", ""),
        "best_static_fill": static_fill,
        "delta_fill": ppo_fill - static_fill,
        "ppo_ret": ppo_ret,
        "best_static_ret": static_ret,
        "delta_ret": ppo_ret - static_ret,
        "ppo_ret_all_orders": ppo_ret_all,
        "best_static_ret_all_orders": static_ret_all,
        "delta_ret_all_orders": ppo_ret_all - static_ret_all,
        "ppo_flow_fill": ppo_flow_fill,
        "best_static_flow_fill": static_flow_fill,
        "delta_flow_fill": ppo_flow_fill - static_flow_fill,
        "ppo_stockout_week_pct": ppo_stockout_week,
        "best_static_stockout_week_pct": static_stockout_week,
        "delta_stockout_week_pct": ppo_stockout_week - static_stockout_week,
        "ppo_re_fr_contribution": as_float(ppo, "re_fr_contribution_all_mean"),
        "best_static_re_fr_contribution": as_float(
            best_static, "re_fr_contribution_all_mean"
        ),
        "ppo_dynamic_ret_contribution": as_float(
            ppo, "dynamic_ret_contribution_all_mean"
        ),
        "best_static_dynamic_ret_contribution": as_float(
            best_static, "dynamic_ret_contribution_all_mean"
        ),
        "ppo_ret_p10_all": as_float(ppo, "ret_p10_all_mean"),
        "best_static_ret_p10_all": as_float(best_static, "ret_p10_all_mean"),
        "delta_ret_p10_all": as_float(ppo, "ret_p10_all_mean")
        - as_float(best_static, "ret_p10_all_mean"),
        "ppo_reward": as_float(ppo, "reward_total_mean"),
        "best_static_reward": as_float(best_static, "reward_total_mean"),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = build_parser().parse_args()
    sweep_root = args.output_root / f"{args.label_prefix}_{utc_now_slug()}"
    run_root = sweep_root / "runs"
    sweep_root.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)

    planned: list[dict[str, Any]] = []
    for algo in args.algos:
        for action_space_mode in args.action_space_modes:
            for reward_profile in args.reward_profiles:
                for risk_level in args.risk_levels:
                    for pt_profile in args.pt_profiles:
                        label = build_label(
                            prefix=args.label_prefix,
                            algo=algo,
                            action_space_mode=action_space_mode,
                            reward_profile=reward_profile,
                            risk_level=risk_level,
                            pt_profile=pt_profile,
                            train_timesteps=args.train_timesteps,
                        )
                        planned.append(
                            {
                                "label": label,
                                "algo": algo,
                                "action_space_mode": action_space_mode,
                                "reward_profile": reward_profile,
                                "risk_level": risk_level,
                                "pt_profile": pt_profile,
                                "norm_reward": bool(args.norm_reward),
                            }
                        )
    if args.max_runs is not None:
        planned = planned[: args.max_runs]

    results: list[dict[str, Any]] = []
    for item in planned:
        label = str(item["label"])
        run_dir = run_root / label
        command = build_command(
            args=args,
            run_root=run_root,
            label=label,
            algo=str(item["algo"]),
            action_space_mode=str(item["action_space_mode"]),
            reward_profile=str(item["reward_profile"]),
            risk_level=str(item["risk_level"]),
            pt_profile=str(item["pt_profile"]),
        )
        item_with_command = {**item, "command": command, "run_dir": str(run_dir)}
        print(" ".join(command), flush=True)
        if args.dry_run:
            results.append({**item_with_command, "status": "dry_run"})
            continue
        if args.skip_existing and (run_dir / "summary.json").exists():
            metrics = read_policy_summary(run_dir, algo=str(item["algo"]))
            results.append({**item_with_command, **metrics, "status": "skipped"})
            continue
        completed = subprocess.run(command, check=False)
        status = "complete" if completed.returncode == 0 else "failed"
        metrics = (
            read_policy_summary(run_dir, algo=str(item["algo"]))
            if completed.returncode == 0
            else {}
        )
        results.append(
            {
                **item_with_command,
                **metrics,
                "status": status,
                "returncode": completed.returncode,
            }
        )
        write_csv(sweep_root / "sweep_summary.csv", results)
        (sweep_root / "sweep_summary.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8"
        )
        if completed.returncode != 0 and args.stop_on_error:
            return completed.returncode

    write_csv(sweep_root / "sweep_summary.csv", results)
    (sweep_root / "sweep_summary.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(f"Saved sweep summary to {sweep_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
