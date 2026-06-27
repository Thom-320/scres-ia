#!/usr/bin/env python3
"""Direct PPO/SAC runner for the Garrido-2024 reward family.

This runner deliberately bypasses the ``control_v1`` survivor filter used by
``benchmark_control_reward.py``.  The goal is not to select a reward by static
control-v1 screens; the goal is to train directly on a chosen resilience reward
candidate, then evaluate the learned policy against Garrido/static baselines on
the Excel-faithful order-level ReT metric.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark_control_reward import (  # noqa: E402
    ALL_STATIC_POLICY_ORDER,
    HEURISTIC_POLICY_NAMES,
    RANDOM_POLICY_NAME,
    aggregate_policy_metrics,
    aggregate_seed_metrics,
    build_backbone_metadata,
    build_metric_contract_metadata,
    build_parser as build_benchmark_parser,
    build_reward_contract_metadata,
    clone_args,
    evaluate_policy,
    learned_phase_name,
    learned_policy_name,
    make_weight_combos,
    resolve_episode_max_steps,
    resolve_eval_risk_levels,
    resolve_git_commit,
    reward_family,
    save_csv,
    train_model,
)


DIRECT_EPISODE_FIELDNAMES = [
    "phase",
    "policy",
    "algo",
    "reward_mode",
    "reward_family",
    "frame_stack",
    "observation_version",
    "seed",
    "episode",
    "eval_seed",
    "w_bo",
    "w_cost",
    "w_disr",
    "steps",
    "reward_total",
    "mean_ret_excel_formula",
    "order_level_ret_mean",
    "fill_rate",
    "backorder_rate",
    "ret_garrido2024_raw_total",
    "ret_garrido2024_train_total",
    "ret_garrido2024_sigmoid_total",
    "ret_thesis_corrected_total",
    "ret_unified_total",
    "demanded_total",
    "delivered_total",
    "backorder_qty_total",
    "flow_fill_rate",
    "flow_backorder_rate",
    "pct_steps_S1",
    "pct_steps_S2",
    "pct_steps_S3",
]

DIRECT_POLICY_FIELDNAMES = [
    "phase",
    "policy",
    "algo",
    "reward_mode",
    "reward_family",
    "frame_stack",
    "observation_version",
    "w_bo",
    "w_cost",
    "w_disr",
    "seed_count",
    "mean_ret_excel_formula_mean",
    "mean_ret_excel_formula_std",
    "mean_ret_excel_formula_ci95_low",
    "mean_ret_excel_formula_ci95_high",
    "reward_total_mean",
    "fill_rate_mean",
    "backorder_rate_mean",
    "ret_garrido2024_train_total_mean",
    "ret_garrido2024_sigmoid_total_mean",
    "pct_steps_S1_mean",
    "pct_steps_S2_mean",
    "pct_steps_S3_mean",
]

DIRECT_COMPARISON_FIELDNAMES = [
    "reward_mode",
    "algo",
    "risk_level",
    "eval_risk_level",
    "learned_policy",
    "learned_mean_ret_excel_formula",
    "static_s2_mean_ret_excel_formula",
    "garrido_cf_s2_mean_ret_excel_formula",
    "best_baseline_policy",
    "best_baseline_mean_ret_excel_formula",
    "delta_vs_static_s2",
    "delta_vs_garrido_cf_s2",
    "delta_vs_best_baseline",
    "learned_beats_static_s2",
    "learned_beats_garrido_cf_s2",
    "learned_beats_best_baseline",
]


def add_excel_aliases(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add explicit Excel ReT aliases to rows produced by the shared benchmark."""
    out: list[dict[str, Any]] = []
    for row in rows:
        copied = dict(row)
        copied["mean_ret_excel_formula"] = float(copied["order_level_ret_mean"])
        out.append(copied)
    return out


def add_policy_excel_aliases(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        copied = dict(row)
        for suffix in ("mean", "std", "ci95_low", "ci95_high"):
            copied[f"mean_ret_excel_formula_{suffix}"] = float(
                copied[f"order_level_ret_mean_{suffix}"]
            )
        out.append(copied)
    return out


def _row_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["phase"]), str(row["policy"])


def _metric(row: dict[str, Any] | None, field: str) -> float | None:
    if row is None:
        return None
    return float(row[field])


def build_direct_comparison_rows(
    policy_rows: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Compare the learned policy to Garrido/static baselines on Excel ReT."""
    rows_by_key = {_row_key(row): row for row in policy_rows}
    learned_phase = learned_phase_name(args)
    learned_policy = learned_policy_name(args)
    learned = rows_by_key.get((learned_phase, learned_policy))
    if learned is None:
        return []

    baseline_rows = [
        row
        for row in policy_rows
        if str(row["phase"]) in {"static_screen", "heuristic_eval", "random_eval"}
    ]
    best_baseline = (
        max(
            baseline_rows,
            key=lambda row: float(row["mean_ret_excel_formula_mean"]),
        )
        if baseline_rows
        else None
    )
    static_s2 = rows_by_key.get(("static_screen", "static_s2"))
    garrido_cf_s2 = rows_by_key.get(("static_screen", "garrido_cf_s2"))
    learned_ret = float(learned["mean_ret_excel_formula_mean"])
    static_s2_ret = _metric(static_s2, "mean_ret_excel_formula_mean")
    garrido_cf_s2_ret = _metric(garrido_cf_s2, "mean_ret_excel_formula_mean")
    best_ret = _metric(best_baseline, "mean_ret_excel_formula_mean")

    def delta(other: float | None) -> float | None:
        return None if other is None else learned_ret - other

    def beats(other: float | None) -> bool | None:
        return None if other is None else learned_ret > other

    return [
        {
            "reward_mode": str(args.reward_mode),
            "algo": str(args.algo),
            "risk_level": str(args.risk_level),
            "eval_risk_level": str(args.risk_level),
            "learned_policy": learned_policy,
            "learned_mean_ret_excel_formula": learned_ret,
            "static_s2_mean_ret_excel_formula": static_s2_ret,
            "garrido_cf_s2_mean_ret_excel_formula": garrido_cf_s2_ret,
            "best_baseline_policy": (
                str(best_baseline["policy"]) if best_baseline else None
            ),
            "best_baseline_mean_ret_excel_formula": best_ret,
            "delta_vs_static_s2": delta(static_s2_ret),
            "delta_vs_garrido_cf_s2": delta(garrido_cf_s2_ret),
            "delta_vs_best_baseline": delta(best_ret),
            "learned_beats_static_s2": beats(static_s2_ret),
            "learned_beats_garrido_cf_s2": beats(garrido_cf_s2_ret),
            "learned_beats_best_baseline": beats(best_ret),
        }
    ]


def _direct_policy_summary_rows(policy_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {field: row.get(field, "") for field in DIRECT_POLICY_FIELDNAMES}
        for row in policy_rows
    ]


def _direct_episode_rows(episode_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {field: row.get(field, "") for field in DIRECT_EPISODE_FIELDNAMES}
        for row in episode_rows
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = build_benchmark_parser()
    parser.description = __doc__
    parser.set_defaults(
        reward_mode="ReT_garrido2024_train",
        output_dir=Path("outputs/benchmarks/garrido2024_direct_rl"),
        w_bo=[1.0],
        w_cost=[0.02],
        w_disr=[0.0],
        skip_artifact_export=True,
    )
    parser.add_argument(
        "--skip-heuristics",
        action="store_true",
        help="Evaluate only static baselines, random, and learned policy.",
    )
    parser.add_argument(
        "--risk-occurrence-mode",
        choices=["legacy_renewal", "thesis_window"],
        default="thesis_window",
        help="DES risk scheduling mode. Default uses the thesis-window lane.",
    )
    parser.add_argument(
        "--risk-frequency-multiplier",
        type=float,
        default=1.0,
        help="Frequency multiplier for tunable non-R3 risks in the DES.",
    )
    parser.add_argument(
        "--risk-impact-multiplier",
        type=float,
        default=1.0,
        help="Recovery/surge impact multiplier for tunable non-R3 risks in the DES.",
    )
    parser.add_argument(
        "--raw-material-flow-mode",
        default="kit_equivalent_order_up_to",
        help="Raw-material flow mode used by the DES.",
    )
    parser.add_argument(
        "--raw-material-order-up-to-multiplier",
        type=float,
        default=2.0,
        help="Order-up-to multiplier for raw material replenishment.",
    )
    parser.add_argument(
        "--ret-g24-shift-cost",
        type=float,
        default=1.0,
        help="Shift-hour cost coefficient added to Garrido-2024 κ.",
    )
    return parser


def run_direct(args: argparse.Namespace) -> dict[str, Any]:
    args.max_steps = resolve_episode_max_steps(args.step_size_hours, args.max_steps)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.reward_mode not in {
        "ReT_garrido2024_raw",
        "ReT_garrido2024_train",
        "ReT_garrido2024",
        "ReT_cd_v1",
        "ReT_seq_v1",
        "control_v1",
        "ReT_excel_delta",
    }:
        raise ValueError(
            "Direct runner expects a trainable resilience/control reward; "
            f"got {args.reward_mode!r}."
        )

    weight_combos = make_weight_combos(args)
    if len(weight_combos) != 1:
        raise ValueError("Direct runner expects exactly one weight combo.")
    weight_combo = weight_combos[0]
    eval_risk_levels = resolve_eval_risk_levels(args)
    if len(eval_risk_levels) != 1 or eval_risk_levels[0] != args.risk_level:
        raise ValueError(
            "This first direct runner supports one eval risk level matching --risk-level. "
            "Use one run per regime for now."
        )

    episode_rows: list[dict[str, Any]] = []
    training_trace_rows: list[dict[str, Any]] = []
    trained_models: list[dict[str, Any]] = []
    model_dir = args.output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        for policy in ALL_STATIC_POLICY_ORDER:
            episode_rows.extend(
                evaluate_policy(
                    "static_screen",
                    policy,
                    args=args,
                    weight_combo=weight_combo,
                    seed=seed,
                )
            )
        episode_rows.extend(
            evaluate_policy(
                "random_eval",
                RANDOM_POLICY_NAME,
                args=args,
                weight_combo=weight_combo,
                seed=seed,
            )
        )
        if not args.skip_heuristics:
            for policy in HEURISTIC_POLICY_NAMES:
                episode_rows.extend(
                    evaluate_policy(
                        "heuristic_eval",
                        policy,
                        args=args,
                        weight_combo=weight_combo,
                        seed=seed,
                    )
                )

        model, vec_env, seed_trace = train_model(args, seed, weight_combo)
        training_trace_rows.extend(seed_trace)
        model_path = model_dir / f"{args.algo}_seed{seed}_{args.reward_mode}.zip"
        model.save(str(model_path))
        trained_models.append(
            {
                "seed": int(seed),
                "algo": str(args.algo),
                "reward_mode": str(args.reward_mode),
                "model_path": str(model_path),
                "train_timesteps": int(args.train_timesteps),
            }
        )
        episode_rows.extend(
            evaluate_policy(
                learned_phase_name(args),
                learned_policy_name(args),
                args=args,
                weight_combo=weight_combo,
                seed=seed,
                model=model,
            )
        )
        vec_env.close()

    episode_rows = add_excel_aliases(episode_rows)
    seed_rows = aggregate_seed_metrics(episode_rows)
    policy_rows = add_policy_excel_aliases(aggregate_policy_metrics(seed_rows))
    comparison_rows = build_direct_comparison_rows(policy_rows, args=args)

    episode_csv = args.output_dir / "episode_metrics.csv"
    policy_csv = args.output_dir / "policy_summary.csv"
    comparison_csv = args.output_dir / "comparison_table.csv"
    training_trace_csv = args.output_dir / "training_trace.csv"
    summary_json = args.output_dir / "summary.json"

    save_csv(episode_csv, _direct_episode_rows(episode_rows), DIRECT_EPISODE_FIELDNAMES)
    save_csv(policy_csv, _direct_policy_summary_rows(policy_rows), DIRECT_POLICY_FIELDNAMES)
    save_csv(comparison_csv, comparison_rows, DIRECT_COMPARISON_FIELDNAMES)
    save_csv(training_trace_csv, training_trace_rows)

    git_commit = resolve_git_commit()
    summary = {
        "description": (
            "Direct learned-policy run without control_v1 survivor filtering; "
            "selection metric is mean_ret_excel_formula."
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "primary_metric": "mean_ret_excel_formula",
        "reward_mode": str(args.reward_mode),
        "reward_family": reward_family(str(args.reward_mode)),
        "config": {
            "algo": str(args.algo),
            "seeds": list(args.seeds),
            "train_timesteps": int(args.train_timesteps),
            "eval_episodes": int(args.eval_episodes),
            "risk_level": str(args.risk_level),
            "stochastic_pt": bool(args.stochastic_pt),
            "observation_version": str(args.observation_version),
            "step_size_hours": float(args.step_size_hours),
            "max_steps": int(args.max_steps),
            "ret_g24_kappa_train_frac": float(args.ret_g24_kappa_train_frac),
            "ret_g24_shift_cost": float(args.ret_g24_shift_cost),
            "risk_occurrence_mode": str(args.risk_occurrence_mode),
            "risk_frequency_multiplier": float(args.risk_frequency_multiplier),
            "risk_impact_multiplier": float(args.risk_impact_multiplier),
            "raw_material_flow_mode": str(args.raw_material_flow_mode),
            "raw_material_order_up_to_multiplier": float(
                args.raw_material_order_up_to_multiplier
            ),
        },
        "backbone": build_backbone_metadata(args, git_commit=git_commit),
        "metric_contract": build_metric_contract_metadata(),
        "reward_contract": build_reward_contract_metadata(args),
        "trained_models": trained_models,
        "policy_summary": policy_rows,
        "comparison_table": comparison_rows,
        "artifacts": {
            "episode_metrics_csv": str(episode_csv),
            "policy_summary_csv": str(policy_csv),
            "comparison_table_csv": str(comparison_csv),
            "training_trace_csv": str(training_trace_csv),
            "summary_json": str(summary_json),
        },
    }
    with summary_json.open("w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, indent=2)
    return summary


def main() -> None:
    args = build_parser().parse_args()
    args.invocation = "python scripts/run_garrido2024_direct_rl.py " + " ".join(
        sys.argv[1:]
    )
    summary = run_direct(args)
    print(f"Wrote direct Garrido-2024 RL artifacts to {args.output_dir}")
    for row in summary["comparison_table"]:
        print(
            "mean_ret_excel_formula: "
            f"learned={row['learned_mean_ret_excel_formula']:.6f}, "
            f"garrido_cf_s2={row['garrido_cf_s2_mean_ret_excel_formula']:.6f}, "
            f"best={row['best_baseline_policy']}:{row['best_baseline_mean_ret_excel_formula']:.6f}, "
            f"delta_best={row['delta_vs_best_baseline']:.6f}"
        )


if __name__ == "__main__":
    main()
