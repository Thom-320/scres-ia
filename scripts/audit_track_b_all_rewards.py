#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys
from typing import Any

import numpy as np
from scipy import stats as scipy_stats
try:
    from sb3_contrib import RecurrentPPO
except ImportError:  # pragma: no cover - optional dependency at runtime.
    RecurrentPPO = None
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_track_b_smoke import (  # noqa: E402
    DOWNSTREAM_NEAR_MAX_THRESHOLD,
    STATIC_POLICY_SPECS,
    StaticPolicySpec,
    build_static_policy_action,
    ci95,
    extract_downstream_multipliers,
)
from supply_chain.env_experimental_shifts import (  # noqa: E402
    REWARD_MODE_OPTIONS,
    RET_SEQ_KAPPA,
    RET_SEQ_W_AE,
    RET_SEQ_W_BC,
    RET_SEQ_W_SC,
)
from supply_chain.external_env_interface import (  # noqa: E402
    get_episode_terminal_metrics,
    make_track_b_env,
)

DEFAULT_PPO_BUNDLE = Path(
    "outputs/track_b_benchmarks/track_b_ret_seq_k020_500k_rerun1"
)
DEFAULT_RECURRENT_BUNDLE = Path(
    "outputs/track_b_benchmarks/track_b_ret_seq_k020_500k_recurrent_ppo_rerun1"
)
DEFAULT_SEEDS = (11, 22, 33, 44, 55)
DEFAULT_EVAL_EPISODES = 3
DEFAULT_MAX_STEPS = 260
DEFAULT_STEP_SIZE_HOURS = 168.0
EVAL_EPISODE_SEED_OFFSET = 90_000
PAPER_REFERENCE_REWARD_MODE = "ReT_seq_v1"

AUDIT_METRICS = (
    "reward_total",
    "ret_thesis_total",
    "ret_thesis_corrected_total",
    "ret_seq_total",
    "ret_unified_total",
    "ret_garrido2024_raw_total",
    "ret_garrido2024_train_total",
    "ret_garrido2024_sigmoid_total",
    "fill_rate",
    "backorder_rate",
    "order_level_ret_mean",
    "flow_fill_rate",
    "flow_backorder_rate",
    "terminal_rolling_fill_rate_4w",
    "terminal_rolling_backorder_rate_4w",
    "delivered_total",
    "produced_total",
    "avg_annual_delivery",
    "avg_annual_production",
    "avg_total_inventory",
    "terminal_pending_backorder_qty",
    "disruption_hours_total",
    "total_step_cost",
    "avg_step_cost",
    "terminal_average_cost",
    "service_continuity_step_mean",
    "backlog_containment_step_mean",
    "adaptive_efficiency_step_mean",
    "ret_seq_step_mean",
    "service_loss_area",
    "service_loss_area_below_095",
    "recovery_streak_count",
    "mean_recovery_streak_steps",
    "mean_recovery_streak_hours",
    "max_recovery_streak_hours",
    "pct_steps_S1",
    "pct_steps_S2",
    "pct_steps_S3",
    "op10_multiplier_step_mean",
    "op12_multiplier_step_mean",
    "op10_multiplier_step_p95",
    "op12_multiplier_step_p95",
    "pct_steps_op10_multiplier_ge_190",
    "pct_steps_op12_multiplier_ge_190",
    "pct_steps_both_downstream_ge_190",
    "pct_ret_case_fill_rate_only",
    "pct_ret_case_recovery",
    "pct_ret_case_autotomy",
    "pct_ret_case_non_recovery",
    "pct_ret_case_no_demand",
    "terminal_zeta_avg",
    "terminal_epsilon_avg",
    "terminal_phi_avg",
    "terminal_tau_avg",
    "terminal_kappa_dot",
)

PAIRWISE_METRICS = (
    "fill_rate",
    "backorder_rate",
    "order_level_ret_mean",
    "delivered_total",
    "avg_annual_delivery",
    "total_step_cost",
    "avg_step_cost",
    "service_continuity_step_mean",
    "backlog_containment_step_mean",
    "adaptive_efficiency_step_mean",
    "ret_seq_total",
    "ret_garrido2024_sigmoid_total",
    "service_loss_area_below_095",
    "mean_recovery_streak_hours",
)

LOWER_IS_BETTER_METRICS = {
    "backorder_rate",
    "total_step_cost",
    "avg_step_cost",
    "service_loss_area_below_095",
    "mean_recovery_streak_hours",
}

PAPER_POLICY_ORDER = (
    "ppo",
    "recurrent_ppo",
    "s1_d1.00",
    "s1_d1.50",
    "s1_d2.00",
    "s2_d1.00",
    "s2_d1.50",
    "s2_d2.00",
    "s3_d1.00",
    "s3_d1.50",
    "s3_d2.00",
)

MANUSCRIPT_POLICY_ORDER = (
    "ppo",
    "recurrent_ppo",
    "s1_d1.00",
    "s2_d1.00",
    "s2_d2.00",
    "s3_d2.00",
)

MANUSCRIPT_POLICY_LABELS = {
    "ppo": "PPO",
    "recurrent_ppo": "RecurrentPPO",
    "s1_d1.00": "S1",
    "s2_d1.00": "S2",
    "s2_d2.00": "S2(d=2.0)",
    "s3_d2.00": "S3(d=2.0)",
}

PAPER_MAIN_TABLE_SPEC = (
    ("fill_rate", "fill_rate"),
    ("backorder_rate", "backorder_rate"),
    ("order_level_ret_mean", "garrido2017_order_level_ret"),
    ("ret_garrido2024_sigmoid_total", "ret_garrido2024_sigmoid_total"),
    ("delivered_total", "delivered_total"),
    ("avg_annual_delivery", "avg_annual_delivery"),
    ("total_step_cost", "total_step_cost"),
    ("avg_step_cost", "avg_step_cost"),
    ("pct_steps_S1", "pct_steps_S1"),
    ("pct_steps_S2", "pct_steps_S2"),
    ("pct_steps_S3", "pct_steps_S3"),
    ("pct_ret_case_autotomy", "pct_ret_case_autotomy"),
    ("pct_ret_case_recovery", "pct_ret_case_recovery"),
    ("pct_ret_case_non_recovery", "pct_ret_case_non_recovery"),
)

PAPER_MECHANISM_TABLE_SPEC = (
    ("service_continuity_step_mean", "service_continuity"),
    ("backlog_containment_step_mean", "backlog_containment"),
    ("adaptive_efficiency_step_mean", "adaptive_efficiency"),
    ("ret_seq_total", "ret_seq_total"),
    ("ret_seq_step_mean", "ret_seq_step_mean"),
    ("ret_thesis_total", "thesis_step_proxy_total"),
    ("ret_thesis_corrected_total", "thesis_step_proxy_corrected_total"),
    ("ret_unified_total", "ret_unified_total"),
    ("service_loss_area", "service_loss_area"),
    ("service_loss_area_below_095", "service_loss_area_below_095"),
    ("mean_recovery_streak_hours", "mean_recovery_streak_hours"),
    ("max_recovery_streak_hours", "max_recovery_streak_hours"),
)

PAPER_RAW_TABLE_SPEC = (
    ("terminal_zeta_avg", "zeta_avg"),
    ("terminal_epsilon_avg", "epsilon_avg"),
    ("terminal_phi_avg", "phi_avg"),
    ("terminal_tau_avg", "tau_avg"),
    ("terminal_kappa_dot", "kappa_dot"),
    ("avg_total_inventory", "avg_total_inventory"),
    ("terminal_pending_backorder_qty", "terminal_pending_backorder_qty"),
    ("produced_total", "produced_total"),
    ("avg_annual_production", "avg_annual_production"),
    ("terminal_average_cost", "terminal_average_cost"),
)

PAPER_CONTROL_TABLE_SPEC = (
    ("op10_multiplier_step_mean", "op10_multiplier_step_mean"),
    ("op10_multiplier_step_p95", "op10_multiplier_step_p95"),
    ("pct_steps_op10_multiplier_ge_190", "pct_steps_op10_multiplier_ge_190"),
    ("op12_multiplier_step_mean", "op12_multiplier_step_mean"),
    ("op12_multiplier_step_p95", "op12_multiplier_step_p95"),
    ("pct_steps_op12_multiplier_ge_190", "pct_steps_op12_multiplier_ge_190"),
    ("pct_steps_both_downstream_ge_190", "pct_steps_both_downstream_ge_190"),
)

RET_CASES: tuple[str, ...] = (
    "fill_rate_only",
    "recovery",
    "autotomy",
    "non_recovery",
    "no_demand",
)


@dataclass(frozen=True)
class LearnedBundle:
    policy: str
    algo: str
    bundle_dir: Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit Track B policies under all implemented resilience functions "
            "without retraining. This keeps the policy fixed and changes only "
            "the evaluation reward lens."
        )
    )
    parser.add_argument(
        "--ppo-bundle",
        type=Path,
        default=DEFAULT_PPO_BUNDLE,
        help="Frozen PPO Track B bundle directory.",
    )
    parser.add_argument(
        "--recurrent-bundle",
        type=Path,
        default=DEFAULT_RECURRENT_BUNDLE,
        help="Frozen RecurrentPPO Track B bundle directory.",
    )
    parser.add_argument(
        "--reward-modes",
        nargs="+",
        choices=list(REWARD_MODE_OPTIONS),
        default=list(REWARD_MODE_OPTIONS),
        help="Audit reward modes to evaluate under the same Track B backbone.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help="Model seeds to audit.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=DEFAULT_EVAL_EPISODES,
        help="Evaluation episodes per policy, seed, and reward mode.",
    )
    parser.add_argument(
        "--risk-level",
        default="adaptive_benchmark_v2",
        help="Track B risk profile.",
    )
    parser.add_argument(
        "--step-size-hours",
        type=float,
        default=DEFAULT_STEP_SIZE_HOURS,
        help="Decision cadence in hours.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help="Episode horizon in decision steps.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write the audit bundle.",
    )
    parser.add_argument(
        "--skip-recurrent",
        action="store_true",
        help="Audit only PPO and statics.",
    )
    return parser


def default_output_dir() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("outputs/track_b_benchmarks") / f"track_b_all_reward_audit_{timestamp}"


def build_env_kwargs(args: argparse.Namespace, reward_mode: str) -> dict[str, Any]:
    return {
        "reward_mode": reward_mode,
        "risk_level": args.risk_level,
        "step_size_hours": args.step_size_hours,
        "max_steps": args.max_steps,
    }


def model_filename(algo: str) -> str:
    return "ppo_model.zip" if algo == "ppo" else "recurrent_ppo_model.zip"


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_learned_bundles(args: argparse.Namespace) -> list[LearnedBundle]:
    bundles = [LearnedBundle(policy="ppo", algo="ppo", bundle_dir=args.ppo_bundle)]
    if not args.skip_recurrent:
        bundles.append(
            LearnedBundle(
                policy="recurrent_ppo",
                algo="recurrent_ppo",
                bundle_dir=args.recurrent_bundle,
            )
        )
    return bundles


def validate_bundle(bundle: LearnedBundle, seeds: list[int]) -> None:
    if not bundle.bundle_dir.exists():
        raise FileNotFoundError(f"Missing bundle directory: {bundle.bundle_dir}")
    for seed in seeds:
        model_path = bundle.bundle_dir / "models" / f"seed{seed}" / model_filename(
            bundle.algo
        )
        vec_path = bundle.bundle_dir / "models" / f"seed{seed}" / "vec_normalize.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model for {bundle.policy} seed {seed}: {model_path}")
        if not vec_path.exists():
            raise FileNotFoundError(f"Missing VecNormalize for {bundle.policy} seed {seed}: {vec_path}")


def load_model(algo: str, model_path: Path) -> Any:
    if algo == "ppo":
        return PPO.load(str(model_path), device="cpu")
    if algo == "recurrent_ppo":
        if RecurrentPPO is None:
            raise RuntimeError(
                "recurrent_ppo audit requested but sb3_contrib is not installed."
            )
        return RecurrentPPO.load(str(model_path), device="cpu")
    raise ValueError(f"Unsupported algo={algo!r}.")


def load_vec_normalize(
    vec_norm_path: Path, *, args: argparse.Namespace, reward_mode: str
) -> VecNormalize:
    env_kwargs = build_env_kwargs(args, reward_mode)

    def _init() -> Any:
        return make_track_b_env(**env_kwargs)

    vec_norm = VecNormalize.load(str(vec_norm_path), DummyVecEnv([_init]))
    vec_norm.training = False
    vec_norm.norm_reward = False
    return vec_norm


def mean_or_zero(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def build_ret_seq_audit_terms(final_info: dict[str, Any]) -> dict[str, float]:
    eps = 1e-6
    new_demanded = float(final_info.get("new_demanded", 0.0))
    new_backorder_qty = float(final_info.get("new_backorder_qty", 0.0))
    pending_backorder_qty = float(final_info.get("pending_backorder_qty", 0.0))
    cumulative_demanded = max(
        float(final_info.get("cumulative_demanded_post_warmup", 1.0)),
        1.0,
    )
    shifts = int(final_info.get("shifts_active", 1))
    kappa = float(final_info.get("ret_seq_kappa", RET_SEQ_KAPPA))

    if new_demanded > 0.0:
        service_continuity = max(eps, 1.0 - new_backorder_qty / new_demanded)
    else:
        service_continuity = 1.0
    backlog_containment = max(
        eps,
        1.0 - min(1.0, pending_backorder_qty / cumulative_demanded),
    )
    adaptive_efficiency = max(eps, 1.0 - kappa * (shifts - 1) / 2.0)
    ret_seq_step = (
        service_continuity**RET_SEQ_W_SC
        * backlog_containment**RET_SEQ_W_BC
        * adaptive_efficiency**RET_SEQ_W_AE
    )
    return {
        "service_continuity_step": float(
            final_info.get("service_continuity_step", service_continuity)
        ),
        "backlog_containment_step": float(
            final_info.get("backlog_containment_step", backlog_containment)
        ),
        "adaptive_efficiency_step": float(
            final_info.get("adaptive_efficiency_step", adaptive_efficiency)
        ),
        "ret_seq_step": float(final_info.get("ret_seq_step", ret_seq_step)),
    }


def summarize_recovery_streaks(
    streaks: list[int], *, step_size_hours: float
) -> dict[str, float]:
    if not streaks:
        return {
            "recovery_streak_count": 0.0,
            "mean_recovery_streak_steps": 0.0,
            "mean_recovery_streak_hours": 0.0,
            "max_recovery_streak_hours": 0.0,
        }
    mean_steps = mean_or_zero([float(step) for step in streaks])
    return {
        "recovery_streak_count": float(len(streaks)),
        "mean_recovery_streak_steps": mean_steps,
        "mean_recovery_streak_hours": mean_steps * step_size_hours,
        "max_recovery_streak_hours": float(max(streaks)) * step_size_hours,
    }


def rank_biserial_from_diffs(diffs: np.ndarray) -> float:
    nonzero = diffs[np.abs(diffs) > 1e-12]
    if nonzero.size == 0:
        return 0.0
    ranks = scipy_stats.rankdata(np.abs(nonzero), method="average")
    positive = float(np.sum(ranks[nonzero > 0]))
    negative = float(np.sum(ranks[nonzero < 0]))
    denom = positive + negative
    if denom <= 0.0:
        return 0.0
    return (positive - negative) / denom


def build_pairwise_stats(
    episode_rows: list[dict[str, Any]],
    *,
    reward_mode: str = PAPER_REFERENCE_REWARD_MODE,
    anchor_policy: str = "ppo",
) -> list[dict[str, Any]]:
    rows_for_mode = [row for row in episode_rows if row["reward_mode"] == reward_mode]
    by_policy: dict[str, dict[tuple[int, int, int], dict[str, Any]]] = {}
    for row in rows_for_mode:
        key = (int(row["seed"]), int(row["episode"]), int(row["eval_seed"]))
        by_policy.setdefault(str(row["policy"]), {})[key] = row

    anchor_rows = by_policy.get(anchor_policy, {})
    if not anchor_rows:
        return []

    comparisons: list[dict[str, Any]] = []
    for comparator_policy, comparator_rows in sorted(by_policy.items()):
        if comparator_policy == anchor_policy:
            continue
        shared_keys = sorted(set(anchor_rows) & set(comparator_rows))
        if not shared_keys:
            continue
        for metric in PAIRWISE_METRICS:
            anchor_values = np.asarray(
                [float(anchor_rows[key][metric]) for key in shared_keys],
                dtype=np.float64,
            )
            comparator_values = np.asarray(
                [float(comparator_rows[key][metric]) for key in shared_keys],
                dtype=np.float64,
            )
            diffs = anchor_values - comparator_values
            nonzero = diffs[np.abs(diffs) > 1e-12]
            if nonzero.size == 0:
                p_value = 1.0
                statistic = 0.0
                effect = 0.0
            else:
                test = scipy_stats.wilcoxon(nonzero, alternative="two-sided")
                p_value = float(test.pvalue)
                statistic = float(test.statistic)
                effect = float(rank_biserial_from_diffs(nonzero))
            ci_low, ci_high = ci95(diffs.tolist())
            mean_diff = float(np.mean(diffs))
            median_diff = float(np.median(diffs))
            wins = int(np.sum(diffs > 0))
            losses = int(np.sum(diffs < 0))
            ties = int(np.sum(np.abs(diffs) <= 1e-12))
            higher_is_better = metric not in LOWER_IS_BETTER_METRICS
            anchor_better = mean_diff > 0.0 if higher_is_better else mean_diff < 0.0
            comparisons.append(
                {
                    "reward_mode": reward_mode,
                    "anchor_policy": anchor_policy,
                    "comparator_policy": comparator_policy,
                    "metric": metric,
                    "orientation": (
                        "higher_is_better" if higher_is_better else "lower_is_better"
                    ),
                    "n_pairs": len(shared_keys),
                    "anchor_mean": float(np.mean(anchor_values)),
                    "comparator_mean": float(np.mean(comparator_values)),
                    "mean_diff_anchor_minus_comparator": mean_diff,
                    "median_diff_anchor_minus_comparator": median_diff,
                    "diff_ci95_low": ci_low,
                    "diff_ci95_high": ci_high,
                    "wilcoxon_statistic": statistic,
                    "wilcoxon_pvalue": p_value,
                    "rank_biserial": effect,
                    "anchor_wins": wins,
                    "anchor_losses": losses,
                    "ties": ties,
                    "anchor_better": anchor_better,
                }
            )
    return comparisons


def build_paper_table(
    rows: list[dict[str, Any]],
    *,
    spec: tuple[tuple[str, str], ...],
) -> list[dict[str, Any]]:
    by_policy = {str(row["policy"]): row for row in rows}
    ordered_policies = [policy for policy in PAPER_POLICY_ORDER if policy in by_policy]
    extra_policies = sorted(set(by_policy) - set(ordered_policies))
    table_rows: list[dict[str, Any]] = []
    for policy in [*ordered_policies, *extra_policies]:
        source = by_policy[policy]
        row: dict[str, Any] = {"policy": policy}
        for metric, label in spec:
            row[f"{label}_mean"] = float(source[f"{metric}_mean"])
            row[f"{label}_ci95_low"] = float(source[f"{metric}_ci95_low"])
            row[f"{label}_ci95_high"] = float(source[f"{metric}_ci95_high"])
        table_rows.append(row)
    return table_rows


def build_manuscript_main_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_policy = {str(row["policy"]): row for row in rows}
    table_rows: list[dict[str, Any]] = []
    for policy in MANUSCRIPT_POLICY_ORDER:
        source = by_policy.get(policy)
        if source is None:
            continue
        table_rows.append(
            {
                "policy": MANUSCRIPT_POLICY_LABELS.get(policy, policy),
                "fill_rate": float(source["fill_rate_mean"]),
                "backorder_rate": float(source["backorder_rate_mean"]),
                "garrido2017_order_level_ret": float(
                    source["order_level_ret_mean_mean"]
                ),
                "ret_garrido2024_sigmoid_total": float(
                    source["ret_garrido2024_sigmoid_total_mean"]
                ),
                "avg_annual_delivery": float(source["avg_annual_delivery_mean"]),
                "pct_steps_S1": float(source["pct_steps_S1_mean"]),
                "pct_steps_S2": float(source["pct_steps_S2_mean"]),
                "pct_steps_S3": float(source["pct_steps_S3_mean"]),
                "pct_ret_case_autotomy": float(source["pct_ret_case_autotomy_mean"]),
                "pct_ret_case_recovery": float(source["pct_ret_case_recovery_mean"]),
                "pct_ret_case_non_recovery": float(
                    source["pct_ret_case_non_recovery_mean"]
                ),
            }
        )
    return table_rows


def render_markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        formatted: list[str] = []
        for value in row.values():
            if isinstance(value, float):
                formatted.append(f"{value:.6f}")
            else:
                formatted.append(str(value))
        lines.append("| " + " | ".join(formatted) + " |")
    return "\n".join(lines) + "\n"


def finalize_episode_row(
    *,
    reward_mode: str,
    policy: str,
    algo: str,
    seed: int,
    episode: int,
    eval_seed: int,
    steps: int,
    reward_total: float,
    ret_thesis_total: float,
    ret_thesis_corrected_total: float,
    ret_unified_total: float,
    ret_garrido2024_raw_total: float,
    ret_garrido2024_train_total: float,
    ret_garrido2024_sigmoid_total: float,
    ret_seq_total: float,
    delivered_total: float,
    produced_total: float,
    demanded_total: float,
    backorder_qty_total: float,
    disruption_hours_total: float,
    inventory_total_sum: float,
    step_cost_total: float,
    service_continuity_total: float,
    backlog_containment_total: float,
    adaptive_efficiency_total: float,
    service_loss_area: float,
    service_loss_area_below_095: float,
    recovery_streaks: list[int],
    shift_counts: dict[int, int],
    ret_case_counts: dict[str, int],
    op10_multipliers: list[float],
    op12_multipliers: list[float],
    final_info: dict[str, Any],
    terminal_metrics: dict[str, float],
    step_size_hours: float,
) -> dict[str, Any]:
    total_steps = max(1, steps)
    episode_years = max((steps * step_size_hours) / (24.0 * 365.0), 1e-9)
    if demanded_total > 0.0:
        flow_backorder_rate = backorder_qty_total / demanded_total
        flow_fill_rate = 1.0 - flow_backorder_rate
    else:
        flow_backorder_rate = 0.0
        flow_fill_rate = 1.0
    track_b_context = final_info["state_constraint_context"]["track_b_context"]
    op10_arr = np.asarray(op10_multipliers or [1.0], dtype=np.float64)
    op12_arr = np.asarray(op12_multipliers or [1.0], dtype=np.float64)
    recovery_summary = summarize_recovery_streaks(
        recovery_streaks,
        step_size_hours=step_size_hours,
    )
    return {
        "reward_mode": reward_mode,
        "policy": policy,
        "algo": algo,
        "seed": seed,
        "episode": episode,
        "eval_seed": eval_seed,
        "steps": steps,
        "reward_total": reward_total,
        "ret_thesis_total": ret_thesis_total,
        "ret_thesis_corrected_total": ret_thesis_corrected_total,
        "ret_seq_total": ret_seq_total,
        "ret_unified_total": ret_unified_total,
        "ret_garrido2024_raw_total": ret_garrido2024_raw_total,
        "ret_garrido2024_train_total": ret_garrido2024_train_total,
        "ret_garrido2024_sigmoid_total": ret_garrido2024_sigmoid_total,
        "delivered_total": delivered_total,
        "produced_total": produced_total,
        "avg_annual_delivery": delivered_total / episode_years,
        "avg_annual_production": produced_total / episode_years,
        "fill_rate": float(terminal_metrics["fill_rate_order_level"]),
        "backorder_rate": float(terminal_metrics["backorder_rate_order_level"]),
        "order_level_ret_mean": float(terminal_metrics["order_level_ret_mean"]),
        "flow_fill_rate": flow_fill_rate,
        "flow_backorder_rate": flow_backorder_rate,
        "avg_total_inventory": inventory_total_sum / total_steps,
        "terminal_pending_backorder_qty": float(
            final_info.get("pending_backorder_qty", 0.0)
        ),
        "disruption_hours_total": disruption_hours_total,
        "total_step_cost": step_cost_total,
        "avg_step_cost": step_cost_total / total_steps,
        "terminal_average_cost": float(final_info.get("average_cost", 0.0)),
        "service_continuity_step_mean": service_continuity_total / total_steps,
        "backlog_containment_step_mean": backlog_containment_total / total_steps,
        "adaptive_efficiency_step_mean": adaptive_efficiency_total / total_steps,
        "ret_seq_step_mean": ret_seq_total / total_steps,
        "service_loss_area": service_loss_area,
        "service_loss_area_below_095": service_loss_area_below_095,
        "terminal_rolling_fill_rate_4w": float(track_b_context["rolling_fill_rate_4w"]),
        "terminal_rolling_backorder_rate_4w": float(
            track_b_context["rolling_backorder_rate_4w"]
        ),
        "pct_steps_S1": 100.0 * shift_counts.get(1, 0) / total_steps,
        "pct_steps_S2": 100.0 * shift_counts.get(2, 0) / total_steps,
        "pct_steps_S3": 100.0 * shift_counts.get(3, 0) / total_steps,
        "op10_multiplier_step_mean": float(np.mean(op10_arr)),
        "op12_multiplier_step_mean": float(np.mean(op12_arr)),
        "op10_multiplier_step_p95": float(np.percentile(op10_arr, 95)),
        "op12_multiplier_step_p95": float(np.percentile(op12_arr, 95)),
        "pct_steps_op10_multiplier_ge_190": 100.0
        * float(np.mean(op10_arr >= DOWNSTREAM_NEAR_MAX_THRESHOLD)),
        "pct_steps_op12_multiplier_ge_190": 100.0
        * float(np.mean(op12_arr >= DOWNSTREAM_NEAR_MAX_THRESHOLD)),
        "pct_steps_both_downstream_ge_190": 100.0
        * float(
            np.mean(
                (op10_arr >= DOWNSTREAM_NEAR_MAX_THRESHOLD)
                & (op12_arr >= DOWNSTREAM_NEAR_MAX_THRESHOLD)
            )
        ),
        "pct_ret_case_fill_rate_only": 100.0
        * ret_case_counts.get("fill_rate_only", 0)
        / total_steps,
        "pct_ret_case_recovery": 100.0 * ret_case_counts.get("recovery", 0) / total_steps,
        "pct_ret_case_autotomy": 100.0 * ret_case_counts.get("autotomy", 0) / total_steps,
        "pct_ret_case_non_recovery": 100.0
        * ret_case_counts.get("non_recovery", 0)
        / total_steps,
        "pct_ret_case_no_demand": 100.0 * ret_case_counts.get("no_demand", 0) / total_steps,
        "terminal_zeta_avg": float(final_info.get("zeta_avg", 0.0)),
        "terminal_epsilon_avg": float(final_info.get("epsilon_avg", 0.0)),
        "terminal_phi_avg": float(final_info.get("phi_avg", 0.0)),
        "terminal_tau_avg": float(final_info.get("tau_avg", 0.0)),
        "terminal_kappa_dot": float(final_info.get("kappa_dot", 0.0)),
        **recovery_summary,
    }


def evaluate_static_policy(
    *,
    args: argparse.Namespace,
    reward_mode: str,
    policy: StaticPolicySpec,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    env_kwargs = build_env_kwargs(args, reward_mode)
    action_payload = build_static_policy_action(policy)

    for episode_idx in range(args.eval_episodes):
        eval_seed = seed + EVAL_EPISODE_SEED_OFFSET + episode_idx
        env = make_track_b_env(**env_kwargs)
        _, info = env.reset(seed=eval_seed)
        terminated = False
        truncated = False
        reward_total = 0.0
        ret_thesis_total = 0.0
        ret_thesis_corrected_total = 0.0
        ret_seq_total = 0.0
        ret_unified_total = 0.0
        ret_garrido2024_raw_total = 0.0
        ret_garrido2024_train_total = 0.0
        ret_garrido2024_sigmoid_total = 0.0
        delivered_total = 0.0
        produced_total = 0.0
        demanded_total = 0.0
        backorder_qty_total = 0.0
        disruption_hours_total = 0.0
        inventory_total_sum = 0.0
        step_cost_total = 0.0
        service_continuity_total = 0.0
        backlog_containment_total = 0.0
        adaptive_efficiency_total = 0.0
        service_loss_area = 0.0
        service_loss_area_below_095 = 0.0
        recovery_streaks: list[int] = []
        current_recovery_streak = 0
        shift_counts = {1: 0, 2: 0, 3: 0}
        ret_case_counts = {case: 0 for case in RET_CASES}
        op10_multipliers: list[float] = []
        op12_multipliers: list[float] = []
        steps = 0
        final_info = info

        while not (terminated or truncated):
            _, reward, terminated, truncated, final_info = env.step(action_payload)
            reward_total += float(reward)
            ret_thesis_total += float(final_info.get("ret_thesis_step", 0.0))
            ret_thesis_corrected_total += float(
                final_info.get("ret_thesis_corrected_step", 0.0)
            )
            ret_unified_total += float(final_info.get("ret_unified_step", 0.0))
            ret_garrido2024_raw_total += float(
                final_info.get("ret_garrido2024_raw_step", 0.0)
            )
            ret_garrido2024_train_total += float(
                final_info.get("ret_garrido2024_train_step", 0.0)
            )
            ret_garrido2024_sigmoid_total += float(
                final_info.get("ret_garrido2024_sigmoid_step", 0.0)
            )
            delivered_total += float(final_info.get("new_delivered", 0.0))
            produced_total += float(final_info.get("new_produced", 0.0))
            demanded_total += float(final_info.get("new_demanded", 0.0))
            backorder_qty_total += float(final_info.get("new_backorder_qty", 0.0))
            disruption_hours_total += float(final_info.get("step_disruption_hours", 0.0))
            inventory_total_sum += float(final_info.get("total_inventory", 0.0))
            step_cost_total += float(final_info.get("step_cost", 0.0))
            ret_seq_terms = build_ret_seq_audit_terms(final_info)
            service_continuity_total += ret_seq_terms["service_continuity_step"]
            backlog_containment_total += ret_seq_terms["backlog_containment_step"]
            adaptive_efficiency_total += ret_seq_terms["adaptive_efficiency_step"]
            ret_seq_total += ret_seq_terms["ret_seq_step"]
            service_loss_area += 1.0 - ret_seq_terms["service_continuity_step"]
            service_loss_area_below_095 += max(
                0.0,
                0.95 - ret_seq_terms["service_continuity_step"],
            )
            ret_case = final_info.get("ret_components", {}).get("ret_case")
            if isinstance(ret_case, str) and ret_case in ret_case_counts:
                ret_case_counts[ret_case] += 1
            if ret_case == "recovery":
                current_recovery_streak += 1
            elif current_recovery_streak > 0:
                recovery_streaks.append(current_recovery_streak)
                current_recovery_streak = 0
            shift_counts[int(final_info.get("shifts_active", 1))] += 1
            op10_mult, op12_mult = extract_downstream_multipliers(final_info)
            op10_multipliers.append(op10_mult)
            op12_multipliers.append(op12_mult)
            steps += 1

        if current_recovery_streak > 0:
            recovery_streaks.append(current_recovery_streak)
        rows.append(
            finalize_episode_row(
                reward_mode=reward_mode,
                policy=policy.label,
                algo="static",
                seed=seed,
                episode=episode_idx + 1,
                eval_seed=eval_seed,
                steps=steps,
                reward_total=reward_total,
                ret_thesis_total=ret_thesis_total,
                ret_thesis_corrected_total=ret_thesis_corrected_total,
                ret_seq_total=ret_seq_total,
                ret_unified_total=ret_unified_total,
                ret_garrido2024_raw_total=ret_garrido2024_raw_total,
                ret_garrido2024_train_total=ret_garrido2024_train_total,
                ret_garrido2024_sigmoid_total=ret_garrido2024_sigmoid_total,
                delivered_total=delivered_total,
                produced_total=produced_total,
                demanded_total=demanded_total,
                backorder_qty_total=backorder_qty_total,
                disruption_hours_total=disruption_hours_total,
                inventory_total_sum=inventory_total_sum,
                step_cost_total=step_cost_total,
                service_continuity_total=service_continuity_total,
                backlog_containment_total=backlog_containment_total,
                adaptive_efficiency_total=adaptive_efficiency_total,
                service_loss_area=service_loss_area,
                service_loss_area_below_095=service_loss_area_below_095,
                recovery_streaks=recovery_streaks,
                shift_counts=shift_counts,
                ret_case_counts=ret_case_counts,
                op10_multipliers=op10_multipliers,
                op12_multipliers=op12_multipliers,
                final_info=final_info,
                terminal_metrics=get_episode_terminal_metrics(env),
                step_size_hours=float(args.step_size_hours),
            )
        )
        env.close()
    return rows


def evaluate_learned_policy(
    *,
    args: argparse.Namespace,
    reward_mode: str,
    bundle: LearnedBundle,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    run_dir = bundle.bundle_dir / "models" / f"seed{seed}"
    model = load_model(bundle.algo, run_dir / model_filename(bundle.algo))
    vec_norm = load_vec_normalize(
        run_dir / "vec_normalize.pkl", args=args, reward_mode=reward_mode
    )
    env_kwargs = build_env_kwargs(args, reward_mode)
    is_recurrent = bundle.algo == "recurrent_ppo"

    for episode_idx in range(args.eval_episodes):
        eval_seed = seed + EVAL_EPISODE_SEED_OFFSET + episode_idx
        env = make_track_b_env(**env_kwargs)
        obs, info = env.reset(seed=eval_seed)
        terminated = False
        truncated = False
        reward_total = 0.0
        ret_thesis_total = 0.0
        ret_thesis_corrected_total = 0.0
        ret_seq_total = 0.0
        ret_unified_total = 0.0
        ret_garrido2024_raw_total = 0.0
        ret_garrido2024_train_total = 0.0
        ret_garrido2024_sigmoid_total = 0.0
        delivered_total = 0.0
        produced_total = 0.0
        demanded_total = 0.0
        backorder_qty_total = 0.0
        disruption_hours_total = 0.0
        inventory_total_sum = 0.0
        step_cost_total = 0.0
        service_continuity_total = 0.0
        backlog_containment_total = 0.0
        adaptive_efficiency_total = 0.0
        service_loss_area = 0.0
        service_loss_area_below_095 = 0.0
        recovery_streaks: list[int] = []
        current_recovery_streak = 0
        shift_counts = {1: 0, 2: 0, 3: 0}
        ret_case_counts = {case: 0 for case in RET_CASES}
        op10_multipliers: list[float] = []
        op12_multipliers: list[float] = []
        steps = 0
        final_info = info
        lstm_states: Any = None
        episode_start = np.ones((1,), dtype=bool)

        while not (terminated or truncated):
            obs_norm = vec_norm.normalize_obs(
                np.asarray(obs, dtype=np.float32)[None, :]
            )
            if is_recurrent:
                action, lstm_states = model.predict(
                    obs_norm,
                    state=lstm_states,
                    episode_start=episode_start,
                    deterministic=True,
                )
            else:
                action, _ = model.predict(obs_norm, deterministic=True)
            obs, reward, terminated, truncated, final_info = env.step(
                np.asarray(action[0], dtype=np.float32)
            )
            reward_total += float(reward)
            ret_thesis_total += float(final_info.get("ret_thesis_step", 0.0))
            ret_thesis_corrected_total += float(
                final_info.get("ret_thesis_corrected_step", 0.0)
            )
            ret_unified_total += float(final_info.get("ret_unified_step", 0.0))
            ret_garrido2024_raw_total += float(
                final_info.get("ret_garrido2024_raw_step", 0.0)
            )
            ret_garrido2024_train_total += float(
                final_info.get("ret_garrido2024_train_step", 0.0)
            )
            ret_garrido2024_sigmoid_total += float(
                final_info.get("ret_garrido2024_sigmoid_step", 0.0)
            )
            delivered_total += float(final_info.get("new_delivered", 0.0))
            produced_total += float(final_info.get("new_produced", 0.0))
            demanded_total += float(final_info.get("new_demanded", 0.0))
            backorder_qty_total += float(final_info.get("new_backorder_qty", 0.0))
            disruption_hours_total += float(final_info.get("step_disruption_hours", 0.0))
            inventory_total_sum += float(final_info.get("total_inventory", 0.0))
            step_cost_total += float(final_info.get("step_cost", 0.0))
            ret_seq_terms = build_ret_seq_audit_terms(final_info)
            service_continuity_total += ret_seq_terms["service_continuity_step"]
            backlog_containment_total += ret_seq_terms["backlog_containment_step"]
            adaptive_efficiency_total += ret_seq_terms["adaptive_efficiency_step"]
            ret_seq_total += ret_seq_terms["ret_seq_step"]
            service_loss_area += 1.0 - ret_seq_terms["service_continuity_step"]
            service_loss_area_below_095 += max(
                0.0,
                0.95 - ret_seq_terms["service_continuity_step"],
            )
            ret_case = final_info.get("ret_components", {}).get("ret_case")
            if isinstance(ret_case, str) and ret_case in ret_case_counts:
                ret_case_counts[ret_case] += 1
            if ret_case == "recovery":
                current_recovery_streak += 1
            elif current_recovery_streak > 0:
                recovery_streaks.append(current_recovery_streak)
                current_recovery_streak = 0
            shift_counts[int(final_info.get("shifts_active", 1))] += 1
            op10_mult, op12_mult = extract_downstream_multipliers(final_info)
            op10_multipliers.append(op10_mult)
            op12_multipliers.append(op12_mult)
            steps += 1
            episode_start = np.array([terminated or truncated], dtype=bool)

        if current_recovery_streak > 0:
            recovery_streaks.append(current_recovery_streak)
        rows.append(
            finalize_episode_row(
                reward_mode=reward_mode,
                policy=bundle.policy,
                algo=bundle.algo,
                seed=seed,
                episode=episode_idx + 1,
                eval_seed=eval_seed,
                steps=steps,
                reward_total=reward_total,
                ret_thesis_total=ret_thesis_total,
                ret_thesis_corrected_total=ret_thesis_corrected_total,
                ret_seq_total=ret_seq_total,
                ret_unified_total=ret_unified_total,
                ret_garrido2024_raw_total=ret_garrido2024_raw_total,
                ret_garrido2024_train_total=ret_garrido2024_train_total,
                ret_garrido2024_sigmoid_total=ret_garrido2024_sigmoid_total,
                delivered_total=delivered_total,
                produced_total=produced_total,
                demanded_total=demanded_total,
                backorder_qty_total=backorder_qty_total,
                disruption_hours_total=disruption_hours_total,
                inventory_total_sum=inventory_total_sum,
                step_cost_total=step_cost_total,
                service_continuity_total=service_continuity_total,
                backlog_containment_total=backlog_containment_total,
                adaptive_efficiency_total=adaptive_efficiency_total,
                service_loss_area=service_loss_area,
                service_loss_area_below_095=service_loss_area_below_095,
                recovery_streaks=recovery_streaks,
                shift_counts=shift_counts,
                ret_case_counts=ret_case_counts,
                op10_multipliers=op10_multipliers,
                op12_multipliers=op12_multipliers,
                final_info=final_info,
                terminal_metrics=get_episode_terminal_metrics(env),
                step_size_hours=float(args.step_size_hours),
            )
        )
        env.close()
    vec_norm.close()
    return rows


def aggregate_policy_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["reward_mode"]), str(row["policy"]), str(row["algo"]))
        grouped.setdefault(key, []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for (reward_mode, policy, algo), group in sorted(grouped.items()):
        out_row: dict[str, Any] = {
            "reward_mode": reward_mode,
            "policy": policy,
            "algo": algo,
            "episodes": len(group),
            "seed_count": len({int(row["seed"]) for row in group}),
        }
        for metric in AUDIT_METRICS:
            values = [float(row[metric]) for row in group]
            ci_low, ci_high = ci95(values)
            out_row[f"{metric}_mean"] = float(statistics.fmean(values))
            out_row[f"{metric}_std"] = (
                float(statistics.stdev(values)) if len(values) > 1 else 0.0
            )
            out_row[f"{metric}_ci95_low"] = ci_low
            out_row[f"{metric}_ci95_high"] = ci_high
        summary_rows.append(out_row)
    return summary_rows


def rank_policy_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in summary_rows:
        grouped.setdefault(str(row["reward_mode"]), []).append(row)

    leaderboard: list[dict[str, Any]] = []
    for reward_mode, rows in sorted(grouped.items()):
        ranked = sorted(
            rows,
            key=lambda row: (
                float(row["fill_rate_mean"]),
                float(row["order_level_ret_mean_mean"]),
                float(row["service_continuity_step_mean_mean"]),
                -float(row["backorder_rate_mean"]),
                float(row["adaptive_efficiency_step_mean_mean"]),
                float(row["ret_garrido2024_sigmoid_total_mean"]),
            ),
            reverse=True,
        )
        for idx, row in enumerate(ranked, start=1):
            leaderboard.append(
                {
                    "reward_mode": reward_mode,
                    "rank": idx,
                    "policy": row["policy"],
                    "algo": row["algo"],
                    "fill_rate_mean": float(row["fill_rate_mean"]),
                    "backorder_rate_mean": float(row["backorder_rate_mean"]),
                    "order_level_ret_mean_mean": float(
                        row["order_level_ret_mean_mean"]
                    ),
                    "service_continuity_step_mean_mean": float(
                        row["service_continuity_step_mean_mean"]
                    ),
                    "adaptive_efficiency_step_mean_mean": float(
                        row["adaptive_efficiency_step_mean_mean"]
                    ),
                    "ret_garrido2024_sigmoid_total_mean": float(
                        row["ret_garrido2024_sigmoid_total_mean"]
                    ),
                    "reward_total_mean": float(row["reward_total_mean"]),
                }
            )
    return leaderboard


def select_reference_policy_rows(
    policy_rows: list[dict[str, Any]],
    *,
    reward_mode: str = PAPER_REFERENCE_REWARD_MODE,
) -> tuple[str, list[dict[str, Any]]]:
    available_modes = {str(row["reward_mode"]) for row in policy_rows}
    selected_mode = reward_mode if reward_mode in available_modes else sorted(available_modes)[0]
    return selected_mode, [
        row for row in policy_rows if str(row["reward_mode"]) == selected_mode
    ]


def build_summary(
    *,
    args: argparse.Namespace,
    bundles: list[LearnedBundle],
    episode_rows: list[dict[str, Any]],
    policy_rows: list[dict[str, Any]],
    leaderboard: list[dict[str, Any]],
    pairwise_stats: list[dict[str, Any]],
    paper_main_table: list[dict[str, Any]],
    paper_mechanism_table: list[dict[str, Any]],
    paper_raw_table: list[dict[str, Any]],
    paper_control_table: list[dict[str, Any]],
    paper_main_table_manuscript: list[dict[str, Any]],
    paper_reference_reward_mode: str,
    output_dir: Path,
) -> dict[str, Any]:
    return {
        "config": {
            "reward_modes": list(args.reward_modes),
            "seeds": [int(seed) for seed in args.seeds],
            "eval_episodes": int(args.eval_episodes),
            "risk_level": str(args.risk_level),
            "step_size_hours": float(args.step_size_hours),
            "max_steps": int(args.max_steps),
            "ppo_bundle": str(args.ppo_bundle.resolve()),
            "recurrent_bundle": (
                None if args.skip_recurrent else str(args.recurrent_bundle.resolve())
            ),
        },
        "benchmark_philosophy": {
            "training_reward_vs_audit": (
                "Policies remain fixed. Only the evaluation reward lens changes, "
                "so cross-policy comparisons stay fair under the same DES dynamics."
            ),
            "cross_mode_reward_comparison_allowed": False,
            "cross_mode_primary_metrics": [
                "fill_rate",
                "backorder_rate",
                "order_level_ret_mean",
                "delivered_total",
                "avg_annual_delivery",
                "service_continuity_step_mean",
                "backlog_containment_step_mean",
                "adaptive_efficiency_step_mean",
                "ret_seq_total",
                "ret_garrido2024_sigmoid_total",
            ],
            "thesis_faithful_metric": (
                "order_level_ret_mean is the paper-facing Garrido-Rios (2017) "
                "order-level aggregate built from Eq. 5.1-5.5 cases."
            ),
            "step_level_thesis_proxy_warning": (
                "ret_thesis_total and ret_thesis_corrected_total are repository "
                "step-level audit proxies. They are retained for diagnostics but "
                "should not be presented as the thesis-faithful primary metric."
            ),
        },
        "learned_bundles": [
            {
                "policy": bundle.policy,
                "algo": bundle.algo,
                "bundle_dir": str(bundle.bundle_dir.resolve()),
            }
            for bundle in bundles
        ],
        "episode_metrics": episode_rows,
        "policy_summary": policy_rows,
        "leaderboard": leaderboard,
        "pairwise_stats": pairwise_stats,
        "paper_reference_reward_mode": paper_reference_reward_mode,
        "paper_main_table": paper_main_table,
        "paper_mechanism_table": paper_mechanism_table,
        "paper_raw_table": paper_raw_table,
        "paper_control_table": paper_control_table,
        "paper_main_table_manuscript": paper_main_table_manuscript,
        "paper_table_notes": {
            "main_table": (
                "Uses order_level_ret_mean as the thesis-faithful Garrido-Rios "
                "(2017) resilience aggregate."
            ),
            "mechanism_table": (
                "thesis_step_proxy_total columns are step-level repository "
                "proxies kept only for internal diagnosis and reward-audit "
                "analysis."
            ),
            "control_table": (
                "Downstream control telemetry documents whether the learned "
                "policy saturates Op10/Op12 dispatch or modulates them."
            ),
        },
        "artifacts": {
            "output_dir": str(output_dir.resolve()),
            "episode_metrics_csv": str((output_dir / "episode_metrics.csv").resolve()),
            "policy_summary_csv": str((output_dir / "policy_summary.csv").resolve()),
            "leaderboard_csv": str((output_dir / "leaderboard.csv").resolve()),
            "pairwise_stats_csv": str((output_dir / "pairwise_stats.csv").resolve()),
            "paper_main_table_csv": str((output_dir / "paper_main_table.csv").resolve()),
            "paper_mechanism_table_csv": str(
                (output_dir / "paper_mechanism_table.csv").resolve()
            ),
            "paper_raw_table_csv": str((output_dir / "paper_raw_table.csv").resolve()),
            "paper_control_table_csv": str(
                (output_dir / "paper_control_table.csv").resolve()
            ),
            "paper_main_table_manuscript_csv": str(
                (output_dir / "paper_main_table_manuscript.csv").resolve()
            ),
            "paper_main_table_manuscript_md": str(
                (output_dir / "paper_main_table_manuscript.md").resolve()
            ),
            "summary_json": str((output_dir / "summary.json").resolve()),
        },
    }


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir or default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    bundles = build_learned_bundles(args)
    for bundle in bundles:
        validate_bundle(bundle, list(args.seeds))

    episode_rows: list[dict[str, Any]] = []
    for reward_mode in args.reward_modes:
        for seed in args.seeds:
            for static_policy in STATIC_POLICY_SPECS:
                episode_rows.extend(
                    evaluate_static_policy(
                        args=args,
                        reward_mode=str(reward_mode),
                        policy=static_policy,
                        seed=int(seed),
                    )
                )
            for bundle in bundles:
                episode_rows.extend(
                    evaluate_learned_policy(
                        args=args,
                        reward_mode=str(reward_mode),
                        bundle=bundle,
                        seed=int(seed),
                    )
                )

    policy_rows = aggregate_policy_rows(episode_rows)
    leaderboard = rank_policy_rows(policy_rows)
    paper_reference_reward_mode, reference_rows = select_reference_policy_rows(policy_rows)
    paper_main_table = build_paper_table(
        reference_rows,
        spec=PAPER_MAIN_TABLE_SPEC,
    )
    paper_mechanism_table = build_paper_table(
        reference_rows,
        spec=PAPER_MECHANISM_TABLE_SPEC,
    )
    paper_raw_table = build_paper_table(
        reference_rows,
        spec=PAPER_RAW_TABLE_SPEC,
    )
    paper_control_table = build_paper_table(
        reference_rows,
        spec=PAPER_CONTROL_TABLE_SPEC,
    )
    paper_main_table_manuscript = build_manuscript_main_table(reference_rows)
    pairwise_stats = build_pairwise_stats(
        episode_rows,
        reward_mode=paper_reference_reward_mode,
    )
    summary = build_summary(
        args=args,
        bundles=bundles,
        episode_rows=episode_rows,
        policy_rows=policy_rows,
        leaderboard=leaderboard,
        pairwise_stats=pairwise_stats,
        paper_main_table=paper_main_table,
        paper_mechanism_table=paper_mechanism_table,
        paper_raw_table=paper_raw_table,
        paper_control_table=paper_control_table,
        paper_main_table_manuscript=paper_main_table_manuscript,
        paper_reference_reward_mode=paper_reference_reward_mode,
        output_dir=output_dir,
    )

    save_csv(output_dir / "episode_metrics.csv", episode_rows)
    save_csv(output_dir / "policy_summary.csv", policy_rows)
    save_csv(output_dir / "leaderboard.csv", leaderboard)
    save_csv(output_dir / "pairwise_stats.csv", pairwise_stats)
    save_csv(output_dir / "paper_main_table.csv", paper_main_table)
    save_csv(output_dir / "paper_mechanism_table.csv", paper_mechanism_table)
    save_csv(output_dir / "paper_raw_table.csv", paper_raw_table)
    save_csv(output_dir / "paper_control_table.csv", paper_control_table)
    save_csv(
        output_dir / "paper_main_table_manuscript.csv",
        paper_main_table_manuscript,
    )
    (output_dir / "paper_main_table_manuscript.md").write_text(
        render_markdown_table(paper_main_table_manuscript),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    args = build_parser().parse_args()
    summary = run_audit(args)
    print(f"Wrote Track B all-reward audit bundle to {summary['artifacts']['output_dir']}")
    for row in summary["leaderboard"][: min(10, len(summary["leaderboard"]))]:
        print(
            f"{row['reward_mode']} | rank={row['rank']} | {row['policy']} "
            f"| fill={row['fill_rate_mean']:.4f} | ret={row['order_level_ret_mean_mean']:.4f}"
        )


if __name__ == "__main__":
    main()
