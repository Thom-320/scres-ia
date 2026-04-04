#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.audit_track_b_all_rewards import (  # noqa: E402
    DEFAULT_EVAL_EPISODES,
    DEFAULT_MAX_STEPS,
    DEFAULT_PPO_BUNDLE,
    DEFAULT_RECURRENT_BUNDLE,
    DEFAULT_SEEDS,
    DEFAULT_STEP_SIZE_HOURS,
    PAPER_REFERENCE_REWARD_MODE,
    LearnedBundle,
    aggregate_policy_rows,
    evaluate_learned_policy,
    evaluate_static_policy,
    save_csv,
    validate_bundle,
)
from scripts.run_track_b_smoke import (  # noqa: E402
    STATIC_POLICY_SPECS,
    evaluate_heuristic_policy,
)
from scripts.track_b_heuristics import make_heuristic_defaults  # noqa: E402
from supply_chain.env_experimental_shifts import REWARD_MODE_OPTIONS  # noqa: E402

DEFAULT_RISK_LEVELS = ("current", "increased", "severe")

MEETING_POLICY_ORDER = (
    "ppo",
    "recurrent_ppo",
    "s1_d1.00",
    "s2_d1.00",
    "s2_d1.50",
    "s3_d2.00",
)

MEETING_POLICY_LABELS = {
    "ppo": "PPO",
    "recurrent_ppo": "RecurrentPPO",
    "s1_d1.00": "S1",
    "s2_d1.00": "S2",
    "s2_d1.50": "S2(d=1.5)",
    "s3_d2.00": "S3(d=2.0)",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate frozen Track B bundles across alternative risk levels "
            "without retraining, then emit meeting-friendly tables."
        )
    )
    parser.add_argument(
        "--ppo-bundle",
        type=Path,
        default=DEFAULT_PPO_BUNDLE,
        help="Frozen PPO bundle directory.",
    )
    parser.add_argument(
        "--recurrent-bundle",
        type=Path,
        default=DEFAULT_RECURRENT_BUNDLE,
        help="Frozen RecurrentPPO bundle directory.",
    )
    parser.add_argument(
        "--skip-recurrent",
        action="store_true",
        help="Evaluate only PPO and static policies.",
    )
    parser.add_argument(
        "--include-heuristics",
        action="store_true",
        help="Evaluate Track B heuristics alongside statics and learned bundles.",
    )
    parser.add_argument(
        "--reward-mode",
        choices=list(REWARD_MODE_OPTIONS),
        default=PAPER_REFERENCE_REWARD_MODE,
        help="Evaluation reward lens. Policies remain fixed.",
    )
    parser.add_argument(
        "--risk-levels",
        nargs="+",
        default=list(DEFAULT_RISK_LEVELS),
        help="Risk levels for frozen-policy cross-scenario evaluation.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help="Model seeds to evaluate.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=DEFAULT_EVAL_EPISODES,
        help="Evaluation episodes per policy, seed, and risk level.",
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
        help="Directory to write the cross-scenario bundle.",
    )
    return parser


def default_output_dir() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return (
        Path("outputs/track_b_benchmarks")
        / f"track_b_cross_scenario_{timestamp}"
    )


def build_bundles(args: argparse.Namespace) -> list[LearnedBundle]:
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


def build_meeting_table_rows(policy_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_policy = {str(row["policy"]): row for row in policy_rows}
    rows: list[dict[str, Any]] = []
    for policy in MEETING_POLICY_ORDER:
        row = by_policy.get(policy)
        if row is None:
            continue
        rows.append(
            {
                "policy": MEETING_POLICY_LABELS.get(policy, policy),
                "fill_rate": float(row["fill_rate_mean"]),
                "backorder_rate": float(row["backorder_rate_mean"]),
                "order_level_ret": float(row["order_level_ret_mean_mean"]),
                "service_continuity": float(
                    row["service_continuity_step_mean_mean"]
                ),
                "backlog_containment": float(
                    row["backlog_containment_step_mean_mean"]
                ),
                "adaptive_efficiency": float(
                    row["adaptive_efficiency_step_mean_mean"]
                ),
                "pct_fill_rate_only": float(row["pct_ret_case_fill_rate_only_mean"]),
                "pct_autotomy": float(row["pct_ret_case_autotomy_mean"]),
                "pct_recovery": float(row["pct_ret_case_recovery_mean"]),
                "pct_non_recovery": float(
                    row["pct_ret_case_non_recovery_mean"]
                ),
            }
        )
    return rows


def render_markdown_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "Policy",
        "Fill",
        "Backorder",
        "Order-level ReT",
        "SC",
        "BC",
        "AE",
        "% Fill-only",
        "% Autotomy",
        "% Recovery",
        "% Non-recovery",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append(
            "| {policy} | {fill:.4f} | {backorder:.4f} | {ret:.4f} | "
            "{sc:.4f} | {bc:.4f} | {ae:.4f} | {fill_only:.2f}% | "
            "{autotomy:.2f}% | {recovery:.2f}% | {non_recovery:.2f}% |".format(
                policy=row["policy"],
                fill=row["fill_rate"],
                backorder=row["backorder_rate"],
                ret=row["order_level_ret"],
                sc=row["service_continuity"],
                bc=row["backlog_containment"],
                ae=row["adaptive_efficiency"],
                fill_only=row["pct_fill_rate_only"],
                autotomy=row["pct_autotomy"],
                recovery=row["pct_recovery"],
                non_recovery=row["pct_non_recovery"],
            )
        )
    return "\n".join(lines)


def build_overview_row(risk_level: str, policy_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_policy = {str(row["policy"]): row for row in policy_rows}
    static_rows = [
        row for row in policy_rows if str(row["algo"]) == "static"
    ]
    best_static = max(
        static_rows,
        key=lambda row: (
            float(row["fill_rate_mean"]),
            float(row["order_level_ret_mean_mean"]),
            float(row["service_continuity_step_mean_mean"]),
            -float(row["backorder_rate_mean"]),
        ),
    )
    ppo = by_policy["ppo"]
    row = {
        "risk_level": risk_level,
        "best_static_policy": str(best_static["policy"]),
        "ppo_fill_rate": float(ppo["fill_rate_mean"]),
        "ppo_backorder_rate": float(ppo["backorder_rate_mean"]),
        "ppo_order_level_ret": float(ppo["order_level_ret_mean_mean"]),
        "ppo_service_continuity": float(ppo["service_continuity_step_mean_mean"]),
        "ppo_pct_recovery": float(ppo["pct_ret_case_recovery_mean"]),
        "best_static_fill_rate": float(best_static["fill_rate_mean"]),
        "best_static_backorder_rate": float(best_static["backorder_rate_mean"]),
        "best_static_order_level_ret": float(
            best_static["order_level_ret_mean_mean"]
        ),
        "best_static_service_continuity": float(
            best_static["service_continuity_step_mean_mean"]
        ),
        "best_static_pct_recovery": float(best_static["pct_ret_case_recovery_mean"]),
        "ppo_fill_gap_vs_best_static_pp": 100.0
        * (
            float(ppo["fill_rate_mean"]) - float(best_static["fill_rate_mean"])
        ),
        "ppo_ret_gap_vs_best_static": float(ppo["order_level_ret_mean_mean"])
        - float(best_static["order_level_ret_mean_mean"]),
    }
    if "recurrent_ppo" in by_policy:
        rppo = by_policy["recurrent_ppo"]
        row.update(
            {
                "recurrent_fill_rate": float(rppo["fill_rate_mean"]),
                "recurrent_order_level_ret": float(
                    rppo["order_level_ret_mean_mean"]
                ),
                "ppo_ret_gap_vs_recurrent": float(ppo["order_level_ret_mean_mean"])
                - float(rppo["order_level_ret_mean_mean"]),
            }
        )
    return row


def run_cross_scenario(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir or default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    bundles = build_bundles(args)
    for bundle in bundles:
        validate_bundle(bundle, list(args.seeds))

    overview_rows: list[dict[str, Any]] = []
    artifacts: dict[str, Any] = {}

    for risk_level in args.risk_levels:
        risk_args = argparse.Namespace(**vars(args))
        risk_args.risk_level = risk_level
        episode_rows: list[dict[str, Any]] = []
        for seed in args.seeds:
            for policy in STATIC_POLICY_SPECS:
                episode_rows.extend(
                    evaluate_static_policy(
                        args=risk_args,
                        reward_mode=args.reward_mode,
                        policy=policy,
                        seed=int(seed),
                    )
                )
            if args.include_heuristics:
                for label, heuristic in make_heuristic_defaults().items():
                    episode_rows.extend(
                        evaluate_heuristic_policy(
                            label,
                            heuristic,
                            args=risk_args,
                            seed=int(seed),
                        )
                    )
            for bundle in bundles:
                episode_rows.extend(
                    evaluate_learned_policy(
                        args=risk_args,
                        reward_mode=args.reward_mode,
                        bundle=bundle,
                        seed=int(seed),
                    )
                )

        policy_rows = aggregate_policy_rows(episode_rows)
        meeting_rows = build_meeting_table_rows(policy_rows)
        overview_rows.append(build_overview_row(risk_level, policy_rows))

        risk_dir = output_dir / risk_level
        risk_dir.mkdir(parents=True, exist_ok=True)
        episode_csv = risk_dir / "episode_metrics.csv"
        policy_csv = risk_dir / "policy_summary.csv"
        meeting_csv = risk_dir / "meeting_table.csv"
        meeting_md = risk_dir / "meeting_table.md"
        save_csv(episode_csv, episode_rows)
        save_csv(policy_csv, policy_rows)
        save_csv(meeting_csv, meeting_rows)
        meeting_md.write_text(render_markdown_table(meeting_rows), encoding="utf-8")
        artifacts[risk_level] = {
            "episode_metrics_csv": str(episode_csv.resolve()),
            "policy_summary_csv": str(policy_csv.resolve()),
            "meeting_table_csv": str(meeting_csv.resolve()),
            "meeting_table_md": str(meeting_md.resolve()),
        }

    overview_csv = output_dir / "risk_overview.csv"
    save_csv(overview_csv, overview_rows)
    summary = {
        "config": {
            "reward_mode": args.reward_mode,
            "risk_levels": list(args.risk_levels),
            "seeds": [int(seed) for seed in args.seeds],
            "eval_episodes": int(args.eval_episodes),
            "include_heuristics": bool(args.include_heuristics),
            "step_size_hours": float(args.step_size_hours),
            "max_steps": int(args.max_steps),
            "ppo_bundle": str(args.ppo_bundle.resolve()),
            "recurrent_bundle": (
                None if args.skip_recurrent else str(args.recurrent_bundle.resolve())
            ),
        },
        "artifacts": {
            "risk_overview_csv": str(overview_csv.resolve()),
            "per_risk": artifacts,
        },
        "risk_overview": overview_rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    args = build_parser().parse_args()
    summary = run_cross_scenario(args)
    print(f"Wrote cross-scenario bundle to {args.output_dir or 'auto output dir'}")
    for row in summary["risk_overview"]:
        print(
            f"{row['risk_level']}: ppo_fill={row['ppo_fill_rate']:.4f}, "
            f"best_static={row['best_static_policy']}, "
            f"gap_pp={row['ppo_fill_gap_vs_best_static_pp']:+.2f}, "
            f"ppo_ret={row['ppo_order_level_ret']:.4f}"
        )


if __name__ == "__main__":
    main()
