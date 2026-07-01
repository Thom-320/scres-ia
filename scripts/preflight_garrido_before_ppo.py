#!/usr/bin/env python3
"""Prepare the Garrido-faithful DES lane before any large PPO run.

This preflight keeps two claims separate:

* ``faithful``: thesis/Excel-like Track A, no risk tuning, deterministic PT.
* ``headroom``: justified stress extension, screened before PPO.

The script reuses ``compare_garrido_dynamic_vs_static.py`` so the same metric
panel is used for static baselines, heuristics, reward screening, and later PPO.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.compare_garrido_dynamic_vs_static import (  # noqa: E402
    FROZEN_STATIC_BY_REGIME,
    build_parser as build_compare_parser,
    run as run_compare,
)
from supply_chain.config import (  # noqa: E402
    BACKORDER_QUEUE_CAP,
    RET_RECOVERY_PERIOD_MODE,
    BACKORDER_OVERFLOW_MODE,
    THESIS_FAITHFUL_PROTOCOL,
)


DEFAULT_OUTPUT_DIR = Path("outputs/preflight/garrido_before_ppo")
REWARD_CANDIDATES = (
    "ReT_garrido2024_raw",
    "ReT_garrido2024",
    "ReT_excel_delta",
    "control_v1",
)
HEADROOM_FREQS = (1.5, 2.0, 3.0)
HEADROOM_IMPACTS = (1.25, 1.5, 2.0)


def _copy_if_exists(src: str | Path, dst: Path) -> None:
    source = Path(src)
    if source.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dst)


def _compare_args(
    *,
    output_dir: Path,
    label: str,
    regimes: str,
    seeds: str,
    eval_episodes: int,
    max_steps: int,
    reward_mode: str,
    train_timesteps: int,
    skip_ppo: bool,
    include_threshold_heuristics: bool,
    stochastic_pt: bool,
    risk_frequency_multiplier: float,
    risk_impact_multiplier: float,
) -> argparse.Namespace:
    argv = [
        "--output-dir",
        str(output_dir),
        "--label",
        label,
        "--regimes",
        regimes,
        "--seeds",
        seeds,
        "--eval-episodes",
        str(eval_episodes),
        "--max-steps",
        str(max_steps),
        "--reward-mode",
        reward_mode,
        "--train-timesteps",
        str(train_timesteps),
        "--risk-frequency-multiplier",
        str(risk_frequency_multiplier),
        "--risk-impact-multiplier",
        str(risk_impact_multiplier),
    ]
    if skip_ppo:
        argv.append("--skip-ppo")
    if include_threshold_heuristics:
        argv.append("--include-threshold-heuristics")
    if stochastic_pt:
        argv.append("--stochastic-pt")
    return build_compare_parser().parse_args(argv)


def _frozen_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in summary.get("comparison_table", []):
        if bool(row.get("is_frozen_efficient_static")):
            rows.append(dict(row))
    return rows


def _screen_score(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "n_frozen_comparisons": 0,
            "strict_wins": 0,
            "resource_pareto_wins": 0,
            "mean_delta_cd_sigmoid": float("nan"),
            "mean_delta_excel_ret": float("nan"),
            "mean_delta_resource_composite": float("nan"),
        }
    return {
        "n_frozen_comparisons": len(rows),
        "strict_wins": sum(
            1 for row in rows if bool(row.get("strict_service_resource_dominates"))
        ),
        "resource_pareto_wins": sum(
            1 for row in rows if bool(row.get("resource_pareto_dominates"))
        ),
        "mean_delta_cd_sigmoid": sum(
            float(row.get("delta_cd_sigmoid_mean", 0.0)) for row in rows
        )
        / len(rows),
        "mean_delta_excel_ret": sum(
            float(row.get("delta_excel_ret", 0.0)) for row in rows
        )
        / len(rows),
        "mean_delta_resource_composite": sum(
            float(row.get("delta_resource_composite_total", 0.0)) for row in rows
        )
        / len(rows),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_readme(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Garrido Before-PPO Preflight",
        "",
        "This directory is the gate before any large PPO training run.",
        "",
        "## Claim Boundary",
        "",
        "- **Faithful lane:** thesis/Excel-like DES. No risk tuning, deterministic processing times, Garrido fulfillment delay, capped backlog ledger.",
        "- **Headroom lane:** justified stress extension. Risk/PT changes are sensitivity experiments, not the 1:1 replication claim.",
        "",
        "## Metrics",
        "",
        "- Primary resilience metric: `mean_ret_excel_formula`.",
        "- Secondary resilience metrics: `mean_ret_text_formula`, `cd_sigmoid_mean`, `cd_raw_mean`, `fill_rate_order_level`.",
        "- Training reward prior: `ReT_garrido2024_raw`.",
        "- Direct Excel reward: screened as ablation (`ReT_excel_delta`), not the default.",
        "",
        "## Artifacts",
        "",
    ]
    for name, artifact_path in payload["artifacts"].items():
        lines.append(f"- `{name}`: `{artifact_path}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _freeze_summary(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ready_for_pre_ppo_screens",
        "claim_boundary": {
            "faithful_lane": (
                "No risk tuning; thesis-window risks, deterministic processing "
                "times, Garrido fulfillment delay, capped backlog ledger."
            ),
            "headroom_lane": (
                "Risk/PT extensions are sensitivity/headroom experiments, not "
                "the 1:1 Garrido replication claim."
            ),
        },
        "primary_resilience_metric": "mean_ret_excel_formula",
        "secondary_resilience_metrics": [
            "mean_ret_text_formula",
            "cd_sigmoid_mean",
            "cd_raw_mean",
            "fill_rate_order_level",
        ],
        "reward_note": (
            "Training rewards may differ from resilience metrics; Excel ReT remains "
            "the primary evaluation bar."
        ),
        "faithful_protocol": {
            **THESIS_FAITHFUL_PROTOCOL,
            "ret_recovery_period_mode": RET_RECOVERY_PERIOD_MODE,
            "backorder_overflow_mode": BACKORDER_OVERFLOW_MODE,
            "backorder_queue_cap": BACKORDER_QUEUE_CAP,
            "risk_frequency_multiplier": 1.0,
            "risk_impact_multiplier": 1.0,
        },
        "gates_to_keep_green": {
            "excel_formula_rows": 47_546,
            "excel_formula_mismatches_required": 0,
            "forensic_replay_mae_required_max": 0.005,
            "risk_frequency_status_required": "PASS",
            "tests_required": "full suite green",
        },
        "preflight_config": {
            "regimes": args.regimes,
            "seeds": args.seeds,
            "eval_episodes": int(args.eval_episodes),
            "max_steps": int(args.max_steps),
            "reward_train_timesteps": int(args.reward_train_timesteps),
            "skip_reward_screen": bool(args.skip_reward_screen),
            "skip_headroom_screen": bool(args.skip_headroom_screen),
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = args.output_dir / args.label
    root.mkdir(parents=True, exist_ok=True)

    freeze = _freeze_summary(args)
    (root / "garrido_des_freeze_summary.json").write_text(
        json.dumps(freeze, indent=2),
        encoding="utf-8",
    )

    static_summary = run_compare(
        _compare_args(
            output_dir=root / "runs",
            label="static_baseline_panel",
            regimes=args.regimes,
            seeds=args.seeds,
            eval_episodes=int(args.eval_episodes),
            max_steps=int(args.max_steps),
            reward_mode="ReT_garrido2024_raw",
            train_timesteps=0,
            skip_ppo=True,
            include_threshold_heuristics=True,
            stochastic_pt=False,
            risk_frequency_multiplier=1.0,
            risk_impact_multiplier=1.0,
        )
    )
    _copy_if_exists(
        static_summary["artifacts"]["episode_metrics_csv"],
        root / "metric_panel_by_seed.csv",
    )
    _copy_if_exists(
        static_summary["artifacts"]["policy_summary_csv"],
        root / "static_baseline_panel.csv",
    )
    _copy_if_exists(
        static_summary["artifacts"]["best_static_by_metric_csv"],
        root / "best_static_by_metric.csv",
    )

    reward_rows: list[dict[str, Any]] = []
    if not bool(args.skip_reward_screen):
        for reward_mode in REWARD_CANDIDATES:
            summary = run_compare(
                _compare_args(
                    output_dir=root / "runs",
                    label=f"reward_screen_{reward_mode}",
                    regimes=args.regimes,
                    seeds=args.seeds,
                    eval_episodes=int(args.eval_episodes),
                    max_steps=int(args.max_steps),
                    reward_mode=reward_mode,
                    train_timesteps=int(args.reward_train_timesteps),
                    skip_ppo=False,
                    include_threshold_heuristics=False,
                    stochastic_pt=False,
                    risk_frequency_multiplier=1.0,
                    risk_impact_multiplier=1.0,
                )
            )
            score = _screen_score(_frozen_rows(summary))
            reward_rows.append(
                {
                    "lane": "faithful",
                    "reward_mode": reward_mode,
                    **score,
                    "summary_json": summary["artifacts"]["summary_json"],
                }
            )
    _write_csv(root / "reward_screen_summary.csv", reward_rows)

    headroom_rows: list[dict[str, Any]] = []
    if not bool(args.skip_headroom_screen):
        cells: list[tuple[str, float, float, bool]] = [
            ("faithful_base", 1.0, 1.0, False),
            *[
                (f"freq_{freq:g}", freq, 1.0, False)
                for freq in HEADROOM_FREQS
            ],
            *[
                (f"impact_{impact:g}", 1.0, impact, False)
                for impact in HEADROOM_IMPACTS
            ],
            ("stochastic_pt", 1.0, 1.0, True),
        ]
        for cell_name, freq, impact, stochastic_pt in cells:
            summary = run_compare(
                _compare_args(
                    output_dir=root / "runs",
                    label=f"headroom_{cell_name}",
                    regimes=args.regimes,
                    seeds=args.seeds,
                    eval_episodes=int(args.eval_episodes),
                    max_steps=int(args.max_steps),
                    reward_mode="ReT_garrido2024_raw",
                    train_timesteps=0,
                    skip_ppo=True,
                    include_threshold_heuristics=True,
                    stochastic_pt=stochastic_pt,
                    risk_frequency_multiplier=freq,
                    risk_impact_multiplier=impact,
                )
            )
            score = _screen_score(_frozen_rows(summary))
            headroom_rows.append(
                {
                    "lane": "headroom",
                    "cell": cell_name,
                    "risk_frequency_multiplier": freq,
                    "risk_impact_multiplier": impact,
                    "stochastic_pt": stochastic_pt,
                    **score,
                    "summary_json": summary["artifacts"]["summary_json"],
                }
            )
    _write_csv(root / "headroom_screen_summary.csv", headroom_rows)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "preflight_complete",
        "primary_resilience_metric": "mean_ret_excel_formula",
        "secondary_resilience_metrics": [
            "mean_ret_text_formula",
            "cd_sigmoid_mean",
            "cd_raw_mean",
            "fill_rate_order_level",
        ],
        "primary_training_reward_prior": "ReT_garrido2024_raw",
        "excel_reward_policy": "screened_ablation_not_default",
        "freeze_summary": freeze,
        "static_summary_json": static_summary["artifacts"]["summary_json"],
        "reward_screen_rows": reward_rows,
        "headroom_screen_rows": headroom_rows,
        "artifacts": {
            "garrido_des_freeze_summary_json": str(
                root / "garrido_des_freeze_summary.json"
            ),
            "static_baseline_panel_csv": str(root / "static_baseline_panel.csv"),
            "best_static_by_metric_csv": str(root / "best_static_by_metric.csv"),
            "reward_screen_summary_csv": str(root / "reward_screen_summary.csv"),
            "headroom_screen_summary_csv": str(root / "headroom_screen_summary.csv"),
            "metric_panel_by_seed_csv": str(root / "metric_panel_by_seed.csv"),
            "summary_json": str(root / "summary.json"),
        },
    }
    (root / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_readme(root / "README.md", payload)
    payload["artifacts"]["readme_md"] = str(root / "README.md")
    (root / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--label", default="smoke")
    parser.add_argument("--regimes", default="current,increased,severe")
    parser.add_argument("--seeds", default="8201,8202,8203")
    parser.add_argument("--eval-episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=52)
    parser.add_argument("--reward-train-timesteps", type=int, default=512)
    parser.add_argument("--skip-reward-screen", action="store_true")
    parser.add_argument("--skip-headroom-screen", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = run(args)
    print(f"Wrote {summary['artifacts']['summary_json']}")
    print(f"Static panel: {summary['artifacts']['static_baseline_panel_csv']}")
    print(f"Reward screen: {summary['artifacts']['reward_screen_summary_csv']}")
    print(f"Headroom screen: {summary['artifacts']['headroom_screen_summary_csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
