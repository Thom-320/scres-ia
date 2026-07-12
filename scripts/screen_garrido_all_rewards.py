#!/usr/bin/env python3
"""Screen every Track-A reward mode before a larger Garrido PPO run.

This is a deliberately small, comparable screen.  It trains PPO under each
reward mode, then evaluates every policy under the same external panel:

* primary resilience: Garrido 2017 Excel ReT
* cost-aware composite: Garrido 2024 C-D sigmoid
* service/resource metrics from ``compare_garrido_dynamic_vs_static``

The output separates two questions that otherwise get conflated:

1. Can PPO beat the frozen efficient static frontier?
2. Can PPO dominate the high-resource Garrido-style ``S3_I1344`` baseline?
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import traceback
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.compare_garrido_dynamic_vs_static import (  # noqa: E402
    build_parser as build_compare_parser,
    run as run_compare,
)
from supply_chain.env_experimental_shifts import REWARD_MODE_OPTIONS  # noqa: E402


DEFAULT_OUTPUT_DIR = Path("outputs/benchmarks/garrido_reward_screens")
DEFAULT_LANES = {
    "faithful": {
        "risk_frequency_multiplier": 1.0,
        "risk_impact_multiplier": 1.0,
        "stochastic_pt": False,
    },
    "headroom_freq1_5": {
        "risk_frequency_multiplier": 1.5,
        "risk_impact_multiplier": 1.0,
        "stochastic_pt": False,
    },
    "envb_impact1_25": {
        "risk_frequency_multiplier": 1.0,
        "risk_impact_multiplier": 1.25,
        "stochastic_pt": False,
    },
    "envb_freq2_impact1_5": {
        "risk_frequency_multiplier": 2.0,
        "risk_impact_multiplier": 1.5,
        "stochastic_pt": False,
    },
}


def _csv_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) not in (None, "")]
    return sum(values) / len(values) if values else float("nan")


def _score_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "n_regime_comparisons": 0,
            "strict_wins": 0,
            "resource_pareto_wins": 0,
            "excel_win_count": 0,
            "cd_win_count": 0,
            "mean_delta_excel_ret": float("nan"),
            "mean_delta_cd_sigmoid": float("nan"),
            "mean_delta_resource": float("nan"),
            "mean_delta_flow_fill": float("nan"),
            "mean_delta_lost_rate": float("nan"),
            "mean_delta_cvar95": float("nan"),
        }
    return {
        "n_regime_comparisons": len(rows),
        "strict_wins": sum(
            1 for row in rows if bool(row.get("strict_service_resource_dominates"))
        ),
        "resource_pareto_wins": sum(
            1 for row in rows if bool(row.get("resource_pareto_dominates"))
        ),
        "excel_win_count": sum(1 for row in rows if float(row["delta_excel_ret"]) > 0.0),
        "cd_win_count": sum(
            1 for row in rows if float(row["delta_cd_sigmoid_mean"]) > 0.0
        ),
        "mean_delta_excel_ret": _mean(rows, "delta_excel_ret"),
        "mean_delta_cd_sigmoid": _mean(rows, "delta_cd_sigmoid_mean"),
        "mean_delta_resource": _mean(rows, "delta_resource_composite_total"),
        "mean_delta_flow_fill": _mean(rows, "delta_flow_fill_rate"),
        "mean_delta_lost_rate": _mean(rows, "delta_lost_rate"),
        "mean_delta_cvar95": _mean(rows, "delta_service_loss_cvar95"),
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


def _compare_args(
    *,
    output_dir: Path,
    label: str,
    reward_mode: str,
    regimes: str,
    seeds: str,
    eval_episodes: int,
    max_steps: int,
    train_timesteps: int,
    algo: str,
    risk_frequency_multiplier: float,
    risk_impact_multiplier: float,
    stochastic_pt: bool,
    w_bo: float,
    w_cost: float,
    w_disr: float,
    control_v2_w_fill: float,
    control_v2_w_service: float,
    control_v2_w_lost: float,
    control_v2_w_inventory: float,
    control_v2_w_shift: float,
    control_v2_w_switch: float,
) -> argparse.Namespace:
    argv = [
        "--output-dir",
        str(output_dir),
        "--label",
        label,
        "--reward-mode",
        reward_mode,
        "--regimes",
        regimes,
        "--seeds",
        seeds,
        "--eval-episodes",
        str(eval_episodes),
        "--max-steps",
        str(max_steps),
        "--train-timesteps",
        str(train_timesteps),
        "--algo",
        algo,
        "--risk-frequency-multiplier",
        str(risk_frequency_multiplier),
        "--risk-impact-multiplier",
        str(risk_impact_multiplier),
        "--w-bo",
        str(w_bo),
        "--w-cost",
        str(w_cost),
        "--w-disr",
        str(w_disr),
        "--control-v2-w-fill",
        str(control_v2_w_fill),
        "--control-v2-w-service",
        str(control_v2_w_service),
        "--control-v2-w-lost",
        str(control_v2_w_lost),
        "--control-v2-w-inventory",
        str(control_v2_w_inventory),
        "--control-v2-w-shift",
        str(control_v2_w_shift),
        "--control-v2-w-switch",
        str(control_v2_w_switch),
    ]
    if stochastic_pt:
        argv.append("--stochastic-pt")
    return build_compare_parser().parse_args(argv)


def _comparison_subset(
    summary: dict[str, Any], *, target: str
) -> list[dict[str, Any]]:
    rows = []
    for row in summary.get("comparison_table", []):
        if row.get("dynamic_policy") != "ppo_dynamic":
            continue
        if target == "frozen_efficient":
            if bool(row.get("is_frozen_efficient_static")):
                rows.append(dict(row))
        elif row.get("static_policy") == target:
            rows.append(dict(row))
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--label",
        default=f"all_rewards_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
    )
    parser.add_argument(
        "--reward-modes",
        default=",".join(REWARD_MODE_OPTIONS),
        help="Comma-separated reward modes. Defaults to every registered Track-A mode.",
    )
    parser.add_argument(
        "--lanes",
        default="faithful,headroom_freq1_5",
        help=(
            "Comma-separated lanes: faithful, headroom_freq1_5, "
            "envb_impact1_25, envb_freq2_impact1_5."
        ),
    )
    parser.add_argument("--regimes", default="current,increased,severe")
    parser.add_argument("--seeds", default="8201")
    parser.add_argument("--eval-episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=52)
    parser.add_argument("--train-timesteps", type=int, default=2048)
    parser.add_argument("--algo", choices=("ppo", "recurrent_ppo"), default="ppo")
    parser.add_argument("--w-bo", type=float, default=4.0)
    parser.add_argument("--w-cost", type=float, default=0.02)
    parser.add_argument("--w-disr", type=float, default=0.0)
    parser.add_argument("--control-v2-w-fill", type=float, default=1.0)
    parser.add_argument("--control-v2-w-service", type=float, default=4.0)
    parser.add_argument("--control-v2-w-lost", type=float, default=2.0)
    parser.add_argument("--control-v2-w-inventory", type=float, default=0.05)
    parser.add_argument("--control-v2-w-shift", type=float, default=0.08)
    parser.add_argument("--control-v2-w-switch", type=float, default=0.02)
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        default=True,
        help="Record reward-mode failures and continue the screen.",
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    reward_modes = _csv_list(args.reward_modes)
    lanes = _csv_list(args.lanes)
    run_dir = args.output_dir / args.label
    runs_dir = run_dir / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)

    score_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    for lane in lanes:
        if lane not in DEFAULT_LANES:
            raise ValueError(f"Unknown lane {lane!r}; expected one of {sorted(DEFAULT_LANES)}")
        lane_cfg = DEFAULT_LANES[lane]
        for reward_mode in reward_modes:
            label = f"{lane}_{reward_mode}"
            print(f"=== {label} ===", flush=True)
            try:
                summary = run_compare(
                    _compare_args(
                        output_dir=runs_dir,
                        label=label,
                        reward_mode=reward_mode,
                        regimes=args.regimes,
                        seeds=args.seeds,
                        eval_episodes=int(args.eval_episodes),
                        max_steps=int(args.max_steps),
                        train_timesteps=int(args.train_timesteps),
                        algo=str(args.algo),
                        risk_frequency_multiplier=float(
                            lane_cfg["risk_frequency_multiplier"]
                        ),
                        risk_impact_multiplier=float(lane_cfg["risk_impact_multiplier"]),
                        stochastic_pt=bool(lane_cfg["stochastic_pt"]),
                        w_bo=float(args.w_bo),
                        w_cost=float(args.w_cost),
                        w_disr=float(args.w_disr),
                        control_v2_w_fill=float(args.control_v2_w_fill),
                        control_v2_w_service=float(args.control_v2_w_service),
                        control_v2_w_lost=float(args.control_v2_w_lost),
                        control_v2_w_inventory=float(args.control_v2_w_inventory),
                        control_v2_w_shift=float(args.control_v2_w_shift),
                        control_v2_w_switch=float(args.control_v2_w_switch),
                    )
                )
            except Exception as exc:  # pragma: no cover - exercised in real screens.
                if not bool(args.continue_on_error):
                    raise
                failure_rows.append(
                    {
                        "lane": lane,
                        "reward_mode": reward_mode,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                continue

            for target in ("frozen_efficient", "static_S3_I1344", "original_S1_I0"):
                rows = _comparison_subset(summary, target=target)
                score_rows.append(
                    {
                        "lane": lane,
                        "reward_mode": reward_mode,
                        "target": target,
                        **_score_rows(rows),
                        "summary_json": summary["artifacts"]["summary_json"],
                    }
                )

    _write_csv(run_dir / "reward_screen_scores.csv", score_rows)
    _write_csv(run_dir / "reward_screen_failures.csv", failure_rows)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "description": "All registered Track-A reward modes screened on Garrido panel.",
        "config": {
            "reward_modes": reward_modes,
            "lanes": lanes,
            "regimes": args.regimes,
            "seeds": args.seeds,
            "eval_episodes": int(args.eval_episodes),
            "max_steps": int(args.max_steps),
            "train_timesteps": int(args.train_timesteps),
            "algo": str(args.algo),
            "w_bo": float(args.w_bo),
            "w_cost": float(args.w_cost),
            "w_disr": float(args.w_disr),
            "control_v2_w_fill": float(args.control_v2_w_fill),
            "control_v2_w_service": float(args.control_v2_w_service),
            "control_v2_w_lost": float(args.control_v2_w_lost),
            "control_v2_w_inventory": float(args.control_v2_w_inventory),
            "control_v2_w_shift": float(args.control_v2_w_shift),
            "control_v2_w_switch": float(args.control_v2_w_switch),
        },
        "lane_configs": DEFAULT_LANES,
        "artifacts": {
            "reward_screen_scores_csv": str(run_dir / "reward_screen_scores.csv"),
            "reward_screen_failures_csv": str(run_dir / "reward_screen_failures.csv"),
            "summary_json": str(run_dir / "summary.json"),
        },
        "scores": score_rows,
        "failures": failure_rows,
    }
    (run_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> int:
    summary = run(build_parser().parse_args())
    print(f"Wrote {summary['artifacts']['summary_json']}")
    print(f"Failures: {len(summary['failures'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
