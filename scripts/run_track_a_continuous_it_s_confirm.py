#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.run_thesis_decision_ppo_smoke as thesis_smoke  # noqa: E402
from scripts.run_track_a_exhaustion_sweep import (  # noqa: E402
    PT_PROFILES,
    REWARD_PROFILES,
)

DEFAULT_OUTPUT_ROOT = Path("outputs/benchmarks/track_a_continuous_it_s_confirm")


def utc_now_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_float_list(value: str) -> list[float]:
    values = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one numeric value.")
    return values


def parse_int_list(value: str) -> list[int]:
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one integer value.")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Confirm Track A continuous I_t,S by comparing PPO against a best "
            "static continuous buffer grid under the same faithful environment."
        )
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--label", default=None)
    parser.add_argument("--seeds", type=parse_int_list, default=[4242, 4243, 4244])
    parser.add_argument("--train-timesteps", type=int, default=40_000)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--risk-level", default="severe")
    parser.add_argument("--garrido-cfis", default="31")
    parser.add_argument(
        "--reward-profile", choices=sorted(REWARD_PROFILES), default="ret_ladder_steep"
    )
    parser.add_argument(
        "--pt-profile", choices=sorted(PT_PROFILES), default="stoch_pt_mean"
    )
    parser.add_argument(
        "--buffer-fractions", type=parse_float_list, default=[i / 10 for i in range(11)]
    )
    parser.add_argument("--shifts", type=parse_int_list, default=[1, 2, 3])
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument(
        "--policy-net-arch", choices=["small", "medium", "large"], default="medium"
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def build_smoke_args(
    args: argparse.Namespace, *, seed: int, run_root: Path
) -> argparse.Namespace:
    label = f"ppo_continuous_it_s_seed{seed}"
    argv = [
        "--label",
        label,
        "--output-root",
        str(run_root),
        "--train-timesteps",
        str(args.train_timesteps),
        "--eval-episodes",
        str(args.eval_episodes),
        "--seed",
        str(seed),
        "--risk-level",
        args.risk_level,
        "--garrido-cfis",
        args.garrido_cfis,
        "--risk-occurrence-mode",
        "thesis_periodic",
        "--raw-material-flow-mode",
        "kit_equivalent_order_up_to",
        "--raw-material-order-up-to-multiplier",
        "2.0",
        "--action-space-mode",
        "continuous_it_s",
        "--inventory-period-mode",
        "thesis_strict",
        "--max-steps",
        str(args.max_steps),
        "--no-include-static-grid",
        "--no-eval-ai-on-garrido-cfis",
        "--n-steps",
        str(args.n_steps),
        "--batch-size",
        str(args.batch_size),
        "--n-epochs",
        str(args.n_epochs),
        "--policy-net-arch",
        args.policy_net_arch,
        "--learning-rate",
        str(args.learning_rate),
        *REWARD_PROFILES[args.reward_profile],
        *PT_PROFILES[args.pt_profile],
    ]
    return thesis_smoke.build_parser().parse_args(argv)


def row_float(row: dict[str, str], key: str) -> float:
    return float(row.get(key) or 0.0)


def read_ppo_summary(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "policy_summary.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    ppo = next(row for row in rows if row["policy"] == "ppo_mlp")
    return {
        "ppo_fill": row_float(ppo, "fill_rate_order_level_mean"),
        "ppo_ret": row_float(ppo, "order_level_ret_mean"),
        "ppo_reward": row_float(ppo, "reward_total_mean"),
    }


def continuous_static_action(buffer_fraction: float, shifts: int) -> np.ndarray:
    return np.array(
        [
            np.clip(float(buffer_fraction), 0.0, 1.0),
            thesis_smoke.shift_signal_for(int(shifts)),
        ],
        dtype=np.float32,
    )


def evaluate_static_continuous_grid(
    *,
    smoke_args: argparse.Namespace,
    seed: int,
    buffer_fractions: list[float],
    shifts_values: list[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for buffer_fraction in buffer_fractions:
        for shifts in shifts_values:
            policy_name = f"static_continuous_b{buffer_fraction:.3f}_S{shifts}"
            rows.extend(
                thesis_smoke.evaluate_action_policy(
                    args=smoke_args,
                    policy_name=policy_name,
                    action_fn=lambda obs, info, b=buffer_fraction, s=shifts: continuous_static_action(
                        b, s
                    ),
                    seed=seed,
                    policy_metadata={
                        "baseline_family": "static_continuous_it_s_grid",
                        "buffer_fraction": float(buffer_fraction),
                        "shifts": int(shifts),
                    },
                )
            )
    aggregate_rows = thesis_smoke.aggregate(rows)
    best_static = max(
        aggregate_rows,
        key=lambda row: (
            float(row["fill_rate_order_level_mean"]),
            float(row["order_level_ret_mean"]),
        ),
    )
    return rows, aggregate_rows, best_static


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_seed(args: argparse.Namespace, *, seed: int, run_root: Path) -> dict[str, Any]:
    smoke_args = build_smoke_args(args, seed=seed, run_root=run_root)
    ppo_run_dir = run_root / f"ppo_continuous_it_s_seed{seed}"
    if args.dry_run:
        return {
            "seed": seed,
            "status": "dry_run",
            "ppo_run_dir": str(ppo_run_dir),
        }

    if not (args.skip_existing and (ppo_run_dir / "policy_summary.csv").exists()):
        ppo_log_path = run_root / f"ppo_continuous_it_s_seed{seed}.log"
        with ppo_log_path.open("w", encoding="utf-8") as log_handle:
            with contextlib.redirect_stdout(log_handle):
                thesis_smoke.run_single(smoke_args, ppo_run_dir)

    static_rows, static_summary, best_static = evaluate_static_continuous_grid(
        smoke_args=smoke_args,
        seed=seed,
        buffer_fractions=args.buffer_fractions,
        shifts_values=args.shifts,
    )
    seed_dir = run_root / f"continuous_static_grid_seed{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    write_csv(seed_dir / "episode_metrics.csv", static_rows)
    write_csv(seed_dir / "policy_summary.csv", static_summary)

    ppo_summary = read_ppo_summary(ppo_run_dir)
    best_fill = float(best_static["fill_rate_order_level_mean"])
    best_ret = float(best_static["order_level_ret_mean"])
    best_reward = float(best_static["reward_total_mean"])
    return {
        "seed": seed,
        "status": "complete",
        "ppo_run_dir": str(ppo_run_dir),
        "static_grid_dir": str(seed_dir),
        "best_static_policy": best_static["policy"],
        "best_static_fill": best_fill,
        "best_static_ret": best_ret,
        "best_static_reward": best_reward,
        **ppo_summary,
        "delta_fill": ppo_summary["ppo_fill"] - best_fill,
        "delta_ret": ppo_summary["ppo_ret"] - best_ret,
        "delta_reward": ppo_summary["ppo_reward"] - best_reward,
    }


def build_overall_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    complete = [row for row in rows if row.get("status") == "complete"]
    if not complete:
        return {"status": "no_complete_rows"}
    delta_fill = [float(row["delta_fill"]) for row in complete]
    delta_ret = [float(row["delta_ret"]) for row in complete]
    return {
        "status": "complete",
        "seed_count": len(complete),
        "positive_fill_seeds": int(sum(delta > 0 for delta in delta_fill)),
        "positive_ret_seeds": int(sum(delta > 0 for delta in delta_ret)),
        "mean_delta_fill": float(np.mean(delta_fill)),
        "mean_delta_ret": float(np.mean(delta_ret)),
        "min_delta_fill": float(np.min(delta_fill)),
        "max_delta_fill": float(np.max(delta_fill)),
        "min_delta_ret": float(np.min(delta_ret)),
        "max_delta_ret": float(np.max(delta_ret)),
    }


def main() -> int:
    args = build_parser().parse_args()
    label = args.label or f"continuous_it_s_confirm_{utc_now_slug()}"
    out_dir = args.output_root / label
    run_root = out_dir / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)

    config = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seeds": args.seeds,
        "train_timesteps": args.train_timesteps,
        "eval_episodes": args.eval_episodes,
        "risk_level": args.risk_level,
        "garrido_cfis": args.garrido_cfis,
        "reward_profile": args.reward_profile,
        "pt_profile": args.pt_profile,
        "buffer_fractions": args.buffer_fractions,
        "shifts": args.shifts,
        "fixed_modes": {
            "risk_occurrence_mode": "thesis_periodic",
            "raw_material_flow_mode": "kit_equivalent_order_up_to",
            "raw_material_order_up_to_multiplier": 2.0,
            "action_space_mode": "continuous_it_s",
        },
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    rows = []
    for seed in args.seeds:
        row = run_seed(args, seed=int(seed), run_root=run_root)
        rows.append(row)
        write_csv(out_dir / "seed_summary.csv", rows)
        (out_dir / "summary.json").write_text(
            json.dumps(
                {
                    "config": config,
                    "overall": build_overall_summary(rows),
                    "seeds": rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(json.dumps(row, indent=2), flush=True)

    summary = {
        "config": config,
        "overall": build_overall_summary(rows),
        "seeds": rows,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    write_csv(out_dir / "seed_summary.csv", rows)
    print(f"Saved to: {out_dir}", flush=True)
    print(json.dumps(summary["overall"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
