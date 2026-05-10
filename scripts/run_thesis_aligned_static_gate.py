#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.config import CAPACITY_BY_SHIFTS, OPERATIONS  # noqa: E402
from supply_chain.external_env_interface import (  # noqa: E402
    get_episode_terminal_metrics,
    get_thesis_aligned_training_env_spec,
    make_thesis_aligned_training_env,
    spec_to_dict,
)

DEFAULT_OUTPUT_ROOT = Path("outputs/thesis_aligned_training")
STATIC_POLICIES = ("garrido_cf_s1", "garrido_cf_s2", "garrido_cf_s3")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a no-training static Garrido gate inside the thesis-aligned "
            "Gym environment. This checks the trainable lane before PPO/SAC."
        )
    )
    parser.add_argument("--label", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--eval-steps", type=int, default=52)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--risk-level",
        choices=["current", "increased", "severe"],
        default="current",
    )
    parser.add_argument("--stochastic-pt", action="store_true")
    return parser


def command_line() -> str:
    return " ".join(
        [
            Path(sys.executable).name,
            "scripts/run_thesis_aligned_static_gate.py",
            *sys.argv[1:],
        ]
    )


def garrido_action(policy: str) -> dict[str, float | int]:
    shift = {"garrido_cf_s1": 1, "garrido_cf_s2": 2, "garrido_cf_s3": 3}[policy]
    return {
        "assembly_shifts": shift,
        "op3_q": float(CAPACITY_BY_SHIFTS[shift]["op3_q"]),
        "op3_rop": float(OPERATIONS[3]["rop"]),
        "op9_q_min": float(OPERATIONS[9]["q"][0]),
        "op9_q_max": float(OPERATIONS[9]["q"][1]),
        "op9_rop": float(OPERATIONS[9]["rop"]),
        "batch_size": float(CAPACITY_BY_SHIFTS[shift]["op7_q"]),
    }


def evaluate_policy(
    *,
    policy: str,
    seed: int,
    eval_steps: int,
    risk_level: str,
    stochastic_pt: bool,
) -> dict[str, Any]:
    env = make_thesis_aligned_training_env(
        max_steps=eval_steps,
        risk_level=risk_level,
        stochastic_pt=stochastic_pt,
    )
    obs, info = env.reset(seed=seed)
    del obs
    reward_total = 0.0
    steps = 0
    terminated = truncated = False
    action = garrido_action(policy)
    while not (terminated or truncated):
        _, reward, terminated, truncated, info = env.step(action)
        reward_total += float(reward)
        steps += 1

    terminal = get_episode_terminal_metrics(env)
    sim = env.sim
    if sim is None:
        raise RuntimeError("Environment closed without simulator state.")
    return {
        "policy": policy,
        "seed": seed,
        "steps": steps,
        "reward_total": reward_total,
        "fill_rate_order_level": terminal["fill_rate_order_level"],
        "backorder_rate_order_level": terminal["backorder_rate_order_level"],
        "order_level_ret_mean": terminal["order_level_ret_mean"],
        "total_delivered": float(sim.total_delivered),
        "total_demanded": float(sim.total_demanded),
        "total_backorders": int(sim.total_backorders),
        "risk_events": len(sim.risk_events),
        "reset_time": float(info.get("post_warmup_start_time", sim.env.now)),
        "training_protocol": info.get("training_protocol"),
        "year_basis": info.get("year_basis"),
    }


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for policy in STATIC_POLICIES:
        bucket = [row for row in rows if row["policy"] == policy]
        out.append(
            {
                "policy": policy,
                "episode_count": len(bucket),
                "reward_total_mean": float(
                    np.mean([row["reward_total"] for row in bucket])
                ),
                "fill_rate_order_level_mean": float(
                    np.mean([row["fill_rate_order_level"] for row in bucket])
                ),
                "backorder_rate_order_level_mean": float(
                    np.mean([row["backorder_rate_order_level"] for row in bucket])
                ),
                "order_level_ret_mean": float(
                    np.mean([row["order_level_ret_mean"] for row in bucket])
                ),
            }
        )
    return out


def main() -> int:
    args = build_parser().parse_args()
    run_dir = args.output_root / args.label
    run_dir.mkdir(parents=True, exist_ok=False)

    manifest = {
        "created_at": utc_now_iso(),
        "command": command_line(),
        "pid": os.getpid(),
        "env_spec": spec_to_dict(get_thesis_aligned_training_env_spec()),
        "risk_level": args.risk_level,
        "stochastic_pt": args.stochastic_pt,
        "eval_steps": args.eval_steps,
        "policies": STATIC_POLICIES,
    }
    (run_dir / "command.txt").write_text(command_line() + "\n", encoding="utf-8")
    (run_dir / "pid.txt").write_text(f"{os.getpid()}\n", encoding="utf-8")
    write_json(run_dir / "manifest.json", manifest)
    write_json(
        run_dir / "status.json", {"status": "RUNNING", "updated_at": utc_now_iso()}
    )

    rows = [
        evaluate_policy(
            policy=policy,
            seed=seed,
            eval_steps=args.eval_steps,
            risk_level=args.risk_level,
            stochastic_pt=args.stochastic_pt,
        )
        for seed in args.seeds
        for policy in STATIC_POLICIES
    ]
    summary = {
        "created_at": utc_now_iso(),
        "runs": rows,
        "policy_summary": aggregate(rows),
    }
    write_json(run_dir / "summary.json", summary)
    write_csv(run_dir / "episode_metrics.csv", rows)
    write_csv(run_dir / "policy_summary.csv", summary["policy_summary"])
    write_json(
        run_dir / "status.json", {"status": "SUCCEEDED", "updated_at": utc_now_iso()}
    )
    print(json.dumps(summary["policy_summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
