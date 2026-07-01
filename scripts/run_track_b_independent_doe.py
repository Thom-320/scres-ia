#!/usr/bin/env python3
"""Dense static Track B gate with independent Op9/Op10/Op12 controls.

This script is intentionally static-only.  It asks whether the downstream
decision surface has real headroom before spending PPO compute.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.config import OPERATIONS
from supply_chain.episode_metrics import METRIC_KEYS, compute_episode_metrics, merge_resource_metrics
from supply_chain.external_env_interface import make_track_b_env


DEFAULT_OUTPUT = Path("outputs/experiments/track_b_independent_doe")
SUMMARY_KEYS = (
    "policy",
    "shift",
    "op9_mult",
    "op10_mult",
    "op12_mult",
    "seed_count",
    "reward_total_mean",
    "reward_total_std",
    *tuple(f"{key}_mean" for key in METRIC_KEYS),
    *tuple(f"{key}_std" for key in METRIC_KEYS),
)


@dataclass(frozen=True)
class Policy:
    shift: int
    op9_mult: float
    op10_mult: float
    op12_mult: float

    @property
    def label(self) -> str:
        return (
            f"S{self.shift}_op9x{self.op9_mult:g}_"
            f"op10x{self.op10_mult:g}_op12x{self.op12_mult:g}"
        )


def parse_float_list(raw: str) -> tuple[float, ...]:
    return tuple(float(x.strip()) for x in raw.split(",") if x.strip())


def parse_int_list(raw: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in raw.split(",") if x.strip())


def parse_risks(raw: str | None) -> tuple[str, ...] | None:
    if not raw:
        return None
    return tuple(x.strip() for x in raw.split(",") if x.strip())


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--reward-mode", default="control_v1")
    ap.add_argument("--risk-level", default="adaptive_benchmark_v2")
    ap.add_argument("--enabled-risks", default=None)
    ap.add_argument("--risk-frequency-multiplier", type=float, default=1.0)
    ap.add_argument("--risk-impact-multiplier", type=float, default=1.0)
    ap.add_argument("--demand-mean-multiplier", type=float, default=1.0)
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--max-steps", type=int, default=104)
    ap.add_argument("--step-size-hours", type=float, default=168.0)
    ap.add_argument("--shifts", default="1,2,3")
    ap.add_argument("--op9-mults", default="1.0")
    ap.add_argument("--op10-mults", default="1.0,1.25,1.5,1.75,2.0")
    ap.add_argument("--op12-mults", default="1.0,1.25,1.5,1.75,2.0")
    return ap


def action_for(policy: Policy) -> dict[str, float | int]:
    return {
        "op3_q": float(OPERATIONS[3]["q"]),
        "op3_rop": float(OPERATIONS[3]["rop"]),
        "op9_q_min": float(OPERATIONS[9]["q"][0]) * policy.op9_mult,
        "op9_q_max": float(OPERATIONS[9]["q"][1]) * policy.op9_mult,
        "op9_rop": float(OPERATIONS[9]["rop"]),
        "op10_q_min": float(OPERATIONS[10]["q"][0]) * policy.op10_mult,
        "op10_q_max": float(OPERATIONS[10]["q"][1]) * policy.op10_mult,
        "op12_q_min": float(OPERATIONS[12]["q"][0]) * policy.op12_mult,
        "op12_q_max": float(OPERATIONS[12]["q"][1]) * policy.op12_mult,
        "assembly_shifts": int(policy.shift),
    }


def run_episode(policy: Policy, *, seed: int, args: argparse.Namespace) -> dict[str, Any]:
    env_kwargs: dict[str, Any] = {
        "reward_mode": args.reward_mode,
        "observation_version": "v7",
        "action_contract": "track_b_v1",
        "risk_level": args.risk_level,
        "step_size_hours": float(args.step_size_hours),
        "max_steps": int(args.max_steps),
        "risk_frequency_multiplier": float(args.risk_frequency_multiplier),
        "risk_impact_multiplier": float(args.risk_impact_multiplier),
        "demand_mean_multiplier": float(getattr(args, "demand_mean_multiplier", 1.0)),
    }
    enabled = parse_risks(args.enabled_risks)
    if enabled is not None:
        env_kwargs["enabled_risks"] = set(enabled)

    env = make_track_b_env(**env_kwargs)
    _, _ = env.reset(seed=seed)
    action = action_for(policy)
    done = False
    reward_total = 0.0
    steps = 0
    while not done:
        _, reward, terminated, truncated, _ = env.step(action)
        reward_total += float(reward)
        done = bool(terminated or truncated)
        steps += 1

    panel = compute_episode_metrics(env.unwrapped.sim)
    panel = merge_resource_metrics(
        panel,
        shift_hours=float(policy.shift) * float(args.step_size_hours) * float(steps),
        extra_shift_hours=float(max(0, policy.shift - 1)) * float(args.step_size_hours) * float(steps),
        strategic_buffer_units=0.0,
    )
    env.close()
    return {
        "policy": policy.label,
        "seed": int(seed),
        "shift": int(policy.shift),
        "op9_mult": float(policy.op9_mult),
        "op10_mult": float(policy.op10_mult),
        "op12_mult": float(policy.op12_mult),
        "reward_total": float(reward_total),
        **panel,
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["policy"]), []).append(row)

    out: list[dict[str, Any]] = []
    for label, bucket in grouped.items():
        first = bucket[0]
        summary: dict[str, Any] = {
            "policy": label,
            "shift": int(first["shift"]),
            "op9_mult": float(first["op9_mult"]),
            "op10_mult": float(first["op10_mult"]),
            "op12_mult": float(first["op12_mult"]),
            "seed_count": len(bucket),
        }
        for key in ("reward_total", *METRIC_KEYS):
            vals = [float(row.get(key, 0.0)) for row in bucket]
            summary[f"{key}_mean"] = float(statistics.fmean(vals))
            summary[f"{key}_std"] = float(statistics.stdev(vals)) if len(vals) > 1 else 0.0
        out.append(summary)
    return sorted(out, key=lambda r: str(r["policy"]))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = build_parser().parse_args()
    output_dir = args.output_dir or (
        DEFAULT_OUTPUT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    policies = [
        Policy(shift=s, op9_mult=o9, op10_mult=o10, op12_mult=o12)
        for s in parse_int_list(args.shifts)
        for o9 in parse_float_list(args.op9_mults)
        for o10 in parse_float_list(args.op10_mults)
        for o12 in parse_float_list(args.op12_mults)
    ]
    seeds = parse_int_list(args.seeds)

    rows: list[dict[str, Any]] = []
    for policy in policies:
        for seed in seeds:
            rows.append(run_episode(policy, seed=seed, args=args))

    summary = summarize(rows)
    best_ret = max(summary, key=lambda row: float(row["ret_excel_mean"]))
    best_cvar = min(summary, key=lambda row: float(row["service_loss_auc_ration_hours_mean"]))
    best_flow = max(summary, key=lambda row: float(row["flow_fill_rate_mean"]))
    payload = {
        "config": {
            **vars(args),
            "output_dir": str(output_dir),
            "n_policies": len(policies),
            "n_episodes": len(rows),
        },
        "best": {
            "ret_excel": best_ret,
            "service_loss_auc_ration_hours": best_cvar,
            "flow_fill_rate": best_flow,
        },
        "summary_rows": summary,
        "seed_rows": rows,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    write_csv(output_dir / "seed_metrics.csv", rows)
    write_csv(output_dir / "policy_summary.csv", summary)
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Track B Independent DOE",
        "",
        f"Policies: {len(policies)}; episodes: {len(rows)}",
        "",
        "## Best by Excel ReT",
        "",
        json.dumps(best_ret, indent=2),
        "",
        "## Best by service-loss AUC (lower)",
        "",
        json.dumps(best_cvar, indent=2),
        "",
        "## Best by flow fill",
        "",
        json.dumps(best_flow, indent=2),
    ]
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"WROTE {output_dir}")
    print(f"BEST_RET {best_ret['policy']} ret_excel={best_ret['ret_excel_mean']:.6f}")


if __name__ == "__main__":
    main()
