#!/usr/bin/env python3
"""Eval-only upstream static bound for the Track B dense-frontier claim.

The canonical Track B dense static frontier sweeps shift x Op10 x Op12 while
holding upstream controls at canonical settings. This script evaluates a small
local 3x3 bound over Op3 and Op9 quantity multipliers at the best canonical
downstream cell: S2, Op10 x2.00, Op12 x1.50.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_track_b_dense_crn_static import (  # noqa: E402
    aggregate_policy,
    aggregate_seed,
    build_env_kwargs,
    build_verdict,
    fallback_eval_plan,
    infer_config,
    infer_eval_plan,
    parse_float_list,
    parse_int_list,
    write_csv,
)
from scripts.run_track_b_smoke import (  # noqa: E402
    _finalize_episode_row,
    append_order_ledger_rows,
    extract_downstream_multipliers,
    get_episode_terminal_metrics,
    init_cd_totals,
    update_cd_totals,
)
from supply_chain.config import OPERATIONS  # noqa: E402
from supply_chain.episode_metrics import compute_episode_metrics  # noqa: E402
from supply_chain.external_env_interface import make_track_b_env  # noqa: E402


DEFAULT_RUN_DIR = Path(
    "outputs/experiments/track_b_gain_2026-06-30/"
    "top_tier_confirm_v3_output/track_b_top_tier_confirm_5seed_60k_h104"
)
DEFAULT_OUTPUT_DIR = Path("outputs/experiments/track_b_upstream_bound_3x3_2026-07-02")


@dataclass(frozen=True)
class UpstreamPolicy:
    op3_mult: float
    op9_mult: float
    shift: int = 2
    op10_mult: float = 2.0
    op12_mult: float = 1.5

    @property
    def label(self) -> str:
        return (
            f"S{self.shift}_op3_{self.op3_mult:.2f}_op9_{self.op9_mult:.2f}_"
            f"op10_{self.op10_mult:.2f}_op12_{self.op12_mult:.2f}"
        )


def action_for(policy: UpstreamPolicy) -> dict[str, float | int]:
    return {
        "op3_q": float(OPERATIONS[3]["q"]) * float(policy.op3_mult),
        "op3_rop": float(OPERATIONS[3]["rop"]),
        "op9_q_min": float(OPERATIONS[9]["q"][0]) * float(policy.op9_mult),
        "op9_q_max": float(OPERATIONS[9]["q"][1]) * float(policy.op9_mult),
        "op9_rop": float(OPERATIONS[9]["rop"]),
        "op10_q_min": float(OPERATIONS[10]["q"][0]) * float(policy.op10_mult),
        "op10_q_max": float(OPERATIONS[10]["q"][1]) * float(policy.op10_mult),
        "op12_q_min": float(OPERATIONS[12]["q"][0]) * float(policy.op12_mult),
        "op12_q_max": float(OPERATIONS[12]["q"][1]) * float(policy.op12_mult),
        "assembly_shifts": int(policy.shift),
    }


def run_static_episode(
    policy: UpstreamPolicy,
    *,
    eval_item: dict[str, int],
    env_kwargs: dict[str, Any],
    order_ledger_rows: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    env = make_track_b_env(**env_kwargs)
    _, info = env.reset(seed=int(eval_item["eval_seed"]))
    terminated = False
    truncated = False
    reward_total = 0.0
    demanded_total = 0.0
    backorder_qty_total = 0.0
    steps = 0
    shift_counts = {1: 0, 2: 0, 3: 0}
    op10_multipliers: list[float] = []
    op12_multipliers: list[float] = []
    cd_totals = init_cd_totals()
    final_info = info
    action = action_for(policy)

    while not (terminated or truncated):
        _, reward, terminated, truncated, final_info = env.step(action)
        reward_total += float(reward)
        update_cd_totals(cd_totals, final_info)
        demanded_total += float(final_info.get("new_demanded", 0.0))
        backorder_qty_total += float(final_info.get("new_backorder_qty", 0.0))
        shift_counts[int(final_info.get("shifts_active", policy.shift))] += 1
        op10_mult, op12_mult = extract_downstream_multipliers(final_info)
        op10_multipliers.append(op10_mult)
        op12_multipliers.append(op12_mult)
        steps += 1

    row = _finalize_episode_row(
        policy=policy.label,
        seed=int(eval_item["seed"]),
        episode=int(eval_item["episode"]),
        eval_seed=int(eval_item["eval_seed"]),
        steps=steps,
        reward_total=reward_total,
        demanded_total=demanded_total,
        backorder_qty_total=backorder_qty_total,
        shift_counts=shift_counts,
        op10_multipliers=op10_multipliers,
        op12_multipliers=op12_multipliers,
        track_b_context=final_info["state_constraint_context"]["track_b_context"],
        terminal_metrics=get_episode_terminal_metrics(env),
        final_info=final_info,
        cd_totals=cd_totals,
        full_episode_metrics=compute_episode_metrics(env.unwrapped.sim),
    )
    row.update(
        {
            "static_shift": int(policy.shift),
            "static_op3_mult": float(policy.op3_mult),
            "static_op9_mult": float(policy.op9_mult),
            "static_op10_mult": float(policy.op10_mult),
            "static_op12_mult": float(policy.op12_mult),
        }
    )
    append_order_ledger_rows(
        order_ledger_rows,
        env,
        policy=policy.label,
        seed=int(eval_item["seed"]),
        episode=int(eval_item["episode"]),
        eval_seed=int(eval_item["eval_seed"]),
    )
    env.close()
    return row


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--op3-mults", default="0.75,1.0,1.25")
    parser.add_argument("--op9-mults", default="0.75,1.0,1.25")
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--eval-episodes", type=int, default=12)
    parser.add_argument("--reward-mode", default="control_v1")
    parser.add_argument("--risk-level", default="adaptive_benchmark_v2")
    parser.add_argument("--observation-version", default="v7")
    parser.add_argument("--step-size-hours", type=float, default=168.0)
    parser.add_argument("--max-steps", type=int, default=104)
    parser.add_argument("--export-order-ledger", action="store_true")
    parser.add_argument("--progress-every", type=int, default=50)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = infer_config(args.run_dir)
    eval_plan = infer_eval_plan(args.run_dir)
    if args.seeds:
        eval_plan = fallback_eval_plan(parse_int_list(args.seeds), int(args.eval_episodes))
    if not eval_plan:
        eval_plan = fallback_eval_plan([1, 2, 3, 4, 5], int(args.eval_episodes))

    env_kwargs = build_env_kwargs(args, config)
    policies = [
        UpstreamPolicy(op3_mult=op3, op9_mult=op9)
        for op3 in parse_float_list(args.op3_mults)
        for op9 in parse_float_list(args.op9_mults)
    ]
    total = len(policies) * len(eval_plan)
    print(
        f"Track B upstream 3x3 bound: {len(policies)} policies x "
        f"{len(eval_plan)} eval episodes = {total} episodes",
        flush=True,
    )
    print(f"Env kwargs: {env_kwargs}", flush=True)

    episode_rows: list[dict[str, Any]] = []
    ledger_rows: list[dict[str, Any]] | None = [] if args.export_order_ledger else None
    done = 0
    for policy in policies:
        for eval_item in eval_plan:
            episode_rows.append(
                run_static_episode(
                    policy,
                    eval_item=eval_item,
                    env_kwargs=env_kwargs,
                    order_ledger_rows=ledger_rows,
                )
            )
            done += 1
            if args.progress_every > 0 and done % args.progress_every == 0:
                print(f"  {done}/{total}", flush=True)

    metrics = [
        "reward_total",
        "order_level_ret_mean",
        "flow_fill_rate",
        "flow_backorder_rate",
        "terminal_rolling_fill_rate_4w",
        "terminal_rolling_backorder_rate_4w",
        "assembly_cost_index",
        "assembly_hours_total",
        "ret_garrido2024_raw_total",
        "ret_garrido2024_train_total",
        "ret_garrido2024_sigmoid_total",
        "ret_garrido2024_sigmoid_mean",
        "terminal_kappa_dot",
        "terminal_epsilon_avg",
        "order_ret_excel",
        "order_lost_rate",
        "order_service_loss_auc_per_order",
        "order_ctj_p99",
        "order_rpj_p99",
        "order_dpj_p99",
        "order_delivered_rations",
        "order_demanded_rations",
    ]
    seed_rows = aggregate_seed(episode_rows, metrics)
    for row in seed_rows:
        first = next(item for item in episode_rows if item["policy"] == row["policy"])
        row["op3_mult"] = float(first["static_op3_mult"])
        row["op9_mult"] = float(first["static_op9_mult"])
    policy_rows = aggregate_policy(seed_rows, metrics)
    for row in policy_rows:
        first = next(item for item in seed_rows if item["policy"] == row["policy"])
        row["op3_mult"] = float(first["op3_mult"])
        row["op9_mult"] = float(first["op9_mult"])

    verdict = build_verdict(
        dynamic_run_dir=args.run_dir,
        static_episode_rows=episode_rows,
        static_seed_rows=seed_rows,
        static_policy_rows=policy_rows,
    )
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "3x3 eval-only upstream static bound at canonical Track B best downstream cell",
        "config": {
            "run_dir": str(args.run_dir),
            "output_dir": str(args.output_dir),
            "env_kwargs": env_kwargs,
            "op3_mults": parse_float_list(args.op3_mults),
            "op9_mults": parse_float_list(args.op9_mults),
            "fixed_shift": 2,
            "fixed_op10_mult": 2.0,
            "fixed_op12_mult": 1.5,
            "n_policies": len(policies),
            "n_eval_episodes": len(eval_plan),
            "export_order_ledger": bool(args.export_order_ledger),
        },
        "verdict": verdict,
        "best_static": policy_rows[0],
    }

    write_csv(args.output_dir / "episode_metrics.csv", episode_rows)
    write_csv(args.output_dir / "seed_metrics.csv", seed_rows)
    write_csv(args.output_dir / "policy_summary.csv", policy_rows)
    write_csv(args.output_dir / "summary.csv", policy_rows)
    if ledger_rows is not None:
        write_csv(args.output_dir / "order_ledger.csv", ledger_rows)
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (args.output_dir / "verdict_vs_dynamic.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    (args.output_dir / "README.md").write_text(
        "# Track B upstream 3x3 static bound\n\n"
        "Eval-only local bound over Op3 and Op9 quantity multipliers at the "
        "canonical best downstream-dispatch static cell (S2, Op10 x2.00, "
        "Op12 x1.50). Uses the canonical Track B CRN plan: seeds 1..5, "
        "12 eval episodes per seed, h104, v7, control_v1, "
        "adaptive_benchmark_v2.\n",
        encoding="utf-8",
    )
    print(f"WROTE {args.output_dir}", flush=True)
    print(
        "BEST_UPSTREAM_BOUND {policy} ret={ret:.6f}; PPO delta={delta:+.6f}".format(
            policy=verdict["best_static_policy"],
            ret=verdict["best_static"]["order_level_ret"],
            delta=verdict["deltas"]["order_level_ret"],
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
