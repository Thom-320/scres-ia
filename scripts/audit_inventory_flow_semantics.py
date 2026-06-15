#!/usr/bin/env python3
"""Audit raw-material flow semantics for thesis inventory buffers.

The thesis states Op2/Op3 quantities as "x units of each rm" and Table 6.1
defines one ration as a bundle of rm1..rm12. The Python DES aggregates raw
materials into single WDC/AL containers. This audit makes the unit convention
visible and checks whether the Table 6.16 inventory levels affect outcomes.

It is read-only: it does not patch the simulation.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.config import (  # noqa: E402
    DAYS_PER_WEEK,
    INVENTORY_BUFFERS,
    NUM_RAW_MATERIALS,
    OPERATIONS,
    RATIONS_PER_SHIFT,
)
from supply_chain.external_env_interface import (  # noqa: E402
    THESIS_INVENTORY_PERIODS,
    get_episode_terminal_metrics,
    make_dkana_thesis_faithful_env,
)


def thesis_factorized_action(period: int | None, shifts: int) -> np.ndarray:
    if period is None:
        return np.array([0, shifts - 1], dtype=np.int64)
    return np.array(
        [THESIS_INVENTORY_PERIODS.index(int(period)) + 1, shifts - 1],
        dtype=np.int64,
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_policy(args: argparse.Namespace, *, period: int | None, rep: int) -> dict[str, Any]:
    action = thesis_factorized_action(period, args.shifts)
    env = make_dkana_thesis_faithful_env(
        reward_mode=args.reward_mode,
        risk_level=args.risk_level,
        observation_version=args.observation_version,
        observation_mode=args.observation_mode,
        action_space_mode="thesis_factorized",
        inventory_period_mode="thesis_strict",
        initial_action=action,
        step_size_hours=args.step_size_hours,
        max_steps=args.max_steps,
        stochastic_pt=args.stochastic_pt,
        learn_initial_decision=False,
    )
    seed = args.seed + rep
    obs, info = env.reset(seed=seed)
    terminated = truncated = False
    reward_total = 0.0
    while not (terminated or truncated):
        obs, reward, terminated, truncated, info = env.step(action)
        reward_total += float(reward)

    sim = env.unwrapped.sim
    terminal = get_episode_terminal_metrics(env)
    detail = sim._inventory_detail()
    pending_qty = sum(
        float(getattr(order, "remaining_qty", 0.0))
        for order in getattr(sim, "pending_backorders", [])
    )
    targets = getattr(sim, "inventory_buffer_targets", {})
    row = {
        "period": 0 if period is None else int(period),
        "policy": "I0" if period is None else f"I{period}",
        "replication": rep,
        "seed": seed,
        "reward_total": reward_total,
        "fill_rate_order_level": float(terminal["fill_rate_order_level"]),
        "order_level_ret_mean": float(terminal["order_level_ret_mean"]),
        "pending_backorder_qty": float(pending_qty),
        "total_demanded": float(getattr(sim, "total_demanded", 0.0)),
        "total_delivered": float(getattr(sim, "total_delivered", 0.0)),
        "target_total": float(sum(float(v) for v in targets.values())),
        "target_op3_rm": float(targets.get("op3_rm", 0.0)),
        "target_op5_rm": float(targets.get("op5_rm", 0.0)),
        "target_op9_rations": float(targets.get("op9_rations", 0.0)),
        "raw_material_wdc": float(detail["raw_material_wdc"]),
        "raw_material_al": float(detail["raw_material_al"]),
        "rations_sb": float(detail["rations_sb"]),
        "rations_cssu": float(detail["rations_cssu"]),
        "rations_theatre": float(detail["rations_theatre"]),
    }
    env.close()
    return row


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for policy in sorted({row["policy"] for row in rows}):
        bucket = [row for row in rows if row["policy"] == policy]
        out.append(
            {
                "policy": policy,
                "episode_count": len(bucket),
                "fill_rate_order_level_mean": float(
                    np.mean([row["fill_rate_order_level"] for row in bucket])
                ),
                "order_level_ret_mean": float(
                    np.mean([row["order_level_ret_mean"] for row in bucket])
                ),
                "pending_backorder_qty_mean": float(
                    np.mean([row["pending_backorder_qty"] for row in bucket])
                ),
                "target_total_mean": float(np.mean([row["target_total"] for row in bucket])),
                "raw_material_wdc_mean": float(
                    np.mean([row["raw_material_wdc"] for row in bucket])
                ),
                "raw_material_al_mean": float(
                    np.mean([row["raw_material_al"] for row in bucket])
                ),
                "rations_sb_mean": float(np.mean([row["rations_sb"] for row in bucket])),
            }
        )
    return out


def write_report(out_dir: Path, args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    summary = summarize(rows)
    write_csv(out_dir / "policy_summary.csv", summary)
    op2_month_aggregate = float(OPERATIONS[2]["q"]) * float(NUM_RAW_MATERIALS)
    op3_month_aggregate = float(OPERATIONS[3]["q"]) * float(NUM_RAW_MATERIALS) * 4.0
    assembly_month = float(RATIONS_PER_SHIFT) * float(DAYS_PER_WEEK) * 4.0
    constants = {
        "num_raw_materials": NUM_RAW_MATERIALS,
        "op2_month_if_summed_rm_units": op2_month_aggregate,
        "op3_month_if_summed_rm_units": op3_month_aggregate,
        "assembly_month_rations_s1": assembly_month,
        "op2_to_assembly_ratio_if_summed": op2_month_aggregate / assembly_month,
        "op3_to_assembly_ratio_if_summed": op3_month_aggregate / assembly_month,
        "op2_month_if_kit_equivalent": float(OPERATIONS[2]["q"]),
        "op3_month_if_kit_equivalent": float(OPERATIONS[3]["q"]) * 4.0,
    }
    (out_dir / "constants.json").write_text(
        json.dumps(constants, indent=2, sort_keys=True), encoding="utf-8"
    )

    lines = [
        "# Inventory Flow Semantics Audit",
        "",
        f"Created UTC: `{datetime.now(timezone.utc).isoformat()}`",
        f"Risk level: `{args.risk_level}`; max_steps: `{args.max_steps}`; reps: `{args.replications}`.",
        "",
        "The DES aggregates rm1..rm12 into single raw-material containers. If Op2/Op3",
        "quantities are summed across all 12 raw materials, the monthly raw-material",
        "inflow is much larger than S1 assembly consumption.",
        "",
        "## Unit Scale Check",
        "",
        f"- Op2 monthly inflow if summed over rm1..rm12: `{op2_month_aggregate:,.0f}`.",
        f"- Op3 monthly dispatch if summed over rm1..rm12: `{op3_month_aggregate:,.0f}`.",
        f"- S1 monthly assembly consumption: `{assembly_month:,.0f}` rations.",
        f"- Op2 / assembly ratio under summed-rm semantics: `{op2_month_aggregate / assembly_month:.2f}`.",
        f"- Op3 / assembly ratio under summed-rm semantics: `{op3_month_aggregate / assembly_month:.2f}`.",
        "",
        "## Static Inventory Probe",
        "",
        "| policy | fill | ReT | target total | WDC raw | AL raw | SB rations | pending qty |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| `{row['policy']}` | {row['fill_rate_order_level_mean']:.4f} | "
            f"{row['order_level_ret_mean']:.4f} | {row['target_total_mean']:.1f} | "
            f"{row['raw_material_wdc_mean']:.1f} | {row['raw_material_al_mean']:.1f} | "
            f"{row['rations_sb_mean']:.1f} | {row['pending_backorder_qty_mean']:.1f} |"
        )
    lines += [
        "",
        "Interpretation: if WDC/AL raw stocks remain orders of magnitude above",
        "Table 6.16 targets, the inventory-period lever is partially masked by",
        "the operating-flow representation. Do not treat L1b/per-node nulls as",
        "final until this audit is resolved against deterministic Table 6.10.",
        "",
        "## Files",
        "",
        "- `episode_metrics.csv`: raw probe rows.",
        "- `policy_summary.csv`: means by inventory period.",
        "- `constants.json`: unit-scale ratios.",
    ]
    (out_dir / "INVENTORY_FLOW_SEMANTICS_AUDIT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/benchmarks/inventory_flow_semantics"),
    )
    parser.add_argument("--periods", default="0,168,504,1344")
    parser.add_argument("--replications", type=int, default=3)
    parser.add_argument("--seed", type=int, default=721031)
    parser.add_argument("--risk-level", default="increased")
    parser.add_argument("--reward-mode", default="ReT_thesis")
    parser.add_argument("--max-steps", type=int, default=260)
    parser.add_argument("--step-size-hours", type=float, default=168.0)
    parser.add_argument("--shifts", type=int, default=1)
    parser.add_argument("--observation-version", default="v5")
    parser.add_argument("--observation-mode", default="env_sdm_history_reward")
    parser.add_argument("--stochastic-pt", action="store_true", default=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    label = args.label or datetime.now(timezone.utc).strftime(
        "inventory_flow_semantics_%Y%m%dT%H%M%SZ"
    )
    out_dir = args.output_root / label
    out_dir.mkdir(parents=True, exist_ok=False)
    periods = [
        None if int(token.strip()) == 0 else int(token.strip())
        for token in args.periods.split(",")
        if token.strip()
    ]
    rows = []
    for period in periods:
        if period is not None and period not in INVENTORY_BUFFERS:
            raise ValueError(f"Unknown inventory period: {period}")
        for rep in range(args.replications):
            rows.append(run_policy(args, period=period, rep=rep))
    write_csv(out_dir / "episode_metrics.csv", rows)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "periods": [0 if period is None else period for period in periods],
        "replications": args.replications,
        "seed": args.seed,
        "risk_level": args.risk_level,
        "reward_mode": args.reward_mode,
        "max_steps": args.max_steps,
        "step_size_hours": args.step_size_hours,
        "shifts": args.shifts,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_report(out_dir, args, rows)
    print(out_dir / "INVENTORY_FLOW_SEMANTICS_AUDIT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
