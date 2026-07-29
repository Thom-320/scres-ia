#!/usr/bin/env python3
"""Audit delayed credit in Track A's Excel-ReT step reward.

The evaluation endpoint must remain Garrido/Excel ReT. This audit keeps live
references to old ``OrderRecord`` objects and therefore measures how much of a
weekly reward delta comes from orders that already existed before the decision
step and were resolved or otherwise mutated during the step.

It does NOT isolate the narrower question "does time passing alone rescore an
unchanged old order?" Use ``scripts/audit_ret_excel_delta_reward_noise.py`` for
that frozen-snapshot audit.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_track_a_v2_conservation_3d_gate import make_env  # noqa: E402
from scripts.run_track_a_v2_conservation_ppo import read_gate, split_regime  # noqa: E402
from supply_chain.ret_thesis import (  # noqa: E402
    compute_ret_per_order_excel_formula,
    order_counts_as_backorder_for_fill_rate,
)


def per_order_excel_values(orders: list[Any], *, current_time: float) -> dict[int, float]:
    ordered = sorted(
        list(orders),
        key=lambda order: (
            int(getattr(order, "j", 0) or 0),
            float(getattr(order, "OPTj", 0.0) or 0.0),
        ),
    )
    out: dict[int, float] = {}
    cumulative_backorders = 0
    cumulative_unattended = 0
    for idx, order in enumerate(ordered, start=1):
        if bool(getattr(order, "lost", False)):
            cumulative_unattended += 1
        elif order_counts_as_backorder_for_fill_rate(order, current_time=current_time):
            cumulative_backorders += 1
        value, _case = compute_ret_per_order_excel_formula(
            order,
            j=idx,
            cumulative_backorders=cumulative_backorders,
            cumulative_unattended=cumulative_unattended,
        )
        out[id(order)] = float(value)
    return out


def parse_regime_subset(value: str, all_regimes: list[str]) -> list[str]:
    if value.strip().lower() in {"all", "*"}:
        return all_regimes
    requested = [x.strip() for x in value.split(",") if x.strip()]
    unknown = [x for x in requested if x not in all_regimes]
    if unknown:
        raise ValueError(f"unknown regimes {unknown}; valid={all_regimes}")
    return requested


def audit_episode(regime: str, action: tuple[float, ...], *, seed: int, max_steps: int) -> list[dict[str, Any]]:
    family, phi, psi = split_regime(regime)
    env = make_env(family=family, phi=phi, psi=psi, max_steps=max_steps, seed=seed)
    obs, _info = env.reset(seed=seed)
    done = truncated = False
    rows: list[dict[str, Any]] = []
    try:
        step = 0
        while not (done or truncated):
            before_orders = list(env.unwrapped.sim.orders)
            before_values = per_order_excel_values(before_orders, current_time=float(env.unwrapped.sim.env.now))
            before_total = float(sum(before_values.values()))

            obs, reward, done, truncated, info = env.step(np.asarray(action, dtype=np.float32))

            after_orders = list(env.unwrapped.sim.orders)
            after_values = per_order_excel_values(after_orders, current_time=float(env.unwrapped.sim.env.now))
            after_total = float(sum(after_values.values()))
            before_ids = set(before_values)
            after_ids = set(after_values)
            old_ids = before_ids & after_ids
            new_ids = after_ids - before_ids

            old_state_change = float(sum(after_values[i] for i in old_ids) - sum(before_values[i] for i in old_ids))
            new_contribution = float(sum(after_values[i] for i in new_ids))
            total_delta = float(after_total - before_total)

            rows.append(
                {
                    "regime": regime,
                    "seed": seed,
                    "step": step,
                    "sim_time_hours": float(env.unwrapped.sim.env.now),
                    "reward": float(reward),
                    "total_delta": total_delta,
                    "old_state_change": old_state_change,
                    "new_contribution": new_contribution,
                    "old_abs_share": abs(old_state_change) / max(abs(total_delta), 1e-12),
                    "n_orders_before": len(before_orders),
                    "n_orders_after": len(after_orders),
                    "n_new_orders": len(new_ids),
                    "flow_fill_rate": float(info.get("flow_fill_rate", np.nan)),
                    "service_loss_auc_ration_hours": float(info.get("service_loss_auc_ration_hours", np.nan)),
                }
            )
            step += 1
    finally:
        env.close()
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_regime: dict[str, dict[str, float]] = {}
    for regime in sorted({str(row["regime"]) for row in rows}):
        sub = [row for row in rows if row["regime"] == regime]
        total_abs_delta = float(sum(abs(float(row["total_delta"])) for row in sub))
        total_abs_old = float(sum(abs(float(row["old_state_change"])) for row in sub))
        total_abs_new = float(sum(abs(float(row["new_contribution"])) for row in sub))
        by_regime[regime] = {
            "steps": len(sub),
            "sum_abs_total_delta": total_abs_delta,
            "sum_abs_old_state_change": total_abs_old,
            "sum_abs_new_contribution": total_abs_new,
            "old_state_change_abs_share_of_total_delta": total_abs_old / max(total_abs_delta, 1e-12),
            "steps_old_state_change_abs_gt_new_contribution_abs": int(
                sum(abs(float(row["old_state_change"])) > abs(float(row["new_contribution"])) for row in sub)
            ),
            "mean_reward_minus_total_delta_abs": float(
                np.mean([abs(float(row["reward"]) - float(row["total_delta"])) for row in sub])
            ),
        }

    totals = Counter()
    for row in rows:
        totals["sum_abs_total_delta"] += abs(float(row["total_delta"]))
        totals["sum_abs_old_state_change"] += abs(float(row["old_state_change"]))
        totals["sum_abs_new_contribution"] += abs(float(row["new_contribution"]))
        totals["steps"] += 1
        if abs(float(row["old_state_change"])) > abs(float(row["new_contribution"])):
            totals["steps_old_state_change_abs_gt_new_contribution_abs"] += 1

    return {
        "steps": int(totals["steps"]),
        "sum_abs_total_delta": float(totals["sum_abs_total_delta"]),
        "sum_abs_old_state_change": float(totals["sum_abs_old_state_change"]),
        "sum_abs_new_contribution": float(totals["sum_abs_new_contribution"]),
        "old_state_change_abs_share_of_total_delta": float(
            totals["sum_abs_old_state_change"] / max(float(totals["sum_abs_total_delta"]), 1e-12)
        ),
        "steps_old_state_change_abs_gt_new_contribution_abs": int(
            totals["steps_old_state_change_abs_gt_new_contribution_abs"]
        ),
        "by_regime": by_regime,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gate-dir", default="outputs/experiments/track_a_v2_conservation_5d_gate_2026-07-03")
    ap.add_argument("--output", default="outputs/audits/track_a_ret_excel_reward_revaluation_2026-07-03")
    ap.add_argument("--regimes", default="all")
    ap.add_argument("--seed", type=int, default=9400)
    ap.add_argument("--max-steps", type=int, default=52)
    args = ap.parse_args()

    gate_summary, all_regimes, candidates = read_gate(Path(args.gate_dir))
    regimes = parse_regime_subset(args.regimes, all_regimes)
    rows: list[dict[str, Any]] = []
    for regime in regimes:
        label = str(gate_summary["best_by_regime"][regime]["candidate"])
        rows.extend(
            audit_episode(
                regime,
                candidates[label].action,
                seed=int(args.seed),
                max_steps=int(args.max_steps),
            )
        )

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "step_decomposition.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=sorted({k for row in rows for k in row}))
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "args": vars(args),
        "policy": "best_by_regime_static_from_gate",
        "primary_question": "training reward decomposition only; evaluation Excel ReT remains unchanged",
        "summary": summarize(rows),
    }
    (out / "summary.json").write_text(json.dumps(payload, indent=2))
    s = payload["summary"]
    (out / "README.md").write_text(
        "\n".join(
            [
                "# Track A Excel-ReT Delayed-Credit Audit",
                "",
                "This is not a new performance run. It decomposes the per-step `ReT_excel_delta` training signal into new-order contribution and contribution from pre-existing orders whose mutable state changed during the step.",
                "",
                "It does not isolate time-passing-only retroactive rescoring; use `scripts/audit_ret_excel_delta_reward_noise.py` for that frozen-snapshot test.",
                "",
                f"Steps: {s['steps']}",
                f"Old-order state-change absolute share of total delta: {s['old_state_change_abs_share_of_total_delta']:.3f}",
                f"Steps where |old state-change| > |new contribution|: {s['steps_old_state_change_abs_gt_new_contribution_abs']}/{s['steps']}",
            ]
        )
    )
    print((out / "README.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
