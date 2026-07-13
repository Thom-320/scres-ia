#!/usr/bin/env python3
"""Compute a rigorous but deliberately loose canonical-ReT bound on locked tapes.

The bound is useful as a falsification of the tempting affected-order shortcut:
because workbook-visible ReT is un-clipped and almost every order lies after the
first action-sensitive event, this valid bound cannot close the 0.01 gate.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.config import HOURS_PER_WEEK  # noqa: E402
from supply_chain.paper2_bottleneck import (  # noqa: E402
    ACTIONS,
    CONTEXTS,
    make_sim,
    materialize_tape,
)
from supply_chain.program_f import advance_including  # noqa: E402
from supply_chain.episode_metrics import compute_episode_metrics  # noqa: E402
from supply_chain.ret_thesis import (  # noqa: E402
    compute_order_level_ret_excel_visible_ledger,
)


ROOT = Path(__file__).resolve().parent.parent
LOCKED_START = 1_110_001
N_LOCKED = 120
WEEKS = 24


def run_constant_m(tape: dict[str, Any]):
    sim, controller, start = make_sim(tape)
    end = start + WEEKS * HOURS_PER_WEEK
    for week in range(WEEKS):
        controller.activate_week(week)
        controller.request(ACTIONS[0])
        advance_including(sim, min(end, start + (week + 1) * HOURS_PER_WEEK))
    metrics = compute_episode_metrics(sim, treatment_start=start)
    orders = [
        order for order in sim.orders
        if not bool(getattr(order, "metrics_excluded", False))
        and float(getattr(order, "OPTj", 0.0)) >= start
    ]
    return sim, controller, start, metrics, orders


def ci95(values: list[float], seed: int = 20260713, n_boot: int = 4000):
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    boot = np.asarray([
        rng.choice(x, len(x), replace=True).mean() for _ in range(n_boot)
    ])
    return [float(x.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]


def tape_bound(seed: int, first_context: str) -> dict[str, Any]:
    tape = materialize_tape(seed, first_context, "locked", weeks=WEEKS)
    _, _, start, metrics, orders = run_constant_m(tape)
    event_starts = [start + float(event["onset_hours"]) for event in tape["base_events"]]
    if not event_starts:
        raise AssertionError("Locked bottleneck tapes are expected to contain events")
    first_event = min(event_starts)
    deltas = []
    for event_start in event_starts:
        fractional = event_start - math.floor(event_start)
        deltas.append(1.0 if abs(fractional) < 1e-12 else 1.0 - fractional)
    delta_min = min(deltas)
    row_max = max(1.0, 0.5 / delta_min)

    visible = compute_order_level_ret_excel_visible_ledger(
        orders,
        current_time=float(start + WEEKS * HOURS_PER_WEEK),
    )
    ret_by_j = {int(row["j"]): float(row["ret"]) for row in visible["ret_rows"]}
    invariant_prefix = [
        order for order in orders
        if getattr(order, "OATj", None) is not None
        and float(order.OATj) < first_event
    ]
    prefix_values = [ret_by_j[int(order.j)] for order in invariant_prefix]
    n_orders = len(orders)
    upper_mean = (
        sum(prefix_values) + (n_orders - len(prefix_values)) * row_max
    ) / max(n_orders, 1)
    return {
        "seed": seed,
        "tape_sha256": tape["threat_sha256"],
        "n_orders": n_orders,
        "n_visible_constant_M": int(visible["n_visible_rows"]),
        "invariant_prefix_orders": len(prefix_values),
        "invariant_prefix_ret_sum": float(sum(prefix_values)),
        "minimum_integer_delivery_gap_after_event": float(delta_min),
        "row_score_upper_bound": float(row_max),
        "policy_mean_ret_upper_bound": float(upper_mean),
        "constant_M_ret_excel": float(metrics["ret_excel"]),
        "upper_gap_vs_constant_M": float(upper_mean - metrics["ret_excel"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "paper2_bottleneck" / "loose_canonical_upper_bound.json",
    )
    args = parser.parse_args()

    rows = [
        tape_bound(LOCKED_START + offset, CONTEXTS[offset % len(CONTEXTS)])
        for offset in range(N_LOCKED)
    ]
    upper_gaps = [row["upper_gap_vs_constant_M"] for row in rows]
    result = {
        "schema_version": "paper2_bottleneck_loose_canonical_bound_v1",
        "scientific_status": "VALID_EMPIRICAL_TAPE_BOUND_TOO_LOOSE_TO_CLOSE",
        "tapes": {"split": "locked_burned", "seed_start": LOCKED_START, "n": N_LOCKED},
        "proof": {
            "policy_invariant_prefix": "Orders completed before the first action-sensitive event are invariant.",
            "per_visible_row_bound": "No-risk <=1; autotomy AP/LT<=1; integer-hour delivery implies recovery 0.5/RP <= 0.5/delta_tau.",
            "sparse_denominator_bound": "For visible count V in [U,N], (S+(V-U)Rmax)/V is maximized at V=N because prefix mean<=Rmax.",
            "tape_mean_bound": "Ubar_tau=(S_prefix+(N-U)Rmax)/N.",
            "scope": "Deterministic upper bound for each already-burned tape; not a population confidence bound and not a domain-valid contract ceiling.",
        },
        "summary": {
            "n_orders_unique": sorted({row["n_orders"] for row in rows}),
            "invariant_prefix_min_mean_max": [
                min(row["invariant_prefix_orders"] for row in rows),
                float(np.mean([row["invariant_prefix_orders"] for row in rows])),
                max(row["invariant_prefix_orders"] for row in rows),
            ],
            "row_score_upper_min_median_mean_max": [
                min(row["row_score_upper_bound"] for row in rows),
                float(np.median([row["row_score_upper_bound"] for row in rows])),
                float(np.mean([row["row_score_upper_bound"] for row in rows])),
                max(row["row_score_upper_bound"] for row in rows),
            ],
            "constant_M_mean_ret_excel": float(np.mean([row["constant_M_ret_excel"] for row in rows])),
            "empirical_upper_gap_ci95": ci95(upper_gaps),
            "tapes_with_upper_gap_at_most_0_01": sum(gap <= 0.01 for gap in upper_gaps),
        },
        "conclusion": "The bound is rigorous but vacuous for the practical gate. Canonical visible-ledger nonmonotonicity and un-clipped recovery scores prevent a cheap affected-order closure; exact quotient execution or a tighter certified optimization remains required.",
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
