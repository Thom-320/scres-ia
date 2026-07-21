#!/usr/bin/env python3
"""Audit whether Program Q's terminal learning reward matches its outcomes.

This is a burned-development diagnostic.  It evaluates the complete 4^8
frontier but never exposes that matrix to a learner or search algorithm.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_program_q_replication import scheduler  # noqa: E402
from supply_chain.program_o_full_des import run_program_o_full_des_episode  # noqa: E402
from supply_chain.program_o_full_des_transducer import (  # noqa: E402
    direct_full_des_vector,
    extract_full_des_skeleton,
    full_action_calendars,
    simulate_full_des_frontier,
)
from supply_chain.program_o_ret_env import CONFIRMED_RET_CELLS  # noqa: E402


DEFAULT_TAPES = (94800001, 94800002, 94800003)
AUDIT_KEYS = (
    "ret_visible",
    "ret_full",
    "quantity_ret_full",
    "unresolved_orders",
    "unresolved_quantity",
    "lost_orders",
    "worst_product_fill",
    "gross_policy_batch_slots",
    "gross_production_quantity",
    "charged_downstream_vehicle_hours",
)


def _safe_correlation(left: np.ndarray, right: np.ndarray, *, rank: bool) -> float:
    if np.ptp(left) <= 1e-15 or np.ptp(right) <= 1e-15:
        return 0.0
    result = spearmanr(left, right) if rank else pearsonr(left, right)
    return float(result.statistic)


def omission_envelope(ret_visible: np.ndarray, unresolved: np.ndarray) -> list[dict[str, float]]:
    """Best visible ReT available at each unresolved-order count."""
    rows = []
    for count in sorted(set(map(float, unresolved))):
        mask = np.isclose(unresolved, count)
        rows.append(
            {
                "unresolved_orders": count,
                "best_ret_visible": float(np.max(ret_visible[mask])),
                "policy_count": int(np.sum(mask)),
            }
        )
    return rows


def summarize_frontier(panel: dict[str, np.ndarray], calendars: np.ndarray) -> dict:
    ret_visible = np.asarray(panel["ret_visible"], dtype=float)
    ret_full = np.asarray(panel["ret_full"], dtype=float)
    quantity_ret = np.asarray(panel["quantity_ret_full"], dtype=float)
    unresolved = np.asarray(panel["unresolved_orders"], dtype=float)
    worst_fill = np.asarray(panel["worst_product_fill"], dtype=float)
    visible_index = int(np.argmax(ret_visible))
    full_index = int(np.argmax(ret_full))
    eligible = (worst_fill >= 0.70) & (np.asarray(panel["lost_orders"]) <= 1e-12)
    constrained_index = int(np.argmax(np.where(eligible, ret_visible, -np.inf))) if np.any(eligible) else None
    top_count = max(1, int(np.ceil(0.01 * len(ret_visible))))
    top = np.argpartition(ret_visible, -top_count)[-top_count:]
    envelope = omission_envelope(ret_visible, unresolved)
    best_zero = max(
        (row["best_ret_visible"] for row in envelope if row["unresolved_orders"] == 0.0),
        default=float("nan"),
    )
    best_positive = max(
        (row["best_ret_visible"] for row in envelope if row["unresolved_orders"] > 0.0),
        default=float("nan"),
    )

    def policy_row(index: int | None) -> dict | None:
        if index is None:
            return None
        return {
            "index": index,
            "calendar": list(map(int, calendars[index])),
            **{key: float(np.asarray(panel[key])[index]) for key in AUDIT_KEYS},
        }

    return {
        "policy_count": int(len(ret_visible)),
        "ret_visible_range": [float(np.min(ret_visible)), float(np.max(ret_visible))],
        "ret_full_range": [float(np.min(ret_full)), float(np.max(ret_full))],
        "ret_full_degenerate": bool(np.ptp(ret_full) <= 1e-15),
        "quantity_ret_full_degenerate": bool(np.ptp(quantity_ret) <= 1e-15),
        "pearson_ret_visible_ret_full": _safe_correlation(ret_visible, ret_full, rank=False),
        "spearman_ret_visible_ret_full": _safe_correlation(ret_visible, ret_full, rank=True),
        "spearman_ret_visible_quantity_ret_full": _safe_correlation(ret_visible, quantity_ret, rank=True),
        "spearman_ret_visible_worst_product_fill": _safe_correlation(ret_visible, worst_fill, rank=True),
        "spearman_ret_visible_unresolved_orders": _safe_correlation(ret_visible, unresolved, rank=True),
        "top_1pct_mean_unresolved_orders": float(np.mean(unresolved[top])),
        "top_1pct_mean_worst_product_fill": float(np.mean(worst_fill[top])),
        "best_positive_unresolved_minus_best_zero_unresolved": (
            float(best_positive - best_zero)
            if np.isfinite(best_positive) and np.isfinite(best_zero)
            else None
        ),
        "visible_champion": policy_row(visible_index),
        "full_ledger_champion": policy_row(full_index),
        "constrained_visible_champion": policy_row(constrained_index),
        "visible_and_full_champion_same": visible_index == full_index,
        "visible_champion_worst_fill_passes_0p70": bool(worst_fill[visible_index] >= 0.70),
        "unresolved_envelope": envelope,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tapes", default=",".join(map(str, DEFAULT_TAPES)))
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/program_q2/reward_audit_v1/result.json",
    )
    parser.add_argument("--skip-direct-parity", action="store_true")
    args = parser.parse_args()
    tapes = tuple(map(int, args.tapes.split(",")))
    calendars = full_action_calendars()
    sched = scheduler()
    cells: dict[str, dict] = {}
    maximum_direct_error = {key: 0.0 for key in AUDIT_KEYS}
    direct_replays = 0
    for cell in CONFIRMED_RET_CELLS:
        per_tape = []
        promoted: set[tuple[int, tuple[int, ...]]] = set()
        for tape in tapes:
            skeleton, _ = extract_full_des_skeleton(
                seed=tape,
                scheduler=sched,
                regime_persistence=cell.regime_persistence,
                dominant_share=cell.dominant_share,
                downstream_freight_physics_mode="fixed_clock_physical_v1",
            )
            panel = simulate_full_des_frontier(
                skeleton=skeleton, scheduler=sched, calendars=calendars
            )
            summary = summarize_frontier(panel, calendars)
            per_tape.append({"tape": tape, **summary})
            for label in ("visible_champion", "full_ledger_champion", "constrained_visible_champion"):
                row = summary[label]
                if row is not None:
                    promoted.add((tape, tuple(row["calendar"])))
            if not args.skip_direct_parity:
                for direct_tape, calendar in sorted(promoted):
                    if direct_tape != tape:
                        continue
                    sim, direct_panel = run_program_o_full_des_episode(
                        seed=direct_tape,
                        calendar=calendar,
                        scheduler=sched,
                        regime_persistence=cell.regime_persistence,
                        dominant_share=cell.dominant_share,
                        downstream_freight_physics_mode="fixed_clock_physical_v1",
                    )
                    direct = direct_full_des_vector(sim, direct_panel)
                    index = int(sum(action * 4 ** (7 - period) for period, action in enumerate(calendar)))
                    for key in AUDIT_KEYS:
                        maximum_direct_error[key] = max(
                            maximum_direct_error[key],
                            abs(float(direct[key]) - float(panel[key][index])),
                        )
                    direct_replays += 1
        cells[cell.cell_id] = {"tapes": per_tape}

    visible_full_correlations = [
        row["spearman_ret_visible_ret_full"]
        for cell in cells.values()
        for row in cell["tapes"]
    ]
    omission_deltas = [
        row["best_positive_unresolved_minus_best_zero_unresolved"]
        for cell in cells.values()
        for row in cell["tapes"]
        if row["best_positive_unresolved_minus_best_zero_unresolved"] is not None
    ]
    frontier_rows = [row for cell in cells.values() for row in cell["tapes"]]
    payload = {
        "schema_version": "program_q2_reward_contract_audit_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "claim_status": "EXPLORATORY_NO_CLAIM",
        "tapes": list(tapes),
        "frontier_size": int(len(calendars)),
        "evaluation_endpoint": "ret_excel_request_snapshot_v2",
        "learning_reward": "terminal_ret_visible",
        "unresolved_orders_are_omitted_from_visible_mean": True,
        "cells": cells,
        "aggregate": {
            "mean_spearman_ret_visible_ret_full": float(np.mean(visible_full_correlations)),
            "ret_full_degenerate_frontiers": int(sum(row["ret_full_degenerate"] for row in frontier_rows)),
            "frontiers_audited": len(frontier_rows),
            "max_positive_unresolved_advantage_over_zero_unresolved": (
                float(max(omission_deltas)) if omission_deltas else None
            ),
            "reward_contract_requires_guardrails": True,
        },
        "direct_parity": {
            "skipped": bool(args.skip_direct_parity),
            "replays": direct_replays,
            "maximum_absolute_error": maximum_direct_error,
            "passed_at_1e_8": bool(
                args.skip_direct_parity or max(maximum_direct_error.values(), default=0.0) <= 1e-8
            ),
        },
        "interpretation_boundary": (
            "This audit detects alignment and omission incentives over the exact static frontier. "
            "It does not establish that a learning algorithm can exploit them."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in payload.items() if k != "cells"}, indent=2))
    return 0 if payload["direct_parity"]["passed_at_1e_8"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
