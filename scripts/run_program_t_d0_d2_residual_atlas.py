#!/usr/bin/env python3
"""Reconcile burned Q/T0 disagreements and compute independent residual ceilings."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_program_q_replication import scheduler  # noqa: E402
from supply_chain.program_o_full_des_transducer import extract_full_des_skeleton, full_action_calendars, simulate_full_des_frontier  # noqa: E402
from supply_chain.program_o_ret_env import CONFIRMED_RET_CELLS  # noqa: E402
from supply_chain.program_t_joint_belief import weekly_product_counts  # noqa: E402

Q_RESULT = ROOT / "results/program_q/confirmation_v1_20260718/artifacts/confirmation/evaluation/result.json"
CANONICAL = ROOT / "results/program_t/t0_ret_transducer_v1/result.json"
COMPACT = ROOT / "results/program_t/t0_full_des_matrix_v1/result.json"


def _mean_lcb(values: np.ndarray, *, resamples: int = 5000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(20260720)
    draws = rng.integers(0, len(values), size=(resamples, len(values)))
    means = values[draws].mean(axis=1)
    return float(values.mean()), float(np.quantile(means, 0.025))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-tapes", type=int, default=48)
    parser.add_argument("--exact-frontier-tapes", type=int, default=0)
    parser.add_argument("--include-records", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "results/program_t/d0_d2_residual_atlas_v1/result.json")
    args = parser.parse_args()
    if not 1 <= args.n_tapes <= 48 or not 0 <= args.exact_frontier_tapes <= args.n_tapes:
        raise ValueError("D0 must remain within the 48-tape burned T0 panel")
    q = json.loads(Q_RESULT.read_text())
    canonical = json.loads(CANONICAL.read_text())
    compact = json.loads(COMPACT.read_text())
    sched = scheduler()
    records = []
    selector_deltas = []
    frontier_deltas = []
    per_cell = {}
    for cell in CONFIRMED_RET_CELLS:
        cell_id = cell.cell_id
        q_audits = q["trajectory_audits"][cell_id]
        learner_seeds = sorted(q_audits, key=int)
        can_cell = canonical["cells"][cell_id]
        primary_id = "ret_proxy_scenario_h3_p4"
        safe_id = "ret_proxy_constraint_aware_h3_p4"
        compact_id = "ret_proxy_nominal_h8_p1"
        primary = can_cell["comparators"][primary_id]
        safe = can_cell["comparators"][safe_id]
        compact_row = compact["cells"][cell_id]["comparators"][compact_id]
        learner_ret = np.asarray(can_cell["learner"]["ret_visible"][: args.n_tapes], dtype=float)
        structured_ret = np.asarray(primary["ret_visible"][: args.n_tapes], dtype=float)
        selector = np.maximum(learner_ret, structured_ret) - structured_ret
        selector_deltas.extend(map(float, selector))
        disagreements = 0
        total = 0
        exact_cell = []
        for offset in range(args.n_tapes):
            tape = 7_490_001 + offset
            skeleton, _ = extract_full_des_skeleton(seed=tape, scheduler=sched, regime_persistence=cell.regime_persistence, dominant_share=cell.dominant_share, downstream_freight_physics_mode="fixed_clock_physical_v1")
            counts = weekly_product_counts(order_times=skeleton.order_times, order_products=skeleton.order_products, decision_start=skeleton.decision_start, weeks=skeleton.decision_weeks)
            primary_calendar = primary["calendar"][offset]
            safe_calendar = safe["calendar"][offset]
            compact_calendar = compact_row["calendar"][offset]
            for learner_seed in learner_seeds:
                q_calendar = q_audits[learner_seed]["calendars"][offset]
                for week, q_action in enumerate(q_calendar):
                    actions = {
                        "q": int(q_action),
                        "scenario_h3_p4": int(primary_calendar[week]),
                        "constraint_aware_h3_p4": int(safe_calendar[week]),
                        "compact_nominal_h8_p1": int(compact_calendar[week]),
                    }
                    total += 1
                    if len(set(actions.values())) > 1:
                        disagreements += 1
                        records.append({
                            "cell": cell_id,
                            "tape": tape,
                            "learner_seed": int(learner_seed),
                            "week": week,
                            "observed_count_c_history": list(counts[:week]),
                            "actions": actions,
                        })
            if offset < args.exact_frontier_tapes:
                panel = simulate_full_des_frontier(skeleton=skeleton, scheduler=sched, calendars=full_action_calendars())
                delta = float(np.max(panel["ret_visible"]) - structured_ret[offset])
                frontier_deltas.append(delta); exact_cell.append(delta)
        per_cell[cell_id] = {
            "disagreement_fraction": disagreements / total,
            "selector_oracle_delta": {"mean": float(selector.mean()), "favorable_fraction": float(np.mean(selector > 0.0))},
            "exact_static_continuation_upper_bound": None if not exact_cell else float(np.mean(exact_cell)),
            "provider_provenance": {
                primary_id: "canonical_ret_transducer",
                safe_id: "canonical_ret_transducer",
                compact_id: "compact_auxiliary",
            },
        }
    selector_mean, selector_lcb = _mean_lcb(np.asarray(selector_deltas))
    selector_promoted = selector_mean >= 0.02
    payload = {
        "schema_version": "program_t_d0_d2_residual_atlas_v1",
        "claim_status": "EXPLORATORY_NO_CLAIM",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "burned_seed_range": [7490001, 7490256],
        "n_tapes": args.n_tapes,
        "provider_rule": "canonical and compact results are never pooled as one benchmark",
        "per_cell": per_cell,
        "bounds": {
            "q_mpc_selector_oracle": {"mean": selector_mean, "lcb95": selector_lcb, "promotion_ceiling_reached": selector_promoted},
            "exact_static_continuation_upper_bound": None if not frontier_deltas else {"mean": float(np.mean(frontier_deltas)), "n": len(frontier_deltas)},
        },
        "disagreement_record_count": len(records),
        "disagreement_records": records if args.include_records else records[:20],
        "records_truncated": not args.include_records and len(records) > 20,
        "verdict": "TRAIN_CROSSFITTED_SELECTOR" if selector_promoted else "STOP_Q_MPC_SELECTOR_CEILING_BELOW_0_02",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"per_cell": per_cell, "bounds": payload["bounds"], "verdict": payload["verdict"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
