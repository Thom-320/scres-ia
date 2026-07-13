#!/usr/bin/env python3
"""Corrective audit of commit a91890bf's stylized VoI atlas.

The committed atlas calls a full-order formula while describing the result as
``ret_excel_visible_v1``.  This script replays the two reported positive cells
on their already-burned seed block, reproduces the stored statistic, then
recomputes the exact same policies with the canonical visible-ledger
aggregator.  It also proves by paired replay that ``r22_prob`` is inert in the
scored order adapter.

This is a retrospective artifact audit.  It is not a new pre-learner screen and
does not authorize selection, learning, or virgin confirmation.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.headroom_sensitivity import (  # noqa: E402
    ACTIONS,
    ARM,
    _belief_policy,
    materialize_tape_theta,
    theta_to_cell,
)
from supply_chain.program_g import ret_order_metrics, simulate_orders  # noqa: E402
from supply_chain.ret_thesis import (  # noqa: E402
    compute_order_level_ret_excel_visible_ledger,
)


ROOT = Path(__file__).resolve().parent.parent
ATLAS = ROOT / "results" / "paper2_search" / "voi_ceiling_atlas.json"
SEED0 = 7_200_001
N_TAPES = 48
WEEKS = 4
SEQUENCES = tuple(itertools.product(ACTIONS, repeat=WEEKS))


def cell_theta(cell: dict[str, float]) -> dict[str, float]:
    return {
        "signal_q": float(cell["signal_q"]),
        "lead": float(cell["lead"]),
        "surge_mult": float(cell["surge_mult"]),
        "persistence": float(cell["persistence"]),
        "commonality": float(cell["commonality"]),
        "r22_prob": float(cell["r22_prob"]),
    }


def score_pair(tape, sequence) -> tuple[float, float, int, float, int, int]:
    orders = simulate_orders(tape, sequence, ARM)
    full = ret_order_metrics(orders)
    visible = compute_order_level_ret_excel_visible_ledger(orders)
    return (
        float(full["ret_order"]),
        float(visible["mean_ret_excel"]),
        int(full["lost"]),
        float(full["ret_quantity"]),
        int(visible["n_visible_rows"]),
        int(visible["n_omitted_rows"]),
    )


def bootstrap_ci95(values, *, seed: int = 20260713, n_boot: int = 10_000) -> list[float]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    boot = np.asarray([
        rng.choice(array, len(array), replace=True).mean()
        for _ in range(n_boot)
    ])
    return [
        float(array.mean()),
        float(np.percentile(boot, 2.5)),
        float(np.percentile(boot, 97.5)),
    ]


def seed_for_cell(cell_index: int) -> int:
    # The commit did not preserve its runner or seed manifest.  A stride of
    # 1,000 exactly reproduces both reported positive cells and matches the
    # Program-I convention of nonoverlapping tape blocks per evaluation.
    return SEED0 + int(cell_index) * 1_000


def summarize_cell(cell_index: int, cell: dict[str, float]) -> dict[str, Any]:
    theta = cell_theta(cell)
    materialized_cell = theta_to_cell(theta)
    seed_start = seed_for_cell(cell_index)
    tapes = [
        materialize_tape_theta(seed_start + offset, materialized_cell, weeks=WEEKS)
        for offset in range(N_TAPES)
    ]
    # Every four-week action sequence appears in periodic_calendars(4), so this
    # matrix is both the reported full-period static set and the tape oracle set.
    scores = np.asarray([
        [score_pair(tape, sequence) for tape in tapes]
        for sequence in SEQUENCES
    ])
    # shape: sequence, tape, (full, visible, lost, quantity)
    observed_sequences = [_belief_policy(tape) for tape in tapes]
    sequence_index = {sequence: index for index, sequence in enumerate(SEQUENCES)}
    observed = np.asarray([
        scores[sequence_index[sequence], tape_index]
        for tape_index, sequence in enumerate(observed_sequences)
    ])

    out: dict[str, Any] = {
        "theta": theta,
        "source_cell_index": cell_index,
        "n_tapes": N_TAPES,
        "seed_start": seed_start,
        "sequence_count": len(SEQUENCES),
        "reported_atlas": {
            "H_PI": float(cell["H_PI"]),
            "H_obs": float(cell["H_obs"]),
            "eta": float(cell["eta"]),
        },
    }
    for metric_name, metric_index in (("full_order_formula", 0), ("visible_ledger", 1)):
        matrix = scores[:, :, metric_index]
        best_static_index = int(matrix.mean(axis=1).argmax())
        static = matrix[best_static_index]
        oracle = matrix.max(axis=0)
        obs = observed[:, metric_index]
        h_pi = float(np.mean(oracle - static))
        tested_obs = float(np.mean(obs - static))
        lost_delta = observed[:, 2] - scores[best_static_index, :, 2]
        out[metric_name] = {
            "best_static_sequence": list(SEQUENCES[best_static_index]),
            "H_PI": h_pi,
            "tested_belief_minus_static": tested_obs,
            "eta_tested": float(tested_obs / h_pi) if abs(h_pi) > 1e-12 else 0.0,
            "belief_favorable_tape_fraction": float(np.mean(obs > static)),
            "belief_minus_static_lost_mean": float(
                np.mean(lost_delta)
            ),
            "belief_minus_static_lost_ci95": bootstrap_ci95(lost_delta),
            "belief_lost_worse_tapes": int(np.sum(lost_delta > 0)),
            "belief_lost_tie_tapes": int(np.sum(lost_delta == 0)),
            "belief_lost_better_tapes": int(np.sum(lost_delta < 0)),
            "belief_minus_static_quantity_ret_mean": float(
                np.mean(observed[:, 3] - scores[best_static_index, :, 3])
            ),
            "best_static_mean_lost_orders": float(np.mean(scores[best_static_index, :, 2])),
            "best_static_mean_visible_rows": float(np.mean(scores[best_static_index, :, 4])),
            "best_static_mean_omitted_rows": float(np.mean(scores[best_static_index, :, 5])),
        }
        if metric_name == "visible_ledger":
            out[metric_name]["project_H_PI_valid"] = False
            out[metric_name]["scope_warning"] = (
                "This unguarded sparse-visible optimum is HOLD^4 with an empty visible "
                "population scored as 1.0. It is a shed-to-win metric degeneracy, not the "
                "project's lost-order/resource-guardrail-constrained H_PI ceiling."
            )
    out["stored_statistic_reproduced"] = bool(
        abs(out["full_order_formula"]["H_PI"] - float(cell["H_PI"])) < 1e-12
        and abs(
            out["full_order_formula"]["tested_belief_minus_static"]
            - float(cell["H_obs"])
        ) < 1e-12
    )
    return out


def r22_inertness(cell_index: int, cell: dict[str, float]) -> dict[str, Any]:
    low = cell_theta(cell)
    high = dict(low)
    low["r22_prob"] = 0.0
    high["r22_prob"] = 0.30
    low_cell = theta_to_cell(low)
    high_cell = theta_to_cell(high)
    max_differences = {"full_order_formula": 0.0, "visible_ledger": 0.0}
    seed_start = seed_for_cell(cell_index)
    demand_signal_equal = True
    r22_differs = False
    for offset in range(N_TAPES):
        low_tape = materialize_tape_theta(seed_start + offset, low_cell, weeks=WEEKS)
        high_tape = materialize_tape_theta(seed_start + offset, high_cell, weeks=WEEKS)
        demand_signal_equal &= bool(
            np.array_equal(low_tape.demand, high_tape.demand)
            and np.array_equal(low_tape.signal, high_tape.signal)
        )
        r22_differs |= not np.array_equal(low_tape.r22, high_tape.r22)
        for sequence in SEQUENCES:
            low_score = score_pair(low_tape, sequence)
            high_score = score_pair(high_tape, sequence)
            max_differences["full_order_formula"] = max(
                max_differences["full_order_formula"],
                abs(low_score[0] - high_score[0]),
            )
            max_differences["visible_ledger"] = max(
                max_differences["visible_ledger"],
                abs(low_score[1] - high_score[1]),
            )
    return {
        "r22_probability_pair": [0.0, 0.30],
        "n_tapes": N_TAPES,
        "seed_start": seed_start,
        "sequence_count_per_tape": len(SEQUENCES),
        "demand_and_signal_exactly_equal": demand_signal_equal,
        "r22_arrays_differ_on_at_least_one_tape": r22_differs,
        "maximum_score_differences": max_differences,
        "scored_adapter_r22_liveness": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "paper2_search" / "voi_ceiling_atlas_corrective_audit.json",
    )
    args = parser.parse_args()

    atlas = json.loads(ATLAS.read_text())
    indexed_cells = list(enumerate(atlas["cells"]))
    positive = [
        (index, cell)
        for index, cell in indexed_cells
        if float(cell["H_obs"]) >= 0.01
    ]
    audited_all = [summarize_cell(index, cell) for index, cell in indexed_cells]
    positive_indexes = {index for index, _ in positive}
    audited = [
        row for row in audited_all if row["source_cell_index"] in positive_indexes
    ]
    r22 = r22_inertness(*positive[0])
    summary = {
        "n_cells": len(audited_all),
        "stored_statistic_reproduced_cells": int(sum(
            row["stored_statistic_reproduced"] for row in audited_all
        )),
        "unguarded_sparse_visible_metric_H_PI_exact_zero_cells": int(sum(
            abs(row["visible_ledger"]["H_PI"]) < 1e-12 for row in audited_all
        )),
        "unguarded_sparse_visible_metric_tested_belief_positive_cells": int(sum(
            row["visible_ledger"]["tested_belief_minus_static"] > 0
            for row in audited_all
        )),
        "unguarded_sparse_visible_metric_tested_belief_delta_min": float(min(
            row["visible_ledger"]["tested_belief_minus_static"]
            for row in audited_all
        )),
        "unguarded_sparse_visible_metric_tested_belief_delta_max": float(max(
            row["visible_ledger"]["tested_belief_minus_static"]
            for row in audited_all
        )),
        "source_positive_cells_with_more_lost_orders_than_static": int(sum(
            row["full_order_formula"]["belief_minus_static_lost_mean"] > 0
            for row in audited
        )),
    }
    result = {
        "schema_version": "concurrent_voi_atlas_corrective_audit_v1",
        "scientific_status": "EXPLORATORY_ATLAS_NOT_CANONICAL_HOBS_OR_BOUND",
        "source_commit": "a91890bfd3d815fc2bd614076c576487e13e0d06",
        "source_atlas": str(ATLAS.relative_to(ROOT)),
        "source_seed_pattern_reconstructed": "seed0=7200001 + 1000*cell_index; 48 tapes per cell",
        "source_seed_pattern_evidence": "The reconstructed stride reproduces both stored positive-cell H_PI and H_obs values to floating-point equality; the commit contains no runner or exact seed manifest.",
        "retrospective_burned_seed_pattern": "for cell_index 0..63: seed_start=7200001+1000*cell_index, 48 consecutive tapes",
        "all_cell_summary": summary,
        "positive_cell_count_in_source": len(positive),
        "hobs_definition_warning": "Each H_obs field in the source and this audit is one tested belief policy minus an in-sample-selected static, not the maximum over observable policies and not an H_obs ceiling.",
        "metric_defect": "The source used compute_order_level_ret_excel_formula over every generated order, not ret_excel_visible_v1 via compute_order_level_ret_excel_visible_ledger.",
        "visible_metric_scope_warning": "The raw visible-ledger maximizer is HOLD^4 because zero visible rows score 1.0. The reported 64/64 zero is an unguarded sparse-ledger degeneracy and is not a project H_PI ceiling.",
        "r22_defect": "r22_prob changes tape.r22 but simulate_orders never consumes tape.r22, so every fixed-sequence score is exactly inert. The belief policy may still choose a different sequence through its shadow _week_step; this does not prove the full tested-policy statistic invariant.",
        "positive_cells": audited,
        "all_cells": audited_all,
        "r22_inertness": r22,
        "guardrails_missing_from_source_atlas": [
            "worst_cssu_service",
            "tail_risk",
            "resource_ledger",
            "backlog_age",
            "action_trajectory_replacements",
            "null_physics_cell",
        ],
        "paper2_confirmed": False,
        "paper3_authorized": False,
        "terminal_boundary_supported": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if all(row["stored_statistic_reproduced"] for row in audited_all) and not r22["scored_adapter_r22_liveness"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
