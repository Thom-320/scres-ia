#!/usr/bin/env python3
"""Retrospective exact 3^3 signal-to-posture mapping screen on burned tapes."""
from __future__ import annotations

from itertools import product
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.paper2_bottleneck import (  # noqa: E402
    ACTIONS,
    ACTION_NAMES,
    CONTEXTS,
    materialize_tape,
    run_policy,
)


ROOT = Path(__file__).resolve().parent.parent


def mapped_policy(mapping):
    def policy(observation):
        scores = (
            observation["equipment_condition_score"],
            observation["route_threat_score"],
            observation["mission_tempo_score"],
        )
        return mapping[max(range(3), key=lambda index: (scores[index], -index))]
    return policy


def constant(action):
    return lambda observation: action


def ci95(values, seed=20260713, n_boot=4000):
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    boot = np.asarray([
        rng.choice(x, len(x), replace=True).mean() for _ in range(n_boot)
    ])
    return [float(x.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]


def main() -> int:
    calibration = [
        materialize_tape(1_100_001 + index, CONTEXTS[index % 3], "calibration", weeks=24)
        for index in range(60)
    ]
    candidates = []
    for indices in product(range(3), repeat=3):
        mapping = tuple(ACTIONS[index] for index in indices)
        rows = [run_policy(tape, mapped_policy(mapping)) for tape in calibration]
        candidates.append({
            "mapping": "".join(ACTION_NAMES[ACTIONS[index]] for index in indices),
            "mean_ret_excel": float(np.mean([row["ret_excel"] for row in rows])),
            "mean_service_loss_auc": float(np.mean([row["service_loss_auc_ration_hours"] for row in rows])),
        })
    candidates.sort(key=lambda row: (row["mean_ret_excel"], row["mapping"]), reverse=True)
    best_name = candidates[0]["mapping"]
    name_to_action = {name: action for action, name in ACTION_NAMES.items()}
    best_mapping = tuple(name_to_action[name] for name in best_name)

    locked = [
        materialize_tape(1_110_001 + index, CONTEXTS[index % 3], "locked", weeks=24)
        for index in range(120)
    ]
    best_rows = [run_policy(tape, mapped_policy(best_mapping)) for tape in locked]
    m_rows = [run_policy(tape, constant(ACTIONS[0])) for tape in locked]
    ret_delta = [row["ret_excel"] - base["ret_excel"] for row, base in zip(best_rows, m_rows)]
    service_delta = [
        (base["service_loss_auc_ration_hours"] - row["service_loss_auc_ration_hours"])
        / max(abs(base["service_loss_auc_ration_hours"]), 1.0)
        for row, base in zip(best_rows, m_rows)
    ]
    lost_delta = [row["n_lost"] - base["n_lost"] for row, base in zip(best_rows, m_rows)]
    result = {
        "schema_version": "paper2_bottleneck_signal_mapping_audit_v1",
        "scientific_use": "retrospective exact 27-mapping screen on already-burned calibration and locked tapes; tested-policy delta, not optimized H_obs",
        "calibration": {
            "seed_start": 1_100_001,
            "n": 60,
            "candidate_count": 27,
            "top_candidates": candidates[:10],
            "selected_mapping_equipment_transport_mission": best_name,
        },
        "locked": {
            "seed_start": 1_110_001,
            "n": 120,
            "ret_minus_constant_M_ci95": ci95(ret_delta),
            "service_reduction_vs_constant_M_ci95": ci95(service_delta),
            "lost_minus_constant_M_ci95": ci95(lost_delta),
            "favorable_tapes": float(np.mean(np.asarray(ret_delta) > 0.0)),
        },
        "conclusion": "This finite interpretable mapping screen does not establish H_obs unless it beats constant M and the still-missing full-horizon open-loop frontier.",
    }
    output = ROOT / "results" / "paper2_bottleneck" / "signal_mapping_audit.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
