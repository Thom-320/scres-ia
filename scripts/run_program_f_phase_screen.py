#!/usr/bin/env python3
"""Execute the preregistered Program F 24-cell prelearner phase diagram."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.program_f import (
    ConstantPortfolio, CONTEXTS, actions_for_budget, branch_from_week,
    materialize_tape, profile_for_cell, run_policy,
)


FEATURES = (
    "equipment_condition_score", "route_threat_score", "mission_tempo_score",
    "equipment_condition", "recent_r11_count", "recent_transport_attack_count",
    "recent_r24_count", "recent_damage_hours", "sb_inventory", "reserve_inventory",
    "backlog_qty", "backlog_count", "active_m", "active_t", "active_r",
    "pending_m", "pending_t", "pending_r", "week_phase",
)


def state_weeks(tape: dict[str, Any]) -> list[int]:
    contexts = tape["context_schedule"]
    changes = [i for i in range(1, len(contexts)) if contexts[i] != contexts[i - 1]]
    first = changes[0] if changes else 6
    second = changes[1] if len(changes) > 1 else min(len(contexts) - 5, first + 6)
    candidates = [max(2, first - 1), first, min(first + 2, len(contexts) - 5), min(second, len(contexts) - 5)]
    result = []
    for value in candidates:
        if value not in result:
            result.append(value)
    while len(result) < 4:
        candidate = min(len(contexts) - 5, 2 + 3 * len(result))
        if candidate not in result:
            result.append(candidate)
        else:
            candidate += 1
            result.append(candidate)
    return result[:4]


def bootstrap_tape(values: dict[str, list[float]], seed: int, n: int = 2000) -> tuple[float, float, float]:
    tape_ids = sorted(values)
    tape_means = np.asarray([np.mean(values[tape]) for tape in tape_ids], dtype=float)
    rng = np.random.default_rng(seed)
    boot = [float(np.mean(tape_means[rng.integers(0, len(tape_means), len(tape_means))])) for _ in range(n)]
    return float(tape_means.mean()), float(np.quantile(boot, .025)), float(np.quantile(boot, .975))


def action_id(action: tuple[int, int, int]) -> str:
    return "".join(map(str, action))


def evaluate_cell(cell: dict[str, Any]) -> tuple[dict[str, Any], list[dict], list[dict]]:
    profile = profile_for_cell(cell)
    actions = actions_for_budget(profile["budget_tokens"])
    tapes = [
        materialize_tape(
            int(cell["screen_seed_start"]) + i, CONTEXTS[i % 3], "phase-screen",
            weeks=32, profile=profile,
        )
        for i in range(12)
    ]
    static_rows = []
    for tape in tapes:
        for action in actions:
            result = run_policy(tape, ConstantPortfolio(action))
            static_rows.append({
                "cell_id": cell["cell_id"], "tape_id": tape["tape_id"],
                "action": action_id(action), "ret": float(result["ret_excel"]),
                "service": float(result["service_loss_auc_ration_hours"]),
                "lost": float(result["n_lost"]), "mass": float(result["mass_residual"]),
                "reserve_issued": float(result["reserve_units_issued"]),
                "r11_hours": sum(e["realized_duration_hours"] for e in result["damage_events"] if e["risk_id"] == "R11"),
                "transport_hours": sum(e["realized_duration_hours"] for e in result["damage_events"] if e["risk_id"] in {"R22", "R23"}),
                "threat_sha256": result["threat_sha256"],
            })

    branch_rows = []
    for tape_index, tape in enumerate(tapes):
        for state_index, week in enumerate(state_weeks(tape)):
            prefix = actions[(tape_index * 4 + state_index) % len(actions)]
            for action in actions:
                result = branch_from_week(
                    tape, prefix_action=prefix, state_week=week,
                    branch_action=action, horizon_weeks=4,
                )
                branch_rows.append({
                    "cell_id": cell["cell_id"], "tape_id": tape["tape_id"],
                    "state_id": f"{tape['tape_id']}|w{week}|p{action_id(prefix)}",
                    "state_week": week, "prefix": action_id(prefix), "action": action_id(action),
                    "ret": result["ret"], "service": result["service"], "lost": result["lost"],
                    "observation": result["observation"], "mass": result["mass_residual"],
                })

    states = sorted({row["state_id"] for row in branch_rows})
    by_state_action = {(row["state_id"], row["action"]): row for row in branch_rows}
    action_names = [action_id(action) for action in actions]
    constant_means = {
        action: float(np.mean([by_state_action[(state, action)]["ret"] for state in states]))
        for action in action_names
    }
    best_constant = max(action_names, key=lambda action: (constant_means[action], action))
    labels, supports, oracle_delta_by_tape, oracle_service = [], {action: 0.0 for action in action_names}, {}, []
    for state in states:
        values = {action: by_state_action[(state, action)]["ret"] for action in action_names}
        best_value = max(values.values())
        winners = [action for action, value in values.items() if abs(value - best_value) <= 1e-10]
        for winner in winners:
            supports[winner] += 1.0 / len(winners)
        label = sorted(winners)[0]
        labels.append((state, label))
        tape_id = by_state_action[(state, label)]["tape_id"]
        delta = best_value - by_state_action[(state, best_constant)]["ret"]
        oracle_delta_by_tape.setdefault(tape_id, []).append(delta)
        base_service = by_state_action[(state, best_constant)]["service"]
        oracle_service.append((base_service - by_state_action[(state, label)]["service"]) / max(abs(base_service), 1.0))
    oracle_ci = bootstrap_tape(oracle_delta_by_tape, 2026071206 + int(cell["cell_id"].split("-")[-1]))
    support_fraction = {key: value / len(states) for key, value in supports.items()}

    # Tape-cross-fitted sequential tree rollout. Static comparator is selected
    # only on each fold's training tapes and then frozen for held-out tapes.
    label_by_state = dict(labels)
    X = np.asarray([[by_state_action[(state, action_names[0])]["observation"][key] for key in FEATURES] for state in states])
    y = np.asarray([action_names.index(label_by_state[state]) for state in states])
    groups = np.asarray([by_state_action[(state, action_names[0])]["tape_id"] for state in states])
    unique_tapes = sorted({tape["tape_id"] for tape in tapes})
    tape_by_id = {tape["tape_id"]: tape for tape in tapes}
    static_index = {(row["tape_id"], row["action"]): row for row in static_rows}
    tree_deltas = []
    for train_idx, test_idx in GroupKFold(n_splits=3).split(X, y, groups):
        model = DecisionTreeClassifier(max_depth=3, min_samples_leaf=3, random_state=20260712)
        model.fit(X[train_idx], y[train_idx])
        train_tapes = sorted(set(groups[train_idx]))
        test_tapes = sorted(set(groups[test_idx]))
        train_static = {
            action: float(np.mean([static_index[(tape_id, action)]["ret"] for tape_id in train_tapes]))
            for action in action_names
        }
        fold_static = max(action_names, key=lambda action: (train_static[action], action))

        def tree_policy(observation: dict[str, float]) -> tuple[int, int, int]:
            pred = int(model.predict(np.asarray([[observation[key] for key in FEATURES]], dtype=float))[0])
            return actions[pred]

        for tape_id in test_tapes:
            candidate = run_policy(tape_by_id[tape_id], tree_policy)
            tree_deltas.append(float(candidate["ret_excel"]) - static_index[(tape_id, fold_static)]["ret"])
    tree_delta = float(np.mean(tree_deltas))
    conversion = tree_delta / max(oracle_ci[0], 1e-12)

    grouped_static: dict[tuple[str, str], list[dict]] = {}
    for row in static_rows:
        grouped_static.setdefault((row["tape_id"], row["action"]), []).append(row)
    def total(key: str, action: str) -> float:
        return sum(row[key] for row in static_rows if row["action"] == action)
    max_m = max(action_names, key=lambda value: int(value[0])); min_m = min(action_names, key=lambda value: int(value[0]))
    max_t = max(action_names, key=lambda value: int(value[1])); min_t = min(action_names, key=lambda value: int(value[1]))
    max_r = max(action_names, key=lambda value: int(value[2])); min_r = min(action_names, key=lambda value: int(value[2]))
    liveness = {
        "manufacturing": total("r11_hours", max_m) < total("r11_hours", min_m),
        "transport": total("transport_hours", max_t) < total("transport_hours", min_t),
        "reserve": total("reserve_issued", max_r) > total("reserve_issued", min_r),
    }
    diverse = [value for value in support_fraction.values() if value >= 0.15]
    gates = {
        "all_three_levers_live": all(liveness.values()),
        "action_diversity": len(diverse) >= 2 and max(support_fraction.values()) <= 0.85,
        "oracle_headroom": oracle_ci[0] >= 0.01 and oracle_ci[1] > 0,
        "observable_conversion": conversion >= 0.50 and tree_delta > 0,
    }
    verdict = {
        **cell, "profile": profile, "n_actions": len(actions), "n_tapes": len(tapes),
        "n_states": len(states), "best_constant": best_constant,
        "support_fraction": support_fraction, "oracle_ret_delta_ci95": oracle_ci,
        "oracle_service_reduction_mean": float(np.mean(oracle_service)),
        "tree_ret_delta": tree_delta, "tree_conversion": conversion,
        "liveness": liveness, "gates": gates, "passes_preselection": all(gates.values()),
        "max_mass_residual": max(row["mass"] for row in static_rows + branch_rows),
        "threat_crn_pass": all(
            len({row["threat_sha256"] for row in static_rows if row["tape_id"] == tape["tape_id"]}) == 1
            for tape in tapes
        ),
    }
    return verdict, static_rows, branch_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    serial = []
    for row in rows:
        serial.append({key: json.dumps(value, sort_keys=True) if isinstance(value, dict) else value for key, value in row.items()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(serial[0]))
        writer.writeheader(); writer.writerows(serial)


def selection_key(row: dict[str, Any]) -> tuple:
    risk = {"current_all_contexts": 0, "dominant_increased_background_current": 1}[row["risk_amplitude"]]
    efficacy = {"low": 0, "base": 1, "high": 2}[row["efficacy_level"]]
    dwell = {(4, 8): 0, (3, 5): 1, (6, 10): 2}[tuple(row["context_dwell_weeks"])]
    return (risk, efficacy, abs(float(row["signal_accuracy"]) - .75), dwell, int(row["minimum_commitment_weeks"]), row["cell_id"])


def main() -> int:
    root = Path("results/program_f/screen")
    design = json.loads((root / "design.json").read_text())
    verdicts, all_static, all_branch = [], [], []
    for cell in design["cells"]:
        verdict, static_rows, branch_rows = evaluate_cell(cell)
        verdicts.append(verdict); all_static.extend(static_rows); all_branch.extend(branch_rows)
        print(f"[program-f-screen] {cell['cell_id']} pass={verdict['passes_preselection']} oracle={verdict['oracle_ret_delta_ci95'][0]:.6f} conversion={verdict['tree_conversion']:.3f}", flush=True)
    admissible = [row for row in verdicts if row["confirmatory_v1_admissible"] and row["passes_preselection"]]
    selected = min(admissible, key=selection_key) if admissible else None
    final = {
        "gate": "PROGRAM_F_PHASE_DIAGRAM_SCREEN", "design_sha256": design["design_sha256"],
        "n_cells": len(verdicts), "n_passing_cells": sum(row["passes_preselection"] for row in verdicts),
        "n_admissible_passing_cells": len(admissible),
        "selected_cell_id": None if selected is None else selected["cell_id"],
        "selection_rule": "frozen lowest-amplitude lexicographic; never maximum ReT",
        "calibration_tapes_opened": 0, "holdout_tapes_opened": 0, "virgin_tapes_opened": 0,
        "ppo_trained": False,
        "interpretation": "PROMOTE_PROGRAM_F_CELL_TO_CONFIRMATION" if selected else "STOP_PROGRAM_F_SCREEN",
        "cells": verdicts,
    }
    (root / "verdict.json").write_text(json.dumps(final, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(root / "static_rows.csv", all_static)
    write_csv(root / "branch_rows.csv", all_branch)
    print(json.dumps({key: final[key] for key in ("interpretation", "n_passing_cells", "n_admissible_passing_cells", "selected_cell_id")}, indent=2))
    return 0 if selected else 2


if __name__ == "__main__":
    raise SystemExit(main())
