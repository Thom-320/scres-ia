#!/usr/bin/env python3
"""Run the frozen Program V pre-learner mechanism gate.

The runner persists one raw row per seed-policy pair.  Selection chooses a
constant and an observable policy using only seeds 8701001--8701030; every
claim interval is then computed once on seeds 8701031--8701060.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from supply_chain.program_v_supplier_memory import (
    ACTIONS,
    HORIZON,
    SUPPLIERS,
    WEEKLY_ORDER,
    avoid_action,
    make_tape,
    policy_library,
    simulate,
)


DEFAULT_OUTPUT = ROOT / "results/program_v/prelearner_gate_v1"
SELECTION_SEEDS = tuple(range(8701001, 8701031))
EVALUATION_SEEDS = tuple(range(8701031, 8701061))
RAW_COLUMNS = (
    "split", "seed", "policy", "family", "deployable", "service", "delivered",
    "demanded", "inventory_final", "backlog_final", "backlog_auc",
    "mean_recovery_weeks", "ordered", "received", "rejected", "mass_residual",
    "action_switches", "unique_actions", "posterior_confidence_mean", "tape_sha256",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def paired_interval(rows: list[dict[str, object]], lhs: str, rhs: str,
                    metric: str = "service") -> dict[str, float | int | str]:
    by_policy = {
        policy: {int(row["seed"]): float(row[metric]) for row in rows if row["policy"] == policy}
        for policy in (lhs, rhs)
    }
    seeds = sorted(set(by_policy[lhs]) & set(by_policy[rhs]))
    differences = np.asarray([by_policy[lhs][seed] - by_policy[rhs][seed] for seed in seeds])
    mean = float(differences.mean())
    if len(differences) > 1 and float(differences.std(ddof=1)) > 1e-12:
        sem = float(stats.sem(differences))
        critical = float(stats.t.ppf(0.975, len(differences) - 1))
        lcb, ucb = mean - critical * sem, mean + critical * sem
        p_greater = float(stats.ttest_1samp(differences, popmean=0.0, alternative="greater").pvalue)
    else:
        lcb = ucb = mean
        p_greater = 0.0 if mean > 0.0 else 1.0
    return {
        "lhs": lhs, "rhs": rhs, "metric": metric, "n_pairs": len(seeds),
        "mean": mean, "lcb95": float(lcb), "ucb95": float(ucb),
        "p_one_sided_gt_zero": p_greater,
    }


def policy_means(rows: list[dict[str, object]], split: str) -> dict[str, dict[str, float]]:
    subset = [row for row in rows if row["split"] == split]
    output: dict[str, dict[str, float]] = {}
    for policy in sorted({str(row["policy"]) for row in subset}):
        selected = [row for row in subset if row["policy"] == policy]
        output[policy] = {
            metric: float(np.mean([float(row[metric]) for row in selected]))
            for metric in (
                "service", "backlog_final", "backlog_auc", "mean_recovery_weeks",
                "received", "rejected", "action_switches", "unique_actions",
            )
        }
    return output


def write_raw(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw_handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, mtime=0) as gz_handle:
            with io.TextIOWrapper(gz_handle, encoding="utf-8", newline="") as text_handle:
                writer = csv.DictWriter(text_handle, fieldnames=RAW_COLUMNS, lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)


def run(output: Path) -> dict[str, object]:
    started = time.perf_counter()
    policies = policy_library()
    rows: list[dict[str, object]] = []
    tapes = {}
    for split, seeds in (("selection", SELECTION_SEEDS), ("evaluation", EVALUATION_SEEDS)):
        for seed in seeds:
            tape = make_tape(seed)
            tapes[seed] = tape
            for policy in policies:
                result = simulate(tape, policy)
                rows.append({
                    "split": split, "seed": seed, "policy": policy.name,
                    "family": policy.family, "deployable": policy.deployable,
                    **result,
                })

    means = {
        "selection": policy_means(rows, "selection"),
        "evaluation": policy_means(rows, "evaluation"),
    }
    by_name = {policy.name: policy for policy in policies}
    best_constant = max(
        (policy.name for policy in policies if policy.family == "constant"),
        key=lambda name: (means["selection"][name]["service"], name),
    )
    observable_candidates = tuple(
        policy.name for policy in policies
        if policy.deployable and policy.family not in {"constant", "placebo", "belief_reset"}
    )
    best_observable = max(
        observable_candidates,
        key=lambda name: (means["selection"][name]["service"], name),
    )
    evaluation_rows = [row for row in rows if row["split"] == "evaluation"]
    effects = {
        "H_priv": paired_interval(evaluation_rows, "privileged_true_state", best_constant),
        "H_obs": paired_interval(evaluation_rows, best_observable, best_constant),
        "H_ret": paired_interval(evaluation_rows, "bayes_retained", "bayes_reset"),
        "retained_vs_delayed": paired_interval(evaluation_rows, "bayes_retained", "placebo_delayed"),
        "retained_vs_shuffled": paired_interval(evaluation_rows, "bayes_retained", "placebo_shuffled"),
        "privileged_residual_over_bayes": paired_interval(
            evaluation_rows, "privileged_true_state", "bayes_retained"
        ),
        "privileged_residual_over_selected": paired_interval(
            evaluation_rows, "privileged_true_state", best_observable
        ),
    }

    all_eval_tapes = [tapes[seed] for seed in EVALUATION_SEEDS]
    warning_hits = [
        warning == regime
        for tape in all_eval_tapes
        for warning, regime in zip(tape.warnings, tape.regimes)
    ]
    degraded_yields, healthy_yields = [], []
    oracle_dominant = True
    for tape in all_eval_tapes:
        for regime, yield_vector in zip(tape.regimes, tape.yields):
            degraded_yields.append(yield_vector[regime])
            healthy_yields.extend(y for supplier, y in enumerate(yield_vector) if supplier != regime)
            scores = {action: float(np.dot(action, yield_vector)) for action in ACTIONS}
            oracle_dominant &= avoid_action(regime) == max(scores, key=lambda action: (scores[action], action))

    expected_rows = len(policies) * (len(SELECTION_SEEDS) + len(EVALUATION_SEEDS))
    tape_counts = {
        seed: len({row["tape_sha256"] for row in rows if int(row["seed"]) == seed})
        for seed in (*SELECTION_SEEDS, *EVALUATION_SEEDS)
    }
    eval_base = means["evaluation"][best_constant]
    eval_obs = means["evaluation"][best_observable]
    falsifiers = {
        "expected_raw_rows": {"passed": len(rows) == expected_rows, "observed": len(rows),
                              "required": expected_rows},
        "policy_library_fixed": {"passed": len(policies) == 13 and len(ACTIONS) == 6,
                                 "policies": [policy.name for policy in policies]},
        "selection_evaluation_disjoint": {
            "passed": set(SELECTION_SEEDS).isdisjoint(EVALUATION_SEEDS),
            "selection": [min(SELECTION_SEEDS), max(SELECTION_SEEDS)],
            "evaluation": [min(EVALUATION_SEEDS), max(EVALUATION_SEEDS)],
        },
        "crn_one_tape_per_seed": {"passed": max(tape_counts.values()) == 1,
                                  "max_distinct_tapes": max(tape_counts.values())},
        "mass_conservation": {
            "passed": max(abs(float(row["mass_residual"])) for row in rows) <= 1e-9,
            "max_abs_residual": max(abs(float(row["mass_residual"])) for row in rows),
            "tolerance": 1e-9,
        },
        "fixed_order_rights": {
            "passed": len({float(row["ordered"]) for row in rows}) == 1,
            "observed": sorted({float(row["ordered"]) for row in rows}),
            "required": WEEKLY_ORDER * HORIZON,
        },
        "oracle_action_reversal_live": {
            "passed": len({avoid_action(i) for i in range(len(SUPPLIERS))}) == 3 and oracle_dominant,
            "actions": [avoid_action(i) for i in range(len(SUPPLIERS))],
            "dominant_on_every_evaluation_tape_week": oracle_dominant,
        },
        "warning_imperfect_but_informative": {
            "passed": 0.34 < float(np.mean(warning_hits)) < 0.95,
            "observed_accuracy": float(np.mean(warning_hits)),
        },
        "supplier_yield_mechanism_moves": {
            "passed": float(np.mean(healthy_yields) - np.mean(degraded_yields)) > 0.5,
            "healthy_mean": float(np.mean(healthy_yields)),
            "degraded_mean": float(np.mean(degraded_yields)),
        },
        "seasonal_demand_nonflat": {
            "passed": float(np.std(np.mean([tape.demand for tape in all_eval_tapes], axis=0))) > 100.0,
            "std_weekly_mean": float(np.std(np.mean([tape.demand for tape in all_eval_tapes], axis=0))),
        },
        "privileged_policy_not_deployable": {
            "passed": not by_name["privileged_true_state"].deployable,
        },
        "selected_policy_guardrails": {
            "passed": (
                eval_obs["backlog_final"] <= eval_base["backlog_final"]
                and eval_obs["mean_recovery_weeks"] <= eval_base["mean_recovery_weeks"]
            ),
            "selected": best_observable,
            "backlog_final_selected": eval_obs["backlog_final"],
            "backlog_final_constant": eval_base["backlog_final"],
            "recovery_selected": eval_obs["mean_recovery_weeks"],
            "recovery_constant": eval_base["mean_recovery_weeks"],
        },
    }
    falsifiers["all_passed"] = all(check["passed"] for check in falsifiers.values())

    mechanism_pass = (
        falsifiers["all_passed"]
        and float(effects["H_priv"]["lcb95"]) >= 0.02
        and float(effects["H_obs"]["lcb95"]) >= 0.01
        and float(effects["H_ret"]["lcb95"]) > 0.0
        and float(effects["retained_vs_delayed"]["lcb95"]) > 0.0
        and float(effects["retained_vs_shuffled"]["lcb95"]) > 0.0
    )
    if not falsifiers["all_passed"]:
        decision = "HALTED_PROGRAM_V_FALSIFIER_FAILED"
    elif float(effects["H_priv"]["lcb95"]) < 0.02:
        decision = "STOP_PROGRAM_V_NO_PHYSICAL_HEADROOM"
    elif float(effects["H_obs"]["lcb95"]) < 0.01:
        decision = "STOP_PROGRAM_V_HEADROOM_NOT_OBSERVABLE"
    elif (
        float(effects["H_ret"]["lcb95"]) <= 0.0
        or float(effects["retained_vs_delayed"]["lcb95"]) <= 0.0
        or float(effects["retained_vs_shuffled"]["lcb95"]) <= 0.0
    ):
        decision = "STOP_PROGRAM_V_NO_RETAINED_HISTORY_VALUE"
    elif float(effects["privileged_residual_over_bayes"]["ucb95"]) <= 0.01:
        decision = "STRUCTURED_BELIEF_SUFFICIENT_FOR_QUALITY"
    else:
        decision = "AUTHORIZE_PROGRAM_V_PLANNER_TIMING_GATE"

    raw_path = output / "raw_rows.csv.gz"
    write_raw(raw_path, rows)
    try:
        raw_label = str(raw_path.relative_to(ROOT))
    except ValueError:
        raw_label = str(raw_path)
    source_paths = (
        ROOT / "contracts/program_v_supplier_memory_v1.json",
        ROOT / "docs/PREREGISTRO_PROGRAM_V_MEMORIA_CARTERA_PROVEEDORES_2026-08-08.md",
        ROOT / "scripts/run_program_v_prelearner_gate_v1.py",
        ROOT / "supply_chain/program_v_supplier_memory.py",
        ROOT / "tests/test_program_v_supplier_memory.py",
    )
    result: dict[str, object] = {
        "schema_version": "program_v_supplier_memory_prelearner_gate_v1",
        "decision": decision,
        "mechanism_gate_passed": mechanism_pass,
        "neural_training_authorized": False,
        "interpretation": (
            "Retained history is causally useful, but a structured belief/last-yield comparator "
            "absorbs the quality headroom. Proceed only to a preregistered planner timing and "
            "amortization gate; this result is not a neural quality premium."
            if decision == "STRUCTURED_BELIEF_SUFFICIENT_FOR_QUALITY" else
            "Decision follows the frozen Program V partition."
        ),
        "selection": {"best_constant": best_constant, "best_observable": best_observable,
                      "observable_candidates": list(observable_candidates)},
        "effects_evaluation": effects,
        "policy_means": means,
        "falsifiers": falsifiers,
        "artifacts": {
            "raw_rows": raw_label,
            "raw_rows_sha256": sha256_file(raw_path),
            "raw_row_count": len(rows),
        },
        "source_manifest_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path) for path in source_paths
        },
        "seed_custody": {
            "selection": [min(SELECTION_SEEDS), max(SELECTION_SEEDS)],
            "evaluation": [min(EVALUATION_SEEDS), max(EVALUATION_SEEDS)],
            "fresh_confirmation_seeds_used": False,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(args.output.resolve())
    print(result["decision"])
    for name, effect in result["effects_evaluation"].items():
        print(f"{name}: {effect['mean']:+.6f} [{effect['lcb95']:+.6f}, {effect['ucb95']:+.6f}]")
    print(f"raw: {result['artifacts']['raw_rows']} ({result['artifacts']['raw_row_count']} rows)")
    return 0 if result["falsifiers"]["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
