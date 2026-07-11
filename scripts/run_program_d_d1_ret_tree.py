#!/usr/bin/env python3
"""Cross-fitted, sequential D1-ReT observable-policy gate.

This runner implements the prospectively frozen ReT-only estimand. It consumes
only the already-open D1 calibration/validation branching artifacts and never
loads the virgin universe.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import joblib
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.tree import DecisionTreeClassifier, export_text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_program_d_d1_branching import features, make_env  # noqa: E402
from supply_chain.config import HOURS_PER_DAY  # noqa: E402
from supply_chain.episode_metrics import compute_episode_metrics  # noqa: E402
from supply_chain.l_program_env import CampaignTape  # noqa: E402
from supply_chain.program_d import PROXY_PATH, RULES, exogenous_hash, paired_bootstrap  # noqa: E402


DEFAULT_BRANCH = Path("results/program_d/d1_v3_visible_branching")
DEFAULT_FRONTIER = Path("results/program_d/d1_v3_visible_frontier")
DEFAULT_OUTPUT = Path("results/program_d/d1_ret_only_tree")
COMPARATOR = "spt_flat"
NUMERIC_FEATURES = (
    "sb_inventory", "queue_qty", "queue_count", "queue_occupancy",
    "contingent_share", "size_p25", "size_p50", "size_p75", "age_p25",
    "age_p50", "age_p75", "oldest_age", "in_transit", "op9_down",
    "op10_down", "op11_down", "op12_down", "recent_demand",
    "recent_delivered", "recent_fill", "operational_day",
)
FEATURE_NAMES = NUMERIC_FEATURES + tuple(f"prior_rule__{rule}" for rule in RULES)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def vector(row: dict[str, Any]) -> list[float]:
    prior = str(row.get("prior_rule", COMPARATOR))
    return [float(row[name]) for name in NUMERIC_FEATURES] + [
        float(prior == rule) for rule in RULES
    ]


def load_tapes(frontier_dir: Path) -> list[CampaignTape]:
    tapes: list[CampaignTape] = []
    for split in ("selection", "validation"):
        payload = json.loads((frontier_dir / f"{split}_tapes.json").read_text())
        tapes.extend(CampaignTape.from_mapping(row) for row in payload)
    if len(tapes) != 60:
        raise RuntimeError(f"Expected 60 non-virgin tapes, found {len(tapes)}")
    if any(t.split not in {"selection", "validation"} for t in tapes):
        raise RuntimeError("Virgin or unknown split reached the D1-ReT loader")
    return tapes


def run_policy(tape: CampaignTape, model: DecisionTreeClassifier | None) -> dict[str, Any]:
    env = make_env(tape, COMPARATOR)
    sim = env.sim
    start = float(env._treatment_start)
    prior_rule = COMPARATOR
    backlog_auc = 0.0
    decisions: list[str] = []
    for _ in range(tape.horizon_weeks * 7):
        if model is None:
            rule = COMPARATOR
        else:
            rule = str(model.predict(np.asarray([vector(features(sim, start, prior_rule))]))[0])
        sim.set_backorder_priority_rule(rule)
        prior_rule = rule
        decisions.append(rule)
        backlog_auc += float(sim.pending_backorder_qty) * HOURS_PER_DAY
        sim.step(None, HOURS_PER_DAY)

    metrics = compute_episode_metrics(sim, treatment_start=start)
    ledger = sim.flow_ledger()
    metrics.update(
        {
            "backlog_auc_ration_hours_daily": float(backlog_auc),
            "raw_mass_residual": float(ledger["raw_residual"]),
            "ration_mass_residual": float(ledger["ration_residual"]),
            "mass_balance_residual": max(
                abs(float(ledger["raw_residual"])), abs(float(ledger["ration_residual"]))
            ),
            "risk_sha256": exogenous_hash(sim, start)["risk_sha256"],
            "demand_sha256": exogenous_hash(sim, start)["demand_sha256"],
            "n_rule_changes": int(sum(a != b for a, b in zip(decisions, decisions[1:]))),
            "rule_counts": json.dumps({rule: decisions.count(rule) for rule in RULES}, sort_keys=True),
        }
    )
    env.close()
    return metrics


def relative_degradation(tree: float, base: float) -> float:
    return (tree - base) / max(abs(base), 1.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch-dir", type=Path, default=DEFAULT_BRANCH)
    parser.add_argument("--frontier-dir", type=Path, default=DEFAULT_FRONTIER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-boot", type=int, default=10_000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    branch_verdict = json.loads((args.branch_dir / "verdict.json").read_text())
    if branch_verdict.get("ret_excel_contract_version") != "ret_excel_visible_v1":
        raise RuntimeError("D1-ReT requires the frozen visible-ledger branch artifact")
    if int(branch_verdict.get("virgin_tapes_opened", -1)) != 0:
        raise RuntimeError("Input branching artifact opened virgin tapes")
    if branch_verdict.get("comparator") != COMPARATOR:
        raise RuntimeError("Frozen comparator mismatch")

    states = read_csv(args.branch_dir / "states.csv")
    oracle = read_csv(args.branch_dir / "oracle_rows.csv")
    branches = read_csv(args.branch_dir / "branch_rows.csv")
    tapes = load_tapes(args.frontier_dir)
    tape_map = {t.campaign_id: t for t in tapes}
    state_map = {row["state_id"]: row for row in states}
    label_map = {row["state_id"]: row["rule"] for row in oracle}
    long_lookup = {
        (row["state_id"], row["rule"]): row
        for row in branches
        if abs(float(row["horizon_hours"]) - 28.0 * 24.0) < 1e-9
    }
    if set(state_map) != set(label_map):
        raise RuntimeError("State and oracle-label populations differ")

    ordered_ids = sorted(state_map)
    x = np.asarray([vector(state_map[sid]) for sid in ordered_ids], dtype=float)
    y = np.asarray([label_map[sid] for sid in ordered_ids])
    groups = np.asarray([state_map[sid]["tape_id"] for sid in ordered_ids])

    fold_rows: list[dict[str, Any]] = []
    tape_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for fold, (train_idx, test_idx) in enumerate(GroupKFold(n_splits=5).split(x, y, groups), 1):
        model = DecisionTreeClassifier(max_depth=3, random_state=0)
        model.fit(x[train_idx], y[train_idx])
        joblib.dump(model, args.output_dir / f"fold_{fold}_tree.joblib")
        (args.output_dir / f"fold_{fold}_tree.txt").write_text(
            export_text(model, feature_names=list(FEATURE_NAMES)), encoding="utf-8"
        )

        heldout_tapes = sorted(set(groups[test_idx]))
        predicted = model.predict(x[test_idx])
        for idx, rule in zip(test_idx, predicted):
            sid = ordered_ids[int(idx)]
            chosen = long_lookup[(sid, str(rule))]
            base = long_lookup[(sid, COMPARATOR)]
            best = long_lookup[(sid, label_map[sid])]
            prediction_rows.append(
                {
                    "fold": fold,
                    "state_id": sid,
                    "tape_id": groups[idx],
                    "predicted_rule": str(rule),
                    "oracle_rule": label_map[sid],
                    "chosen_ret": float(chosen["ret_excel"]),
                    "base_ret": float(base["ret_excel"]),
                    "oracle_ret": float(best["ret_excel"]),
                    "chosen_delta": float(chosen["ret_excel"]) - float(base["ret_excel"]),
                    "oracle_delta": float(best["ret_excel"]) - float(base["ret_excel"]),
                }
            )

        for tape_id in heldout_tapes:
            tape = tape_map[str(tape_id)]
            base = run_policy(tape, None)
            tree = run_policy(tape, model)
            if (base["risk_sha256"], base["demand_sha256"]) != (
                tree["risk_sha256"], tree["demand_sha256"]
            ):
                raise RuntimeError(f"FAIL_CLOSED exogenous mismatch for {tape_id}")
            if max(base["mass_balance_residual"], tree["mass_balance_residual"]) > 1e-6:
                raise RuntimeError(f"FAIL_CLOSED mass-balance mismatch for {tape_id}")
            tape_rows.append(
                {
                    "fold": fold,
                    "tape_id": tape_id,
                    "tape_sha256": tape.digest(),
                    "family": tape.family,
                    "risk_level": tape.risk_level,
                    "tree_ret_excel": tree["ret_excel"],
                    "base_ret_excel": base["ret_excel"],
                    "delta_ret_excel": tree["ret_excel"] - base["ret_excel"],
                    "tree_ret_clipped": tree["ret_excel_visible_clipped_0_1"],
                    "base_ret_clipped": base["ret_excel_visible_clipped_0_1"],
                    "delta_ret_clipped": tree["ret_excel_visible_clipped_0_1"] - base["ret_excel_visible_clipped_0_1"],
                    "tree_service_loss": tree["service_loss_auc_ration_hours"],
                    "base_service_loss": base["service_loss_auc_ration_hours"],
                    "service_loss_degradation": relative_degradation(tree["service_loss_auc_ration_hours"], base["service_loss_auc_ration_hours"]),
                    "tree_lost": tree["lost_orders"],
                    "base_lost": base["lost_orders"],
                    "lost_degradation": relative_degradation(tree["lost_orders"], base["lost_orders"]),
                    "tree_backlog_auc": tree["backlog_auc_ration_hours_daily"],
                    "base_backlog_auc": base["backlog_auc_ration_hours_daily"],
                    "backlog_degradation": relative_degradation(tree["backlog_auc_ration_hours_daily"], base["backlog_auc_ration_hours_daily"]),
                    "tree_rule_changes": tree["n_rule_changes"],
                    "tree_rule_counts": tree["rule_counts"],
                    "risk_sha256": tree["risk_sha256"],
                    "demand_sha256": tree["demand_sha256"],
                    "mass_balance_residual": max(base["mass_balance_residual"], tree["mass_balance_residual"]),
                }
            )
        fold_rows.append(
            {
                "fold": fold,
                "n_train_states": len(train_idx),
                "n_test_states": len(test_idx),
                "n_test_tapes": len(heldout_tapes),
                "depth": model.get_depth(),
                "leaves": model.get_n_leaves(),
                "accuracy": float(np.mean(predicted == y[test_idx])),
            }
        )
        print(f"[d1-ret-tree] fold {fold}/5 complete ({len(heldout_tapes)} held-out tapes)", flush=True)

    primary = paired_bootstrap([r["delta_ret_excel"] for r in tape_rows], seed=0xD141, n_boot=args.n_boot)
    clipped = paired_bootstrap([r["delta_ret_clipped"] for r in tape_rows], seed=0xD142, n_boot=args.n_boot)
    lost = paired_bootstrap([r["lost_degradation"] for r in tape_rows], seed=0xD143, n_boot=args.n_boot)
    service = paired_bootstrap([r["service_loss_degradation"] for r in tape_rows], seed=0xD144, n_boot=args.n_boot)
    backlog = paired_bootstrap([r["backlog_degradation"] for r in tape_rows], seed=0xD145, n_boot=args.n_boot)
    positive_share = float(np.mean([r["delta_ret_excel"] > 0 for r in tape_rows]))
    chosen_headroom = float(np.mean([r["chosen_delta"] for r in prediction_rows]))
    oracle_headroom = float(np.mean([r["oracle_delta"] for r in prediction_rows]))
    capture = chosen_headroom / oracle_headroom if oracle_headroom > 0 else float("nan")
    criteria = {
        "ret_ci_positive": primary["ci95"][0] > 0,
        "clipped_ret_ci_positive": clipped["ci95"][0] > 0,
        "positive_on_70pct_tapes": positive_share >= 0.70,
        "captures_50pct_oracle": capture >= 0.50,
        "lost_upper_ci_at_most_2pct": lost["ci95"][1] <= 0.02,
        "service_upper_ci_at_most_2pct": service["ci95"][1] <= 0.02,
        "backlog_upper_ci_at_most_2pct": backlog["ci95"][1] <= 0.02,
        "identity_and_conservation": all(r["mass_balance_residual"] <= 1e-6 for r in tape_rows),
    }
    promote = all(criteria.values())
    verdict = {
        "kind": "program_d_d1_ret_only_observable_tree",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha_input": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip(),
        "preregistration": "docs/PROGRAM_D_D1_RET_ONLY_TREE_PREREGISTRATION_2026-07-11.md",
        "proxy_sha256": sha256(PROXY_PATH.read_bytes()).hexdigest(),
        "comparator": COMPARATOR,
        "ret_excel_contract_version": "ret_excel_visible_v1",
        "n_tapes": len(tape_rows),
        "n_states": len(prediction_rows),
        "primary_delta": primary,
        "clipped_delta": clipped,
        "positive_tape_share": positive_share,
        "crossfit_branch_chosen_headroom": chosen_headroom,
        "crossfit_branch_oracle_headroom": oracle_headroom,
        "oracle_capture_ratio": capture,
        "lost_relative_degradation": lost,
        "service_loss_relative_degradation": service,
        "backlog_auc_relative_degradation": backlog,
        "criteria": criteria,
        "virgin_tapes_opened": 0,
        "ppo_trained": False,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__},
        "promoted": promote,
        "verdict": "PROMOTE_D1_RET_OBSERVABLE_POLICY" if promote else "STOP_D1_RET_NOT_OBSERVABLY_CONVERTIBLE",
    }
    write_csv(args.output_dir / "folds.csv", fold_rows)
    write_csv(args.output_dir / "crossfit_state_predictions.csv", prediction_rows)
    write_csv(args.output_dir / "sequential_tape_effects.csv", tape_rows)
    (args.output_dir / "verdict.json").write_text(json.dumps(verdict, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(verdict, indent=2, sort_keys=True))
    return 0 if promote else 2


if __name__ == "__main__":
    raise SystemExit(main())
