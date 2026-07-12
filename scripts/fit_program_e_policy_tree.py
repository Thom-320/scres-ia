#!/usr/bin/env python3
"""Fit the preregistered depth-3 observable Program E policy tree."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.tree import DecisionTreeClassifier, export_text

from supply_chain.dra2_policy_env import OBSERVATION_KEYS


TREE_FEATURES = tuple(key for key in OBSERVATION_KEYS if key not in {
    "departures_to_date", "unavailable_hours_to_date"
})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle-dir", type=Path, required=True)
    parser.add_argument("--normalizers", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)

    states = pd.read_csv(args.oracle_dir / "states.csv")
    labels = pd.read_csv(args.oracle_dir / "oracle_state_summary.csv")
    data = states.merge(
        labels[["state_id", "supported_first_action_14d"]], on="state_id", how="inner"
    )
    data = data[data["supported_first_action_14d"] != "NONE_TIE_OR_ZERO"].copy()
    normalizers = json.loads(args.normalizers.read_text())
    scales = normalizers["observation_scales"]
    x = np.column_stack([
        np.clip(data[key].to_numpy(float) / max(float(scales[key]), 1e-9), -10, 10)
        for key in TREE_FEATURES
    ])
    y = (data["supported_first_action_14d"] == "DISPATCH_NOW").astype(int).to_numpy()
    groups = data["tape_id"].to_numpy()
    predictions = np.zeros_like(y)
    fold_rows = []
    splitter = GroupKFold(n_splits=5)
    for fold, (train, test) in enumerate(splitter.split(x, y, groups), start=1):
        model = DecisionTreeClassifier(max_depth=3, random_state=20260712)
        model.fit(x[train], y[train])
        predictions[test] = model.predict(x[test])
        fold_rows.append({
            "fold": fold, "n_train": len(train), "n_test": len(test),
            "accuracy": float(accuracy_score(y[test], predictions[test])),
        })
    final = DecisionTreeClassifier(max_depth=3, random_state=20260712)
    final.fit(x, y)
    joblib.dump({"model": final, "features": TREE_FEATURES, "scales": scales},
                args.output_dir / "policy_tree.joblib")
    (args.output_dir / "policy_tree.txt").write_text(
        export_text(final, feature_names=list(TREE_FEATURES)), encoding="utf-8"
    )
    pd.DataFrame({
        "state_id": data["state_id"], "tape_id": data["tape_id"],
        "label": y, "crossfit_prediction": predictions,
    }).to_csv(args.output_dir / "crossfit_predictions.csv", index=False)
    verdict = {
        "n_labeled_states": len(y),
        "n_ties_excluded": int(len(states) - len(y)),
        "crossfit_accuracy": float(accuracy_score(y, predictions)),
        "confusion_matrix": confusion_matrix(y, predictions, labels=[0, 1]).tolist(),
        "folds": fold_rows,
        "tree_depth": int(final.get_depth()),
        "tree_leaves": int(final.get_n_leaves()),
        "validation_tapes_opened": 0,
        "virgin_tapes_opened": 0,
        "interpretation": "PROGRAM_E_TREE_FIT_TRAINING_ONLY",
    }
    (args.output_dir / "verdict.json").write_text(
        json.dumps(verdict, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(verdict, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
