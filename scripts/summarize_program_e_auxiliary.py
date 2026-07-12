#!/usr/bin/env python3
"""Summarize frozen heuristic/tree validation results without opening new tapes."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.evaluate_program_e_validation import convex_mixture


ROOT = Path("results/program_e/validation")


def main() -> int:
    static = pd.read_csv(ROOT / "static_rows.csv")
    auxiliary = pd.read_csv(ROOT / "auxiliary_rows.csv")
    static_summary = [
        {
            "policy_id": policy_id,
            "mean_ret": float(group.ret.mean()),
            "mean_departures": float(group.departures.mean()),
            "mean_unavailable_hours": float(group.unavailable_hours.mean()),
        }
        for policy_id, group in static.groupby("policy_id", sort=False)
    ]
    static_index = {
        (row.policy_id, row.tape_id): row for row in static.itertuples()
    }
    summaries = []
    for policy, group in auxiliary.groupby("policy", sort=False):
        candidate = {
            "mean_ret": float(group.ret.mean()),
            "mean_service_loss_auc": float(group.service.mean()),
            "mean_lost_orders": float(group.lost.mean()),
            "mean_departures": float(group.departures.mean()),
            "mean_unavailable_hours": float(group.unavailable_hours.mean()),
        }
        summary = {"policy": policy, "candidate": candidate}
        try:
            mixture = convex_mixture(
                static_summary,
                candidate["mean_departures"],
                candidate["mean_unavailable_hours"],
            )
        except RuntimeError as error:
            by_policy = static.groupby("policy_id", as_index=False).agg(
                ret=("ret", "mean"), service=("service", "mean"),
                lost=("lost", "mean"), departures=("departures", "mean"),
                unavailable_hours=("unavailable_hours", "mean"),
            )
            baseline = by_policy.sort_values(
                ["departures", "unavailable_hours"]
            ).iloc[0]
            summary.update({
                "resource_comparator_exists": False,
                "reason": str(error),
                "descriptive_min_resource_static": baseline.to_dict(),
                "descriptive_ret_delta": float(group.ret.mean() - baseline.ret),
            })
        else:
            weights = mixture["weights"]
            deltas, reductions, lost_deltas = [], [], []
            for row in group.itertuples():
                baseline = {
                    metric: sum(
                        float(weights[k]) * float(getattr(
                            static_index[(item["policy_id"], row.tape_id)], metric
                        ))
                        for k, item in enumerate(static_summary)
                    )
                    for metric in ("ret", "service", "lost")
                }
                deltas.append(float(row.ret - baseline["ret"]))
                reductions.append(float(
                    (baseline["service"] - row.service)
                    / max(abs(baseline["service"]), 1.0)
                ))
                lost_deltas.append(float(row.lost - baseline["lost"]))
            summary.update({
                "resource_comparator_exists": True,
                "mixture": {
                    "weights": {
                        static_summary[k]["policy_id"]: float(value)
                        for k, value in enumerate(weights) if value > 1e-10
                    },
                    "mean_ret": mixture["mean_ret"],
                    "mean_departures": mixture["mean_departures"],
                    "mean_unavailable_hours": mixture["mean_unavailable_hours"],
                },
                "ret_delta_mean": float(np.mean(deltas)),
                "ret_delta_tape_fraction_positive": float(np.mean(np.asarray(deltas) > 0)),
                "service_loss_reduction_mean": float(np.mean(reductions)),
                "lost_orders_delta_mean": float(np.mean(lost_deltas)),
            })
        summaries.append(summary)
    verdict = {
        "gate": "PROGRAM_E_AUXILIARY_OBSERVABLE_BASELINES",
        "data_source": "existing validation outputs only",
        "new_tapes_opened": 0,
        "summaries": summaries,
        "interpretation": "NO_AUXILIARY_POLICY_CONVERTED_HEADROOM",
    }
    (ROOT / "auxiliary_verdict.json").write_text(
        json.dumps(verdict, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(verdict, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
