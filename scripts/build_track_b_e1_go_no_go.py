#!/usr/bin/env python3
"""E1 go/no-go verdict: PPO vs regime-conditioned static table and heuristics.

`scripts/run_track_b_e1_regime_static_heuristic_crn.py` evaluates zero-learning
comparators (common static, fitted regime table, 6 heuristics) reusing the same
CRN eval-plan keys (`seed`, `episode`, `eval_seed`) as the canonical Track B PPO
run. Neither script loads the PPO policy itself. This script merges the two
`episode_metrics.csv` files on those keys and answers the question the audit
(docs/REVIEWER2_DEEP_AUDIT_2026-07-01.md, T3) actually asked: does PPO beat a
zero-learning comparator that has access to the same privileged observation
fields (true regime, forecasts) it does?

Verdict rule:
- If PPO's CI95 lower bound over the best zero-learning comparator (by primary
  metric) is > 0, the win survives the privileged-observation attack.
- If the best zero-learning comparator matches or beats PPO, the paper's
  central claim (as currently framed) does not survive and must be reframed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.build_track_b_q1_stats import (  # noqa: E402
    PRIMARY_METRIC,
    bootstrap_ci,
    cohens_d_paired,
    paired_rows,
)

DEFAULT_PPO_RUN_DIR = Path(
    "outputs/experiments/track_b_gain_2026-06-30/"
    "top_tier_confirm_v3_output/track_b_top_tier_confirm_5seed_60k_h104"
)
DEFAULT_E1_DIR = Path("outputs/experiments/track_b_e1_confirmatory_2026-07-02")
DEFAULT_OUT_DIR = Path("outputs/audits/track_b_e1_go_no_go_2026-07-02")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ppo-run-dir", type=Path, default=DEFAULT_PPO_RUN_DIR)
    parser.add_argument("--e1-dir", type=Path, default=DEFAULT_E1_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--primary-metric", default=PRIMARY_METRIC)
    parser.add_argument("--learned-policy", default="ppo")
    return parser.parse_args()


def compare(learned: pd.DataFrame, comparator: pd.DataFrame, label: str, metric: str) -> dict[str, Any]:
    pairs = paired_rows(learned, comparator, metric)
    if pairs.empty:
        return {
            "comparator": label,
            "n_pairs": 0,
            "note": "no overlapping (seed, episode, eval_seed) keys — CRN mismatch",
        }
    delta = pairs["learned"].to_numpy(dtype=float) - pairs["static"].to_numpy(dtype=float)
    lo, hi = bootstrap_ci(delta)
    return {
        "comparator": label,
        "n_pairs": int(len(pairs)),
        "ppo_mean": float(pairs["learned"].mean()),
        "comparator_mean": float(pairs["static"].mean()),
        "delta_mean": float(delta.mean()),
        "delta_ci95_low": lo,
        "delta_ci95_high": hi,
        "paired_cohens_d": cohens_d_paired(delta),
        "ppo_wins": bool(delta.mean() > 0.0),
        "ppo_wins_ci95": bool(lo > 0.0),
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ppo_all = pd.read_csv(args.ppo_run_dir / "episode_metrics.csv")
    ppo_rows = ppo_all[ppo_all["policy"].astype(str) == args.learned_policy].copy()
    if ppo_rows.empty:
        raise SystemExit(f"No rows with policy=={args.learned_policy!r} in {args.ppo_run_dir}")

    e1_all = pd.read_csv(args.e1_dir / "episode_metrics.csv")
    policies = sorted(e1_all["policy"].astype(str).unique())

    regime_table_policies = [p for p in policies if p.startswith("regime_table_")]
    heuristic_policies = [p for p in policies if p.startswith("heur_")]
    common_static_policies = [
        p for p in policies if not p.startswith("regime_table_") and not p.startswith("heur_")
    ]

    results: list[dict[str, Any]] = []
    for label in regime_table_policies + heuristic_policies + common_static_policies:
        comparator_rows = e1_all[e1_all["policy"].astype(str) == label].copy()
        results.append(compare(ppo_rows, comparator_rows, label, args.primary_metric))

    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output_dir / "e1_go_no_go_comparisons.csv", index=False)

    valid = results_df[results_df.get("n_pairs", 0) > 0].copy() if not results_df.empty else results_df
    verdict: dict[str, Any]
    if valid.empty:
        verdict = {"status": "NO_DATA", "reason": "no comparator had overlapping CRN keys with PPO"}
    else:
        # best zero-learning comparator = the one with the highest comparator_mean
        best_row = valid.loc[valid["comparator_mean"].idxmax()]
        ppo_beats_best = bool(best_row["ppo_wins"])
        ppo_beats_best_ci = bool(best_row["ppo_wins_ci95"])
        if ppo_beats_best_ci:
            status = "GO — PPO beats the best zero-learning comparator with CI95 > 0"
        elif ppo_beats_best:
            status = "MARGINAL — PPO beats the best zero-learning comparator but CI95 crosses zero"
        else:
            status = "NO-GO — the best zero-learning comparator matches or beats PPO; reframe required"
        verdict = {
            "status": status,
            "best_zero_learning_comparator": str(best_row["comparator"]),
            "best_zero_learning_comparator_mean": float(best_row["comparator_mean"]),
            "ppo_mean": float(best_row["ppo_mean"]),
            "delta_mean": float(best_row["delta_mean"]),
            "delta_ci95": [float(best_row["delta_ci95_low"]), float(best_row["delta_ci95_high"])],
        }

    (args.output_dir / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    print(json.dumps(verdict, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
