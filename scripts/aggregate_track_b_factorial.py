#!/usr/bin/env python3
"""Consolidated, versioned aggregation for the corrected contract factorial.

The three factorial arms (Blocker 2) were executed as separate invocations of
`run_track_b_contract_factorial.py --skip-static`, so each run's summary.json
holds only its own arm means. This script recomputes ALL cross-arm and
arm-vs-static contrasts from the raw per-episode CSVs and writes a single
`factorial_aggregate_summary.json`, so the verdict numbers in
`docs/TRACK_B_CONTRACT_FACTORIAL_VERDICT_2026-07-10.md` trace to one versioned
analysis rather than ad-hoc recomputation.

Static baseline rows come from the crossed-eval ledger (same canonical env,
same fresh tapes; the prespecified S2/Op10x2.00/Op12x1.50 comparator),
restricted to the factorial's tape battery.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats as scipy_stats


def two_way_bootstrap(delta: np.ndarray, n_boot: int = 10_000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n_s, n_t = delta.shape
    means = np.empty(n_boot)
    for b in range(n_boot):
        s_idx = rng.integers(0, n_s, n_s)
        t_idx = rng.integers(0, n_t, n_t)
        means[b] = delta[np.ix_(s_idx, t_idx)].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def load_arm(csv_path: Path, arm: str) -> dict[tuple[int, int], dict[str, float]]:
    out: dict[tuple[int, int], dict[str, float]] = {}
    with csv_path.open() as fh:
        for row in csv.DictReader(fh):
            if row["arm"] != arm:
                continue
            out[(int(row["train_seed"]), int(row["eval_seed"]))] = {
                k: float(v) for k, v in row.items()
                if k not in ("arm", "train_seed", "eval_seed") and v != ""
            }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/experiments"))
    parser.add_argument("--stamp", default="2026-07-09")
    parser.add_argument("--arms", nargs="+",
                        default=["joint", "upstream_shift", "dispatch_only"])
    parser.add_argument("--static-csv", type=Path,
                        default=Path("outputs/experiments/track_b_crossed_eval_2026-07-09/crossed_rows.csv"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    data: dict[str, dict[tuple[int, int], dict[str, float]]] = {}
    for arm in args.arms:
        csv_path = args.root / f"track_b_factorial_{arm}_{args.stamp}" / "factorial_rows.csv"
        data[arm] = load_arm(csv_path, arm)
    seeds = sorted({s for rows in data.values() for (s, _t) in rows})
    tapes = sorted({t for rows in data.values() for (_s, t) in rows})
    for arm, rows in data.items():
        missing = [(s, t) for s in seeds for t in tapes if (s, t) not in rows]
        if missing:
            raise SystemExit(f"arm {arm} missing cells: {missing[:5]}...")

    static_by_tape: dict[int, dict[str, float]] = {}
    with args.static_csv.open() as fh:
        for row in csv.DictReader(fh):
            if row["arm"] == "static" and int(row["eval_seed"]) in tapes:
                static_by_tape[int(row["eval_seed"])] = {
                    k: float(v) for k, v in row.items()
                    if k not in ("arm", "train_seed", "eval_seed", "obs_dim") and v != ""
                }
    if set(static_by_tape) != set(tapes):
        raise SystemExit("static ledger does not cover the factorial tape battery")

    def contrast(a: np.ndarray, b: np.ndarray | None = None) -> dict[str, Any]:
        delta = a - b if b is not None else a
        per_seed = delta.mean(axis=1)
        lo2, hi2 = two_way_bootstrap(delta)
        tci = scipy_stats.t.interval(
            0.95, len(per_seed) - 1, loc=per_seed.mean(), scale=scipy_stats.sem(per_seed))
        return {
            "mean": float(delta.mean()),
            "two_way_ci95": [lo2, hi2],
            "seed_t_ci95": [float(tci[0]), float(tci[1])],
            "per_seed": [float(v) for v in per_seed],
            "seeds_positive": int((per_seed > 0).sum()),
            "tapes_positive": int((delta.mean(axis=0) > 0).sum()),
            "n_tapes": len(tapes),
        }

    result: dict[str, Any] = {
        "arms": args.arms,
        "seeds": seeds,
        "tapes": tapes,
        "static_source": str(args.static_csv),
    }
    for key in ("ret_excel", "ret_excel_cvar05"):
        mat = {arm: np.array([[data[arm][(s, t)][key] for t in tapes] for s in seeds])
               for arm in args.arms}
        static = np.array([static_by_tape[t][key] for t in tapes])[None, :]
        block: dict[str, Any] = {
            "arm_means": {arm: float(mat[arm].mean()) for arm in args.arms},
            "static_mean": float(static.mean()),
            "vs_static": {arm: contrast(mat[arm], np.repeat(static, len(seeds), axis=0))
                          for arm in args.arms},
        }
        if "joint" in mat and "upstream_shift" in mat:
            block["joint_minus_upstream_shift_PRIMARY"] = contrast(mat["joint"], mat["upstream_shift"])
        if "joint" in mat and "dispatch_only" in mat:
            block["joint_minus_dispatch_only"] = contrast(mat["joint"], mat["dispatch_only"])
        result[key] = block

    out_path = args.output or (args.root / f"track_b_factorial_aggregate_{args.stamp}"
                               / "factorial_aggregate_summary.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("ret_excel",)}, indent=2))
    print(f"written: {out_path}")


if __name__ == "__main__":
    main()
