#!/usr/bin/env python3
"""T0 inference with comparator reselection inside every tape bootstrap."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=20000)
    args = parser.parse_args()
    matrix = json.loads(args.matrix.read_text())
    out = {"schema_version": "program_t_t0_reselection_bootstrap_v1", "created_at": datetime.now(timezone.utc).isoformat(), "resamples": args.resamples, "cells": {}}
    rng = np.random.default_rng(20260720)
    for cell, payload in matrix["cells"].items():
        q = np.asarray(payload["learner"]["ret_visible"], dtype=float)
        ids = sorted(payload["comparators"])
        mpc = np.asarray([payload["comparators"][k]["ret_visible"] for k in ids], dtype=float)
        selected_index = int(np.argmax(mpc.mean(axis=1)))
        selected = ids[selected_index]
        point = float(q.mean() - mpc.mean(axis=1).max())
        bootstrap = np.empty(args.resamples, dtype=float)
        for start in range(0, args.resamples, 1000):
            width = min(1000, args.resamples - start)
            index = rng.integers(0, len(q), size=(width, len(q)))
            q_mean = q[index].mean(axis=1)
            competitor_mean = np.stack([row[index].mean(axis=1) for row in mpc], axis=1)
            bootstrap[start:start + width] = q_mean - competitor_mean.max(axis=1)
        row = payload["comparators"][selected]
        worst_delta = np.asarray(payload["learner"]["worst_product_fill"]) - np.asarray(row["worst_product_fill"])
        worst_index = rng.integers(0, len(q), size=(args.resamples, len(q)))
        worst_boot = worst_delta[worst_index].mean(axis=1)
        lost_delta = np.asarray(payload["learner"]["lost_orders"]) - np.asarray(row["lost_orders"])
        resource_delta = np.asarray(payload["learner"]["gross_production_quantity"]) - np.asarray(row["gross_production_quantity"])
        checks = {
            "residual_lcb95_at_least_0_015": float(np.quantile(bootstrap, .05)) >= .015,
            "worst_product_lcb_at_least_minus_0_02": float(np.quantile(worst_boot, .05)) >= -.02,
            "lost_orders_nonincrease": float(lost_delta.mean()) <= 1e-12,
            "resources_exact": bool(np.all(np.abs(resource_delta) <= 1e-12)),
        }
        out["cells"][cell] = {
            "selected_comparator": selected,
            "point_residual": point,
            "reselection_lcb95": float(np.quantile(bootstrap, .05)),
            "reselection_ucb95": float(np.quantile(bootstrap, .95)),
            "worst_product_lcb95": float(np.quantile(worst_boot, .05)),
            "checks": checks,
            "status": "PASS_T0_RESIDUAL_HEADROOM" if all(checks.values()) else "STOP_T0_NO_SAFE_RESIDUAL_HEADROOM",
        }
    out["verdict"] = "PASS_T0_ALL_CELLS" if all(v["status"].startswith("PASS") for v in out["cells"].values()) else "STOP_T0_NO_SAFE_RESIDUAL_HEADROOM"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
