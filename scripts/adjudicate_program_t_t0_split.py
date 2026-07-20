#!/usr/bin/env python3
"""Select T0 comparators on one burned half and adjudicate on the other."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from supply_chain.program_t_t0_gate import adjudicate_t0_residual  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    matrix = json.loads(args.matrix.read_text())
    n = int(matrix["n_tapes"])
    if n < 8 or n % 2:
        raise ValueError("split adjudication requires an even N >= 8")
    split = n // 2
    out = {
        "schema_version": "program_t_t0_burned_split_adjudication_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_matrix": str(args.matrix),
        "selection_tapes": [7490001, 7490001 + split - 1],
        "evaluation_tapes": [7490001 + split, 7490001 + n - 1],
        "cells": {},
    }
    for cell, payload in matrix["cells"].items():
        candidates = []
        for config_id, values in payload["comparators"].items():
            candidates.append((
                float(np.mean(values["ret_visible"][:split])),
                -float(np.percentile(values["online_ms"][:split], 95)),
                config_id,
            ))
        _score, _latency, selected = max(candidates)
        learner = payload["learner"]
        mpc = payload["comparators"][selected]
        sl = slice(split, n)
        learner_ret = np.asarray(learner["ret_visible"])[sl]
        mpc_ret = np.asarray(mpc["ret_visible"])[sl]
        result = adjudicate_t0_residual(
            best_observable_ret=learner_ret,
            reinforced_mpc_ret=mpc_ret,
            worst_product_delta=np.asarray(learner["worst_product_fill"])[sl] - np.asarray(mpc["worst_product_fill"])[sl],
            lost_order_delta=np.asarray(learner["lost_orders"])[sl] - np.asarray(mpc["lost_orders"])[sl],
            resource_delta=np.asarray(learner["gross_production_quantity"])[sl] - np.asarray(mpc["gross_production_quantity"])[sl],
        )
        delta = learner_ret - mpc_ret
        rng = np.random.default_rng(20260720)
        indices = rng.integers(0, len(delta), size=(20000, len(delta)))
        boot = delta[indices].mean(axis=1)
        result.update({
            "selected_comparator": selected,
            "selection_mean_ret": float(_score),
            "evaluation_learner_mean_ret": float(np.mean(learner_ret)),
            "evaluation_mpc_mean_ret": float(np.mean(mpc_ret)),
            "evaluation_point_delta": float(np.mean(delta)),
            "evaluation_ci90": [float(np.quantile(boot, 0.05)), float(np.quantile(boot, 0.95))],
            "selected_p95_online_ms": float(np.percentile(mpc["online_ms"][sl], 95)),
            "unique_calendars_evaluation": len({tuple(row) for row in mpc["calendar"][split:n]}),
        })
        out["cells"][cell] = result
    out["verdict"] = (
        "PASS_T0_ALL_CELLS" if all(row["status"].startswith("PASS") for row in out["cells"].values())
        else "STOP_T0_NO_SAFE_RESIDUAL_HEADROOM"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
