#!/usr/bin/env python3
"""Prospective sample-size audit from burned comparator-v2 paired rows."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np


def paired_root_deltas(
    payload: dict[str, object], *, config_id: str, persistence_mode: str
) -> dict[int, float]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in payload.get("pareto_pairs", []):
        if row["config_id"] != config_id:
            continue
        if row["persistence_mode"] != persistence_mode:
            continue
        grouped[int(row["history_root"])].append(
            float(row["retained"]["early_ret_complete_cohort"])
            - float(row["reset"]["early_ret_complete_cohort"])
        )
    return {root: float(np.mean(values)) for root, values in grouped.items()}


def required_histories(*, sd: float, sesoi: float, alpha: float, power: float) -> int:
    if sd <= 0.0:
        return 1
    normal = NormalDist()
    z_alpha = normal.inv_cdf(1.0 - alpha)
    z_power = normal.inv_cdf(power)
    return int(math.ceil(((z_alpha + z_power) * sd / sesoi) ** 2))


def audit(
    payload: dict[str, object],
    *,
    config_id: str,
    persistence_mode: str,
    sesoi: float,
    alpha: float,
    bootstrap_draws: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    by_root = paired_root_deltas(
        payload, config_id=config_id, persistence_mode=persistence_mode
    )
    roots = sorted(by_root)
    values = np.asarray([by_root[root] for root in roots], dtype=float)
    if len(values) < 2:
        raise ValueError("power audit requires at least two independent history roots")
    sd = float(values.std(ddof=1))
    rng = np.random.default_rng(bootstrap_seed)
    bootstrap = np.asarray(
        [float(rng.choice(values, size=len(values), replace=True).mean()) for _ in range(bootstrap_draws)],
        dtype=float,
    )
    requirements = {
        str(power): required_histories(
            sd=sd, sesoi=sesoi, alpha=alpha, power=power
        )
        for power in (0.80, 0.90)
    }
    return {
        "schema_version": "q_r1_comparator_power_audit_v1",
        "claim_status": "BURNED_POWER_PLANNING_NO_CLAIM",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_id": config_id,
        "persistence_mode": persistence_mode,
        "sesoi": sesoi,
        "one_sided_alpha": alpha,
        "history_roots": roots,
        "n_history_roots": len(roots),
        "paired_root_deltas": {str(root): by_root[root] for root in roots},
        "burned_mean_delta": float(values.mean()),
        "burned_sd_delta": sd,
        "burned_bootstrap_ci95": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "required_histories": requirements,
        "bootstrap_draws": bootstrap_draws,
        "bootstrap_seed": bootstrap_seed,
        "selection_performed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config-id", required=True)
    parser.add_argument("--persistence-mode", default="binary_0.9")
    parser.add_argument("--sesoi", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_722)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    result = audit(
        json.loads(args.input.read_text()),
        config_id=args.config_id,
        persistence_mode=args.persistence_mode,
        sesoi=args.sesoi,
        alpha=args.alpha,
        bootstrap_draws=args.bootstrap_draws,
        bootstrap_seed=args.bootstrap_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
