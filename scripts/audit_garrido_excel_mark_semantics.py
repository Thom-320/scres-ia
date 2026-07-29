#!/usr/bin/env python3
"""Reproducible monotonicity discriminator for Garrido workbook risk marks.

If a binary mark is determined solely by whether an order window overlaps a
fixed set of event intervals, a containing order window must inherit every mark
present on the contained window. Violations reject that *pure window-only*
model. Non-violations do not uniquely identify it.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.garrido_replication import (  # noqa: E402
    DEFAULT_RAW_WORKBOOKS,
    GarridoCFTarget,
    load_raw_garrido_targets,
)


ODD_CFS = (1, 3, 5, 7, 9, 11, 13, 15, 17, 19)


def risk_family(label: str) -> str:
    match = re.match(r"R\d+", str(label))
    return match.group(0) if match else str(label)


def superset_violations(target: GarridoCFTarget) -> dict[str, Any]:
    """Count ordered containing-window pairs and column-level violations."""
    orders = target.orders
    opt = np.asarray([order.optj for order in orders], dtype=float)
    oat = np.asarray([order.oatj for order in orders], dtype=float)
    active = {
        label: np.asarray(
            [float(order.risk_values.get(label, 0.0)) > 0.0 for order in orders],
            dtype=bool,
        )
        for label in target.risk_columns
    }
    opportunities: dict[str, int] = defaultdict(int)
    violations: dict[str, int] = defaultdict(int)
    nested_pairs = 0
    for contained_index in range(len(orders)):
        containers = (opt <= opt[contained_index]) & (oat >= oat[contained_index])
        containers[contained_index] = False
        n_containers = int(containers.sum())
        nested_pairs += n_containers
        if n_containers == 0:
            continue
        for label, values in active.items():
            if not values[contained_index]:
                continue
            family = risk_family(label)
            opportunities[family] += n_containers
            violations[family] += int((containers & ~values).sum())
    return {
        "cfi": int(target.cfi),
        "orders": len(orders),
        "nested_pairs": int(nested_pairs),
        "families": {
            family: {
                "violations": int(violations[family]),
                "opportunities": int(opportunities[family]),
                "rate": (
                    float(violations[family] / opportunities[family])
                    if opportunities[family]
                    else None
                ),
            }
            for family in sorted(opportunities)
        },
    }


def run(cfs: Iterable[int] = ODD_CFS) -> dict[str, Any]:
    targets = load_raw_garrido_targets(DEFAULT_RAW_WORKBOOKS)
    results = {f"cf{cfi}": superset_violations(targets[cfi]) for cfi in cfs}
    return {
        "test": "containing-window mark monotonicity",
        "null_model": "binary mark is a function only of overlap with fixed event intervals",
        "interpretation": {
            "violation": "rejects the pure window-only null model",
            "no_violation": (
                "consistent with the null but not exclusive evidence for it; "
                "quantity, route, or genealogy rules can also be monotone"
            ),
        },
        "split": "odd_cf_calibration_only",
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/audits/excel_mark_semantics/"
            "superset_violations_all_odd_cfs.json"
        ),
    )
    args = parser.parse_args()
    payload = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for cf, result in payload["results"].items():
        rates = ", ".join(
            f"{risk}={stats['rate']:.3f}"
            for risk, stats in result["families"].items()
            if stats["rate"] is not None
        )
        print(f"{cf}: pairs={result['nested_pairs']}; {rates}")


if __name__ == "__main__":
    main()
