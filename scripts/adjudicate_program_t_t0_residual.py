#!/usr/bin/env python3
"""Fail-closed T0 residual-headroom adjudication from frozen paired vectors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from supply_chain.program_t_t0_gate import adjudicate_t0_residual  # noqa: E402


REQUIRED = (
    "best_observable_ret",
    "reinforced_mpc_ret",
    "worst_product_delta",
    "lost_order_delta",
    "resource_delta",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    with np.load(args.input, allow_pickle=False) as payload:
        if any(key not in payload.files for key in REQUIRED):
            raise ValueError(f"input must contain {REQUIRED}")
        vectors = {key: np.asarray(payload[key], dtype=float) for key in REQUIRED}
    lengths = {len(value) for value in vectors.values()}
    if len(lengths) != 1:
        raise ValueError("all paired tape vectors must have the same length")
    result = adjudicate_t0_residual(**vectors)
    result.update(
        schema_version="program_t_t0_residual_adjudication_v1",
        claim_status="BURNED_DEVELOPMENT_GATE_ONLY",
        input=str(args.input),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if str(result["status"]).startswith("PASS") else 3


if __name__ == "__main__":
    raise SystemExit(main())

