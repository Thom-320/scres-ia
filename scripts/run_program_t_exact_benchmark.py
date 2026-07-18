#!/usr/bin/env python3
"""Run the deterministic Program T exact POMDP benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.paper2_exhaustive_search.program_t_exact_pomdp import solve_grid  # noqa: E402
from supply_chain.program_t_exact_pomdp import ExactProductMixPOMDP  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = {
        "schema_version": "program_t_exact_benchmark_bundle_v1",
        "minimal_policy_ladder": ExactProductMixPOMDP(horizon=6).diagnostic(),
        "latent_model_uncertainty_grid": solve_grid(),
        "claim_limit": (
            "Exact only for the reduced finite benchmarks; no full-DES or "
            "Paper-2 claim is authorized."
        ),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
