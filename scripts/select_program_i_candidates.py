#!/usr/bin/env python3
"""Select Program I families from sensitivity output without authorizing RL."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from supply_chain.decision_right_discovery import select_candidate_families


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verdict", type=Path, default=Path("results/program_i/morris/verdict.json"))
    parser.add_argument("--catalog", type=Path, default=Path("contracts/decision_right_catalog_v1.json"))
    parser.add_argument("--output", type=Path, default=Path("results/program_i/candidate_families.json"))
    args = parser.parse_args()
    verdict = json.loads(args.verdict.read_text())
    catalog = json.loads(args.catalog.read_text())
    candidates = select_candidate_families(verdict["metrics"]["ret_excel"], catalog["factors"])
    result = {
        "contract_id": "global_sensitivity_v1",
        "source_design_sha256": verdict["design_sha256"],
        "candidate_families": candidates,
        "environment_factors_excluded": True,
        "requires_branching": True,
        "promote_to_rl": False,
        "interpretation": "CANDIDATES_REQUIRE_ACTION_CONTRACT" if candidates else "INSUFFICIENT_IMPLEMENTED_DECISION_FAMILIES",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if candidates else 2


if __name__ == "__main__":
    raise SystemExit(main())
