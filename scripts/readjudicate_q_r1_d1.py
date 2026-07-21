#!/usr/bin/env python3
"""Recompute D1 gates from frozen raw rows after guardrail code review."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_q_r1_d1_demand_memory import summarize  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "results/q_r1/d1_demand_memory_v1/result.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/q_r1/d1_demand_memory_v1/adjudication.json")
    args = parser.parse_args()
    raw = json.loads(args.source.read_text())
    payload = {
        "schema_version": "q_r1_d1_demand_memory_adjudication_v1",
        "claim_status": "EXPLORATORY_NO_CLAIM",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(args.source),
        "oracle_parameters_boundary": "theta-known information arm under the same stochastic MPC; never observes Z_t or future demand and is not a mathematical value upper bound",
        "summary": summarize(raw["rows"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
