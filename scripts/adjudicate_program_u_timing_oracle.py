#!/usr/bin/env python3
"""Adjudicate U2 from a frozen complete calendar-by-tape ReT matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from supply_chain.program_u_timing_oracle import timing_oracle_gate  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    with np.load(args.matrix, allow_pickle=False) as payload:
        calendars = np.asarray(payload["calendars"], dtype=np.uint8)
        ret = np.asarray(payload["ret"], dtype=float)
    if calendars.shape != (65_536, 8) or ret.shape[0] != 65_536:
        raise ValueError("expected complete (65536,8) calendars and matching ReT rows")
    score_map = {
        tuple(map(int, calendar)): tuple(map(float, row))
        for calendar, row in zip(calendars, ret, strict=True)
    }
    result = timing_oracle_gate(score_by_calendar=score_map)
    result["claim_status"] = "PRE_LEARNER_ORACLE_GATE_ONLY"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if str(result["status"]).startswith("PASS") else 3


if __name__ == "__main__":
    raise SystemExit(main())

