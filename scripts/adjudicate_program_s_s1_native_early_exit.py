#!/usr/bin/env python3
"""Apply the frozen native-stratum H_PI early-exit without selecting a cell."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


THRESHOLD = 0.01


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(path.read_text()) for path in args.summaries]
    if any(row.get("stratum") != "THESIS_NATIVE_INDEPENDENT" for row in rows):
        raise ValueError("early-exit accepts native-stratum summaries only")
    lcbs = [float(row["H_PI_safe"]["lcb95"]) for row in rows]
    maximum = max(lcbs)
    stop = maximum < THRESHOLD
    payload = {
        "schema_version": "program_s_s1_native_early_exit_v1_1",
        "threshold": THRESHOLD,
        "n_points": len(rows),
        "max_lcb95_H_PI_safe": maximum,
        "promotion_selection_performed": False,
        "verdict": "STOP_S1_NO_CONNECTED_PHYSICAL_HEADROOM" if stop else "PASS_S1_NATIVE_HEADROOM_CONTINUE_TO_CONNECTED_REGION_AUDIT"
    }
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 3 if stop else 0


if __name__ == "__main__":
    raise SystemExit(main())
