#!/usr/bin/env python3
"""Fail-closed entry point for D1 Phase 4.

The script intentionally refuses to fit or open virgin tapes unless the frozen
branching verdict promoted the lever. D1-v2 stopped, so this guard is the only
scientifically valid Phase-4 behavior for the current artifacts.
"""
from __future__ import annotations

import json
from pathlib import Path

VERDICT = Path("results/program_d/d1_branching/verdict.json")


def main() -> int:
    payload = json.loads(VERDICT.read_text(encoding="utf-8"))
    if not bool(payload.get("promoted_to_observable_tree", False)):
        raise SystemExit(
            "BLOCKED_BY_PREREGISTRATION: branching verdict is "
            f"{payload.get('verdict')}; tree fitting and virgin tapes are forbidden."
        )
    raise SystemExit(
        "Promotion artifact detected, but this repository snapshot was frozen after "
        "the terminal D1 stop. Implement Phase 4 in a new commit without altering "
        "the frozen branching artifact."
    )


if __name__ == "__main__":
    raise SystemExit(main())

