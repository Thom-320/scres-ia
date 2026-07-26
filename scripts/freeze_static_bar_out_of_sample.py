#!/usr/bin/env python3
"""Freeze the deployable static bar by selecting it OUTSIDE the evaluation campaigns.

The 2026-07-24 report selected the best static calendar on the same 48 campaigns it then
graded. That is in-sample selection, and it inflated the bar (mean 0.7376 vs 0.6894) while
manufacturing 27 "already optimal" campaigns that do not survive an honest selection.

This script enumerates the exact 4^8 frontier for a calibration block disjoint from every
evaluation campaign, picks the calendar with the highest mean exact resilience there, and
writes it with its provenance so downstream reports consume an artifact instead of a number
someone typed.

Usage:
    .venv/bin/python scripts/freeze_static_bar_out_of_sample.py
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.c6_perbatch_ceiling import OBJECTIVE, SCHED_PATTERN  # noqa: E402
from scripts.evaluate_program_q_replication import scheduler  # noqa: E402
from scripts.run_oracle_curve_v2 import CALIB_ROOTS, histories  # noqa: E402
from supply_chain.oracle_curve_v2 import load_history  # noqa: E402
from supply_chain.program_o_full_des_transducer import (  # noqa: E402
    simulate_full_des_frontier,
)

OUT = ROOT / "results/oracle_capture_v1/static_bar_out_of_sample.json"
FRONTIERS = ROOT / "results/fig5_surrogate_v1/frontiers"


def main() -> int:
    sched = scheduler()
    calendars = np.load(FRONTIERS / "calendars.npz")["calendars"]
    started = time.perf_counter()
    labels = []
    identities = []
    for spec in histories(CALIB_ROOTS):
        campaigns, _ = load_history(spec, sched)
        for campaign in campaigns:
            metrics = simulate_full_des_frontier(
                skeleton=campaign.skeleton, scheduler=SCHED_PATTERN,
                calendars=calendars, include_q_r1_metrics=True)
            labels.append(np.asarray(metrics[OBJECTIVE], dtype=float))
            identities.append([int(campaign.history_root), int(campaign.campaign_index),
                               float(spec.kappa)])
    stack = np.vstack(labels)
    row = int(stack.mean(axis=0).argmax())
    calendar = [(row // 4 ** (7 - w)) % 4 for w in range(8)]

    payload = {
        "schema": "static_bar_out_of_sample_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "the deployable static bar: the best fixed calendar a planner could have "
                   "chosen WITHOUT seeing the evaluation campaigns",
        "calendar": calendar,
        "frontier_row": row,
        "calibration_block": [CALIB_ROOTS.start, CALIB_ROOTS.stop - 1],
        "calibration_campaigns": int(stack.shape[0]),
        "calibration_mean_label": float(stack.mean(axis=0)[row]),
        "disjoint_from_evaluation": True,
        "supersedes": "the in-sample bar [0,0,3,3,3,3,3,3] selected on the 48 evaluation "
                      "campaigns themselves; that bar is retained only as an adversarial "
                      "hindsight reference, never as the deployable one",
        "objective": OBJECTIVE,
        "elapsed_seconds": time.perf_counter() - started,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(f"static bar {calendar} (row {row}) from {stack.shape[0]} calibration campaigns "
          f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
