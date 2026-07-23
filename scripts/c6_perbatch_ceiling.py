#!/usr/bin/env python3
"""Per-batch (8^8) action-space ceiling for the Q-R1 burned campaigns.

The deployable residual closed the coarse 4^8 weekly-count space: the frozen c256 comparator
hits the exact global optimum in 23/24 campaigns at kappa 0.90. The per-batch space is 8^8 =
16,777,216, distinguishing intra-week batch ordering, which was shown to move the cohort ReT.
This module computes the clairvoyant ceiling of that richer space per campaign, which
upper-bounds any learner in it.

Top-level (picklable) functions so a ProcessPoolExecutor works under both fork (Linux/Kaggle)
and spawn (macOS). Nothing here trains a learner; it is a gate.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from supply_chain.program_o_full_des_transducer import simulate_full_des_frontier
from supply_chain.retained_context_discovery import build_campaign_history

_ROOT = Path(__file__).resolve().parents[1]


def _count_scheduler() -> dict:
    """The frozen weekly-count scheduler, read from the same contract as
    scripts/evaluate_program_q_replication.py::scheduler.

    Re-read here instead of imported from that module because it drags in
    sb3_contrib -> torch at module import time, which is absent on lean cloud
    runtimes (Kaggle) and irrelevant to this gate.
    """
    parent = json.loads(
        (_ROOT / "contracts/program_o_full_des_hpi_translation_v1.json").read_text()
    )
    key = parent["action"]["primary_scheduler"]
    return parent["action"]["within_week_schedulers"][key]


OBJECTIVE = "early_ret_complete_cohort"
TOTAL = 8 ** 8  # 16,777,216

# 8-pattern scheduler (per-batch space) and the count->canonical-pattern embedding.
SCHED_PATTERN = {str(i): ["P_C" if (i >> j) & 1 else "P_H" for j in range(3)] for i in range(8)}
COUNT_TO_PAT = np.array([0, 2, 5, 7], dtype=np.uint8)   # weekly count -> the pattern the count scheduler emits
COARSE_PATTERNS = np.array([0, 2, 5, 7], dtype=np.uint8)  # the 4^8 count subset inside pattern space

CAMPAIGNS_PER_HISTORY = 12
REGIME_PERSISTENCE = 0.90
DOMINANT_SHARE = 0.90
MODE_TO_KAPPA = {"binary_0.9": 0.90, "binary_0.75": 0.75}


def rebuild_campaign(root: int, kappa: float, index: int):
    """Reconstruct one burned campaign identically to the Pareto/residual path."""
    hist = build_campaign_history(
        history_root=int(root), campaigns=CAMPAIGNS_PER_HISTORY, kappa=float(kappa),
        scheduler=_count_scheduler(), regime_persistence=REGIME_PERSISTENCE,
        dominant_share=DOMINANT_SHARE,
    )
    camp = hist[int(index)]
    assert int(camp.history_root) == int(root) and int(camp.campaign_index) == int(index)
    return camp


def _digits_base8(start: int, n: int) -> np.ndarray:
    """Rows [start, start+n) of the 8^8 calendar table, base-8 little-endian -> (n, 8) uint8."""
    idx = np.arange(start, start + n, dtype=np.int64)
    out = np.empty((n, 8), dtype=np.uint8)
    for week in range(8):
        out[:, week] = (idx % 8).astype(np.uint8)
        idx //= 8
    return out


def _max_objective(skeleton, calendars: np.ndarray) -> float:
    metrics = simulate_full_des_frontier(
        skeleton=skeleton, scheduler=SCHED_PATTERN, calendars=calendars, include_q_r1_metrics=True
    )
    return float(np.asarray(metrics[OBJECTIVE]).max())


def _value_of(skeleton, calendar: np.ndarray) -> float:
    metrics = simulate_full_des_frontier(
        skeleton=skeleton, scheduler=SCHED_PATTERN,
        calendars=np.asarray(calendar, dtype=np.uint8)[None, :], include_q_r1_metrics=True,
    )
    return float(np.asarray(metrics[OBJECTIVE])[0])


def campaign_ceilings(root, kappa, index, frozen_calendar, *, sample=None, chunk=65_536, seed=0):
    """Per-batch ceiling (exact or sampled lower bound), coarse 4^8 ceiling, and a self-check.

    ``frozen_calendar`` is the frozen comparator's weekly-count calendar; replaying it through
    the pattern scheduler (via COUNT_TO_PAT) must reproduce the value the Pareto recorded.
    """
    camp = rebuild_campaign(root, kappa, index)

    coarse = np.stack(np.meshgrid(*([COARSE_PATTERNS] * 8), indexing="ij"), -1).reshape(-1, 8).astype(np.uint8)
    coarse_ceiling = _max_objective(camp.skeleton, coarse)

    replay = _value_of(camp.skeleton, COUNT_TO_PAT[np.asarray(frozen_calendar, dtype=np.uint8)])

    best = -1.0
    if sample is None:  # EXACT 8^8
        start = 0
        while start < TOTAL:
            n = min(chunk, TOTAL - start)
            best = max(best, _max_objective(camp.skeleton, _digits_base8(start, n)))
            start += n
    else:  # sampled lower bound
        rng = np.random.default_rng(seed)
        left = int(sample)
        while left > 0:
            n = min(chunk, left)
            left -= n
            best = max(best, _max_objective(camp.skeleton, rng.integers(0, 8, size=(n, 8), dtype=np.uint8)))
    return {"perbatch_ceiling": best, "coarse_ceiling": coarse_ceiling, "replay": replay}


def ceiling_worker(task: dict) -> dict:
    """Picklable ProcessPool entry point. ``task`` carries one Pareto row's identity."""
    out = campaign_ceilings(
        task["history_root"], MODE_TO_KAPPA[task["persistence_mode"]], task["campaign_index"],
        task["frozen_calendar"], sample=task.get("sample"), chunk=task.get("chunk", 65_536),
        seed=task["history_root"] * 100 + task["campaign_index"],
    )
    out.update({
        "history_root": task["history_root"],
        "campaign_index": task["campaign_index"],
        "persistence_mode": task["persistence_mode"],
        "frozen": task["frozen"],
    })
    return out
