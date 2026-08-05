"""Instrument tests for the E*-C replay repair.

These tests deliberately feed the falsifier helpers a manipulated row.  A check that cannot
reject these inputs is not evidence for the replay artifact.
"""
from __future__ import annotations

import numpy as np

from scripts.run_estar_capacity_sweep_v1 import (
    demand_identity_evidence,
    selected_guardrail_evidence,
)


def _row(*, demand=100.0, worst=0.8, lost=0.0):
    return {
        "demanded_total": demand,
        "demanded_by_claimant": {"A": demand / 2.0, "B": demand / 2.0},
        "worst_claimant_fill": worst,
        "lost_orders": lost,
    }


def test_f2_rejects_demand_drift_against_same_tape_baseline():
    baseline = {"R": [_row()]}
    cells = {"budget=600|R": {0.5: [_row(demand=99.0)]}}

    evidence = demand_identity_evidence(cells, baseline, {"budget=600|R": "R"})

    assert evidence["passed"] is False
    assert evidence["mismatches"] == 1
    assert evidence["max_total_abs_delta"] == 1.0


def test_f2_accepts_identical_total_and_claimant_demand():
    baseline = {"R": [_row()]}
    cells = {"budget=600|R": {0.5: [_row()]}}

    evidence = demand_identity_evidence(cells, baseline, {"budget=600|R": "R"})

    assert evidence["passed"] is True
    assert evidence["mismatches"] == 0


def test_f6_rejects_selected_arm_that_buys_fill_with_lost_orders():
    baseline = {"R": [_row(lost=0.0), _row(lost=0.0)]}
    cells = {"budget=600|R": {0.5: [_row(lost=2.0), _row(lost=2.0)]}}

    evidence = selected_guardrail_evidence(
        cells,
        baseline,
        {"budget=600|R": 0.5},
        {"budget=600|R": "R"},
        n_boot=500,
        rng=np.random.default_rng(1234),
    )

    assert evidence["passed"] is False
    assert evidence["by_cell"]["budget=600|R"]["metrics"]["lost_orders"]["passes"] is False

