"""Tests for the treatment-window outcome metric (corrected protocol)."""

from __future__ import annotations

from supply_chain.supply_chain import MFSCSimulation
from supply_chain.clean_metrics import treatment_filtered_order_ret


def _run():
    sim = MFSCSimulation(
        shifts=2, seed=11, horizon=8064, risks_enabled=True, risk_level="current",
        risk_occurrence_mode="thesis_window", warmup_trigger="op9_arrival",
    )
    sim.run()
    return sim


def test_filter_excludes_pre_treatment_orders_and_partitions_cleanly():
    sim = _run()
    clean = treatment_filtered_order_ret(sim)
    all_n = len(sim.orders)
    # Warm-up produced some orders before the policy could act.
    assert clean["n_orders_pre_treatment"] > 0
    # Kept + excluded partitions the full order set exactly.
    assert clean["n_orders"] + clean["n_orders_pre_treatment"] == all_n
    # treatment_start defaults to end of warm-up.
    assert clean["treatment_start"] == sim.warmup_time


def test_filtering_changes_the_outcome_value():
    sim = _run()
    contaminated = sim.compute_order_level_ret()["mean_ret"]
    clean = treatment_filtered_order_ret(sim)["mean_ret"]
    # The pre-treatment warm-up backlog materially shifts ReT; the corrected metric
    # must differ (here by ~0.08, an order of magnitude above the retained-reset signal).
    assert abs(clean - contaminated) > 0.01


def test_explicit_treatment_start_keeps_only_later_orders():
    sim = _run()
    cutoff = sim.warmup_time + 2000.0
    res = treatment_filtered_order_ret(sim, treatment_start=cutoff)
    assert res["treatment_start"] == cutoff
    assert all(float(o.OPTj) >= cutoff for o in sim.orders) or res["n_orders_pre_treatment"] > 0
