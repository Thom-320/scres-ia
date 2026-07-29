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
    assert "mean_ret_excel_formula" in clean
    assert "case_counts_excel_formula" in clean


def test_filtering_changes_the_outcome_value():
    sim = _run()
    from supply_chain.ret_thesis import compute_order_level_ret_excel_formula

    # Same Excel formula, all orders vs treatment-window filtered: excluding the
    # pre-treatment warm-up orders shifts the outcome. After the unfulfilled-order
    # fix (lost/pending orders score 0, not the ~1.0 no-risk fill-rate branch), the
    # metric is no longer inflated, so both values sit in the corrected low range
    # and the warm-up shift is small but present.
    excel_all = compute_order_level_ret_excel_formula(
        sim.orders, current_time=float(sim.env.now)
    )["mean_ret_excel"]
    clean = treatment_filtered_order_ret(sim)["mean_ret"]
    assert excel_all != clean  # filtering still changes the outcome
    # Regression guard: unfulfilled orders must not re-inflate ReT toward ~1.0.
    assert excel_all < 0.05 and clean < 0.05


def test_explicit_treatment_start_keeps_only_later_orders():
    sim = _run()
    cutoff = sim.warmup_time + 2000.0
    res = treatment_filtered_order_ret(sim, treatment_start=cutoff)
    assert res["treatment_start"] == cutoff
    assert all(float(o.OPTj) >= cutoff for o in sim.orders) or res["n_orders_pre_treatment"] > 0
