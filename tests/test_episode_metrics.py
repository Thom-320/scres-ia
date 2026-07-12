"""Tests for the unified episode metrics panel."""

from __future__ import annotations

from supply_chain.supply_chain import MFSCSimulation
from supply_chain.episode_metrics import (
    compute_episode_metrics,
    merge_resource_metrics,
    METRIC_KEYS,
)


def _run(shifts=1, period=0, seed=7, horizon=16128, risk="increased"):
    bufs = None
    repl = None
    if period:
        from supply_chain.config import INVENTORY_BUFFERS
        bufs = dict(INVENTORY_BUFFERS[period])
        repl = float(period)
    sim = MFSCSimulation(
        shifts=shifts, seed=seed, horizon=horizon, risks_enabled=True, risk_level=risk,
        risk_occurrence_mode="thesis_window", warmup_trigger="op9_arrival",
        initial_buffers=bufs, inventory_replenishment_period=repl,
    )
    sim.run()
    return sim


def test_panel_has_all_keys_and_sane_ranges():
    m = compute_episode_metrics(_run())
    # resilience bars bounded
    assert 0.0 <= m["ret_excel"] <= 1.0
    assert 0.0 <= m["ret_thesis"] <= 1.0
    assert 0.0 <= m["ret_continuous"] <= 1.0
    assert 0.0 <= m["fill_rate"] <= 1.0
    assert 0.0 <= m["fill_rate_on_time"] <= 1.0
    # non-negative service/throughput
    assert m["lost_orders"] >= 0
    assert m["service_loss_auc_ration_hours"] >= 0
    assert m["delivered_rations"] >= 0
    assert m["n_orders"] > 0


def test_distribution_quantiles_are_monotone():
    m = compute_episode_metrics(_run())
    assert m["ctj_p50"] <= m["ctj_p90"] <= m["ctj_p99"]
    assert m["rpj_p50"] <= m["rpj_p90"] <= m["rpj_p99"]
    assert m["ttr_p95"] >= 0


def test_buffer_panel_has_comparable_service_metrics_under_stress():
    no_buf = compute_episode_metrics(_run(shifts=1, period=0))
    buf = compute_episode_metrics(_run(shifts=1, period=168))

    assert no_buf["n_orders"] > 0
    assert buf["n_orders"] > 0
    for panel in (no_buf, buf):
        assert 0.0 <= panel["ret_excel"] <= 1.0
        assert 0.0 <= panel["flow_fill_rate"] <= 1.5
        assert panel["service_loss_auc_ration_hours"] >= 0.0


def test_merge_resource_metrics():
    m = compute_episode_metrics(_run())
    merged = merge_resource_metrics(
        m, shift_hours=161280.0, extra_shift_hours=0.0,
        strategic_buffer_units=0.0, end_state_inventory=123.0,
    )
    assert merged["shift_hours"] == 161280.0
    assert merged["surge_hours"] == 0.0
    assert merged["unit_surge_hours_per_ration"] == 0.0
    assert merged["end_state_inventory"] == 123.0
    # every declared metric key is present after merge
    for k in METRIC_KEYS:
        assert k in merged, f"missing metric {k}"
