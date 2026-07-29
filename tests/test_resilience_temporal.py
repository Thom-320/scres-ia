from __future__ import annotations

from types import SimpleNamespace

from supply_chain.resilience_temporal import (
    cluster_risk_events,
    compute_temporal_resilience_panel,
)
from supply_chain.episode_metrics import compute_episode_metrics
from supply_chain.supply_chain import OrderRecord, RiskEvent


def _sim(*, orders, events, horizon=1600.0):
    return SimpleNamespace(
        env=SimpleNamespace(now=float(horizon)),
        warmup_time=0.0,
        orders=list(orders),
        risk_events=list(events),
    )


def test_overlapping_risks_form_one_cluster() -> None:
    events = [
        RiskEvent("R22", 100.0, 140.0, 40.0, [10]),
        RiskEvent("R24", 200.0, 200.0, 0.0, [13]),
        RiskEvent("R22", 500.0, 520.0, 20.0, [12]),
    ]
    clusters = cluster_risk_events(
        events, treatment_start=0.0, treatment_end=1000.0, gap_hours=168.0
    )
    assert len(clusters) == 2
    assert clusters[0]["risk_ids"] == ["R22", "R24"]
    assert clusters[1]["risk_ids"] == ["R22"]


def test_temporal_panel_sees_depth_auc_and_recovery() -> None:
    onset = 400.0
    order = OrderRecord(
        j=1,
        OPTj=onset - 48.0,
        LTj=48.0,
        quantity=100.0,
        remaining_qty=0.0,
        OATj=onset + 48.0,
        CTj=96.0,
    )
    event = RiskEvent("R22", onset, onset + 24.0, 24.0, [10])
    panel = compute_temporal_resilience_panel(_sim(orders=[order], events=[event]))

    assert panel["system_ttr_n_clusters"] == 1.0
    assert panel["system_ttr_n_recovered"] == 1.0
    assert panel["system_ttr_n_censored"] == 0.0
    assert panel["temporal_maximum_service_drop"] > 0.0
    assert panel["temporal_service_loss_auc_ration_hours"] > 0.0
    assert panel["system_ttr_mean"] >= 48.0


def test_no_risk_has_empty_secondary_panel() -> None:
    panel = compute_temporal_resilience_panel(_sim(orders=[], events=[]))
    assert panel["system_ttr_n_clusters"] == 0.0
    assert panel["system_ttr_censored_fraction"] == 0.0
    assert panel["temporal_maximum_service_drop"] == 0.0
    assert panel["temporal_service_loss_auc_ration_hours"] == 0.0


def test_optional_temporal_panel_does_not_change_canonical_metrics() -> None:
    order = OrderRecord(
        j=1,
        OPTj=100.0,
        LTj=48.0,
        quantity=100.0,
        remaining_qty=0.0,
        OATj=160.0,
        CTj=60.0,
    )
    sim = _sim(
        orders=[order],
        events=[RiskEvent("R22", 120.0, 144.0, 24.0, [10])],
    )
    canonical = compute_episode_metrics(sim, include_temporal_panel=False)
    augmented = compute_episode_metrics(sim, include_temporal_panel=True)

    assert canonical.items() <= augmented.items()
    assert augmented["ret_excel_contract_version"] == "ret_excel_request_snapshot_v2"
    assert "temporal_panel_version" in augmented
