from __future__ import annotations

from scripts.audit_garrido_wrap_custody import build_manifest


def test_custody_manifest_keeps_active_runs_pending_and_retired_claims_explicit():
    payload = build_manifest()
    records = {item["artifact"]: item for item in payload["records"]}
    assert records["results/garrido_wrap_q1/result.json"]["status"] == (
        "NO_GO_NEURAL_PREMIUM_IN_WRAP_PANEL"
    )
    assert records["results/garrido_meta_learner_h3power_local/result.json"]["status"] == (
        "PENDING_ACTIVE_LOCAL_RUN"
    )
    assert records["results/garrido_neural_headroom_gate_v1/result.json"]["status"] == (
        "NO_GO_NEURAL_PREMIUM_E1_HEADROOM_CLOSED"
    )
    assert payload["retired_claims"]["old_meta_learner_contrasts"] == (
        "RETIRED_DRIVER_LEAK"
    )
