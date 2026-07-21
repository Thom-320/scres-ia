from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_causal_residual_campaign_remains_exploratory_and_opens_no_seed() -> None:
    result = json.loads(
        (ROOT / "results/program_t/causal_residual_campaign_v1/verdict.json").read_text()
    )
    assert result["claim_status"] == "EXPLORATORY_NO_CLAIM"
    assert result["new_scientific_seeds_opened"] == []
    assert result["hybrid_training_authorized"] is False
    assert result["r1_retention_authorized"] is False
    assert result["next_route"] == "PROGRAM_Q_PUBLICATION_FALLBACK"
    assert result["u1_direct_preflight"]["status"].startswith("PASS_U1_DIRECT_PREFLIGHT")
    assert result["u1_direct_discovery"]["status"] == "STOP_U1_NO_CONNECTED_CLASSICAL_CONVERSION_REGION"


def test_q_mpc_oracle_ceiling_does_not_override_convertibility_stop() -> None:
    result = json.loads(
        (ROOT / "results/program_t/causal_residual_campaign_v1/verdict.json").read_text()
    )
    selector = result["mechanisms"]["q_mpc_complementarity"]
    assert selector["oracle_ceiling_lcb95"] > 0.015
    assert selector["observable_one_step_lcb95"] < 0.012
    assert selector["status"] == "STOP_Q_MPC_SELECTOR_NOT_OBSERVABLY_CONVERTIBLE"
