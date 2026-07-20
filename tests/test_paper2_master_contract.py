from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = json.loads(
    (ROOT / "contracts/paper2_learning_augmented_event_triggered_mpc_v1.json").read_text()
)


def test_master_contract_opens_no_learner_or_confirmation_seed() -> None:
    assert CONTRACT["status"] == (
        "FROZEN_INTEGRATION_CHARTER_GATES_ONLY_NO_LEARNER_OR_CONFIRMATION_SEEDS_AUTHORIZED"
    )
    assert CONTRACT["current_authorization"]["hybrid_training"] is False
    assert CONTRACT["current_authorization"]["virgin_confirmation"] is False
    assert CONTRACT["current_authorization"]["paper3"] is False


def test_timing_estimand_matches_review_rights_and_uses_weekly_as_ceiling() -> None:
    gate = CONTRACT["promotion_gates"]["U2_timing"]
    assert "same four review rights" in gate["estimand"]
    assert "upper ceiling only" in gate["unconstrained_weekly_oracle_role"]


def test_historical_stops_and_q_fallback_are_preserved() -> None:
    assert CONTRACT["historical_verdicts_unchanged"]["program_q"] == (
        "STOP_Q_NO_REPLICATED_LEARNED_ADAPTATION"
    )
    assert "Program Q" in CONTRACT["failure_route"]
