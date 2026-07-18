from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = json.loads(
    (ROOT / "contracts/program_t_confidence_gated_bv_mpc_t0_v1.json").read_text()
)


def test_t0_opens_no_new_scientific_seed() -> None:
    assert CONTRACT["status"] == "FROZEN_DEVELOPMENT_ONLY_NO_NEW_SCIENTIFIC_SEEDS"
    assert CONTRACT["data_custody"]["new_scientific_seeds_authorized"] is False
    assert set(CONTRACT["data_custody"]["forbidden"]) == {
        "751", "752", "753", "754", "755", "756"
    }


def test_t0_preserves_program_q_stop_and_does_not_freeze_architecture_early() -> None:
    assert CONTRACT["ancestry"]["program_q_verdict_preserved"] == (
        "STOP_Q_NO_REPLICATED_LEARNED_ADAPTATION"
    )
    freeze = CONTRACT["architecture_freeze"]
    assert freeze["gru_units"] is None
    assert freeze["quantile_count"] is None
    assert freeze["loss_weights"] is None


def test_t0_requires_strong_mpc_before_learning() -> None:
    questions = " ".join(CONTRACT["t0_questions"])
    assert "ReT-aligned" in questions
    assert "residual observable headroom" in CONTRACT["purpose"]
    assert CONTRACT["nested_arms"]["M0"].startswith("strong ReT-aligned")
