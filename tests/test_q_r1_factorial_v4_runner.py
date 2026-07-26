from __future__ import annotations

import numpy as np
import pytest

from scripts.run_q_r1_matched_retention_factorial_v4 import (
    build_histories,
    estimands,
    evaluate_neural_arm,
    load_authority,
    static_rows,
)


class ConstantModel:
    def predict(self, observation, *, state, episode_start, deterministic):
        assert observation.shape == (23,)
        assert deterministic is True
        return np.asarray(0), state


def test_draft_authorizes_only_burned_instrument_preflight() -> None:
    contract, receipt = load_authority("instrument-preflight")
    assert contract["status"] == "DRAFT_PROSPECTIVE_UNOPENED_NOT_AUTHORITY"
    assert receipt is None
    with pytest.raises(RuntimeError, match="forbidden until the contract is frozen"):
        load_authority("development-worker")


def test_neural_arm_emits_complete_campaign_rows_with_one_checkpoint_hash() -> None:
    histories = build_histories([7_570_801], (0.90,))
    rows = evaluate_neural_arm(
        ConstantModel(),
        histories=histories,
        arm="P1_H1",
        retained_prior=True,
        reset_hidden_at_boundaries=False,
        checkpoint_sha256="abc",
    )
    assert len(rows) == 12
    assert {row["checkpoint_sha256"] for row in rows} == {"abc"}
    assert {row["arm"] for row in rows} == {"P1_H1"}
    for field in (
        "early_ret_complete_cohort",
        "early_ret_visible",
        "worst_product_fill",
        "unresolved_orders",
        "unresolved_quantity",
        "lost_orders",
        "lost_quantity",
        "service_loss",
    ):
        assert all(field in row for row in rows)


def test_estimands_include_the_neural_premium_from_common_rows() -> None:
    rows = []
    values = {
        "P0_H0": 0.50,
        "P1_H0": 0.55,
        "P0_H1": 0.52,
        "P1_H1": 0.60,
        "structured_reset": 0.53,
        "structured_retained": 0.58,
    }
    for arm, value in values.items():
        rows.append(
            {
                "arm": arm,
                "history_root": 1,
                "kappa": 0.9,
                "campaign_index": 0,
                "early_ret_complete_cohort": value,
            }
        )
    result = estimands(rows)
    assert result["explicit_context_value"]["mean"] == pytest.approx(0.05)
    assert result["raw_recurrent_memory_value"]["mean"] == pytest.approx(0.02)
    assert result["structured_retained_value"]["mean"] == pytest.approx(0.05)
    assert result["neural_premium"]["mean"] == pytest.approx(0.02)


def test_static_rows_emit_the_same_mandatory_service_aliases() -> None:
    histories = build_histories([7_570_801], (0.90,))
    rows = static_rows(histories, calendar=[0] * 8)
    assert len(rows) == 12
    assert all(row["service_loss"] == row["service_loss_auc"] for row in rows)
    assert all(row["whole_campaign_ret"] == row["ret_visible"] for row in rows)
