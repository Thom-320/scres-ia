from __future__ import annotations

import numpy as np

from scripts.audit_program_q2_reward_contract import summarize_frontier


def test_reward_audit_detects_visible_reward_omission_incentive() -> None:
    calendars = np.asarray([[0] * 8, [1] * 8, [2] * 8], dtype=np.uint8)
    panel = {
        "ret_visible": np.asarray([0.80, 0.90, 0.85]),
        "ret_full": np.asarray([0.80, 0.60, 0.82]),
        "quantity_ret_full": np.asarray([0.80, 0.55, 0.83]),
        "unresolved_orders": np.asarray([0.0, 3.0, 0.0]),
        "unresolved_quantity": np.asarray([0.0, 100.0, 0.0]),
        "lost_orders": np.zeros(3),
        "worst_product_fill": np.asarray([0.80, 0.60, 0.75]),
        "gross_policy_batch_slots": np.full(3, 24.0),
        "gross_production_quantity": np.full(3, 120000.0),
        "charged_downstream_vehicle_hours": np.full(3, 100.0),
    }
    result = summarize_frontier(panel, calendars)
    assert result["visible_champion"]["index"] == 1
    assert result["full_ledger_champion"]["index"] == 2
    assert result["constrained_visible_champion"]["index"] == 2
    assert result["best_positive_unresolved_minus_best_zero_unresolved"] > 0
    assert not result["visible_and_full_champion_same"]
