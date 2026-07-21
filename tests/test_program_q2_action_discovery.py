import numpy as np
import pytest

from supply_chain.program_o_full_des import product_demand_tape
from supply_chain.program_o_full_des_transducer import normalize_scheduler
from supply_chain.program_q2_action_discovery import (
    CENTERED_ACTIONS,
    SEQUENCES,
    count4_calendar_to_sequence8,
    perfect_information_timing_identity,
    sequence8_scheduler,
)


def test_sequence8_is_complete_and_centered_mapping_preserves_q() -> None:
    assert len(SEQUENCES) == 8
    assert len(set(SEQUENCES)) == 8
    assert SEQUENCES[0] == ("P_H", "P_H", "P_H")
    assert SEQUENCES[-1] == ("P_C", "P_C", "P_C")
    assert count4_calendar_to_sequence8((0, 1, 2, 3)) == CENTERED_ACTIONS
    array = normalize_scheduler(sequence8_scheduler())
    assert array.shape == (8, 3)
    # Product index zero is P_C, so summing the encoded row counts P_H.
    assert np.array_equal(array.sum(axis=1), np.asarray([3, 2, 2, 1, 2, 1, 1, 0]))


def test_clairvoyant_weekly_and_batch_feasible_sets_are_identical() -> None:
    result = perfect_information_timing_identity()
    assert result["counts_equal"] is True
    assert result["weekly_clairvoyant_feasible_calendars"] == 16_777_216
    assert result["batch_clairvoyant_feasible_calendars"] == 16_777_216
    assert result["representative_flattened_sequence_count"] == 8


def test_default_tape_still_deterministic_after_scheduler_generalization() -> None:
    first = product_demand_tape(7_400_048, regime_persistence=0.75, dominant_share=0.9)
    second = product_demand_tape(7_400_048, regime_persistence=0.75, dominant_share=0.9)
    assert first == second


def test_scheduler_rejects_nonconsecutive_actions() -> None:
    with pytest.raises(ValueError):
        normalize_scheduler({"0": ("P_H",) * 3, "2": ("P_C",) * 3})
