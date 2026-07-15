import json
from pathlib import Path

import numpy as np
import pytest

from scripts.screen_program_o_exact_transducer import (
    complete_calendars as binary_calendars,
    make_tape,
    simulate as scalar_simulate,
)
from scripts.screen_program_o_full_action_transducer import (
    MATRIX_KEYS,
    SAFE_EQUAL,
    SAFE_HIGHER,
    SAFE_LOWER,
    calendar_index,
    execute_stage,
    frozen_profiles,
    full_action_calendars,
    safe_oracle_indices,
    scheduler_array,
    simulate_frontier,
)


ROOT = Path(__file__).resolve().parent.parent
CONTRACT = json.loads(
    (ROOT / "contracts/program_o_full_action_transducer_v1.json").read_text()
)
OLD_CONTRACT = json.loads(
    (ROOT / "contracts/program_o_exact_transducer_v1.json").read_text()
)


def test_complete_base4_frontier_and_index_bijection():
    calendars = full_action_calendars()

    assert calendars.shape == (4**8, 8)
    assert len(np.unique(calendars, axis=0)) == 4**8
    for index in (0, 1, 255, 256, 49151, 65535):
        assert calendar_index(calendars[index]) == index


def test_action_mapping_and_frozen_profile_registry():
    centered = scheduler_array(CONTRACT, "centered_minority_v1")
    blocked = scheduler_array(CONTRACT, "blocked_right_v1")

    assert centered.tolist() == [[1, 1, 1], [1, 0, 1], [0, 1, 0], [0, 0, 0]]
    assert blocked.tolist() == [[1, 1, 1], [1, 1, 0], [0, 0, 1], [0, 0, 0]]
    profiles = frozen_profiles(CONTRACT)
    assert len(profiles) == 13
    assert len({profile["profile_id"] for profile in profiles}) == 13
    assert sum(profile["role"] == "primary" for profile in profiles) == 4
    assert sum(profile["role"] == "exact_null" for profile in profiles) == 3


def test_vectorized_full_action_transducer_matches_historical_scalar_binary_subset():
    tape = make_tape(7410000, persistence=0.75, dominant_share=0.9)
    panel = simulate_frontier(
        tape=tape,
        contract=CONTRACT,
        scheduler_id="centered_minority_v1",
        complete_substitution=False,
    )
    binary = binary_calendars()
    for binary_index in (0, 1, 85, 170, 255):
        old_calendar = binary[binary_index]
        base4_calendar = tuple(2 if action == "C_MAJOR" else 1 for action in old_calendar)
        index = calendar_index(base4_calendar)
        scalar = scalar_simulate(
            tape,
            old_calendar,
            OLD_CONTRACT,
            complete_substitution=False,
        )
        assert panel["ret_visible"][index] == pytest.approx(scalar["ret"], abs=1e-14)
        assert panel["ret_full"][index] == pytest.approx(scalar["ret_full"], abs=1e-14)
        assert panel["unresolved_quantity"][index] == pytest.approx(
            scalar["unfulfilled_quantity"]
        )
        assert panel["worst_order_fill"][index] == pytest.approx(
            scalar["worst_product_fill"]
        )
        assert panel["visible_rows"][index] == scalar["visible_rows"]


@pytest.mark.parametrize(
    "scheduler_id",
    ("centered_minority_v1", "blocked_right_v1", "blocked_left_v1"),
)
def test_fungible_null_is_exact_for_every_calendar_and_guardrail(scheduler_id):
    tape = make_tape(7410000, persistence=0.9, dominant_share=0.9)
    panel = simulate_frontier(
        tape=tape,
        contract=CONTRACT,
        scheduler_id=scheduler_id,
        complete_substitution=True,
    )

    assert tuple(panel) == MATRIX_KEYS
    assert all(np.all(values == values[:1]) for values in panel.values())
    assert np.max(np.abs(panel["mass_residual_aggregate"])) <= 1e-8


def test_safe_oracle_rejects_visible_ret_gain_bought_with_omissions():
    panel = {
        key: np.zeros((1, 2), dtype=float)
        for key in set((*MATRIX_KEYS, *SAFE_HIGHER, *SAFE_LOWER, *SAFE_EQUAL))
    }
    panel["ret_visible"][:] = [[0.5, 1.0]]
    for key in SAFE_HIGHER:
        panel[key][:] = [[1.0, 1.0]]
    for key in SAFE_LOWER:
        panel[key][:] = [[0.0, 0.0]]
    for key in SAFE_EQUAL:
        panel[key][:] = [[1.0, 1.0]]
    panel["omitted_rows"][:] = [[0.0, 1.0]]

    assert safe_oracle_indices(panel, 0).tolist() == [0]


def test_validation_cannot_open_without_additive_freeze(tmp_path):
    with pytest.raises(RuntimeError, match="additive validation freeze is absent"):
        execute_stage(
            contract_path=ROOT / "contracts/program_o_full_action_transducer_v1.json",
            output_root=tmp_path / "results",
            stage="validation",
            workers=1,
            validation_freeze_path=tmp_path / "missing-freeze.json",
        )
