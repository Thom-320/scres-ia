import numpy as np

from supply_chain.program_o_state_rich import StateRichObservation
from supply_chain.program_q2_observation_discovery import Q21_NAMES, q21_preclip


def observation(**overrides):
    payload = {
        "week": 0,
        "decision_time": 100.0,
        "on_hand": (0.0, 0.0),
        "locked_pipeline": (0.0, 0.0),
        "backlog_quantity": (0.0, 0.0),
        "backlog_orders": (0, 0),
        "max_backlog_age": (0.0, 0.0),
        "in_flight_quantity": (0.0, 0.0),
        "belief_c": 0.5,
        "predicted_share_c": 0.5,
        "previous_action": None,
        "remaining_decisions": 8,
        "observation_sha256": "x",
    }
    payload.update(overrides)
    return StateRichObservation(**payload)


def test_q21_preclip_exposes_saturation_instead_of_hiding_it() -> None:
    vector = q21_preclip(
        observation(backlog_quantity=(240_000.0, 0.0), max_backlog_age=(2_688.0, 0.0))
    )
    assert vector.shape == (len(Q21_NAMES),)
    assert vector[4] == 2.0
    assert vector[8] == 2.0
    assert np.clip(vector, 0.0, 1.0)[4] == 1.0


def test_previous_action_one_hot_is_stable() -> None:
    none = q21_preclip(observation(previous_action=None))
    action = q21_preclip(observation(previous_action=2))
    assert none[18] == 1.0
    assert action[16] == 1.0
