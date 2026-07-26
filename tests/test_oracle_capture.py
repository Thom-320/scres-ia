from __future__ import annotations

import numpy as np

from supply_chain.oracle_capture import (
    Campaign,
    N_CALENDARS,
    calendar_index,
    pooled_capture,
)


def _campaign(root: int, labels: np.ndarray) -> Campaign:
    return Campaign(
        history_root=root,
        campaign_index=1,
        persistence_mode="binary_0.9",
        kappa=0.9,
        retained_prior=0.9,
        initial_regime="P_C",
        labels=labels,
        frozen_indices={"retained": 0, "reset": 0},
    )


def test_calendar_index_matches_base_four_contract() -> None:
    assert calendar_index([0, 0, 3, 3, 3, 3, 3, 3]) == 4095
    assert calendar_index([3] * 8) == N_CALENDARS - 1


def test_pooled_capture_penalizes_regression_in_zero_headroom_campaign() -> None:
    zero_headroom_labels = np.zeros(N_CALENDARS, dtype=np.float64)
    zero_headroom_labels[0] = 1.0
    zero_headroom = _campaign(1, zero_headroom_labels)

    headroom_labels = np.zeros(N_CALENDARS, dtype=np.float64)
    headroom_labels[0] = 1.0
    headroom = _campaign(2, headroom_labels)

    result = pooled_capture(
        [zero_headroom, headroom],
        calendars={
            zero_headroom.key: [0, 0, 0, 0, 0, 0, 0, 1],
            headroom.key: [0, 0, 0, 0, 0, 0, 0, 0],
        },
        bar_values={zero_headroom.key: 1.0, headroom.key: 0.0},
        rng=np.random.default_rng(7),
    )

    assert result["n_zero_headroom_campaigns"] == 1
    assert result["n_zero_headroom_regressions"] == 1
    assert result["zero_headroom_numerator"] == -1.0
    assert result["pooled_ratio"] == 0.0
