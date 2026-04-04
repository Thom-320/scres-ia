from __future__ import annotations

import numpy as np

from scripts.eval_track_b_forecast_sensitivity import (
    FORECAST_FIELD_NAMES,
    ForecastScrambleWrapper,
    ForecastZeroWrapper,
)
from supply_chain.external_env_interface import get_observation_fields, make_track_b_env


def test_forecast_zero_wrapper_zeros_only_forecast_channels() -> None:
    env = ForecastZeroWrapper(make_track_b_env(observation_version="v7", max_steps=1))
    obs, _ = env.reset(seed=5)
    fields = tuple(get_observation_fields("v7"))
    indices = [fields.index(name) for name in FORECAST_FIELD_NAMES]
    assert all(obs[idx] == 0.0 for idx in indices)
    env.close()


def test_forecast_scramble_wrapper_replaces_with_bank_values() -> None:
    bank = np.asarray([[0.11, 0.22], [0.33, 0.44], [0.55, 0.66]], dtype=np.float32)
    env = ForecastScrambleWrapper(
        make_track_b_env(observation_version="v7", max_steps=1),
        forecast_bank=bank,
    )
    obs, _ = env.reset(seed=9)
    fields = tuple(get_observation_fields("v7"))
    idx_48h = fields.index(FORECAST_FIELD_NAMES[0])
    idx_168h = fields.index(FORECAST_FIELD_NAMES[1])
    assert any(
        np.allclose([obs[idx_48h], obs[idx_168h]], pair, atol=1e-6)
        for pair in bank
    )
    env.close()
