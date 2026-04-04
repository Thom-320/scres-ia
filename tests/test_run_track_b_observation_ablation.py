from __future__ import annotations

from scripts.run_track_b_observation_ablation import (
    FORECAST_FIELD_NAMES,
    ForecastMaskWrapper,
    OBS_ABLATION_CONFIGS,
)
from supply_chain.external_env_interface import get_observation_fields, make_track_b_env


def test_forecast_mask_wrapper_zeros_only_forecast_channels() -> None:
    env = ForecastMaskWrapper(make_track_b_env(observation_version="v7", max_steps=1))
    obs, _ = env.reset(seed=7)

    fields = tuple(get_observation_fields("v7"))
    forecast_indices = [fields.index(name) for name in FORECAST_FIELD_NAMES]

    assert all(obs[idx] == 0.0 for idx in forecast_indices)
    assert env.observation_space.shape == (46,)
    env.close()


def test_observation_ablation_configs_match_intended_contracts() -> None:
    assert OBS_ABLATION_CONFIGS["v7_full"].observation_version == "v7"
    assert OBS_ABLATION_CONFIGS["v7_no_forecast"].wrapper is ForecastMaskWrapper
    assert OBS_ABLATION_CONFIGS["v5_7d"].observation_version == "v5"
