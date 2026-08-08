from __future__ import annotations

import numpy as np
import pytest

from supply_chain.demand_seasonal import SeasonalDemandContract, SeasonalDemandProcess


def test_source_generator_and_observable_mode_are_explicitly_distinct() -> None:
    source = SeasonalDemandProcess(
        SeasonalDemandContract(alpha_range=(0.4, 0.4), gamma_range=(0.2, 0.2)),
        np.random.default_rng(1),
    )
    observable = SeasonalDemandProcess(
        SeasonalDemandContract(
            alpha_range=(0.4, 0.4), gamma_range=(0.2, 0.2),
            forecast_mode="holt_winters_observable",
        ),
        np.random.default_rng(1),
    )
    for week in range(24):
        realised = 2_500.0 + 200.0 * np.sin(2.0 * np.pi * week / 12.0)
        source.observe((week + 1) * 168.0, realised)
        observable.observe((week + 1) * 168.0, realised)
    assert source.forecast_mode == "garrido_generator"
    assert observable.forecast_mode == "holt_winters_observable"
    assert np.isfinite(source.gross_requirements())
    assert np.isfinite(observable.gross_requirements())
    assert source.gross_requirements() != pytest.approx(observable.gross_requirements())


def test_invalid_forecast_mode_fails_closed() -> None:
    with pytest.raises(ValueError, match="forecast_mode"):
        SeasonalDemandContract(forecast_mode="oracle")


def test_sampler_diagnostic_is_not_an_episode_seed() -> None:
    from scripts.characterise_seasonal_demand_engine_v1 import sampler_diagnostics

    result = sampler_diagnostics(2_000)
    assert result["instrument_seed"] == 20260808
    for key in ("alpha", "gamma"):
        assert 0.0 <= result[key]["min"] < result[key]["max"] <= 1.0
        assert result[key]["ks_uniform"] < 0.05
