from __future__ import annotations

import numpy as np

from supply_chain.continuous_its_env import make_continuous_its_track_a_env
from supply_chain.external_env_interface import (
    REALIZED_RISK_OBSERVATION_IDS,
    get_observation_fields,
)


def _field(obs: np.ndarray, fields: tuple[str, ...], name: str) -> float:
    return float(obs[fields.index(name)])


def test_v8_observation_shape_matches_realized_risk_schema() -> None:
    env = make_continuous_its_track_a_env(
        reward_mode="ReT_excel_delta",
        observation_version="v8",
        risk_level="current",
        max_steps=2,
        init_frac=0.0,
    )

    obs, _info = env.reset(seed=123)
    fields = get_observation_fields("v8")

    assert obs.shape == (len(fields),)
    for risk_id in REALIZED_RISK_OBSERVATION_IDS:
        suffix = risk_id.lower()
        assert f"active_{suffix}" in fields
        assert f"recent_{suffix}" in fields
        assert f"recent_{suffix}_duration_norm" in fields


def test_v8_observation_exposes_recent_realized_risk_ids() -> None:
    env = make_continuous_its_track_a_env(
        reward_mode="ReT_excel_delta",
        observation_version="v8",
        risk_level="current",
        risk_frequency_multiplier=4.0,
        risk_impact_multiplier=1.5,
        max_steps=12,
        init_frac=0.75,
    )
    fields = get_observation_fields("v8")
    obs, _info = env.reset(seed=123)

    recent_seen = False
    duration_seen = False
    for _ in range(12):
        obs, _reward, done, truncated, _info = env.step(
            np.array([0.1, 0.0], dtype=np.float32)
        )
        recent_seen = recent_seen or any(
            _field(obs, fields, f"recent_{risk_id.lower()}") > 0.0
            for risk_id in REALIZED_RISK_OBSERVATION_IDS
        )
        duration_seen = duration_seen or any(
            _field(obs, fields, f"recent_{risk_id.lower()}_duration_norm") > 0.0
            for risk_id in REALIZED_RISK_OBSERVATION_IDS
        )
        if done or truncated:
            break

    assert recent_seen
    assert duration_seen
