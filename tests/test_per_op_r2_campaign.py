from __future__ import annotations

import numpy as np

from scripts.run_per_op_r2_campaign import (
    campaign_sequence,
    phase_from_name,
)
from supply_chain.continuous_its_env import make_per_op_buffer_track_a_env


def test_campaign_sequence_is_deterministic_and_sticky() -> None:
    phases = [
        phase_from_name("calm", enabled_risks=("R22", "R23", "R24"), psi=1.0),
        phase_from_name("r2_phi4", enabled_risks=("R22", "R23", "R24"), psi=1.0),
    ]
    a = campaign_sequence(phases, n_blocks=200, rho=0.85, seed=909)
    b = campaign_sequence(phases, n_blocks=200, rho=0.85, seed=909)

    assert [p.name for p in a] == [p.name for p in b]
    stay_rate = np.mean([a[i].name == a[i - 1].name for i in range(1, len(a))])
    assert stay_rate > 0.70
    assert {p.name for p in a} == {"calm", "r2_phi4"}


def test_per_op_r2_campaign_env_restricts_realized_risks_to_r2() -> None:
    env = make_per_op_buffer_track_a_env(
        reward_mode="ReT_excel_delta",
        observation_version="v6",
        risk_level="current",
        risk_frequency_multiplier=4.0,
        risk_impact_multiplier=1.0,
        enabled_risks=("R22", "R23", "R24"),
        max_steps=3,
        priming_enabled=False,
        risk_obs=True,
        init_fracs=[0.0, 0.0, 0.1],
    )
    env.reset(seed=77)
    done = truncated = False
    while not (done or truncated):
        _obs, _reward, done, truncated, _info = env.step(
            np.asarray([0.0, 0.0, 0.1, -1.0], dtype=np.float32)
        )
    risk_ids = {event.risk_id for event in env.unwrapped.sim.risk_events}

    assert risk_ids
    assert risk_ids <= {"R22", "R23", "R24"}
    env.close()
