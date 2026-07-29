from __future__ import annotations

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from scripts.run_track_a_headroom_campaign import (
    behavior_clone_policy,
    build_env,
    listed_sequence,
    markov_sequence,
    oracle_action_map,
    parse_phase,
)
from supply_chain.continuous_its_env import make_continuous_its_track_a_env


def test_parse_phase_maps_family_to_enabled_risks() -> None:
    r1 = parse_phase("R1_phi4_psi1.5")
    r24 = parse_phase("R24_phi8_psi2")
    mixed = parse_phase("mixed_phi1_psi1")

    assert r1.family == "R1"
    assert r1.phi == 4.0
    assert r1.psi == 1.5
    assert r1.enabled_risks is not None
    assert set(r1.enabled_risks) == {"R11", "R12", "R13", "R14"}
    assert r24.enabled_risks == ("R24",)
    assert mixed.enabled_risks is None


def test_listed_sequence_repeats_in_order() -> None:
    phases = [parse_phase("R1_phi1_psi1"), parse_phase("R2_phi4_psi1")]

    seq = listed_sequence(phases, 5)

    assert [p.name for p in seq] == [
        "R1_phi1_psi1",
        "R2_phi4_psi1",
        "R1_phi1_psi1",
        "R2_phi4_psi1",
        "R1_phi1_psi1",
    ]


def test_markov_sequence_is_deterministic_and_has_all_phases() -> None:
    phases = [
        parse_phase("R1_phi1_psi1"),
        parse_phase("R2_phi4_psi1"),
        parse_phase("mixed_phi8_psi2"),
    ]

    a = markov_sequence(phases, n_blocks=100, rho=0.75, seed=123)
    b = markov_sequence(phases, n_blocks=100, rho=0.75, seed=123)

    assert [p.name for p in a] == [p.name for p in b]
    assert {p.name for p in a} == {p.name for p in phases}


def test_continuous_env_restricts_r24_phase_to_r24_risks() -> None:
    phase = parse_phase("R24_phi4_psi1")
    env = make_continuous_its_track_a_env(
        reward_mode="ReT_excel_delta",
        observation_version="v6",
        risk_level="current",
        risk_frequency_multiplier=phase.phi,
        risk_impact_multiplier=phase.psi,
        enabled_risks=phase.enabled_risks,
        max_steps=3,
        priming_enabled=False,
        risk_obs=True,
        init_frac=0.1,
    )
    env.reset(seed=77)
    done = truncated = False
    while not (done or truncated):
        _obs, _reward, done, truncated, _info = env.step(
            np.asarray([0.1, -1.0], dtype=np.float32)
        )
    risk_ids = {event.risk_id for event in env.unwrapped.sim.risk_events}

    assert risk_ids
    assert risk_ids <= {"R24"}
    env.close()


def test_oracle_action_map_selects_best_action_by_phase() -> None:
    phases = [parse_phase("R1_phi1_psi1"), parse_phase("R2_phi4_psi1")]
    sequence = listed_sequence(phases, 2)
    statics = [
        {
            "label": "a",
            "action": (0.0, -1.0),
            "by_phase": {
                "R1_phi1_psi1": {"excel": 1.0},
                "R2_phi4_psi1": {"excel": 0.0},
            },
        },
        {
            "label": "b",
            "action": (0.1, 0.0),
            "by_phase": {
                "R1_phi1_psi1": {"excel": 0.0},
                "R2_phi4_psi1": {"excel": 1.0},
            },
        },
    ]

    actions = oracle_action_map(statics, sequence)

    assert actions["R1_phi1_psi1"] == (0.0, -1.0)
    assert actions["R2_phi4_psi1"] == (0.1, 0.0)


def test_behavior_clone_policy_reduces_action_mse() -> None:
    phase = parse_phase("R1_phi1_psi1")
    venv = DummyVecEnv(
        [
            lambda: build_env(
                phase=phase,
                reward="ReT_excel_delta",
                obs_v="v6",
                max_steps=2,
                init_frac=0.1,
                holding_cost=0.0,
                shift_cost=0.001,
                cvar_alpha=0.2,
                step_size_hours=168.0,
            )
        ]
    )
    model = PPO("MlpPolicy", venv, seed=123, verbose=0, n_steps=2, batch_size=2)
    obs = np.repeat(venv.reset(), repeats=8, axis=0).astype(np.float32)
    actions = np.tile(np.asarray([[0.1, -1.0]], dtype=np.float32), (8, 1))

    stats = behavior_clone_policy(
        model,
        obs,
        actions,
        epochs=20,
        batch_size=4,
        seed=123,
    )

    assert stats["bc_loss_final"] < stats["bc_loss_initial"]
