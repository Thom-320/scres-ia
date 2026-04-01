from __future__ import annotations

import pytest

from supply_chain.external_env_interface import (
    get_observation_fields,
    get_track_b_env_spec,
    make_track_b_env,
)
from supply_chain.supply_chain import MFSCSimulation


def test_track_b_env_spec_matches_minimal_contract() -> None:
    spec = get_track_b_env_spec()
    env = make_track_b_env(max_steps=1)

    assert spec.env_variant == "track_b_adaptive_control"
    assert spec.reward_mode == "ReT_seq_v1"
    assert spec.observation_version == "v7"
    assert tuple(spec.observation_fields) == get_observation_fields("v7")
    assert len(spec.observation_fields) == 46
    assert len(spec.action_fields) == 7
    assert env.observation_space.shape == (46,)
    assert env.action_space.shape == (7,)


def test_v7_observation_exposes_track_b_features_and_contract_metadata() -> None:
    env = make_track_b_env(max_steps=2)
    obs, info = env.reset(seed=7)

    assert info["action_contract"] == "track_b_v1"
    assert obs.shape == (46,)
    track_b_context = info["state_constraint_context"]["track_b_context"]
    assert obs[40] == pytest.approx(track_b_context["op10_down"])
    assert obs[41] == pytest.approx(track_b_context["op12_down"])
    assert obs[42] == pytest.approx(track_b_context["op10_queue_pressure_norm"])
    assert obs[43] == pytest.approx(track_b_context["op12_queue_pressure_norm"])
    assert obs[44] == pytest.approx(track_b_context["rolling_fill_rate_4w"])
    assert obs[45] == pytest.approx(track_b_context["rolling_backorder_rate_4w"])
    assert "op10_q_max" in info["action_constraints"]["base_control_parameters"]
    assert "op12_q_max" in info["action_constraints"]["base_control_parameters"]

    next_obs, _, _, _, step_info = env.step([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    next_track_b = step_info["state_constraint_context"]["track_b_context"]
    assert next_obs.shape == (46,)
    assert next_obs[40] == pytest.approx(next_track_b["op10_down"])
    assert next_obs[41] == pytest.approx(next_track_b["op12_down"])
    assert 0.0 <= next_obs[42] <= 1.0
    assert 0.0 <= next_obs[43] <= 1.0
    assert 0.0 <= next_obs[44] <= 1.0
    assert 0.0 <= next_obs[45] <= 1.0


def test_track_b_action_updates_downstream_dispatch_parameters() -> None:
    env = make_track_b_env(max_steps=1)
    env.reset(seed=11)

    env.step([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0])

    assert env.sim is not None
    assert env.sim.params["op10_q_min"] == pytest.approx(4800.0)
    assert env.sim.params["op10_q_max"] == pytest.approx(5200.0)
    assert env.sim.params["op12_q_min"] == pytest.approx(1200.0)
    assert env.sim.params["op12_q_max"] == pytest.approx(1300.0)


@pytest.mark.parametrize(
    ("buffer_attr", "process_name", "q_min_key", "q_max_key", "target_attr", "qty"),
    [
        (
            "rations_sb_dispatch",
            "_op10_transport_to_cssu",
            "op10_q_min",
            "op10_q_max",
            "rations_cssu",
            111.0,
        ),
        (
            "rations_cssu",
            "_op12_transport_to_theatre",
            "op12_q_min",
            "op12_q_max",
            "rations_theatre",
            222.0,
        ),
    ],
)
def test_simulation_uses_mutable_downstream_dispatch_parameters(
    buffer_attr: str,
    process_name: str,
    q_min_key: str,
    q_max_key: str,
    target_attr: str,
    qty: float,
) -> None:
    sim = MFSCSimulation(risks_enabled=False, seed=7)
    getattr(sim, buffer_attr)._level = 1000.0
    sim.params[q_min_key] = qty
    sim.params[q_max_key] = qty
    if process_name == "_op10_transport_to_cssu":
        sim.params["op10_pt"] = 0.0
    else:
        sim.params["op12_pt"] = 0.0
    sim.env.process(getattr(sim, process_name)())

    sim.env.run(until=25.0)

    assert getattr(sim, buffer_attr).level == pytest.approx(1000.0 - qty)
    assert getattr(sim, target_attr).level == pytest.approx(qty)
