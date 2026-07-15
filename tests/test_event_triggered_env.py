from __future__ import annotations

import math

import numpy as np

from supply_chain.continuous_its_env import ContinuousItsTrackAEnv
from supply_chain.event_triggered_env import make_event_triggered_track_a_env
from supply_chain.external_env_interface import make_thesis_aligned_training_env


def _kwargs() -> dict:
    return {
        "risks_enabled": False,
        "stochastic_pt": False,
        "priming_enabled": False,
        "max_steps": 16,
        "warmup_hours_override": 0.0,
    }


def _order_signature(sim) -> list[tuple]:
    return [
        (
            int(order.j),
            float(order.OPTj),
            None if order.OATj is None else float(order.OATj),
            float(order.quantity),
            bool(order.lost),
        )
        for order in sim.orders
    ]


def test_seven_daily_holds_match_one_weekly_advance() -> None:
    daily = make_event_triggered_track_a_env(**_kwargs())
    weekly = make_thesis_aligned_training_env(step_size_hours=168.0, **_kwargs())
    daily.reset(seed=90210)
    weekly.reset(seed=90210)

    for _ in range(7):
        _, _, terminated, truncated, info = daily.step([-1.0, 1.0, 1.0])
        assert not terminated and not truncated
        assert info["action_phase"] == "daily_hold"

    posture = ContinuousItsTrackAEnv._action_dict(daily, 1)
    weekly.step(posture)

    assert daily.unwrapped.sim.env.now == weekly.sim.env.now
    assert daily.unwrapped.sim._inventory_detail() == weekly.sim._inventory_detail()
    assert _order_signature(daily.unwrapped.sim) == _order_signature(weekly.sim)
    assert daily.intervention_count == 0
    assert daily.buffer_commitment_count == 0


def test_hold_never_reissues_buffer_commitment() -> None:
    env = make_event_triggered_track_a_env(**_kwargs())
    env.reset(seed=123)
    initial_targets = dict(env.unwrapped.sim.inventory_buffer_targets)
    initial_events = list(env.unwrapped.sim.risk_events)

    for _ in range(3):
        env.step([-1.0, 1.0, 1.0])

    assert env.intervention_count == 0
    assert env.buffer_commitment_count == 0
    assert env.pending_buffer_commitment is None
    assert env.unwrapped.sim.inventory_buffer_targets == initial_targets
    assert env.unwrapped.sim.risk_events == initial_events


def test_pending_buffer_commitment_is_immutable_and_activates_at_168h() -> None:
    env = make_event_triggered_track_a_env(**_kwargs())
    env.reset(seed=456)
    start = float(env.unwrapped.sim.env.now)

    env.step([1.0, 0.5, -1.0])
    pending = dict(env.pending_buffer_commitment or {})
    assert pending["fraction"] == 0.5
    assert pending["activates_at"] == start + 168.0
    assert env.current_frac == 0.0

    # A second intervention may update shifts but cannot overwrite the buffer.
    env.step([1.0, 1.0, 1.0])
    assert env.requested_shift == 3
    assert env.pending_buffer_commitment == pending
    assert env.buffer_commitment_count == 1
    assert env.rejected_buffer_commitment_count == 1

    for _ in range(5):
        env.step([-1.0, 0.0, -1.0])
    assert math.isclose(float(env.unwrapped.sim.env.now), start + 168.0)
    assert env.pending_buffer_commitment is None
    assert env.current_frac == 0.5
    assert env.unwrapped.sim.inventory_buffer_targets


def test_shift_ramps_one_level_per_day_and_hold_preserves_request() -> None:
    env = make_event_triggered_track_a_env(
        **_kwargs(), surge_budget_hours=10_000.0
    )
    env.reset(seed=789)
    _, _, _, _, info1 = env.step([1.0, 0.0, 1.0])
    assert info1["requested_shift"] == 3
    assert info1["effective_shift"] == 2
    _, _, _, _, info2 = env.step([-1.0, 0.0, -1.0])
    assert info2["requested_shift"] == 3
    assert info2["effective_shift"] == 3


def test_action_validation_and_observation_contains_only_control_state_tail() -> None:
    env = make_event_triggered_track_a_env(**_kwargs())
    obs, _ = env.reset(seed=1)
    assert obs.shape[0] == env.env.observation_space.shape[0] + 7
    assert np.all((obs[-7:] >= 0.0) & (obs[-7:] <= 1.0))
    try:
        env.step([1.0, 0.0])
    except ValueError as exc:
        assert "shape (3,)" in str(exc)
    else:
        raise AssertionError("invalid action shape was accepted")


def test_hold_does_not_change_per_risk_crn_or_scale_r3() -> None:
    common = {
        "risks_enabled": True,
        "stochastic_pt": False,
        "priming_enabled": False,
        "max_steps": 16,
        "warmup_hours_override": 0.0,
        "risk_rng_mode": "per_risk",
        "risk_frequency_multipliers_by_id": {"R22": 4.0, "R24": 4.0},
        "risk_impact_multipliers_by_id": {"R22": 1.5, "R24": 1.5},
    }
    daily = make_event_triggered_track_a_env(**common)
    weekly = make_thesis_aligned_training_env(step_size_hours=168.0, **common)
    daily.reset(seed=314159)
    weekly.reset(seed=314159)

    for _ in range(7):
        daily.step([-1.0, 1.0, 1.0])
    weekly.step(ContinuousItsTrackAEnv._action_dict(daily, 1))

    daily_events = [
        (e.risk_id, e.start_time, e.end_time, e.duration) for e in daily.unwrapped.sim.risk_events
    ]
    weekly_events = [
        (e.risk_id, e.start_time, e.end_time, e.duration) for e in weekly.sim.risk_events
    ]
    assert daily_events == weekly_events
    assert daily.unwrapped.sim.risk_frequency_multipliers_by_id.get("R3", 1.0) == 1.0
    assert daily.unwrapped.sim.risk_impact_multipliers_by_id.get("R3", 1.0) == 1.0
