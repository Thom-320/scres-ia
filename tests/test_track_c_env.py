"""Track C machinery tests: campaign schedule, thinning, route-aware top-ups."""

from __future__ import annotations

import numpy as np
import pytest

from supply_chain.config import CAMPAIGN_V1_CONFIG
from supply_chain.supply_chain import MFSCSimulation
from supply_chain.track_c_env import make_track_c_env


def _sim(seed: int, campaign=None, route_aware=False, **kw):
    return MFSCSimulation(
        shifts=2,
        risks_enabled=True,
        risk_level="current",
        seed=seed,
        strict_exogenous_crn=True,
        horizon=168.0 * 60,
        campaign_config=campaign,
        replenishment_route_aware=route_aware,
        **kw,
    )


def test_campaign_path_deterministic_and_seed_dependent():
    a = _sim(600001, campaign=dict(CAMPAIGN_V1_CONFIG))
    b = _sim(600001, campaign=dict(CAMPAIGN_V1_CONFIG))
    c = _sim(600002, campaign=dict(CAMPAIGN_V1_CONFIG))
    assert a.campaign_path == b.campaign_path
    assert a.campaign_path != c.campaign_path
    assert a.campaign_path[0][0] == 0.0
    states = {s for _, s in a.campaign_path}
    assert states == {"calm", "campaign"}


def test_campaign_off_is_inert():
    sim = _sim(600001, campaign=None)
    assert sim.campaign_path == []
    assert sim.campaign_state_at(1000.0) == "calm"
    assert sim._campaign_freq_max("R22") == 1.0
    assert sim._campaign_accept_event("R22") is True  # no RNG consumed


def test_thinning_rates_calm_vs_campaign():
    """Realized R24 event rate ~ native in all-calm, ~x3 in all-campaign."""
    counts = {}
    for state in ("calm", "campaign"):
        cfg = dict(CAMPAIGN_V1_CONFIG)
        cfg["initial_state"] = state
        cfg["dwell_calm_weeks_mean"] = 10_000.0
        cfg["dwell_campaign_weeks_mean"] = 10_000.0
        total = 0
        for seed in (1, 2, 3):
            sim = _sim(seed, campaign=cfg)
            sim._start_processes()
            sim.env.run(until=168.0 * 52)
            total += sum(1 for e in sim.risk_events if e.risk_id == "R24")
        counts[state] = total
    # Native R24 ~ 13/yr -> ~39 over 3 seeds; campaign x3 -> ~117.
    assert counts["campaign"] > 1.8 * counts["calm"], counts


def test_route_aware_blocks_topup_until_route_reopens():
    # No _start_processes(): only the route-aware waiter runs, so container
    # movements are attributable to the top-up path alone.
    sim = _sim(
        600001,
        campaign=None,
        route_aware=True,
        initial_buffers={"op9_rations": 1000.0},
    )
    sim._take_down(8)
    sim.inventory_buffer_targets = {"op9_rations": 5000.0}
    level_before = float(sim.rations_sb.level)
    ev = sim._top_up_inventory_buffer("op9_rations", 5000.0)
    assert ev is None  # blocked: queued waiter, no immediate injection
    sim.env.run(until=24.0 * 5)
    assert float(sim.rations_sb.level) == level_before  # blocked while op8 down
    sim._bring_up(8)
    sim.env.run(until=24.0 * 8)
    assert float(sim.rations_sb.level) >= 5000.0  # delivered after reopening


def test_route_aware_off_injects_regardless():
    sim = _sim(
        600001,
        campaign=None,
        route_aware=False,
        initial_buffers={"op9_rations": 1000.0},
        inventory_replenishment_period=168.0,
    )
    sim._start_processes()
    sim.env.run(until=100.0)
    sim._take_down(8)
    sim.inventory_buffer_targets = {"op9_rations": 5000.0}
    ev = sim._top_up_inventory_buffer("op9_rations", 5000.0)
    assert ev is not None  # legacy behavior: injects through the downed route


def test_track_c_env_smoke_and_econ():
    env = make_track_c_env(max_steps=8)
    obs, info = env.reset(seed=600003)
    action = np.zeros(11, dtype=np.float32)
    action[8:11] = 0.25
    for _ in range(8):
        obs, r, term, trunc, info = env.step(action)
        if term or trunc:
            break
    econ = info["track_c_econ"]
    assert econ["n_steps"] >= 1
    assert econ["holding_frac_mean"] > 0.0
    assert 0.0 <= econ["campaign_frac"] <= 1.0
    assert info.get("campaign_state") in ("calm", "campaign")
    assert env.action_space.shape == (11,)


def test_campaign_cycle_format_three_states():
    cfg = {
        "initial_state": "calm",
        "dwell_min_weeks": 1.0,
        "cycle": [
            {"name": "calm", "dwell_mean_weeks": 6.0},
            {"name": "pre_campaign", "dwell_mean_weeks": 3.0,
             "frequency_multipliers": {"R22": 2.0}},
            {"name": "campaign", "dwell_mean_weeks": 5.0,
             "frequency_multipliers": {"R22": 6.0},
             "impact_multipliers": {"R22": 4.0}},
        ],
    }
    sim = _sim(600005, campaign=cfg)
    states = [s for _, s in sim.campaign_path[:9]]
    assert states[:3] == ["calm", "pre_campaign", "campaign"]
    assert states[3:6] == ["calm", "pre_campaign", "campaign"]
    assert sim._campaign_freq_max("R22") == 6.0
    # State-dependent multipliers resolve by current time.
    t_pre = next(t for t, s in sim.campaign_path if s == "pre_campaign")
    t_camp = next(t for t, s in sim.campaign_path if s == "campaign")
    assert sim.campaign_state_at(t_pre + 1.0) == "pre_campaign"
    assert sim.campaign_state_at(t_camp + 1.0) == "campaign"
