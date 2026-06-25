"""Tests for the exogenous scenario-tape generator (`learning_extension_v1`).

These pin the properties the paper contract relies on: determinism / CRN replay,
that `rho` is a real persistence knob (H2 needs `d/d_rho > 0`), that the two phase
processes are independent (the user's 2026-06-24 decision), and that a memoryless
baseline exists.
"""

from __future__ import annotations

import numpy as np
import pytest

from supply_chain.scenario_tape import (
    DEFAULT_DEMAND_MULTIPLIERS,
    DEFAULT_DISRUPTION_LEVELS,
    generate_scenario_tape,
)


def _stay_rate(values) -> float:
    arr = np.asarray(values)
    if len(arr) < 2:
        return float("nan")
    return float(np.mean(arr[1:] == arr[:-1]))


def test_tape_is_deterministic_and_replayable() -> None:
    a = generate_scenario_tape(200, rho_disruption=0.7, rho_demand=0.6, seed=11)
    b = generate_scenario_tape(200, rho_disruption=0.7, rho_demand=0.6, seed=11)
    assert [p.disruption_level for p in a.blocks] == [p.disruption_level for p in b.blocks]
    assert [p.demand_multiplier for p in a.blocks] == [p.demand_multiplier for p in b.blocks]
    # Different seed -> different tape (overwhelmingly likely at this length).
    c = generate_scenario_tape(200, rho_disruption=0.7, rho_demand=0.6, seed=12)
    assert [p.disruption_phase for p in a.blocks] != [p.disruption_phase for p in c.blocks]


def test_phases_map_to_declared_grids() -> None:
    tape = generate_scenario_tape(50, rho_disruption=0.5, rho_demand=0.5, seed=3)
    assert tape.disruption_levels == DEFAULT_DISRUPTION_LEVELS
    assert tape.demand_multipliers == DEFAULT_DEMAND_MULTIPLIERS
    for p in tape.blocks:
        assert p.disruption_level == DEFAULT_DISRUPTION_LEVELS[p.disruption_phase]
        assert p.demand_multiplier == DEFAULT_DEMAND_MULTIPLIERS[p.demand_phase]
    assert len(tape) == 50


def test_rho_is_a_monotone_persistence_knob() -> None:
    # Higher rho -> blocks stick to the same phase more often. This is the knob H2's
    # d/d_rho gradient sweeps over, so it must be monotone and roughly calibrated.
    n = 6000
    low = generate_scenario_tape(n, rho_disruption=1 / 3, rho_demand=1 / 3, seed=21)
    high = generate_scenario_tape(n, rho_disruption=0.9, rho_demand=1 / 3, seed=21)
    low_stay = _stay_rate([p.disruption_phase for p in low.blocks])
    high_stay = _stay_rate([p.disruption_phase for p in high.blocks])
    # Memoryless baseline ~ 1/3 stay-rate; sticky ~ 0.9.
    assert low_stay == pytest.approx(1 / 3, abs=0.05)
    assert high_stay == pytest.approx(0.9, abs=0.05)
    assert high_stay > low_stay + 0.4


def test_disruption_and_demand_processes_are_independent() -> None:
    # Two independent processes (user decision): the demand chain must not co-move with
    # the disruption chain even when their rho values differ.
    n = 8000
    tape = generate_scenario_tape(n, rho_disruption=0.85, rho_demand=0.4, seed=7)
    d = np.array([p.disruption_phase for p in tape.blocks], dtype=float)
    m = np.array([p.demand_phase for p in tape.blocks], dtype=float)
    corr = np.corrcoef(d, m)[0, 1]
    assert abs(corr) < 0.05
    # And each kept its own persistence.
    assert _stay_rate(d) == pytest.approx(0.85, abs=0.05)
    assert _stay_rate(m) == pytest.approx(0.4, abs=0.05)


def test_rho_out_of_range_is_rejected() -> None:
    # Below 1/n_states is not a valid symmetric-persistence chain.
    with pytest.raises(ValueError, match="rho_disruption"):
        generate_scenario_tape(10, rho_disruption=0.1, rho_demand=0.5, seed=1)
    with pytest.raises(ValueError, match="rho_demand"):
        generate_scenario_tape(10, rho_disruption=0.5, rho_demand=1.5, seed=1)


def test_empty_tape_is_allowed() -> None:
    tape = generate_scenario_tape(0, rho_disruption=0.5, rho_demand=0.5, seed=1)
    assert len(tape) == 0
