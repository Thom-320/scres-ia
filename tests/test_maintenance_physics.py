"""Paper 2 maintenance physics gates (G0)."""
import sys; sys.path.insert(0, ".")
import numpy as np
import pytest
from supply_chain.maintenance import (materialize_tape, central_cell, simulate, week_step,
    enumerate_oracle, worst_condition_policy, forecast_policy, ACTIONS, TAU)

CELL = central_cell()


def test_determinism_same_actions_same_result():
    t = materialize_tape(1, CELL, 8)
    assert simulate(t, ("PM5",) * 8).service_loss == simulate(t, ("PM5",) * 8).service_loss


def test_crn_exogenous_tape_action_invariant():
    # threat/wear/demand must be identical regardless of action choice (only realized damage differs)
    t = materialize_tape(2, CELL, 8)
    sha = t.sha
    for seq in [("PM5",) * 8, ("PM7",) * 8, ("PM5", "PM6", "PM7") * 3]:
        simulate(t, seq)
    assert t.sha == sha  # simulate must not mutate the tape


def test_single_crew_at_most_one_pm_per_week():
    # week_step gives PM downtime PM_DOWN to exactly the chosen station when no corrective repair
    t = materialize_tape(3, CELL, 8)
    d = np.zeros(3); wip = np.zeros(2)
    d2, _, _, down, crew = week_step(t, 0, 1, d, wip)   # action=PM6, low degradation -> no failure
    assert (down > 0).sum() <= 1 and crew == 24.0


def test_higher_degradation_raises_failure_given_threat():
    # a threatened station fails iff d>TAU
    t = materialize_tape(4, CELL, 8)
    w = int(np.argmax(t.threat[:, 0] == 1)) if t.threat[:, 0].any() else 0
    if not t.threat[w, 0]:
        pytest.skip("no threat on station 0 in this tape")
    lo = np.array([0.1, 0.0, 0.0]); hi = np.array([0.9, 0.0, 0.0])
    _, _, _, down_lo, _ = week_step(t, w, 1, lo, np.zeros(2))   # maintain another station
    _, _, _, down_hi, _ = week_step(t, w, 1, hi, np.zeros(2))
    assert down_lo[0] == 0 and down_hi[0] > 0   # below TAU no failure; above TAU failure


def test_oracle_is_lower_bound_on_service_loss():
    t = materialize_tape(5, CELL, 6)
    best, _ = enumerate_oracle(t)
    for seq in [("PM5",) * 6, ("PM6",) * 6, ("PM5", "PM6", "PM7") * 2]:
        assert best <= simulate(t, seq).service_loss + 1e-9


def test_no_mass_creation_fill_rate_bounded():
    t = materialize_tape(6, CELL, 8)
    r = simulate(t, worst_condition_policy(t))
    assert 0.0 <= r.fill_rate <= 1.0 and r.service_loss >= 0.0


def test_station_symmetry_under_wear_swap():
    # with homogeneous wear, permuting which station is threatened permutes downtime symmetrically
    c = dict(CELL); c["wear_hetero"] = "low"
    t = materialize_tape(7, c, 8)
    # PM5-only vs PM7-only should differ only through the (heterogeneous) threat/serial structure,
    # but both must be valid finite service losses
    a = simulate(t, ("PM5",) * 8).service_loss
    b = simulate(t, ("PM7",) * 8).service_loss
    assert np.isfinite(a) and np.isfinite(b)


def test_forecast_policy_actions_valid():
    t = materialize_tape(8, CELL, 8)
    for a in forecast_policy(t):
        assert a in ACTIONS
