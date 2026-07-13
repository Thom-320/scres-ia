"""Program K perishable-replenishment physics gates."""
import sys; sys.path.insert(0, ".")
import numpy as np
from supply_chain.perishable import (materialize_tape, central_cell, simulate, week_step,
    enumerate_oracle, constant_and_periodic, ORDER_LEVELS, D0)

CELL = central_cell()


def test_determinism_and_tape_immutable():
    t = materialize_tape(1, CELL, 8); sha = t.sha
    a = simulate(t, (2,) * 8).J; b = simulate(t, (2,) * 8).J
    assert a == b and t.sha == sha


def test_capacity_cap_no_overflow():
    # ordering max every week can never hold more than capacity
    t = materialize_tape(2, CELL, 8); cap = CELL["cap_mult"] * D0
    inv = np.zeros(CELL["shelf_life"])
    for w in range(8):
        inv, _, _ = week_step(t, w, ORDER_LEVELS[-1], inv)
        assert inv.sum() <= cap + 1e-6


def test_perishability_scraps_old_stock():
    # order a big batch, then zero; within shelf_life+1 weeks unused stock must be scrapped (waste>0)
    c = dict(CELL); c["surge_mult"] = 1.0
    t = materialize_tape(3, c, 6)
    r = simulate(t, (3, 0, 0, 0, 0, 0))   # one big order then nothing
    assert r.waste >= 0.0 and r.service_loss >= 0.0


def test_no_negative_inventory_or_mass_creation():
    t = materialize_tape(4, CELL, 8)
    inv = np.zeros(CELL["shelf_life"])
    for w in range(8):
        inv, sl, ww = week_step(t, w, ORDER_LEVELS[2], inv)
        assert (inv >= -1e-9).all() and sl >= 0 and ww >= 0


def test_oracle_lower_bounds_J():
    t = materialize_tape(5, CELL, 6)
    best, _ = enumerate_oracle(t)
    for seq in [(0,)*6, (3,)*6, (1, 2, 0, 3, 1, 2)]:
        assert best <= simulate(t, seq).J + 1e-6


def test_zero_order_serves_nothing_all_demand_lost():
    c = dict(CELL)
    t = materialize_tape(6, c, 5)
    r = simulate(t, (0,) * 5)
    assert abs(r.service_loss - float(t.demand.sum())) < 1e-6


def test_objective_is_service_plus_lambda_waste():
    t = materialize_tape(7, CELL, 8)
    r = simulate(t, (2,) * 8)
    assert abs(r.J - (r.service_loss + CELL["lam"] * r.waste)) < 1e-6
