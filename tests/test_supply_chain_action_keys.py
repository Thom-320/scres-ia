from __future__ import annotations

import pytest

from supply_chain.supply_chain import MFSCSimulation


def test_step_rejects_unknown_action_key_before_mutating_params() -> None:
    sim = MFSCSimulation(horizon=24)
    original = dict(sim.params)

    with pytest.raises(KeyError, match=r"unknown action key\(s\): op8_rop"):
        sim.step({"op8_rop": 24}, step_hours=1)

    assert sim.params == original
    assert sim.env.now == 0


def test_step_accepts_mutable_and_declared_pseudo_action_keys() -> None:
    sim = MFSCSimulation(horizon=24)

    sim.step(
        {
            "assembly_shifts": 2,
            # This is a declared compatibility pseudo-action. Liveness is
            # contract-dependent and is audited separately.
            "op5_q": 1.0,
        },
        step_hours=1,
    )

    assert sim.params["assembly_shifts"] == 2
    assert sim.env.now == 1
