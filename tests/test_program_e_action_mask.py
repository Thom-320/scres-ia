import json
from pathlib import Path

from supply_chain.dra2_experiment import materialize_tape
from supply_chain.dra2_policy_env import ProgramEConvoyEnv, make_identity_normalizers


def make_env():
    tape = materialize_tape(
        899001, "routine", 16, "disposable",
        contract_id="program_e_policy_realizability_v1", tape_prefix="program-e",
    )
    env = ProgramEConvoyEnv([tape], make_identity_normalizers(), random_tapes=False)
    env.reset(seed=1)
    return env


def test_hold_is_always_valid_and_dispatch_mask_tracks_physics():
    env = make_env()
    assert env.action_masks()[0]
    assert env.action_masks()[1] == env.sim.op8_convoy_dispatch_feasible()


def test_invalid_dispatch_is_masked_to_hold_without_creating_resources():
    env = make_env()
    env.sim.op8_convoy_available = False
    before = env.sim.op8_convoy_departures
    _, _, _, _, info = env.step(1)
    assert info["masked"]
    assert info["effective_action"] == 0
    assert env.sim.op8_convoy_departures == before
