from supply_chain.dra2_experiment import materialize_tape
from supply_chain.dra2_policy_env import ProgramEConvoyEnv, make_identity_normalizers


def make_env():
    tape = materialize_tape(
        899002, "routine", 16, "disposable",
        contract_id="program_e_policy_realizability_v1", tape_prefix="program-e",
    )
    env = ProgramEConvoyEnv([tape], make_identity_normalizers(), random_tapes=False)
    env.reset(seed=2)
    return env


def test_episode_resource_ledger_matches_convoy_counters():
    env = make_env()
    while True:
        action = 1 if env.action_masks()[1] else 0
        _, _, done, _, info = env.step(action)
        if done:
            break
    assert info["episode_departures"] >= 0
    assert info["episode_unavailable_hours"] >= 0
    assert env.sim.op8_convoy_metrics()["op8_convoy_resource_residual"] == 0
