from supply_chain.paper2_bottleneck import ACTIONS, CONTEXTS, materialize_tape, run_policy


def constant(action):
    return lambda obs: action


def test_equal_team_hours_crn_and_mass():
    tape = materialize_tape(1099001, CONTEXTS[0], "disposable", weeks=4)
    rows = [run_policy(tape, constant(a)) for a in ACTIONS]
    assert len({r["total_token_hours"] for r in rows}) == 1
    assert len({r["consumed_base_threat_sha256"] for r in rows}) == 1
    assert len({r["realized_demand_sha256"] for r in rows}) == 1
    assert max(r["mass_residual"] for r in rows) < 1e-6


def test_each_action_consumes_exactly_one_team():
    tape = materialize_tape(1099002, CONTEXTS[1], "disposable", weeks=3)
    for action in ACTIONS:
        row = run_policy(tape, constant(action))
        assert row["total_token_hours"] == 3 * 168
