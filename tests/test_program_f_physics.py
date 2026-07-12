from supply_chain.program_f import ACTIONS, ConstantPortfolio, materialize_tape, run_policy


def test_program_f_actions_have_exact_equal_budget():
    assert len(ACTIONS) == 6
    assert all(sum(action) == 2 for action in ACTIONS)


def test_program_f_tape_is_deterministic_and_policy_independent():
    left = materialize_tape(939001, "equipment_pressure", "smoke", weeks=4)
    right = materialize_tape(939001, "equipment_pressure", "smoke", weeks=4)
    assert left == right


def test_program_f_episode_conserves_mass_and_threats_across_actions():
    tape = materialize_tape(939002, "mission_surge", "smoke", weeks=4)
    rows = [run_policy(tape, ConstantPortfolio(action)) for action in ACTIONS]
    assert max(abs(row["mass_residual"]) for row in rows) < 1e-6
    assert {row["threat_sha256"] for row in rows} == {tape["threat_sha256"]}
    assert {row["total_token_hours"] for row in rows} == {4 * 168 * 2}


def test_specialized_levers_change_realized_physics():
    tape = materialize_tape(939003, "interdiction_campaign", "smoke", weeks=8)
    m = run_policy(tape, ConstantPortfolio((2, 0, 0)))
    t = run_policy(tape, ConstantPortfolio((0, 2, 0)))
    r = run_policy(tape, ConstantPortfolio((0, 0, 2)))
    m_r11 = sum(row["realized_duration_hours"] for row in m["damage_events"] if row["risk_id"] == "R11")
    t_r11 = sum(row["realized_duration_hours"] for row in t["damage_events"] if row["risk_id"] == "R11")
    m_transport = sum(row["realized_duration_hours"] for row in m["damage_events"] if row["risk_id"] in {"R22", "R23"})
    t_transport = sum(row["realized_duration_hours"] for row in t["damage_events"] if row["risk_id"] in {"R22", "R23"})
    assert m_r11 < t_r11
    assert t_transport < m_transport
    assert r["reserve_units_issued"] >= m["reserve_units_issued"]
