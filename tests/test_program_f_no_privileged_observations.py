from supply_chain.program_f import (
    OBSERVATION_KEYS, make_sim, materialize_tape,
)


FORBIDDEN_EXACT = {
    "context_schedule", "latent_context", "future_events", "future_duration",
    "oracle_action", "oracle_label", "future_ret", "future_service",
}
FORBIDDEN_TOKENS = ("future", "oracle", "latent", "next_risk", "repair_duration", "ret_excel")


def test_program_f_observation_schema_is_exact_whitelist():
    tape = materialize_tape(939050, "equipment_pressure", "schema-audit", weeks=4)
    _, controller, _ = make_sim(tape)
    observation = controller.observation()
    assert tuple(observation) == OBSERVATION_KEYS
    assert not (set(observation) & FORBIDDEN_EXACT)
    assert not any(token in key for key in observation for token in FORBIDDEN_TOKENS)


def test_future_event_payload_is_not_exposed_by_observation():
    tape = materialize_tape(939051, "mission_surge", "schema-audit", weeks=4)
    _, controller, _ = make_sim(tape)
    before = controller.observation()
    # Mutating an exact future event after construction must not alter the
    # contemporaneous observation. Signals are left unchanged because their
    # noisy one-week lead is the explicitly allowed information channel.
    tape["base_events"][-1]["magnitude"] += 999999
    after = controller.observation()
    assert before == after
