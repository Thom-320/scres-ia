from supply_chain.program_f import (
    CONTEXTS, OBSERVATION_KEYS, make_sim, materialize_tape,
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


def test_risk_scores_are_noisy_signal_not_true_latent_context():
    """The three *_score channels must equal the tape's noisy one-week-lead signal
    (values on the {0.1, 0.8} grid), never a clean {0,1} one-hot of the true next
    context -- a hard fail-closed guard against a latent-context leak."""
    tape = materialize_tape(939052, "interdiction_campaign", "schema-audit", weeks=4)
    _, controller, _ = make_sim(tape)
    obs = controller.observation()
    week = min(controller.current_week, int(tape["weeks"]) - 1)
    scores = tape["signals"][week]["scores"]
    assert obs["equipment_condition_score"] == float(scores["equipment_pressure"])
    assert obs["route_threat_score"] == float(scores["interdiction_campaign"])
    assert obs["mission_tempo_score"] == float(scores["mission_surge"])
    for value in (obs["equipment_condition_score"], obs["route_threat_score"],
                  obs["mission_tempo_score"]):
        assert value in (0.1, 0.8), value


def test_observation_never_exposes_the_true_context_string():
    tape = materialize_tape(939053, "equipment_pressure", "schema-audit", weeks=4)
    _, controller, _ = make_sim(tape)
    obs = controller.observation()
    true_context = tape["context_schedule"][controller.current_week]
    assert true_context in CONTEXTS
    assert true_context not in obs
    assert true_context not in set(map(str, obs.values()))
