import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import simpy

from scripts.run_paper2_bottleneck_exact_transducer import (
    CONTAINER_FIELDS,
    KEY_SCHEMA_VERSION,
    RESULT_SCHEMA_VERSION,
    REDUCED_CERTIFICATION_SUITES,
    RNG_FIELDS,
    SIM_STATE_FIELDS,
    _semantic,
    _digest,
    _queued_event_token,
    _resource_token,
    _runtime_alias_token,
    certify_exhaustive,
    certification_provenance,
    feasible_calendar_count,
    feasible_calendars,
    validate_reduced_certification_payload,
    validate_collision_bisimulation_certificate,
)
from supply_chain.paper2_bottleneck import CONTEXTS, materialize_tape


ROOT = Path(__file__).resolve().parent.parent


def _zero_collision_certificate(weeks: int) -> dict:
    nodes = []
    for week in reversed(range(1, weeks + 1)):
        terminal = week == weeks
        nodes.append(
            {
                "obligation_id": f"node:w{week}:n0",
                "week": week,
                "state_id": 0,
                "state_sha256": f"{week:064x}"[-64:],
                "state_sha512": f"{week:0128x}"[-128:],
                "state_bytes": 1,
                "expected_actions": [] if terminal else [0],
                "edges": [] if terminal else [{
                    "action": 0,
                    "incremental_label_sha256": "a" * 64,
                    "child_obligation_id": f"node:w{week + 1}:n0",
                    "child_complete": True,
                }],
                "status": "COMPLETE",
            }
        )
    roots = []
    body = {
        "schema_version": "paper2_collision_bisimulation_v2",
        "key_schema_version": KEY_SCHEMA_VERSION,
        "complete_state_serialization": True,
        "event_payload_serialized": True,
        "resource_users_serialized": True,
        "callback_closure_state_serialized": True,
        "process_target_state_serialized_or_fail_closed": True,
        "runtime_alias_graph_serialized": True,
        "collision_payload_checks": 0,
        "collision_root_count": 0,
        "transition_congruence_checks": 0,
        "node_obligation_count": len(nodes),
        "terminal_node_obligation_count": 1,
        "unresolved_node_obligation_count": 0,
        "unresolved_collision_root_count": 0,
        "all_actions_covered": True,
        "backward_induction_complete": True,
        "node_obligations": nodes,
        "collision_roots": roots,
        "mismatch_examples": [],
        "induction_rule": "test fixture",
        "passed": True,
    }
    body["node_obligation_records_sha256"] = _digest(nodes)
    body["collision_root_records_sha256"] = _digest(roots)
    body["transition_record_sha256"] = _digest({
        "nodes": body["node_obligation_records_sha256"],
        "roots": body["collision_root_records_sha256"],
    })
    body["certificate_sha256"] = _digest(body)
    return body


def _valid_calendar(sequence: tuple[int, ...]) -> bool:
    if not sequence or sequence[0] != 0:
        return False
    return not any(
        sequence[week] != sequence[week - 1]
        and sequence[week - 1] != sequence[week - 2]
        for week in range(2, len(sequence))
    )


def test_calendar_enumerator_matches_closed_form_and_dwell_rule():
    expected = {1: 1, 6: 43, 12: 2_731, 16: 43_691, 24: 11_184_811}
    for weeks, count in expected.items():
        assert feasible_calendar_count(weeks) == count
    calendars = list(feasible_calendars(12))
    assert len(calendars) == expected[12]
    assert len(set(calendars)) == expected[12]
    assert all(_valid_calendar(calendar) for calendar in calendars)


def test_frozen_reduced_suite_hashes_match_committed_tape_generator():
    split_by_role = {
        "w12_five_tape": "transducer_collision_suite_burned",
        "w16_hard_tape": "transducer_hard_state_burned",
    }
    for role, suite in REDUCED_CERTIFICATION_SUITES.items():
        weeks = int(suite["weeks"])
        for seed, context, expected_sha256 in suite["tapes"]:
            tape = materialize_tape(
                seed,
                context,
                split_by_role[role],
                weeks=weeks,
            )
            assert tape["threat_sha256"] == expected_sha256


def test_unknown_mutable_object_fails_closed_in_semantic_key():
    class UnknownMutable:
        def __init__(self):
            self.value = 1

    with pytest.raises(TypeError, match="unknown mutable value"):
        _semantic(UnknownMutable())


def test_simpy_timeout_payload_and_unknown_fields_fail_closed():
    left_env = simpy.Environment()
    right_env = simpy.Environment()
    left = simpy.Timeout(left_env, 4, value={"signal": "left"})
    right = simpy.Timeout(right_env, 4, value={"signal": "right"})
    assert _queued_event_token(left, owner_registry={}) != _queued_event_token(
        right, owner_registry={}
    )
    left.researcher_hidden_state = 7
    with pytest.raises(TypeError, match="unclassified fields"):
        _queued_event_token(left, owner_registry={})


def test_resource_user_state_and_runtime_alias_classes_are_serialized():
    env = simpy.Environment()
    resource = simpy.Resource(env, capacity=1)
    request = resource.request()
    def serializable_process():
        yield "paused"

    generator = serializable_process()
    next(generator)
    request.proc = SimpleNamespace(
        _generator=generator,
        _target=request,
        callbacks=[],
    )
    owners = {id(resource): "resource"}
    before = _resource_token(resource, owner_registry=owners)
    request.usage_since = 12.5
    after = _resource_token(resource, owner_registry=owners)
    assert before != after

    def fake_sim(contract_aliases_queue_event: bool):
        local_env = simpy.Environment()
        queued = simpy.Timeout(local_env, 1, value="same")
        separate = simpy.Event(local_env)
        containers = {
            name: simpy.Container(local_env, capacity=1, init=0)
            for name in CONTAINER_FIELDS
        }
        return SimpleNamespace(
            env=local_env,
            _contract_renewed_event=(
                queued if contract_aliases_queue_event else separate
            ),
            op10_convoy=simpy.Resource(local_env, capacity=1),
            op12_convoy=simpy.Resource(local_env, capacity=1),
            **containers,
        )

    assert _runtime_alias_token(fake_sim(True)) != _runtime_alias_token(
        fake_sim(False)
    )


def test_parent_awaiting_child_process_fails_closed_until_recursive_graph_support():
    env = simpy.Environment()

    def child():
        yield env.timeout(1)

    def parent():
        yield env.process(child())

    env.process(parent())
    env.step()  # start parent and schedule child
    env.step()  # start child and schedule its Timeout
    timeout = next(event for *_prefix, event in env._queue if isinstance(event, simpy.Timeout))
    with pytest.raises(TypeError, match="nested/awaited Process callbacks"):
        _queued_event_token(timeout, owner_registry={})


def test_delayed_divergence_requires_complete_child_obligation_chain():
    certificate = _zero_collision_certificate(4)
    node = next(
        row for row in certificate["node_obligations"]
        if row["obligation_id"] == "node:w2:n0"
    )
    node["edges"][0]["child_obligation_id"] = "node:w4:n0"
    # Recompute all hashes so this exercises graph semantics, not tamper detection.
    certificate["node_obligation_records_sha256"] = _digest(
        certificate["node_obligations"]
    )
    certificate["transition_record_sha256"] = _digest({
        "nodes": certificate["node_obligation_records_sha256"],
        "roots": certificate["collision_root_records_sha256"],
    })
    body = dict(certificate)
    body.pop("certificate_sha256")
    certificate["certificate_sha256"] = _digest(body)
    failures = validate_collision_bisimulation_certificate(
        certificate,
        expected_collision_count=0,
        weeks=4,
    )
    assert any("dangling" in failure for failure in failures)


def test_key_schema_names_required_future_state_families():
    assert KEY_SCHEMA_VERSION.endswith("_v3")
    assert {"rations_sb", "emergency_theatre_reserve"}.issubset(CONTAINER_FIELDS)
    assert set(RNG_FIELDS) == {"rng", "demand_rng", "risk_rng", "regime_rng"}
    assert {
        "pending_backorder_qty",
        "total_unattended_orders",
        "op_down_count",
        "_ret_quantity_risk_units",
        "emergency_reserve_in_transit",
    }.issubset(SIM_STATE_FIELDS)


def test_small_horizon_primary_transducer_matches_every_brute_calendar():
    tape = materialize_tape(
        1_110_001,
        CONTEXTS[0],
        "transducer_pytest_burned",
        weeks=4,
    )
    result = certify_exhaustive(tape, weeks=4, workers=1)
    assert result["complete_horizon_enumeration"] is True
    assert result["calendars_compared"] == feasible_calendar_count(4)
    assert result["primary_transducer_bitwise_certified"] is True
    assert result["mismatch_examples"] == []
    assert result["full_guardrail_label_certified"] is False
    callback_audit = result["all_prefix_callback_audit"]
    assert callback_audit["passed"] is True
    assert callback_audit["semantic_key_evaluations"] == result["prefix_replays"]
    assert callback_audit["unknown_callback_owner_count"] == 0
    assert callback_audit["prefixes_with_nonempty_callback_inventory"] == result[
        "prefix_replays"
    ]
    assert len(callback_audit["layer_prefix_callback_records_sha256"]) == 4


def test_saved_w12_certification_is_complete_but_full_bound_remains_closed():
    path = (
        ROOT
        / "results"
        / "paper2_bottleneck"
        / "exact_transducer_certification_w12.json"
    )
    if not path.exists():
        pytest.skip("long-form W12 certification artifact not generated")
    result = json.loads(path.read_text())
    if result.get("schema_version") != RESULT_SCHEMA_VERSION:
        pytest.skip("saved W12 artifact predates provenance-bound schema")
    assert result["scientific_status"] == (
        "REDUCED_HORIZON_PRIMARY_CERTIFIED_FULL_CONTRACT_FAIL_CLOSED"
    )
    assert len(result["tapes"]) == 5
    assert all(row["calendars_compared"] == 2_731 for row in result["tapes"])
    assert all(row["mismatch_examples"] == [] for row in result["tapes"])
    assert result["summary"]["all_tapes_primary_bitwise_certified"] is True
    assert result["summary"]["full_guardrail_label_certified"] is False
    assert result["summary"]["full_24_week_transducer_authorized"] is False
    assert result["summary"]["h_pi_computed"] is False
    provenance = result["provenance"]
    assert len(provenance["git_commit"]) == 40
    assert len(provenance["producer_sha256"]) == 64
    assert len(provenance["provenance_sha256"]) == 64
    assert provenance["environment"]["environment_sha256"]
    assert all(row["all_prefix_callback_audit"]["passed"] for row in result["tapes"])


def _reduced_payload(role: str) -> dict:
    suite = REDUCED_CERTIFICATION_SUITES[role]
    provenance = certification_provenance()
    count = feasible_calendar_count(suite["weeks"])
    tapes = []
    for seed, context, tape_sha in suite["tapes"]:
        tapes.append({
            "seed": seed,
            "requested_first_context": context,
            "tape_sha256": tape_sha,
            "split": "reduced_certification_burned",
            "complete_horizon_enumeration": True,
            "primary_transducer_bitwise_certified": True,
            "calendars_compared": count,
            "collision_count": 0,
            "prefix_replays": suite["weeks"],
            "collision_bisimulation": _zero_collision_certificate(
                int(suite["weeks"])
            ),
            "all_prefix_callback_audit": {
                "passed": True,
                "unknown_callback_owner_count": 0,
                "semantic_key_evaluations": suite["weeks"],
                "layer_semantic_key_evaluations": [1] * suite["weeks"],
                "layer_callback_inventory": [[{"owner": "MFSCSimulation"}]] * suite["weeks"],
                "prefix_callback_records_sha256": "a" * 64,
                "layer_prefix_callback_records_sha256": ["b" * 64] * suite["weeks"],
                "prefixes_with_nonempty_callback_inventory": suite["weeks"],
                "layer_prefixes_with_nonempty_callback_inventory": [1] * suite["weeks"],
            },
        })
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "scientific_status": "REDUCED_HORIZON_PRIMARY_CERTIFIED_FULL_CONTRACT_FAIL_CLOSED",
        "key_schema_version": KEY_SCHEMA_VERSION,
        "contract_sha256": __import__("hashlib").sha256(
            (ROOT / "contracts" / "paper2_bottleneck_full_horizon_bound_v1.json").read_bytes()
        ).hexdigest(),
        "provenance": provenance,
        "weeks": suite["weeks"],
        "tapes": tapes,
        "summary": {"all_tapes_primary_bitwise_certified": True},
    }


def test_reduced_v3_provenance_roles_callbacks_and_environment_fail_closed():
    payload = _reduced_payload("w16_hard_tape")
    assert validate_reduced_certification_payload(payload, "w16_hard_tape") == []

    payload["tapes"][0]["requested_first_context"] = "mission_surge"
    assert any(
        "seed/context/tape" in failure
        for failure in validate_reduced_certification_payload(payload, "w16_hard_tape")
    )
    payload = _reduced_payload("w16_hard_tape")
    payload["tapes"][0]["all_prefix_callback_audit"]["layer_semantic_key_evaluations"][-1] = 0
    assert any(
        "callback" in failure
        for failure in validate_reduced_certification_payload(payload, "w16_hard_tape")
    )
    payload = _reduced_payload("w16_hard_tape")
    payload["provenance"]["environment"]["packages"]["numpy"] = "tampered"
    assert any(
        "provenance" in failure or "environment" in failure
        for failure in validate_reduced_certification_payload(payload, "w16_hard_tape")
    )
