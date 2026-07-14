import builtins
import json
import copy
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import simpy

import scripts.run_paper2_bottleneck_exact_transducer as exact_module
from scripts.run_paper2_bottleneck_exact_transducer import (
    CONTAINER_FIELDS,
    KEY_SCHEMA_VERSION,
    RESULT_SCHEMA_VERSION,
    REDUCED_CERTIFICATION_SUITES,
    RNG_FIELDS,
    SIM_STATE_FIELDS,
    _semantic,
    _digest,
    _generator_stack,
    _feasible_next,
    _queued_event_token,
    _resource_token,
    _runtime_alias_token,
    markov_completeness_certificate,
    semantic_markov_payload,
    semantic_markov_fingerprint,
    build_transducer,
    certify_exhaustive,
    certification_provenance,
    create_reduced_execution_launch_authorization,
    feasible_calendar_count,
    feasible_calendars,
    launch_reduced_execution_fresh_process,
    validate_reduced_certification_structure,
    validate_reduced_certification_payload,
    verify_independent_reduced_execution,
    validate_collision_bisimulation_certificate,
    validate_markov_completeness_certificate,
)
from supply_chain.paper2_bottleneck import ACTIONS, CONTEXTS, make_sim, materialize_tape
import supply_chain.paper2_bottleneck as paper2_bottleneck_module
from supply_chain.program_f import advance_including

ROOT = Path(__file__).resolve().parent.parent


def _rehash_collision_certificate(certificate: dict) -> None:
    certificate["node_obligation_records_sha256"] = _digest(
        certificate["node_obligations"]
    )
    certificate["collision_root_records_sha256"] = _digest(
        certificate["collision_roots"]
    )
    certificate["request_lag_equivalence_records_sha256"] = _digest(
        certificate["request_lag_equivalence_records"]
    )
    certificate["layer_multiplicity_records_sha256"] = _digest(
        certificate["layer_multiplicity_records"]
    )
    certificate["discarded_transition_ids_sha256"] = _digest(
        [
            row["discarded_transition_id"]
            for row in certificate["collision_roots"]
        ]
    )
    certificate["transition_record_sha256"] = _digest(
        {
            "nodes": certificate["node_obligation_records_sha256"],
            "roots": certificate["collision_root_records_sha256"],
        }
    )
    body = dict(certificate)
    body.pop("certificate_sha256", None)
    certificate["certificate_sha256"] = _digest(body)


def _zero_collision_certificate(weeks: int) -> dict:
    layers = [[{"state_id": 0, "last": 0, "switched": False, "count": 1}]]
    transitions = []
    discarded = []
    for layer_index in range(weeks - 1):
        next_layer = []
        index = {}
        table = {}
        for parent in layers[-1]:
            for action in _feasible_next(parent["last"], parent["switched"]):
                key = (action, action != parent["last"])
                if key not in index:
                    index[key] = len(next_layer)
                    next_layer.append(
                        {
                            "state_id": len(next_layer),
                            "last": action,
                            "switched": action != parent["last"],
                            "count": 0,
                        }
                    )
                else:
                    discarded.append(
                        {
                            "week": layer_index + 2,
                            "parent": parent,
                            "action": action,
                            "target": index[key],
                        }
                    )
                target = index[key]
                next_layer[target]["count"] += parent["count"]
                table[(parent["state_id"], action)] = target
        layers.append(next_layer)
        transitions.append(table)

    nodes = []
    lag_records = []
    for layer_index in reversed(range(weeks)):
        terminal = layer_index == weeks - 1
        for state in layers[layer_index]:
            actions = (
                []
                if terminal
                else list(_feasible_next(state["last"], state["switched"]))
            )
            edges = []
            for action in actions:
                child = transitions[layer_index][(state["state_id"], action)]
                edges.append(
                    {
                        "action": action,
                        "represented_prefix_count": state["count"],
                        "incremental_label_sha256": "a" * 64,
                        "child_obligation_id": (
                            f"node:w{layer_index + 2}:n{child}"
                        ),
                        "child_complete": True,
                    }
                )
                lag_records.append(
                    {
                        "obligation_id": (
                            f"request-lag:w{layer_index + 2}:p"
                            f"{state['state_id']}:a{action}"
                        ),
                        "parent_week": layer_index + 1,
                        "parent_state_id": state["state_id"],
                        "action": action,
                        "state_bytes_equal": True,
                        "checkpoint_equal": True,
                        "callback_inventory_equal": True,
                        "markov_binding_equal": True,
                        "status": "COMPLETE",
                    }
                )
            nodes.append(
                {
                    "obligation_id": (
                        f"node:w{layer_index + 1}:n{state['state_id']}"
                    ),
                    "week": layer_index + 1,
                    "state_id": state["state_id"],
                    "state_sha256": f"{layer_index + 1:064x}"[-64:],
                    "state_sha512": f"{layer_index + 1:0128x}"[-128:],
                    "state_bytes": 1,
                    "last_action": state["last"],
                    "switched_previous": state["switched"],
                    "represented_prefix_count": state["count"],
                    "expected_actions": actions,
                    "edges": edges,
                    "status": "COMPLETE",
                }
            )

    node_by_week_state = {
        (row["week"], row["state_id"]): row for row in nodes
    }
    roots = []
    for collision_id, item in enumerate(discarded):
        week = item["week"]
        child = node_by_week_state[(week, item["target"])]
        actions = child["expected_actions"]
        edges = []
        for action in actions:
            target = transitions[week - 1][(child["state_id"], action)]
            edges.append(
                {
                    "action": action,
                    "state_bytes_equal": True,
                    "incremental_labels_bitwise_equal": True,
                    "callback_inventory_equal": True,
                    "markov_binding_equal": True,
                    "child_obligation_id": f"node:w{week + 1}:n{target}",
                    "child_obligation_complete": True,
                    "status": "COMPLETE",
                }
            )
        roots.append(
            {
                "collision_id": collision_id,
                "root_id": f"collision:{collision_id:08d}",
                "discarded_transition_id": (
                    f"discarded:w{week}:p{item['parent']['state_id']}"
                    f":a{item['action']}"
                ),
                "discarded_parent_state_id": item["parent"]["state_id"],
                "discarded_action": item["action"],
                "represented_prefix_count": item["parent"]["count"],
                "week": week,
                "representative_state_id": child["state_id"],
                "last_action": child["last_action"],
                "switched_previous": child["switched_previous"],
                "canonical_bytes_equal": True,
                "callback_inventory_equal": True,
                "markov_binding_equal": True,
                "expected_actions": actions,
                "edges": edges,
                "status": "COMPLETE",
            }
        )

    multiplicity_records = []
    for layer_index, layer in enumerate(layers):
        terminal = layer_index == weeks - 1
        outgoing = (
            0
            if terminal
            else sum(
                len(_feasible_next(row["last"], row["switched"]))
                for row in layer
            )
        )
        kept = 0 if terminal else len(layers[layer_index + 1])
        represented_successors = (
            0
            if terminal
            else sum(
                row["count"]
                * len(_feasible_next(row["last"], row["switched"]))
                for row in layer
            )
        )
        multiplicity_records.append(
            {
                "week": layer_index + 1,
                "node_count": len(layer),
                "represented_prefix_count": sum(row["count"] for row in layer),
                "closed_form_prefix_count": feasible_calendar_count(layer_index + 1),
                "outgoing_quotient_transition_count": outgoing,
                "represented_successor_count": represented_successors,
                "kept_successor_state_count": kept,
                "discarded_transition_count": 0 if terminal else outgoing - kept,
                "status": "COMPLETE",
            }
        )
    completeness = markov_completeness_certificate()
    discarded_ids = [row["discarded_transition_id"] for row in roots]
    body = {
        "schema_version": "paper2_collision_bisimulation_v2",
        "key_schema_version": KEY_SCHEMA_VERSION,
        "complete_state_serialization": True,
        "event_payload_serialized": True,
        "resource_users_serialized": True,
        "callback_closure_state_serialized": True,
        "process_target_state_serialized_or_fail_closed": True,
        "runtime_alias_graph_serialized": True,
        "markov_completeness_certificate": completeness,
        "markov_completeness_certificate_sha256": completeness["certificate_sha256"],
        "markov_completeness_validated": True,
        "per_key_runtime_schema_enforced": True,
        "runtime_schema_sha256": "c" * 64,
        "tape_binding_sha256": "d" * 64,
        "deterministic_transition_semantics_bound": True,
        "control_state_bound_into_obligations": True,
        "exact_feasible_action_sets_validated": True,
        "quotient_multiplicity_validated": True,
        "request_lag_equivalence_validated": True,
        "request_lag_equivalence_check_count": len(lag_records),
        "request_lag_equivalence_records": lag_records,
        "request_lag_equivalence_records_sha256": _digest(lag_records),
        "layer_multiplicity_records": multiplicity_records,
        "layer_multiplicity_records_sha256": _digest(multiplicity_records),
        "terminal_represented_prefix_count": feasible_calendar_count(weeks),
        "terminal_closed_form_prefix_count": feasible_calendar_count(weeks),
        "w24_terminal_prefix_target": 11_184_811,
        "discarded_transition_count": len(discarded),
        "discarded_transition_witness_count": len(roots),
        "discarded_transition_witness_bijection": True,
        "discarded_transition_ids_sha256": _digest(discarded_ids),
        "collision_payload_checks": len(roots),
        "collision_root_count": len(roots),
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
    body["transition_record_sha256"] = _digest(
        {
            "nodes": body["node_obligation_records_sha256"],
            "roots": body["collision_root_records_sha256"],
        }
    )
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
    for role, suite in REDUCED_CERTIFICATION_SUITES.items():
        weeks = int(suite["weeks"])
        for seed, context, expected_sha256 in suite["tapes"]:
            tape = materialize_tape(
                seed,
                context,
                suite["split"],
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

    assert _runtime_alias_token(fake_sim(True)) != _runtime_alias_token(fake_sim(False))


def test_parent_awaiting_child_process_fails_closed_until_recursive_graph_support():
    env = simpy.Environment()

    def child():
        yield env.timeout(1)

    def parent():
        yield env.process(child())

    env.process(parent())
    env.step()  # start parent and schedule child
    env.step()  # start child and schedule its Timeout
    timeout = next(
        event for *_prefix, event in env._queue if isinstance(event, simpy.Timeout)
    )
    with pytest.raises(TypeError, match="nested/awaited Process callbacks"):
        _queued_event_token(timeout, owner_registry={})


def _post_week_state():
    tape = materialize_tape(
        1_110_001,
        CONTEXTS[0],
        "markov_completeness_pytest_burned",
        weeks=4,
    )
    sim, controller, start = make_sim(tape)
    controller.activate_week(0)
    controller.request(ACTIONS[0])
    advance_including(sim, start + 168.0)
    return sim, controller


def test_markov_completeness_certificate_is_current_source_runtime_bound():
    certificate = markov_completeness_certificate()
    assert validate_markov_completeness_certificate(certificate) == []
    assert certificate["passed"] is True
    assert (
        certificate["reachable_read_inventory"][
            "observation_excluded_from_open_loop_runner"
        ]
        is True
    )
    assert certificate["primary_output_projection"]["access_inventory_matches"] is True
    assert certificate["primary_output_projection"]["reflective_access_absent"] is True
    callable_inventory = certificate["loaded_proof_callable_bindings"]
    assert {
        "run_prefix",
        "semantic_markov_payload",
        "semantic_markov_fingerprint",
        "build_transducer",
    }.issubset(callable_inventory["root_names"])
    reachable = {
        qualname: (source_sha256, code_binding_sha256)
        for _module, qualname, source_sha256, code_binding_sha256 in callable_inventory[
            "reachable_callable_bindings"
        ]
    }
    assert {
        "run_prefix",
        "semantic_markov_payload",
        "semantic_markov_fingerprint",
        "build_transducer",
        "audit_collision_bisimulation",
        "validate_collision_bisimulation_certificate",
        "checkpoint",
        "_endpoint_panel",
        "make_sim",
        "advance_including",
    }.issubset(reachable)
    assert all(
        source_sha256 == "SOURCE_UNAVAILABLE" or len(source_sha256) == 64
        for source_sha256, _code_binding_sha256 in reachable.values()
    )
    assert all(
        len(code_binding_sha256) == 64
        for _source_sha256, code_binding_sha256 in reachable.values()
    )
    assert len(callable_inventory["binding_sha256"]) == 64
    assert callable_inventory["unsupported"] == []

    tampered = copy.deepcopy(certificate)
    tampered["determinism"]["all_rng_states_serialized"] = False
    body = dict(tampered)
    body.pop("certificate_sha256")
    tampered["certificate_sha256"] = _digest(body)
    failures = validate_markov_completeness_certificate(tampered)
    assert any("current source/runtime theorem" in failure for failure in failures)


@pytest.mark.parametrize(
    ("target", "message"),
    (
        ("controller", "controller live fields"),
        ("environment", "Environment live fields"),
        ("resource", "Resource op10_convoy live fields"),
        ("container", "Container rations_sb live fields"),
    ),
)
def test_every_key_fails_closed_on_late_runtime_schema_drift(target, message):
    sim, controller = _post_week_state()
    owner = {
        "controller": controller,
        "environment": sim.env,
        "resource": sim.op10_convoy,
        "container": sim.rations_sb,
    }[target]
    owner.researcher_hidden_state = 7
    with pytest.raises(TypeError, match=message):
        semantic_markov_fingerprint(sim, controller)


def test_controller_tape_root_and_domain_order_aliases_fail_closed():
    sim, controller = _post_week_state()
    controller.profile["researcher_hidden_profile"] = 1
    with pytest.raises(TypeError, match="profile does not match"):
        semantic_markov_fingerprint(sim, controller)

    sim, controller = _post_week_state()
    assert sim.orders
    cloned = copy.copy(sim.orders[-1])
    sim.pending_backorders.append(cloned)
    with pytest.raises(TypeError, match="not the authoritative"):
        semantic_markov_fingerprint(sim, controller)

    sim, controller = _post_week_state()
    controller.sim = object()
    with pytest.raises(TypeError, match="controller.sim is not"):
        semantic_markov_fingerprint(sim, controller)


def test_immutable_sim_contract_state_is_bound_into_every_key():
    sim, controller = _post_week_state()
    before_payload = semantic_markov_payload(sim, controller)
    before_fingerprint = semantic_markov_fingerprint(sim, controller)
    sim.horizon = float(sim.horizon) + 1.0
    after_payload = semantic_markov_payload(sim, controller)
    after_fingerprint = semantic_markov_fingerprint(sim, controller)

    assert (
        before_payload["sim_immutable_contract"]
        != after_payload["sim_immutable_contract"]
    )
    assert (
        before_payload["markov_completeness"]["tape_binding"]["binding_sha256"]
        != after_payload["markov_completeness"]["tape_binding"]["binding_sha256"]
    )
    assert before_fingerprint[3] != after_fingerprint[3]


def test_every_key_fails_closed_on_runtime_global_rebinding(monkeypatch):
    markov_completeness_certificate()
    sim, controller = _post_week_state()
    monkeypatch.setattr(
        paper2_bottleneck_module,
        "ACTIONS",
        tuple(reversed(paper2_bottleneck_module.ACTIONS)),
    )
    with pytest.raises(TypeError, match="runtime global bindings drifted"):
        semantic_markov_fingerprint(sim, controller)


def test_every_key_fails_closed_on_runtime_method_rebinding(monkeypatch):
    markov_completeness_certificate()
    sim, controller = _post_week_state()

    def replacement_activate_week(self, week):
        self.current_week = int(week)

    monkeypatch.setattr(
        paper2_bottleneck_module.BottleneckController,
        "activate_week",
        replacement_activate_week,
    )
    with pytest.raises(TypeError, match="binding"):
        semantic_markov_fingerprint(sim, controller)


def test_build_fails_before_shared_run_prefix_poison_can_fool_both_paths(
    monkeypatch,
):
    tape = materialize_tape(
        1_110_001,
        CONTEXTS[0],
        "run_prefix_poison_pytest_burned",
        weeks=2,
    )
    original = exact_module.run_prefix
    exact_module.markov_completeness_certificate()

    def poisoned_run_prefix(*args, **kwargs):
        return original(*args, **kwargs)

    monkeypatch.setattr(exact_module, "run_prefix", poisoned_run_prefix)
    with pytest.raises(RuntimeError, match="binding"):
        exact_module.build_transducer(tape, 2)


def test_build_fails_before_semantic_payload_can_delete_condition(monkeypatch):
    tape = materialize_tape(
        1_110_001,
        CONTEXTS[0],
        "semantic_payload_poison_pytest_burned",
        weeks=2,
    )
    original = exact_module.semantic_markov_payload
    exact_module.markov_completeness_certificate()

    def payload_without_condition(*args, **kwargs):
        payload = original(*args, **kwargs)
        del payload["controller"]["condition"]
        return payload

    monkeypatch.setattr(
        exact_module,
        "semantic_markov_payload",
        payload_without_condition,
    )
    with pytest.raises(RuntimeError, match="binding"):
        exact_module.build_transducer(tape, 2)


def test_build_fails_when_simulator_init_defaults_change_after_freeze(monkeypatch):
    tape = materialize_tape(
        1_110_001,
        CONTEXTS[0],
        "method_defaults_poison_pytest_burned",
        weeks=2,
    )
    function = exact_module.MFSCSimulation.__init__
    original = function.__defaults__
    assert original is not None and original[0] == 1
    exact_module.markov_completeness_certificate()

    monkeypatch.setattr(function, "__defaults__", (3, *original[1:]))
    with pytest.raises(RuntimeError, match="binding"):
        exact_module.build_transducer(tape, 2)


def test_build_fails_when_statically_loaded_numpy_mean_is_rebound(monkeypatch):
    tape = materialize_tape(
        1_110_001,
        CONTEXTS[0],
        "numpy_mean_poison_pytest_burned",
        weeks=2,
    )
    exact_module.markov_completeness_certificate()

    monkeypatch.setattr(exact_module.np, "mean", lambda _values: 0.123456789)
    with pytest.raises(RuntimeError, match="binding"):
        exact_module.build_transducer(tape, 2)


def _w2_builtin_attack_tape(split: str) -> dict:
    return materialize_tape(
        1_110_001,
        CONTEXTS[0],
        split,
        weeks=2,
    )


def test_runtime_inventory_binds_getattr_and_zip_builtins():
    names = {row[3] for row in exact_module._PROOF_BUILTIN_SPECS}
    assert {"getattr", "zip"} <= names
    assert exact_module._PROOF_BUILTIN_BASELINE["unsupported"] == ()
    assert (
        exact_module._PROOF_BUILTIN_BASELINE[
            "all_identities_match_import_freeze"
        ]
        is True
    )


def test_proof_roots_exclude_cli_identity_but_retain_reachable_transition_helpers(
    monkeypatch,
):
    assert "main" not in exact_module._PROOF_CALLABLE_ROOT_NAMES
    assert not any(
        name.startswith("_execution_")
        for name in exact_module._PROOF_CALLABLE_ROOT_NAMES
    )
    assert any(
        function is exact_module._treatment_orders
        for function, _loaded_names in exact_module._PROOF_FAST_CALLABLE_SPECS
    )
    tape = _w2_builtin_attack_tape("reachable_helper_poison_pytest_burned")
    original = exact_module._treatment_orders

    def poisoned_treatment_orders(*args, **kwargs):
        return original(*args, **kwargs)

    monkeypatch.setattr(exact_module, "_treatment_orders", poisoned_treatment_orders)
    with pytest.raises(RuntimeError, match="binding"):
        build_transducer(tape, 2)


def test_build_fails_when_caller_selective_getattr_fabricates_ret_fields(
    monkeypatch,
):
    tape = _w2_builtin_attack_tape("builtin_getattr_poison_pytest_burned")
    original_getattr = builtins.getattr

    def poisoned_getattr(obj, name, *default):
        caller = sys._getframe(1).f_code.co_name
        if caller == "order_has_ret_risk_indicator" and name == "ret_risk_indicators":
            return {"fabricated": 1.0}
        if caller == "compute_ret_per_order_excel_formula" and name == "OATj":
            return None
        return original_getattr(obj, name, *default)

    monkeypatch.setattr(builtins, "getattr", poisoned_getattr)
    with pytest.raises(RuntimeError, match="builtin bindings drifted"):
        build_transducer(tape, 2)


def test_build_fails_when_caller_selective_zip_hides_active_m_ledger(monkeypatch):
    tape = _w2_builtin_attack_tape("builtin_zip_poison_pytest_burned")
    original_zip = builtins.zip

    def poisoned_zip(*args, **kwargs):
        if sys._getframe(1).f_code.co_name == "activate_week":
            return iter(())
        return original_zip(*args, **kwargs)

    monkeypatch.setattr(builtins, "zip", poisoned_zip)
    with pytest.raises(RuntimeError, match="builtin bindings drifted"):
        build_transducer(tape, 2)


def test_build_fails_when_plain_method_is_rewrapped_as_staticmethod(monkeypatch):
    tape = _w2_builtin_attack_tape("descriptor_staticmethod_poison_pytest_burned")
    original = vars(paper2_bottleneck_module.BottleneckController)["activate_week"]
    monkeypatch.setattr(
        paper2_bottleneck_module.BottleneckController,
        "activate_week",
        staticmethod(original),
    )
    with pytest.raises(RuntimeError, match="method bindings drifted"):
        build_transducer(tape, 2)


def test_build_fails_when_simpy_container_put_init_is_wrapped(monkeypatch):
    tape = _w2_builtin_attack_tape("simpy_container_put_poison_pytest_burned")
    container_put = exact_module.simpy_resources_container.ContainerPut
    original = vars(container_put)["__init__"]

    def halve_amount(self, resource, amount):
        return original(self, resource, amount * 0.5)

    monkeypatch.setattr(container_put, "__init__", halve_amount)
    with pytest.raises(RuntimeError, match="method bindings drifted"):
        build_transducer(tape, 2)


def test_build_fails_when_simpy_boundclass_target_mutates(monkeypatch):
    tape = _w2_builtin_attack_tape("simpy_boundclass_poison_pytest_burned")
    descriptor = vars(exact_module.simpy_core.Environment)["process"]
    monkeypatch.setattr(descriptor, "cls", exact_module.simpy_events.Timeout)
    with pytest.raises(RuntimeError, match="method bindings drifted"):
        build_transducer(tape, 2)


def test_build_fails_when_simpy_container_put_base_and_mro_mutate():
    tape = _w2_builtin_attack_tape("simpy_mro_poison_pytest_burned")
    container_put = exact_module.simpy_resources_container.ContainerPut
    put_base = exact_module.simpy_resources_base.Put

    class EvilPut(put_base):
        def __init__(self, resource, amount):
            super().__init__(resource)
            self.amount = float(amount) * 0.5

    original_bases = container_put.__bases__
    container_put.__bases__ = (EvilPut,)
    try:
        with pytest.raises(RuntimeError, match="bases/MRO bindings drifted"):
            build_transducer(tape, 2)
    finally:
        container_put.__bases__ = original_bases


def test_sequential_run_prefix_calls_do_not_create_natural_binding_drift():
    tape = _w2_builtin_attack_tape("sequential_prefix_regression_pytest_burned")
    first = exact_module.run_prefix(tape, (0,))
    second = exact_module.run_prefix(tape, (0, 1))
    assert first.key
    assert second.key
    exact_module._assert_loaded_proof_bindings(full=True)


def test_class_topology_certificate_is_allowlisted_and_source_attested():
    certificate = markov_completeness_certificate()
    topology = certificate["runtime_class_topology"]
    source_attestation = certificate["source_to_loaded_class_topology_attestation"]
    assert topology["all_bases_and_mro_members_allowlisted"] is True
    assert source_attestation["passed"] is True
    assert certificate["runtime_class_topology_matches_import_freeze"] is True
    assert (
        certificate[
            "source_to_loaded_class_topology_attestation_matches_import_freeze"
        ]
        is True
    )


def test_custom_source_loader_cannot_bless_disk_source_loaded_code_mismatch():
    runner = ROOT / "scripts" / "run_paper2_bottleneck_exact_transducer.py"
    program = f"""
from importlib.abc import SourceLoader
from importlib.util import module_from_spec, spec_from_loader
from pathlib import Path
import sys

path = Path({str(runner)!r})
original = path.read_text()
needle = '\"condition\": _float_token(controller.condition),'
if original.count(needle) != 1:
    raise SystemExit('poison target is not unique')
modified = original.replace(needle, '\"condition\": _float_token(0.0),')

class PoisonLoader(SourceLoader):
    def get_filename(self, fullname):
        return str(path)

    def get_data(self, path_arg):
        return modified.encode()

    def get_source(self, fullname):
        return original

    def get_code(self, fullname):
        return compile(
            modified,
            str(path),
            'exec',
            flags=0,
            dont_inherit=True,
            optimize=int(sys.flags.optimize),
        )

name = 'scripts.run_paper2_bottleneck_exact_transducer_loader_poison'
loader = PoisonLoader()
spec = spec_from_loader(name, loader, origin=str(path))
module = module_from_spec(spec)
module.__file__ = str(path)
sys.modules[name] = module
if loader.get_source(name) != original or path.read_text() != original:
    raise SystemExit('source view was not clean')
try:
    loader.exec_module(module)
except RuntimeError as exc:
    if 'source-to-loaded-code attestation failed' not in str(exc):
        raise
    print('SOURCE_LOADER_ATTACK_REJECTED')
else:
    raise SystemExit('poisoned loaded code escaped source attestation')
"""
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "SOURCE_LOADER_ATTACK_REJECTED" in result.stdout


def test_precertificate_loaded_method_patch_is_not_blessed(monkeypatch):
    original = paper2_bottleneck_module.BottleneckController.activate_week

    def activate_week_reading_projected_history(self, week):
        prior_events = len(self.action_events)
        original(self, week)
        if prior_events < 0:
            raise AssertionError("unreachable")

    with monkeypatch.context() as context:
        context.setattr(
            paper2_bottleneck_module.BottleneckController,
            "activate_week",
            activate_week_reading_projected_history,
        )
        exact_module._markov_completeness_certificate_cached.cache_clear()
        certificate = exact_module._markov_completeness_certificate_cached()
        assert certificate["passed"] is False
        assert certificate["primary_output_projection"][
            "access_inventory_matches"
        ] is False
        with pytest.raises(RuntimeError, match="binding"):
            exact_module.markov_completeness_certificate()
    exact_module._markov_completeness_certificate_cached.cache_clear()
    assert exact_module.markov_completeness_certificate()["passed"] is True


def test_full_completed_order_and_risk_histories_are_conservatively_serialized():
    sim, controller = _post_week_state()
    before = semantic_markov_payload(sim, controller)
    assert "all_order_history" in before
    assert "all_risk_event_history" in before
    assert len(before["all_order_history"]) == len(sim.orders)
    assert len(before["all_risk_event_history"]) == len(sim.risk_events)
    assert sim.orders
    sim.orders[0].quantity = float(sim.orders[0].quantity) + 1.0
    after = semantic_markov_payload(sim, controller)
    assert before["all_order_history"] != after["all_order_history"]


def test_generator_root_binding_uses_identity_not_reserved_local_name():
    class HiddenMutable:
        pass

    hidden = HiddenMutable()

    def suspended(value=hidden):
        self = value
        yield self

    generator = suspended()
    next(generator)
    with pytest.raises(TypeError, match="unknown mutable value"):
        _generator_stack(generator, root_registry={})
    token = _generator_stack(
        generator, root_registry={id(hidden): "explicit.test.root"}
    )
    assert "runtime_root_ref" in repr(token)


def test_collision_certificate_rejects_rehashed_false_completeness_theorem():
    certificate = _zero_collision_certificate(4)
    completeness = certificate["markov_completeness_certificate"]
    completeness["runtime_graph_rules"]["per_key_live_schema_enforced"] = False
    inner = dict(completeness)
    inner.pop("certificate_sha256")
    completeness["certificate_sha256"] = _digest(inner)
    certificate["markov_completeness_certificate_sha256"] = completeness[
        "certificate_sha256"
    ]
    outer = dict(certificate)
    outer.pop("certificate_sha256")
    certificate["certificate_sha256"] = _digest(outer)
    failures = validate_collision_bisimulation_certificate(
        certificate,
        expected_collision_count=len(certificate["collision_roots"]),
        weeks=4,
    )
    assert any("current source/runtime theorem" in failure for failure in failures)


def test_delayed_divergence_requires_complete_child_obligation_chain():
    certificate = _zero_collision_certificate(4)
    node = next(
        row
        for row in certificate["node_obligations"]
        if row["obligation_id"] == "node:w2:n0"
    )
    node["edges"][0]["child_obligation_id"] = "node:w4:n0"
    # Recompute all hashes so this exercises graph semantics, not tamper detection.
    certificate["node_obligation_records_sha256"] = _digest(
        certificate["node_obligations"]
    )
    certificate["transition_record_sha256"] = _digest(
        {
            "nodes": certificate["node_obligation_records_sha256"],
            "roots": certificate["collision_root_records_sha256"],
        }
    )
    body = dict(certificate)
    body.pop("certificate_sha256")
    certificate["certificate_sha256"] = _digest(body)
    failures = validate_collision_bisimulation_certificate(
        certificate,
        expected_collision_count=len(certificate["collision_roots"]),
        weeks=4,
    )
    assert any("dangling" in failure for failure in failures)


def test_rehashed_impossible_action_multiset_is_rejected_independently():
    certificate = _zero_collision_certificate(4)
    node = next(
        row
        for row in certificate["node_obligations"]
        if row["week"] == 2 and row["switched_previous"] is False
    )
    node["expected_actions"] = [0, 0, 2]
    for edge, action in zip(node["edges"], node["expected_actions"]):
        edge["action"] = action
    _rehash_collision_certificate(certificate)
    failures = validate_collision_bisimulation_certificate(
        certificate,
        expected_collision_count=len(certificate["collision_roots"]),
        weeks=4,
    )
    assert any("incomplete node obligation" in failure for failure in failures)


def test_rehashed_multiplicity_tamper_is_rejected():
    certificate = _zero_collision_certificate(4)
    terminal = next(
        row for row in certificate["node_obligations"] if row["week"] == 4
    )
    terminal["represented_prefix_count"] += 1
    _rehash_collision_certificate(certificate)
    failures = validate_collision_bisimulation_certificate(
        certificate,
        expected_collision_count=len(certificate["collision_roots"]),
        weeks=4,
    )
    assert any("multiplicity" in failure for failure in failures)


def test_rehashed_request_lag_tamper_is_rejected():
    certificate = _zero_collision_certificate(4)
    certificate["request_lag_equivalence_records"][0][
        "state_bytes_equal"
    ] = False
    _rehash_collision_certificate(certificate)
    failures = validate_collision_bisimulation_certificate(
        certificate,
        expected_collision_count=len(certificate["collision_roots"]),
        weeks=4,
    )
    assert any("request-lag" in failure for failure in failures)


def test_key_schema_names_required_future_state_families():
    assert KEY_SCHEMA_VERSION.endswith("_v4")
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
    assert (
        callback_audit["prefixes_with_nonempty_callback_inventory"]
        == result["prefix_replays"]
    )
    assert len(callback_audit["layer_prefix_callback_records_sha256"]) == 4


def test_real_collision_certificate_binds_finite_markov_theorem():
    tape = materialize_tape(
        1_110_001,
        CONTEXTS[0],
        "markov_collision_pytest_burned",
        weeks=6,
    )
    transducer = build_transducer(tape, 6)
    assert transducer.collisions
    certificate = transducer.collision_bisimulation
    assert certificate["passed"] is True
    assert certificate["markov_completeness_validated"] is True
    assert certificate["per_key_runtime_schema_enforced"] is True
    assert certificate["deterministic_transition_semantics_bound"] is True
    assert certificate["request_lag_equivalence_validated"] is True
    assert certificate["request_lag_equivalence_check_count"] == sum(
        len(row["expected_actions"])
        for row in certificate["node_obligations"]
    )
    assert certificate["terminal_represented_prefix_count"] == 43
    assert certificate["terminal_closed_form_prefix_count"] == 43
    assert certificate["discarded_transition_count"] == len(
        transducer.collisions
    )
    assert certificate["discarded_transition_witness_bijection"] is True
    assert (
        validate_collision_bisimulation_certificate(
            certificate,
            expected_collision_count=len(transducer.collisions),
            weeks=6,
        )
        == []
    )


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
        collision_certificate = _zero_collision_certificate(int(suite["weeks"]))
        tapes.append(
            {
                "seed": seed,
                "requested_first_context": context,
                "tape_sha256": tape_sha,
                "split": suite["split"],
                "complete_horizon_enumeration": True,
                "primary_transducer_bitwise_certified": True,
                "calendars_compared": count,
                "collision_count": len(collision_certificate["collision_roots"]),
                "prefix_replays": suite["weeks"],
                "collision_bisimulation": collision_certificate,
                "all_prefix_callback_audit": {
                    "passed": True,
                    "unknown_callback_owner_count": 0,
                    "semantic_key_evaluations": suite["weeks"],
                    "layer_semantic_key_evaluations": [1] * suite["weeks"],
                    "layer_callback_inventory": [[{"owner": "MFSCSimulation"}]]
                    * suite["weeks"],
                    "prefix_callback_records_sha256": "a" * 64,
                    "layer_prefix_callback_records_sha256": ["b" * 64] * suite["weeks"],
                    "prefixes_with_nonempty_callback_inventory": suite["weeks"],
                    "layer_prefixes_with_nonempty_callback_inventory": [1]
                    * suite["weeks"],
                },
            }
        )
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "scientific_status": "REDUCED_HORIZON_PRIMARY_CERTIFIED_FULL_CONTRACT_FAIL_CLOSED",
        "key_schema_version": KEY_SCHEMA_VERSION,
        "contract_sha256": __import__("hashlib")
        .sha256(
            (
                ROOT / "contracts" / "paper2_bottleneck_full_horizon_bound_v1.json"
            ).read_bytes()
        )
        .hexdigest(),
        "provenance": provenance,
        "weeks": suite["weeks"],
        "tapes": tapes,
        "summary": {"all_tapes_primary_bitwise_certified": True},
    }


def test_reduced_v3_provenance_roles_callbacks_and_environment_fail_closed():
    payload = _reduced_payload("w16_hard_tape")
    assert validate_reduced_certification_structure(payload, "w16_hard_tape") == []
    assert any(
        "independent custody-bound execution replay" in failure
        for failure in validate_reduced_certification_payload(
            payload, "w16_hard_tape"
        )
    )

    payload["tapes"][0]["requested_first_context"] = "mission_surge"
    assert any(
        "seed/context/tape" in failure
        for failure in validate_reduced_certification_structure(
            payload, "w16_hard_tape"
        )
    )
    payload = _reduced_payload("w16_hard_tape")
    payload["tapes"][0]["all_prefix_callback_audit"]["layer_semantic_key_evaluations"][
        -1
    ] = 0
    assert any(
        "callback" in failure
        for failure in validate_reduced_certification_structure(
            payload, "w16_hard_tape"
        )
    )
    payload = _reduced_payload("w16_hard_tape")
    payload["provenance"]["environment"]["packages"]["numpy"] = "tampered"
    assert any(
        "provenance" in failure or "environment" in failure
        for failure in validate_reduced_certification_structure(
            payload, "w16_hard_tape"
        )
    )


def test_reduced_structure_accepts_caller_bound_cross_host_environment():
    payload = _reduced_payload("w16_hard_tape")
    environment = copy.deepcopy(payload["provenance"]["environment"])
    environment.pop("environment_sha256")
    environment["python_soabi"] = "cpython-311-x86_64-linux-gnu"
    environment["environment_sha256"] = _digest(environment)
    payload["provenance"]["environment"] = environment
    payload["provenance"].pop("provenance_sha256")
    payload["provenance"]["provenance_sha256"] = _digest(payload["provenance"])

    assert validate_reduced_certification_structure(
        payload,
        "w16_hard_tape",
        expected_environment_sha256=environment["environment_sha256"],
    ) == []
    assert any(
        "environment digest mismatch" in failure
        for failure in validate_reduced_certification_structure(
            payload, "w16_hard_tape"
        )
    )


def test_reduced_suite_split_is_exact_and_wrong_split_fails_closed(tmp_path):
    from scripts.paper2_bound_execution_harness import scientific_child_environment

    suite = REDUCED_CERTIFICATION_SUITES["w16_hard_tape"]
    seeds = tuple((seed, context) for seed, context, _sha in suite["tapes"])
    wrong_signer = exact_module.EphemeralOpenSSLEd25519Signer()
    try:
        with pytest.raises(ValueError, match="split does not match"):
            create_reduced_execution_launch_authorization(
                tmp_path / "wrong.authorization.json",
                tmp_path / "wrong.output.json",
                tmp_path / "wrong.receipt.json",
                role="w16_hard_tape",
                execution_role="producer",
                replay_pair_id="b" * 64,
                weeks=suite["weeks"],
                seeds=seeds,
                split="arbitrary_unregistered_split",
                receipt_signer=wrong_signer,
                launch_mode="isolated_bootstrap",
                isolated_bootstrap_path=(
                    ROOT / "scripts" / "paper2_isolated_bootstrap.py"
                ),
                runtime_attestation_path=tmp_path / "wrong.runtime.json",
                host_runtime_sha256="1" * 64,
                portable_runtime_sha256="2" * 64,
                scientific_child_environment=scientific_child_environment(),
                harness_execution_nonce="3" * 64,
            )
    finally:
        wrong_signer.close()

    payload = _reduced_payload("w16_hard_tape")
    payload["tapes"][0]["split"] = "arbitrary_unregistered_split"
    assert any(
        "split mismatch" in failure
        for failure in validate_reduced_certification_structure(
            payload, "w16_hard_tape"
        )
    )

    signer = exact_module.EphemeralOpenSSLEd25519Signer()
    authorization_info = create_reduced_execution_launch_authorization(
        tmp_path / "correct.authorization.json",
        tmp_path / "correct.output.json",
        tmp_path / "correct.receipt.json",
        role="w16_hard_tape",
        execution_role="producer",
        replay_pair_id="b" * 64,
        weeks=suite["weeks"],
        seeds=seeds,
        split=suite["split"],
        receipt_signer=signer,
        launch_mode="isolated_bootstrap",
        isolated_bootstrap_path=ROOT / "scripts" / "paper2_isolated_bootstrap.py",
        runtime_attestation_path=tmp_path / "correct.runtime.json",
        host_runtime_sha256="1" * 64,
        portable_runtime_sha256="2" * 64,
        scientific_child_environment=scientific_child_environment(),
        harness_execution_nonce="3" * 64,
    )
    signer.close()
    authorization = json.loads(
        Path(authorization_info["authorization_path"]).read_text()
    )
    authorization["seed_identity"][0]["split"] = "arbitrary_unregistered_split"
    authorization.pop("authorization_body_sha256")
    authorization["authorization_body_sha256"] = _digest(authorization)
    assert any(
        "scope mismatch" in failure
        for failure in exact_module._execution_authorization_failures(
            authorization, "c" * 64
        )
    )
    authorization["schema_version"] = (
        "paper2_reduced_execution_launch_authorization_v1"
    )
    authorization.pop("authorization_body_sha256")
    authorization["authorization_body_sha256"] = _digest(authorization)
    assert any(
        "legacy unsigned-custody authorization v1 is rejected" in failure
        for failure in exact_module._execution_authorization_failures(
            authorization, "c" * 64
        )
    )


def test_scientific_child_environment_allowlist_rejects_injection_variables():
    from scripts.paper2_bound_execution_harness import scientific_child_environment

    environment = scientific_child_environment()
    assert exact_module._scientific_child_environment_failures(environment) == []
    for injected_name in ("PYTHONPATH", "LD_PRELOAD", "DYLD_INSERT_LIBRARIES"):
        injected = dict(environment)
        injected[injected_name] = "/tmp/attacker"
        assert exact_module._scientific_child_environment_failures(injected)


def _add_fabricated_execution_fields(payload: dict) -> None:
    payload["scientific_run"] = True
    for tape in payload["tapes"]:
        tape.update(
            {
                "mismatch_examples": [],
                "state_counts_by_week": [1] * payload["weeks"],
                "terminal_state_count": 1,
                "collision_examples": [],
                "endpoint_replay_hash": "c" * 64,
                "proof_audit": {"unknown_callback_owner_count": 0},
                "policy_output_count": tape["calendars_compared"],
                "policy_output_records_sha256": "d" * 64,
                "transducer_build_seconds": 1.0,
                "brute_replay_seconds": 1.0,
            }
        )


@pytest.mark.parametrize("timing_only_difference", [False, True])
def test_copied_or_timing_forged_payloads_cannot_pass_receipt_verification(
    tmp_path, timing_only_difference
):
    producer = _reduced_payload("w16_hard_tape")
    _add_fabricated_execution_fields(producer)
    independent = copy.deepcopy(producer)
    if timing_only_difference:
        independent["tapes"][0]["transducer_build_seconds"] = 2.0
    producer_path = tmp_path / "producer.json"
    independent_path = tmp_path / "independent.json"
    producer_path.write_text(json.dumps(producer, sort_keys=True))
    independent_path.write_text(json.dumps(independent, sort_keys=True))
    producer_receipt = tmp_path / "producer.receipt.json"
    independent_receipt = tmp_path / "independent.receipt.json"
    producer_receipt.write_text(json.dumps({"fabricated": True}))
    independent_receipt.write_text(
        json.dumps({"fabricated": True, "timing": timing_only_difference})
    )

    assert validate_reduced_certification_structure(
        producer, "w16_hard_tape"
    ) == []
    producer_sha = __import__("hashlib").sha256(producer_path.read_bytes()).hexdigest()
    independent_sha = (
        __import__("hashlib").sha256(independent_path.read_bytes()).hexdigest()
    )
    producer_receipt_sha = (
        __import__("hashlib").sha256(producer_receipt.read_bytes()).hexdigest()
    )
    independent_receipt_sha = (
        __import__("hashlib").sha256(independent_receipt.read_bytes()).hexdigest()
    )
    verification = verify_independent_reduced_execution(
        producer_path,
        independent_path,
        "w16_hard_tape",
        expected_producer_sha256=producer_sha,
        expected_independent_sha256=independent_sha,
        expected_producer_authorization_sha256="e" * 64,
        expected_independent_authorization_sha256="f" * 64,
        producer_receipt_path=producer_receipt,
        expected_producer_receipt_sha256=producer_receipt_sha,
        independent_receipt_path=independent_receipt,
        expected_independent_receipt_sha256=independent_receipt_sha,
        expected_producer_public_key_fingerprint="1" * 64,
        expected_independent_public_key_fingerprint="2" * 64,
        producer_runtime_attestation_path=tmp_path / "producer.runtime.json",
        expected_producer_runtime_attestation_sha256="3" * 64,
        independent_runtime_attestation_path=tmp_path / "independent.runtime.json",
        expected_independent_runtime_attestation_sha256="4" * 64,
    )
    assert verification["passed"] is False
    assert verification["exact_execution_witness_match"] is True
    assert any(
        "receipt schema mismatch" in failure
        or "authorization chain read failed" in failure
        for failure in verification["failures"]
    )


def test_two_fresh_isolated_w2_processes_emit_distinct_bound_receipts(tmp_path):
    from scripts.paper2_bound_execution_harness import (
        capture_runtime_attestation,
        scientific_child_environment,
    )

    replay_pair_id = "a" * 64
    seeds = ((1_110_001, CONTEXTS[0]),)
    runtime = capture_runtime_attestation(
        python=sys.executable,
        repo_root=ROOT,
        runner_path=Path(exact_module.__file__),
    )
    launches = []
    authorizations = []
    outputs = []
    receipts = []
    signers = []
    for execution_role in ("producer", "independent_replay"):
        role_root = tmp_path / execution_role
        role_root.mkdir(mode=0o700)
        authorization_path = role_root / "execution.authorization.json"
        output_path = role_root / "output.json"
        receipt_path = role_root / "execution.receipt.json"
        runtime_path = role_root / "runtime.json"
        signer = exact_module.EphemeralOpenSSLEd25519Signer()
        authorization = create_reduced_execution_launch_authorization(
            authorization_path,
            output_path,
            receipt_path,
            role="pytest_w2_fresh_process_not_evidence",
            execution_role=execution_role,
            replay_pair_id=replay_pair_id,
            weeks=2,
            seeds=seeds,
            split="fresh_process_w2_pytest_burned",
            workers=1,
            non_scientific_smoke=True,
            max_calendars=feasible_calendar_count(2),
            receipt_signer=signer,
            launch_mode="isolated_bootstrap",
            custody_root=role_root,
            isolated_bootstrap_path=(
                ROOT / "scripts" / "paper2_isolated_bootstrap.py"
            ),
            runtime_attestation_path=runtime_path,
            host_runtime_sha256=runtime["runtime_sha256"],
            portable_runtime_sha256=runtime["portable_sha256"],
            scientific_child_environment=scientific_child_environment(),
            harness_execution_nonce=("1" if execution_role == "producer" else "2")
            * 64,
            parent_launcher_path=Path(exact_module.__file__),
        )
        launch = launch_reduced_execution_fresh_process(
            authorization_path,
            expected_authorization_sha256=authorization["authorization_sha256"],
            receipt_signer=signer,
            expected_public_key_fingerprint=authorization[
                "prelaunch_signing_public_key_fingerprint"
            ],
            timeout_seconds=60,
        )
        assert launch["passed"] is True, launch
        authorizations.append(authorization)
        launches.append(launch)
        outputs.append(json.loads(output_path.read_text()))
        receipts.append(json.loads(receipt_path.read_text()))
        signers.append(authorization["prelaunch_signing_public_key_fingerprint"])

    assert launches[0]["child_pid"] != launches[1]["child_pid"]
    assert launches[0]["output_sha256"] != launches[1]["output_sha256"]
    assert (
        launches[0]["execution_receipt_sha256"]
        != launches[1]["execution_receipt_sha256"]
    )
    assert [row["execution_role"] for row in receipts] == [
        "producer",
        "independent_replay",
    ]
    assert {row["replay_pair_id"] for row in receipts} == {replay_pair_id}
    assert all(row["fresh_child_process"] is True for row in receipts)
    assert all(row["materialized_argv"][1:4] == ["-I", "-B", "-S"] for row in receipts)
    assert len(set(signers)) == 2
    assert (
        exact_module._reduced_execution_witness(outputs[0])
        == exact_module._reduced_execution_witness(outputs[1])
    )
    for output in outputs:
        tape = output["tapes"][0]
        assert tape["policy_output_count"] == feasible_calendar_count(2)
        assert len(tape["policy_output_records_sha256"]) == 64
        assert len(tape["endpoint_replay_hash"]) == 64

    for authorization, launch in zip(authorizations, launches):
        runtime_path = Path(authorization["execution_identity"]["runtime_attestation_path"])
        _output, _receipt, failures = exact_module._load_bound_execution(
            Path(launch["output_path"]),
            Path(launch["execution_receipt_path"]),
            expected_output_sha256=launch["output_sha256"],
            expected_receipt_sha256=launch["execution_receipt_sha256"],
            expected_authorization_sha256=authorization["authorization_sha256"],
            expected_public_key_fingerprint=authorization[
                "prelaunch_signing_public_key_fingerprint"
            ],
            runtime_attestation_path=runtime_path,
            expected_runtime_attestation_sha256=launch[
                "runtime_attestation_file_sha256"
            ],
            role="pytest_w2_fresh_process_not_evidence",
            label="focused-smoke",
        )
        assert failures == []

    original_authorization_path = Path(authorizations[0]["authorization_path"])
    relocated_authorization_path = tmp_path / "relocated.authorization.json"
    relocated_authorization_path.write_bytes(original_authorization_path.read_bytes())
    forged_receipt = copy.deepcopy(receipts[0])
    forged_receipt["authorization_path"] = str(relocated_authorization_path.resolve())
    forged_receipt_path = tmp_path / "relocated.receipt.json"
    forged_receipt_path.write_text(json.dumps(forged_receipt, sort_keys=True))
    forged_receipt_sha = (
        __import__("hashlib").sha256(forged_receipt_path.read_bytes()).hexdigest()
    )
    _output, _receipt, relocation_failures = exact_module._load_bound_execution(
        Path(launches[0]["output_path"]),
        forged_receipt_path,
        expected_output_sha256=launches[0]["output_sha256"],
        expected_receipt_sha256=forged_receipt_sha,
        expected_authorization_sha256=authorizations[0]["authorization_sha256"],
        expected_public_key_fingerprint=authorizations[0][
            "prelaunch_signing_public_key_fingerprint"
        ],
        runtime_attestation_path=Path(
            authorizations[0]["execution_identity"]["runtime_attestation_path"]
        ),
        expected_runtime_attestation_sha256=launches[0][
            "runtime_attestation_file_sha256"
        ],
        role="pytest_w2_fresh_process_not_evidence",
        label="relocated",
    )
    assert any("Ed25519 signature is invalid" in failure for failure in relocation_failures)
    assert any("custody path mismatch" in failure for failure in relocation_failures)


def test_signed_harness_receipt_normalizes_to_exact_chain_w1(tmp_path):
    from scripts.paper2_bound_execution_harness import (
        capture_runtime_attestation,
        execute_reduced_signed_session,
    )

    runtime = capture_runtime_attestation(
        python=sys.executable,
        repo_root=ROOT,
        runner_path=Path(exact_module.__file__),
    )
    custody_root = tmp_path / "signed-harness"
    custody_root.mkdir(mode=0o700)

    def acknowledge(prelaunch):
        return {
            "schema_version": "paper2_reduced_signed_prelaunch_ack_v1",
            "prelaunch_record_sha256": prelaunch["prelaunch_record_sha256"],
            "public_key_fingerprint": prelaunch["public_key_fingerprint"],
            "authorization_sha256": prelaunch["authorization_sha256"],
            "host_runtime_sha256": prelaunch["host_runtime_sha256"],
            "acknowledged_before_child_launch": True,
        }

    result = execute_reduced_signed_session(
        custody_root=custody_root,
        role="pytest_w1_signed_harness_not_evidence",
        execution_role="producer",
        replay_pair_id="7" * 64,
        weeks=1,
        seeds=((1_110_001, CONTEXTS[0]),),
        split="signed_harness_w1_pytest_burned",
        workers=1,
        output_path=custody_root / "output.json",
        authorization_path=custody_root / "authorization.json",
        exact_receipt_path=custody_root / "exact-receipt.json",
        runtime_attestation_path=custody_root / "runtime.json",
        harness_receipt_path=custody_root / "harness-receipt.json",
        host_runtime_sha256=runtime["runtime_sha256"],
        portable_runtime_sha256=runtime["portable_sha256"],
        harness_execution_nonce="8" * 64,
        acknowledgement_callback=acknowledge,
        non_scientific_smoke=True,
        max_calendars=feasible_calendar_count(1),
        timeout_seconds=60,
    )
    prelaunch = result["prelaunch"]
    launch = result["launch"]
    _output, receipt, failures = exact_module._load_bound_execution(
        Path(launch["output_path"]),
        Path(result["harness_receipt_path"]),
        expected_output_sha256=launch["output_sha256"],
        expected_receipt_sha256=result["harness_receipt_sha256"],
        expected_authorization_sha256=prelaunch["authorization_sha256"],
        expected_public_key_fingerprint=prelaunch["public_key_fingerprint"],
        runtime_attestation_path=custody_root / "runtime.json",
        expected_runtime_attestation_sha256=launch[
            "runtime_attestation_file_sha256"
        ],
        role="pytest_w1_signed_harness_not_evidence",
        label="signed-harness",
    )
    assert failures == []
    assert receipt["receipt_signing_public_key_fingerprint"] == prelaunch[
        "public_key_fingerprint"
    ]
