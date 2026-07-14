from fractions import Fraction
from hashlib import sha256
import copy
import json
from pathlib import Path
import pickle
import subprocess
import tempfile

import pytest

import scripts.run_paper2_bottleneck_full_frontier as frontier

from scripts.run_paper2_bottleneck_exact_transducer import (
    Checkpoint,
    StateNode,
    Transition,
    Transducer,
    _digest,
    build_transducer,
    feasible_calendar_count,
    feasible_calendars,
    markov_completeness_certificate,
    run_prefix,
)
from scripts.run_paper2_bottleneck_full_frontier import (
    BuildSpec,
    PRIMARY_CONTRACT_PATH,
    _build_score_checkpoint,
    _build_fingerprint,
    _compiled_transducer_proof,
    _contract_seed_rows,
    _load_calibration_winner,
    _load_score_checkpoint,
    _paired_h_pi_inference,
    _tracked_head_json,
    assemble_w24_profile_state_audit,
    calendar_at_index,
    calendar_index,
    compile_score_transducer,
    replay_calendar,
    resolve_calibration_tie_break,
    validate_selected_replay_set,
    resolve_frontier,
    screen_frontier,
    validate_acceleration_authorization,
    validate_w24_profile_state_audit_payload,
)
from supply_chain.paper2_bottleneck import CONTEXTS, materialize_tape


ROOT = Path(__file__).resolve().parent.parent


def _checkpoint(values=()):
    values = tuple(map(float, values))
    primary = sum(values) / len(values) if values else 1.0
    return Checkpoint(
        visible_values=values,
        visible_order_ids=tuple(range(len(values))),
        primary_hex=float(primary).hex(),
        endpoint_digest="synthetic",
    )


def _synthetic_transducer(weeks: int, tape_offset: int) -> Transducer:
    """Small Markov machine with binary-exact labels and variable row counts."""
    layers = [[
        StateNode(
            state_id=0,
            key="synthetic-0",
            representative=(0,),
            checkpoint=_checkpoint((Fraction(8 + tape_offset, 32),)),
            last_action=0,
            switched_previous=False,
            represented_prefix_count=1,
        )
    ]]
    transitions = []
    collisions = []
    for week in range(1, weeks):
        prior = layers[-1]
        next_nodes = []
        next_index = {}
        table = {}
        for parent in prior:
            choices = (
                (parent.last_action,)
                if parent.switched_previous
                else (0, 1, 2)
            )
            for action in choices:
                switched = action != parent.last_action
                key = (action, switched)
                if key not in next_index:
                    state_id = len(next_nodes)
                    next_index[key] = state_id
                    next_nodes.append(
                        StateNode(
                            state_id=state_id,
                            key=f"synthetic-{week}-{action}-{int(switched)}",
                            representative=parent.representative + (action,),
                            checkpoint=_checkpoint(),
                            last_action=action,
                            switched_previous=switched,
                            represented_prefix_count=0,
                        )
                    )
                else:
                    collisions.append(
                        {
                            "week": week + 1,
                            "parent_state_id": parent.state_id,
                            "action": action,
                            "target_state_id": next_index[key],
                            "represented_prefix_count": (
                                parent.represented_prefix_count
                            ),
                        }
                    )
                next_nodes[next_index[key]].represented_prefix_count += (
                    parent.represented_prefix_count
                )
                # Some edges append no visible row, so policy denominators vary.
                code = (week * 11 + parent.last_action * 5 + action * 7 + tape_offset) % 29
                values = () if code % 7 == 0 else (Fraction(code + 1, 64),)
                table[(parent.state_id, action)] = Transition(
                    next_state_id=next_index[key],
                    appended_visible_values=tuple(map(float, values)),
                    appended_visible_order_ids=(() if not values else (week,)),
                )
        layers.append(next_nodes)
        transitions.append(table)
    transducer = Transducer(
        weeks=weeks,
        layers=layers,
        transitions=transitions,
        collisions=collisions,
        prefix_replays=0,
    )
    transducer.collision_bisimulation = _synthetic_structural_certificate(
        transducer
    )
    return transducer


def _synthetic_structural_certificate(transducer: Transducer) -> dict:
    nodes = []
    lag_records = []
    for layer_index in reversed(range(transducer.weeks)):
        for node in transducer.layers[layer_index]:
            terminal = layer_index == transducer.weeks - 1
            actions = [] if terminal else sorted(
                action
                for (state_id, action) in transducer.transitions[layer_index]
                if state_id == node.state_id
            )
            edges = [] if terminal else [
                {
                    "action": action,
                    "represented_prefix_count": node.represented_prefix_count,
                    "incremental_label_sha256": "a" * 64,
                    "child_obligation_id": (
                        f"node:w{layer_index + 2}:n"
                        f"{transducer.transitions[layer_index][(node.state_id, action)].next_state_id}"
                    ),
                    "child_complete": True,
                }
                for action in actions
            ]
            for action in actions:
                lag_records.append(
                    {
                        "obligation_id": (
                            f"request-lag:w{layer_index + 2}:p"
                            f"{node.state_id}:a{action}"
                        ),
                        "parent_week": layer_index + 1,
                        "parent_state_id": node.state_id,
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
                    "obligation_id": f"node:w{layer_index + 1}:n{node.state_id}",
                    "week": layer_index + 1,
                    "state_id": node.state_id,
                    "state_sha256": node.key,
                    "state_sha512": "b" * 128,
                    "state_bytes": 1,
                    "last_action": node.last_action,
                    "switched_previous": node.switched_previous,
                    "represented_prefix_count": node.represented_prefix_count,
                    "expected_actions": actions,
                    "edges": edges,
                    "status": "COMPLETE",
                }
            )
    node_map = {
        (row["week"], row["state_id"]): row for row in nodes
    }
    roots = []
    for collision_id, collision in enumerate(transducer.collisions):
        week = collision["week"]
        child = node_map[(week, collision["target_state_id"])]
        actions = child["expected_actions"]
        edges = []
        for action in actions:
            target = transducer.transitions[week - 1][
                (child["state_id"], action)
            ].next_state_id
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
                    f"discarded:w{week}:p{collision['parent_state_id']}"
                    f":a{collision['action']}"
                ),
                "discarded_parent_state_id": collision["parent_state_id"],
                "discarded_action": collision["action"],
                "represented_prefix_count": collision[
                    "represented_prefix_count"
                ],
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
    for layer_index, layer in enumerate(transducer.layers):
        terminal = layer_index == transducer.weeks - 1
        outgoing = (
            0
            if terminal
            else sum(
                len((node.last_action,) if node.switched_previous else (0, 1, 2))
                for node in layer
            )
        )
        kept = 0 if terminal else len(transducer.layers[layer_index + 1])
        represented_successors = (
            0
            if terminal
            else sum(
                node.represented_prefix_count
                * len(
                    (node.last_action,)
                    if node.switched_previous
                    else (0, 1, 2)
                )
                for node in layer
            )
        )
        multiplicity_records.append(
            {
                "week": layer_index + 1,
                "node_count": len(layer),
                "represented_prefix_count": sum(
                    node.represented_prefix_count for node in layer
                ),
                "closed_form_prefix_count": feasible_calendar_count(layer_index + 1),
                "outgoing_quotient_transition_count": outgoing,
                "represented_successor_count": represented_successors,
                "kept_successor_state_count": kept,
                "discarded_transition_count": 0 if terminal else outgoing - kept,
                "status": "COMPLETE",
            }
        )
    markov_completeness = markov_completeness_certificate()
    discarded_ids = [row["discarded_transition_id"] for row in roots]
    body = {
        "schema_version": "paper2_collision_bisimulation_v2",
        "key_schema_version": frontier.KEY_SCHEMA_VERSION,
        "complete_state_serialization": True,
        "event_payload_serialized": True,
        "resource_users_serialized": True,
        "callback_closure_state_serialized": True,
        "process_target_state_serialized_or_fail_closed": True,
        "runtime_alias_graph_serialized": True,
        "markov_completeness_certificate": markov_completeness,
        "markov_completeness_certificate_sha256": markov_completeness[
            "certificate_sha256"
        ],
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
        "terminal_represented_prefix_count": feasible_calendar_count(
            transducer.weeks
        ),
        "terminal_closed_form_prefix_count": feasible_calendar_count(
            transducer.weeks
        ),
        "w24_terminal_prefix_target": 11_184_811,
        "discarded_transition_count": len(roots),
        "discarded_transition_witness_count": len(roots),
        "discarded_transition_witness_bijection": True,
        "discarded_transition_ids_sha256": _digest(discarded_ids),
        "collision_payload_checks": len(roots),
        "collision_root_count": len(roots),
        "transition_congruence_checks": 0,
        "node_obligation_count": len(nodes),
        "terminal_node_obligation_count": len(transducer.layers[-1]),
        "unresolved_node_obligation_count": 0,
        "unresolved_collision_root_count": 0,
        "all_actions_covered": True,
        "backward_induction_complete": True,
        "node_obligations": nodes,
        "collision_roots": roots,
        "mismatch_examples": [],
        "induction_rule": "synthetic unit-test graph",
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


def _brute_exact(transducers):
    weeks = transducers[0].weeks
    aggregate = {}
    per_tape = [dict() for _ in transducers]
    for index, sequence in enumerate(feasible_calendars(weeks)):
        values = []
        for tape_index, transducer in enumerate(transducers):
            visible, _ids = transducer.predict_visible_ledger(sequence)
            score = (
                Fraction(1)
                if not visible
                else sum(map(Fraction.from_float, visible), Fraction(0)) / len(visible)
            )
            per_tape[tape_index][index] = score
            values.append(score)
        aggregate[index] = sum(values, Fraction(0)) / len(values)
    aggregate_max = max(aggregate.values())
    aggregate_winners = [
        index for index, value in aggregate.items() if value == aggregate_max
    ]
    tape_winners = []
    for rows in per_tape:
        maximum = max(rows.values())
        tape_winners.append(
            (maximum, [index for index, value in rows.items() if value == maximum])
        )
    return aggregate_max, aggregate_winners, tape_winners


def test_w6_and_w12_stable_index_matches_brute_dfs_enumeration():
    for weeks in (6, 12):
        enumerated = list(feasible_calendars(weeks))
        assert len(enumerated) == feasible_calendar_count(weeks)
        for index, sequence in enumerate(enumerated):
            assert calendar_at_index(index, weeks) == sequence
            assert calendar_index(sequence) == index


def test_w12_streaming_frontier_matches_brute_exact_enumeration():
    sources = [_synthetic_transducer(12, offset) for offset in (0, 3, 9)]
    compiled = [
        compile_score_transducer(
            source,
            seed=9_000_000 + index,
            tape_sha256=f"synthetic-{index}",
        )
        for index, source in enumerate(sources)
    ]
    brute_value, brute_winners, brute_tapes = _brute_exact(sources)
    screening = screen_frontier(
        compiled, batch_size=37, max_contenders=10_000
    )
    resolved = resolve_frontier(
        compiled, screening, acceleration_certified=True
    )
    assert screening.pass1_count == 2_731
    assert screening.pass2_count == 2_731
    assert resolved["exact_maximum_certified"] is True
    assert Fraction(
        int(resolved["aggregate"]["score"]["numerator"]),
        int(resolved["aggregate"]["score"]["denominator"]),
    ) == brute_value
    assert resolved["aggregate"]["winner_indices"] == brute_winners
    for actual, (expected_score, expected_winners) in zip(
        resolved["per_tape"], brute_tapes
    ):
        assert Fraction(
            int(actual["oracle_score"]["numerator"]),
            int(actual["oracle_score"]["denominator"]),
        ) == expected_score
        assert actual["winner_indices"] == expected_winners
        assert actual["oracle_tie_count"] == len(expected_winners)
        assert actual["oracle_tie_indices_sha256"] == frontier._json_digest(
            expected_winners
        )
        assert actual["oracle_tie_selection_rule"] == "none_all_exact_ties_retained"
        assert actual["display_representative_index"] == min(expected_winners)
        assert actual["representative_semantics"].startswith("display_only")


def test_phase_specific_screen_does_not_accumulate_or_veto_inactive_objective():
    sources = [_synthetic_transducer(6, offset) for offset in (0, 3, 9)]
    compiled = [
        compile_score_transducer(
            source,
            seed=9_100_000 + index,
            tape_sha256=f"phase-synthetic-{index}",
        )
        for index, source in enumerate(sources)
    ]

    calibration = screen_frontier(
        compiled,
        batch_size=11,
        max_contenders=1,
        objective_scope="aggregate_only",
    )
    assert calibration.aggregate_contenders
    assert calibration.per_tape_contenders == []
    assert calibration.per_tape_best_lower == []
    assert calibration.contender_overflow == {"aggregate": False, "per_tape": []}
    # Even a transported/stale inactive overflow marker has no veto power.
    calibration.contender_overflow["per_tape"] = [compiled[1].seed]
    calibration_result = resolve_frontier(
        compiled,
        calibration,
        acceleration_certified=True,
    )
    assert calibration_result["exact_maximum_certified"] is True
    assert calibration_result["aggregate"] is not None
    assert calibration_result["per_tape"] == []

    locked = screen_frontier(
        compiled,
        batch_size=11,
        max_contenders=3,
        objective_scope="per_tape_only",
    )
    assert locked.aggregate_contenders == []
    assert locked.aggregate_best_lower == float("-inf")
    assert [len(rows) for rows in locked.per_tape_contenders] == [1, 3, 1]
    assert locked.contender_overflow == {"aggregate": False, "per_tape": []}
    locked.contender_overflow["aggregate"] = True
    locked_result = resolve_frontier(
        compiled,
        locked,
        acceleration_certified=True,
    )
    assert locked_result["exact_maximum_certified"] is True
    assert locked_result["aggregate"] is None
    assert len(locked_result["per_tape"]) == len(compiled)


def test_w6_real_canonical_frontier_matches_unaccelerated_brute_and_replay():
    weeks = 6
    tape = materialize_tape(
        1_110_001, CONTEXTS[0], "frontier_pytest_burned", weeks=weeks
    )
    source = build_transducer(tape, weeks)
    compiled = compile_score_transducer(
        source, seed=tape["seed"], tape_sha256=tape["threat_sha256"]
    )
    brute = {}
    for index, sequence in enumerate(feasible_calendars(weeks)):
        checkpoint = run_prefix(tape, sequence).checkpoint
        brute[index] = Fraction.from_float(float.fromhex(checkpoint.primary_hex))
    maximum = max(brute.values())
    winners = [index for index, value in brute.items() if value == maximum]

    screening = screen_frontier(
        [compiled], batch_size=11, max_contenders=1_000
    )
    resolved = resolve_frontier(
        [compiled],
        screening,
        score_provider=lambda _tape_index, index: brute[index],
        acceleration_certified=True,
    )
    assert screening.calendar_count == 43
    assert resolved["aggregate"]["winner_indices"] == winners
    assert resolved["per_tape"][0]["winner_indices"] == winners
    replay = replay_calendar(
        tape, calendar_at_index(winners[0], weeks), brute[winners[0]]
    )
    assert replay["active_sequence_matches"] is True
    assert replay["primary_exact_match"] is True
    assert replay["crn_hashes_present"] is True
    assert replay["resource_semantics_match"] is True


def test_selected_replay_audit_requires_same_crn_and_complete_resources():
    base = {
        "seed": 1,
        "tape_sha256": "a" * 64,
        "primary_exact_match": True,
        "active_sequence_matches": True,
        "resource_semantics_match": True,
        "crn_hashes_present": True,
        "guardrails": {
            "consumed_base_threat_sha256": "b" * 64,
            "realized_demand_sha256": "c" * 64,
        },
    }
    rows = [{**base, "role": "fixed"}, {**base, "role": "oracle"}]
    assert validate_selected_replay_set(rows)["passed"] is True
    rows[1] = copy.deepcopy(rows[1])
    rows[1]["guardrails"]["realized_demand_sha256"] = "d" * 64
    audit = validate_selected_replay_set(rows)
    assert audit["passed"] is False
    assert any("CRN" in failure for failure in audit["failures"])


def _tie_guardrails(*, lost_orders, threat_sha, demand_sha):
    return {
        "lost_orders": lost_orders,
        "n_lost": 0,
        "ration_ret_excel": 0.8,
        "ret_excel_cvar05": 0.7,
        "ret_excel_cvar10": 0.75,
        "service_loss_auc_ration_hours": 1.0,
        "backorder_qty_final": 0.0,
        "backlog_age_max": 0.0,
        "reserve_units_issued": 0.0,
        "consumed_base_threat_sha256": threat_sha,
        "realized_demand_sha256": demand_sha,
    }


def _calibration_tie_rows(winners):
    contract = json.loads(PRIMARY_CONTRACT_PATH.read_text())
    rows = []
    for expected in _contract_seed_rows(contract, "calibration"):
        tape_sha = sha256(f"tie-tape-{expected['seed']}".encode()).hexdigest()
        threat_sha = sha256(f"tie-threat-{expected['seed']}".encode()).hexdigest()
        demand_sha = sha256(f"tie-demand-{expected['seed']}".encode()).hexdigest()
        for winner in winners:
            rows.append(
                {
                    "role": "calibration_aggregate_winner",
                    "seed": expected["seed"],
                    "tape_sha256": tape_sha,
                    "calendar_index": winner,
                    "primary_exact_match": True,
                    "active_sequence_matches": True,
                    "resource_semantics_match": True,
                    "crn_hashes_present": True,
                    "guardrails": _tie_guardrails(
                        lost_orders=1 if winner == min(winners) else 0,
                        threat_sha=threat_sha,
                        demand_sha=demand_sha,
                    ),
                }
            )
    return rows


def test_calibration_tie_break_replays_all_primary_ties_and_uses_frozen_order():
    resolved = {"aggregate": {"winner_indices": [5, 7]}}
    rows = _calibration_tie_rows([5, 7])
    audit = resolve_calibration_tie_break(resolved, rows)
    assert audit["passed"] is True
    assert audit["required_replay_count"] == 120
    assert audit["observed_replay_count"] == 120
    assert audit["selected_calendar_index"] == 7
    assert audit["guardrail_constrained_frontier_certified"] is False

    for row in rows:
        row["guardrails"]["lost_orders"] = 0
    calendar_index_only = resolve_calibration_tie_break(resolved, rows)
    assert calendar_index_only["passed"] is True
    assert calendar_index_only["selected_calendar_index"] == 5

    rows[0]["guardrails"]["lost_orders"] = float("nan")
    malformed = resolve_calibration_tie_break(resolved, rows)
    assert malformed["passed"] is False
    assert malformed["selected_calendar_index"] is None
    assert any("nonfinite" in failure for failure in malformed["failures"])


def test_phase_aware_replay_audit_accepts_single_calibration_tie_and_locked_pairs():
    contract = json.loads(PRIMARY_CONTRACT_PATH.read_text())
    calibration = _calibration_tie_rows([5])
    calibration_expected = {
        row["seed"]: {
            "tape_sha256": row["tape_sha256"],
            "consumed_base_threat_sha256": row["guardrails"][
                "consumed_base_threat_sha256"
            ],
            "realized_demand_sha256": row["guardrails"][
                "realized_demand_sha256"
            ],
        }
        for row in calibration
    }
    audit = validate_selected_replay_set(
        calibration,
        phase="calibration",
        aggregate_winner_indices=[5],
        expected_exogenous_by_seed=calibration_expected,
        contract=contract,
    )
    assert audit["passed"] is True
    assert audit["replay_count"] == 60
    assert all(group["replay_count"] == 1 for group in audit["groups"])

    missing = validate_selected_replay_set(
        calibration[:-1],
        phase="calibration",
        aggregate_winner_indices=[5],
        expected_exogenous_by_seed=calibration_expected,
        contract=contract,
    )
    assert missing["passed"] is False
    assert any("coverage" in failure for failure in missing["failures"])

    locked = []
    oracle_by_seed = {}
    locked_expected = {}
    for expected in _contract_seed_rows(contract, "locked_bound"):
        seed = expected["seed"]
        tape_sha = sha256(f"locked-tape-{seed}".encode()).hexdigest()
        threat_sha = sha256(f"locked-threat-{seed}".encode()).hexdigest()
        demand_sha = sha256(f"locked-demand-{seed}".encode()).hexdigest()
        common = {
            "seed": seed,
            "tape_sha256": tape_sha,
            "primary_exact_match": True,
            "active_sequence_matches": True,
            "resource_semantics_match": True,
            "crn_hashes_present": True,
            "guardrails": _tie_guardrails(
                lost_orders=0,
                threat_sha=threat_sha,
                demand_sha=demand_sha,
            ),
        }
        locked.extend(
            [
                {
                    **common,
                    "role": "fixed_calibration_comparator",
                    "calendar_index": 1,
                },
                {
                    **common,
                    "role": "per_tape_oracle_winner",
                    "calendar_index": 5,
                },
            ]
        )
        oracle_by_seed[seed] = [5]
        locked_expected[seed] = {
            "tape_sha256": tape_sha,
            "consumed_base_threat_sha256": threat_sha,
            "realized_demand_sha256": demand_sha,
        }
    locked_audit = validate_selected_replay_set(
        locked,
        phase="locked",
        per_tape_winner_indices=oracle_by_seed,
        expected_exogenous_by_seed=locked_expected,
        contract=contract,
    )
    assert locked_audit["passed"] is True
    assert locked_audit["replay_count"] == 238

    locked[1]["calendar_index"] = 6
    tampered = validate_selected_replay_set(
        locked,
        phase="locked",
        per_tape_winner_indices=oracle_by_seed,
        expected_exogenous_by_seed=locked_expected,
        contract=contract,
    )
    assert tampered["passed"] is False

    wrong_expected = copy.deepcopy(locked_expected)
    wrong_expected[next(iter(wrong_expected))]["realized_demand_sha256"] = "0" * 64
    wrong_crn = validate_selected_replay_set(
        locked,
        phase="locked",
        per_tape_winner_indices=oracle_by_seed,
        expected_exogenous_by_seed=wrong_expected,
        contract=contract,
    )
    assert wrong_crn["passed"] is False


def test_contender_overflow_and_missing_certificate_fail_closed():
    compiled = compile_score_transducer(
        _synthetic_transducer(6, 0), seed=1, tape_sha256="synthetic"
    )
    overflow = screen_frontier(
        [compiled], batch_size=8, max_contenders=0
    )
    result = resolve_frontier(
        [compiled], overflow, acceleration_certified=True
    )
    assert result["exact_maximum_certified"] is False
    assert result["overflow_is_terminal"] is True
    assert result["partial_selection_performed"] is False
    assert result["aggregate"] is None
    assert result["per_tape"] == []

    complete = screen_frontier(
        [compiled], batch_size=8, max_contenders=1_000
    )
    result = resolve_frontier(
        [compiled], complete, acceleration_certified=False
    )
    assert result["exact_maximum_certified"] is False
    assert result["fail_closed_reason"] == "acceleration_not_certified"


def test_checkpoint_resume_invalidates_on_dependency_hash_drift(
    tmp_path, monkeypatch
):
    spec = BuildSpec(
        index=0,
        seed=1_110_001,
        context=CONTEXTS[0],
        split="checkpoint_pytest_burned",
        weeks=4,
    )
    _build_score_checkpoint(spec, tmp_path)
    assert _load_score_checkpoint(spec, tmp_path) is not None

    original = frontier._file_sha256
    changed = ROOT / "supply_chain" / "supply_chain.py"

    def drifted(path):
        if Path(path).resolve() == changed.resolve():
            return "0" * 64
        return original(Path(path))

    monkeypatch.setattr(frontier, "_file_sha256", drifted)
    assert _load_score_checkpoint(spec, tmp_path) is None


def test_w24_audit_rejects_synthetic_shortcut_and_binds_complete_checkpoint_chain(
    tmp_path,
):
    tape = {"seed": 1_110_001, "threat_sha256": "7" * 64}
    proof = {
        "state_inventory": {
            "classification_complete": True,
            "all_frozen_invariants_hold": True,
            "unclassified_live_attributes": [],
            "static_live_reads_unclassified": [],
        },
        "unknown_callback_owner_count": 0,
        "callback_inventory": [{"owner": "MFSCSimulation"}],
    }
    source = _synthetic_transducer(24, 0)
    callback = (("process", "callback", "MFSCSimulation"),)
    source.prefix_replays = 24
    source.callback_inventory = callback
    source.semantic_key_evaluations = 24
    source.layer_callback_inventory = tuple(callback for _ in range(24))
    source.layer_semantic_key_evaluations = tuple(1 for _ in range(24))
    source.prefix_callback_records_sha256 = "a" * 64
    source.layer_prefix_callback_records_sha256 = tuple("b" * 64 for _ in range(24))
    source.prefixes_with_nonempty_callback_inventory = 24
    source.layer_prefixes_with_nonempty_callback_inventory = tuple(1 for _ in range(24))
    compiled = compile_score_transducer(
        source, seed=1_110_001, tape_sha256=tape["threat_sha256"]
    )
    checkpoint_path = tmp_path / "w24_profile.pickle"
    with checkpoint_path.open("wb") as handle:
        pickle.dump(compiled, handle, protocol=pickle.HIGHEST_PROTOCOL)

    shortcut = {
        "data_path": str(checkpoint_path),
        "data_sha256": sha256(checkpoint_path.read_bytes()).hexdigest(),
        "data_bytes": checkpoint_path.stat().st_size,
        "score_table_sha256": compiled.table_sha256,
    }
    assert assemble_w24_profile_state_audit(
        compiled, shortcut, proof, tape
    )["profile_audit_passed"] is False

    spec = BuildSpec(
        index=0,
        seed=1_110_001,
        context=CONTEXTS[0],
        split="w24_profile_state_audit_burned",
        weeks=24,
    )
    fingerprint = _build_fingerprint(spec)
    metadata = {
        **fingerprint,
        "fingerprint_sha256": frontier._json_digest(fingerprint),
        **shortcut,
        "source_transducer_proof": _compiled_transducer_proof(compiled),
    }
    audit = assemble_w24_profile_state_audit(compiled, metadata, proof, tape)
    assert audit["profile_audit_passed"] is True
    assert validate_w24_profile_state_audit_payload(audit) == []
    tampered = copy.deepcopy(audit)
    tampered["checkpoint_summary"]["source_transducer_proof"][
        "semantic_key_evaluations"
    ] += 1
    assert any(
        "proof" in failure or "callback" in failure
        for failure in validate_w24_profile_state_audit_payload(tampered)
    )


def test_tracked_calibration_json_fails_after_worktree_tampering(tmp_path, monkeypatch):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "codex-test@example.invalid"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Codex Test"], cwd=tmp_path, check=True
    )
    artifact = tmp_path / "calibration.json"
    artifact.write_text('{"phase":"calibration"}\n')
    subprocess.run(["git", "add", "calibration.json"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=tmp_path, check=True)
    monkeypatch.setattr(frontier, "ROOT", tmp_path)
    payload, digest, relative = _tracked_head_json(artifact)
    assert payload["phase"] == "calibration"
    assert len(digest) == 64
    assert relative == "calibration.json"

    artifact.write_text('{"phase":"locked"}\n')
    with pytest.raises(ValueError, match="differs from HEAD"):
        _tracked_head_json(artifact)


def test_dependency_identity_survives_vps_to_local_path_relocation():
    remote = [
        {
            "role": "w12_five_tape",
            "path": "/vps/run/source/results/paper2_bottleneck/w12.json",
            "sha256": "a" * 64,
        }
    ]
    local = [
        {
            "role": "w12_five_tape",
            "path": "/Users/test/repo/results/paper2_bottleneck/w12.json",
            "sha256": "a" * 64,
        }
    ]
    assert frontier._dependency_identities(remote) == frontier._dependency_identities(local)


def _synthetic_calibration_chain(monkeypatch):
    contract = json.loads(PRIMARY_CONTRACT_PATH.read_text())
    expected = _contract_seed_rows(contract, "calibration")
    winner_indices = [0, 1]
    selected_index = 1
    tape_rows = [
        {
            **row,
            "tape_sha256": sha256(f"tape-{row['seed']}".encode()).hexdigest(),
        }
        for row in expected
    ]
    checkpoints = [
        {
            "seed": row["seed"],
            "data_sha256": sha256(f"checkpoint-{row['seed']}".encode()).hexdigest(),
            "score_table_sha256": sha256(f"score-{row['seed']}".encode()).hexdigest(),
        }
        for row in expected
    ]
    result_sha = "1" * 64
    auth_sha = "2" * 64
    source_commit = "3" * 40
    selected_replays = []
    for row in tape_rows:
        threat_sha = sha256(f"threat-{row['seed']}".encode()).hexdigest()
        demand_sha = sha256(f"demand-{row['seed']}".encode()).hexdigest()
        for winner_index in winner_indices:
            selected_replays.append(
                {
                    "role": "calibration_aggregate_winner",
                    "seed": row["seed"],
                    "tape_sha256": row["tape_sha256"],
                    "calendar_index": winner_index,
                    "primary_exact_match": True,
                    "active_sequence_matches": True,
                    "resource_semantics_match": True,
                    "crn_hashes_present": True,
                    "guardrails": _tie_guardrails(
                        lost_orders=1 if winner_index == 0 else 0,
                        threat_sha=threat_sha,
                        demand_sha=demand_sha,
                    ),
                }
            )
    resolved_frontier = {
        "objective_scope": "aggregate_only",
        "exact_maximum_certified": True,
        "aggregate": {
            "unique_winner": False,
            "winner_indices": winner_indices,
            "primary_tie_count": len(winner_indices),
            "primary_tie_indices_sha256": frontier._json_digest(winner_indices),
        },
        "per_tape": [],
    }
    tie_audit = resolve_calibration_tie_break(
        resolved_frontier,
        selected_replays,
        contract=contract,
    )
    assert tie_audit["selected_calendar_index"] == selected_index
    result = {
        "scientific_status": "EXACT_PRIMARY_FRONTIER_CERTIFIED_SELECTED_REPLAYS_PASSED",
        "scientific_run": True,
        "primary_contract_id": contract["contract_id"],
        "phase": "calibration",
        "weeks": 24,
        "primary_contract_sha256": frontier._file_sha256(PRIMARY_CONTRACT_PATH),
        "phase_execution_complete": True,
        "exact_maximum_certified": True,
        "frontier_scope": "unconstrained_primary_metric",
        "guardrail_constrained_frontier_certified": False,
        "resolved_frontier": resolved_frontier,
        "calibration_tie_break_audit": tie_audit,
        "selected_calibration_index": selected_index,
        "calendar_index": {
            "calendar_count": contract["physics"]["effective_calendar_count"]
        },
        "screening": {
            "objective_scope": "aggregate_only",
            "pass1_count": contract["physics"]["effective_calendar_count"],
            "pass2_count": contract["physics"]["effective_calendar_count"],
            "passes_identical": True,
            "aggregate_contender_count": 2,
            "per_tape_contender_counts": [],
            "contender_overflow": {"aggregate": False, "per_tape": []},
        },
        "selected_replay_complete": True,
        "transducers": [
            {"seed": row["seed"], "tape_sha256": row["tape_sha256"]}
            for row in tape_rows
        ],
        "selected_replays": selected_replays,
        "runner_manifest": "/transport/control/artifacts/frontier_calibration/runner_manifest.json",
        "acceleration_authorization": {
            "authorized": True,
            "failures": [],
            "sha256": auth_sha,
            "schema_version": frontier.AUTHORIZATION_SCHEMA_VERSION,
            "key_schema_version": frontier.KEY_SCHEMA_VERSION,
            "contract_sha256": frontier._file_sha256(PRIMARY_CONTRACT_PATH),
            "dependencies": [],
        },
        "build": {"checkpoints": checkpoints},
    }
    expected_code_paths = (
        PRIMARY_CONTRACT_PATH,
        Path(frontier.__file__),
        frontier.TRANSDUCER_RUNNER_PATH,
        ROOT / "supply_chain" / "paper2_bottleneck.py",
        ROOT / "supply_chain" / "episode_metrics.py",
    )
    code_sha = {
        str(path.relative_to(ROOT)): frontier._file_sha256(path)
        for path in expected_code_paths
    }
    manifest = {
        "schema_version": frontier.MANIFEST_SCHEMA_VERSION,
        "git_head": source_commit,
        "git_status_sha256": frontier._json_digest(""),
        "command": [
            "python",
            "runner.py",
            "--phase",
            "calibration",
            "--authorization",
            "authorization.json",
            "--weeks",
            "24",
        ],
        "seed_manifest": tape_rows,
        "input_artifacts": {
            "primary_contract": {
                "sha256": frontier._file_sha256(PRIMARY_CONTRACT_PATH)
            },
            "authorization": {
                "path": "/transport/control/authorization.json",
                "sha256": auth_sha,
            },
            "authorization_dependencies": [],
        },
        "result_sha256": result_sha,
        "phase_execution_complete": True,
        "exact_maximum_certified": True,
        "key_schema_version": frontier.KEY_SCHEMA_VERSION,
        "full_execution_was_explicitly_invoked": True,
        "code_sha256": code_sha,
        "checkpoint_artifacts": checkpoints,
    }
    authorization = {"git_commit": source_commit}

    def fake_tracked(_path):
        return result, result_sha, "results/calibration/result.json"

    def fake_resolve(_recorded, _digest, *, result_path, role):
        del result_path
        if role == "runner_manifest":
            return manifest, "4" * 64, "results/calibration/runner_manifest.json"
        return authorization, auth_sha, "results/calibration/authorization.json"

    def fake_run(args, **_kwargs):
        if args[1] == "cat-file":
            return subprocess.CompletedProcess(args, 0, stdout=b"", stderr=b"")
        if args[1] == "show":
            relative = args[2].split(":", 1)[1]
            return subprocess.CompletedProcess(
                args, 0, stdout=(ROOT / relative).read_bytes(), stderr=b""
            )
        raise AssertionError(args)

    monkeypatch.setattr(frontier, "_tracked_head_json", fake_tracked)
    monkeypatch.setattr(frontier, "_resolve_tracked_json", fake_resolve)
    monkeypatch.setattr(frontier.subprocess, "run", fake_run)
    monkeypatch.setattr(
        frontier,
        "validate_acceleration_authorization",
        lambda *_args, **_kwargs: {
            "authorized": True,
            "failures": [],
            "dependencies": [],
            "schema_version": frontier.AUTHORIZATION_SCHEMA_VERSION,
            "key_schema_version": frontier.KEY_SCHEMA_VERSION,
            "contract_sha256": frontier._file_sha256(PRIMARY_CONTRACT_PATH),
        },
    )
    return result


def test_calibration_loader_accepts_complete_chain_and_rejects_tampering(
    tmp_path, monkeypatch
):
    result = _synthetic_calibration_chain(monkeypatch)
    calendar, provenance = _load_calibration_winner(tmp_path / "result.json", 24)
    assert calendar_index(calendar) == 1
    assert provenance["primary_tie_count"] == 2
    assert provenance["source_git_commit"] == "3" * 40
    assert len(provenance["seed_manifest_sha256"]) == 64

    original_tie_audit = copy.deepcopy(result["calibration_tie_break_audit"])
    forged_tie_audit = copy.deepcopy(original_tie_audit)
    forged_tie_audit.pop("audit_sha256")
    forged_tie_audit["selected_calendar_index"] = 0
    forged_tie_audit["audit_sha256"] = frontier._json_digest(forged_tie_audit)
    result["calibration_tie_break_audit"] = forged_tie_audit
    result["selected_calibration_index"] = 0
    with pytest.raises(ValueError, match="tie-break audit does not revalidate"):
        _load_calibration_winner(tmp_path / "result.json", 24)

    result["calibration_tie_break_audit"] = original_tie_audit
    result["selected_calibration_index"] = 1
    result["guardrail_constrained_frontier_certified"] = True
    with pytest.raises(ValueError, match="top_level_not_guardrail_frontier"):
        _load_calibration_winner(tmp_path / "result.json", 24)

    result["guardrail_constrained_frontier_certified"] = False
    result["phase_execution_complete"] = False
    with pytest.raises(ValueError, match="phase_complete"):
        _load_calibration_winner(tmp_path / "result.json", 24)


def _locked_h_pi_rows(delta_values):
    contract = json.loads(PRIMARY_CONTRACT_PATH.read_text())
    expected = _contract_seed_rows(contract, "locked_bound")
    fixed = Fraction(1, 2)
    rows = []
    for row, delta in zip(expected, delta_values):
        delta = Fraction(delta)
        rows.append(
            {
                "seed": row["seed"],
                "tape_sha256": sha256(f"locked-{row['seed']}".encode()).hexdigest(),
                "oracle_score": frontier._fraction_payload(fixed + delta),
                "fixed_calibration_score": frontier._fraction_payload(fixed),
                "h_pi": frontier._fraction_payload(delta),
                "oracle_tie_indices": [0],
                "oracle_tie_count": 1,
                "oracle_tie_indices_sha256": frontier._json_digest([0]),
                "oracle_tie_selection_rule": "none_all_exact_ties_retained",
                "display_representative_index": 0,
                "representative_semantics": (
                    "display_only_not_used_for_h_pi_or_guardrail_feasibility"
                ),
            }
        )
    return rows


def test_paired_h_pi_bootstrap_boundary_material_and_ambiguous_decisions():
    n = json.loads(PRIMARY_CONTRACT_PATH.read_text())["seed_blocks"]["locked_bound"]["n"]
    boundary = _paired_h_pi_inference(_locked_h_pi_rows([Fraction(1, 200)] * n))
    assert boundary["n_tapes"] == 119
    assert boundary["bootstrap_resamples"] == 10_000
    assert boundary["bootstrap_seed"] == 20260713
    assert boundary["mean"] == pytest.approx(0.005)
    assert boundary["lcb95"] == pytest.approx(0.005)
    assert boundary["ucb95"] == pytest.approx(0.005)
    assert boundary["boundary_decision"].startswith("BOUNDARY_CLOSE")

    material = _paired_h_pi_inference(_locked_h_pi_rows([Fraction(1, 50)] * n))
    assert material["boundary_decision"] == "MATERIAL_H_PI_DIAGNOSTIC_ONLY_NO_PROMOTION"

    mixed = [Fraction(0) if index % 2 == 0 else Fraction(1, 50) for index in range(n)]
    ambiguous = _paired_h_pi_inference(_locked_h_pi_rows(mixed))
    assert ambiguous["lcb95"] < 0.01 <= ambiguous["ucb95"]
    assert ambiguous["boundary_decision"] == "AMBIGUOUS_H_PI_FAMILY_REMAINS_ACTIVE"


def test_paired_h_pi_bootstrap_rejects_missing_or_tampered_exact_rows():
    contract = json.loads(PRIMARY_CONTRACT_PATH.read_text())
    n = contract["seed_blocks"]["locked_bound"]["n"]
    rows = _locked_h_pi_rows([Fraction(1, 200)] * n)
    with pytest.raises(ValueError, match="exactly 119"):
        _paired_h_pi_inference(rows[:-1])
    rows[0]["h_pi"] = frontier._fraction_payload(Fraction(1, 10))
    with pytest.raises(ValueError, match="arithmetic mismatch"):
        _paired_h_pi_inference(rows)
