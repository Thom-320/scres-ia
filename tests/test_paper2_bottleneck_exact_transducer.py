import json
from pathlib import Path

import pytest

from scripts.run_paper2_bottleneck_exact_transducer import (
    CONTAINER_FIELDS,
    KEY_SCHEMA_VERSION,
    RESULT_SCHEMA_VERSION,
    REDUCED_CERTIFICATION_SUITES,
    RNG_FIELDS,
    SIM_STATE_FIELDS,
    _semantic,
    certify_exhaustive,
    certification_provenance,
    feasible_calendar_count,
    feasible_calendars,
    validate_reduced_certification_payload,
)
from supply_chain.paper2_bottleneck import CONTEXTS, materialize_tape


ROOT = Path(__file__).resolve().parent.parent


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


def test_unknown_mutable_object_fails_closed_in_semantic_key():
    class UnknownMutable:
        def __init__(self):
            self.value = 1

    with pytest.raises(TypeError, match="unknown mutable value"):
        _semantic(UnknownMutable())


def test_key_schema_names_required_future_state_families():
    assert KEY_SCHEMA_VERSION.endswith("_v2")
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
            "prefix_replays": suite["weeks"],
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
