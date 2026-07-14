import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.recompute_program_j_visible_v1_frontier import (
    FIELDS,
    SEQUENCES,
    WEEKS,
)


ROOT = Path(__file__).resolve().parent.parent
AUDIT = (
    ROOT
    / "research/paper2_exhaustive_search/program_j_request_snapshot_v2_frontier_structure_audit_20260714.json"
)


def json_sha256(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_audit():
    return json.loads(AUDIT.read_text())


def test_program_j_structure_audit_is_content_and_source_addressed():
    payload = load_audit()
    expected = payload.pop("content_sha256")
    assert json_sha256(payload) == expected

    for relative, expected_hash in payload["source_bindings"].items():
        assert file_sha256(ROOT / relative) == expected_hash


def test_program_j_structure_audit_binds_complete_frontier_and_metric_scope():
    payload = load_audit()
    contract = payload["evaluated_contract"]

    assert WEEKS == contract["weeks"] == 8
    assert len(SEQUENCES) == contract["complete_open_loop_schedule_count"] == 3**8
    assert payload["metric_contract"]["id"] == "ret_excel_request_snapshot_v2"
    assert payload["metric_contract"]["workbook_replay"]["mismatches"] == 0
    assert payload["subject"]["result_values_required_for_this_audit"] is True
    assert contract["tapes"]["virgin"] is False


def test_program_j_structure_audit_fails_closed_on_resources_and_claims():
    payload = load_audit()
    required = payload["resource_acceptance"]["required_result_conditions"]

    assert required["resource_audit.executed_pm_hours_equal"] is True
    assert required["resource_audit.corrective_hours_constrained_non_superior"] is True
    assert required["resource_audit.maximum_mass_residual_lte"] == 1e-5
    assert set((
        "lost_orders",
        "service_loss_auc",
        "corrective_hours",
        "ret_quantity",
        "ret_cvar05",
    )).issubset(FIELDS)
    assert payload["guardrail_completeness"]["paper2_gate_complete"] is False
    assert payload["strongest_comparator_and_observability"]["h_obs_established"] is False
    assert payload["strongest_comparator_and_observability"]["learner_authorized"] is False
    assert "strongest-comparator H_PI" in payload["claim_boundary"]["invalid"]
    assert "Paper 2 result" in payload["claim_boundary"]["invalid"]


def test_program_j_structure_audit_requires_strong_classical_comparators():
    payload = load_audit()
    missing = " ".join(
        payload["strongest_comparator_and_observability"][
            "mandatory_missing_comparators"
        ]
    ).lower()

    assert "hysteresis" in missing
    assert "whittle" in missing
    assert "mpc" in missing
    assert "dp" in missing
    assert payload["statistical_scope"]["selection_and_evaluation_same_block"] is True
    assert payload["statistical_scope"]["confidence_interval_implemented"] is False


def test_program_j_resource_only_frontier_recalculation_matches_raw_matrix():
    payload = load_audit()
    completed = payload["completed_result_independent_recalculation"]
    expected = completed["resource_only_envelope"]
    raw = np.load(
        ROOT
        / "results/paper2_maintenance/request_snapshot_v2_full_frontier/raw_matrices.npz",
        allow_pickle=False,
    )
    fields = raw["field_names"].tolist()
    arrays = {
        field: raw["matrix"][:, :, index]
        for index, field in enumerate(fields)
    }
    means = {field: values.mean(axis=0) for field, values in arrays.items()}
    budget = expected["B_corrective_hours"]
    feasible = (
        (means["corrective_hours"] <= budget + 1e-12)
        & np.isclose(means["executed_pm_hours"], 192.0)
        & (means["mass_residual"] <= 1e-5)
    )
    best_index = int(
        np.argmax(np.where(feasible, means["ret_visible"], -np.inf))
    )
    tape_oracle_indices = arrays["ret_visible"].argmax(axis=1)
    tape_rows = np.arange(arrays["ret_visible"].shape[0])
    oracle_ret = arrays["ret_visible"][tape_rows, tape_oracle_indices].mean()
    oracle_corrective = arrays["corrective_hours"][
        tape_rows, tape_oracle_indices
    ].mean()

    assert raw["matrix"].shape == tuple(completed["raw_matrix_shape"])
    assert np.isfinite(raw["matrix"]).all()
    assert raw["seeds"].tolist() == completed["seed_sequence_exact"]
    assert int(feasible.sum()) == expected["feasible_deterministic_schedule_count"]
    assert best_index == expected["best_static_first_tie_index"]
    assert means["ret_visible"][best_index] == expected["best_static_ret_visible"]
    assert oracle_ret == expected["deterministic_tape_contingent_pi_ret_visible"]
    assert oracle_corrective <= budget
    assert (
        oracle_ret - means["ret_visible"][best_index]
        == expected["resource_only_pi_delta_ret_visible"]
    )
    assert expected["resource_only_pi_delta_ret_visible"] < 0.01
    assert (
        completed["deterministic_resource_only_pi_minus_best_static_guardrails"][
            "tail_noninferiority_point_pass"
        ]
        is False
    )
