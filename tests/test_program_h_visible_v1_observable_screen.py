import json
import subprocess

import numpy as np

from scripts.screen_program_h_visible_v1_observable_policies import (
    ELIGIBLE_POLICIES,
    INELIGIBLE_DIAGNOSTICS,
    METRICS,
    PLACEBOS,
    REFERENCE,
    SIGNAL_DEPENDENT,
    adapter_liveness_audit,
    bootstrap_indices,
    evaluate,
    evaluate_split,
    make_tape,
    metric_governance_audit,
    paired_ci,
    placebo_tape,
    policy_actions,
    resource_ledger,
    signal_donors,
)


def test_placebos_change_only_signal_and_block_shuffle_is_deranged():
    tapes = [make_tape(index, 1_070_001) for index in range(24)]
    donors = signal_donors(tapes)

    assert all(donor != index for index, donor in enumerate(donors))
    assert all(
        tapes[donor].cell["cell_id"] == tapes[index].cell["cell_id"]
        for index, donor in enumerate(donors)
    )

    current = tapes[0]
    for kind in PLACEBOS:
        changed = placebo_tape(current, kind=kind, donor_tape=tapes[donors[0]])
        assert np.array_equal(changed.demand, current.demand)
        assert np.array_equal(changed.r22, current.r22)
        assert np.array_equal(changed.z, current.z)
        assert changed.seed == current.seed


def test_quarantined_diagnostic_evaluator_and_resource_ledger_are_deterministic():
    current = make_tape(0, 1_070_001)
    row = evaluate(current, REFERENCE)
    resources = resource_ledger(current, REFERENCE)

    assert set(row) == set(METRICS)
    assert 0.0 <= row["ret_visible"] <= 1.0
    assert row["scheduled_dispatches"] == 12.0
    assert row["cargo_departures"] <= row["scheduled_dispatches"]
    assert row["dispatched_rations"] <= 12.0 * 5000.0
    assert resources == {
        key: row[key]
        for key in ("scheduled_dispatches", "cargo_departures", "dispatched_rations")
    }


def test_all_named_policies_and_placebos_emit_four_action_trajectories():
    tapes = [make_tape(index, 1_070_001) for index in range(24)]
    result = evaluate_split(tapes)

    expected = {"ABAB", *ELIGIBLE_POLICIES, *INELIGIBLE_DIAGNOSTICS}
    expected.update(
        f"{base}__placebo_{placebo}"
        for base in SIGNAL_DEPENDENT
        for placebo in PLACEBOS
    )
    assert set(result["actions"]) == expected
    assert set(result["outcomes"]) == expected
    for name in expected:
        assert len(result["actions"][name]) == len(tapes)
        assert all(len(sequence) == 4 for sequence in result["actions"][name])
        assert set(result["outcomes"][name]) == set(METRICS)
        assert all(
            values.shape == (len(tapes),)
            for values in result["outcomes"][name].values()
        )


def test_historical_mpc_is_explicitly_outside_eligible_whitelist():
    current = make_tape(0, 1_070_001)
    actions = policy_actions("historical_mpc_h2_privileged", current)

    assert len(actions) == 4
    assert "historical_mpc_h2_privileged" in INELIGIBLE_DIAGNOSTICS
    assert "historical_mpc_h2_privileged" not in ELIGIBLE_POLICIES


def test_governing_adapter_fails_closed_when_r22_and_arm_are_dead():
    audit = adapter_liveness_audit(make_tape(0, 1_060_001))

    assert audit["r22_bit_identical"] is True
    assert audit["arm_bit_identical"] is True
    assert audit["r22_all_zero"] == audit["r22_all_one"]
    assert audit["arm_T"] == audit["arm_TRS"]
    assert audit["source_mechanism_check"] == {
        "metrics_all_reads_arm": False,
        "metrics_all_reads_tape_r22": False,
        "evidence": "supply_chain/program_g.py metrics_all accepts arm but never references it and its daily delivery loop never references tape.r22.",
    }
    weekly = audit["weekly_reference_physics"]
    assert weekly["service_loss_bit_identical"] is True
    assert weekly["mission_ledger_changes"] is True
    assert weekly["structural_delivery_ceiling"] == {
        "normal_cycles_times_load": 15000,
        "closed_cycles_times_load": 10000,
        "cssu_capacity": 10000,
        "proof": "A route closure reduces three cycles to two, but two 5,000-ration loads already equal the 10,000-ration CSSU capacity. Since free capacity is at most 10,000, the closure cannot reduce weekly delivered quantity in the frozen weekly kernel; it changes only the mission ledger.",
    }


def test_visible_v1_is_quarantined_and_request_snapshot_v2_rescore_is_required():
    governance = metric_governance_audit()

    assert governance["diagnostic_metric"] == "ret_excel_visible_v1"
    assert governance["diagnostic_metric_status"] == "HOLD_FOR_LEDGER_SEMANTICS_REPAIR"
    assert governance["diagnostic_metric_disposition"] == (
        "QUARANTINED_OAT_LEDGER_NOT_SOURCE_VALIDATED"
    )
    assert governance["replacement_development_contract"] == (
        "ret_excel_request_snapshot_v2"
    )
    assert governance["prior_h_results_restored"] is False
    assert "rescore identical burned/calibration tapes" in governance[
        "required_rescore"
    ]
    assert governance["can_support_h_result"] is False


def test_paired_bootstrap_is_deterministic_and_keeps_crn_pairs():
    candidate = np.asarray([1.0, 2.0, 3.0, 4.0])
    baseline = np.asarray([0.0, 1.0, 2.0, 3.0])
    indices = bootstrap_indices(4, 100, 20260714)

    first = paired_ci(candidate, baseline, indices)
    second = paired_ci(candidate, baseline, indices)
    assert first == second
    assert first == {"mean": 1.0, "lcb95": 1.0, "ucb95": 1.0}


def test_cli_fails_closed_before_policy_screen_and_writes_hashed_invalidation(tmp_path):
    output = tmp_path / "observable.json"
    completed = subprocess.run(
        [
            ".venv/bin/python",
            "scripts/screen_program_h_visible_v1_observable_policies.py",
            "--calibration-tapes",
            "12",
            "--locked-tapes",
            "24",
            "--bootstrap-resamples",
            "50",
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    result = json.loads(output.read_text())

    assert completed.returncode == 2
    assert result["governing_metric"] is None
    assert result["diagnostic_metric"] == "ret_excel_visible_v1"
    assert result["scientific_status"] == (
        "INVALID_PROGRAM_H_DEAD_ROUTE_AND_QUARANTINED_VISIBLE_V1"
    )
    assert result["tapes"]["virgin_opened"] is False
    assert result["policies_evaluated"] is False
    assert result["retrospective_screen_pass"] is False
    assert result["valid_for_H_obs"] is False
    assert result["adapter_liveness_audit"]["r22_bit_identical"] is True
    assert result["adapter_liveness_audit"]["arm_bit_identical"] is True
    assert result["metric_governance"]["can_support_h_result"] is False
    assert result["metric_governance"]["replacement_development_contract"] == (
        "ret_excel_request_snapshot_v2"
    )
    assert any(
        "authorized R22" in requirement
        for requirement in result["reopening_requirements"]
    )
    assert any(
        "ret_excel_request_snapshot_v2" in requirement
        for requirement in result["reopening_requirements"]
    )
    assert result["claim_limit"].startswith(
        "The quarantined visible-v1 metric cannot support any Program-H"
    )
    assert len(result["content_sha256"]) == 64
