"""Evaluator amendment v1_1 — fail-closed gates (anti-022abd0) + confirmation preconditions."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.evaluate_program_o_ret_learner import (  # noqa: E402
    PLACEBO_FAMILIES,
    compute_provisional_primary_pass,
    derive_placebo_calendars,
    encode_calendar,
    verify_confirmation_preconditions,
)


def _passing_inputs():
    inference = {"estimates": {
        "c1::H_learned": {"lcb95": 0.02}, "c1::guard": {"lcb95": 0.0}}}
    summary = {"c1": {
        "favorable_tapes_vs_open_loop": 40, "favorable_tapes_vs_classical": 40,
        "positive_learner_seeds_vs_both": 9, "max_abs_mass_residual": 0.0,
        "max_abs_partition_residual": 0.0,
        "point_guardrails": {"g": {"m": 0.01}}}}
    audits = {"c1": {"8101": {"passed": True}, "8102": {"passed": True}}}
    placebos = {"c1": {f: {"executed": True, "learner_seeds_beating": 9}
                       for f in PLACEBO_FAMILIES}}
    resources = {"populated": True, "max_abs_diff": 0.0}
    return inference, summary, audits, placebos, resources


def test_all_populated_and_passing_passes():
    inf, s, a, p, r = _passing_inputs()
    ok, gates = compute_provisional_primary_pass(
        inference=inf, summary_rows=s, audit_rows=a, placebo_rows=p, resource_equality=r)
    assert ok and all(gates.values())


@pytest.mark.parametrize("mutation", [
    "placebos_empty", "placebo_family_missing", "placebo_not_executed",
    "placebo_not_beaten", "audit_false", "audits_empty",
    "resources_unpopulated", "resources_nonzero", "inference_empty",
])
def test_fail_closed_on_every_missing_or_failing_component(mutation):
    inf, s, a, p, r = _passing_inputs()
    if mutation == "placebos_empty":
        p = {}
    elif mutation == "placebo_family_missing":
        del p["c1"]["modal"]
    elif mutation == "placebo_not_executed":
        p["c1"]["modal"]["executed"] = False
    elif mutation == "placebo_not_beaten":
        p["c1"]["modal"]["learner_seeds_beating"] = 7
    elif mutation == "audit_false":
        a["c1"]["8102"]["passed"] = False
    elif mutation == "audits_empty":
        a = {}
    elif mutation == "resources_unpopulated":
        r = {"max_abs_diff": 0.0}          # populated flag ABSENT -> must fail, not default-pass
    elif mutation == "resources_nonzero":
        r["max_abs_diff"] = 1e-9
    elif mutation == "inference_empty":
        inf = {"estimates": {}}
    ok, gates = compute_provisional_primary_pass(
        inference=inf, summary_rows=s, audit_rows=a, placebo_rows=p, resource_equality=r)
    assert not ok, f"gate must FAIL CLOSED under {mutation}: {gates}"


def test_encode_calendar_roundtrip():
    assert encode_calendar((0,) * 8) == 0
    assert encode_calendar((3,) * 8) == 4 ** 8 - 1
    assert encode_calendar((0, 0, 0, 0, 0, 0, 0, 1)) == 1
    assert encode_calendar((1, 0, 0, 0, 0, 0, 0, 0)) == 4 ** 7


def test_placebo_calendars_deterministic_and_well_formed():
    cals = [(0, 1, 2, 3, 0, 1, 2, 3)] * 5 + [(1, 1, 2, 3, 0, 1, 2, 3)] * 3
    a = derive_placebo_calendars(cals, rng_seed=42)
    b = derive_placebo_calendars(cals, rng_seed=42)
    assert a == b
    assert a["modal"] == (0, 1, 2, 3, 0, 1, 2, 3)
    assert a["phase_only"] == (0, 1, 2, 3, 0, 1, 2, 3)
    assert len(a["frequency_matched"]) == 8
    assert set(a["frequency_matched"]).issubset({0, 1, 2, 3})


def test_confirmation_refuses_without_artifacts(tmp_path):
    contract = tmp_path / "contract.json"
    contract.write_text("{}")
    with pytest.raises(SystemExit, match="mandatory"):
        verify_confirmation_preconditions(
            calibration_result_path=None, authorization_path=None, contract_path=contract)
    cal = tmp_path / "cal.json"
    cal.write_text(json.dumps({"phase": "calibration", "provisional_primary_pass": False}))
    auth = tmp_path / "auth.json"
    auth.write_text("{}")
    with pytest.raises(SystemExit, match="provisional_primary_pass"):
        verify_confirmation_preconditions(
            calibration_result_path=cal, authorization_path=auth, contract_path=contract)
    cal.write_text(json.dumps({
        "phase": "calibration", "provisional_primary_pass": True,
        "amendment_gates": {"base_lcb": True, "trajectory_feedback": False}}))
    with pytest.raises(SystemExit, match="amendment gates"):
        verify_confirmation_preconditions(
            calibration_result_path=cal, authorization_path=auth, contract_path=contract)
    cal.write_text(json.dumps({
        "phase": "calibration", "provisional_primary_pass": True,
        "amendment_gates": {"base_lcb": True}}))
    with pytest.raises(SystemExit, match="independent auditor"):
        verify_confirmation_preconditions(
            calibration_result_path=cal, authorization_path=auth, contract_path=contract)
