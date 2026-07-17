"""Evaluator amendment v1_1 — fail-closed gates (anti-022abd0) + confirmation preconditions."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.evaluate_program_o_ret_learner import (  # noqa: E402
    DEMAND_LEDGER_IDENTITIES,
    PLACEBO_FAMILIES,
    compute_provisional_primary_pass,
    demand_ledger_residuals,
    derive_placebo_calendars,
    encode_calendar,
    verify_confirmation_preconditions,
)
from scripts.adjudicate_program_o_ret_calibration import adjudicate  # noqa: E402
from supply_chain.program_o_eval_custody import (  # noqa: E402
    sha256,
    verify_sha256_manifest,
    write_sha256_manifest,
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
    resources = {
        "populated": True,
        "full_open_loop_frontier_included": True,
        "max_abs_diff": 0.0,
    }
    demand = {
        "populated": True,
        "identities": {name: 0.0 for name in DEMAND_LEDGER_IDENTITIES},
        "max_abs_residual": 0.0,
    }
    return inference, summary, audits, placebos, resources, demand


def test_all_populated_and_passing_passes():
    inf, s, a, p, r, d = _passing_inputs()
    ok, gates = compute_provisional_primary_pass(
        inference=inf, summary_rows=s, audit_rows=a, placebo_rows=p, resource_equality=r, demand_preservation=d)
    assert ok and all(gates.values())


@pytest.mark.parametrize("mutation", [
    "placebos_empty", "placebo_family_missing", "placebo_not_executed",
    "placebo_not_beaten", "audit_false", "audits_empty",
    "resources_unpopulated", "resources_frontier_missing", "resources_nonzero",
    "inference_empty", "demand_unpopulated", "demand_identity_missing",
    "demand_residual_large",
])
def test_fail_closed_on_every_missing_or_failing_component(mutation):
    inf, s, a, p, r, d = _passing_inputs()
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
    elif mutation == "resources_frontier_missing":
        r["full_open_loop_frontier_included"] = False
    elif mutation == "inference_empty":
        inf = {"estimates": {}}
    elif mutation == "demand_unpopulated":
        d["populated"] = False
    elif mutation == "demand_identity_missing":
        del d["identities"][DEMAND_LEDGER_IDENTITIES[0]]
    elif mutation == "demand_residual_large":
        d["max_abs_residual"] = 1e-3
    ok, gates = compute_provisional_primary_pass(
        inference=inf, summary_rows=s, audit_rows=a, placebo_rows=p, resource_equality=r, demand_preservation=d)
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


def test_confirmation_refusal_ladder(tmp_path):
    import hashlib
    from supply_chain.program_o_eval_custody import sha256, write_sha256_manifest

    def make_chain(base, *, audit_passed=True, status="ELIGIBLE_FOR_INDEPENDENT_AUTHORIZATION"):
        base.mkdir(exist_ok=True)
        contract = base / "contract.json"; contract.write_text("{}")
        cal_dir = base / "calibration"; cal_dir.mkdir()
        cal = cal_dir / "result.json"
        cal.write_text(json.dumps({"phase": "calibration", "provisional_primary_pass": True,
                                   "amendment_gates": {"base_lcb": True}}))
        raw = cal_dir / "raw_calendar_matrix" / "c1" / "tape.npz"
        raw.parent.mkdir(parents=True)
        raw.write_bytes(b"raw")
        raw_manifest = cal_dir / "raw_files.sha256"
        write_sha256_manifest(cal_dir, [raw], raw_manifest)
        write_sha256_manifest(
            cal_dir, [cal, raw_manifest, raw], cal_dir / "evaluation_files.sha256"
        )
        audit_dir = base / "audit"; audit_dir.mkdir()
        audit = audit_dir / "full_des_audit.json"
        audit.write_text(json.dumps({"passed": audit_passed, "phase": "calibration",
                                     "evaluation_result_sha256": sha256(cal)}))
        write_sha256_manifest(audit_dir, [audit], audit_dir / "audit_files.sha256")
        adj = base / "adjudication.json"
        adj.write_text(json.dumps({"status": status,
                                   "calibration_result_sha256": sha256(cal),
                                   "direct_audit_sha256": sha256(audit)}))
        auth = base / "authorization.json"
        auth.write_text(json.dumps({"authorized_by": "independent_auditor",
                                    "contract_sha256": hashlib.sha256(contract.read_bytes()).hexdigest(),
                                    "calibration_result_sha256": sha256(cal),
                                    "direct_audit_sha256": sha256(audit),
                                    "adjudication_sha256": sha256(adj)}))
        return contract, cal, audit, adj, auth

    contract, cal, audit, adj, auth = make_chain(tmp_path / "ok")
    with pytest.raises(SystemExit, match="mandatory"):
        verify_confirmation_preconditions(
            calibration_result_path=None, authorization_path=None, contract_path=contract)
    c2, cal2, audit2, adj2, auth2 = make_chain(tmp_path / "bad_audit", audit_passed=False)
    with pytest.raises(SystemExit, match="did not pass"):
        verify_confirmation_preconditions(
            calibration_result_path=cal2, authorization_path=auth2, contract_path=c2,
            full_des_audit_path=audit2, adjudication_path=adj2)
    c3, cal3, audit3, adj3, auth3 = make_chain(tmp_path / "bad_adj", status="NOT_ELIGIBLE")
    with pytest.raises(SystemExit, match="not eligible"):
        verify_confirmation_preconditions(
            calibration_result_path=cal3, authorization_path=auth3, contract_path=c3,
            full_des_audit_path=audit3, adjudication_path=adj3)
    # complete, hash-bound chain passes without raising
    verify_confirmation_preconditions(
        calibration_result_path=cal, authorization_path=auth, contract_path=contract,
        full_des_audit_path=audit, adjudication_path=adj)
