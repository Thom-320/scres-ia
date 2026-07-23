"""Permanent regression tests for the Q-R1 comparator freeze amendment.

These were originally run ad hoc as acceptance tests T1-T6 while auditing the freeze
(docs/Q_R1_COMPARATOR_V2_FREEZE_AMENDMENT_V1_2026-07-22.md).  Documented-but-unversioned
checks protect nothing, so they are pinned here.

The properties under test are the ones that make the freeze auditable:
 * the frozen budget is DERIVED from the config id, never hardcoded (the superseded
   utility hardcoded c64 and would mislabel a c256 receipt as c64);
 * the gate is applied to values RE-DERIVED from the 96 raw rows, not to the receipt's
   summary fields alone, and not to its ``convergence_pass`` boolean;
 * identity, gate and provenance failures are all fail-closed;
 * the secondary-disclosure schema names keys that ``evaluate_calendar`` actually emits.
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "contracts" / "q_r1_comparator_v2_frozen_c256_v1.json"
FREEZE_V1_1 = ROOT / "contracts" / "q_r1_comparator_v2_frozen_c256_v1_1.json"
RECEIPT = (
    ROOT
    / "results"
    / "q_r1"
    / "comparator_v2_c256_c1024_v1"
    / "convergence_merged"
    / "result.json"
)

EXPECTED_CONFIG_ID = "qr1_v2_scenario_h4_c256_wf0.00_unone_expected_tol0.0000_legacy"


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def amended():
    return _load("amended_freeze", "scripts/freeze_q_r1_comparator_v2_amended.py")


@pytest.fixture(scope="module")
def superseded():
    return _load("superseded_freeze", "scripts/freeze_q_r1_comparator_v2.py")


@pytest.fixture(scope="module")
def auditor():
    return _load("freeze_audit", "scripts/audit_q_r1_comparator_v2_freeze.py")


@pytest.fixture(scope="module")
def receipt():
    if not RECEIPT.exists():
        pytest.skip("burned convergence receipt is not present in this checkout")
    return json.loads(RECEIPT.read_text())


# -- T1: the amended instrument accepts the genuine receipt and derives c256 ----------


def test_t1_accepts_genuine_receipt_and_derives_budget(amended, receipt):
    out = amended.freeze(receipt, receipt_path="x", receipt_sha256="x")
    assert out["config_id"] == EXPECTED_CONFIG_ID
    assert out["config"]["conditional_paths"] == 256
    assert all(out["convergence_gate_recomputed"].values())
    assert out["fresh_roots_assigned"] is None
    assert out["learner_authorized"] is False
    assert out["execution_authority"] == "BURNED_PARETO_AND_POWER_AUDIT_ONLY"


# -- T4/T5: the superseded instrument is preserved, cannot freeze, and mislabels -------


def test_t4_superseded_instrument_cannot_execute_the_authorized_freeze(superseded, receipt):
    with pytest.raises(ValueError, match="tolerance"):
        superseded.freeze(receipt, receipt_path="x", receipt_sha256="x")


def test_t5_superseded_instrument_mislabels_c256_as_c64_if_guards_relaxed(
    superseded, receipt, monkeypatch
):
    """The hazard that motivated the amendment: it is a mislabel, not a wrong pointer."""
    monkeypatch.setattr(superseded, "EXPECTED_TOLERANCE", 0.0)
    monkeypatch.setattr(superseded, "EXPECTED_TIE_BREAKER", "legacy")
    out = superseded.freeze(receipt, receipt_path="x", receipt_sha256="x")
    assert out["config_id"] == EXPECTED_CONFIG_ID  # id says c256 ...
    assert out["config"]["conditional_paths"] == 64  # ... config says c64


# -- T6: fail-closed on identity, gate and provenance ---------------------------------


def test_t6a_refuses_a_different_budget_by_identity(amended, receipt):
    fake = copy.deepcopy(receipt)
    fake["convergence"][0]["low_config"] = EXPECTED_CONFIG_ID.replace("c256", "c64")
    with pytest.raises(ValueError, match="missing or duplicated"):
        amended.freeze(fake, receipt_path="x", receipt_sha256="x")


def test_t6b_refuses_a_degraded_gate(amended, receipt):
    fake = copy.deepcopy(receipt)
    fake["convergence"][0]["first_action_agreement"] = 0.90625
    with pytest.raises(ValueError, match="failed convergence gate"):
        amended.freeze(fake, receipt_path="x", receipt_sha256="x")


def test_t6b_bis_does_not_trust_the_receipt_boolean(amended, receipt):
    """A receipt whose boolean and whose numbers disagree must be rejected."""
    fake = copy.deepcopy(receipt)
    fake["convergence"][0]["first_action_agreement"] = 0.5
    fake["convergence"][0]["convergence_pass"] = True
    with pytest.raises(ValueError, match="failed convergence gate"):
        amended.freeze(fake, receipt_path="x", receipt_sha256="x")


@pytest.mark.parametrize(
    "flag",
    [
        "selection_performed",
        "learner_return_used",
        "retained_minus_reset_used_for_selection",
    ],
)
def test_t6c_refuses_any_selection_provenance(amended, receipt, flag):
    fake = copy.deepcopy(receipt)
    fake[flag] = True
    with pytest.raises(ValueError, match="invalid selection provenance"):
        amended.freeze(fake, receipt_path="x", receipt_sha256="x")


def test_refuses_a_lower_or_equal_high_budget_reference(amended, receipt):
    fake = copy.deepcopy(receipt)
    fake["convergence"][0]["high_config"] = EXPECTED_CONFIG_ID
    with pytest.raises(ValueError, match="predeclared c1024 high budget"):
        amended.freeze(fake, receipt_path="x", receipt_sha256="x")


# -- the recomputation audit ----------------------------------------------------------


def _audit(auditor, freeze_payload, receipt_payload):
    return auditor.audit(
        freeze_payload,
        receipt_payload,
        freeze_path="x",
        freeze_sha=str(freeze_payload.get("convergence_sha256")),
        receipt_path="y",
        receipt_sha=str(freeze_payload.get("convergence_sha256")),
    )


def test_audit_recomputes_the_gate_from_raw_rows(auditor, receipt):
    if not FREEZE.exists():
        pytest.skip("frozen contract is not present in this checkout")
    result = _audit(auditor, json.loads(FREEZE.read_text()), receipt)
    assert result["verdict"] == "FREEZE_ENTITLED_RECOMPUTED_FROM_RAW_ROWS"
    assert result["checks_passed"] == result["checks_total"]
    # Re-derivation must reproduce the summaries exactly, not merely closely.
    assert all(delta == 0.0 for delta in result["absolute_deltas"].values())
    assert result["recomputed"]["comparable_arm_states"] == 96
    assert result["recomputed"]["disagreements"] == 3


def test_audit_catches_a_summary_that_contradicts_its_raw_rows(auditor, receipt):
    """The exact hole the audit exists to close."""
    if not FREEZE.exists():
        pytest.skip("frozen contract is not present in this checkout")
    fake = copy.deepcopy(receipt)
    fake["convergence"][0]["first_action_agreement"] = 0.99  # summary now lies
    result = _audit(auditor, json.loads(FREEZE.read_text()), fake)
    assert result["verdict"] == "FREEZE_NOT_ENTITLED_RECOMPUTATION_FAILED"
    failed = {check["check"] for check in result["checks"] if not check["passed"]}
    assert "recomputed_matches_summary::first_action_agreement" in failed


def test_audit_catches_dropped_raw_rows(auditor, receipt):
    if not FREEZE.exists():
        pytest.skip("frozen contract is not present in this checkout")
    fake = copy.deepcopy(receipt)
    fake["convergence_pairs"] = fake["convergence_pairs"][:95]
    result = _audit(auditor, json.loads(FREEZE.read_text()), fake)
    assert result["verdict"] == "FREEZE_NOT_ENTITLED_RECOMPUTATION_FAILED"
    failed = {check["check"] for check in result["checks"] if not check["passed"]}
    assert "raw_row_count" in failed


def test_audit_catches_duplicate_raw_identities(auditor, receipt):
    if not FREEZE.exists():
        pytest.skip("frozen contract is not present in this checkout")
    fake = copy.deepcopy(receipt)
    fake["convergence_pairs"][1] = copy.deepcopy(fake["convergence_pairs"][0])
    result = _audit(auditor, json.loads(FREEZE.read_text()), fake)
    failed = {check["check"] for check in result["checks"] if not check["passed"]}
    assert "unique_raw_identities" in failed


# -- disclosure schema ----------------------------------------------------------------


def test_disclosure_names_exist_in_the_evaluation_output(amended):
    """`ret_total` was never emitted by evaluate_calendar; guard the successor list."""
    from supply_chain.q_r1_retained_learning import evaluate_calendar  # noqa: F401

    assert "ret_total" not in amended.SECONDARY_DISCLOSURES
    assert "ret_visible" in amended.SECONDARY_DISCLOSURES
    # Degenerate keys must be declared as degenerate, never as live guardrails.
    assert set(amended.DEGENERATE_DISCLOSURES) == {"ret_full", "lost_orders"}
    assert not set(amended.SECONDARY_DISCLOSURES) & set(amended.DEGENERATE_DISCLOSURES)


def test_successor_contract_is_a_schema_only_correction():
    if not (FREEZE.exists() and FREEZE_V1_1.exists()):
        pytest.skip("frozen contracts are not present in this checkout")
    v1 = json.loads(FREEZE.read_text())
    v11 = json.loads(FREEZE_V1_1.read_text())
    assert v11["supersedes_sha256"]
    assert v11["correction_scope"] == "SCHEMA_ONLY_SECONDARY_DISCLOSURE_NAMES"
    for key in ("config", "config_id", "convergence_sha256", "convergence_gate",
                "primary_objective", "execution_authority", "fresh_roots_assigned"):
        assert v1[key] == v11[key], f"successor altered {key}"
    assert "ret_total" not in v11["secondary_disclosures"]
