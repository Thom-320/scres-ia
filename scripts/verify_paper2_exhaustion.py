#!/usr/bin/env python3
"""Fail-closed verifier for the Paper-2 evidence and approach registries.

This verifier does not turn an unbounded researcher-extension class into a
scientific impossibility result.  It checks that the versioned evidence is
internally consistent, that superseded positive claims are not promoted, and
that unresolved/domain-blocked families remain visibly nonterminal.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
SEARCH = ROOT / "research" / "paper2_exhaustive_search"

SOURCE_HASHES = {
    Path("/Users/thom/Downloads/Raw_data1+Re.xlsx"):
        "30b88c9b9fe68ef527dbfcc70d8e653ea7bd152ab891b3fc0ecf53cb6f043486",
    Path("/Users/thom/Downloads/Raw_data2+Re.xlsx"):
        "4bd462771fefff16fc5666a851256b3780198d474832dec1423c0b6f94be86b0",
    Path("/Users/thom/Downloads/Rsult_1.xlsx"):
        "1901f683f6014cf75237c17233b8eba04f541b956f2d19dcecf2edc00e83b00a",
    Path("/Users/thom/Downloads/garrido et al 2024 factory resilience.pdf"):
        "1260863dc295232faf24b820e1f67d53f25f81ffa2d221f7ef02a02310519c43",
    Path("/Users/thom/Downloads/v.0_neuralNet-scres.docx"):
        "b111070a05c8f4d1afa058454138bed9b4b74900ab87eaaf6eb5186b6e8293f2",
    Path("/Users/thom/Downloads/v.0_neuralNet-scres.pdf"):
        "521b12770e94f3e70c4c88ce1e38613f4e0aad3e1dab114632c9c89dbfad182d",
    Path("/Users/thom/Library/CloudStorage/GoogleDrive-chisicathomas@gmail.com/My Drive/Supernote/Document/20_RESEARCH/PhD-Papers/garrido2024 scres+AI.pdf"):
        "3e3bc8f82e20b891ee163fb8a035dd37be4312fa11f58dde77452dc1bb903ae6",
    Path("/Users/thom/Library/CloudStorage/GoogleDrive-chisicathomas@gmail.com/My Drive/Archive/Misc_Unsorted/Unsorted/WRAP_Theses_Garrido_Rios_2017.pdf"):
        "de9192d233b0c728ece6156b754fc64543146868121358b8a95c73b3edaa55cf",
}

TERMINAL_B_PROOF_CLASSES = {
    "certified_quantitative_global_upper_bound",
    "exact_global_zero",
    "formal_global_dominance",
}

CANONICAL_METRIC_CONTRACT = "ret_excel_request_snapshot_v2"
LEGACY_VISIBLE_METRIC_CONTRACT = "ret_excel_visible_v1"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def close(actual: float, expected: float, tol: float = 1e-12) -> bool:
    return math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tol)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def validate_metric_governance(
    governance: dict[str, Any],
    source_semantics: dict[str, Any],
    implementation_audit: dict[str, Any],
    excel_reaudit: dict[str, Any],
    *,
    root: Path = ROOT,
) -> dict[str, Any]:
    """Validate the v2 lock and fail closed on every visible-v1 result.

    Exact workbook formula replay is necessary but does not authorize a result.
    The current release gate also requires source-aligned request snapshots,
    explicit quarantine of the OAT-derived v1 ledger, and false Paper-2/Paper-3
    authorization until same-time semantics and v2 rescoring are complete.
    """
    failures: list[str] = []
    hashes_checked = 0
    implementation_source_hashes_checked = 0

    if governance.get("schema_version") != "paper2_metric_governance_audit_v2":
        failures.append("invalid metric-governance schema")
    if governance.get("status") != (
        "CANONICAL_RET_EXCEL_REQUEST_SNAPSHOT_V2__PROVISIONAL__RESCORE_REQUIRED"
    ):
        failures.append("metric governance does not retain the provisional v2 hold")

    canonical = governance.get("canonical_endpoint", {})
    if canonical.get("contract_id") != CANONICAL_METRIC_CONTRACT:
        failures.append("canonical endpoint is not request-snapshot v2")
    if canonical.get("implementation_status") != (
        "SOURCE_ALIGNED_IMPLEMENTED__PROVISIONAL_PENDING_SAME_TIME_GARRIDO_CONFIRMATION"
    ):
        failures.append("v2 implementation status is not fail-closed provisional")
    request = canonical.get("request_snapshot_semantics", {})
    required_fields = {
        "ret_bt_at_request",
        "ret_ut_at_request",
        "ret_ledger_snapshot_time",
        "ret_ledger_event_sequence",
    }
    if set(request.get("fields", [])) != required_fields:
        failures.append("request-snapshot field set is incomplete")
    if request.get("same_timestamp_authority") != (
        "PROVISIONAL_PENDING_GARRIDO_SIMULINK_CONFIRMATION"
    ):
        failures.append("same-time convention was promoted without Garrido authority")

    evidence = governance.get("source_evidence", {})
    evidence_rows = {
        "excel_formula_reaudit": excel_reaudit,
        "source_semantics_audit": source_semantics,
        "v2_implementation_audit": implementation_audit,
    }
    for label, payload in evidence_rows.items():
        row = evidence.get(label, {})
        relative = row.get("path")
        expected = row.get("sha256")
        path = root / relative if isinstance(relative, str) else None
        if path is None or not path.is_file():
            failures.append(f"missing metric evidence: {label}")
            continue
        hashes_checked += 1
        if not isinstance(expected, str) or sha256(path) != expected:
            failures.append(f"metric evidence hash mismatch: {label}")
        try:
            if load(path) != payload:
                failures.append(f"metric evidence payload mismatch: {label}")
        except (json.JSONDecodeError, OSError):
            failures.append(f"invalid metric evidence JSON: {label}")

    raw_formula = excel_reaudit.get("raw_formula_audit", {})
    if (
        raw_formula.get("total_rows") != 47_546
        or raw_formula.get("total_mismatches") != 0
        or raw_formula.get("max_abs_diff") != 0.0
        or len(raw_formula.get("sheets", {})) != 20
    ):
        failures.append("Excel formula reaudit is not exact on 47,546 rows")

    source_formula = source_semantics.get("workbook_formula_replay", {})
    source_totals = source_semantics.get("timing_and_population_evidence", {}).get(
        "totals", {}
    )
    if source_semantics.get("status") != (
        "FORMULA_EXACT__VISIBLE_ROWS_OBSERVED__OAT_BT_UT_RECONSTRUCTION_NOT_SOURCE_VALIDATED"
    ):
        failures.append("source audit no longer rejects OAT-derived Bt/Ut")
    if evidence.get("source_semantics_audit", {}).get("content_sha256") != (
        source_semantics.get("content_sha256")
    ):
        failures.append("source-semantics content hash mismatch")
    if (
        source_formula.get("formula_rows") != 47_546
        or source_formula.get("mismatches") != 0
        or source_formula.get("max_abs_diff") != 0.0
        or source_totals.get("OATj_adjacent_inversions_in_j_order") != 16_391
        or source_totals.get("sumUt_adjacent_decreases_in_j_order") != 0
        or source_totals.get("sumUt_adjacent_decreases_in_OATj_order") != 8_340
        or source_totals.get("visible_rows_only_OAT_Bt_matches") != 590
        or source_totals.get("visible_rows_only_OAT_Bt_total") != 47_546
    ):
        failures.append("source-semantics audit counters mismatch")

    replay = implementation_audit.get("canonical_aggregator_workbook_replay", {})
    if implementation_audit.get("canonical_development_contract") != (
        CANONICAL_METRIC_CONTRACT
    ):
        failures.append("implementation audit contract mismatch")
    if evidence.get("v2_implementation_audit", {}).get("content_sha256") != (
        implementation_audit.get("content_sha256")
    ):
        failures.append("v2 implementation content hash mismatch")
    if implementation_audit.get("status") != (
        "SOURCE_ALIGNED_IMPLEMENTED__PROVISIONAL_PENDING_SAME_TIME_GARRIDO_CONFIRMATION"
    ):
        failures.append("implementation audit lost its provisional hold")
    if (
        replay.get("passes") is not True
        or replay.get("formula_rows") != 47_546
        or replay.get("mismatches") != 0
        or replay.get("max_abs_diff") != 0.0
        or len(replay.get("sheets", {})) != 20
    ):
        failures.append("canonical v2 aggregator does not exactly replay the workbook")

    implementation_sources = implementation_audit.get("implementation_sources", {})
    required_implementation_sources = {
        "supply_chain/episode_metrics.py",
        "supply_chain/garrido_replication.py",
        "supply_chain/ret_thesis.py",
        "supply_chain/supply_chain.py",
        "tests/test_ret_excel_request_snapshot_contract.py",
    }
    if set(implementation_sources) != required_implementation_sources:
        failures.append("v2 implementation source manifest is incomplete")
    for relative in sorted(required_implementation_sources & set(implementation_sources)):
        path = root / relative
        implementation_source_hashes_checked += 1
        if not path.is_file() or sha256(path) != implementation_sources[relative]:
            failures.append(f"v2 implementation source hash mismatch: {relative}")
    native_test = evidence.get("v2_implementation_audit", {})
    if native_test.get("native_des_snapshot_integration_test") != (
        "tests/test_ret_excel_request_snapshot_contract.py::"
        "test_native_des_captures_request_snapshots_before_queue_entry"
    ):
        failures.append("native DES snapshot integration test is not pinned")
    if native_test.get("native_des_snapshot_test_sha256") != implementation_sources.get(
        "tests/test_ret_excel_request_snapshot_contract.py"
    ):
        failures.append("native DES snapshot integration test hash mismatch")

    legacy = governance.get("superseded_contracts", {}).get(
        LEGACY_VISIBLE_METRIC_CONTRACT, {}
    )
    required_prohibitions = {
        "Paper-2 positive result",
        "Paper-2 null result",
        "H_PI",
        "H_obs",
        "family or comparator ceiling",
        "terminal Return A or Return B evidence",
    }
    if legacy.get("status") != "QUARANTINED_METRIC_DEVELOPMENT_ONLY":
        failures.append("visible-v1 is not quarantined")
    if not required_prohibitions.issubset(set(legacy.get("prohibited_claims", []))):
        failures.append("visible-v1 prohibited-claim set is incomplete")

    quarantines = {
        row.get("scope_id"): row
        for row in governance.get("quarantine_registry", [])
    }
    required_quarantines = {
        "all_ret_excel_visible_v1_oat_ledger_outputs",
        "program_h_visible_v1",
        "program_j_visible_v1",
        "mtr_visible_v1_switch_frontiers",
    }
    if not required_quarantines.issubset(quarantines):
        failures.append("H/J/MTR or global visible-v1 quarantine is missing")
    for scope_id in required_quarantines & set(quarantines):
        if quarantines[scope_id].get("paper2_authority") is not False:
            failures.append(f"quarantined scope retains Paper-2 authority: {scope_id}")

    release = governance.get("v2_release_gates", {})
    if (
        release.get("same_timestamp_garrido_confirmation") != "PENDING"
        or release.get("identical_tape_rescore_h_j_mtr") != "REQUIRED"
        or release.get("complete_same_contract_comparators_rebuilt") != "REQUIRED"
        or release.get("all_guardrails_rebuilt") != "REQUIRED"
        or release.get("virgin_confirmation")
        != "PROHIBITED_UNTIL_ALL_PRIOR_GATES_PASS"
    ):
        failures.append("v2 release gates do not fail closed")

    authorization = governance.get("paper2_authorization", {})
    for field in (
        "paper2_positive_confirmed",
        "paper2_null_confirmed",
        "paper2_ceiling_confirmed",
        "learner_authorized",
        "paper3_authorized",
        "prior_h_j_mtr_results_restored",
    ):
        if authorization.get(field) is not False:
            failures.append(f"scientific authorization must remain false: {field}")

    return {
        "passed": not failures,
        "failures": failures,
        "hashes_checked": hashes_checked,
        "implementation_source_hashes_checked": implementation_source_hashes_checked,
        "canonical_contract": canonical.get("contract_id"),
        "canonical_status": canonical.get("implementation_status"),
        "visible_v1_disposition": legacy.get("status"),
        "quarantined_scopes": sorted(required_quarantines & set(quarantines)),
        "paper2_positive_confirmed": authorization.get("paper2_positive_confirmed"),
        "paper2_null_confirmed": authorization.get("paper2_null_confirmed"),
        "paper2_ceiling_confirmed": authorization.get("paper2_ceiling_confirmed"),
        "prior_h_j_mtr_results_restored": authorization.get(
            "prior_h_j_mtr_results_restored"
        ),
    }


def validate_paper3_claim_supersession(
    supersession: dict[str, Any],
    *,
    root: Path = ROOT,
) -> dict[str, Any]:
    """Validate the fail-closed precedence rule for historical retained claims.

    C12 is preserved as bounded historical evidence.  It may not be interpreted
    as current Paper-3 authorization unless the current task independently
    establishes the Paper-2 learned-value prerequisite.  The current artifact
    intentionally records that this prerequisite is unmet.
    """
    failures: list[str] = []
    hashes_checked = 0

    if supersession.get("schema_version") != "paper3_claim_supersession_v1":
        failures.append("invalid schema")
    if supersession.get("status") != (
        "PAPER3_NOT_AUTHORIZED__DEPENDENCY_PAPER2_LEARNED_VALUE_UNMET"
    ):
        failures.append("invalid current Paper-3 status")

    historical = supersession.get("historical_claim", {})
    if historical.get("claim_id") != "C12":
        failures.append("historical claim must be C12")
    if historical.get("historical_status") != (
        "Supported at small effect size, both observation arms"
    ):
        failures.append("historical C12 status mismatch")
    if historical.get("bounded_interpretation_retained") is not True:
        failures.append("bounded historical interpretation was not retained")
    if historical.get("authorization_transferable_to_current_paper3") is not False:
        failures.append("historical C12 was made transferable to current Paper 3")

    def validate_hashed_file(row: dict[str, Any], label: str) -> Path | None:
        nonlocal hashes_checked
        relative = row.get("path")
        expected = row.get("sha256")
        path = root / relative if isinstance(relative, str) else None
        if path is None or not path.is_file():
            failures.append(f"missing {label} artifact")
            return None
        if not isinstance(expected, str) or sha256(path) != expected:
            failures.append(f"{label} artifact hash mismatch")
            return None
        hashes_checked += 1
        return path

    c12_source = validate_hashed_file(historical.get("source", {}), "C12 source")
    if c12_source is not None:
        source_text = c12_source.read_text()
        if "| C12 | Retained learning" not in source_text or (
            "Supported at small effect size, both observation arms" not in source_text
        ):
            failures.append("hashed C12 source does not contain the historical claim")
    h4_source = validate_hashed_file(historical.get("evidence", {}), "C12 evidence")
    if h4_source is not None:
        h4_text = h4_source.read_text()
        if "8 online-adaptation cycles" not in h4_text or (
            "not a claim of" not in h4_text
            or "cross-campaign organizational learning" not in h4_text
        ):
            failures.append("hashed C12 evidence lacks its own scope limitation")

    correction = supersession.get("later_same_contract_correction", {})
    correction_path = validate_hashed_file(correction, "same-contract correction")
    if correction_path is not None:
        correction_text = correction_path.read_text()
        if "PPO minus static is `−0.000018049`" not in correction_text or (
            "Retire the claim that PPO has a Track B adaptive" not in correction_text
        ):
            failures.append("same-contract correction does not retire learned advantage")

    def dotted_value(payload: dict[str, Any], dotted: str) -> Any:
        value: Any = payload
        for key in dotted.split("."):
            if not isinstance(value, dict) or key not in value:
                return object()
            value = value[key]
        return value

    current_inputs = supersession.get("current_task_inputs", [])
    if len(current_inputs) != 3:
        failures.append("current-task input set must contain exactly three artifacts")
    for index, row in enumerate(current_inputs):
        path = validate_hashed_file(row, f"current-task input {index}")
        if path is None:
            continue
        if "required_values" in row:
            try:
                payload = load(path)
            except (json.JSONDecodeError, OSError):
                failures.append(f"current-task input {index} is not valid JSON")
                continue
            for dotted, expected in row["required_values"].items():
                if dotted_value(payload, dotted) != expected:
                    failures.append(
                        f"current-task input {index} required value mismatch: {dotted}"
                    )
        required_text = row.get("required_text")
        if required_text is not None and required_text not in path.read_text():
            failures.append(f"current-task input {index} required text missing")

    gate = supersession.get("current_task_authorization_gate", {})
    if gate.get("canonical_endpoint") != CANONICAL_METRIC_CONTRACT:
        failures.append("current task canonical endpoint mismatch")
    if gate.get("canonical_endpoint_status") != (
        "SOURCE_ALIGNED_IMPLEMENTED__PROVISIONAL_PENDING_SAME_TIME_GARRIDO_CONFIRMATION_AND_V2_RESCORE"
    ):
        failures.append("current task metric gate is not provisional and fail closed")
    for field in (
        "paper2_observable_adaptive_value_confirmed",
        "paper2_learned_adaptive_value_confirmed",
        "h_learned_positive_against_strongest_same_contract_comparator",
        "virgin_paper2_confirmation_passed",
    ):
        if gate.get(field) is not False:
            failures.append(f"current prerequisite must remain false: {field}")
    if len(gate.get("required_before_paper3", [])) != 4:
        failures.append("Paper-3 prerequisite set is incomplete")
    if len(gate.get("unmet_requirements", [])) != 4:
        failures.append("current unmet-requirement set is incomplete")

    disposition = supersession.get("effective_current_disposition", {})
    required_disposition = {
        "historical_c12_result_retracted": False,
        "historical_c12_may_be_promoted_as_current_paper3_result": False,
        "paper3_authorized": False,
        "new_retained_learning_execution_authorized": False,
        "paper3_claim_allowed": (
            "Only PAPER3_NOT_AUTHORIZED__DEPENDENCY_PAPER2_LEARNED_VALUE_UNMET"
        ),
    }
    for field, expected in required_disposition.items():
        if disposition.get(field) != expected:
            failures.append(f"invalid effective disposition: {field}")
    if "Historical C12 alone can never reopen the gate" not in disposition.get(
        "reopening_rule", ""
    ):
        failures.append("reopening rule does not fail closed on historical C12")

    return {
        "passed": not failures,
        "failures": failures,
        "hashes_checked": hashes_checked,
        "historical_claim_preserved": (
            historical.get("bounded_interpretation_retained") is True
        ),
        "paper3_authorized": disposition.get("paper3_authorized"),
    }


def validate_boundary_family_proof_ledger(
    registry: dict[str, Any],
    ledger: dict[str, Any],
    *,
    root: Path = ROOT,
) -> dict[str, Any]:
    """Validate family-wide Return-B proof objects independently of state labels.

    A registry label is routing metadata, not a scientific ceiling.  This check
    therefore requires a complete proof object, canonical metric, named and
    matched resources, a complete comparator, and content-addressed evidence
    before any family can count toward a terminal boundary.
    """
    failures: list[dict[str, Any]] = []
    registry_rows = {row["family_id"]: row for row in registry["approaches"]}
    ledger_rows = {row["family_id"]: row for row in ledger.get("families", [])}

    schema_ok = ledger.get("schema_version") == "paper2_boundary_family_proof_ledger_v2"
    if not schema_ok:
        failures.append({"scope": "ledger", "reason": "invalid_schema"})

    coverage_ok = set(registry_rows) == set(ledger_rows) and len(ledger_rows) == len(
        ledger.get("families", [])
    )
    if not coverage_ok:
        failures.append(
            {
                "scope": "ledger",
                "reason": "family_coverage_mismatch_or_duplicate",
                "registry_only": sorted(set(registry_rows) - set(ledger_rows)),
                "ledger_only": sorted(set(ledger_rows) - set(registry_rows)),
            }
        )

    terminal_ids: list[str] = []
    nonterminal_ids: list[str] = []
    artifact_hashes_checked = 0
    for family_id in sorted(set(registry_rows) & set(ledger_rows)):
        registry_row = registry_rows[family_id]
        row = ledger_rows[family_id]
        row_failures: list[str] = []
        if row.get("registry_state") != registry_row.get("state"):
            row_failures.append("registry_state_mismatch")
        governing_metric = row.get("governing_metric")
        if governing_metric not in {
            LEGACY_VISIBLE_METRIC_CONTRACT,
            CANONICAL_METRIC_CONTRACT,
        }:
            row_failures.append("unknown_or_missing_governing_metric")
        if not row.get("evidence_metric"):
            row_failures.append("missing_evidence_metric")
        if not row.get("named_resource_vector"):
            row_failures.append("missing_named_resource_vector")

        artifact_failures: list[str] = []
        for artifact in row.get("proof_artifacts", []):
            relative = artifact.get("path")
            expected = artifact.get("sha256")
            if not relative or not expected:
                artifact_failures.append("missing_path_or_sha256")
                continue
            path = Path(relative)
            if not path.is_absolute():
                path = root / path
            if not path.is_file():
                artifact_failures.append(f"missing:{relative}")
                continue
            artifact_hashes_checked += 1
            if sha256(path) != expected:
                artifact_failures.append(f"hash_mismatch:{relative}")
        if artifact_failures:
            row_failures.extend(artifact_failures)

        claimed_terminal = row.get("terminal_b_eligible") is True
        if claimed_terminal:
            if governing_metric == LEGACY_VISIBLE_METRIC_CONTRACT:
                row_failures.append("legacy_visible_v1_cannot_support_terminal_claim")
            if governing_metric != CANONICAL_METRIC_CONTRACT:
                row_failures.append("terminal_claim_not_scored_under_canonical_v2")
            if row.get("evidence_metric") != row.get("governing_metric"):
                row_failures.append("evidence_metric_does_not_match_governing_metric")
            if row.get("closure_kind") not in TERMINAL_B_PROOF_CLASSES:
                row_failures.append("ineligible_closure_kind")
            if row.get("comparator_complete") is not True:
                row_failures.append("comparator_incomplete")
            if row.get("resource_matching_certified") is not True:
                row_failures.append("resources_not_certified_matched")
            if row.get("scope_closes_registered_family") is not True:
                row_failures.append("proof_scope_not_family_wide")
            if row.get("unresolved_extension_or_domain_fact") is not False:
                row_failures.append("unresolved_extension_or_domain_fact")
            if not row.get("proof_artifacts"):
                row_failures.append("missing_content_addressed_proof")
            if row.get("closure_kind") == "certified_quantitative_global_upper_bound":
                ceiling = row.get("quantitative_ceiling") or {}
                ucb95 = ceiling.get("ucb95")
                if ucb95 is None or float(ucb95) >= 0.01:
                    row_failures.append("quantitative_ucb95_not_below_gate")
            elif row.get("closure_kind") in {"exact_global_zero", "formal_global_dominance"}:
                ceiling = row.get("quantitative_ceiling") or {}
                value = ceiling.get("value")
                if value is None or float(value) > 0.0:
                    row_failures.append("exact_or_dominance_ceiling_not_nonpositive")

        if row_failures:
            failures.append(
                {"scope": "family", "family_id": family_id, "reasons": row_failures}
            )
        if claimed_terminal and not row_failures:
            terminal_ids.append(family_id)
        else:
            nonterminal_ids.append(family_id)

    all_families_terminal = (
        schema_ok
        and coverage_ok
        and not failures
        and bool(ledger_rows)
        and len(terminal_ids) == len(registry_rows)
    )
    summary = ledger.get("summary", {})
    summary_consistent = (
        summary.get("registered_families") == len(ledger_rows)
        and summary.get("terminal_b_eligible_families") == len(terminal_ids)
        and summary.get("nonterminal_families") == len(nonterminal_ids)
        and ledger.get("terminal_b_supported") is all_families_terminal
    )
    if not summary_consistent:
        failures.append({"scope": "ledger", "reason": "summary_or_terminal_flag_mismatch"})
        all_families_terminal = False

    return {
        "schema_ok": schema_ok,
        "coverage_ok": coverage_ok,
        "summary_consistent": summary_consistent,
        "artifact_hashes_checked": artifact_hashes_checked,
        "terminal_family_ids": terminal_ids,
        "nonterminal_family_ids": nonterminal_ids,
        "all_families_terminal_b_eligible": all_families_terminal,
        "failures": failures,
    }


def validate_historical_visible_v1_ceiling_audit(
    audit: dict[str, Any],
    *,
    root: Path = ROOT,
) -> dict[str, Any]:
    failures: list[str] = []
    rows = audit.get("lanes", [])
    hashes_checked = 0
    if audit.get("schema_version") != "paper2_historical_visible_v1_ceiling_audit_v1":
        failures.append("invalid schema")
    if audit.get("governing_metric") != "ret_excel_visible_v1":
        failures.append("invalid governing metric")
    summary = audit.get("summary", {})
    if summary.get("lanes_audited") != len(rows) or len(rows) != 10:
        failures.append("lane count mismatch")
    if summary.get("existing_family_wide_visible_v1_matched_resource_ceilings") != 0:
        failures.append("unsupported existing visible-v1 ceiling count")
    if summary.get("paper2_confirmed") is not False:
        failures.append("audit cannot confirm Paper 2")
    if summary.get("terminal_boundary_supported") is not False:
        failures.append("audit cannot support terminal boundary")

    by_lane = {row.get("lane"): row for row in rows}
    if len(by_lane) != len(rows):
        failures.append("duplicate lane")
    if by_lane.get("Program H", {}).get("evidence_metric") != (
        "ret_excel_full_ledger_order_adapter"
    ):
        failures.append("Program H evidence metric mismatch")
    if by_lane.get("Program J", {}).get("evidence_metric") != "ret_excel_full_ledger":
        failures.append("Program J evidence metric mismatch")

    for row in rows:
        artifacts = row.get("artifacts")
        if artifacts is None:
            artifacts = [row.get("artifact")]
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                failures.append(f"missing artifact for {row.get('lane')}")
                continue
            relative = artifact.get("path")
            expected = artifact.get("sha256")
            path = root / relative if isinstance(relative, str) else None
            if path is None or not path.is_file():
                failures.append(f"missing artifact: {relative}")
                continue
            hashes_checked += 1
            if not isinstance(expected, str) or sha256(path) != expected:
                failures.append(f"artifact hash mismatch: {relative}")
    return {
        "passed": not failures,
        "failures": failures,
        "lane_count": len(rows),
        "hashes_checked": hashes_checked,
        "existing_visible_v1_ceilings": summary.get(
            "existing_family_wide_visible_v1_matched_resource_ceilings"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=SEARCH / "boundary_verification.json",
    )
    args = parser.parse_args()

    checks: list[dict[str, Any]] = []

    def record(name: str, passed: bool, detail: Any) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    taxonomy = load(SEARCH / "phase0_failure_taxonomy.json")
    registry = load(SEARCH / "approach_registry.json")
    boundary_proofs = load(SEARCH / "boundary_family_proof_ledger.json")
    prelearner = load(SEARCH / "prelearner_contract.json")
    interventions = load(SEARCH / "candidate_intervention_ledger.json")
    decision_coverage = load(SEARCH / "decision_right_catalog_coverage.json")
    decision_catalog = load(ROOT / "contracts" / "decision_right_catalog_v1.json")
    paper3_supersession = load(SEARCH / "paper3_claim_supersession.json")
    historical_ceiling_audit = load(
        SEARCH / "historical_visible_v1_ceiling_audit_20260714.json"
    )
    metric_audit = load(SEARCH / "metric_governance_audit.json")
    metric_source_semantics = load(
        SEARCH / "ret_excel_visible_v1_source_semantics_audit_20260714.json"
    )
    metric_v2_implementation = load(
        SEARCH / "ret_excel_request_snapshot_v2_implementation_audit_20260714.json"
    )
    excel_metric_reaudit = load(SEARCH / "excel_metric_reaudit_20260713.json")

    record("taxonomy_schema", taxonomy.get("schema_version") == "paper2_failure_taxonomy_v1",
           taxonomy.get("schema_version"))
    family_ids = [row["family_id"] for row in taxonomy["decision_families"]]
    record("taxonomy_unique_families", len(family_ids) == len(set(family_ids)), len(family_ids))
    record("taxonomy_covers_A_through_K3", len(family_ids) >= 17,
           {"n": len(family_ids), "first": family_ids[:3], "last": family_ids[-3:]})

    approach_ids = {row["family_id"] for row in registry["approaches"]}
    intervention_ids = {row["family_id"] for row in interventions["interventions"]}
    record(
        "intervention_ledger_covers_registry",
        approach_ids == intervention_ids,
        {"approaches": len(approach_ids), "interventions": len(intervention_ids)},
    )
    record(
        "prelearner_gate_is_fail_closed",
        prelearner["status"] == "NO_CURRENT_CANDIDATE_ELIGIBLE"
        and prelearner["primary_endpoint"].startswith(CANONICAL_METRIC_CONTRACT)
        and prelearner["metric_readiness"]["status"]
        == "HOLD_CANONICAL_V2_RESCORE_AND_DOMAIN_CONFIRMATION_REQUIRED"
        and prelearner["metric_readiness"][
            "paper2_null_positive_or_ceiling_authorized"
        ]
        is False
        and prelearner["learner_authorized"] is False
        and prelearner["paper3_authorized"] is False
        and set(prelearner["candidate_entry_status"]) == approach_ids,
        {
            "status": prelearner["status"],
            "mandatory_gates": len(prelearner["mandatory_gates"]),
            "learner_authorized": prelearner["learner_authorized"],
        },
    )
    paper3_supersession_validation = validate_paper3_claim_supersession(
        paper3_supersession
    )
    record(
        "historical_c12_is_superseded_for_current_paper3_authorization",
        paper3_supersession_validation["passed"]
        and paper3_supersession_validation["paper3_authorized"] is False,
        paper3_supersession_validation,
    )
    historical_ceiling_validation = validate_historical_visible_v1_ceiling_audit(
        historical_ceiling_audit
    )
    record(
        "historical_visible_v1_ceiling_audit_is_fail_closed",
        historical_ceiling_validation["passed"]
        and historical_ceiling_validation["existing_visible_v1_ceilings"] == 0,
        historical_ceiling_validation,
    )
    boundary_proof_validation = validate_boundary_family_proof_ledger(
        registry,
        boundary_proofs,
    )
    record(
        "boundary_proof_ledger_schema_and_family_coverage",
        boundary_proof_validation["schema_ok"]
        and boundary_proof_validation["coverage_ok"],
        {
            "schema_ok": boundary_proof_validation["schema_ok"],
            "coverage_ok": boundary_proof_validation["coverage_ok"],
        },
    )
    record(
        "boundary_proof_artifact_hashes_and_summary_are_consistent",
        not boundary_proof_validation["failures"]
        and boundary_proof_validation["summary_consistent"],
        boundary_proof_validation,
    )
    catalog_decision_ids = {
        row["id"] for row in decision_catalog["factors"]
        if row["class"] == "decision_right"
    }
    coverage_ids = {row["factor_id"] for row in decision_coverage["rows"]}
    coverage_evidence_paths = {
        evidence
        for row in decision_coverage["rows"]
        for evidence in row["evidence"]
    }
    op7_release = next(
        row for row in decision_coverage["rows"]
        if row["factor_id"] == "op7_release_period"
    )
    op6_rework = next(
        row for row in decision_coverage["rows"]
        if row["factor_id"] == "op6_rework_rule"
    )
    record(
        "declared_decision_right_catalog_is_routed_with_known_gaps",
        decision_coverage["decision_right_count"] == 32
        and decision_coverage["all_decision_rights_covered_once"] is True
        and decision_coverage["mechanism_family_complete"] is False
        and decision_coverage["newly_identified_gap_count"] == 2
        and decision_coverage["new_executable_source_native_candidate_count"] == 0
        and decision_coverage["exact_current_kernel_zero_count"] == 12
        and coverage_ids == catalog_decision_ids
        and all((ROOT / path).exists() for path in coverage_evidence_paths)
        and op7_release["catalog_status"] == "implemented"
        and op7_release["disposition"] == "transition_dead_configuration_field"
        and op7_release["current_kernel_h_pi_ceiling"] == 0.0
        and op6_rework["disposition"]
        == "transition_live_fidelity_configuration_not_adaptive_action"
        and op6_rework["current_kernel_h_pi_ceiling"] is None
        and "op7_rop" in (ROOT / "supply_chain" / "config.py").read_text()
        and "op7_rop" not in (ROOT / "supply_chain" / "supply_chain.py").read_text(),
        {
            "catalog_count": len(catalog_decision_ids),
            "coverage_count": len(coverage_ids),
            "exact_current_kernel_zero_count": decision_coverage["exact_current_kernel_zero_count"],
            "new_executable_source_native_candidate_count": decision_coverage["new_executable_source_native_candidate_count"],
            "mechanism_family_complete": decision_coverage["mechanism_family_complete"],
            "newly_identified_gap_count": decision_coverage["newly_identified_gap_count"],
            "op7_release_period": op7_release,
            "op6_rework_rule": op6_rework,
        },
    )

    source_details = {}
    source_ok = True
    for path, expected in SOURCE_HASHES.items():
        actual = sha256(path) if path.exists() else None
        source_details[str(path)] = {"expected": expected, "actual": actual}
        source_ok &= actual == expected
    record("source_hashes", source_ok, source_details)

    metric_validation = validate_metric_governance(
        metric_audit,
        metric_source_semantics,
        metric_v2_implementation,
        excel_metric_reaudit,
    )
    record(
        "canonical_v2_metric_is_provisional_and_visible_v1_is_quarantined",
        metric_validation["passed"]
        and metric_validation["paper2_positive_confirmed"] is False
        and metric_validation["paper2_null_confirmed"] is False
        and metric_validation["paper2_ceiling_confirmed"] is False
        and metric_validation["prior_h_j_mtr_results_restored"] is False,
        metric_validation,
    )

    ancestor = "ef6b53b7cac9c1cbdcdc4347a31c8300d1941fc0"
    is_ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, "HEAD"],
        cwd=ROOT,
        check=False,
    ).returncode == 0
    record("k3_corrective_commit_is_ancestor", is_ancestor,
           {"corrective_commit": ancestor, "head": git("rev-parse", "HEAD")})

    k3_path = ROOT / "results" / "k3" / "open_loop_confound_audit.json"
    k3 = load(k3_path)
    k3_ok = (
        k3["verdict"] == "RETRACT_K3_ADAPTIVE_AND_NEURAL_CLAIMS_STATIC_PERIOD8_CONFOUND"
        and k3["paper2_adaptive_confirmed"] is False
        and k3["paper3_neural_retention_authorized"] is False
        and k3["ppo_seed0_unique_test_sequences"] == 1
        and k3["ppo_seed0_modal_sequence"] == [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.0, 0.0]
        and k3["ppo_seed0_minus_fixed_ret"] == [0.0, 0.0, 0.0]
        and k3["learner_test_6900001_6900120"]["fixed_minus_mpc_ordered_D0"] == [0.0, 0.0, 0.0]
    )
    record("k3_static_period8_confound", k3_ok,
           {"sha256": sha256(k3_path), "verdict": k3["verdict"]})

    k3_dominance_path = SEARCH / "k3_frontloading_dominance_certificate.json"
    k3_dominance = load(k3_dominance_path)
    k3_graph = k3_dominance["resource_graph"]
    k3_prefix = k3_dominance["exhaustive_prefix_certificate"]
    k3_envelope = k3_dominance["non_superior_resource_envelope_certificate"]
    k3_metric = k3_dominance["full_ledger_metric_monotonicity"]
    k3_full = k3_dominance["metric_conclusions"]["frozen_k3_full_ledger"]
    k3_visible = k3_dominance["metric_conclusions"]["ret_excel_visible_v1"]
    k3_visible_counterexample = k3_dominance[
        "visible_ledger_nonmonotonicity_counterexample"
    ]
    k3_dominance_ok = (
        k3_dominance["status"]
        == "PASS_EXACT_PATHWISE_FRONTLOADING_DOMINANCE__FROZEN_K3_FULL_LEDGER_H_PI_ZERO"
        and k3_dominance["generated_without_stochastic_tapes"] is True
        and k3_graph["reachable_state_count"] == 61
        and k3_graph["reachable_edge_count"] == 260
        and k3_graph["effective_exact_budget_schedule_count"] == 6_371
        and k3_graph["all_terminal_spends_equal_budget"] is True
        and k3_prefix["schedules_checked"] == 6_371
        and k3_prefix["prefix_comparisons_checked"] == 50_968
        and k3_prefix["violation_count"] == 0
        and k3_envelope["schedule_count_total_spend_le_budget"] == 5_758_374
        and k3_envelope["exact_budget_schedule_count"] == 6_371
        and k3_envelope["strictly_under_budget_schedule_count"] == 5_752_003
        and k3_envelope["prefix_dominance_violation_count"] == 0
        and k3_prefix["unique_nondominated_schedule_count"] == 1
        and k3_metric["status_vectors_checked_against_live_aggregator"] == 6_561
        and k3_metric["coordinatewise_status_pairs_checked"] == 1_679_616
        and k3_metric["abstraction_mismatch_count"] == 0
        and k3_metric["metric_dominance_violation_count"] == 0
        and all(k3_dominance["source_semantics"]["checks"].values())
        and k3_full["metric"] == "ret_excel_full_ledger_order"
        and k3_full["h_pi"] == 0.0
        and k3_full["h_obs"] == 0.0
        and k3_visible["result"]
        == "NOT_UNCONDITIONALLY_CERTIFIED_BY_THIS_THEOREM"
        and k3_visible_counterexample["status"]
        == "PASS_LIVE_AGGREGATOR_COUNTEREXAMPLE"
        and k3_visible_counterexample["front_loaded_path"][
            "mean_ret_excel"
        ]
        == 6 / 7
        and k3_visible_counterexample["later_batched_path"][
            "mean_ret_excel"
        ]
        == 1.0
    )
    record(
        "k3_exact_frontloading_full_ledger_hpi_zero",
        k3_dominance_ok,
        {
            "sha256": sha256(k3_dominance_path),
            "states": k3_graph["reachable_state_count"],
            "schedules": k3_graph["effective_exact_budget_schedule_count"],
            "prefix_violations": k3_prefix["violation_count"],
            "full_ledger_h_pi": k3_full["h_pi"],
            "visible_v1_scope": k3_visible["result"],
        },
    )

    track_b = load(ROOT / "outputs" / "experiments" /
                   "track_b_same_contract_challenge_2026-07-10" / "summary.json")
    track_b_primary = track_b["ret_excel"]["canonical_joint_minus_best_full_static"]
    record(
        "track_b_same_contract_reversal",
        track_b["stop_rule"]["passed"] is False
        and track_b_primary["two_way_ci95"][1] < 0
        and track_b_primary["tapes_positive"] == 2,
        track_b_primary,
    )

    program_h = load(ROOT / "results" / "program_h" / "bound" / "verdict.json")
    h_bound = program_h["certified_upper_bound_J_PI_minus_ABAB"]
    record(
        "program_h_bound_remains_above_mcid",
        close(h_bound["mean"], 0.01641384654689033)
        and h_bound["ci95"][1] > 0.01,
        h_bound,
    )

    program_i = load(ROOT / "results" / "program_i" / "branching" / "verdict.json")
    i_transport = program_i["families"]["transport"]["horizons"]["8"]
    record(
        "program_i_resource_equal_transport_tiny",
        i_transport["gates"]["resource_equivalence"] is True
        and i_transport["oracle_delta"]["ci95"][1] < 0.01,
        i_transport["oracle_delta"],
    )

    maintenance = load(ROOT / "results" / "paper2_maintenance" / "terminal_verdict.json")
    record(
        "maintenance_prelearner_stop",
        maintenance["verdict"] == "STOP_NO_OBSERVABLE_MAINTENANCE_HEADROOM"
        and maintenance["paper2_learner_authorized"] is False
        and maintenance["paper3_authorized"] is False
        and maintenance["central_screen"]["oracle_ret_delta_ci95"][1] < 0.01,
        maintenance["central_screen"],
    )

    bottleneck = load(ROOT / "results" / "paper2_bottleneck" /
                      "locked_confirmation" / "verdict.json")
    record(
        "bottleneck_observable_policy_stop",
        bottleneck["verdict"] == "STOP_NO_ADAPTIVE_BOTTLENECK_VALUE"
        and bottleneck["ppo_trained"] is False
        and bottleneck["gates"]["equal_team_hours"] is True
        and bottleneck["gates"]["crn"] is True,
        bottleneck["signal_policy_minus_best_constant"],
    )
    bottleneck_corrective = load(
        ROOT / "results" / "paper2_bottleneck" / "corrective_completeness_audit.json"
    )
    effect_quotient = load(
        ROOT / "results" / "paper2_bottleneck" / "effect_quotient_audit.json"
    )
    loose_bound = load(
        ROOT / "results" / "paper2_bottleneck" / "loose_canonical_upper_bound.json"
    )
    signal_mapping = load(
        ROOT / "results" / "paper2_bottleneck" / "signal_mapping_audit.json"
    )
    bottleneck_bound_contract = load(
        ROOT / "contracts" / "paper2_bottleneck_full_horizon_bound_v1.json"
    )
    record(
        "bottleneck_family_bound_gap_is_explicit",
        bottleneck_corrective["status"] == "ACTIVE_FOR_BOUND_NOT_TERMINAL"
        and bottleneck_corrective["calendar_frontier"]["effective_full_horizon_calendar_count"] == 11_184_811
        and bottleneck_corrective["calendar_frontier"]["exact_bound_available"] is False
        and bottleneck_corrective["component_resource_audit"]["total_team_hours_equal"] is True
        and bottleneck_corrective["component_resource_audit"]["allocation_destination_hours_are_separate_contract_resources"] is False
        and bottleneck_corrective["component_resource_audit"]["reserve_resource_semantics_resolved"] is False
        and bottleneck_corrective["null_physics"]["exact_selected_metric_equivalence"] is True
        and bottleneck_corrective["null_physics"]["n_retrospective_tapes"] == 3
        and bottleneck_corrective["trajectory_audit"]["locked_unique_signal_sequences"] == 120,
        {
            "calendar_frontier": bottleneck_corrective["calendar_frontier"],
            "component_resource_audit": bottleneck_corrective["component_resource_audit"],
            "null_physics": bottleneck_corrective["null_physics"],
        },
    )
    record(
        "bottleneck_exact_effect_quotient_is_only_an_acceleration",
        effect_quotient["scientific_status"] == "EXACT_ACCELERATION_DESIGN_NOT_H_PI_RESULT"
        and effect_quotient["full_calendar_count"] == 11_184_811
        and effect_quotient["splits"]["calibration"]["distinct_effect_executions"] == 30_765_821
        and effect_quotient["splits"]["locked"]["distinct_effect_executions"] == 57_918_762
        and effect_quotient["totals"]["effect_quotient_des_runs"] == 88_684_583
        and effect_quotient["collision_validation"]["selected_outcomes_exactly_equal"] is True
        and bool(effect_quotient["not_done"]),
        {
            "scientific_status": effect_quotient["scientific_status"],
            "totals": effect_quotient["totals"],
            "not_done": effect_quotient["not_done"],
        },
    )
    record(
        "bottleneck_full_horizon_bound_protocol_is_frozen_fail_closed",
        bottleneck_bound_contract["status"] == "RETROSPECTIVE_BOUND_PROTOCOL_FROZEN_BEFORE_FULL_EXECUTION"
        and bottleneck_bound_contract["complete_open_loop_family"]["effective_calendar_count"] == 11_184_811
        and bottleneck_bound_contract["complete_open_loop_family"]["locked_bound"]["algorithm_development_seed_excluded"] == 1_110_001
        and bottleneck_bound_contract["complete_open_loop_family"]["locked_bound"]["n_tapes"] == 119
        and bottleneck_bound_contract["exact_acceleration_requirements"]["failure_rule"].startswith("Any state-key collision")
        and bottleneck_bound_contract["inference"]["learner_authorized"] is False
        and bottleneck_bound_contract["inference"]["paper3_authorized"] is False,
        {
            "contract_id": bottleneck_bound_contract["contract_id"],
            "locked_bound": bottleneck_bound_contract["complete_open_loop_family"]["locked_bound"],
            "state_merge_rule": bottleneck_bound_contract["exact_acceleration_requirements"]["state_merge_rule"],
        },
    )
    record(
        "bottleneck_cheap_canonical_bound_is_valid_but_vacuous",
        loose_bound["scientific_status"] == "VALID_EMPIRICAL_TAPE_BOUND_TOO_LOOSE_TO_CLOSE"
        and loose_bound["summary"]["n_orders_unique"] == [143]
        and loose_bound["summary"]["tapes_with_upper_gap_at_most_0_01"] == 0
        and loose_bound["summary"]["empirical_upper_gap_ci95"][0] > 1.0,
        {
            "scientific_status": loose_bound["scientific_status"],
            "summary": loose_bound["summary"],
        },
    )
    record(
        "bottleneck_memoryless_signal_mapping_reduces_to_constant_M",
        signal_mapping["calibration"]["candidate_count"] == 27
        and signal_mapping["calibration"]["selected_mapping_equipment_transport_mission"] == "MMM"
        and signal_mapping["locked"]["ret_minus_constant_M_ci95"] == [0.0, 0.0, 0.0]
        and "not optimized H_obs" in signal_mapping["scientific_use"],
        {
            "selected_mapping": signal_mapping["calibration"]["selected_mapping_equipment_transport_mission"],
            "locked": signal_mapping["locked"],
        },
    )

    # A concurrently generated local artifact incorrectly labels the maximum of
    # three constant policies as a full-horizon PI ceiling.  That statistic is a
    # lower bound on the weekly oracle, and its bootstrap LCB cannot certify an
    # upper ceiling.  Keep it outside the evidence allowlist and prove that it
    # did not change the registry state.
    invalid_ceiling = SEARCH / "bottleneck_pi_ceiling.json"
    invalid_detail: dict[str, Any] = {
        "present": invalid_ceiling.exists(),
        "reason": "best-of-three constants is not an upper bound on the 24-week oracle; LCB<0.01 is not a ceiling test",
    }
    if invalid_ceiling.exists():
        invalid_detail.update({
            "sha256": sha256(invalid_ceiling),
            "local_unverified_verdict": load(invalid_ceiling).get("verdict"),
        })
    integrated = next(
        row for row in registry["approaches"]
        if row["family_id"] == "integrated_production_maintenance_routing_recovery_resource"
    )
    record(
        "invalid_constants_only_PI_claim_is_excluded",
        integrated["state"] == "active_for_bound"
        and integrated["current_physics_ceiling"]["available"] is False,
        invalid_detail,
    )

    # Commit a91890bf was pushed concurrently with a terminal Boundary-B label.
    # The tracked certificate is now explicitly retracted; its underlying atlas
    # remains useful only as corrective exploratory evidence.
    concurrent_audit = load(SEARCH / "concurrent_boundary_commit_audit.json")
    concurrent_certificate = load(
        ROOT / "results" / "paper2_search" / "boundary_certificate.json"
    )
    concurrent_atlas = load(
        ROOT / "results" / "paper2_search" / "voi_ceiling_atlas.json"
    )
    concurrent_atlas_corrective = load(
        ROOT / "results" / "paper2_search" / "voi_ceiling_atlas_corrective_audit.json"
    )
    concurrent_seed_ledger_path = (
        ROOT / "results" / "paper2_search" / "seed_burn_ledger.json"
    )
    concurrent_seed_ledger = load(concurrent_seed_ledger_path)
    concurrent_seed_ledger_text = concurrent_seed_ledger_path.read_text()
    corrected_seed_ledger = load(SEARCH / "seed_burn_ledger_correction.json")
    atlas_burn = next(
        row for row in corrected_seed_ledger["blocks"]
        if row["label"] == "concurrent_stylized_voi_atlas"
    )
    missing_referenced_files = [
        str(path.relative_to(ROOT))
        for path in (
            ROOT / "results" / "paper2_search" / "voi_ceiling_atlas_figure.py",
            ROOT / "tests" / "test_voi_ceiling_atlas.py",
        )
        if not path.exists()
    ]
    record(
        "concurrent_a918_terminal_boundary_is_superseded",
        concurrent_audit["audited_commit"].startswith("a91890bf")
        and concurrent_audit["verdict"] == "SUPERSEDED_NONTERMINAL_INSUFFICIENT_FOR_RETURN_B"
        and concurrent_audit["terminal_boundary_certified"] is False
        and concurrent_certificate["terminal_return"]
        == "RETRACTED_PENDING_EXACT_MTR_BOUND_AND_DOMAIN_FACTS"
        and concurrent_certificate["scientific_status"].startswith(
            "NONTERMINAL_LEGACY"
        )
        and concurrent_certificate["program_l_route_recourse_followup"][
            "verdict"
        ].startswith("REJECT_PROGRAM_L_TERMINAL_REAFFIRMATION")
        and concurrent_certificate["evidence_budget_discharged"]["families_with_quantitative_bounds"].startswith(">=3")
        and "NOT full Op1-Op13 DES" in concurrent_atlas["contract"]
        and "7_200_001+" in concurrent_atlas["design"]
        and "7200001" not in concurrent_seed_ledger_text
        and concurrent_seed_ledger["known_virgin_blocks_from_source_of_truth"]["k3_confirmation"] == "6800001-6800120"
        and concurrent_seed_ledger["known_virgin_blocks_from_source_of_truth"]["k3_learner_test"] == "6900001-6900120"
        and atlas_burn["range"] == "7200001-7299999"
        and atlas_burn["status"] == "burned_conservative_reservation"
        and set(missing_referenced_files) == {
            "results/paper2_search/voi_ceiling_atlas_figure.py",
            "tests/test_voi_ceiling_atlas.py",
        }
        and integrated["state"] == "active_for_bound"
        and integrated["current_physics_ceiling"]["available"] is False,
        {
            "concurrent_claim": concurrent_certificate["terminal_return"],
            "reported_quantitative_bounds": concurrent_certificate["evidence_budget_discharged"]["families_with_quantitative_bounds"],
            "atlas_contract": concurrent_atlas["contract"],
            "opened_unledgered_seed_block": "7200001+",
            "incorrectly_listed_known_virgin_k3_blocks": {
                "confirmation": concurrent_seed_ledger["known_virgin_blocks_from_source_of_truth"]["k3_confirmation"],
                "learner_test": concurrent_seed_ledger["known_virgin_blocks_from_source_of_truth"]["k3_learner_test"],
            },
            "corrected_conservative_burn": atlas_burn,
            "missing_referenced_files": missing_referenced_files,
            "governing_status": concurrent_audit["governing_current_status"],
        },
    )
    program_g_text = (ROOT / "supply_chain" / "program_g.py").read_text()
    simulate_orders_source = program_g_text.split("def simulate_orders", 1)[1].split(
        "def ret_order_metrics", 1
    )[0]
    ret_order_source = program_g_text.split("def ret_order_metrics", 1)[1].split(
        "# Cobb-Douglas", 1
    )[0]
    headroom_source = (ROOT / "supply_chain" / "headroom_sensitivity.py").read_text()
    positive_atlas_cells = [
        cell for cell in concurrent_atlas["cells"] if cell["H_obs"] >= 0.01
    ]
    atlas_cell_keys = set().union(*(cell.keys() for cell in concurrent_atlas["cells"]))
    record(
        "concurrent_atlas_is_not_a_canonical_Hobs_ceiling",
        "compute_order_level_ret_excel_formula" in ret_order_source
        and "compute_order_level_ret_excel_visible_ledger" not in ret_order_source
        and "tape.r22" not in simulate_orders_source
        and "obs = np.array([_ret(t, _belief_policy(t))" in headroom_source
        and len(positive_atlas_cells) == 2
        and not ({"lost", "worst_cssu", "quantity_ret", "tail_risk", "resource_ledger"} & atlas_cell_keys),
        {
            "metric_used": "compute_order_level_ret_excel_formula_not_visible_ledger",
            "r22_consumed_by_scored_simulate_orders": False,
            "Hobs_estimator": "one_belief_policy_tested_delta_not_maximum_or_ceiling",
            "cells_Hobs_ge_0_01": len(positive_atlas_cells),
            "missing_guardrail_fields": sorted(
                {"lost", "worst_cssu", "quantity_ret", "tail_risk", "resource_ledger"}
                - atlas_cell_keys
            ),
        },
    )
    corrected_positive = concurrent_atlas_corrective["positive_cells"]
    corrected_atlas_summary = concurrent_atlas_corrective["all_cell_summary"]
    record(
        "concurrent_atlas_positive_cells_fail_lost_guardrail_and_visible_optimum_is_degenerate",
        concurrent_atlas_corrective["scientific_status"] == "EXPLORATORY_ATLAS_NOT_CANONICAL_HOBS_OR_BOUND"
        and corrected_atlas_summary["n_cells"] == 64
        and corrected_atlas_summary["stored_statistic_reproduced_cells"] == 64
        and corrected_atlas_summary["unguarded_sparse_visible_metric_H_PI_exact_zero_cells"] == 64
        and corrected_atlas_summary["unguarded_sparse_visible_metric_tested_belief_positive_cells"] == 0
        and concurrent_atlas_corrective["positive_cell_count_in_source"] == 2
        and all(row["stored_statistic_reproduced"] for row in corrected_positive)
        and all(row["full_order_formula"]["tested_belief_minus_static"] >= 0.01 for row in corrected_positive)
        and all(row["full_order_formula"]["belief_minus_static_lost_ci95"][1] > 0 for row in corrected_positive)
        and all(row["full_order_formula"]["belief_lost_better_tapes"] == 0 for row in corrected_positive)
        and all(close(row["visible_ledger"]["H_PI"], 0.0) for row in corrected_positive)
        and all(row["visible_ledger"]["best_static_sequence"] == ["HOLD"] * 4 for row in corrected_positive)
        and all(row["visible_ledger"]["best_static_mean_lost_orders"] == 48.0 for row in corrected_positive)
        and all(row["visible_ledger"]["best_static_mean_visible_rows"] == 0.0 for row in corrected_positive)
        and all(row["visible_ledger"]["project_H_PI_valid"] is False for row in corrected_positive)
        and concurrent_atlas_corrective["r22_inertness"]["r22_arrays_differ_on_at_least_one_tape"] is True
        and concurrent_atlas_corrective["r22_inertness"]["maximum_score_differences"] == {
            "full_order_formula": 0.0,
            "visible_ledger": 0.0,
        }
        and concurrent_atlas_corrective["terminal_boundary_supported"] is False,
        {
            "source_seed_pattern": concurrent_atlas_corrective["source_seed_pattern_reconstructed"],
            "all_cell_summary": corrected_atlas_summary,
            "cells": [
                {
                    "source_cell_index": row["source_cell_index"],
                    "stored_reproduced": row["stored_statistic_reproduced"],
                    "full_formula_tested_delta": row["full_order_formula"]["tested_belief_minus_static"],
                    "full_formula_lost_delta_ci95": row["full_order_formula"]["belief_minus_static_lost_ci95"],
                    "lost_guardrail_pass": row["full_order_formula"]["belief_minus_static_lost_ci95"][1] <= 0,
                    "unguarded_sparse_visible_metric_H_PI": row["visible_ledger"]["H_PI"],
                    "unguarded_best_static_lost_mean": row["visible_ledger"]["best_static_mean_lost_orders"],
                    "project_H_PI_valid": row["visible_ledger"]["project_H_PI_valid"],
                }
                for row in corrected_positive
            ],
            "r22_inertness": concurrent_atlas_corrective["r22_inertness"],
        },
    )

    states = [row["state"] for row in registry["approaches"]]
    active = [row["family_id"] for row in registry["approaches"]
              if row["state"] == "active_for_bound"]
    blocked = [row["family_id"] for row in registry["approaches"]
               if row["state"].startswith("blocked")]
    record("registry_has_no_promotable_family", registry["registry_summary"]["promotable"] == 0,
           {"states": sorted(set(states))})
    record("paper3_not_authorized", registry["registry_summary"]["paper3_authorized"] is False,
           registry["registry_summary"])
    record("unresolved_families_are_explicit", bool(active or blocked),
           {"active_for_bound": active, "blocked": blocked})

    failed = [check for check in checks if not check["passed"]]
    metric_rescore_hold = (
        metric_audit["paper2_authorization"]["current_scientific_state"]
        == "HOLD_CANONICAL_V2_RESCORE_AND_DOMAIN_CONFIRMATION_REQUIRED"
        and metric_audit["v2_release_gates"]["identical_tape_rescore_h_j_mtr"]
        == "REQUIRED"
    )
    terminal_boundary = (
        not failed
        and not metric_rescore_hold
        and not active
        and not blocked
        and boundary_proof_validation["all_families_terminal_b_eligible"]
    )
    if failed:
        status = "FAIL_EVIDENCE_INCONSISTENT"
    elif metric_rescore_hold:
        status = "HOLD_CANONICAL_V2_RESCORE_AND_DOMAIN_CONFIRMATION_REQUIRED"
    elif active:
        status = "OPEN_ACTIVE_BOUND_REQUIRED"
    elif blocked:
        status = "BOUNDARY_CONDITIONAL_ON_DOMAIN_FACTS"
    elif not boundary_proof_validation["all_families_terminal_b_eligible"]:
        status = "NONTERMINAL_FAMILY_CEILINGS_INCOMPLETE"
    else:
        status = "TERMINAL_FINITE_ENVELOPE_BOUNDARY"

    result = {
        "schema_version": "paper2_boundary_verification_v1",
        "repository_head": git("rev-parse", "HEAD"),
        "status": status,
        "terminal_boundary": terminal_boundary,
        "paper2_confirmed": False,
        "paper2_null_confirmed": False,
        "paper2_ceiling_confirmed": False,
        "paper3_authorized": False,
        "canonical_metric_contract": CANONICAL_METRIC_CONTRACT,
        "metric_rescore_hold": metric_rescore_hold,
        "checks_passed": len(checks) - len(failed),
        "checks_total": len(checks),
        "active_for_bound": active,
        "blocked_domain_or_claim_families": blocked,
        "family_proof_validation": boundary_proof_validation,
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
