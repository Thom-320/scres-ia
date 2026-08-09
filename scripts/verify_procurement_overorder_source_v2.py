#!/usr/bin/env python3
"""Independent source-only audit of the Op2 190,000-unit interpretation.

The canonical v1 artifact already records D5 as ``MATCHES-PUBLISHED`` and
includes the Table 6.20/S=3 evidence.  This companion audit independently
checks two bounded text sections: the Op2 operation enumerates rm1 through
rm12, and Table 6.20 says ``190,000 units of each rm`` for S=1, S=2 and S=3.

This is a diagnostic text-and-arithmetic audit.  It opens no seed, runs no
simulation and makes no learner or headroom claim.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = ROOT / "thesis.txt"
DEFAULT_OUTPUT = ROOT / "results/procurement_overorder_source_v2/result.json"
REFERENCE = ROOT / "results/procurement_overorder_source/result.json"
DEFAULT_CONTRACT = ROOT / "docs/CORRECCION_FUENTE_OP2_190K_2026-08-09.md"
HOURS_PER_WEEK = 168.0
RATIONS_PER_DAY = 2_500.0
DAYS_PER_WEEK = 6
NUM_RAW_MATERIALS = 12
OP2_QUANTITY_PER_RM_MONTH = 190_000.0
OP2_REORDER_PERIOD_HOURS = 672.0
OP3_QUANTITY_PER_RM_WEEK_S1 = 15_500.0


def _compact(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def _between(text: str, start_marker: str, end_marker: str) -> str:
    start = text.find(start_marker)
    if start < 0:
        return ""
    end = text.find(end_marker, start + len(start_marker))
    if end < 0:
        return ""
    return text[start:end]


def _check(
    passed: bool,
    *,
    observed: Any,
    threshold: Any,
    why_it_can_fail: str,
    evidence_class: str,
    textual_status_gate: bool = False,
) -> dict[str, Any]:
    return {
        "computed": True,
        "passed": bool(passed),
        "evidence_class": evidence_class,
        "textual_status_gate": textual_status_gate,
        "evidence": {
            "computed_from": {"observed": observed, "threshold": threshold},
            "why_it_can_fail": why_it_can_fail,
        },
    }


def _disclosure(statement: str, evidence: dict[str, Any]) -> dict[str, Any]:
    return {
        "computed": False,
        "passed": None,
        "disclosure": True,
        "evidence": {"statement": statement, "evidence": evidence},
    }


def _summarise(checks: dict[str, dict[str, Any]]) -> dict[str, Any]:
    computed = {name: row for name, row in checks.items() if row.get("computed")}
    failed = [name for name, row in computed.items() if not row.get("passed")]
    disclosures = [name for name, row in checks.items() if row.get("disclosure")]
    return {
        "all_passed": not failed,
        "n_computed": len(computed),
        "n_failed": len(failed),
        "failed": failed,
        "n_disclosures": len(disclosures),
        "disclosures": disclosures,
    }


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_sealed(
    payload: dict[str, Any],
    output: Path,
    *,
    source: Path,
    reference: Path,
    contract: Path,
) -> None:
    sealed = dict(payload)
    sealed["source_path"] = str(source.relative_to(ROOT))
    sealed["source_sha256"] = _sha256_bytes(source.read_bytes())
    sealed["reference_path"] = str(reference.relative_to(ROOT))
    sealed["reference_sha256"] = _sha256_bytes(reference.read_bytes())
    sealed["contract_path"] = str(contract.relative_to(ROOT))
    sealed["contract_sha256"] = _sha256_bytes(contract.read_bytes())
    canonical = json.dumps(
        sealed, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    sealed["self_sha256"] = _sha256_bytes(canonical)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(sealed, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def build_payload(thesis_text: str) -> dict[str, Any]:
    compact = _compact(thesis_text)
    operation_excerpt = _between(
        compact,
        "Operation 2 (Op2, j):",
        "Operation 3 (Op3, j):",
    )
    operation_enumerates_all = bool(
        re.search(
            r"Q = \{190,000 rm1.{0,80}?190,000(?: \d{1,3})? rm12\}",
            operation_excerpt,
        )
    )

    table_heading = "Table 6.20 Short-term manufacturing capacity (S) of the MFSC"
    table_start = compact.rfind(table_heading)
    table_end = compact.find(
        "Table 6.21 Design matrix with short-term manufacturing capacity",
        table_start + len(table_heading),
    )
    table_excerpt = (
        compact[table_start:table_end]
        if table_start >= 0 and table_end > table_start
        else ""
    )
    op2_each_rm_occurrences = table_excerpt.count("190,000 units of each rm")
    op3_s_levels_present = all(
        phrase in table_excerpt
        for phrase in (
            "15,500 units of each rm",
            "31,000 units of each rm",
            "47,000 units of each rm",
        )
    )

    op2_q = OP2_QUANTITY_PER_RM_MONTH
    op2_rop_hours = OP2_REORDER_PERIOD_HOURS
    op3_q_s1 = OP3_QUANTITY_PER_RM_WEEK_S1
    op2_reorder_period_weeks = op2_rop_hours / HOURS_PER_WEEK
    op2_per_rm_week = op2_q / op2_reorder_period_weeks
    demand_components_week = (
        RATIONS_PER_DAY * DAYS_PER_WEEK * NUM_RAW_MATERIALS
    )
    procurement_components_week = op2_per_rm_week * NUM_RAW_MATERIALS
    distribution_components_week_s1 = op3_q_s1 * NUM_RAW_MATERIALS
    source_s3_components_week = 47_000.0 * NUM_RAW_MATERIALS

    checks = {
        "f1_operation_2_enumerates_rm1_through_rm12": _check(
            operation_enumerates_all is True,
            observed=operation_enumerates_all,
            threshold=True,
            evidence_class="DIRECT_TEXTUAL_EVIDENCE",
            textual_status_gate=True,
            why_it_can_fail=(
                "if the operation description did not enumerate rm1 through rm12, "
                "the prose "
                "could still support a total-across-materials interpretation"
            ),
        ),
        "f2_table_6_20_repeats_each_rm_for_all_three_shift_levels": _check(
            op2_each_rm_occurrences >= 3,
            observed=op2_each_rm_occurrences,
            threshold=3,
            evidence_class="DIRECT_TEXTUAL_EVIDENCE",
            textual_status_gate=True,
            why_it_can_fail=(
                "if Table 6.20 did not explicitly say 'each rm' at S=1, S=2 and S=3, "
                "the interpretation would still depend on the operation prose alone"
            ),
        ),
        "f3_table_6_20_contains_the_shift_scaled_op3_levels": _check(
            op3_s_levels_present is True,
            observed=op3_s_levels_present,
            threshold=True,
            evidence_class="DERIVED_CONTEXT__NOT_TEXTUAL_STATUS_GATE",
            why_it_can_fail=(
                "if the table did not show 15,500/31,000/47,000 for Op3, the "
                "design inference that Op2 is held at an S=3-sized ceiling would "
                "lack its contextual premise; the per-rm textual status is unaffected"
            ),
        ),
        "f4_published_op2_exceeds_mean_s1_demand_threefold": _check(
            procurement_components_week / demand_components_week >= 3.0,
            observed=procurement_components_week / demand_components_week,
            threshold=3.0,
            evidence_class="DERIVED_ARITHMETIC__NOT_TEXTUAL_STATUS_GATE",
            why_it_can_fail=(
                "if the derived ratio were below three, the threefold-flow claim "
                "would fail; the per-rm textual status is unaffected"
            ),
        ),
        "f5_op2_weekly_rate_is_close_to_the_published_s3_flow": _check(
            abs(op2_per_rm_week - 47_000.0) / 47_000.0 < 0.02,
            observed=abs(op2_per_rm_week - 47_000.0) / 47_000.0,
            threshold=0.02,
            evidence_class="DERIVED_ARITHMETIC__NOT_TEXTUAL_STATUS_GATE",
            why_it_can_fail=(
                "if the monthly Op2 rate were not close to the S=3 weekly Op3 "
                "rate, the numerical premise for the capacity-ceiling design "
                "inference would fail; the per-rm textual status is unaffected"
            ),
        ),
    }
    checks["d1_relationship_to_canonical_v1"] = _disclosure(
        "This source-only audit is complementary to and independent of canonical "
        "v1.  It does not supersede v1, rerun its simulation, or adjudicate its "
        "simulation claims.",
        evidence={
            "canonical_v1_d5_status": "MATCHES-PUBLISHED",
            "canonical_v1_result": str(REFERENCE.relative_to(ROOT)),
            "companion_audit_addition": (
                "bounded independent checks of the Op2 and Table 6.20 text"
            ),
        },
    )
    summary = _summarise(checks)
    textual_checks = tuple(
        name
        for name, check in checks.items()
        if check.get("textual_status_gate") is True
    )
    source_explicit = all(checks[name]["passed"] for name in textual_checks)

    return {
        "schema_version": "procurement_overorder_source_v2",
        "claim_status": (
            "SOURCE_EXPLICIT_PER_RAW_MATERIAL__INDEPENDENTLY_CONFIRMED"
            if source_explicit
            else "SOURCE_TEXT_NOT_INDEPENDENTLY_CONFIRMED"
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_role": "DIAGNOSTIC_SOURCE_AUDIT",
        "scope": (
            "DIAGNOSTIC_SOURCE_TEXT_AND_ARITHMETIC_NO_SEEDS_NO_SIMULATION_NO_LEARNER"
        ),
        "source_locations": {
            "operation_2": "Garrido-Rios thesis, printed page 84",
            "capacity_table": "Garrido-Rios thesis, Table 6.20, printed page 108",
        },
        "source_evidence": {
            "operation_2_enumerates_rm1_through_rm12": operation_enumerates_all,
            "table_6_20_op2_each_rm_occurrences": op2_each_rm_occurrences,
            "table_6_20_op3_shift_levels_present": op3_s_levels_present,
            "textual_evidence_all_passed": source_explicit,
            "textual_status_depends_only_on": list(textual_checks),
        },
        "published_parameters": {
            "op2_quantity_per_rm_month": op2_q,
            "op2_reorder_period_hours": op2_rop_hours,
            "op3_quantity_per_rm_week_s1": op3_q_s1,
            "op3_quantity_per_rm_week_s2": 31_000.0,
            "op3_quantity_per_rm_week_s3": 47_000.0,
            "raw_material_count": NUM_RAW_MATERIALS,
            "mean_rations_per_day": RATIONS_PER_DAY,
            "days_per_week": DAYS_PER_WEEK,
        },
        "derived": {
            "op2_reorder_period_weeks": op2_reorder_period_weeks,
            "op2_per_rm_week": op2_per_rm_week,
            "mean_s1_demand_components_week": demand_components_week,
            "op2_procurement_components_week": procurement_components_week,
            "op3_distribution_components_week_s1": distribution_components_week_s1,
            "op3_distribution_components_week_s3": source_s3_components_week,
            "op2_over_mean_s1_demand": (
                procurement_components_week / demand_components_week
            ),
            "op3_s1_over_mean_s1_demand": (
                distribution_components_week_s1 / demand_components_week
            ),
            "op2_per_rm_week_over_op3_s3": op2_per_rm_week / 47_000.0,
        },
        "interpretation": {
            "source_fact": (
                "Op2 supplies 190,000 units of every one of the 12 raw materials "
                "every 672 hours."
            ),
            "design_inference": (
                "Holding Op2 at 47,500 units per raw material per week while Op3 "
                "scales from 15,500 to 47,000 is consistent with procurement being "
                "sized to the S=3 experiment ceiling; this is an inference, not an "
                "author quotation."
            ),
            "extension_boundary": (
                "Reducing Op2 enough to make supplier yield binding is permissible "
                "only as a researcher-declared extension with a measured "
                "fidelity price."
            ),
        },
        "relationship_to_canonical_v1": {
            "artifact": str(REFERENCE.relative_to(ROOT)),
            "relationship": "COMPLEMENTARY_INDEPENDENT_SOURCE_ONLY_AUDIT",
            "canonical_v1_d5_status": "MATCHES-PUBLISHED",
            "does_not_supersede_v1": True,
            "does_not_rerun_v1_simulation": True,
        },
        "falsifiers": checks,
        "falsifier_summary": summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    args = parser.parse_args()
    payload = build_payload(args.source.read_text(encoding="utf-8"))
    _write_sealed(
        payload,
        args.output,
        source=args.source,
        reference=REFERENCE,
        contract=args.contract,
    )
    print(payload["claim_status"])
    print(
        f"Op2={payload['derived']['op2_procurement_components_week']:,.0f} "
        f"components/week; "
        f"ratio={payload['derived']['op2_over_mean_s1_demand']:.4f}"
    )
    print(
        f"falsifiers={payload['falsifier_summary']['n_computed']}; "
        f"failed={payload['falsifier_summary']['n_failed']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
