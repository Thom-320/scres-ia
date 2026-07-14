#!/usr/bin/env python3
"""Content-address and schema-check every Phase-0 failure-taxonomy row."""
from __future__ import annotations

import argparse
from datetime import date
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
SEARCH = ROOT / "research" / "paper2_exhaustive_search"
REQUIRED_FIELDS = {
    "family_id",
    "primary_metric",
    "physical_mechanism",
    "decision_rights",
    "physical_liveness",
    "action_ranking_diversity",
    "h_pi",
    "h_obs",
    "learned_value",
    "retained_value",
    "strongest_comparator",
    "resource_matching",
    "guardrail_result",
    "final_verdict",
    "failure_class",
    "exact_failure",
    "evidence",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(taxonomy: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    rows = taxonomy.get("decision_families", [])
    family_ids = [row.get("family_id") for row in rows]
    if len(rows) != 17 or len(set(family_ids)) != len(rows):
        failures.append("taxonomy must contain 17 unique family ids")
    evidence_rows = []
    hashes_checked = 0
    for row in rows:
        family_id = row.get("family_id")
        missing = sorted(REQUIRED_FIELDS - set(row))
        if missing:
            failures.append(f"{family_id}: missing fields {missing}")
        if not str(row.get("primary_metric", "")).startswith("ret_excel_visible_v1"):
            failures.append(f"{family_id}: governing primary metric is not visible-v1")
        if not row.get("exact_failure"):
            failures.append(f"{family_id}: exact failure is empty")
        artifacts = []
        for relative in row.get("evidence", []):
            path = ROOT / relative
            if not path.is_file():
                failures.append(f"{family_id}: missing evidence {relative}")
                artifacts.append({"path": relative, "sha256": None})
            else:
                digest = sha256(path)
                hashes_checked += 1
                artifacts.append({"path": relative, "sha256": digest})
        if not artifacts:
            failures.append(f"{family_id}: no evidence artifacts")
        evidence_rows.append({"family_id": family_id, "artifacts": artifacts})
    return {
        "passed": not failures,
        "failures": failures,
        "family_count": len(rows),
        "evidence_hashes_checked": hashes_checked,
        "families": evidence_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=SEARCH / "phase0_failure_taxonomy_validation.json",
    )
    args = parser.parse_args()
    taxonomy_path = SEARCH / "phase0_failure_taxonomy.json"
    taxonomy = json.loads(taxonomy_path.read_text())
    validation = validate(taxonomy)
    result = {
        "schema_version": "paper2_phase0_failure_taxonomy_validation_v1",
        "generated_date": date.today().isoformat(),
        "taxonomy_path": str(taxonomy_path.relative_to(ROOT)),
        "taxonomy_sha256": sha256(taxonomy_path),
        **validation,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
