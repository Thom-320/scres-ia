#!/usr/bin/env python3
"""Independent fail-closed audit of Program Q confirmation inputs before 749 opens."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.audit_program_q_seed_custody import scan
from scripts.evaluate_program_q_replication import (
    ADJUDICATOR,
    CONTRACT,
    DIRECT_AUDIT,
    FREEZE,
    LAUNCHER,
    POWER_VERDICT,
    RUNNER,
    SEED_AUDIT,
    SMOKE_MANIFEST,
    SMOKE_REPORT,
    SMOKE_ROOT,
    SMOKE_SCRIPT,
    WATCHER,
    verify_model_hashes,
)
from supply_chain.program_o_eval_custody import sha256, verify_sha256_manifest


EVALUATOR = ROOT / "scripts/evaluate_program_q_replication.py"


def audit(*, models: Path, plan: Path) -> dict:
    failures: list[str] = []
    contract = json.loads(CONTRACT.read_text())
    execution_plan = json.loads(plan.read_text())
    seed_scan = scan(ROOT)
    try:
        model_hashes = verify_model_hashes(models)
    except RuntimeError as error:
        failures.append(str(error))
        model_hashes = {}
    expected = {
        "contract_sha256": sha256(CONTRACT),
        "evaluator_sha256": sha256(EVALUATOR),
        "candidate_freeze_sha256": sha256(FREEZE),
        "power_verdict_sha256": sha256(POWER_VERDICT),
        "seed_audit_sha256": sha256(SEED_AUDIT),
        "direct_audit_sha256": sha256(DIRECT_AUDIT),
        "adjudicator_sha256": sha256(ADJUDICATOR),
        "runner_sha256": sha256(RUNNER),
        "launcher_sha256": sha256(LAUNCHER),
        "watcher_sha256": sha256(WATCHER),
        "smoke_script_sha256": sha256(SMOKE_SCRIPT),
        "smoke_report_sha256": sha256(SMOKE_REPORT),
        "smoke_manifest_sha256": sha256(SMOKE_MANIFEST),
    }
    for key, value in expected.items():
        if execution_plan.get(key) != value:
            failures.append(f"plan_{key}_mismatch")
    if contract.get("confirmation", {}).get("opened") is not False:
        failures.append("contract_does_not_record_confirmation_closed")
    if contract.get("confirmation", {}).get("N") != 256:
        failures.append("contract_N_is_not_256")
    if seed_scan.get("status") != "PROGRAM_Q_SEEDS_VIRGIN":
        failures.append("reserved_seed_scan_failed")
    if execution_plan.get("expected_shards") != 768:
        failures.append("expected_shard_count_is_not_768")
    if execution_plan.get("external_collaborator_dependency") is not False:
        failures.append("external_collaborator_dependency_present")
    try:
        smoke_manifest = verify_sha256_manifest(SMOKE_ROOT, SMOKE_MANIFEST)
        smoke = json.loads(SMOKE_REPORT.read_text())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        failures.append(f"development_smoke_custody_failed: {error}")
    else:
        if len(smoke_manifest) != 7 or "report.json" not in smoke_manifest:
            failures.append("development_smoke_manifest_is_not_complete")
        if smoke.get("status") != "PASS_NONPROMOTABLE_END_TO_END_SMOKE":
            failures.append("development_smoke_did_not_pass")
        if smoke.get("scientific_749_seeds_opened") is not False:
            failures.append("development_smoke_touched_scientific_seeds")
        if smoke.get("reduced_design_adjudication") != "STOP_Q_NO_REPLICATED_LEARNED_ADAPTATION":
            failures.append("development_smoke_did_not_fail_closed_at_adjudication")
    return {
        "schema_version": "program_q_confirmation_preopen_audit_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": not failures,
        "status": (
            "PASS_CONFIRMATION_IMPLEMENTATION_PREOPEN_AUDIT"
            if not failures
            else "STOP_CONFIRMATION_IMPLEMENTATION_PREOPEN_AUDIT"
        ),
        "source_hashes": expected,
        "model_hashes": model_hashes,
        "seed_scan_status": seed_scan.get("status"),
        "scientific_seeds_opened": False,
        "failures": failures,
        "authorization_boundary": (
            "This audit does not authorize opening 749. A separately hashed independent "
            "authorization is still mandatory."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    payload = audit(models=args.models, plan=args.plan)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    raise SystemExit(0 if payload["passed"] else 1)


if __name__ == "__main__":
    main()
