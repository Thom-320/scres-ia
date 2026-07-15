#!/usr/bin/env python3
"""Independently audit a retrieved Garrido risk-sensitivity custody package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_garrido_risk_headroom_sensitivity import (
    build_profiles,
    summarize_group,
)
from scripts.run_track_a_headroom_search import continuous_candidates


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksum_manifest(run_dir: Path) -> list[str]:
    failures: list[str] = []
    manifest = run_dir / "custody" / "remote_files.sha256"
    if not manifest.is_file():
        return ["missing_remote_files_sha256"]
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        path = run_dir / relative
        if not path.is_file():
            failures.append(f"missing:{relative}")
        elif sha256(path) != expected:
            failures.append(f"hash_mismatch:{relative}")
    return failures


def approximately_equal(left: Any, right: Any, *, path: str = "") -> list[str]:
    failures: list[str] = []
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            failures.append(f"keys:{path}:{sorted(set(left) ^ set(right))}")
            return failures
        for key in left:
            failures.extend(
                approximately_equal(left[key], right[key], path=f"{path}/{key}")
            )
        return failures
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return [f"length:{path}:{len(left)}!={len(right)}"]
        for index, (a, b) in enumerate(zip(left, right)):
            failures.extend(approximately_equal(a, b, path=f"{path}/{index}"))
        return failures
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if not math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12):
            failures.append(f"value:{path}:{left}!={right}")
        return failures
    if left != right:
        failures.append(f"value:{path}:{left!r}!={right!r}")
    return failures


def audit(
    run_dir: Path,
    *,
    contract_path: Path,
    expected_source_commit: str,
) -> dict[str, Any]:
    failures = verify_checksum_manifest(run_dir)
    custody = run_dir / "custody"
    artifacts = run_dir / "artifacts" / "development"
    required = [
        custody / "producer_control.json",
        custody / "producer_exit.json",
        custody / "watcher_state.json",
        artifacts / "result.json",
        artifacts / "raw_rows.csv",
        artifacts / "profile_summary.csv",
        artifacts / "progress.json",
    ]
    failures.extend(f"missing:{path.relative_to(run_dir)}" for path in required if not path.is_file())
    if failures:
        return {"status": "FAIL_GARRIDO_RISK_AUDIT", "failures": failures}

    control = json.loads((custody / "producer_control.json").read_text())
    exit_state = json.loads((custody / "producer_exit.json").read_text())
    watcher = json.loads((custody / "watcher_state.json").read_text())
    result = json.loads((artifacts / "result.json").read_text())
    contract = json.loads(contract_path.read_text())
    if control.get("source_commit") != expected_source_commit:
        failures.append("source_commit_mismatch")
    if int(exit_state.get("returncode", 1)) != 0 or not exit_state.get("result_exists"):
        failures.append("producer_exit_not_successful")
    if watcher.get("status") != "COMPLETE_PENDING_RETRIEVAL":
        failures.append("watcher_not_terminal_complete")
    if result.get("contract_sha256") != sha256(contract_path):
        failures.append("contract_hash_mismatch")
    if result.get("metric") != "ret_excel_request_snapshot_v2":
        failures.append("canonical_metric_mismatch")
    if result.get("black_swan_R3_scaled") is not False:
        failures.append("R3_scaled_or_unreported")
    if any(bool(result.get("claim_boundary", {}).get(key)) for key in (
        "H_PI_established", "H_obs_established", "learner_authorized",
        "paper2_confirmed", "paper3_authorized",
    )):
        failures.append("claim_boundary_overpromotion")

    with (artifacts / "raw_rows.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    profiles = build_profiles()
    candidates = continuous_candidates(
        contract["policy_frontier"]["buffer_fractions_of_I1344"],
        contract["policy_frontier"]["shift_levels"],
    )
    seeds = list(contract["development"]["seeds"])
    expected_count = len(profiles) * len(candidates) * len(seeds)
    keys = [(row["profile"], row["candidate"], int(row["seed"])) for row in rows]
    if len(rows) != expected_count or len(set(keys)) != expected_count:
        failures.append(f"raw_matrix_incomplete_or_duplicate:{len(rows)}/{expected_count}")
    expected_keys = {
        (profile["id"], candidate.label, int(seed))
        for profile in profiles for candidate in candidates for seed in seeds
    }
    if set(keys) != expected_keys:
        failures.append("raw_matrix_key_set_mismatch")
    if any("R3" in row["enabled_risks"].split(",") for row in rows):
        failures.append("R3_enabled_in_raw_matrix")
    if any("R3" in row["impact_multipliers"] for row in rows):
        failures.append("R3_impact_in_raw_matrix")
    numeric_fields = (
        "ret_excel", "ration_ret_excel", "ret_excel_cvar10", "lost_orders",
        "backorder_qty_final", "backlog_age_max", "service_loss_auc_ration_hours",
        "resource", "risk_events",
    )
    if any(not math.isfinite(float(row[field])) for row in rows for field in numeric_fields):
        failures.append("nonfinite_raw_metric")

    profile_map = {profile["id"]: profile for profile in profiles}
    for row in rows:
        profile = profile_map[row["profile"]]
        if json.loads(row["risk_overrides"]) != profile["overrides"]:
            failures.append(f"risk_override_mismatch:{row['profile']}")
            break
        if json.loads(row["impact_multipliers"]) != profile["impact"]:
            failures.append(f"impact_mismatch:{row['profile']}")
            break

    recomputed: dict[str, list[dict[str, Any]]] = {}
    groups = sorted({profile["group"] for profile in profiles})
    for group_index, group in enumerate(groups):
        members = [profile["id"] for profile in profiles if profile["group"] == group]
        summaries = []
        for budget_index, budget in enumerate(contract["policy_frontier"]["resource_budget_caps"]):
            summary = summarize_group(
                rows,
                members,
                candidates,
                seeds,
                float(budget),
                bootstrap_seed=2026071500 + 100 * group_index + budget_index,
            )
            if summary is not None:
                summaries.append(summary)
        recomputed[group] = summaries
    failures.extend(
        f"summary_recompute:{failure}"
        for failure in approximately_equal(
            recomputed, result.get("group_budget_summaries", {}), path="group_budget_summaries"
        )[:50]
    )
    passing = [
        {"group": group, **summary}
        for group, summaries in recomputed.items() for summary in summaries
        if summary["door_pass"]
    ]
    expected_status = (
        "DEVELOPMENT_DOOR_FOUND" if passing
        else "DEVELOPMENT_NO_DOOR_UNDER_TESTED_FRONTIER"
    )
    if result.get("status") != expected_status:
        failures.append("status_not_recomputed")
    failures.extend(
        f"passing_doors:{failure}"
        for failure in approximately_equal(
            passing, result.get("passing_doors", []), path="passing_doors"
        )[:50]
    )
    progress = json.loads((artifacts / "progress.json").read_text())
    if int(progress.get("completed", -1)) != expected_count:
        failures.append("progress_not_complete")

    return {
        "schema_version": "garrido_risk_headroom_independent_audit_v1",
        "status": "PASS_GARRIDO_RISK_AUDIT" if not failures else "FAIL_GARRIDO_RISK_AUDIT",
        "failures": failures,
        "source_commit": control.get("source_commit"),
        "result_sha256": sha256(artifacts / "result.json"),
        "raw_rows_sha256": sha256(artifacts / "raw_rows.csv"),
        "n_rows": len(rows),
        "expected_rows": expected_count,
        "recomputed_status": expected_status,
        "recomputed_passing_door_count": len(passing),
        "R3_scaled": False,
        "claim_boundary_verified": not any("claim_boundary" in item for item in failures),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(
        args.run_dir.resolve(),
        contract_path=args.contract.resolve(),
        expected_source_commit=args.expected_source_commit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS_GARRIDO_RISK_AUDIT" else 1


if __name__ == "__main__":
    raise SystemExit(main())
