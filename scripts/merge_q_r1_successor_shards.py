#!/usr/bin/env python3
"""Merge only the exact frozen Q-R1 successor shards."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from scripts.run_q_r1_successor_abc import contract_identity, summarize


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def merge_payloads(contract: dict, payloads: list[dict]) -> dict:
    expected_shards = [row["history_roots"] for row in contract["shards"]]
    observed_shards = [payload["history_roots"] for payload in payloads]
    if sorted(observed_shards) != sorted(expected_shards):
        raise ValueError("shards do not exactly cover the frozen plan")
    expected_contract_sha = contract_identity(contract)
    if contract.get("contract_identity_sha256") != expected_contract_sha:
        raise ValueError("frozen contract identity digest mismatch")
    if any(payload.get("contract_sha256") != expected_contract_sha for payload in payloads):
        raise ValueError("shard contract digest mismatch")
    expected_planner = contract["selected_universal_planner"]["config_id"]
    if any(payload.get("planner") != expected_planner for payload in payloads):
        raise ValueError("shard planner mismatch")
    commits = {payload["runtime"]["commit"] for payload in payloads}
    if len(commits) != 1:
        raise ValueError("shards were not executed from one commit")
    rows = [row for payload in payloads for row in payload["rows"]]
    keys = {
        (
            row["estimand"], row["kappa"], row["history_root"],
            row["campaign_index"], row["arm"],
        )
        for row in rows
    }
    if len(keys) != len(rows):
        raise ValueError("duplicate campaign-arm rows across shards")
    observed_roots = sorted({int(row["history_root"]) for row in rows})
    expected_roots = list(range(contract["history_roots"][0], contract["history_roots"][1] + 1))
    if observed_roots != expected_roots:
        raise ValueError("merged rows do not contain every frozen history root")
    return {
        "schema_version": "q_r1_successor_abc_merged_v1",
        "claim_status": "PROSPECTIVE_CONFIRMATION",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "history_roots": contract["history_roots"],
        "histories": len(expected_roots),
        "campaigns_per_history": contract["campaigns_per_history"],
        "planner": expected_planner,
        "contract_sha256": expected_contract_sha,
        "execution_commit": commits.pop(),
        "source_shards": observed_shards,
        "summary": summarize(rows),
        "rows": rows,
        "elapsed_seconds_sum": sum(float(payload["elapsed_seconds"]) for payload in payloads),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--shard", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    contract = json.loads(args.contract.read_text())
    payloads = [json.loads(path.read_text()) for path in args.shard]
    merged = merge_payloads(contract, payloads)
    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n")
    receipt = {
        "result": str(result_path),
        "result_sha256": sha256(result_path),
        "shard_sha256": {str(path): sha256(path) for path in args.shard},
        "complete_frozen_coverage": True,
    }
    (output_dir / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"result": str(result_path), "rows": len(merged["rows"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
