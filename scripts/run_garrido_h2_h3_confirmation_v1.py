#!/usr/bin/env python3
"""Run frozen Garrido H2/H3 confirmation shards on fresh paired tapes."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_garrido_h2_h3_corrective_v1 import (
    canonical_sha256,
    file_sha256,
    git_commit,
    require_clean_worktree,
    run_configuration,
    trace_preflight,
    verify_sources,
    workbook_audit,
)

DEFAULT_CONTRACT = ROOT / "contracts/garrido_h2_h3_confirmation_v1.json"
DEFAULT_FREEZE = (
    ROOT / "contracts/garrido_h2_h3_confirmation_v1_freeze_receipt.json"
)


def load_authority(
    contract_path: Path, freeze_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    contract = json.loads(contract_path.read_text())
    receipt = json.loads(freeze_path.read_text())
    if contract.get("contract_id") != "garrido_h2_h3_confirmation_v1":
        raise RuntimeError("STOP_CONFIRMATION_AUTHORITY:contract")
    if contract.get("status") != "DRAFT_PROSPECTIVE_CONFIRMATION_UNOPENED":
        raise RuntimeError("STOP_CONFIRMATION_AUTHORITY:status")
    if receipt.get("status") != "FROZEN_PROSPECTIVE_CONFIRMATION_UNOPENED":
        raise RuntimeError("STOP_CONFIRMATION_AUTHORITY:freeze")
    if receipt.get("contract_sha256") != file_sha256(contract_path):
        raise RuntimeError("STOP_CONFIRMATION_AUTHORITY:hash")
    if receipt.get("runner_sha256") != file_sha256(Path(__file__)):
        raise RuntimeError("STOP_CONFIRMATION_AUTHORITY:runner")
    if receipt.get("confirmation_roots_opened") is not False:
        raise RuntimeError("STOP_CONFIRMATION_AUTHORITY:already-open")
    parent = contract["parent_development"]
    parent_path = ROOT / parent["result"]
    if file_sha256(parent_path) != parent["result_sha256"]:
        raise RuntimeError("STOP_CONFIRMATION_AUTHORITY:parent-result")
    development = set(map(int, parent["development_tape_roots"]))
    confirmation = list(
        map(int, contract["execution"]["confirmation_tape_roots"])
    )
    if len(confirmation) != len(set(confirmation)) or development & set(confirmation):
        raise RuntimeError("STOP_CONFIRMATION_DUPLICATE_OR_DEVELOPMENT_ROOT")
    return contract, receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--freeze-receipt", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tape-indices", type=int, nargs="+", required=True)
    args = parser.parse_args()

    require_clean_worktree()
    if args.output_dir.exists():
        raise RuntimeError("STOP_CONFIRMATION_OUTPUT_DIR_EXISTS")
    contract, freeze = load_authority(args.contract, args.freeze_receipt)
    roots_all = list(
        map(int, contract["execution"]["confirmation_tape_roots"])
    )
    indices = list(map(int, args.tape_indices))
    if (
        len(indices) != len(set(indices))
        or not indices
        or min(indices) < 0
        or max(indices) >= len(roots_all)
    ):
        raise RuntimeError("STOP_CONFIRMATION_TAPE_INDEX")
    roots = [roots_all[index] for index in indices]
    args.output_dir.mkdir(parents=True)
    source_hashes = verify_sources(contract, args.source_dir)
    contract_hash = file_sha256(args.contract)
    commit = git_commit()
    opening = {
        "status": "OPENED_CONFIRMATION",
        "opened_at": datetime.now(timezone.utc).isoformat(),
        "contract_sha256": contract_hash,
        "freeze_receipt_sha256": file_sha256(args.freeze_receipt),
        "code_commit": commit,
        "source_hashes": source_hashes,
        "tape_indices": indices,
        "confirmation_tape_roots": roots,
        "development_roots_opened": False,
        "confirmation_roots_opened": True,
    }
    (args.output_dir / "opening_receipt.json").write_text(
        json.dumps(opening, indent=2, sort_keys=True) + "\n"
    )
    source_audit = workbook_audit(args.source_dir)
    (args.output_dir / "source_workbook_audit.json").write_text(
        json.dumps(source_audit, indent=2, sort_keys=True) + "\n"
    )
    trace = trace_preflight(contract)
    (args.output_dir / "table_6_20_trace_preflight.json").write_text(
        json.dumps(trace, indent=2, sort_keys=True) + "\n"
    )

    started = time.perf_counter()
    rows_path = args.output_dir / "rows.jsonl"
    with rows_path.open("x") as stream:
        for tape_root in roots:
            for config_index in range(1, 91):
                row = run_configuration(
                    contract,
                    config_index=config_index,
                    tape_root=tape_root,
                )
                stream.write(
                    json.dumps(row, sort_keys=True, separators=(",", ":"))
                    + "\n"
                )
                stream.flush()
            print(
                f"completed confirmation_tape_root={tape_root}",
                flush=True,
            )
    rows = [
        json.loads(line)
        for line in rows_path.read_text().splitlines()
        if line.strip()
    ]
    expected = 90 * len(roots)
    identities = {(row["tape_root"], row["cf"]) for row in rows}
    if len(rows) != expected or len(identities) != expected:
        raise RuntimeError("STOP_CONFIRMATION_INCOMPLETE_MATRIX")
    completion = {
        "status": "COMPLETE_VALID_CONFIRMATION_SHARD",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "contract_sha256": contract_hash,
        "freeze_receipt_sha256": file_sha256(args.freeze_receipt),
        "code_commit": commit,
        "row_count": len(rows),
        "rows_sha256": file_sha256(rows_path),
        "row_identity_digest": canonical_sha256(sorted(identities)),
        "confirmation_tape_roots": roots,
        "development_roots_opened": False,
        "confirmation_roots_opened": True,
        "authority_review_commit": freeze["review_commit"],
    }
    (args.output_dir / "completion_receipt.json").write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(completion, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
