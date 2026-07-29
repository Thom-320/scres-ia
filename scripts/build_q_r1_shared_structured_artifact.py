#!/usr/bin/env python3
"""Build the one immutable structured-comparator artifact for Q-R1 factorial v4.

The structured comparator depends on the frozen physical histories, retained
prior path, and comparator configuration. It does not depend on the neural
configuration, optimizer seed, checkpoint, or learned action. This instrument
computes those rows once and binds their complete identity to a receipt before
any worker may grade a checkpoint.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_q_r1_matched_retention_factorial_v4 import (  # noqa: E402
    COMPARATOR_FREEZE_PATH,
    CONTRACT_PATH,
    KAPPAS,
    STRUCTURED_AMENDMENT_FREEZE_PATH,
    STRUCTURED_AMENDMENT_PATH,
    build_histories,
    evaluate_structured,
    integer_range,
    json_sha256,
    load_authority,
    load_shared_structured_authority,
    runtime_receipt,
    sha256,
    validate_shared_structured_bar,
    write_json,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite {args.output_dir}")

    contract, base_freeze = load_authority("development-worker")
    amendment, amendment_freeze = load_shared_structured_authority()
    runtime = runtime_receipt()
    if runtime["git_status_porcelain"]:
        raise RuntimeError("builder requires a clean worktree before output creation")

    roots = integer_range(
        contract["data_splits"]["checkpoint_selection_history_roots"]
    )
    args.output_dir.mkdir(parents=True)
    opening = {
        "schema_version": "q_r1_shared_structured_opening_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_contract_sha256": sha256(CONTRACT_PATH),
        "base_freeze_receipt_sha256": sha256(
            ROOT / "contracts/q_r1_matched_retention_factorial_v4_freeze_receipt.json"
        ),
        "amendment_sha256": sha256(STRUCTURED_AMENDMENT_PATH),
        "amendment_freeze_receipt_sha256": sha256(
            STRUCTURED_AMENDMENT_FREEZE_PATH
        ),
        "comparator_freeze_sha256": sha256(COMPARATOR_FREEZE_PATH),
        "selection_roots_opened": roots,
        "campaign_indices": [0, 1],
        "kappa_cells": list(KAPPAS),
        "neural_configuration_opened": False,
        "optimizer_seed_opened": False,
        "confirmation_roots_opened": False,
        "runtime": runtime,
    }
    opening_path = args.output_dir / "structured_bar_opening_receipt.json"
    write_json(opening_path, opening)

    histories = build_histories(roots, KAPPAS)
    rows, source_receipt = evaluate_structured(
        histories,
        campaign_indices={0, 1},
        contract=contract,
        progress_dir=args.output_dir,
    )
    rows_path = args.output_dir / "structured_rows.json"
    write_json(rows_path, rows)
    identities = [
        {
            "history_root": int(row["history_root"]),
            "kappa": float(row["kappa"]),
            "campaign_index": int(row["campaign_index"]),
            "arm": str(row["arm"]),
            "skeleton_sha256": str(row["skeleton_sha256"]),
            "prefix_state_hash": str(row["prefix_state_hash"]),
        }
        for row in rows
    ]
    completion = {
        "schema_version": "q_r1_shared_structured_completion_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_contract_sha256": sha256(CONTRACT_PATH),
        "amendment_sha256": sha256(STRUCTURED_AMENDMENT_PATH),
        "opening_receipt_sha256": sha256(opening_path),
        "structured_rows_sha256": sha256(rows_path),
        "rows_digest_sha256": json_sha256(rows),
        "identities_sha256": json_sha256(identities),
        "rows": len(rows),
        "elapsed_seconds": float(source_receipt["elapsed_seconds"]),
        "cache_entries": int(source_receipt["cache_entries"]),
        "selection_roots": roots,
        "campaign_indices": [0, 1],
        "kappa_cells": list(KAPPAS),
        "neural_configuration_opened": False,
        "optimizer_seed_opened": False,
        "confirmation_roots_opened": False,
        "immutable": True,
    }
    completion_path = args.output_dir / "structured_bar_completion_receipt.json"
    write_json(completion_path, completion)
    validate_shared_structured_bar(
        rows_path=rows_path,
        completion_receipt_path=completion_path,
        opening_receipt_path=opening_path,
        expected_contract_sha256=sha256(CONTRACT_PATH),
        expected_amendment_sha256=sha256(STRUCTURED_AMENDMENT_PATH),
        expected_roots=roots,
    )
    write_json(
        args.output_dir / "result.json",
        {
            "schema_version": "q_r1_shared_structured_artifact_run_v1",
            "claim_status": amendment["claim_status"],
            "base_contract_sha256": sha256(CONTRACT_PATH),
            "amendment_sha256": sha256(STRUCTURED_AMENDMENT_PATH),
            "structured_rows_sha256": sha256(rows_path),
            "completion_receipt_sha256": sha256(completion_path),
            "rows": len(rows),
            "confirmation_roots_opened": False,
        },
    )
    print(json.dumps(completion, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
