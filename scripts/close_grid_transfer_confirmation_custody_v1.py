#!/usr/bin/env python3
"""Close the reserved seed block after the confirmation result is sealed.

This is deliberately a separate post-run action.  A reservation is not quietly left as
``RESERVED_NOT_OPENED`` after fresh tapes have been consumed, and a failed confirmation is still
recorded as consumed rather than being relabelled as a failed preflight.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from supply_chain.arm_runner import seal_and_write  # noqa: E402

BLOCK_ID = "garrido_grid_transfer_v1_confirmation"
START, END = 8_100_001, 8_100_060
CONTRACT = Path("docs/PREREGISTRO_CONFIRMACION_TRANSFERENCIA_REJILLA_2026-08-05.md")
REGISTRY = Path("research/seed_custody_registry.json")
RESULT = Path("results/grid_transfer_confirmation/result.json")
RECEIPT = Path("results/custody/garrido_grid_transfer_confirmation_closeout.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=RESULT)
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    parser.add_argument("--receipt", type=Path, default=RECEIPT)
    args = parser.parse_args()

    result = json.loads(args.result.read_text())
    seeds = result.get("seeds")
    expected = list(range(START, END + 1))
    if result.get("run_role") != "CONFIRMATION":
        raise SystemExit("result is not marked as a confirmation")
    if seeds != expected:
        raise SystemExit("result does not contain the exact reserved seed block")
    if result.get("contract_sha256") != hashlib.sha256(args.contract.read_bytes()).hexdigest():
        raise SystemExit("result contract hash does not match the closeout contract")

    registry = json.loads(args.registry.read_text())
    block = next((row for row in registry.get("blocks", []) if row.get("id") == BLOCK_ID), None)
    if block is None:
        raise SystemExit(f"missing registry block: {BLOCK_ID}")
    if (int(block["start"]), int(block["end"])) != (START, END):
        raise SystemExit("registry block range mismatch")
    if block.get("status") != "RESERVED_NOT_OPENED":
        raise SystemExit(f"block is not awaiting closeout: {block.get('status')!r}")

    closed_at = datetime.now(timezone.utc).isoformat()
    block["status"] = "USED_CONFIRMATION"
    block["artifact"] = str(args.result)
    block["artifact_sha256"] = result.get("self_sha256")
    block["claim_status"] = result.get("claim_status")
    block["closed_at"] = closed_at
    block["purpose"] = (
        "Confirmation block consumed. The result may pass or fail its preregistered estimand; "
        "either way these tapes are not reusable as virgin evidence."
    )
    args.registry.write_text(json.dumps(registry, indent=1) + "\n")
    payload = {
        "schema_version": "garrido_grid_transfer_confirmation_closeout_v1",
        "claim_status": "SEED_BLOCK_CLOSED_AS_USED_CONFIRMATION",
        "block_id": BLOCK_ID,
        "seed_range": [START, END],
        "result": str(args.result),
        "result_sha256": result.get("self_sha256"),
        "result_claim_status": result.get("claim_status"),
        "closed_at": closed_at,
        "registry": str(args.registry),
        "rule": "a failed confirmation is still consumed and cannot be reused as virgin evidence",
    }
    digest = seal_and_write(payload, args.receipt, contract=args.contract, reference=args.result)
    print(f"  block closed as USED_CONFIRMATION · receipt seal {digest[:16]}…")
    print(f"  -> {args.receipt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
