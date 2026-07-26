#!/usr/bin/env python3
"""Mechanically freeze the externally reviewed factorial-v4 contract bytes.

The reviewed JSON is renamed without changing a byte.  Its historical internal
status remains a provenance marker; the separate immutable receipt confers
``FROZEN_PROSPECTIVE_UNOPENED`` authority.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "contracts/q_r1_matched_retention_factorial_v4.json"
RECEIPT = (
    ROOT / "contracts/q_r1_matched_retention_factorial_v4_freeze_receipt.json"
)
PASS_DOC = ROOT / "docs/FACTORIAL_V4_PASS_PRE_FREEZE_2026-07-26.md"
IMPLEMENTATION_RECEIPT = (
    ROOT
    / "results/q_r1/matched_retention_factorial_v4_pre_freeze"
    / "implementation_receipt.json"
)
COMPARATOR_RECEIPT = ROOT / "contracts/q_r1_comparator_v2_frozen_c256_v1.json"

EXPECTED_CONTRACT_SHA256 = (
    "bb92a2cbfcd3691a77f7f9ab8a269d7ffab65823d37b41f70d0b13795d92e764"
)
EXTERNAL_PASS_ORIGIN_COMMIT = "d2aa607d404c274ad89e37355d5e6570b797df7f"
REVIEWED_IMPLEMENTATION_COMMIT = "1ce6e6244b09c1579fea29839abb6dc96447cfb6"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
    ).strip()


def build_receipt() -> dict[str, object]:
    if RECEIPT.exists():
        raise RuntimeError("refusing to overwrite the factorial-v4 freeze receipt")
    if git("status", "--porcelain"):
        raise RuntimeError("freeze requires a clean worktree")
    if sha256(CONTRACT) != EXPECTED_CONTRACT_SHA256:
        raise RuntimeError("contract bytes differ from the externally reviewed draft")
    contract = json.loads(CONTRACT.read_text())
    if contract.get("status") != "DRAFT_PROSPECTIVE_UNOPENED_NOT_AUTHORITY":
        raise RuntimeError("reviewed contract content marker changed")
    if contract["data_splits"].get("opened") is not False:
        raise RuntimeError("development roots were opened before freeze")
    pass_text = PASS_DOC.read_text()
    if "Verdict: `PASS_PRE_FREEZE`" not in pass_text:
        raise RuntimeError("external PASS_PRE_FREEZE is missing")
    implementation = json.loads(IMPLEMENTATION_RECEIPT.read_text())
    if implementation.get("fresh_development_roots_opened") is not False:
        raise RuntimeError("implementation receipt reports opened development roots")
    if implementation.get("confirmation_roots_opened") is not False:
        raise RuntimeError("implementation receipt reports opened confirmation roots")
    return {
        "schema_version": "q_r1_matched_retention_factorial_v4_freeze_receipt_v1",
        "status": "FROZEN_PROSPECTIVE_UNOPENED",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "contract_path": str(CONTRACT.relative_to(ROOT)),
        "contract_sha256": sha256(CONTRACT),
        "reviewed_contract_internal_status": contract["status"],
        "content_byte_identical_to_externally_reviewed_draft": True,
        "external_pass_document": str(PASS_DOC.relative_to(ROOT)),
        "external_pass_document_sha256": sha256(PASS_DOC),
        "external_pass_origin_commit": EXTERNAL_PASS_ORIGIN_COMMIT,
        "reviewed_implementation_commit": REVIEWED_IMPLEMENTATION_COMMIT,
        "implementation_receipt": str(IMPLEMENTATION_RECEIPT.relative_to(ROOT)),
        "implementation_receipt_sha256": sha256(IMPLEMENTATION_RECEIPT),
        "comparator_freeze_receipt": str(COMPARATOR_RECEIPT.relative_to(ROOT)),
        "comparator_freeze_receipt_sha256": sha256(COMPARATOR_RECEIPT),
        "freeze_execution_parent_commit": git("rev-parse", "HEAD"),
        "worktree_clean_before_receipt": True,
        "fresh_development_roots_opened": False,
        "checkpoint_selection_roots_opened": False,
        "confirmation_roots_opened": False,
        "development_optimizer_seeds_opened": False,
        "instrument_preflight_seed": 7672001,
        "instrument_preflight_seed_status": (
            "BURNED_INSTRUMENT_ONLY_NOT_DEVELOPMENT_ELIGIBLE"
        ),
        "fresh_development_optimizer_seeds": [
            7672101,
            7672102,
            7672103,
            7672104,
            7672105,
        ],
        "learner_return_observed_before_freeze": False,
        "confirmation_return_observed_before_freeze": False,
        "next_authorized_step": (
            "Audit this receipt, then run static-bar; do not open confirmation."
        ),
    }


def main() -> int:
    payload = build_receipt()
    RECEIPT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
