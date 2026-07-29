#!/usr/bin/env python3
"""Audit the source-aligned request-snapshot Excel ReT v2 implementation."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.garrido_replication import (
    DEFAULT_RAW_WORKBOOKS,
    load_raw_garrido_targets,
)
from supply_chain.ret_thesis import (
    compute_order_level_ret_excel_request_snapshot_ledger,
)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE_AUDIT = (
    ROOT
    / "research/paper2_exhaustive_search/ret_excel_visible_v1_source_semantics_audit_20260714.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "research/paper2_exhaustive_search/ret_excel_request_snapshot_v2_implementation_audit_20260714.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def workbook_order(row) -> SimpleNamespace:
    return SimpleNamespace(
        j=row.j,
        OPTj=row.optj,
        OATj=row.oatj,
        CTj=row.ctj,
        LTj=row.ltj,
        APj=row.apj,
        RPj=row.rpj,
        DPj=row.dpj,
        quantity=row.q,
        remaining_qty=0.0,
        lost=False,
        lost_time=None,
        ret_risk_indicators=row.risk_values,
        ret_bt_at_request=int(row.sum_bt),
        ret_ut_at_request=int(row.sum_ut),
        ret_ledger_snapshot_time=row.optj,
        ret_ledger_event_sequence=row.j,
    )


def main() -> int:
    if DEFAULT_OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {DEFAULT_OUTPUT}")
    source_audit = json.loads(DEFAULT_SOURCE_AUDIT.read_text())
    unhashed_source = dict(source_audit)
    expected_source_hash = unhashed_source.pop("content_sha256")
    if json_sha256(unhashed_source) != expected_source_hash:
        raise ValueError("source-semantics audit content hash mismatch")

    targets = load_raw_garrido_targets(DEFAULT_RAW_WORKBOOKS)
    rows = 0
    mismatches = 0
    max_abs_diff = 0.0
    sheets: dict[str, Any] = {}
    for cfi, target in sorted(targets.items()):
        orders = [workbook_order(row) for row in target.orders]
        result = compute_order_level_ret_excel_request_snapshot_ledger(orders)
        sheet_mismatches = 0
        sheet_max = 0.0
        for actual, expected in zip(
            result["ret_values"],
            (row.ret for row in target.orders),
            strict=True,
        ):
            difference = abs(float(actual) - float(expected))
            sheet_max = max(sheet_max, difference)
            if difference > 1e-12:
                sheet_mismatches += 1
        sheets[f"CF{cfi}"] = {
            "rows": len(target.orders),
            "mismatches": sheet_mismatches,
            "max_abs_diff": sheet_max,
        }
        rows += len(target.orders)
        mismatches += sheet_mismatches
        max_abs_diff = max(max_abs_diff, sheet_max)

    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    result = {
        "schema_version": "ret_excel_request_snapshot_v2_implementation_audit_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_head_before_uncommitted_implementation": head,
        "status": "SOURCE_ALIGNED_IMPLEMENTED__PROVISIONAL_PENDING_SAME_TIME_GARRIDO_CONFIRMATION",
        "canonical_development_contract": "ret_excel_request_snapshot_v2",
        "source_semantics_audit": {
            "path": str(DEFAULT_SOURCE_AUDIT.relative_to(ROOT)),
            "sha256": sha256(DEFAULT_SOURCE_AUDIT),
            "content_sha256": expected_source_hash,
        },
        "implementation_sources": {
            path: sha256(ROOT / path)
            for path in (
                "supply_chain/ret_thesis.py",
                "supply_chain/supply_chain.py",
                "supply_chain/episode_metrics.py",
                "supply_chain/garrido_replication.py",
                "tests/test_ret_excel_request_snapshot_contract.py",
            )
        },
        "workbooks": [
            {"path": str(path), "sha256": sha256(path)}
            for path in DEFAULT_RAW_WORKBOOKS
        ],
        "canonical_aggregator_workbook_replay": {
            "formula_rows": rows,
            "mismatches": mismatches,
            "max_abs_diff": max_abs_diff,
            "tolerance": 1e-12,
            "sheets": sheets,
            "passes": bool(rows == 47_546 and mismatches == 0),
        },
        "request_snapshot_contract": {
            "capture_time": "OPTj before inserting request j into the Op9 queue",
            "fields": [
                "ret_bt_at_request",
                "ret_ut_at_request",
                "ret_ledger_snapshot_time",
                "ret_ledger_event_sequence",
            ],
            "row_population": "completed non-lost rows only, emitted in original j/OPTj order, original j preserved, no clipping",
            "fallback": "half-open [OPTj+LTj, min(OATj,lost_time)) reconstruction for adapters without captured snapshots; diagnostic unless the complete event history is present",
            "same_timestamp_convention": "events already scheduled before request generation run first; snapshot; then enqueue request j; record event sequence",
        },
        "superseded_contract": {
            "id": "ret_excel_visible_v1",
            "disposition": "QUARANTINED_OAT_LEDGER_NOT_SOURCE_VALIDATED",
        },
        "scientific_authorization": {
            "paper2_confirmed": False,
            "paper3_authorized": False,
            "prior_h_j_mtr_results_restored": False,
            "required_next": "rescore identical burned/calibration tapes and rebuild every comparator under v2; obtain Garrido confirmation of same-timestamp event order before virgin confirmation",
        },
        "claim_limit": "The v2 implementation is source-aligned and exactly reproduces workbook cells when source Bt/Ut snapshots are injected. It does not prove endogenous DES ledger fidelity, adaptive headroom, or a Paper-2 result.",
    }
    result["content_sha256"] = json_sha256(result)
    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(DEFAULT_OUTPUT),
        "formula_rows": rows,
        "mismatches": mismatches,
        "max_abs_diff": max_abs_diff,
        "content_sha256": result["content_sha256"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
