#!/usr/bin/env python3
"""Aggregate completed corrective shards and infer only at tape level."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import statistics
from typing import Any

from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONTRACT = ROOT / "contracts" / "garrido_h2_h3_corrective_v1.json"
METRICS = (
    "ret_excel",
    "ret_excel_full_ledger",
    "ret_thesis",
    "ret_continuous",
    "flow_fill_rate",
    "delivered_rations",
    "generated_orders",
    "scored_rows",
    "omitted_rows",
    "served_orders",
    "unresolved_orders",
    "lost_orders",
)


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def interval(values: list[float]) -> dict[str, float | int]:
    n = len(values)
    mean = statistics.fmean(values)
    sd = statistics.stdev(values)
    critical = float(student_t.ppf(0.975, n - 1))
    half = critical * sd / math.sqrt(n)
    return {
        "n_tapes": n,
        "mean": mean,
        "sd_between_tapes": sd,
        "lcb95": mean - half,
        "ucb95": mean + half,
        "positive_tapes": sum(value > 0.0 for value in values),
        "zero_tapes": sum(value == 0.0 for value in values),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--shards", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise RuntimeError("STOP_AGGREGATE_OUTPUT_DIR_EXISTS")
    args.output_dir.mkdir(parents=True)
    contract = json.loads(args.contract.read_text())
    contract_hash = file_sha256(args.contract)
    expected_roots = set(contract["execution"]["tape_roots"])
    commits: set[str] = set()
    roots_seen: set[int] = set()
    rows: list[dict[str, Any]] = []
    shard_receipts: list[dict[str, Any]] = []
    for shard in args.shards:
        receipt = json.loads((shard / "completion_receipt.json").read_text())
        if receipt["status"] != "COMPLETE_VALID_SHARD":
            raise RuntimeError("STOP_AGGREGATE_INVALID_SHARD")
        if receipt["contract_sha256"] != contract_hash:
            raise RuntimeError("STOP_AGGREGATE_CONTRACT_HASH")
        rows_path = shard / "rows.jsonl"
        if file_sha256(rows_path) != receipt["rows_sha256"]:
            raise RuntimeError("STOP_AGGREGATE_ROWS_HASH")
        shard_rows = [
            json.loads(line)
            for line in rows_path.read_text().splitlines()
            if line.strip()
        ]
        if len(shard_rows) != receipt["row_count"]:
            raise RuntimeError("STOP_AGGREGATE_ROW_COUNT")
        rows.extend(shard_rows)
        commits.add(receipt["code_commit"])
        roots_seen.update(receipt["tape_roots"])
        shard_receipts.append(receipt)
    if len(commits) != 1:
        raise RuntimeError("STOP_AGGREGATE_CODE_COMMIT")
    if roots_seen != expected_roots:
        raise RuntimeError(
            f"STOP_AGGREGATE_TAPE_COVERAGE:{sorted(roots_seen)}"
        )
    identities = [(row["tape_root"], row["cf"]) for row in rows]
    if len(rows) != 90 * len(expected_roots) or len(set(identities)) != len(rows):
        raise RuntimeError("STOP_AGGREGATE_MATRIX_COVERAGE")
    by_identity = {
        (row["tape_root"], row["cf"]): row for row in rows
    }

    neutral_checks: list[dict[str, Any]] = []
    for tape_root in sorted(expected_roots):
        for base in range(1, 31):
            baseline = by_identity[(tape_root, base)]
            shifted = by_identity[(tape_root, base + 60)]
            if shifted["shifts"] == 1:
                equal = all(
                    baseline[metric] == shifted[metric] for metric in METRICS
                )
                neutral_checks.append(
                    {
                        "tape_root": tape_root,
                        "base_index": base,
                        "equal": equal,
                    }
                )
                if not equal:
                    raise RuntimeError(
                        "STOP_AGGREGATE_NEUTRAL_SHIFT_IDENTITY"
                    )

    tape_deltas: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    raw_tape_rows: list[dict[str, Any]] = []
    for tape_root in sorted(expected_roots):
        for family, bases in (
            ("R1r", range(1, 11)),
            ("R2r", range(11, 21)),
            ("R3", range(21, 31)),
        ):
            for treatment, offset in (("H2_buffer", 30), ("H3_shift", 60)):
                summary: dict[str, Any] = {
                    "tape_root": tape_root,
                    "family": family,
                    "treatment": treatment,
                }
                for metric in METRICS:
                    deltas = [
                        float(by_identity[(tape_root, base + offset)][metric])
                        - float(by_identity[(tape_root, base)][metric])
                        for base in bases
                    ]
                    value = statistics.fmean(deltas)
                    summary[f"delta_{metric}"] = value
                    tape_deltas[(family, treatment, metric)].append(value)
                raw_tape_rows.append(summary)

    results: dict[str, Any] = {}
    for family in ("R1r", "R2r", "R3"):
        results[family] = {}
        for treatment in ("H2_buffer", "H3_shift"):
            panel = {
                metric: interval(tape_deltas[(family, treatment, metric)])
                for metric in METRICS
            }
            panel["ret_excel"]["directional_gate"] = (
                "PASS_DIRECTIONAL_TAPE_LEVEL"
                if float(panel["ret_excel"]["lcb95"]) > 0.0
                else "NO_PASS_DIRECTIONAL_TAPE_LEVEL"
            )
            results[family][treatment] = panel

    tape_path = args.output_dir / "tape_level_deltas.json"
    tape_path.write_text(
        json.dumps(raw_tape_rows, indent=2, sort_keys=True) + "\n"
    )
    result_payload = {
        "schema_version": "garrido_h2_h3_corrective_analysis_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "COMPLETE_DEVELOPMENT_TAPE_LEVEL_ANALYSIS",
        "contract_sha256": contract_hash,
        "code_commit": next(iter(commits)),
        "tape_roots": sorted(expected_roots),
        "row_count": len(rows),
        "neutral_shift_checks": {
            "count": len(neutral_checks),
            "all_equal": all(item["equal"] for item in neutral_checks),
        },
        "results": results,
        "claim_boundary": (
            "Development directional evidence only; no independent "
            "order-level replication and no confirmatory claim."
        ),
        "confirmation_roots_opened": False,
    }
    result_path = args.output_dir / "result.json"
    result_path.write_text(
        json.dumps(result_payload, indent=2, sort_keys=True) + "\n"
    )
    receipt = {
        "status": "COMPLETE_VALID_AGGREGATE",
        "contract_sha256": contract_hash,
        "code_commit": next(iter(commits)),
        "result_sha256": file_sha256(result_path),
        "tape_level_deltas_sha256": file_sha256(tape_path),
        "source_shard_rows_sha256": sorted(
            receipt["rows_sha256"] for receipt in shard_receipts
        ),
        "confirmation_roots_opened": False,
    }
    (args.output_dir / "completion_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
