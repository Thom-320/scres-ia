#!/usr/bin/env python3
"""Aggregate fresh H2/H3 confirmation shards with frozen multiplicity gates."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any

from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_garrido_h2_h3_confirmation_v1 import file_sha256

DEFAULT_CONTRACT = ROOT / "contracts/garrido_h2_h3_confirmation_v1.json"
DEFAULT_FREEZE = (
    ROOT / "contracts/garrido_h2_h3_confirmation_v1_freeze_receipt.json"
)
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


def interval(values: list[float]) -> dict[str, float | int]:
    n = len(values)
    mean = statistics.fmean(values)
    sd = statistics.stdev(values)
    critical = float(student_t.ppf(0.975, n - 1))
    half = critical * sd / math.sqrt(n)
    statistic = math.inf if sd == 0.0 and mean > 0.0 else (
        0.0 if sd == 0.0 else mean / (sd / math.sqrt(n))
    )
    return {
        "n_tapes": n,
        "mean": mean,
        "sd_between_tapes": sd,
        "lcb95": mean - half,
        "ucb95": mean + half,
        "positive_tapes": sum(value > 0.0 for value in values),
        "zero_tapes": sum(value == 0.0 for value in values),
        "one_sided_positive_p": float(student_t.sf(statistic, n - 1)),
    }


def holm_passes(panels: dict[str, float], alpha: float) -> dict[str, Any]:
    ordered = sorted(panels.items(), key=lambda item: (item[1], item[0]))
    output: dict[str, Any] = {}
    still_rejecting = True
    m = len(ordered)
    for rank, (name, p_value) in enumerate(ordered, start=1):
        threshold = alpha / (m - rank + 1)
        passed = still_rejecting and p_value <= threshold
        output[name] = {
            "rank": rank,
            "p_value": p_value,
            "holm_threshold": threshold,
            "pass": passed,
        }
        if not passed:
            still_rejecting = False
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--freeze-receipt", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--shards", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise RuntimeError("STOP_CONFIRMATION_AGGREGATE_OUTPUT_EXISTS")
    contract = json.loads(args.contract.read_text())
    freeze = json.loads(args.freeze_receipt.read_text())
    if freeze.get("contract_sha256") != file_sha256(args.contract):
        raise RuntimeError("STOP_CONFIRMATION_AGGREGATE_AUTHORITY")
    if freeze.get("analyzer_sha256") != file_sha256(Path(__file__)):
        raise RuntimeError("STOP_CONFIRMATION_AGGREGATE_ANALYZER")
    expected_roots = set(
        map(int, contract["execution"]["confirmation_tape_roots"])
    )
    roots_seen: set[int] = set()
    commits: set[str] = set()
    rows: list[dict[str, Any]] = []
    shard_hashes: list[str] = []
    for shard in args.shards:
        receipt = json.loads((shard / "completion_receipt.json").read_text())
        if receipt.get("status") != "COMPLETE_VALID_CONFIRMATION_SHARD":
            raise RuntimeError("STOP_CONFIRMATION_AGGREGATE_SHARD")
        if receipt.get("contract_sha256") != file_sha256(args.contract):
            raise RuntimeError("STOP_CONFIRMATION_AGGREGATE_CONTRACT")
        rows_path = shard / "rows.jsonl"
        if receipt.get("rows_sha256") != file_sha256(rows_path):
            raise RuntimeError("STOP_CONFIRMATION_AGGREGATE_ROWS_HASH")
        shard_rows = [
            json.loads(line)
            for line in rows_path.read_text().splitlines()
            if line.strip()
        ]
        if len(shard_rows) != int(receipt["row_count"]):
            raise RuntimeError("STOP_CONFIRMATION_AGGREGATE_ROW_COUNT")
        rows.extend(shard_rows)
        roots_seen.update(map(int, receipt["confirmation_tape_roots"]))
        commits.add(str(receipt["code_commit"]))
        shard_hashes.append(str(receipt["rows_sha256"]))
    if roots_seen != expected_roots or len(commits) != 1:
        raise RuntimeError("STOP_CONFIRMATION_AGGREGATE_COVERAGE")
    identities = [(int(row["tape_root"]), int(row["cf"])) for row in rows]
    if len(rows) != 90 * len(expected_roots) or len(set(identities)) != len(rows):
        raise RuntimeError("STOP_CONFIRMATION_AGGREGATE_MATRIX")
    by_identity = {(int(row["tape_root"]), int(row["cf"])): row for row in rows}

    neutral_equal = 0
    for root in expected_roots:
        for base in range(1, 31):
            baseline = by_identity[(root, base)]
            shifted = by_identity[(root, base + 60)]
            if int(shifted["shifts"]) == 1:
                if not all(baseline[m] == shifted[m] for m in METRICS):
                    raise RuntimeError("STOP_CONFIRMATION_NEUTRAL_IDENTITY")
                neutral_equal += 1

    tape_values: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    tape_rows: list[dict[str, Any]] = []
    for root in sorted(expected_roots):
        for family, bases in (
            ("R1r", range(1, 11)),
            ("R2r", range(11, 21)),
            ("R3", range(21, 31)),
        ):
            for treatment, offset in (("H2_buffer", 30), ("H3_shift", 60)):
                item: dict[str, Any] = {
                    "confirmation_tape_root": root,
                    "family": family,
                    "treatment": treatment,
                }
                for metric in METRICS:
                    value = statistics.fmean(
                        float(by_identity[(root, base + offset)][metric])
                        - float(by_identity[(root, base)][metric])
                        for base in bases
                    )
                    item[f"delta_{metric}"] = value
                    tape_values[(family, treatment, metric)].append(value)
                tape_rows.append(item)

    results: dict[str, Any] = {}
    primary_p: dict[str, float] = {}
    for family in ("R1r", "R2r", "R3"):
        results[family] = {}
        for treatment in ("H2_buffer", "H3_shift"):
            panel = {
                metric: interval(tape_values[(family, treatment, metric)])
                for metric in METRICS
            }
            name = f"{family}:{treatment}"
            primary_p[name] = float(
                panel["ret_excel"]["one_sided_positive_p"]
            )
            results[family][treatment] = panel
    holm = holm_passes(
        primary_p,
        float(contract["confirmation_inference"]["familywise_alpha"]),
    )
    panel_gates: dict[str, Any] = {}
    for family in ("R1r", "R2r", "R3"):
        for treatment in ("H2_buffer", "H3_shift"):
            name = f"{family}:{treatment}"
            panel = results[family][treatment]
            gates = {
                "primary_holm": bool(holm[name]["pass"]),
                "full_ledger_lcb_positive": (
                    float(panel["ret_excel_full_ledger"]["lcb95"]) > 0.0
                ),
                "fill_lcb_positive": (
                    float(panel["flow_fill_rate"]["lcb95"]) > 0.0
                ),
                "delivered_lcb_positive": (
                    float(panel["delivered_rations"]["lcb95"]) > 0.0
                ),
                "unresolved_ucb_negative": (
                    float(panel["unresolved_orders"]["ucb95"]) < 0.0
                ),
                "generated_orders_exact_zero": all(
                    value == 0.0
                    for value in tape_values[
                        (family, treatment, "generated_orders")
                    ]
                ),
            }
            panel_gates[name] = {
                **gates,
                "confirmed": all(gates.values()),
            }
    global_pass = all(
        bool(item["confirmed"]) for item in panel_gates.values()
    )
    args.output_dir.mkdir(parents=True)
    tape_path = args.output_dir / "tape_level_deltas.json"
    tape_path.write_text(json.dumps(tape_rows, indent=2, sort_keys=True) + "\n")
    payload = {
        "schema_version": "garrido_h2_h3_confirmation_analysis_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": (
            "CONFIRM_H2_H3_ALL_SIX_PANELS"
            if global_pass
            else "MIXED_OR_NULL_H2_H3_CONFIRMATION"
        ),
        "contract_sha256": file_sha256(args.contract),
        "freeze_receipt_sha256": file_sha256(args.freeze_receipt),
        "code_commit": next(iter(commits)),
        "confirmation_tape_roots": sorted(expected_roots),
        "row_count": len(rows),
        "neutral_shift_checks": {
            "count": neutral_equal,
            "all_equal": True,
        },
        "holm": holm,
        "panel_gates": panel_gates,
        "results": results,
        "global_confirmation_pass": global_pass,
        "claim_boundary": (
            "Confirmation applies only to H2/H3 resource interventions in "
            "the frozen thesis-grounded reconstructed DES; it does not "
            "establish learner, feedback, or architectural value."
        ),
        "development_roots_opened": False,
        "confirmation_roots_opened": True,
    }
    result_path = args.output_dir / "result.json"
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    receipt = {
        "status": "COMPLETE_VALID_CONFIRMATION_AGGREGATE",
        "contract_sha256": file_sha256(args.contract),
        "freeze_receipt_sha256": file_sha256(args.freeze_receipt),
        "result_sha256": file_sha256(result_path),
        "tape_level_deltas_sha256": file_sha256(tape_path),
        "source_shard_rows_sha256": sorted(shard_hashes),
        "global_confirmation_pass": global_pass,
        "development_roots_opened": False,
        "confirmation_roots_opened": True,
    }
    (args.output_dir / "completion_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
