#!/usr/bin/env python3
"""Canonical workbook-row ceiling for Program O's affected-order gate."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from supply_chain.garrido_replication import (  # noqa: E402
    DEFAULT_RAW_WORKBOOKS,
    load_raw_garrido_targets,
)
from supply_chain.ret_thesis import (  # noqa: E402
    compute_order_level_ret_excel_request_snapshot_ledger,
)

DEFAULT_CONTRACT = ROOT / "contracts/program_o_multi_ration_product_mix_v1.json"
DEFAULT_OUTPUT = ROOT / "results/program_o/affected_order_bound_v1/result.json"


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


def workbook_order(row: Any) -> SimpleNamespace:
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


def row_ceiling(values: list[float], gate: float) -> dict[str, Any]:
    if not values:
        raise ValueError("affected-order ceiling requires visible rows")
    deficits = sorted((max(0.0, 1.0 - float(value)) for value in values), reverse=True)
    target = float(gate) * len(values)
    cumulative = 0.0
    minimum_rows: int | None = None
    for index, deficit in enumerate(deficits, start=1):
        cumulative += deficit
        if cumulative + 1e-15 >= target:
            minimum_rows = index
            break
    return {
        "n_visible_rows": len(values),
        "mean_ret": sum(values) / len(values),
        "all_rows_perfect_rescue_upper_delta": sum(deficits) / len(values),
        "minimum_perfectly_rescued_rows_for_gate": minimum_rows,
        "minimum_fraction_visible_rows_for_gate": (
            minimum_rows / len(values) if minimum_rows is not None else None
        ),
        "largest_single_row_upper_delta_on_mean": deficits[0] / len(values),
    }


def produce(contract_path: Path) -> dict[str, Any]:
    contract = json.loads(contract_path.read_text())
    gate = float(contract["frozen_endpoint_gate"])
    fraction_gate = 0.10
    targets = load_raw_garrido_targets(DEFAULT_RAW_WORKBOOKS)
    sheets: dict[str, Any] = {}
    all_values: list[float] = []
    for cfi, target in sorted(targets.items()):
        orders = [workbook_order(row) for row in target.orders]
        ledger = compute_order_level_ret_excel_request_snapshot_ledger(orders)
        values = [float(value) for value in ledger["ret_values"]]
        bound = row_ceiling(values, gate)
        bound["passes_fraction_gate"] = bool(
            bound["minimum_fraction_visible_rows_for_gate"] is not None
            and bound["minimum_fraction_visible_rows_for_gate"] <= fraction_gate
        )
        sheets[f"CF{cfi}"] = bound
        all_values.extend(values)
    aggregate = row_ceiling(all_values, gate)
    pass_sheets = sorted(
        name for name, row in sheets.items() if row["passes_fraction_gate"]
    )
    result: dict[str, Any] = {
        "schema_version": "program_o_affected_order_bound_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": (
            "PASS_METRIC_LIVENESS_ONLY__TRANSDUCER_FREEZE_ALLOWED"
            if pass_sheets
            else "STOP_PROGRAM_O_METRIC_CEILING_BELOW_FRACTION_GATE"
        ),
        "contract": str(contract_path.relative_to(ROOT)),
        "contract_sha256": sha256(contract_path),
        "metric": "ret_excel_request_snapshot_v2",
        "delta_gate": gate,
        "maximum_affected_visible_fraction": fraction_gate,
        "method": "Canonical v2 row values are held fixed; each affected row is optimistically raised to its mathematical ceiling ReT=1. Sorted row deficits give a rigorous row-score ceiling, not a physical or policy result.",
        "workbooks": {
            str(path): sha256(path) for path in DEFAULT_RAW_WORKBOOKS
        },
        "sheets": sheets,
        "aggregate": aggregate,
        "passing_sheets": pass_sheets,
        "claim_boundary": {
            "h_pi_established": False,
            "h_obs_established": False,
            "product_physics_established": False,
            "transducer_authorized": bool(pass_sheets),
            "full_des_authorized": False,
            "learner_authorized": False,
            "paper3_authorized": False,
        },
    }
    result["content_sha256"] = json_sha256(result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    contract = args.contract.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    result = produce(contract)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
