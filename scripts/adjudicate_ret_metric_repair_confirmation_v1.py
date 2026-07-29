#!/usr/bin/env python3
"""Fail-closed adjudication for the prospective ReT repair confirmation."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.expanded_contract_controllers_v2 import posture_name


def canonical_sha(value: Any) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def paired_interval(
    deltas: list[float],
    *,
    seed: int,
    draws: int,
) -> dict[str, Any]:
    values = np.asarray(deltas, dtype=float)
    rng = np.random.default_rng(seed)
    boot = np.asarray(
        [rng.choice(values, len(values), replace=True).mean() for _ in range(draws)]
    )
    return {
        "n_tapes": int(len(values)),
        "delta_mean": float(values.mean()),
        "ci95": [
            float(np.quantile(boot, 0.025)),
            float(np.quantile(boot, 0.975)),
        ],
        "n_positive": int((values > 0.0).sum()),
        "deltas": values.tolist(),
    }


def validate_run(
    run_dir: Path,
    contract: dict[str, Any],
    contract_path: Path,
) -> tuple[dict, list[dict]]:
    result_path = run_dir / "result.json"
    rows_path = run_dir / "rows.json"
    traces_path = run_dir / "traces.json"
    receipt_path = run_dir / "completion_receipt.json"
    for path in (result_path, rows_path, traces_path, receipt_path):
        if not path.is_file():
            raise ValueError(f"missing confirmation artifact: {path}")

    result = json.loads(result_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    if result.get("confirmation_roots_opened") is not True:
        raise ValueError("run is not labelled as confirmation")
    if result.get("contract_sha256") != file_sha(contract_path):
        raise ValueError("run contract hash differs from local frozen contract")
    if receipt.get("status") != "COMPLETE":
        raise ValueError("completion receipt is not terminal")
    expected_receipt = {
        "result_sha256": file_sha(result_path),
        "rows_sha256": file_sha(rows_path),
        "traces_sha256": file_sha(traces_path),
    }
    for key, observed in expected_receipt.items():
        if receipt.get(key) != observed:
            raise ValueError(f"completion receipt mismatch: {key}")
    if result.get("all_prefix_state_hashes_match") is not True:
        raise ValueError("prefix replay gate failed")
    if result.get("metric") != contract["primary_endpoint"]:
        raise ValueError("run endpoint differs from contract")

    for family, family_result in result["family_results"].items():
        if family_result["roots"] != contract["roots"][family]:
            raise ValueError(f"root mismatch for {family}")
        if family_result["incumbent_posture"] != contract["frozen_incumbents"][family]:
            raise ValueError(f"incumbent mismatch for {family}")
        if int(family_result["candidate_count"]) != 216:
            raise ValueError(f"candidate domain incomplete for {family}")
    return result, json.loads(rows_path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--run-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    contract = json.loads(args.contract.read_text())
    all_rows: list[dict[str, Any]] = []
    observed_families: set[str] = set()
    run_hashes: list[str] = []
    for run_dir in args.run_dirs:
        result, rows = validate_run(run_dir, contract, args.contract)
        observed_families.update(result["family_results"])
        all_rows.extend(rows)
        run_hashes.append(file_sha(run_dir / "result.json"))
    if observed_families != set(contract["families"]):
        raise ValueError("confirmation family coverage is incomplete")

    draws = int(contract["inference"]["bootstrap_draws"])
    endpoints = [
        contract["primary_endpoint"],
        contract["mandatory_semantic_sensitivity"]["endpoint"],
        "ret_excel",
        "ret_excel_full_ledger",
        "ret_thesis",
        "flow_fill_rate",
        "lost_orders",
        "delivered_rations",
        "unresolved",
        "strategic_injected",
        "terminal_stock",
    ]
    families: dict[str, Any] = {}
    for family_index, family in enumerate(contract["families"]):
        frozen_name = posture_name(tuple(contract["frozen_incumbents"][family]))
        mpc_rows = sorted(
            (
                row
                for row in all_rows
                if row["family"] == family and row["arm"] == "replay_mpc_v2"
            ),
            key=lambda row: int(row["tape_seed"]),
        )
        static_rows = sorted(
            (
                row
                for row in all_rows
                if row["family"] == family and row["arm"] == frozen_name
            ),
            key=lambda row: int(row["tape_seed"]),
        )
        if [row["tape_seed"] for row in mpc_rows] != contract["roots"][family]:
            raise ValueError(f"MPC tape coverage mismatch for {family}")
        if [row["tape_seed"] for row in static_rows] != contract["roots"][family]:
            raise ValueError(f"static tape coverage mismatch for {family}")

        comparisons: dict[str, Any] = {}
        for endpoint_index, endpoint in enumerate(endpoints):
            deltas = [
                float(mpc[endpoint]) - float(static[endpoint])
                for mpc, static in zip(mpc_rows, static_rows)
            ]
            comparisons[endpoint] = paired_interval(
                deltas,
                seed=2026072900 + family_index * 100 + endpoint_index,
                draws=draws,
            )

        primary_lcb = comparisons[contract["primary_endpoint"]]["ci95"][0]
        semantic_lcb = comparisons[
            contract["mandatory_semantic_sensitivity"]["endpoint"]
        ]["ci95"][0]
        fill_lcb = comparisons["flow_fill_rate"]["ci95"][0]
        materiality = float(contract["inference"]["materiality_threshold"])
        fill_floor = -0.005
        if (
            primary_lcb > materiality
            and semantic_lcb > 0.0
            and fill_lcb >= fill_floor
        ):
            verdict = "PASS_MATERIAL_REPAIRED_MPC"
        elif primary_lcb > 0.0 and semantic_lcb > 0.0 and fill_lcb >= fill_floor:
            verdict = "PASS_DIRECTIONAL_ONLY"
        else:
            verdict = "NOT_CONFIRMED"
        families[family] = {
            "frozen_incumbent": contract["frozen_incumbents"][family],
            "comparisons": comparisons,
            "verdict": verdict,
        }

    payload = {
        "schema_version": "ret_metric_repair_confirmation_v1",
        "claim_status": "PROSPECTIVE_CORRECTIVE_CONFIRMATION",
        "contract_path": str(args.contract),
        "contract_sha256": file_sha(args.contract),
        "source_result_sha256": run_hashes,
        "families": families,
        "historical_endpoint_unchanged": True,
        "neural_authorization": False,
        "quantity_time_causal_status": "DISCLOSED_PROXY_NOT_EXACT_ATTRIBUTION",
    }
    payload["self_sha256"] = canonical_sha(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
