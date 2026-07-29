#!/usr/bin/env python3
"""Run the frozen Garrido H2/H3 corrective development matrix."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.config import CAPACITY_BY_SHIFTS, INVENTORY_BUFFERS
from supply_chain.episode_metrics import compute_episode_metrics
from supply_chain.garrido_thesis_design import (
    DESIGN,
    SOURCE_VALIDATION_QUARANTINE,
    THESIS_SEEDS,
)
from supply_chain.supply_chain import MFSCSimulation

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONTRACT = ROOT / "contracts" / "garrido_h2_h3_corrective_v1.json"
FAMILY_RISKS = {
    "R1r": ("R11", "R12", "R13", "R14"),
    "R2r": ("R21", "R22", "R23", "R24"),
    "R3": ("R3",),
}


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return sha256(encoded).hexdigest()


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def require_clean_worktree() -> None:
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    )
    if status.strip():
        raise RuntimeError("STOP_INSTRUMENT_DIRTY_WORKTREE")


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("contract_id") != "garrido_h2_h3_corrective_v1":
        raise RuntimeError("STOP_INSTRUMENT_WRONG_CONTRACT")
    return payload


def verify_sources(contract: dict[str, Any], source_dir: Path) -> dict[str, str]:
    mapping = {
        "thesis": (
            contract["sources"]["thesis_file"],
            contract["sources"]["thesis_sha256"],
        ),
        "raw_data_1": (
            contract["sources"]["raw_data_1_file"],
            contract["sources"]["raw_data_1_sha256"],
        ),
        "raw_data_2": (
            contract["sources"]["raw_data_2_file"],
            contract["sources"]["raw_data_2_sha256"],
        ),
    }
    verified: dict[str, str] = {}
    for label, (name, expected) in mapping.items():
        path = source_dir / name
        if not path.exists() or file_sha256(path) != expected:
            raise RuntimeError(f"STOP_INSTRUMENT_SOURCE_HASH:{label}")
        verified[label] = expected
    return verified


def workbook_audit(source_dir: Path) -> list[dict[str, Any]]:
    import openpyxl

    outputs: list[dict[str, Any]] = []
    for filename, indices, ret_col in (
        ("Raw_data1+Re.xlsx", range(1, 11), 21),
        ("Raw_data2+Re.xlsx", range(11, 21), 27),
    ):
        workbook = openpyxl.load_workbook(
            source_dir / filename, read_only=True, data_only=True
        )
        for index in indices:
            sheet = workbook[f"CF{index}"]
            seed = None
            ret_values: list[float] = []
            order_ids: list[int] = []
            order_times: list[float] = []
            for row in sheet.iter_rows(values_only=True):
                if row and row[0] == "Seed" and isinstance(row[1], (int, float)):
                    seed = int(row[1])
                if len(row) >= ret_col and isinstance(row[ret_col - 1], (int, float)):
                    ret_values.append(float(row[ret_col - 1]))
                    if len(row) > 5 and isinstance(row[4], (int, float)):
                        order_ids.append(int(row[4]))
                    if len(row) > 5 and isinstance(row[5], (int, float)):
                        order_times.append(float(row[5]))
            if seed is None or not ret_values:
                raise RuntimeError(f"STOP_INSTRUMENT_WORKBOOK_PARSE:CF{index}")
            outputs.append(
                {
                    "cf": index,
                    "seed": seed,
                    "seed_matches_design": seed == THESIS_SEEDS[index],
                    "scored_rows": len(ret_values),
                    "max_order_id": max(order_ids),
                    "max_order_time": max(order_times),
                    "inferred_horizon_years": max(order_times) / 8064.0,
                    "mean_ret_excel": statistics.fmean(ret_values),
                    "quarantined": index in SOURCE_VALIDATION_QUARANTINE,
                    "quarantine_reason": SOURCE_VALIDATION_QUARANTINE.get(index),
                }
            )
        workbook.close()
    return outputs


def simulation_kwargs(
    contract: dict[str, Any],
    *,
    config_index: int,
    seed: int,
    risks_enabled: bool = True,
) -> dict[str, Any]:
    cfg = DESIGN[config_index]
    execution = contract["execution"]
    buffers = None
    if cfg.buffer_hours:
        level = INVENTORY_BUFFERS[cfg.buffer_hours]
        buffers = {
            "op3_rm": float(level["op3_rm"]),
            "op5_rm": float(level["op5_rm"]),
            "op9_rations": float(level["op9_rations"]),
        }
    return {
        "shifts": cfg.shifts,
        "initial_buffers": buffers,
        "inventory_replenishment_period": (
            float(cfg.buffer_hours) if cfg.buffer_hours else None
        ),
        "seed": seed,
        "horizon": cfg.horizon_hours,
        "risks_enabled": risks_enabled,
        "risk_level": "current",
        "enabled_risks": set(FAMILY_RISKS[cfg.risk_family]),
        "risk_overrides": {
            risk: "increased" for risk in cfg.increased_risks
        },
        "strict_exogenous_crn": execution["strict_exogenous_crn"],
        "periodic_release_mode": execution["periodic_release_mode"],
        "assembly_batch_release_mode": execution[
            "assembly_batch_release_mode"
        ],
        "raw_material_flow_mode": execution["raw_material_flow_mode"],
        "raw_material_order_up_to_multiplier": execution[
            "raw_material_order_up_to_multiplier"
        ],
        "downstream_q_source": execution["downstream_q_source"],
        "year_basis": "thesis",
        "warmup_trigger": "op9_arrival",
        "r14_defect_mode": "thesis_strict_op6",
        "risk_occurrence_mode": "thesis_window",
    }


def run_configuration(
    contract: dict[str, Any],
    *,
    config_index: int,
    tape_root: int,
) -> dict[str, Any]:
    cfg = DESIGN[config_index]
    seed = int(tape_root) + cfg.base_index
    sim = MFSCSimulation(**simulation_kwargs(
        contract, config_index=config_index, seed=seed
    ))
    sim.step(action=None, step_hours=cfg.horizon_hours)
    start = float(contract["execution"]["common_evaluation_start_hours"])
    if sim.warmup_time > start:
        raise RuntimeError(
            f"STOP_INSTRUMENT_COMMON_EVALUATION:CF{config_index}:"
            f"warmup={sim.warmup_time}"
        )
    metrics = compute_episode_metrics(sim, treatment_start=start)
    generated = int(metrics["n_orders"])
    served = int(metrics["n_served"])
    scored = int(metrics["ret_excel_visible_n"])
    omitted = int(metrics["ret_excel_omitted_n"])
    if scored + omitted != generated:
        raise RuntimeError(
            f"STOP_INSTRUMENT_ROW_PARTITION:CF{config_index}"
        )
    initial_raw = 0.0
    initial_rations = 0.0
    if cfg.buffer_hours:
        level = INVENTORY_BUFFERS[cfg.buffer_hours]
        initial_raw = 12.0 * (
            float(level["op3_rm"]) + float(level["op5_rm"])
        )
        initial_rations = float(level["op9_rations"])
        after_initial = (
            float(sim.total_strategic_raw_injected)
            + float(sim.total_strategic_rations_injected)
            - initial_raw
            - initial_rations
        )
        if after_initial <= 0.0:
            raise RuntimeError(
                f"STOP_INSTRUMENT_PERIODIC_BUFFER:CF{config_index}"
            )
    return {
        "cf": config_index,
        "base_index": cfg.base_index,
        "scenario": cfg.scenario,
        "hypothesis": cfg.hypothesis,
        "family": cfg.risk_family,
        "pattern": cfg.risk_pattern,
        "buffer_hours": cfg.buffer_hours,
        "shifts": cfg.shifts,
        "tape_root": int(tape_root),
        "seed": seed,
        "common_evaluation_start_hours": start,
        "warmup_time": float(sim.warmup_time),
        "generated_orders": generated,
        "scored_rows": scored,
        "omitted_rows": omitted,
        "served_orders": served,
        "unresolved_orders": generated - served,
        "lost_orders": int(metrics["lost_orders"]),
        "ret_excel": float(metrics["ret_excel"]),
        "ret_excel_full_ledger": float(metrics["ret_excel_full_ledger"]),
        "ret_thesis": float(metrics["ret_thesis"]),
        "ret_continuous": float(metrics["ret_continuous"]),
        "flow_fill_rate": float(metrics["flow_fill_rate"]),
        "delivered_rations": float(metrics["delivered_rations"]),
        "strategic_raw_injected": float(sim.total_strategic_raw_injected),
        "strategic_rations_injected": float(
            sim.total_strategic_rations_injected
        ),
        "initial_strategic_raw": initial_raw,
        "initial_strategic_rations": initial_rations,
    }


def trace_preflight(contract: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for shifts, config_index in ((1, 3), (2, 61), (3, 63)):
        kwargs = simulation_kwargs(
            contract,
            config_index=config_index,
            seed=991_000 + shifts,
            risks_enabled=False,
        )
        kwargs["shifts"] = shifts
        kwargs["horizon"] = 5_000.0
        kwargs["initial_buffers"] = None
        kwargs["inventory_replenishment_period"] = None
        sim = MFSCSimulation(**kwargs)
        sim.step(action=None, step_hours=5_000.0)
        raw = sim.material_availability_events["raw_material_al"]
        staged = sim.material_availability_events["rations_al"]
        raw_gaps = [
            raw[i][0] - raw[i - 1][0] for i in range(1, len(raw))
        ]
        staged_gaps = [
            staged[i][0] - staged[i - 1][0]
            for i in range(1, len(staged))
        ]
        expected = CAPACITY_BY_SHIFTS[shifts]
        expected_raw = 12.0 * float(expected["op3_q"])
        raw_quantities = sorted({float(qty) for _, qty in raw})
        staged_quantities = sorted({float(qty) for _, qty in staged})
        row = {
            "shifts": shifts,
            "op3_q_per_rm": float(sim.params["op3_q"]),
            "op3_arrival_quantities_total_rm": raw_quantities,
            "op3_median_start_gap": statistics.median(raw_gaps),
            "op7_batch_size": float(sim.params["batch_size"]),
            "op7_staged_quantities": staged_quantities,
            "op7_median_gap": statistics.median(staged_gaps),
            "expected_op3_total_rm": expected_raw,
            "expected_op3_rop": 168.0,
            "expected_op7_q": float(expected["op7_q"]),
            "expected_op7_rop": float(expected["op7_rop"]),
        }
        if (
            raw_quantities != [expected_raw]
            or row["op3_median_start_gap"] != 168.0
            or staged_quantities != [float(expected["op7_q"])]
            or row["op7_median_gap"] != float(expected["op7_rop"])
        ):
            raise RuntimeError(
                f"STOP_INSTRUMENT_TRACE_TABLE_6_20:S={shifts}:{row}"
            )
        rows.append(row)
    return {"status": "PASS_TABLE_6_20_TRACE", "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tape-indices", type=int, nargs="*", default=None)
    args = parser.parse_args()

    require_clean_worktree()
    if args.output_dir.exists():
        raise RuntimeError("STOP_INSTRUMENT_OUTPUT_DIR_EXISTS")
    args.output_dir.mkdir(parents=True)
    contract = load_contract(args.contract)
    source_hashes = verify_sources(contract, args.source_dir)
    contract_hash = file_sha256(args.contract)
    commit = git_commit()
    selected = (
        list(range(len(contract["execution"]["tape_roots"])))
        if args.tape_indices is None
        else args.tape_indices
    )
    roots = [contract["execution"]["tape_roots"][index] for index in selected]
    opening = {
        "status": "OPENED_DEVELOPMENT",
        "opened_at": datetime.now(timezone.utc).isoformat(),
        "contract_sha256": contract_hash,
        "code_commit": commit,
        "source_hashes": source_hashes,
        "tape_indices": selected,
        "tape_roots": roots,
        "confirmation_roots_opened": False,
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
                f"completed tape_root={tape_root} "
                f"rows={sum(1 for _ in rows_path.open())}",
                flush=True,
            )
    rows = [
        json.loads(line) for line in rows_path.read_text().splitlines()
        if line.strip()
    ]
    expected = 90 * len(roots)
    if len(rows) != expected:
        raise RuntimeError("STOP_INSTRUMENT_INCOMPLETE_MATRIX")
    identities = {
        (row["tape_root"], row["cf"]) for row in rows
    }
    if len(identities) != expected:
        raise RuntimeError("STOP_INSTRUMENT_PAIRING")
    completion = {
        "status": "COMPLETE_VALID_SHARD",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "contract_sha256": contract_hash,
        "code_commit": commit,
        "row_count": len(rows),
        "rows_sha256": file_sha256(rows_path),
        "row_identity_digest": canonical_sha256(sorted(identities)),
        "tape_roots": roots,
        "confirmation_roots_opened": False,
    }
    (args.output_dir / "completion_receipt.json").write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(completion, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
