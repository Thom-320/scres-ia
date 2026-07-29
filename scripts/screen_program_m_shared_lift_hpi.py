#!/usr/bin/env python3
"""Program M complete-calendar, burned-development H_PI screen.

This producer is deliberately pre-result infrastructure.  It enumerates all
2**8 reservation calendars in each of the 19 frozen cells, using the real MFSC
DES and the request-snapshot-v2 ReT ledger.  Screen results select candidate
regions only: this script computes no confidence interval and makes no
promotion, Paper 2, learner, or virgin-tape claim.

The contract's 7,300,001+ blocks are opened only by an explicit CLI run.  Unit
tests import the pure helpers and use synthetic, non-project seeds.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from hashlib import sha256
import itertools
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from supply_chain.config import HOURS_PER_WEEK, SIMULATION_HORIZON  # noqa: E402
from supply_chain.episode_metrics import compute_episode_metrics  # noqa: E402
from supply_chain.program_m_shared_lift import (  # noqa: E402
    ACTION_NAMES,
    ACTIVATION_LEAD_HOURS,
    DECISION_WEEKS,
    LOCAL_BYPASS_OPS,
    SLOT_WINDOW_HOURS,
    ProgramMSharedLiftSimulation,
)
from supply_chain.ret_thesis import (  # noqa: E402
    compute_order_level_ret_excel_request_snapshot_ledger,
)
from supply_chain.supply_chain import MFSCSimulation  # noqa: E402


CONTRACT_PATH = ROOT / "contracts/program_m_shared_lift_reservation_v1.json"
PROXY_PATH = ROOT / "supply_chain/data/garrido_proxy_v1_freeze_2026-07-10.json"
METRIC_ID = "ret_excel_request_snapshot_v2"
SCREEN_SEEDS = tuple(range(7_300_001, 7_300_025))
POSITIVE_HAZARDS = (0.25, 0.50, 0.75)
DURATIONS = (24, 72, 120)
SIGNALS = ((0.70, 0.80), (0.85, 0.90))
PRACTICAL_GATE = 0.01
CAMPAIGN_HOURS = 1_392.0
EVENT_RISK_ID = "researcher_introduced_localized_access_disruption"
SOURCE_PATHS = (
    "contracts/program_m_shared_lift_reservation_v1.json",
    "scripts/screen_program_m_shared_lift_hpi.py",
    "supply_chain/program_m_shared_lift.py",
    "supply_chain/episode_metrics.py",
    "supply_chain/ret_thesis.py",
    "supply_chain/supply_chain.py",
    "supply_chain/data/garrido_proxy_v1_freeze_2026-07-10.json",
)


@dataclass(frozen=True, order=True)
class Cell:
    cell_id: str
    hazard: float
    duration_hours: int
    sensitivity: float
    specificity: float
    is_null: bool = False


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest_json(value: Any) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    h = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def hash_uniform(seed: int, week: int, stream: str) -> float:
    """Stable independent U[0,1) draw; streams never depend on policy state."""

    raw = sha256(f"program-m-tape-v1:{int(seed)}:{int(week)}:{stream}".encode()).digest()
    return int.from_bytes(raw[:8], "big") / float(2**64)


def frozen_cells() -> tuple[Cell, ...]:
    null = Cell("null-h0_d24_s85", 0.0, 24, 0.85, 0.90, True)
    positive = tuple(
        Cell(
            f"h{int(hazard * 100):02d}_d{duration}_s{int(sensitivity * 100):02d}",
            hazard,
            duration,
            sensitivity,
            specificity,
        )
        for hazard, duration, (sensitivity, specificity) in itertools.product(
            POSITIVE_HAZARDS, DURATIONS, SIGNALS
        )
    )
    cells = (null, *positive)
    assert len(cells) == 19 and sum(not cell.is_null for cell in cells) == 18
    assert sum(cell.hazard == 0.0 for cell in cells) == 1
    return cells


def all_calendars() -> tuple[tuple[str, ...], ...]:
    calendars = tuple(itertools.product(ACTION_NAMES, repeat=DECISION_WEEKS))
    assert len(calendars) == 256
    return calendars


def generate_exogenous_tape(
    *, seed: int, cell: Cell, decision_start_time: float
) -> dict[str, Any]:
    """Materialize policy-independent weekly events with nested hazard CRN.

    The hazard draw, location draw and start draw are separate hash streams.
    None includes hazard, duration, signal, calendar, order or departure data.
    Hence increasing the hazard creates a nested event set for a fixed seed;
    changing duration changes only event end time.
    """

    events: list[dict[str, Any]] = []
    draws: list[dict[str, Any]] = []
    for week in range(DECISION_WEEKS):
        hazard_u = hash_uniform(seed, week, "hazard")
        location_u = hash_uniform(seed, week, "location")
        start_u = hash_uniform(seed, week, "start")
        destination = "A" if location_u < 0.5 else "B"
        decision_time = float(decision_start_time) + week * HOURS_PER_WEEK
        start_time = decision_time + ACTIVATION_LEAD_HOURS + start_u * SLOT_WINDOW_HOURS
        draws.append(
            {
                "week": week,
                "hazard_u": hazard_u,
                "location_u": location_u,
                "start_u": start_u,
                "destination": destination,
                "start_time": start_time,
            }
        )
        if hazard_u >= cell.hazard:
            continue
        events.append(
            {
                "risk_id": EVENT_RISK_ID,
                "start_time": start_time,
                "end_time": start_time + cell.duration_hours,
                "duration": float(cell.duration_hours),
                "affected_ops": sorted(LOCAL_BYPASS_OPS),
                "affected_cssu": destination,
                "description": "Program M destination-local access disruption",
                "magnitude": 1.0,
                "unit": "binary local-route outage",
            }
        )
    return {
        "schema_version": "program_m_shared_lift_exogenous_tape_v1",
        "seed": int(seed),
        "cell_id": cell.cell_id,
        "decision_start_time": float(decision_start_time),
        "draws": draws,
        "risk_events": events,
        "draw_sha256": digest_json(draws),
        "risk_sha256": digest_json(events),
    }


def proxy_kwargs() -> dict[str, Any]:
    payload = json.loads(PROXY_PATH.read_text(encoding="utf-8"))
    kwargs = dict(payload["sim_kwargs"])
    for key in (
        "risk_level",
        "seed_stream_mode",
        "cssu_topology_mode",
        "order_fulfillment_mode",
        "downstream_transport_capacity_mode",
    ):
        kwargs.pop(key, None)
    return kwargs


def determine_warmup_time(seed: int) -> float:
    """Determine the canonical warmup before any Program M event exists."""

    horizon = max(float(SIMULATION_HORIZON), 20_000.0)
    sim = MFSCSimulation(
        seed=int(seed),
        horizon=horizon,
        risks_enabled=False,
        strict_exogenous_crn=True,
        cssu_topology_mode="split_v1",
        order_fulfillment_mode="op9_linked",
        downstream_transport_capacity_mode="tandem_capacity_one",
        **proxy_kwargs(),
    )
    sim._start_processes()
    while not sim.warmup_complete:
        if float(sim.env.now) >= horizon - 1.0:
            raise RuntimeError(f"warmup did not complete for seed {seed}")
        sim.env.run(until=min(float(sim.env.now) + 1.0, horizon))
    return float(sim.warmup_time)


def _quantity_ret_v2(sim: ProgramMSharedLiftSimulation, start: float) -> float:
    orders = [
        order
        for order in sim.orders
        if not bool(getattr(order, "metrics_excluded", False))
        and float(getattr(order, "OPTj", 0.0) or 0.0) >= start
    ]
    ledger = compute_order_level_ret_excel_request_snapshot_ledger(
        orders, current_time=float(sim.env.now)
    )
    denominator = sum(float(row["quantity"]) for row in ledger["ret_rows"])
    if denominator <= 0.0:
        return 1.0
    return float(
        sum(float(row["ret"]) * float(row["quantity"]) for row in ledger["ret_rows"])
        / denominator
    )


def _worst_cssu_fill(sim: ProgramMSharedLiftSimulation, start: float) -> tuple[float, dict[str, float]]:
    fills: dict[str, float] = {}
    for destination in ("A", "B"):
        demand = sum(
            float(qty)
            for time, dest, qty in sim.cssu_demand_events
            if dest == destination and float(time) >= start - 1e-9
        )
        delivered = sum(
            float(qty)
            for time, dest, qty in sim.cssu_delivery_events
            if dest == destination and float(time) >= start - 1e-9
        )
        fills[destination] = min(1.0, delivered / demand) if demand > 0.0 else 1.0
    return min(fills.values()), fills


def _campaign_orders(sim: ProgramMSharedLiftSimulation, start: float) -> list[Any]:
    return [
        order
        for order in sim.orders
        if not bool(getattr(order, "metrics_excluded", False))
        and float(getattr(order, "OPTj", 0.0) or 0.0) >= start
    ]


def run_calendar(
    *, seed: int, cell: Cell, calendar_index: int, warmup_time: float
) -> dict[str, Any]:
    calendars = all_calendars()
    calendar = calendars[int(calendar_index)]
    tape = generate_exogenous_tape(
        seed=int(seed), cell=cell, decision_start_time=float(warmup_time)
    )
    horizon = float(warmup_time) + CAMPAIGN_HOURS
    sim = ProgramMSharedLiftSimulation(
        seed=int(seed),
        horizon=horizon,
        risks_enabled=True,
        risk_event_tape=tape["risk_events"],
        strict_exogenous_crn=True,
        program_m_enabled=True,
        reservation_calendar=calendar,
        decision_start_time=float(warmup_time),
        warning_sensitivity=cell.sensitivity,
        warning_specificity=cell.specificity,
        cssu_topology_mode="split_v1",
        order_fulfillment_mode="op9_linked",
        downstream_transport_capacity_mode="tandem_capacity_one",
        **proxy_kwargs(),
    ).run()
    if not math.isclose(float(sim.warmup_time), float(warmup_time), abs_tol=1e-9):
        raise AssertionError("pre-risk warmup anchor drifted during policy replay")
    sim.assert_complete_calendar()
    metrics = compute_episode_metrics(
        sim,
        treatment_start=float(warmup_time),
        ret_excel_contract_version=METRIC_ID,
    )
    if metrics["ret_excel_contract_version"] != METRIC_ID:
        raise AssertionError("canonical metric contract drift")
    ledger = sim.program_m_ledger()
    resources = ledger["resources"]
    flow = ledger["flow_ledger"]
    worst_fill, cssu_fill = _worst_cssu_fill(sim, float(warmup_time))
    orders = _campaign_orders(sim, float(warmup_time))
    protected_order_ids = {
        int(row["order_j"])
        for row in ledger["movements"]
        if row["loaded"] and row["order_j"] is not None
    }
    released_order_ids = {
        int(order.j) for order in orders if order.op9_release_time is not None
    }
    base_departures = len(released_order_ids - protected_order_ids)
    result = {
        "calendar_index": int(calendar_index),
        "calendar": list(calendar),
        "ret_request_snapshot_v2": float(metrics["ret_excel_visible"]),
        "quantity_ret_request_snapshot_v2": _quantity_ret_v2(sim, float(warmup_time)),
        "service_loss_auc_ration_hours": float(metrics["service_loss_auc_ration_hours"]),
        "attended_orders": int(metrics["n_served"]),
        "lost_orders": int(metrics["n_lost"]),
        "remaining_backlog_quantity": float(metrics["backorder_qty_final"]),
        "remaining_backlog_orders": sum(
            int(order.OATj is None and not bool(order.lost)) for order in orders
        ),
        "maximum_backlog_age_hours": float(metrics["backlog_age_max"]),
        "worst_cssu_fill": float(worst_fill),
        "cssu_fill": cssu_fill,
        "ret_cvar05": float(metrics["ret_excel_cvar05"]),
        "mass_residual": max(abs(float(flow["raw_residual"])), abs(float(flow["ration_residual"]))),
        "reserved_slots": int(resources["reserved_slots"]),
        "reserved_payload_capacity_rations": float(resources["reserved_payload_capacity_rations"]),
        "reserved_vehicle_hours": float(resources["reserved_vehicle_hours"]),
        "protected_loaded_departures": int(resources["loaded_departures"]),
        "protected_empty_departures": int(resources["empty_departures"]),
        "protected_actual_payload_rations": float(resources["actual_payload_rations"]),
        "protected_actual_loaded_vehicle_hours": float(resources["actual_loaded_vehicle_hours"]),
        "base_terrestrial_departures": int(base_departures),
        "base_terrestrial_payload_rations": float(
            sum(
                float(order.quantity)
                for order in orders
                if int(order.j) in released_order_ids - protected_order_ids
            )
        ),
        "base_terrestrial_vehicle_hours": float(48.0 * base_departures),
        "risk_sha256": tape["risk_sha256"],
        "demand_sha256": digest_json(
            [
                [round(float(time), 9), destination, round(float(qty), 9)]
                for time, destination, qty in sim.cssu_demand_events
                if float(time) >= float(warmup_time) - 1e-9
            ]
        ),
    }
    if result["reserved_slots"] != 8 or result["reserved_vehicle_hours"] != 384.0:
        raise AssertionError("fixed protected resource envelope violated")
    return result


def evaluate_cell_tape(task: Mapping[str, Any]) -> dict[str, Any]:
    cell = Cell(**task["cell"])
    seed = int(task["seed"])
    warmup = determine_warmup_time(seed)
    tape = generate_exogenous_tape(seed=seed, cell=cell, decision_start_time=warmup)
    rows = [
        run_calendar(
            seed=seed, cell=cell, calendar_index=index, warmup_time=warmup
        )
        for index in range(len(all_calendars()))
    ]
    risk_hashes = {row["risk_sha256"] for row in rows}
    demand_hashes = {row["demand_sha256"] for row in rows}
    if risk_hashes != {tape["risk_sha256"]} or len(demand_hashes) != 1:
        raise AssertionError("CRN tape drift across calendars")
    return {
        "schema_version": "program_m_shared_lift_hpi_raw_shard_v1",
        "cell": asdict(cell),
        "seed": seed,
        "warmup_time": warmup,
        "tape": tape,
        "n_calendars": len(rows),
        "evaluations": rows,
    }


def adjacent(left: Cell, right: Cell) -> bool:
    if left.is_null or right.is_null:
        return False
    factors = (
        (POSITIVE_HAZARDS, left.hazard, right.hazard),
        (DURATIONS, left.duration_hours, right.duration_hours),
        (SIGNALS, (left.sensitivity, left.specificity), (right.sensitivity, right.specificity)),
    )
    changed = 0
    for levels, a, b in factors:
        if a == b:
            continue
        changed += 1
        if abs(levels.index(a) - levels.index(b)) != 1:
            return False
    return changed == 1


def candidate_regions(cell_summaries: Sequence[Mapping[str, Any]]) -> list[list[str]]:
    """Selection-only connected components clearing the raw mean screen gate."""

    by_id = {cell.cell_id: cell for cell in frozen_cells()}
    eligible = {
        row["cell_id"]
        for row in cell_summaries
        if not by_id[row["cell_id"]].is_null
        and float(row["h_pi_mean"]) >= PRACTICAL_GATE
    }
    components: list[list[str]] = []
    while eligible:
        pending = [min(eligible)]
        eligible.remove(pending[0])
        component: list[str] = []
        while pending:
            current = pending.pop()
            component.append(current)
            neighbors = {
                other
                for other in eligible
                if adjacent(by_id[current], by_id[other])
            }
            eligible -= neighbors
            pending.extend(sorted(neighbors))
        cells = [by_id[cell_id] for cell_id in component]
        if (
            len(cells) >= 3
            and len({cell.hazard for cell in cells}) >= 2
            and len({cell.duration_hours for cell in cells}) >= 2
        ):
            components.append(sorted(component))
    return sorted(components)


def summarize_cell(shards: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not shards:
        raise ValueError("cannot summarize an empty cell")
    cell_id = shards[0]["cell"]["cell_id"]
    if any(shard["cell"]["cell_id"] != cell_id for shard in shards):
        raise ValueError("mixed cells")
    n_calendars = 256
    matrix = np.asarray(
        [
            [float(row["ret_request_snapshot_v2"]) for row in shard["evaluations"]]
            for shard in shards
        ],
        dtype=float,
    )
    if matrix.shape != (len(shards), n_calendars):
        raise AssertionError("incomplete 256-calendar frontier")
    calendar_means = matrix.mean(axis=0)
    static_index = int(np.flatnonzero(calendar_means == calendar_means.max())[0])
    oracle_indices = np.argmax(matrix, axis=1)
    deltas = matrix[np.arange(len(shards)), oracle_indices] - matrix[:, static_index]
    oracle_ret = matrix[np.arange(len(shards)), oracle_indices]
    static_ret = matrix[:, static_index]
    tail_n = max(1, int(math.ceil(0.05 * len(shards))))
    endpoint_names = (
        "quantity_ret_request_snapshot_v2",
        "service_loss_auc_ration_hours",
        "attended_orders",
        "lost_orders",
        "remaining_backlog_quantity",
        "remaining_backlog_orders",
        "maximum_backlog_age_hours",
        "worst_cssu_fill",
        "ret_cvar05",
        "mass_residual",
        "reserved_slots",
        "reserved_payload_capacity_rations",
        "reserved_vehicle_hours",
        "protected_loaded_departures",
        "protected_empty_departures",
        "protected_actual_payload_rations",
        "protected_actual_loaded_vehicle_hours",
        "base_terrestrial_departures",
        "base_terrestrial_payload_rations",
        "base_terrestrial_vehicle_hours",
    )
    endpoint_deltas: dict[str, Any] = {}
    for endpoint in endpoint_names:
        values = np.asarray(
            [[float(row[endpoint]) for row in shard["evaluations"]] for shard in shards]
        )
        paired = values[np.arange(len(shards)), oracle_indices] - values[:, static_index]
        endpoint_deltas[endpoint] = {
            "mean_oracle_minus_static": float(paired.mean()),
            "per_tape": paired.tolist(),
        }
    cell = shards[0]["cell"]
    return {
        "cell_id": cell_id,
        "cell": cell,
        "n_tapes": len(shards),
        "n_calendars": n_calendars,
        "best_static_calendar_index": static_index,
        "best_static_calendar": list(all_calendars()[static_index]),
        "oracle_calendar_indices": oracle_indices.astype(int).tolist(),
        "unique_oracle_calendars": int(len(set(map(int, oracle_indices)))),
        "h_pi_mean": float(deltas.mean()),
        "h_pi_per_tape": deltas.tolist(),
        "tape_tail_cvar05": {
            "tail_n": tail_n,
            "oracle": float(np.sort(oracle_ret)[:tail_n].mean()),
            "static": float(np.sort(static_ret)[:tail_n].mean()),
            "marginal_delta": float(
                np.sort(oracle_ret)[:tail_n].mean()
                - np.sort(static_ret)[:tail_n].mean()
            ),
            "paired_h_pi_lower_tail_mean": float(np.sort(deltas)[:tail_n].mean()),
        },
        "screen_gate_mean_only": bool(float(deltas.mean()) >= PRACTICAL_GATE),
        "endpoint_deltas_oracle_minus_static": endpoint_deltas,
        "max_abs_mass_residual": float(
            max(float(row["mass_residual"]) for shard in shards for row in shard["evaluations"])
        ),
        "risk_hashes_by_tape": [shard["tape"]["risk_sha256"] for shard in shards],
    }


def source_manifest() -> dict[str, str]:
    return {path: file_sha256(ROOT / path) for path in SOURCE_PATHS}


def build_run_contract(*, seeds: Sequence[int]) -> dict[str, Any]:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    if tuple(seeds) != SCREEN_SEEDS:
        split = "synthetic_test_or_explicit_noncanonical"
    else:
        split = "burned_screen"
    payload = {
        "schema_version": "program_m_shared_lift_hpi_run_contract_v1",
        "contract_id": contract["contract_id"],
        "governing_metric": METRIC_ID,
        "scientific_role": "screen_selection_only_no_ci_no_promotion",
        "seed_role": split,
        "seeds": list(map(int, seeds)),
        "cells": [asdict(cell) for cell in frozen_cells()],
        "calendars": [list(calendar) for calendar in all_calendars()],
        "campaign_hours": CAMPAIGN_HOURS,
        "treatment_start_rule": "per-seed canonical warmup determined before risk; absolute thereafter",
        "adjacency": "one ordered factor step in exactly one positive-cell factor",
        "candidate_region_rule": contract["risk_and_signal_design"]["connected_region_rule"],
        "source_sha256": source_manifest(),
    }
    payload["content_sha256"] = digest_json(payload)
    return payload


def validate_resume(run_dir: Path, expected_contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    frozen = json.loads((run_dir / "run_contract.json").read_text(encoding="utf-8"))
    if frozen != expected_contract:
        raise RuntimeError("resume run contract or source hash mismatch")
    progress_path = run_dir / "progress.json"
    if not progress_path.exists():
        raise FileNotFoundError("--resume requires progress.json")
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    for record in progress.get("completed_shards", []):
        path = run_dir / record["path"]
        if not path.is_file() or file_sha256(path) != record["sha256"]:
            raise RuntimeError(f"resume shard custody mismatch: {record['path']}")
    return list(progress.get("completed_shards", []))


def execute(
    *,
    run_dir: Path,
    seeds: Sequence[int],
    workers: int,
    resume: bool,
    evaluator: Callable[[Mapping[str, Any]], dict[str, Any]] = evaluate_cell_tape,
) -> dict[str, Any]:
    run_contract = build_run_contract(seeds=seeds)
    contract_path = run_dir / "run_contract.json"
    progress_path = run_dir / "progress.json"
    if resume:
        completed = validate_resume(run_dir, run_contract)
    else:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError("refusing to overwrite non-empty run directory")
        run_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(contract_path, run_contract)
        completed = []
        atomic_json(
            progress_path,
            {
                "schema_version": "program_m_shared_lift_hpi_progress_v1",
                "run_contract_sha256": file_sha256(contract_path),
                "completed_shards": [],
                "complete": False,
            },
        )
    completed_keys = {(row["cell_id"], int(row["seed"])) for row in completed}
    tasks = [
        {"cell": asdict(cell), "seed": int(seed)}
        for cell in frozen_cells()
        for seed in seeds
        if (cell.cell_id, int(seed)) not in completed_keys
    ]

    def persist(payload: Mapping[str, Any]) -> None:
        cell_id = payload["cell"]["cell_id"]
        seed = int(payload["seed"])
        relative = Path("raw") / cell_id / f"seed_{seed}.json"
        path = run_dir / relative
        atomic_json(path, payload)
        completed.append(
            {
                "cell_id": cell_id,
                "seed": seed,
                "path": relative.as_posix(),
                "sha256": file_sha256(path),
            }
        )
        completed.sort(key=lambda row: (row["cell_id"], int(row["seed"])))
        atomic_json(
            progress_path,
            {
                "schema_version": "program_m_shared_lift_hpi_progress_v1",
                "run_contract_sha256": file_sha256(contract_path),
                "completed_shards": completed,
                "completed_count": len(completed),
                "total_count": len(frozen_cells()) * len(seeds),
                "complete": False,
            },
        )

    if int(workers) <= 1 or evaluator is not evaluate_cell_tape:
        for task in tasks:
            persist(evaluator(task))
    else:
        with ProcessPoolExecutor(max_workers=int(workers)) as pool:
            futures = {pool.submit(evaluate_cell_tape, task): task for task in tasks}
            for future in as_completed(futures):
                persist(future.result())

    shards_by_cell: dict[str, list[dict[str, Any]]] = {
        cell.cell_id: [] for cell in frozen_cells()
    }
    for record in completed:
        shards_by_cell[record["cell_id"]].append(
            json.loads((run_dir / record["path"]).read_text(encoding="utf-8"))
        )
    summaries = [
        summarize_cell(sorted(shards_by_cell[cell.cell_id], key=lambda row: row["seed"]))
        for cell in frozen_cells()
    ]
    regions = candidate_regions(summaries)
    result = {
        "schema_version": "program_m_shared_lift_hpi_screen_v1",
        "status": "SCREEN_COMPLETE_SELECTION_ONLY__NO_CI_NO_PROMOTION",
        "run_contract_sha256": file_sha256(contract_path),
        "governing_metric": METRIC_ID,
        "n_cells": len(summaries),
        "n_tapes_per_cell": len(seeds),
        "n_calendars": 256,
        "n_des_evaluations": len(summaries) * len(seeds) * 256,
        "cell_summaries": summaries,
        "candidate_regions_mean_screen_only": regions,
        "selection_note": (
            "Raw screen means may nominate a region only. No CI was computed; "
            "no cell passes validation or authorizes H_obs, a learner, Paper 2, "
            "Paper 3, or virgin tapes."
        ),
    }
    atomic_json(run_dir / "result.json", result)
    atomic_json(
        progress_path,
        {
            "schema_version": "program_m_shared_lift_hpi_progress_v1",
            "run_contract_sha256": file_sha256(contract_path),
            "completed_shards": completed,
            "completed_count": len(completed),
            "total_count": len(frozen_cells()) * len(seeds),
            "complete": True,
            "result_sha256": file_sha256(run_dir / "result.json"),
        },
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=ROOT / "outputs/program_m_shared_lift_hpi_screen_v1",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    execute(
        run_dir=args.run_dir.resolve(),
        seeds=SCREEN_SEEDS,
        workers=max(1, int(args.workers)),
        resume=bool(args.resume),
    )


if __name__ == "__main__":
    main()
