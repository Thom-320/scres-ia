#!/usr/bin/env python3
"""Build the sealed 216-posture recovery surface on declared burned-seed replays."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from supply_chain.arm_runner import canonical_payload_sha256, seal_and_write  # noqa: E402
from supply_chain.config import HOURS_PER_WEEK  # noqa: E402
from supply_chain.config import THESIS_FAITHFUL_PROTOCOL as P  # noqa: E402
from supply_chain.episode_metrics import compute_episode_metrics  # noqa: E402
from supply_chain.expanded_contract_controllers_v2 import (  # noqa: E402
    ALL_POSTURES,
    NODES,
    Posture,
    posture_targets,
)
from supply_chain.garrido_v0_recovery import (  # noqa: E402
    CONTEXT_ORDER,
    RECOVERY_CONSECUTIVE_DAYS,
    RECOVERY_FRACTION,
    RECOVERY_WINDOW_HOURS,
    placebo_event_rows,
    recovery_utility,
    restricted_recovery_summary,
    risk_event_rows,
)
from supply_chain.resilience_temporal import compute_temporal_resilience_panel  # noqa: E402
from supply_chain.seed_custody import custody_falsifier, module_manifest  # noqa: E402
from supply_chain.supply_chain import MFSCSimulation  # noqa: E402


GRID_ID = "garrido_v0_recovery216_v1"
EVENT_ONSET_HOURS = 4031.0
HORIZON_HOURS = 36.0 * HOURS_PER_WEEK
SEED_BASE = 5_300_001
CONTRACT = Path("docs/PREREGISTRO_GARRIDO_V0_RECOVERY_SURFACE_V1_2026-08-06.md")
REFERENCE = Path("results/garrido_v0_recovery_gate_v2/result.json")
MODULES = (
    "supply_chain/garrido_v0_recovery.py",
    "supply_chain/resilience_temporal.py",
    "supply_chain/episode_metrics.py",
    "supply_chain/supply_chain.py",
    "supply_chain/expanded_contract_controllers_v2.py",
    "supply_chain/arm_runner.py",
    "supply_chain/seed_custody.py",
)
EPISODE_KEYS = (
    "demanded_rations",
    "delivered_rations",
    "flow_fill_rate",
    "service_loss_auc_ration_hours",
    "backorder_qty_final",
    "lost_orders",
    "ret_excel",
    "ret_excel_full_ledger",
    "ret_excel_risk_conditional",
)


def canonical_sha(value: Any) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def order_tape(sim: MFSCSimulation) -> list[dict[str, Any]]:
    return [
        {
            "j": int(order.j),
            "OPTj": float(order.OPTj),
            "Q": float(order.quantity),
            "contingent": bool(order.contingent),
        }
        for order in sim.orders
    ]


def evaluate_simulation(
    *, seed: int, posture: Posture, events: list[dict[str, Any]], horizon: float
) -> dict[str, Any]:
    sim = MFSCSimulation(
        shifts=1,
        initial_buffers={node: 0.0 for node in NODES},
        inventory_replenishment_period=168.0,
        seed=int(seed),
        horizon=float(horizon),
        risks_enabled=True,
        enabled_risks=set(),
        strict_exogenous_crn=True,
        year_basis=P["year_basis"],
        warmup_trigger=P["warmup_trigger"],
        r14_defect_mode=P["r14_defect_mode"],
        risk_event_tape=events,
    )
    sim.inventory_buffer_targets.update(posture_targets(posture))
    sim.step(action=None, step_hours=horizon)
    episode = compute_episode_metrics(sim)
    temporal = compute_temporal_resilience_panel(
        sim,
        cluster_window_hours=RECOVERY_WINDOW_HOURS,
        recovery_fraction=RECOVERY_FRACTION,
        recovery_consecutive_days=RECOVERY_CONSECUTIVE_DAYS,
    )
    tape = order_tape(sim)
    return {
        "episode": {key: float(episode[key]) for key in EPISODE_KEYS},
        "temporal": temporal,
        "risk_event_count": int(len(sim.risk_events)),
        "order_tape_sha256": canonical_sha(tape),
        "n_orders": int(len(tape)),
    }


def derive_cell(
    *, posture: Posture, risk: Mapping[str, Any], placebo: Mapping[str, Any]
) -> dict[str, Any]:
    recovery = restricted_recovery_summary(risk["temporal"], placebo["temporal"])
    utility = recovery_utility(
        recovery,
        demanded_rations=float(risk["episode"]["demanded_rations"]),
        flow_fill_rate=float(risk["episode"]["flow_fill_rate"]),
    )
    return {
        "posture": list(posture),
        "utility": float(utility),
        "recovery": recovery,
        "risk": dict(risk),
        "placebo": dict(placebo),
    }


def scientific_hash(payload: Mapping[str, Any]) -> str:
    return canonical_payload_sha256(
        payload,
        extra_exclude=frozenset(
            {
                "scientific_payload_sha256",
                "cache_custody",
                "contract_sha256",
                "reference_sha256",
            }
        ),
    )


def verify_surface(payload: Mapping[str, Any]) -> None:
    if payload.get("grid_id") != GRID_ID:
        raise ValueError("wrong recovery grid id")
    if tuple(payload.get("context_order", ())) != CONTEXT_ORDER:
        raise ValueError("context order drifted")
    expected = [list(posture) for posture in ALL_POSTURES]
    placebo = payload.get("placebo_cells")
    contexts = payload.get("contexts")
    if not isinstance(placebo, list) or [row.get("posture") for row in placebo] != expected:
        raise ValueError("placebo surface is incomplete or out of order")
    if (
        not isinstance(contexts, Mapping)
        or len(contexts) != len(CONTEXT_ORDER)
        or set(contexts) != set(CONTEXT_ORDER)
    ):
        raise ValueError("risk surface is incomplete or out of order")
    placebo_order_hashes = {
        row["panel"]["order_tape_sha256"] for row in placebo
    }
    if len(placebo_order_hashes) != 1:
        raise ValueError("placebo violates exogenous-demand CRN across postures")
    for context in CONTEXT_ORDER:
        cells = contexts[context]
        if not isinstance(cells, list) or [row.get("posture") for row in cells] != expected:
            raise ValueError(f"{context} surface is incomplete or out of order")
        if any(int(row["risk"]["temporal"]["system_ttr_n_clusters"]) != 1 for row in cells):
            raise ValueError(f"{context} does not have one recovery cluster per cell")
        if len({row["risk"]["order_tape_sha256"] for row in cells}) != 1:
            raise ValueError(f"{context} violates exogenous-demand CRN across postures")
        for index, row in enumerate(cells):
            ttr = float(row["recovery"]["restricted_ttr_hours"])
            if not 0.0 <= ttr <= RECOVERY_WINDOW_HOURS + 1e-9:
                raise ValueError(f"{context} has an invalid restricted TTR")
            if row["placebo"]["order_tape_sha256"] != placebo[index]["panel"]["order_tape_sha256"]:
                raise ValueError(f"{context} is not paired to its posture placebo")
    stored_science = payload.get("scientific_payload_sha256")
    if not stored_science or stored_science != scientific_hash(payload):
        raise ValueError("scientific payload hash is invalid")
    stored_self = payload.get("self_sha256")
    if stored_self:
        body = json.dumps(
            {key: value for key, value in payload.items() if key != "self_sha256"},
            indent=1,
            sort_keys=True,
            default=str,
        )
        if sha256(body.encode()).hexdigest() != stored_self:
            raise ValueError("surface envelope hash is invalid")


def build_seed(seed: int, horizon: float) -> dict[str, Any]:
    started = time.perf_counter()
    placebo_runs: dict[Posture, dict[str, Any]] = {}
    placebo_events = placebo_event_rows(onset_hours=EVENT_ONSET_HOURS)
    for posture in ALL_POSTURES:
        placebo_runs[posture] = evaluate_simulation(
            seed=seed, posture=posture, events=placebo_events, horizon=horizon
        )

    contexts: dict[str, list[dict[str, Any]]] = {}
    for context in CONTEXT_ORDER:
        events = risk_event_rows(context, onset_hours=EVENT_ONSET_HOURS)
        cells: list[dict[str, Any]] = []
        for posture in ALL_POSTURES:
            risk = evaluate_simulation(
                seed=seed, posture=posture, events=events, horizon=horizon
            )
            cells.append(
                derive_cell(posture=posture, risk=risk, placebo=placebo_runs[posture])
            )
        contexts[context] = cells

    payload: dict[str, Any] = {
        "schema_version": "garrido_v0_recovery_surface_v1",
        "claim_status": "DEVELOPMENT_SURFACE_DECLARED_BURNED_REPLAY",
        "scope": "FULL_PANEL_CACHE_NO_HYPOTHESIS_ADJUDICATION",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "grid_id": GRID_ID,
        "seed": int(seed),
        "replay_of": "garrido_q2_des288",
        "horizon_hours": float(horizon),
        "event_onset_hours": EVENT_ONSET_HOURS,
        "recovery_window_hours": RECOVERY_WINDOW_HOURS,
        "demand_mode": "natural_split_rng_strict_exogenous_crn",
        "posture_levels_hours": [0, 168, 336, 504, 672, 1344],
        "posture_nodes": list(NODES),
        "context_order": list(CONTEXT_ORDER),
        "placebo_event_tape_sha256": canonical_sha(placebo_events),
        "placebo_cells": [
            {"posture": list(posture), "panel": placebo_runs[posture]}
            for posture in ALL_POSTURES
        ],
        "contexts": contexts,
        "module_manifest": module_manifest(MODULES, script=__file__),
        "cache_custody": custody_falsifier(
            [seed], replay_of="garrido_q2_des288"
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    payload["scientific_payload_sha256"] = scientific_hash(payload)
    verify_surface(payload)
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed-start", type=int, default=SEED_BASE)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--horizon-weeks", type=float, default=36.0)
    ap.add_argument("--contract", type=Path, default=CONTRACT)
    ap.add_argument("--reference", type=Path, default=REFERENCE)
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/surface_cache") / GRID_ID / "development",
    )
    args = ap.parse_args()
    horizon = float(args.horizon_weeks) * HOURS_PER_WEEK
    if horizon < EVENT_ONSET_HOURS + RECOVERY_WINDOW_HOURS:
        raise ValueError("horizon does not include the frozen recovery window")
    seeds = [args.seed_start + offset for offset in range(args.seeds)]
    total_started = time.perf_counter()
    for seed in seeds:
        output = args.output_dir / f"{seed}.json"
        if output.exists():
            verify_surface(json.loads(output.read_text()))
            print(f"  seed {seed}: verified existing slice", flush=True)
            continue
        payload = build_seed(seed, horizon)
        digest = seal_and_write(
            payload, output, contract=args.contract, reference=args.reference
        )
        verify_surface(json.loads(output.read_text()))
        print(
            f"  seed {seed}: {len(ALL_POSTURES) * (len(CONTEXT_ORDER) + 1):,} episodes "
            f"({time.perf_counter() - total_started:.1f}s; {digest[:12]}...)",
            flush=True,
        )
    print(f"  complete in {time.perf_counter() - total_started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
