#!/usr/bin/env python3
"""Corrected recovery gate: clock-supported shocks and natural-demand CRN."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from supply_chain.arm_runner import seal_and_write  # noqa: E402
from supply_chain.config import HOURS_PER_WEEK  # noqa: E402
from supply_chain.config import THESIS_FAITHFUL_PROTOCOL as P  # noqa: E402
from supply_chain.episode_metrics import compute_episode_metrics  # noqa: E402
from supply_chain.expanded_contract_controllers_v2 import (  # noqa: E402
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


EVENT_ONSET_HOURS = 4031.0
HORIZON_HOURS = 36.0 * HOURS_PER_WEEK
SENTINEL_POSTURES: tuple[Posture, ...] = (
    (0, 0, 0),
    (1344, 0, 0),
    (0, 1344, 0),
    (0, 0, 1344),
    (336, 504, 672),
    (1344, 1344, 1344),
)
REFERENCE = Path("results/garrido_v0_recovery_gate_v1/smoke_result.json")
MODULES = (
    "supply_chain/garrido_v0_recovery.py",
    "supply_chain/resilience_temporal.py",
    "supply_chain/episode_metrics.py",
    "supply_chain/supply_chain.py",
    "supply_chain/expanded_contract_controllers_v2.py",
    "supply_chain/arm_runner.py",
    "supply_chain/seed_custody.py",
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


def evaluate(
    *,
    seed: int,
    posture: Posture,
    events: list[dict[str, Any]],
    horizon: float,
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
        "episode": {
            key: float(episode[key])
            for key in (
                "demanded_rations",
                "delivered_rations",
                "flow_fill_rate",
                "service_loss_auc_ration_hours",
                "backorder_qty_final",
                "ret_excel_full_ledger",
            )
        },
        "temporal": temporal,
        "risk_event_count": len(sim.risk_events),
        "order_tape_sha256": canonical_sha(tape),
        "n_orders": len(tape),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default="5300001,5300002")
    ap.add_argument("--horizon-weeks", type=float, default=36.0)
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--replay-of", default="garrido_q2_des288")
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("results/garrido_v0_recovery_gate_v2/result.json"),
    )
    args = ap.parse_args()
    started = time.perf_counter()
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    horizon = float(args.horizon_weeks) * HOURS_PER_WEEK
    if horizon < EVENT_ONSET_HOURS + RECOVERY_WINDOW_HOURS:
        raise ValueError("horizon must include the full preregistered recovery window")

    rows: list[dict[str, Any]] = []
    placebo_hashes: dict[int, dict[str, str]] = {}
    for seed in seeds:
        placebo_by_posture: dict[Posture, dict[str, Any]] = {}
        for posture in SENTINEL_POSTURES:
            placebo_by_posture[posture] = evaluate(
                seed=seed,
                posture=posture,
                events=placebo_event_rows(onset_hours=EVENT_ONSET_HOURS),
                horizon=horizon,
            )
        placebo_hashes[seed] = {
            str(posture): result["order_tape_sha256"]
            for posture, result in placebo_by_posture.items()
        }
        for context in CONTEXT_ORDER:
            events = risk_event_rows(context, onset_hours=EVENT_ONSET_HOURS)
            for posture in SENTINEL_POSTURES:
                risk = evaluate(
                    seed=seed,
                    posture=posture,
                    events=events,
                    horizon=horizon,
                )
                placebo = placebo_by_posture[posture]
                recovery = restricted_recovery_summary(
                    risk["temporal"], placebo["temporal"]
                )
                utility = recovery_utility(
                    recovery,
                    demanded_rations=risk["episode"]["demanded_rations"],
                    flow_fill_rate=risk["episode"]["flow_fill_rate"],
                )
                rows.append(
                    {
                        "seed": seed,
                        "context": context,
                        "posture": list(posture),
                        "order_tape_sha256": risk["order_tape_sha256"],
                        "placebo_order_tape_sha256": placebo["order_tape_sha256"],
                        "risk_event_tape_sha256": canonical_sha(events),
                        "n_clusters": int(risk["temporal"]["system_ttr_n_clusters"]),
                        "utility": utility,
                        **risk["episode"],
                        **recovery,
                    }
                )
        print(f"  seed {seed} complete ({time.perf_counter() - started:.1f}s)", flush=True)

    by_context: dict[str, Any] = {}
    impact_contexts = 0
    responsive_contexts = 0
    for context in CONTEXT_ORDER:
        selected = [row for row in rows if row["context"] == context]
        excess_auc = np.asarray(
            [row["excess_service_loss_auc_ration_hours"] for row in selected], dtype=float
        )
        excess_drop = np.asarray(
            [row["excess_maximum_service_drop"] for row in selected], dtype=float
        )
        restricted = np.asarray([row["restricted_ttr_hours"] for row in selected], dtype=float)
        utilities = np.asarray([row["utility"] for row in selected], dtype=float)
        impact_flags = np.asarray([bool(row["impacted"]) for row in selected], dtype=bool)
        impact_fraction = float(np.mean(impact_flags))
        auc_norm = np.asarray(
            [
                row["excess_service_loss_auc_ration_hours"]
                / max(1.0, row["demanded_rations"] * RECOVERY_WINDOW_HOURS)
                for row in selected
            ],
            dtype=float,
        )
        responsive = bool(
            np.ptp(restricted) >= 24.0 - 1e-9 or np.ptp(auc_norm) >= 0.01 - 1e-12
        )
        impacted = bool(impact_fraction >= 0.25 and responsive)
        impact_contexts += int(impacted)
        responsive_contexts += int(responsive)
        by_context[context] = {
            "n_cells": len(selected),
            "impact_fraction": impact_fraction,
            "median_excess_service_loss_auc": float(np.median(excess_auc)),
            "median_excess_maximum_service_drop": float(np.median(excess_drop)),
            "restricted_ttr_range_hours": float(np.ptp(restricted)),
            "normalised_excess_auc_range": float(np.ptp(auc_norm)),
            "utility_range": float(np.ptp(utilities)),
            "impacted": impacted,
            "posture_responsive": responsive,
        }

    impacted_rows = [row for row in rows if row["impacted"]]
    recovered_fraction = (
        float(np.mean([row["recovered_within_tau"] for row in impacted_rows]))
        if impacted_rows
        else 0.0
    )
    one_cluster = all(row["n_clusters"] == 1 for row in rows)
    bounded = all(
        0.0 <= row["restricted_ttr_hours"] <= RECOVERY_WINDOW_HOURS + 1e-9
        and 0.0 <= row["utility"] <= 1.001001 + 1e-9
        for row in rows
    )
    order_crn_failures: list[dict[str, Any]] = []
    for seed in seeds:
        placebo_distinct = set(placebo_hashes[seed].values())
        if len(placebo_distinct) != 1:
            order_crn_failures.append(
                {"seed": seed, "context": "PLACEBO", "n_distinct": len(placebo_distinct)}
            )
        for context in CONTEXT_ORDER:
            hashes = {
                row["order_tape_sha256"]
                for row in rows
                if row["seed"] == seed and row["context"] == context
            }
            if len(hashes) != 1:
                order_crn_failures.append(
                    {"seed": seed, "context": context, "n_distinct": len(hashes)}
                )
    custody = custody_falsifier(seeds, replay_of=args.replay_of, exclude=args.output)
    falsifiers = {
        "f1_every_cell_has_one_temporal_cluster": {
            "passed": one_cluster,
            "evidence": {
                "why_it_can_fail": "a missing or split cluster changes the TTR estimand",
                "distinct_cluster_counts": sorted({row["n_clusters"] for row in rows}),
            },
        },
        "f2_natural_demand_is_common_random_numbers": {
            "passed": not order_crn_failures,
            "evidence": {
                "why_it_can_fail": "natural demand replaces the R24-erasing Excel tape; its realised order tape must remain identical across postures",
                "failures": order_crn_failures,
            },
        },
        "f3_restricted_ttr_cannot_reward_censoring": {
            "passed": bounded,
            "evidence": {
                "why_it_can_fail": "the historical instrument assigned zero when every cluster was censored",
                "tau_hours": RECOVERY_WINDOW_HOURS,
                "censored_values": sorted(
                    {
                        row["restricted_ttr_hours"]
                        for row in rows
                        if row["right_censored_at_tau"]
                    }
                ),
            },
        },
        "f4_event_clock_hits_op1_and_op2_cycles": {
            "passed": bool(EVENT_ONSET_HOURS < 4032.0 <= EVENT_ONSET_HOURS + 1.0),
            "evidence": {
                "why_it_can_fail": "v1 placed R12/R13 between procurement cycles, making them decorative",
                "event_onset_hours": EVENT_ONSET_HOURS,
                "op1_rop_hours": 4032.0,
                "op2_rop_hours": 672.0,
            },
        },
        "f5_seed_custody_is_a_declared_replay": custody,
    }
    falsifiers["all_passed"] = all(
        value.get("passed") is True
        for value in falsifiers.values()
        if isinstance(value, dict) and not value.get("not_applicable")
    )
    gates = {
        "g1_shocks_have_incremental_service_effect": {
            "passed": impact_contexts >= 6,
            "observed_contexts": impact_contexts,
            "required_contexts": 6,
        },
        "g2_postures_change_recovery": {
            "passed": responsive_contexts >= 4,
            "observed_contexts": responsive_contexts,
            "required_contexts": 4,
        },
        "g3_recovery_is_observed_somewhere": {
            "passed": recovered_fraction >= 0.25,
            "observed_fraction": recovered_fraction,
            "required_fraction": 0.25,
        },
    }
    gate_passed = bool(falsifiers["all_passed"] and all(g["passed"] for g in gates.values()))
    verdict = "GO_BUILD_V0_RECOVERY_SURFACE" if gate_passed else "STOP_V0_RECOVERY_SURFACE_GATE_FAILED"

    payload = {
        "schema_version": "garrido_v0_recovery_gate_v2",
        "claim_status": verdict,
        "scope": "DEVELOPMENT_GATE_ON_DECLARED_BURNED_SEED_REPLAY_NO_HYPOTHESIS_ADJUDICATION",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "contract_path": str(args.contract),
        "reference_path": str(REFERENCE),
        "module_manifest": module_manifest(MODULES, script=__file__),
        "replay_of": args.replay_of,
        "seeds": seeds,
        "horizon_hours": horizon,
        "event_onset_hours": EVENT_ONSET_HOURS,
        "recovery_window_hours": RECOVERY_WINDOW_HOURS,
        "demand_mode": "natural_split_rng_strict_exogenous_crn",
        "sentinel_postures": [list(posture) for posture in SENTINEL_POSTURES],
        "context_order": list(CONTEXT_ORDER),
        "by_context": by_context,
        "rows": rows,
        "gates": gates,
        "falsifiers": falsifiers,
        "elapsed_seconds": time.perf_counter() - started,
    }
    digest = seal_and_write(
        payload,
        args.output,
        contract=args.contract,
        reference=REFERENCE,
    )
    print(json.dumps({"claim_status": verdict, "gates": gates}, indent=2))
    print(f"  -> {args.output} ({digest[:16]}…)")
    return 0 if falsifiers["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
