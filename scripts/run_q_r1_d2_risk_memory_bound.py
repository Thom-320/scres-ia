#!/usr/bin/env python3
"""Burned direct-SimPy Q-R1 risk-parameter cold-start information bounds.

This is deliberately not a learner experiment.  It asks whether knowing the
Garrido-native current/increased occurrence level changes the best first-two-
week product-mix prefix under a common continuation.  Selection is leave-one-
history-out and never uses the held-out campaign's realized future shocks.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.paper2_exhaustive_search.program_s_design import build_program_s_risk_tape  # noqa: E402
from research.paper2_exhaustive_search.program_s_transducer import run_program_s_direct  # noqa: E402
from scripts.run_program_u1_direct_classical_conversion import scheduler  # noqa: E402
from supply_chain.program_o_full_des_transducer import direct_full_des_vector  # noqa: E402
from supply_chain.program_s_risk_interaction import ProgramSCell  # noqa: E402
from supply_chain.q_r1_retained_learning import early_cohort_metrics_from_orders  # noqa: E402


PERSISTENCE = {"iid": 0.5, "persistent_0p75": 0.75, "persistent_0p90": 0.90}
FAMILIES = ("R24", "R22")
PREFIXES = tuple((a, b) for a in range(4) for b in range(4))
COMMON_CONTINUATION = (2, 2, 2, 2, 2, 2)
RESOURCE_KEYS = (
    "gross_policy_batch_slots",
    "gross_production_quantity",
    "charged_daily_dispatch_slots",
    "charged_downstream_vehicle_hours",
)


def transition_binary(level: int, persistence: float, rng: np.random.Generator) -> int:
    return int(level) if rng.random() < persistence else 1 - int(level)


def risk_level_history(root: int, campaigns: int, mode: str, family: str) -> tuple[int, ...]:
    digest = hashlib.sha256(f"q-r1-d2:{root}:{mode}:{family}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    level = int(rng.integers(0, 2))
    output = []
    for _ in range(campaigns):
        output.append(level)
        level = transition_binary(level, PERSISTENCE[mode], rng)
    return tuple(output)


def cell_for(family: str, level: int) -> ProgramSCell:
    r22 = 3.0 if family == "R22" and level == 1 else 1.0
    r24 = 2.0 if family == "R24" and level == 1 else 1.0
    return ProgramSCell(
        stratum="THESIS_NATIVE_INDEPENDENT",
        mask="LOC_SURGE",
        coupling="independent",
        phi_by_risk={"R22": r22, "R24": r24},
        psi_by_risk={"R22": 1.0, "R24": 1.0},
        r14_probability_multiplier=1.0,
        baseline_capacity_multiplier=1.0,
        regime_persistence=0.90,
        dominant_share=0.90,
        alarm_lead_hours=0.0,
        alarm_balanced_accuracy=0.50,
    )


def clustered_lcb(values: dict[int, list[float]], seed: int = 20260721) -> float:
    roots = sorted(values)
    means = np.asarray([np.mean(values[root]) for root in roots], dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(means), size=(5000, len(means)))
    return float(np.quantile(means[draws].mean(axis=1), 0.025))


def evaluate_all(args: argparse.Namespace) -> list[dict]:
    sched = scheduler()
    rows: list[dict] = []
    started = time.perf_counter()
    for family in FAMILIES:
        for mode in PERSISTENCE:
            for offset in range(args.histories):
                root = args.seed_start + offset
                levels = risk_level_history(root, args.campaigns, mode, family)
                for campaign, level in enumerate(levels):
                    if time.perf_counter() - started > args.hard_cap_seconds:
                        raise TimeoutError("Q-R1 D2 hard cap exceeded")
                    seed = root * 100 + campaign
                    cell = cell_for(family, level)
                    tape = build_program_s_risk_tape(
                        cell, tape_id=seed, horizon_hours=8 * 168
                    )
                    early_events = [
                        event for event in tape["events"]
                        if event["risk_id"] == family and float(event["start_time"]) < 2 * 168
                    ]
                    all_family_events = [
                        event for event in tape["events"] if event["risk_id"] == family
                    ]
                    for prefix in PREFIXES:
                        calendar = prefix + COMMON_CONTINUATION
                        sim = run_program_s_direct(
                            seed=seed,
                            calendar=calendar,
                            scheduler=sched,
                            cell=cell,
                            risk_event_tape=tape["events"],
                        )
                        vector = direct_full_des_vector(sim, sim.product_outcome_panel())
                        early = early_cohort_metrics_from_orders(
                            orders=sim.orders,
                            decision_start=float(sim.program_o_decision_start or 0.0),
                            score_time=float(sim.horizon),
                        )
                        rows.append({
                            "family": family,
                            "persistence_mode": mode,
                            "history_root": root,
                            "campaign_index": campaign,
                            "level": "increased" if level else "current",
                            "prefix": list(prefix),
                            "calendar": list(calendar),
                            "early_family_event_count": len(early_events),
                            "campaign_family_event_count": len(all_family_events),
                            "event_tape_sha256": tape["event_tape_sha256"],
                            **early,
                            "ret_visible": float(vector["ret_visible"]),
                            "worst_product_fill": float(vector["worst_product_fill"]),
                            "lost_orders": float(vector["lost_orders"]),
                            "mass_residual": float(vector["mass_residual"]),
                            "partition_residual": float(vector["partition_residual"]),
                            **{key: float(vector[key]) for key in RESOURCE_KEYS},
                        })
                    print(json.dumps({"family": family, "mode": mode, "root": root, "campaign": campaign}), flush=True)
    return rows


def summarize(rows: list[dict]) -> dict:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["family"], row["persistence_mode"])].append(row)
    output = {}
    for (family, mode), subset in grouped.items():
        roots = sorted({row["history_root"] for row in subset})
        by_episode_prefix = {
            (row["history_root"], row["campaign_index"], tuple(row["prefix"])): row
            for row in subset
        }
        delta_by_root: dict[int, list[float]] = defaultdict(list)
        favorable = []
        divergence = []
        fill = []
        early_omitted_delta = []
        unresolved_delta = []
        lost_delta = []
        resource_error = 0.0
        selected = []
        for test_root in roots:
            train = [row for row in subset if row["history_root"] != test_root]
            global_prefix = max(
                PREFIXES,
                key=lambda prefix: np.mean([row["early_ret_2w"] for row in train if tuple(row["prefix"]) == prefix]),
            )
            level_prefix = {
                level: max(
                    PREFIXES,
                    key=lambda prefix: np.mean([
                        row["early_ret_2w"] for row in train
                        if row["level"] == level and tuple(row["prefix"]) == prefix
                    ]),
                )
                for level in ("current", "increased")
            }
            campaigns = sorted({row["campaign_index"] for row in subset if row["history_root"] == test_root})
            for campaign in campaigns:
                any_row = by_episode_prefix[(test_root, campaign, PREFIXES[0])]
                known = by_episode_prefix[(test_root, campaign, level_prefix[any_row["level"]])]
                reset = by_episode_prefix[(test_root, campaign, global_prefix)]
                delta = known["early_ret_2w"] - reset["early_ret_2w"]
                delta_by_root[test_root].append(delta)
                favorable.append(delta > 0.0)
                divergence.append(tuple(known["prefix"]) != tuple(reset["prefix"]))
                fill.append(known["worst_product_fill"] - reset["worst_product_fill"])
                early_omitted_delta.append(known["early_omitted_rows"] - reset["early_omitted_rows"])
                unresolved_delta.append(known["early_unresolved_orders"] - reset["early_unresolved_orders"])
                lost_delta.append(known["lost_orders"] - reset["lost_orders"])
                resource_error = max(resource_error, *[
                    abs(known[key] - reset[key]) for key in RESOURCE_KEYS
                ])
                selected.append({
                    "history_root": test_root,
                    "campaign_index": campaign,
                    "level": any_row["level"],
                    "known_level_prefix": list(level_prefix[any_row["level"]]),
                    "reset_prefix": list(global_prefix),
                    "delta_early_ret_2w": delta,
                })
        deltas = [value for root in roots for value in delta_by_root[root]]
        level_event_counts = {
            level: [row["early_family_event_count"] for row in subset if row["level"] == level and tuple(row["prefix"]) == PREFIXES[0]]
            for level in ("current", "increased")
        }
        mean_delta = float(np.mean(deltas))
        lcb = clustered_lcb(delta_by_root)
        action_divergence = float(np.mean(divergence))
        gate = (
            mean_delta >= 0.02
            and lcb > 0.0
            and action_divergence >= 0.10
            and resource_error == 0.0
            and float(np.mean(fill)) >= -0.02
            and float(np.mean(early_omitted_delta)) <= 0.0
            and float(np.max(unresolved_delta)) <= 0.0
            and float(np.max(lost_delta)) <= 0.0
        )
        output[f"{family}:{mode}"] = {
            "known_level_minus_reset_mean_early_ret_2w": mean_delta,
            "history_clustered_lcb95": lcb,
            "favorable_fraction": float(np.mean(favorable)),
            "action_divergence": action_divergence,
            "mean_worst_product_delta": float(np.mean(fill)),
            "mean_early_omitted_rows_delta": float(np.mean(early_omitted_delta)),
            "max_unresolved_orders_delta": float(np.max(unresolved_delta)),
            "max_lost_orders_delta": float(np.max(lost_delta)),
            "anti_shedding_pass": (
                float(np.mean(early_omitted_delta)) <= 0.0
                and float(np.max(unresolved_delta)) <= 0.0
                and float(np.max(lost_delta)) <= 0.0
            ),
            "max_resource_error": resource_error,
            "early_event_count_mean_by_level": {
                level: float(np.mean(values)) for level, values in level_event_counts.items()
            },
            "gate_pass": gate,
            "selected_rows": selected,
        }
    r24_pass = any(output[f"R24:{mode}"]["gate_pass"] for mode in PERSISTENCE)
    r22_pass = any(output[f"R22:{mode}"]["gate_pass"] for mode in PERSISTENCE)
    return {
        "by_family_persistence": output,
        "r24_carrier_pass": r24_pass,
        "r22_negative_control_pass": r22_pass,
        "verdict": "PASS_R24_COLD_START_HEADROOM_EXPLORATORY" if r24_pass else "STOP_CARRIER_R24_NO_COLD_START_HEADROOM",
        "learner_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--histories", type=int, default=24)
    parser.add_argument("--campaigns", type=int, default=12)
    parser.add_argument("--seed-start", type=int, default=7_570_601)
    parser.add_argument("--hard-cap-seconds", type=float, default=1800.0)
    parser.add_argument("--output", type=Path, default=ROOT / "results/q_r1/d2_risk_memory_bound_v1/result.json")
    args = parser.parse_args()
    started = time.perf_counter()
    rows = evaluate_all(args)
    summary = summarize(rows)
    payload = {
        "schema_version": "q_r1_d2_risk_memory_bound_v1",
        "claim_status": "EXPLORATORY_NO_CLAIM",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "engine": "direct_simpy_only",
        "selection": "leave_one_history_out_known_level_prefix_vs_level_agnostic_prefix",
        "learner_return_used_for_selection": False,
        "common_continuation_weeks_3_to_8": list(COMMON_CONTINUATION),
        "source_levels": {
            "R24": "Garrido Table 6.12 current 672h vs increased 336h; frequency x1/x2",
            "R22": "Garrido Table 6.12 current 4032h vs increased 1344h; frequency x1/x3",
        },
        "config": vars(args) | {"output": str(args.output)},
        "rows": rows,
        "summary": summary,
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
