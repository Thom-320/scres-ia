#!/usr/bin/env python3
"""Evaluate fixed static and state-rich classical policies on B validation tapes.

This is an exploratory, post-hoc comparator screen. It reuses only the burned
Program B validation tapes and never opens a DES seed. The primary B learner
adjudication remains in evaluate_program_b_service_safe_learner.py; this file
adds a same-tape classical ladder so a learner is not compared only with a weak
static baseline.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from supply_chain.program_o_full_des import run_program_o_full_des_episode  # noqa: E402
from supply_chain.program_o_full_des_transducer import (  # noqa: E402
    direct_full_des_vector,
    extract_full_des_skeleton,
    simulate_full_des_frontier,
)
from supply_chain.program_o_ret_env import (  # noqa: E402
    compute_service_safe_reward,
)
from supply_chain.program_o_state_rich import (  # noqa: E402
    StateRichConfiguration,
    finite_state_rich_configurations,
    state_rich_calendar,
)

SCHEDULER_CONTRACT = ROOT / "contracts/program_o_full_des_hpi_translation_v1.json"
GATE_RESULT = ROOT / "results/program_b_gate_v2/development_service_safe.json"
CONTRACT = ROOT / "contracts/program_b_service_safe_learner_v1.json"
SEEDS = tuple(range(7400097, 7400121))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scheduler() -> dict[str, list[str]]:
    parent = json.loads(SCHEDULER_CONTRACT.read_text())
    key = parent["action"]["primary_scheduler"]
    return parent["action"]["within_week_schedulers"][key]


def bootstrap(values: list[float], seed: int) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(array), size=(20000, len(array)))
    means = array[draws].mean(axis=1)
    return {
        "n": int(len(array)),
        "mean": float(array.mean()),
        "sd": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "lcb95_one_sided": float(np.quantile(means, 0.05)),
        "ucb95_one_sided": float(np.quantile(means, 0.95)),
        "positive_tapes": int((array > 0.0).sum()),
    }


def guardrail_report(
    candidate_rows: list[dict[str, Any]],
    comparator_rows: list[dict[str, Any]],
) -> dict[str, object]:
    rules = {
        "lost_orders_leq": lambda c, b: float(c["lost_orders"]) <= float(b["lost_orders"]) + 1e-12,
        "unresolved_orders_leq": lambda c, b: float(c["unresolved_orders"]) <= float(b["unresolved_orders"]) + 1e-12,
        "delivered_rations_geq": lambda c, b: float(c["delivered_rations"]) + 1e-12 >= float(b["delivered_rations"]),
        "worst_product_fill_geq": lambda c, b: float(c["worst_product_fill"]) + 1e-12 >= float(b["worst_product_fill"]),
        "flow_fill_rate_geq_minus_0_01": lambda c, b: float(c["flow_fill_rate"]) + 1e-12 >= float(b["flow_fill_rate"]) - 0.01,
    }
    failures = {
        name: [
            int(candidate["seed"])
            for candidate, comparator in zip(candidate_rows, comparator_rows)
            if not rule(candidate, comparator)
        ]
        for name, rule in rules.items()
    }
    return {"n": len(candidate_rows), "failures_by_rule": failures, "all_pass": not any(failures.values())}


def evaluate_calendar(seed: int, calendar: tuple[int, ...], sched: dict[str, list[str]]) -> dict[str, float | int]:
    skeleton, _ = extract_full_des_skeleton(
        seed=int(seed),
        scheduler=sched,
        regime_persistence=0.75,
        dominant_share=0.90,
        downstream_freight_physics_mode="fixed_clock_physical_v1",
    )
    matrix = simulate_full_des_frontier(
        skeleton=skeleton,
        scheduler=sched,
        calendars=np.asarray([calendar], dtype=np.uint8),
    )
    metrics = {key: float(value[0]) for key, value in matrix.items()}
    metrics["ret_service_full_clipped"] = compute_service_safe_reward(
        metrics, demanded_quantity=float(sum(skeleton.order_quantities))
    )
    sim, panel = run_program_o_full_des_episode(
        seed=int(seed),
        calendar=calendar,
        scheduler=sched,
        regime_persistence=0.75,
        dominant_share=0.90,
        downstream_freight_physics_mode="fixed_clock_physical_v1",
        risks_enabled=False,
    )
    direct = panel["metrics"]
    resources = panel["resources"]
    vector = direct_full_des_vector(sim, panel)
    return {
        "ret_service_full_clipped": float(metrics["ret_service_full_clipped"]),
        "ret_excel_clipped_0_1": float(direct["ret_excel_visible_clipped_0_1"]),
        "ret_excel_full_ledger": float(direct["ret_excel_full_ledger"]),
        "ret_thesis": float(direct["ret_thesis"]),
        "flow_fill_rate": float(direct["flow_fill_rate"]),
        "delivered_rations": float(direct["delivered_rations"]),
        "lost_orders": float(direct["lost_orders"]),
        "unresolved_orders": float(direct["n_orders"] - direct["n_served"]),
        "terminal_stock": float(vector["ending_inventory_total"]),
        "worst_product_fill": float(panel["worst_product_fill"]),
        "actual_payload": float(resources["actual_payload"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sched = scheduler()
    gate = json.loads(GATE_RESULT.read_text())
    contract = json.loads(CONTRACT.read_text())
    frozen = tuple(map(int, gate["frozen_incumbent"]["calendar"]))
    in_sample = tuple(map(int, gate["in_sample_static_incumbent"]["calendar"]))

    calendars: dict[str, dict[str, list[int]]] = {
        "frozen_static": {"calendar": list(frozen)},
        "in_sample_static": {"calendar": list(in_sample)},
    }
    configs = list(finite_state_rich_configurations())
    policy_configs: dict[str, StateRichConfiguration] = {
        config.config_id: config for config in configs
    }
    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for policy, calendar_payload in list(calendars.items()):
        calendar = tuple(calendar_payload["calendar"])
        for seed in SEEDS:
            values = evaluate_calendar(seed, calendar, sched)
            values["seed"] = int(seed)
            values["calendar"] = list(calendar)
            rows[policy].append(values)
    for policy, config in policy_configs.items():
        calendars[policy] = {}
        for seed in SEEDS:
            skeleton, _ = extract_full_des_skeleton(
                seed=int(seed),
                scheduler=sched,
                regime_persistence=0.75,
                dominant_share=0.90,
                downstream_freight_physics_mode="fixed_clock_physical_v1",
            )
            calendar, decisions = state_rich_calendar(
                skeleton=skeleton.as_dict(),
                scheduler=sched,
                config=config,
                regime_persistence=0.75,
                dominant_share=0.90,
            )
            calendars[policy][str(seed)] = {
                "calendar": list(calendar),
                "decision_count": len(decisions),
            }
            values = evaluate_calendar(seed, tuple(calendar), sched)
            values["seed"] = int(seed)
            values["calendar"] = list(calendar)
            rows[policy].append(values)

    summary: dict[str, dict[str, Any]] = {}
    metrics = (
        "ret_excel_clipped_0_1", "ret_excel_full_ledger", "ret_thesis",
        "flow_fill_rate", "delivered_rations", "lost_orders", "unresolved_orders",
        "terminal_stock", "worst_product_fill", "actual_payload",
    )
    for policy, policy_rows in rows.items():
        summary[policy] = {
            "primary": bootstrap(
                [float(row["ret_service_full_clipped"]) for row in policy_rows],
                seed=830000 + len(policy),
            ),
            "secondary_means": {
                metric: float(np.mean([float(row[metric]) for row in policy_rows]))
                for metric in metrics
            },
            "n_unique_calendars": len({tuple(row["calendar"]) for row in policy_rows}),
        }
    for policy in rows:
        if policy != "frozen_static":
            summary[policy]["guardrails_vs_frozen_static"] = guardrail_report(
                rows[policy], rows["frozen_static"]
            )

    contrasts: dict[str, Any] = {}
    baseline_rows = rows["frozen_static"]
    for policy, policy_rows in rows.items():
        if policy == "frozen_static":
            continue
        diff = [
            float(candidate["ret_service_full_clipped"])
            - float(base["ret_service_full_clipped"])
            for candidate, base in zip(policy_rows, baseline_rows)
        ]
        contrasts[policy + "_minus_frozen_static"] = {
            "primary": bootstrap(diff, seed=831000 + len(contrasts)),
            "secondary_mean_deltas": {
                metric: float(np.mean([
                    float(candidate[metric]) - float(base[metric])
                    for candidate, base in zip(policy_rows, baseline_rows)
                ]))
                for metric in metrics
            },
        }

    result = {
        "schema_version": "program_b_classical_baselines_exploratory_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "contract_sha256": digest(CONTRACT),
        "validation_tapes": [SEEDS[0], SEEDS[-1]],
        "primary": "ret_service_full_clipped_v1",
        "secondary": list(metrics),
        "policies": ["frozen_static", "in_sample_static", *sorted(policy_configs)],
        "summary": summary,
        "contrasts": contrasts,
        "calendars": calendars,
        "claim_boundary": "Exploratory same-tape comparator screen; no fresh DES seeds; no promotion or neural superiority claim.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "policies": result["policies"], "validation_tapes": result["validation_tapes"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
