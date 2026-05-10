#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.config import (  # noqa: E402
    HOURS_PER_YEAR_THESIS,
    INVENTORY_BUFFERS,
    R14_DEFECT_MODE_OPTIONS,
    SIMULATION_HORIZON,
    THESIS_DOWNSTREAM_Q_RANGES,
    THESIS_FAITHFUL_PROTOCOL,
    VALIDATION_TABLE_6_10,
)
from supply_chain.supply_chain import MFSCSimulation  # noqa: E402

DEFAULT_OUTPUT_ROOT = Path("outputs/thesis_faithful")
SCENARIOS = (
    "cf0",
    "stochastic_current",
    "capacity_s1",
    "capacity_s2",
    "capacity_s3",
    "inventory_168",
    "inventory_336",
    "inventory_504",
    "inventory_672",
    "inventory_1344",
)

THESIS_BACKBONE = {
    **THESIS_FAITHFUL_PROTOCOL,
    "benchmark_protocol": "thesis_faithful",
    "allowed_scenarios": SCENARIOS,
}


@dataclass(frozen=True)
class ScenarioSpec:
    cfi: str
    shifts: int
    risks_enabled: bool
    deterministic_baseline: bool
    risk_level: str
    initial_buffers: dict[str, float] | None = None
    inventory_replenishment_period: float | None = None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def scenario_spec(name: str) -> ScenarioSpec:
    if name == "cf0":
        return ScenarioSpec("Cf0", 1, False, True, "current")
    if name == "stochastic_current":
        return ScenarioSpec("Cf0_Rcr_current", 1, True, False, "current")
    if name.startswith("capacity_s"):
        shifts = int(name[-1])
        return ScenarioSpec(f"Cf0_S{shifts}", shifts, False, True, "current")
    if name.startswith("inventory_"):
        period = int(name.split("_", maxsplit=1)[1])
        return ScenarioSpec(
            cfi=f"Cf0_It_{period}h",
            shifts=1,
            risks_enabled=False,
            deterministic_baseline=True,
            risk_level="current",
            initial_buffers={k: float(v) for k, v in INVENTORY_BUFFERS[period].items()},
            inventory_replenishment_period=float(period),
        )
    raise ValueError(f"Unsupported scenario {name!r}.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a Garrido-Rios thesis-faithful simulation lane. This launcher "
            "does not train RL policies and rejects reward/action-multiplier paths."
        )
    )
    parser.add_argument("--label", required=True, help="Auditable run folder label.")
    parser.add_argument("--scenario", choices=SCENARIOS, default="cf0")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--horizon-hours",
        type=float,
        default=float(SIMULATION_HORIZON),
        help="Simulation horizon. Default is thesis Section 6.8.1: 161,280h.",
    )
    parser.add_argument(
        "--downstream-q-source",
        choices=tuple(sorted(THESIS_DOWNSTREAM_Q_RANGES)),
        default=str(THESIS_FAITHFUL_PROTOCOL["downstream_q_source"]),
        help="Resolve the Figure 6.2 vs Table 6.20 downstream Q discrepancy.",
    )
    parser.add_argument(
        "--r14-defect-mode",
        choices=R14_DEFECT_MODE_OPTIONS,
        default=str(THESIS_FAITHFUL_PROTOCOL["r14_defect_mode"]),
    )
    return parser


def command_line(args: argparse.Namespace) -> str:
    return " ".join(
        [Path(sys.executable).name, "scripts/run_thesis_faithful.py", *sys.argv[1:]]
    )


def throughput_summary(sim: MFSCSimulation) -> dict[str, Any]:
    return sim.get_annual_throughput(start_time=sim.warmup_time)


def order_rows(sim: MFSCSimulation, cfi: str) -> list[dict[str, Any]]:
    rows = []
    for order in sim.orders:
        rows.append(
            {
                "Cfi": cfi,
                "j": order.j,
                "OPTj": order.OPTj,
                "OATj": order.OATj,
                "CTj": order.CTj,
                "LTj": order.LTj,
                "quantity": order.quantity,
                "remaining_qty": order.remaining_qty,
                "backorder": order.backorder,
                "lost": order.lost,
                "contingent": order.contingent,
                "APj": order.APj,
                "RPj": order.RPj,
                "DPj": order.DPj,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_one(
    *,
    spec: ScenarioSpec,
    seed: int,
    horizon_hours: float,
    downstream_q_source: str,
    r14_defect_mode: str,
) -> MFSCSimulation:
    sim = MFSCSimulation(
        shifts=spec.shifts,
        initial_buffers=spec.initial_buffers,
        seed=seed,
        horizon=horizon_hours,
        risks_enabled=spec.risks_enabled,
        risk_level=spec.risk_level,
        year_basis="thesis",
        deterministic_baseline=spec.deterministic_baseline,
        warmup_trigger="op9_arrival",
        downstream_q_source=downstream_q_source,
        r14_defect_mode=r14_defect_mode,
        inventory_replenishment_period=spec.inventory_replenishment_period,
    )
    return sim.run()


def summarize_run(sim: MFSCSimulation, spec: ScenarioSpec, seed: int) -> dict[str, Any]:
    throughput = throughput_summary(sim)
    thesis_ecs_avg = float(np.mean(VALIDATION_TABLE_6_10["ECS_simulated"]))
    avg_delivery = float(throughput["avg_annual_delivery"])
    return {
        "cfi": spec.cfi,
        "seed": seed,
        "shifts": spec.shifts,
        "risks_enabled": spec.risks_enabled,
        "risk_level": spec.risk_level,
        "horizon_hours": float(sim.horizon),
        "year_basis": sim.year_basis,
        "hours_per_year": HOURS_PER_YEAR_THESIS,
        "warmup_trigger": sim.warmup_trigger,
        "warmup_time": float(sim.warmup_time),
        "downstream_q_source": sim.downstream_q_source,
        "r14_defect_mode": sim.r14_defect_mode,
        "total_produced": float(sim.total_produced),
        "total_delivered": float(sim.total_delivered),
        "total_demanded": float(sim.total_demanded),
        "orders": len(sim.orders),
        "total_backorders": int(sim.total_backorders),
        "unattended_orders": int(sim.total_unattended_orders),
        "risk_events": len(sim.risk_events),
        "avg_annual_delivery_post_warmup": avg_delivery,
        "avg_annual_production_post_warmup": float(throughput["avg_annual_production"]),
        "thesis_table_6_10_ecs_avg": thesis_ecs_avg,
        "relative_delta_vs_table_6_10_ecs_avg": (
            (avg_delivery - thesis_ecs_avg) / thesis_ecs_avg
        ),
    }


def aggregate_runs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    delivery = [float(row["avg_annual_delivery_post_warmup"]) for row in rows]
    production = [float(row["avg_annual_production_post_warmup"]) for row in rows]
    return {
        "run_count": len(rows),
        "avg_annual_delivery_post_warmup_mean": float(np.mean(delivery)),
        "avg_annual_delivery_post_warmup_std": float(np.std(delivery, ddof=0)),
        "avg_annual_production_post_warmup_mean": float(np.mean(production)),
        "avg_annual_production_post_warmup_std": float(np.std(production, ddof=0)),
        "thesis_table_6_10_ecs_avg": float(
            np.mean(VALIDATION_TABLE_6_10["ECS_simulated"])
        ),
    }


def main() -> int:
    args = build_parser().parse_args()
    spec = scenario_spec(args.scenario)
    run_dir = args.output_root / args.label
    run_dir.mkdir(parents=True, exist_ok=False)

    manifest = {
        "created_at": utc_now_iso(),
        "protocol": THESIS_BACKBONE,
        "scenario": args.scenario,
        "scenario_spec": spec.__dict__,
        "command": command_line(args),
        "pid": os.getpid(),
    }
    (run_dir / "command.txt").write_text(command_line(args) + "\n", encoding="utf-8")
    (run_dir / "pid.txt").write_text(f"{os.getpid()}\n", encoding="utf-8")
    write_json(run_dir / "manifest.json", manifest)
    write_json(
        run_dir / "status.json", {"status": "RUNNING", "updated_at": utc_now_iso()}
    )

    rows: list[dict[str, Any]] = []
    try:
        for seed in args.seeds:
            sim = run_one(
                spec=spec,
                seed=seed,
                horizon_hours=args.horizon_hours,
                downstream_q_source=args.downstream_q_source,
                r14_defect_mode=args.r14_defect_mode,
            )
            rows.append(summarize_run(sim, spec, seed))
            write_csv(run_dir / f"orders_seed_{seed}.csv", order_rows(sim, spec.cfi))
            write_csv(
                run_dir / f"risk_events_seed_{seed}.csv",
                [
                    {
                        "risk_id": event.risk_id,
                        "start_time": event.start_time,
                        "end_time": event.end_time,
                        "duration": event.duration,
                        "affected_ops": ",".join(str(op) for op in event.affected_ops),
                        "description": event.description,
                    }
                    for event in sim.risk_events
                ],
            )
        summary = {
            "created_at": utc_now_iso(),
            "scenario": args.scenario,
            "runs": rows,
            "aggregate": aggregate_runs(rows),
        }
        write_json(run_dir / "summary.json", summary)
        write_json(
            run_dir / "status.json",
            {"status": "SUCCEEDED", "updated_at": utc_now_iso()},
        )
    except Exception as exc:
        write_json(
            run_dir / "FAILED.json",
            {"status": "FAILED", "updated_at": utc_now_iso(), "error": repr(exc)},
        )
        write_json(
            run_dir / "status.json",
            {"status": "FAILED", "updated_at": utc_now_iso()},
        )
        raise

    print(json.dumps(summary["aggregate"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
