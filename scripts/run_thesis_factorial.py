#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.config import (  # noqa: E402
    R14_DEFECT_MODE_OPTIONS,
    THESIS_DOWNSTREAM_Q_RANGES,
    THESIS_FAITHFUL_PROTOCOL,
)
from supply_chain.supply_chain import MFSCSimulation  # noqa: E402
from supply_chain.thesis_design import (  # noqa: E402
    ThesisDesignSpec,
    design_matrix,
    design_spec_for_cfi,
    parse_cf_range,
)

DEFAULT_OUTPUT_ROOT = Path("outputs/thesis_faithful/factorial")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def spec_row(spec: ThesisDesignSpec) -> dict[str, Any]:
    return {
        "Cfi": spec.cfi,
        "family": spec.family,
        "source_cfi": spec.source_cfi,
        "enabled_risks": ",".join(spec.enabled_risks),
        "risk_overrides": json.dumps(spec.risk_overrides, sort_keys=True),
        "shifts": spec.shifts,
        "inventory_replenishment_period": spec.inventory_replenishment_period,
        "initial_buffers": json.dumps(spec.initial_buffers or {}, sort_keys=True),
        "horizon_hours": spec.horizon_hours,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run Garrido-Rios thesis factorial configurations Cf1..Cf90. "
            "This is a reproduction harness, not an RL trainer."
        )
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--cf-range",
        default="1-90",
        help="Cfi selection, e.g. '1-30', '31-60', or '1-30,47,85'.",
    )
    selection.add_argument("--cfi", type=int, nargs="+", help="Explicit Cfi values.")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42],
        help=(
            "Seeds to run. If a single integer N is supplied with --seed-count-mode, "
            "it is interpreted as N generated seeds."
        ),
    )
    parser.add_argument(
        "--seed-count-mode",
        action="store_true",
        help="Interpret --seeds N as seed count and generate deterministic seeds 1..N.",
    )
    parser.add_argument("--label", default=None, help="Auditable run folder label.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--downstream-q-source",
        choices=tuple(sorted(THESIS_DOWNSTREAM_Q_RANGES)),
        default=str(THESIS_FAITHFUL_PROTOCOL["downstream_q_source"]),
    )
    parser.add_argument(
        "--r14-defect-mode",
        choices=R14_DEFECT_MODE_OPTIONS,
        default=str(THESIS_FAITHFUL_PROTOCOL["r14_defect_mode"]),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write only the design matrix and manifest; do not run SimPy.",
    )
    return parser


def selected_specs(args: argparse.Namespace) -> list[ThesisDesignSpec]:
    values = args.cfi if args.cfi is not None else parse_cf_range(str(args.cf_range))
    return design_matrix(values)


def selected_seeds(args: argparse.Namespace) -> list[int]:
    if args.seed_count_mode:
        if len(args.seeds) != 1:
            raise ValueError("--seed-count-mode expects exactly one --seeds value.")
        return list(range(1, int(args.seeds[0]) + 1))
    return [int(seed) for seed in args.seeds]


def command_line() -> str:
    return " ".join(
        [Path(sys.executable).name, "scripts/run_thesis_factorial.py", *sys.argv[1:]]
    )


def run_one(
    *,
    spec: ThesisDesignSpec,
    seed: int,
    downstream_q_source: str,
    r14_defect_mode: str,
) -> MFSCSimulation:
    sim = MFSCSimulation(
        shifts=spec.shifts,
        initial_buffers=spec.initial_buffers,
        seed=seed,
        horizon=spec.horizon_hours,
        risks_enabled=True,
        risk_level="current",
        year_basis="thesis",
        deterministic_baseline=False,
        warmup_trigger="op9_arrival",
        downstream_q_source=downstream_q_source,
        r14_defect_mode=r14_defect_mode,
        enabled_risks=set(spec.enabled_risks),
        risk_overrides=spec.risk_overrides,
        inventory_replenishment_period=spec.inventory_replenishment_period,
    )
    return sim.run()


def summarize_run(
    sim: MFSCSimulation, spec: ThesisDesignSpec, seed: int
) -> dict[str, Any]:
    throughput = sim.get_annual_throughput(start_time=sim.warmup_time)
    ret = sim.compute_order_level_ret()
    risk_counts: dict[str, int] = {}
    for event in sim.risk_events:
        risk_counts[event.risk_id] = risk_counts.get(event.risk_id, 0) + 1
    return {
        **spec_row(spec),
        "seed": seed,
        "warmup_time": float(sim.warmup_time),
        "total_produced": float(sim.total_produced),
        "total_delivered": float(sim.total_delivered),
        "total_demanded": float(sim.total_demanded),
        "orders": len(sim.orders),
        "total_backorders": int(sim.total_backorders),
        "unattended_orders": int(sim.total_unattended_orders),
        "risk_events": len(sim.risk_events),
        "risk_event_counts": json.dumps(risk_counts, sort_keys=True),
        "avg_annual_delivery_post_warmup": float(throughput["avg_annual_delivery"]),
        "avg_annual_production_post_warmup": float(throughput["avg_annual_production"]),
        "mean_ret": float(ret["mean_ret"]),
        "fill_rate_order_level": float(ret["fill_rate_order_level"]),
        "ret_case_counts": json.dumps(ret["case_counts"], sort_keys=True),
    }


def risk_event_rows(
    sim: MFSCSimulation, spec: ThesisDesignSpec, seed: int
) -> list[dict[str, Any]]:
    return [
        {
            "Cfi": spec.cfi,
            "seed": seed,
            "risk_id": event.risk_id,
            "start_time": event.start_time,
            "end_time": event.end_time,
            "duration": event.duration,
            "affected_ops": ",".join(str(op) for op in event.affected_ops),
            "description": event.description,
        }
        for event in sim.risk_events
    ]


def main() -> int:
    args = build_parser().parse_args()
    specs = selected_specs(args)
    seeds = selected_seeds(args)
    label = (
        args.label or f"factorial_{utc_now_iso().replace(':', '').replace('+', 'Z')}"
    )
    run_dir = args.output_root / label
    run_dir.mkdir(parents=True, exist_ok=False)

    manifest = {
        "created_at": utc_now_iso(),
        "protocol": THESIS_FAITHFUL_PROTOCOL,
        "command": command_line(),
        "pid": os.getpid(),
        "dry_run": bool(args.dry_run),
        "seeds": seeds,
        "downstream_q_source": args.downstream_q_source,
        "r14_defect_mode": args.r14_defect_mode,
    }
    write_json(run_dir / "manifest.json", manifest)
    (run_dir / "command.txt").write_text(command_line() + "\n", encoding="utf-8")
    (run_dir / "pid.txt").write_text(f"{os.getpid()}\n", encoding="utf-8")
    write_csv(run_dir / "design_matrix.csv", [spec_row(spec) for spec in specs])

    if args.dry_run:
        write_json(
            run_dir / "status.json",
            {"status": "SUCCEEDED", "updated_at": utc_now_iso()},
        )
        print(
            json.dumps({"run_dir": str(run_dir), "specs": len(specs), "dry_run": True})
        )
        return 0

    write_json(
        run_dir / "status.json", {"status": "RUNNING", "updated_at": utc_now_iso()}
    )
    summary_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    try:
        for spec in specs:
            design_spec_for_cfi(spec.cfi)
            for seed in seeds:
                sim = run_one(
                    spec=spec,
                    seed=seed,
                    downstream_q_source=args.downstream_q_source,
                    r14_defect_mode=args.r14_defect_mode,
                )
                summary_rows.append(summarize_run(sim, spec, seed))
                event_rows.extend(risk_event_rows(sim, spec, seed))
        write_csv(run_dir / "summary.csv", summary_rows)
        write_csv(run_dir / "risk_events.csv", event_rows)
        write_json(
            run_dir / "summary.json",
            {
                "created_at": utc_now_iso(),
                "run_count": len(summary_rows),
                "cfi_count": len(specs),
                "seed_count": len(seeds),
                "runs": summary_rows,
            },
        )
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

    print(json.dumps({"run_dir": str(run_dir), "runs": len(summary_rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
