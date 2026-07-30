#!/usr/bin/env python3
"""The five-column metric panel: no single number decides, service is a constraint.

Three independent screens converged on one conclusion, by three different
mechanisms, and it is the reason this panel exists rather than a fourth metric:

  ret_excel              loses ordinal discrimination under risk, because the
                         omitted-order fraction is policy-dependent (3.9%-18.6%)
  ret_thesis             collapses to a single case bucket under risk
  R_cobb_douglas         prices no lost order at all -- an order that is never
                         served leaves the backorder queue and stops costing,
                         which is why it ranks a 76%-fill/16-lost posture second

**A resilience index is not a service guarantee.** So service is carried here as
separate, declared constraints that a policy passes or fails, never as a term
inside an objective that a policy can trade away. Resource use is reported on a
Pareto front rather than scalarised, for the same reason.

This panel also spans what no existing screen spans alone. The Cobb-Douglas static
screen varied shifts but not heterogeneous buffer postures; the v2 comparator
instrument varies the full 216 postures but pins `shifts=1`. Neither crosses the
two. The expanded contract Garrido asked about is buffers AND shifts, so the panel
crosses them.

Development screen. Four tapes per cell decides nothing; this reports a panel, not
an adjudication.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.cobb_douglas_resilience import (  # noqa: E402
    CobbDouglasRecorder,
    score_comparison_set,
)
from supply_chain.provenance import calibration_stamp  # noqa: E402
from supply_chain.config import HOURS_PER_WEEK  # noqa: E402
from supply_chain.config import THESIS_FAITHFUL_PROTOCOL as P  # noqa: E402
from supply_chain.episode_metrics import compute_episode_metrics  # noqa: E402
from supply_chain.expanded_contract_controllers_v2 import (  # noqa: E402
    ProjectedDDMRPController,
    VectorStaticPosture,
)
from supply_chain.supply_chain import MFSCSimulation  # noqa: E402

FAMILIES = {"R1r": ("R11", "R12", "R13", "R14"), "R2r": ("R21", "R22", "R23", "R24")}

# Buffer postures, declared before evaluation. (672, 0, 1344) is the 216-posture
# buffer-gate incumbent under all-risks-current; (168, 0, 168) won the
# fully-increased single-family regime. Both are carried because the two regimes
# disagree, which is itself a finding.
POSTURES: tuple[tuple[int, int, int], ...] = (
    (672, 0, 1344),
    (168, 0, 168),
    (168, 168, 168),
    (0, 0, 0),
    (1344, 1344, 1344),
)
SHIFTS: tuple[int, ...] = (1, 2, 3)

# Service floors, declared BEFORE evaluation. These are constraints, not terms:
# a policy that fails one is out regardless of how well it scores on any index.
SERVICE_FLOORS = {
    "flow_fill_rate_min": 0.90,
    "lost_orders_max": 0.0,
    "backorder_qty_final_max": 50_000.0,
}


def run_cell(controller, *, seed, horizon, family, shifts, epoch_hours,
             period_hours, replenishment) -> dict:
    sim = MFSCSimulation(
        shifts=shifts,
        initial_buffers={n: 0.0 for n in ("op3_rm", "op5_rm", "op9_rations")},
        inventory_replenishment_period=replenishment,
        seed=seed, horizon=horizon, risks_enabled=True, risk_level="current",
        enabled_risks=set(FAMILIES[family]),
        risk_overrides={r: "increased" for r in FAMILIES[family]},
        strict_exogenous_crn=True, year_basis=P["year_basis"],
        warmup_trigger=P["warmup_trigger"], r14_defect_mode=P["r14_defect_mode"],
    )
    controller.reset()
    rec = CobbDouglasRecorder(period_hours=period_hours)
    elapsed, epoch, since = 0.0, 0, float("inf")
    postures: list[list[int]] = []
    while elapsed < horizon:
        if since >= epoch_hours:
            targets = controller.act(sim, epoch)
            sim.inventory_buffer_targets.update(
                {k: float(v) for k, v in targets.items()})
            diag = getattr(controller, "last_diagnostic", {})
            postures.append(list(diag.get("posture",
                                          getattr(controller, "posture", ()))))
            since, epoch = 0.0, epoch + 1
        step = min(period_hours, horizon - elapsed)
        sim.step(action=None, step_hours=step)
        elapsed += step
        since += step
        rec.sample(sim)

    agg = rec.aggregate()
    m = compute_episode_metrics(sim)
    agg.update({
        k: float(m[k]) for k in (
            "ret_excel", "ret_excel_full_ledger", "ret_thesis", "ret_excel_cvar10",
            "ret_excel_cvar05", "flow_fill_rate", "fill_rate_on_time", "lost_orders",
            "backorder_qty_final", "delivered_rations")
    })
    agg["strategic_injected"] = float(sim.total_strategic_raw_injected
                                      + sim.total_strategic_rations_injected)
    agg["distinct_postures"] = len({tuple(p) for p in postures if p})
    agg["posture_changes"] = sum(
        1 for a, b in zip(postures, postures[1:]) if a != b)
    return agg


def pareto_front(rows: dict[str, dict]) -> list[str]:
    """Non-dominated on (resource down, fill up, lost down).

    Reported instead of a scalarised resource penalty: weighting service against
    material is a decision for the reader, not something a metric should hide.
    """
    def dominates(a: dict, b: dict) -> bool:
        better_eq = (a["kappa"] <= b["kappa"]
                     and a["flow_fill_rate"] >= b["flow_fill_rate"]
                     and a["lost_orders"] <= b["lost_orders"])
        strictly = (a["kappa"] < b["kappa"]
                    or a["flow_fill_rate"] > b["flow_fill_rate"]
                    or a["lost_orders"] < b["lost_orders"])
        return better_eq and strictly

    return sorted(n for n, v in rows.items()
                  if not any(dominates(o, v) for k, o in rows.items() if k != n))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--families", nargs="+", default=["R1r", "R2r"])
    ap.add_argument("--tapes", nargs="+", type=int,
                    default=[1_620_001, 1_620_002, 1_620_003, 1_620_004])
    ap.add_argument("--horizon-weeks", type=int, default=52)
    ap.add_argument("--epoch-weeks", type=int, default=4)
    ap.add_argument("--period-hours", type=float, default=24.0)
    ap.add_argument("--replenishment-hours", type=float, default=168.0)
    ap.add_argument("--contract", type=Path,
                    default=Path("contracts/cobb_douglas_calibration_v1.json"))
    ap.add_argument("--output", type=Path,
                    default=Path("results/metric_panel/panel_v1.json"))
    args = ap.parse_args()

    contract = json.loads(args.contract.read_text())
    exponents = contract["exponents"]
    horizon = args.horizon_weeks * HOURS_PER_WEEK
    epoch_hours = args.epoch_weeks * HOURS_PER_WEEK
    started = time.perf_counter()

    out: dict[str, dict] = {}
    for family in args.families:
        cells: dict[str, list[dict]] = {}
        for shifts in SHIFTS:
            for posture in POSTURES:
                c = VectorStaticPosture(posture)
                name = f"{'/'.join(str(h) for h in posture)}|S{shifts}"
                cells[name] = [
                    run_cell(c, seed=t, horizon=horizon, family=family,
                             shifts=shifts, epoch_hours=epoch_hours,
                             period_hours=args.period_hours,
                             replenishment=args.replenishment_hours)
                    for t in args.tapes]
            d = ProjectedDDMRPController()
            cells[f"ddmrp|S{shifts}"] = [
                run_cell(d, seed=t, horizon=horizon, family=family, shifts=shifts,
                         epoch_hours=epoch_hours, period_hours=args.period_hours,
                         replenishment=args.replenishment_hours)
                for t in args.tapes]

        keys = ("zeta", "epsilon", "phi", "tau", "kappa", "ret_excel",
                "ret_excel_full_ledger", "ret_thesis", "ret_excel_cvar10",
                "ret_excel_cvar05", "flow_fill_rate", "fill_rate_on_time",
                "lost_orders", "backorder_qty_final", "delivered_rations",
                "strategic_injected", "distinct_postures", "posture_changes")
        per_cell = {n: {k: sum(e[k] for e in eps) / len(eps) for k in keys}
                    for n, eps in cells.items()}
        # Means alone cannot support paired inference. The 4 x 18 rows are kept.
        per_tape_rows = {n: [{k: e[k] for k in keys} | {"tape": t}
                             for e, t in zip(eps, args.tapes)]
                         for n, eps in cells.items()}
        scored = score_comparison_set(per_cell, exponents)

        for n, v in per_cell.items():
            v.update(scored[n])
            v["service_pass"] = bool(
                v["flow_fill_rate"] >= SERVICE_FLOORS["flow_fill_rate_min"]
                and v["lost_orders"] <= SERVICE_FLOORS["lost_orders_max"]
                and v["backorder_qty_final"]
                <= SERVICE_FLOORS["backorder_qty_final_max"])

        eligible = {n: v for n, v in per_cell.items() if v["service_pass"]}

        # A declared floor is a decision, and a panel that hides its sensitivity to
        # that decision is worse than no panel. On first run the four metrics all
        # agreed on one winner in R2r -- which looked like the headline until this
        # sweep showed the agreement existed only because the 50,000 floor happened
        # to land in a 5,000-wide window admitting exactly two cells. At 49,000 none
        # pass; at 55,000 fifteen pass and three different winners reappear.
        floor_sensitivity = []
        for floor in (20_000, 40_000, 49_000, 50_000, 55_000, 60_000,
                      100_000, 200_000):
            ok = {n: v for n, v in per_cell.items()
                  if v["flow_fill_rate"] >= SERVICE_FLOORS["flow_fill_rate_min"]
                  and v["lost_orders"] <= SERVICE_FLOORS["lost_orders_max"]
                  and v["backorder_qty_final"] <= floor}
            winners = {m: (max(ok, key=lambda n: ok[n][m]) if ok else None)
                       for m in ("ret_excel", "ret_excel_full_ledger",
                                 "R_cobb_douglas", "ret_excel_cvar10")}
            floor_sensitivity.append({
                "backorder_qty_final_max": floor,
                "n_pass": len(ok),
                "winner_by_metric": winners,
                "n_distinct_winners": len(set(winners.values())),
                "all_metrics_agree": len(set(winners.values())) == 1 and bool(ok),
            })
        rank_by = {
            m: sorted(per_cell, key=lambda n: -per_cell[n][m])
            for m in ("ret_excel", "ret_excel_full_ledger", "R_cobb_douglas",
                      "ret_excel_cvar10")
        }
        out[family] = {
            "per_cell": per_cell,
            "service_floors": SERVICE_FLOORS,
            "n_cells": len(per_cell),
            "n_service_pass": len(eligible),
            "service_failures": sorted(set(per_cell) - set(eligible)),
            "rank_by_metric": rank_by,
            "winner_by_metric": {m: r[0] for m, r in rank_by.items()},
            "winner_by_metric_among_service_pass": {
                m: next((n for n in r if n in eligible), None)
                for m, r in rank_by.items()},
            "pareto_front_kappa_fill_lost_only": pareto_front(per_cell),
            "pareto_front_axes": ["kappa (down)", "flow_fill_rate (up)",
                                  "lost_orders (down)"],
            "pareto_front_excluded_axes": [
                "backorder_qty_final", "delivered_rations", "strategic_injected",
                "fill_rate_on_time", "tau"],
            "per_tape_rows": per_tape_rows,
            "service_floor_sensitivity": floor_sensitivity,
            # Compares the exact winner VECTOR across floors. The first version
            # compared `n_distinct_winners`, i.e. how many winners there were --
            # which would call a panel "robust" if the winners changed completely
            # but happened to stay the same in number. Both families gave the
            # right answer by luck; the check did not.
            "agreement_is_floor_robust": len({
                json.dumps(fs["winner_by_metric"], sort_keys=True)
                for fs in floor_sensitivity if fs["n_pass"] > 0}) == 1,
        }
        print(f"  {family}: {len(per_cell)} cells "
              f"({time.perf_counter() - started:.0f}s)", flush=True)

    payload = {
        "schema_version": "metric_panel_v1",
        "calibration_provenance": calibration_stamp(),
        "claim_status": "DEVELOPMENT_SCREEN_NO_CLAIM",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": ("five-column panel: no single metric decides; service is a "
                    "declared constraint; resources on a Pareto front"),
        "metric_panel": ["ret_excel", "ret_excel_full_ledger", "R_cobb_douglas",
                         "ret_excel_cvar10"],
        "service_constraints_are_not_objective_terms": True,
        "spans_buffers_and_shifts": True,
        "step_hours": args.period_hours,
        "ret_excel_is_step_cadence_dependent": True,
        "cadence_warning": (
            "ret_excel depends on step() cadence: identical trajectories (same "
            "fill, same delivered, same risk events) score 0.004369 at one step "
            "and 0.005981 at hourly steps, a 37% spread, because RPj differs in "
            "175 of 311 orders. Numbers here are comparable ONLY to artifacts "
            "produced at the same step_hours. Winners were verified stable across "
            "24h and 672h; full rank order was not (2/18 positions held in R1r)."),
        "contract_path": str(args.contract),
        "contract_self_sha256": contract.get("self_sha256"),
        "exponents": exponents,
        "degenerate_variables": contract.get("degenerate_variables", []),
        "postures_declared": [list(p) for p in POSTURES],
        "shifts_declared": list(SHIFTS),
        "tapes": list(args.tapes),
        "results": out,
        "elapsed_seconds": time.perf_counter() - started,
    }
    body = json.dumps(payload, indent=1, sort_keys=True)
    payload["self_sha256"] = sha256(body.encode()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(f"\n-> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
