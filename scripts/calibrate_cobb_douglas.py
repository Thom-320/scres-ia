#!/usr/bin/env python3
"""Calibrate and freeze the Cobb-Douglas exponents, then score a comparison set.

Two phases, deliberately separate commands so the exponents are frozen to a file
before any policy is scored:

    --phase calibrate   sweep development tapes, record x_max, derive exponents
                        with Garrido's rule (0.20 / ln x_max), write the contract
    --phase score       load the frozen contract, score a declared comparison set

Re-deriving exponents after seeing results would convert a scale normaliser into a
tuned preference weight, which is the Program G error. The contract file carries the
sweep's hash so a scoring run can prove which calibration it used.

The comparison set matters as much as the exponents: kappa_dot is normalised by the
whole set's cost (Eq. 5), so every policy's R depends on which others are present.
The set is declared on the command line and echoed into the result.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.cobb_douglas_resilience import (  # noqa: E402
    FLOORS,
    WELL_CONDITIONED_LOG_MAX,
    assert_terms_bounded,
    conditioning,
    UNIT_COSTS,
    VARIABLES,
    CobbDouglasRecorder,
    derive_exponents,
    kappa_dot,
    score_comparison_set,
)
from supply_chain.config import HOURS_PER_WEEK  # noqa: E402
from supply_chain.config import THESIS_FAITHFUL_PROTOCOL as P  # noqa: E402
from supply_chain.episode_metrics import compute_episode_metrics  # noqa: E402
from supply_chain.expanded_contract_controllers import LADDER_HOURS, level_targets  # noqa: E402
from supply_chain.supply_chain import MFSCSimulation  # noqa: E402

FAMILIES = {"R1r": ("R11", "R12", "R13", "R14"), "R2r": ("R21", "R22", "R23", "R24")}
# Development tapes for calibration. Disjoint from every sealed evaluation block.
CALIBRATION_SEEDS = (1_610_001, 1_610_002, 1_610_003)


def run_episode(*, seed: int, horizon: float, family: str, buffer_hours: int,
                shifts: int, period_hours: float, replenishment: float) -> dict:
    """One episode, sampling the five C-D output variables every `period_hours`."""
    sim = MFSCSimulation(
        shifts=shifts,
        initial_buffers=level_targets(buffer_hours),
        inventory_replenishment_period=replenishment if buffer_hours else 0.0,
        seed=seed,
        horizon=horizon,
        risks_enabled=True,
        risk_level="current",
        enabled_risks=set(FAMILIES[family]),
        risk_overrides={r: "increased" for r in FAMILIES[family]},
        strict_exogenous_crn=True,
        year_basis=P["year_basis"],
        warmup_trigger=P["warmup_trigger"],
        r14_defect_mode=P["r14_defect_mode"],
    )
    rec = CobbDouglasRecorder(period_hours=period_hours)
    elapsed = 0.0
    while elapsed < horizon:
        step = min(period_hours, horizon - elapsed)
        sim.step(action=None, step_hours=step)
        elapsed += step
        rec.sample(sim)
    agg = rec.aggregate()
    m = compute_episode_metrics(sim)
    agg.update({
        "ret_excel": float(m["ret_excel"]),
        "ret_excel_full_ledger": float(m["ret_excel_full_ledger"]),
        "ret_thesis": float(m["ret_thesis"]),
        "flow_fill_rate": float(m["flow_fill_rate"]),
        "delivered_rations": float(m["delivered_rations"]),
        "lost_orders": float(m["lost_orders"]),
    })
    return agg


def phase_calibrate(args) -> dict:
    """Sweep the development block wide enough to bracket every variable's range."""
    horizon = args.horizon_weeks * HOURS_PER_WEEK
    rows: list[dict] = []
    started = time.perf_counter()
    for family in args.families:
        for buffer_hours in LADDER_HOURS:
            for shifts in (1, 2, 3):
                for seed in CALIBRATION_SEEDS:
                    agg = run_episode(
                        seed=seed, horizon=horizon, family=family,
                        buffer_hours=buffer_hours, shifts=shifts,
                        period_hours=args.period_hours,
                        replenishment=args.replenishment_hours)
                    rows.append({"family": family, "buffer_hours": buffer_hours,
                                 "shifts": shifts, "seed": seed, **agg})
        print(f"  {family} done ({time.perf_counter() - started:.0f}s, "
              f"{len(rows)} episodes)", flush=True)

    # kappa_dot needs a set. During calibration the "set" is the whole sweep, which
    # is the widest bracket available and is what Garrido's own 10,000-run
    # calibration amounts to.
    kd = kappa_dot({str(i): r["kappa"] for i, r in enumerate(rows)})
    maxima = {
        "zeta": max(r["zeta"] for r in rows),
        "epsilon": max(r["epsilon"] for r in rows),
        "phi": max(r["phi"] for r in rows),
        "tau": max(r["tau"] for r in rows),
        "kappa_dot": max(kd.values()),
    }

    # A variable whose observed maximum is <= 1 cannot be normalised by
    # 0.20/ln(x_max) and carries no information: floored identically for every
    # policy, its term is a constant that shifts R's level without changing any
    # ranking. Record it as degenerate with exponent 0 rather than crash or
    # silently clamp -- and say so in the artifact.
    degenerate = sorted(k for k in VARIABLES if maxima[k] <= 1.0)
    live = {k: v for k, v in maxima.items() if k not in degenerate}
    exponents = {k: 0.0 for k in degenerate}
    if live:
        exponents.update(derive_exponents({**{k: 10.0 for k in VARIABLES}, **live}))
        for k in degenerate:
            exponents[k] = 0.0

    cond = conditioning({**maxima, **{k: math.e for k in degenerate}})
    # The whole point of the rule: no term may exceed 1/5 anywhere in range.
    assert_terms_bounded(exponents, {**maxima, **{k: 1.0 for k in degenerate}})

    payload = {
        "schema_version": "cobb_douglas_calibration_v1",
        "claim_status": "CALIBRATION_ONLY_NO_POLICY_ADJUDICATION",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "provenance": (
            "Garrido, Ponguta & Garcia-Reyes (2024) IJPR, DOI "
            "10.1080/00207543.2024.2425771, Eq. (3)-(6); exponents re-derived with "
            "his own rule 0.20/ln(x_max) from OUR maxima, never copied from his fit"
        ),
        "exponent_rule": "0.20 / ln(x_max)",
        "maxima": maxima,
        "exponents": exponents,
        "degenerate_variables": degenerate,
        "conditioning": cond,
        "ill_conditioned_variables": sorted(
            k for k, v in cond.items() if not v["well_conditioned"]),
        "well_conditioned_log_max_threshold": WELL_CONDITIONED_LOG_MAX,
        "floors": dict(FLOORS),
        "unit_costs": dict(UNIT_COSTS),
        "unit_costs_provenance": (
            "Garrido 2024 §3.1 assumption (6): all seven cost parameters share c = 1. "
            "His §5 varies the sensitive ones over [1,2] and finds the ranking "
            "unchanged, so this is his published baseline, not a placeholder."
        ),
        "calibration_seeds": list(CALIBRATION_SEEDS),
        "families": list(args.families),
        "period_hours": args.period_hours,
        "horizon_weeks": args.horizon_weeks,
        "n_episodes": len(rows),
        "episodes": rows,
        "elapsed_seconds": time.perf_counter() - started,
    }
    return payload


def phase_score(args) -> dict:
    """Score a frozen comparison set of (buffer, shift) postures under one contract."""
    contract = json.loads(Path(args.contract).read_text())
    exponents = contract["exponents"]
    horizon = args.horizon_weeks * HOURS_PER_WEEK
    started = time.perf_counter()

    # The comparison set, declared before evaluation. kappa_dot depends on all of it.
    postures = [(b, s) for b in args.buffer_hours for s in args.shifts]
    out: dict[str, dict] = {}
    for family in args.families:
        per_policy: dict[str, dict] = {}
        rows: dict[str, list[dict]] = {}
        for buffer_hours, shifts in postures:
            name = f"I{buffer_hours}_S{shifts}"
            eps = [run_episode(seed=t, horizon=horizon, family=family,
                               buffer_hours=buffer_hours, shifts=shifts,
                               period_hours=args.period_hours,
                               replenishment=args.replenishment_hours)
                   for t in args.tapes]
            rows[name] = eps
            per_policy[name] = {
                k: sum(e[k] for e in eps) / len(eps)
                for k in ("zeta", "epsilon", "phi", "tau", "kappa", "ret_excel",
                          "ret_excel_full_ledger", "ret_thesis", "flow_fill_rate",
                          "delivered_rations", "lost_orders")
            }
        scored = score_comparison_set(per_policy, exponents)
        out[family] = {
            "comparison_set": sorted(per_policy),
            "per_policy": {
                name: {**per_policy[name], **scored[name],
                       "per_tape_R_inputs": rows[name]}
                for name in per_policy
            },
        }
        print(f"  {family} scored ({time.perf_counter() - started:.0f}s)", flush=True)

    return {
        "schema_version": "cobb_douglas_score_v1",
        "claim_status": "DEVELOPMENT_SCREEN_NO_CLAIM",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "contract_path": str(args.contract),
        "contract_self_sha256": contract.get("self_sha256"),
        "unit_costs": dict(UNIT_COSTS),
        "exponents": exponents,
        "degenerate_variables": contract.get("degenerate_variables", []),
        "comparison_set_declared": [f"I{b}_S{s}" for b in args.buffer_hours
                                    for s in args.shifts],
        "tapes": list(args.tapes),
        "metric_panel": ["ret_excel", "ret_excel_full_ledger", "ret_thesis",
                         "R_cobb_douglas"],
        "results": out,
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=("calibrate", "score"), required=True)
    ap.add_argument("--families", nargs="+", default=["R1r", "R2r"])
    ap.add_argument("--horizon-weeks", type=int, default=52)
    ap.add_argument("--period-hours", type=float, default=24.0,
                    help="C-D sampling period; Garrido's t is one planning period")
    ap.add_argument("--replenishment-hours", type=float, default=168.0)
    ap.add_argument("--contract", type=Path,
                    default=Path("contracts/cobb_douglas_calibration_v1.json"))
    ap.add_argument("--buffer-hours", nargs="+", type=int, default=[0, 168, 1344])
    ap.add_argument("--shifts", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--tapes", nargs="+", type=int,
                    default=[1_620_001, 1_620_002, 1_620_003, 1_620_004])
    ap.add_argument("--cost", nargs="+", default=[], metavar="c_x=VALUE",
                    help="override kappa cost coefficients, e.g. c_b=2 c_i=2; "
                         "Garrido's own §5 sensitivity varies the sensitive ones "
                         "over [1,2] and checks the ranking is unchanged")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    if args.cost:
        for spec in args.cost:
            k, _, v = spec.partition("=")
            if k not in UNIT_COSTS:
                raise SystemExit(f"unknown cost coefficient {k!r}; "
                                 f"expected one of {sorted(UNIT_COSTS)}")
            UNIT_COSTS[k] = float(v)
        print(f"cost overrides in effect: {dict(UNIT_COSTS)}")

    payload = phase_calibrate(args) if args.phase == "calibrate" else phase_score(args)
    body = json.dumps(payload, indent=1, sort_keys=True)
    payload["self_sha256"] = sha256(body.encode()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(f"\n-> {args.output}")

    if args.phase == "calibrate":
        print(f"maxima:    {payload['maxima']}")
        print(f"exponents: {payload['exponents']}")
        if payload["degenerate_variables"]:
            print(f"DEGENERATE (exponent forced to 0, carries no information): "
                  f"{payload['degenerate_variables']}")
        if payload["ill_conditioned_variables"]:
            print("ILL-CONDITIONED (exponent highly sensitive to one episode):")
            for k in payload["ill_conditioned_variables"]:
                c = payload["conditioning"][k]
                print(f"  {k}: ln(x_max)={c['log_max']:.3f} "
                      f"relative_sensitivity={c['relative_sensitivity']:.1f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
