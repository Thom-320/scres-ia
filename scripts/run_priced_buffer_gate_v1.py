#!/usr/bin/env python3
"""Price the buffer, then ask whether the priced decision space is eligible.

Contract: `docs/ENMIENDA_COSTE_BUFFER_DECLARADO_2026-08-08.md`, committed before this file.
Custody: declared replay, no fresh seeds. Uses `supply_chain.falsifiers`, so the checks learned on
2026-08-08 are inherited rather than retyped, and a literal `passed` cannot be constructed.

WHY A HOLDING COST CAN EXIST NOW, and could not before. Until the release path landed, switching
the strategic target off left delivered units in place: K = 4 and K = 26 produced an IDENTICAL
0.302680. Charging for inventory-hours then would have priced something the policy could not
control. With release enabled the same block runs 0.541315 down to 0.376187, so holding duration is
a real decision and pricing it is meaningful.

THE COST IS OUR DECLARED ASSUMPTION, AND IT IS NOT A MONETARY RATE. Garrido-Rios (2017) excludes
cost deliberately (p.147) and lists it as a future extension without values (p.148), so inventing a
currency figure would be fabricating provenance -- the failure mode this project already measured
when it copied Cobb-Douglas exponents across scales.

Instead the cost enters in the endpoint's OWN units, through

    J(lambda) = L* + lambda * (inventory_hours / max_inventory_hours)

`lambda` is an exchange rate between held inventory and unserved-demand exposure. The reference
`lambda = 1` is the equal-maximum convention: holding the maximum possible buffer for the whole
horizon costs exactly as much as total exposure. That is Garrido's own "each argument equated at
its maximum" logic (IJPR 2024 section 3.4) applied where it is defensible -- to two quantities we
measure ourselves -- rather than to five variables on borrowed maxima.

AND THE ANSWER IS REPORTED AS A FUNCTION OF lambda, NOT AT ONE POINT. The break-even lambda -- where
the optimal holding duration changes -- is DERIVED from the data, not assumed. A reader who rejects
our reference can read their own.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.arm_runner import seal_and_write  # noqa: E402
from supply_chain.continuous_its_env import make_continuous_its_track_a_env  # noqa: E402
from supply_chain.falsifiers import (  # noqa: E402
    disclosure, ge, gt, not_applicable, preflight, summarise,
)
from supply_chain.seed_custody import custody_falsifier, module_manifest  # noqa: E402

R1 = ("R11", "R12", "R13", "R14")
R2 = ("R21", "R22", "R23", "R24")
MAX_STEPS, STEP_HOURS = 26, 168.0
LEAD_HOURS = 336.0                 # OUR assumption; the thesis's 48 h is delivery, not rebuild
LAMBDAS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
MIN_FRONT = 3
SEED_BLOCK = tuple(range(8600001, 8600013))
MODULES = ("supply_chain/supply_chain.py", "supply_chain/continuous_its_env.py",
           "supply_chain/falsifiers.py", "supply_chain/arm_runner.py",
           "supply_chain/seed_custody.py")

SCENARIO = {"demand_process": "garrido_seasonal_v1",
            "strategic_buffer_release_mode": "immediate",
            "inventory_replenishment_lead_time": LEAD_HOURS}


def make_env():
    return make_continuous_its_track_a_env(
        init_frac=0.0, reward_mode="ReT_excel_delta", observation_version="v6",
        risk_level="current", enabled_risks=R1 + R2, risk_rng_mode="per_risk",
        stochastic_pt=False, max_steps=MAX_STEPS, step_size_hours=STEP_HOURS,
        risk_obs=True, holding_cost=0.0, shift_cost=0.0,
        demand_process="garrido_seasonal_v1",
        demand_seasonal_contract={"forecast_mode": "garrido_generator"},
        inventory_replenishment_lead_time=LEAD_HOURS,
        strategic_buffer_release_mode="immediate")


def exposure(sim) -> float:
    horizon, start = float(sim.env.now), float(sim.warmup_time)
    num = den = 0.0
    for o in sim.orders:
        if bool(getattr(o, "metrics_excluded", False)):
            continue
        opt = float(getattr(o, "OPTj", 0.0) or 0.0)
        if opt < start:
            continue
        q = float(o.quantity or 0.0)
        due = opt + float(o.LTj or 0.0)
        end = float(o.OATj) if getattr(o, "OATj", None) is not None else horizon
        num += q * max(0.0, end - due)
        den += q * max(0.0, horizon - due)
    return num / den if den > 0 else 0.0


def play(option, seed: int) -> dict:
    start, k = option
    weeks = set(range(start, start + k))
    env = make_env()
    env.reset(seed=int(seed))
    sim = env.unwrapped.sim
    done = truncated = False
    step, inv_hours = 0, 0.0
    try:
        while not (done or truncated):
            on = step in weeks
            _o, _r, done, truncated, _i = env.step(
                np.array([1.0 if on else 0.0, -1.0], dtype=np.float32))
            inv_hours += (1.0 if on else 0.0) * STEP_HOURS
            step += 1
        return {"L": exposure(sim), "inventory_hours": inv_hours,
                "released": float(getattr(sim, "strategic_buffer_released_units", 0.0)),
                "n_steps": step}
    finally:
        env.close()


def options() -> list[tuple[int, int]]:
    return [(s, k) for k in (0, 4, 8, 13, 18, 22) for s in range(0, 27 - k, 4) if s + k <= MAX_STEPS]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--replay-of", required=True)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--output", type=Path,
                    default=Path("results/priced_buffer_gate/result.json"))
    args = ap.parse_args()
    started = time.perf_counter()
    seeds = list(SEED_BLOCK[:args.seeds])
    opts = options()
    print(f"  {len(opts)} calendarios x {len(seeds)} semillas = {len(opts) * len(seeds)} episodios")

    # ---- PRE-FLIGHT, from the shared module, before anything expensive ---------------------
    env = make_env()
    env.reset(seed=seeds[0])
    reset_now = float(env.unwrapped.sim.env.now)
    live_scenario = {"demand_process": getattr(env.unwrapped.sim, "demand_process", None),
                     "strategic_buffer_release_mode":
                         getattr(env.unwrapped.sim, "strategic_buffer_release_mode", None),
                     "inventory_replenishment_lead_time":
                         float(getattr(env.unwrapped.sim,
                                       "inventory_replenishment_lead_time", 0.0))}
    env.close()
    pre = preflight(probe=lambda o: play(o, seeds[0])["L"], options=opts,
                    reset_now=reset_now, horizon=MAX_STEPS * STEP_HOURS,
                    scenario=live_scenario, expected_scenario=SCENARIO)
    pre_summary = summarise(pre)
    print("  pre-vuelo:", {k: v["passed"] for k, v in pre.items()})
    if not pre_summary["all_passed"]:
        print("  PRE-VUELO FALLA — no se corre la campaña")

    rows = {}
    if pre_summary["all_passed"]:
        for o in opts:
            runs = [play(o, s) for s in seeds]
            rows[str(o)] = {
                "option": list(o), "start": o[0], "k_weeks": o[1],
                "L_mean": float(np.mean([r["L"] for r in runs])),
                "L_by_seed": [float(r["L"]) for r in runs],
                "inventory_hours": float(np.mean([r["inventory_hours"] for r in runs])),
                "released_units": float(np.mean([r["released"] for r in runs])),
            }

    max_ih = max((v["inventory_hours"] for v in rows.values()), default=1.0) or 1.0
    priced, best_by_lambda = {}, {}
    for lam in LAMBDAS:
        j = {k: v["L_mean"] + lam * (v["inventory_hours"] / max_ih) for k, v in rows.items()}
        best = min(j, key=j.get) if j else None
        priced[str(lam)] = {"J_by_option": j, "best_option": best,
                            "best_k_weeks": rows[best]["k_weeks"] if best else None,
                            "best_J": j[best] if best else None}
        if best:
            best_by_lambda[lam] = rows[best]["k_weeks"]

    # The break-even lambda is DERIVED: the smallest swept lambda at which the optimal holding
    # duration stops being the longest one. Nothing about it is assumed.
    ks = [best_by_lambda[l] for l in LAMBDAS if l in best_by_lambda]
    switches = [(LAMBDAS[i], ks[i - 1], ks[i]) for i in range(1, len(ks)) if ks[i] != ks[i - 1]]
    n_distinct_optima = len(set(ks))

    pts = [(v["L_mean"], v["inventory_hours"]) for v in rows.values()]
    front = [p for p in pts if not any(q[0] <= p[0] and q[1] <= p[1] and q != p for q in pts)]
    n_front = len({(round(a, 9), round(b, 9)) for a, b in front})

    falsifiers = dict(pre)
    falsifiers["f5_pareto_front_is_wide_enough"] = ge(
        n_front, MIN_FRONT,
        "a front of one or two points is hold-or-do-not, which is the collapsed space the "
        "benchmark measured; a priced decision needs an interior trade-off",
        front=[[float(a), float(b)] for a, b in sorted(front, key=lambda x: x[1])])
    falsifiers["f6_price_actually_moves_the_optimum"] = ge(
        n_distinct_optima, 2,
        "if the optimal holding duration is the same at every lambda, the price is inert and the "
        "cost is decoration rather than a decision variable",
        best_k_by_lambda={str(k): v for k, v in best_by_lambda.items()},
        switches=[[float(a), int(b), int(c)] for a, b, c in switches])
    falsifiers["f7_release_actually_fired"] = gt(
        max((v["released_units"] for v in rows.values()), default=0.0), 0.0,
        "with release enabled some schedule must free stock; zero released would mean the physics "
        "change is inert and the whole pricing rests on nothing",
        released_by_option={k: v["released_units"] for k, v in rows.items()})
    falsifiers["d1_cost_has_no_thesis_provenance"] = disclosure(
        "the holding price is OUR declared assumption, in endpoint units, not a monetary rate",
        thesis_p147="cost deliberately excluded from the wartime military model",
        thesis_p148="cost and SC lead time listed as future extensions, no values given",
        lead_hours_is_also_ours=LEAD_HOURS,
        thesis_48h_is="delivery lead time to the user (p.111), not buffer rebuild")
    falsifiers["d2_reference_lambda_convention"] = disclosure(
        "lambda = 1 means holding the maximum possible buffer for the whole horizon costs exactly "
        "as much as total exposure; the answer is reported across a sweep, and the break-even is "
        "derived from the data rather than assumed",
        lambdas=list(LAMBDAS), max_inventory_hours=float(max_ih))
    falsifiers["d3_no_fresh_seeds"] = not_applicable(
        "declared replay of an already-consumed development block",
        custody=custody_falsifier(seeds, replay_of=args.replay_of, exclude=args.output))

    summary = summarise(falsifiers)
    verdict = ("BLOCKED_INSTRUMENT" if not pre_summary["all_passed"]
               else "PRICED_DECISION_SPACE_ELIGIBLE" if summary["all_passed"]
               else "PRICED_DECISION_SPACE_NOT_ELIGIBLE")

    print(f"\n  frente de Pareto distinto: {n_front} (exige >= {MIN_FRONT})")
    print(f"  K optimo por lambda: {best_by_lambda}")
    print(f"  cambios de optimo: {switches or 'ninguno'}")
    print(f"\n  veredicto: {verdict}")
    print(f"  falsadores: {summary['n_computed']} computados, {summary['n_failed']} fallidos, "
          f"{summary['n_disclosures']} divulgaciones, {summary['n_not_applicable']} no aplicables")
    for k, v in falsifiers.items():
        if v.get("computed"):
            print(f"    {k:52s} {'PASA' if v['passed'] else 'FALLA'}")

    payload = {
        "schema_version": "priced_buffer_gate_v1",
        "claim_status": verdict,
        "scope": "DEVELOPMENT_ON_DECLARED_REPLAY_ELIGIBILITY_GATE_NOT_A_RESULT",
        "run_role": "PRICED_ELIGIBILITY_GATE", "replay_of": args.replay_of,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "module_manifest": module_manifest(MODULES, script=__file__),
        "declared_assumptions": {
            "holding_price": ("J = L* + lambda * inventory_hours / max_inventory_hours; lambda is "
                              "an exchange rate in ENDPOINT units, never a currency figure"),
            "reference_lambda": ("1.0 = holding the maximum buffer for the whole horizon costs as "
                                 "much as total exposure (Garrido IJPR 2024 s3.4 equal-at-maximum "
                                 "logic, applied to two quantities we measure ourselves)"),
            "lead_hours": LEAD_HOURS,
            "why_not_monetary": ("the thesis excludes cost (p.147) and gives no values (p.148); "
                                 "inventing a currency rate would fabricate provenance"),
            "fidelity_price": ("release and lead time are extensions with no source event, so any "
                               "result here is OURS and never presented as reproducing "
                               "Garrido-Rios (2017)")},
        "scenario": SCENARIO, "live_scenario": live_scenario,
        "lambdas": list(LAMBDAS), "max_inventory_hours": float(max_ih),
        "options": [list(o) for o in opts], "seeds": seeds,
        "rows": rows, "priced": priced,
        "best_k_by_lambda": {str(k): v for k, v in best_by_lambda.items()},
        "optimum_switches": [[float(a), int(b), int(c)] for a, b, c in switches],
        "pareto_front_distinct": n_front,
        "falsifiers": falsifiers, "falsifier_summary": summary,
        "preflight_summary": pre_summary,
        "elapsed_seconds": time.perf_counter() - started,
    }
    digest = seal_and_write(payload, args.output, contract=args.contract,
                            reference=Path("results/kan_mlp_r2_benchmark_v2/result.json"))
    print(f"\n  -> {args.output} (sello {digest[:16]}…)")
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
