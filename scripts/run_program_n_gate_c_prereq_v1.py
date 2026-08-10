#!/usr/bin/env python3
"""Gate C prerequisite: does the DES-MPC beat the cheap heuristic, and on what?

Preregistration: `docs/PREREGISTRO_PUERTA_C_AMORTIZACION_2026-08-09.md`, stage C0 said the E*
planner is expensive and its quality was NEVER MEASURED. This measures it.

THE PLANNER IS REAL HERE, unlike K3's. At each weekly epoch it enumerates candidate actions and
evaluates each by a fresh DES rollout, exactly as `DirectDESMPC` specifies. The adapter cannot be
cloned -- simpy generators are unpicklable -- so a candidate is evaluated by REPLAYING the episode
from the burned tape up to the current epoch and stepping the candidate. That is what makes it
expensive and it is the honest implementation, not a shortcut.

FIRST FALSIFIER FIRST. Before comparing anything, `p1` asks whether the objective the planner
maximises responds to the action at all. A planner scoring 192 candidates on a flat function is not
a better decision maker; it is 192 evaluations of a constant. The rest of the run is only
interpretable if p1 passes.

No seed is opened: the 24 tapes are already-consumed step-3 development tapes, declared as a replay.
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

from supply_chain import falsifiers as F                                        # noqa: E402
from supply_chain.arm_runner import seal_and_write                              # noqa: E402
from supply_chain.episode_metrics import compute_episode_metrics                # noqa: E402
from supply_chain.estar_bridge import make_expanded_sim                         # noqa: E402
from supply_chain.estar_kernel import (                                         # noqa: E402
    MASKS, SUPPLIER_LANES, DISPATCH_LANES, EStarAction)

CONTRACT = Path("docs/PREREGISTRO_PUERTA_C_AMORTIZACION_2026-08-09.md")
OUT = Path("results/program_n/gate_c_prereq_mpc_quality/result.json")
MASK = "M111"
EPOCHS, STEP_HOURS = 52, 168.0
#: Declared and closed. The planner enumerates these; the constant arm picks one on calibration.
GRID = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 1.0]
TAPE_SHARDS = ("step3_s1_r1r_a", "step3_s2_r1r_b", "step3_s3_r2r_a", "step3_s4_r2r_b")


def load_tapes():
    out = []
    for shard in TAPE_SHARDS:
        for path in sorted(Path("results", shard, "full").glob("*_actual_tapes.json")):
            for tape in json.loads(path.read_text()):
                out.append((shard, tape))
    return out


def action_for(sim, frac: float) -> EStarAction:
    """The declared action family: a fraction of every headroom the mask permits."""
    mask = MASKS[MASK]
    sup = tuple(SUPPLIER_LANES) if mask["P"] else ()
    dis = tuple(DISPATCH_LANES) if mask["D"] else ()
    proc = {lane: frac * min(sim._e_star_source_stock[lane],
                             sim.e_star_supplier_capacity[lane],
                             sim.e_star_transport_capacity[lane]) for lane in sup}
    targets = {node: (frac * sim.e_star_node_capacities[node] if mask["U"]
                      else sim._e_star_targets[node])
               for node in ("wdc", "al", "sb", "cssu_a", "cssu_b")}
    if not mask["U"]:
        for node in ("cssu_a", "cssu_b"):
            targets[node] = frac * sim.e_star_node_capacities[node]
    return EStarAction(procurement_qty=proc, buffer_targets=targets,
                       dispatch_qty={lane: 0.0 for lane in dis},
                       active_supplier_lanes=sup, active_dispatch_lanes=dis)


def run_sequence(tape, fracs):
    """Run one episode applying `fracs[k]` at epoch k. Returns the objective and the ledger."""
    sim = make_expanded_sim(tape, MASK)
    total = 0.0
    for k in range(EPOCHS):
        _, reward, done, _ = sim.step_e_star(action_for(sim, fracs[k]), step_hours=STEP_HOURS)
        total += float(reward)
        if done:
            break
    metrics = compute_episode_metrics(sim)
    return {"objective": total, "n_served": float(metrics["n_served"]),
            "n_lost": float(metrics["n_lost"]),
            "ret_full_ledger": float(metrics["ret_excel_full_ledger"])}


def des_mpc(tape, counters):
    """DirectDESMPC: at each epoch, one fresh DES rollout per candidate, then commit the best."""
    committed = []
    for k in range(EPOCHS):
        best, best_score = GRID[0], -np.inf
        for frac in GRID:
            sim = make_expanded_sim(tape, MASK)
            for j, prior in enumerate(committed):            # replay the committed prefix
                sim.step_e_star(action_for(sim, prior), step_hours=STEP_HOURS)
                counters["replay_steps"] += 1
            _, reward, _, _ = sim.step_e_star(action_for(sim, frac), step_hours=STEP_HOURS)
            counters["des_calls"] += 1
            counters["candidate_evaluations"] += 1
            if float(reward) > best_score:
                best, best_score = frac, float(reward)
        committed.append(best)
        counters["decisions"] += 1
    return committed


def paired(a, b, higher_is_better=True):
    d = np.array(a, float) - np.array(b, float)
    if not higher_is_better:
        d = -d
    n = d.size
    se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    t = 2.069 if n >= 24 else 2.228
    return {"mean": float(d.mean()), "ci95_low": float(d.mean() - t * se),
            "ci95_high": float(d.mean() + t * se), "n": int(n),
            "favourable": int((d > 0).sum()), "lcb_positive": bool(d.mean() - t * se > 0.0)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tapes", type=int, default=0, help="0 = all")
    ap.add_argument("--output", type=Path,
                    default=Path(__file__).resolve().parent.parent / OUT)
    args = ap.parse_args()
    started = time.perf_counter()

    tapes = load_tapes()
    if args.tapes:
        tapes = tapes[:args.tapes]
    print(f"  {len(tapes)} tapas", flush=True)

    # ---- p1 first: does the objective respond to the action at all? -------------------------
    constant_runs = {frac: [run_sequence(t, [frac] * EPOCHS) for _, t in tapes] for frac in GRID}
    print(f"  constantes listas ({time.perf_counter() - started:.0f}s)", flush=True)
    obj_by_frac = {f: float(np.mean([r["objective"] for r in rs]))
                   for f, rs in constant_runs.items()}
    lost_by_frac = {f: float(np.mean([r["n_lost"] for r in rs]))
                    for f, rs in constant_runs.items()}
    objective_spread = max(obj_by_frac.values()) - min(obj_by_frac.values())
    ledger_spread = max(lost_by_frac.values()) - min(lost_by_frac.values())

    counters = {"decisions": 0, "des_calls": 0, "candidate_evaluations": 0, "replay_steps": 0}
    mpc_rows, mpc_plans, mpc_seconds = [], [], []
    for shard, tape in tapes:
        t0 = time.perf_counter()
        plan = des_mpc(tape, counters)
        mpc_seconds.append(time.perf_counter() - t0)
        mpc_plans.append(plan)
        mpc_rows.append(run_sequence(tape, plan))
    print(f"  mpc listo ({time.perf_counter() - started:.0f}s)", flush=True)

    best_frac = max(GRID, key=lambda f: obj_by_frac[f])
    best_frac_ledger = min(GRID, key=lambda f: lost_by_frac[f])
    rng = np.random.default_rng(20260810)
    random_rows = [run_sequence(t, list(rng.choice(GRID, size=EPOCHS))) for _, t in tapes]

    def col(rows, key):
        return [r[key] for r in rows]

    vs = {
        "objective_vs_best_constant": paired(col(mpc_rows, "objective"),
                                             col(constant_runs[best_frac], "objective")),
        "lost_vs_best_constant": paired(col(mpc_rows, "n_lost"),
                                        col(constant_runs[best_frac_ledger], "n_lost"),
                                        higher_is_better=False),
        "objective_vs_random": paired(col(mpc_rows, "objective"), col(random_rows, "objective")),
        "lost_vs_random": paired(col(mpc_rows, "n_lost"), col(random_rows, "n_lost"),
                                 higher_is_better=False),
    }
    unique_plans = len({tuple(p) for p in mpc_plans})
    plan_is_constant = all(len(set(p)) == 1 for p in mpc_plans)

    checks = {
        "p1_the_objective_responds_to_the_action": F.check(
            objective_spread > 0.0,
            "a planner that scores every candidate on a flat objective is not a better decision "
            "maker; it is N evaluations of a constant, and nothing downstream of this is "
            "interpretable. This fails if the eight constant arms return the identical objective",
            computed_from={"objective_spread": objective_spread, "n_levels": len(GRID)},
            objective_by_frac=obj_by_frac, ledger_spread_n_lost=ledger_spread,
            ledger_by_frac=lost_by_frac),
        "p2_the_planner_beats_the_best_constant_on_its_own_objective": F.check(
            vs["objective_vs_best_constant"]["lcb_positive"],
            "the planner may simply not beat a well-searched constant, which is the outcome this "
            "project has measured in every other decision class",
            computed_from={"mean": vs["objective_vs_best_constant"]["mean"],
                           "ci95_low": vs["objective_vs_best_constant"]["ci95_low"]}),
        "p3_the_planner_beats_the_best_constant_on_the_physical_ledger": F.check(
            vs["lost_vs_best_constant"]["lcb_positive"],
            "winning an engineering reward while losing more orders would make the objective the "
            "wrong one, and this separates those two outcomes instead of conflating them",
            computed_from={"mean": vs["lost_vs_best_constant"]["mean"],
                           "ci95_low": vs["lost_vs_best_constant"]["ci95_low"]}),
        "p4_a_control_must_be_worse": F.check(
            vs["objective_vs_random"]["lcb_positive"] or vs["lost_vs_random"]["lcb_positive"],
            "if a random action sequence matches the planner on both endpoints, the comparison "
            "has no resolution and neither does the planner",
            computed_from={"objective_lcb": vs["objective_vs_random"]["ci95_low"],
                           "lost_lcb": vs["lost_vs_random"]["ci95_low"]}),
        "p5_the_plan_is_not_degenerate": F.check(
            not plan_is_constant,
            "a planner whose chosen sequence is a single repeated level IS a constant policy, "
            "however expensively it arrived there",
            computed_from={"n_unique_plans": unique_plans, "n_tapes": len(tapes)},
            plans_are_all_constant=plan_is_constant),
    }
    checks["custody"] = {
        "passed": None, "not_applicable": True,
        "evidence": {"why_it_can_fail": "it cannot: these are already-consumed step-3 development "
                                        "tapes, replayed on purpose. No seed is opened and the run "
                                        "carries no confirmatory grade.",
                     "shards": list(TAPE_SHARDS), "n_tapes": len(tapes), "seeds_opened": 0}}
    summary = F.summarise(checks)

    if not checks["p1_the_objective_responds_to_the_action"]["passed"]:
        status = "PLANNER_OBJECTIVE_IS_FLAT_NO_QUALITY_TO_MEASURE"
    elif not checks["p4_a_control_must_be_worse"]["passed"]:
        status = "BLOCKED_NO_RESOLUTION"
    elif (checks["p2_the_planner_beats_the_best_constant_on_its_own_objective"]["passed"]
          and checks["p3_the_planner_beats_the_best_constant_on_the_physical_ledger"]["passed"]):
        status = "EXPENSIVE_PLANNER_QUALIFIES_GATE_C_MAY_PROCEED"
    else:
        status = "EXPENSIVE_PLANNER_DOES_NOT_BEAT_THE_CHEAP_HEURISTIC"

    payload = {
        "schema_version": "program_n_gate_c_prereq_v1", "claim_status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_role": "PREREQUISITE_QUALITY_MEASUREMENT",
        "scope": "REPLAY_OF_CONSUMED_STEP3_TAPES_NO_SEEDS_OPENED_NO_CONFIRMATORY_GRADE",
        "mask": MASK, "epochs": EPOCHS, "grid": GRID, "n_tapes": len(tapes),
        "planner_work": counters,
        "planner_seconds_per_episode": {"mean": float(np.mean(mpc_seconds)),
                                        "p95": float(np.quantile(mpc_seconds, 0.95))},
        "objective_by_constant": obj_by_frac, "n_lost_by_constant": lost_by_frac,
        "best_constant_by_objective": best_frac, "best_constant_by_ledger": best_frac_ledger,
        "mpc_mean": {k: float(np.mean(col(mpc_rows, k))) for k in mpc_rows[0]},
        "random_mean": {k: float(np.mean(col(random_rows, k))) for k in random_rows[0]},
        "comparisons": vs, "n_unique_plans": unique_plans,
        "falsifiers": checks, "falsifier_summary": summary,
        "elapsed_seconds": time.perf_counter() - started,
        "contract_path": str(CONTRACT),
    }
    seal_and_write(payload, args.output, contract=CONTRACT,
                   reference=Path("results/program_n/gate_c0_expert_audit/result.json"))

    print(f"\nveredicto: {status}\n")
    print(f"  objetivo por constante: {json.dumps(obj_by_frac)}")
    print(f"  n_lost por constante:   {json.dumps({k: round(v, 2) for k, v in lost_by_frac.items()})}")
    print(f"\n  trabajo del planificador: {counters}")
    print(f"  {np.mean(mpc_seconds):.2f}s por episodio planificado\n")
    for name, v in vs.items():
        print(f"    {name:34} {v['mean']:+.4f} [{v['ci95_low']:+.4f}, {v['ci95_high']:+.4f}]  "
              f"{v['favourable']}/{v['n']}")
    print("\n  falsadores:")
    for k, v in checks.items():
        mark = "N/A " if v.get("not_applicable") else ("PASA" if v["passed"] else "FALLA")
        print(f"    {k:58} {mark}")
    print(f"\n  -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
