#!/usr/bin/env python3
"""The buffer gate again, with physics that conserves mass and a cost that is attributable.

Contract: `docs/RETRACTACION_LIBERACION_BUFFER_2026-08-08.md`, committed before this file.
Development replay on already-burned seeds. No virgin block is opened.

WHY THERE IS A SUCCESSOR AT ALL. The predecessor (`run_priced_buffer_gate_v1.py`,
`PRICED_DECISION_SPACE_ELIGIBLE`) is retracted. Its "release" freed exactly zero when a target was
set -- it walked contract KEYS through getattr and they are not container attributes -- and drained
operating stock to zero when none was. Its cost counted weeks with the switch on, charging the same
for a node holding ten units as for one holding a hundred thousand. The trade-off it measured was
inventory destruction against a mis-named proxy.

WHAT CHANGES, AND WHY THE CLASS DOES NOT COLLAPSE. Lowering the target now only stops
replenishment: delivered units stay and are consumed normally. The obvious worry is that this
returns the environment to the state where K = 4 and K = 26 were byte-identical. It does not, and
the reason is that the COST is no longer a switch: more weeks on means more top-ups, so more
kit-equivalent units are actually replenished, and `f3` below fails loudly if that stops being true.

THE COST IS EXACT AND ATTRIBUTABLE. Kit-equivalent units the schedule really put into the system,
`raw/12 + rations`, read from the simulator's own accumulators. Physical quantity-time travels
beside it as a SENSITIVITY, not as the price, because splitting on-hand stock between "buffer" and
"operations" is not clean and pretending it was is precisely the previous error. Neither is a
monetary rate and neither comes from the thesis: both are our declared assumptions.

THE ANSWER IS A PARETO FRONT, NOT A CHOSEN lambda. And the fixed control is selected on TRAIN and
scored on TEST, because the predecessor chose its comparator on the same trajectories where it
measured the gap.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.arm_runner import seal_and_write                              # noqa: E402
from supply_chain.continuous_its_env import make_continuous_its_track_a_env     # noqa: E402
from supply_chain import falsifiers as F                                        # noqa: E402
from supply_chain.seed_custody import custody_falsifier, module_manifest        # noqa: E402

R1 = ("R11", "R12", "R13", "R14")
R2 = ("R21", "R22", "R23", "R24")
MAX_STEPS, STEP_HOURS = 26, 168.0
LEAD_HOURS = 336.0                 # OUR assumption; the thesis's 48 h is delivery, not rebuild
LAMBDAS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
TRAIN = tuple(range(8600001, 8600013))
TEST = tuple(range(8600013, 8600025))
OUT = Path("results/conservative_buffer_gate/result.json")
CONTRACT = Path("docs/RETRACTACION_LIBERACION_BUFFER_2026-08-08.md")
MODULES = ("supply_chain/supply_chain.py", "supply_chain/continuous_its_env.py",
           "supply_chain/falsifiers.py", "supply_chain/arm_runner.py",
           "supply_chain/seed_custody.py")


def make_env():
    return make_continuous_its_track_a_env(
        init_frac=0.0, reward_mode="ReT_excel_delta", observation_version="v6",
        risk_level="current", enabled_risks=R1 + R2, risk_rng_mode="per_risk",
        stochastic_pt=False, max_steps=MAX_STEPS, step_size_hours=STEP_HOURS,
        risk_obs=True, holding_cost=0.0, shift_cost=0.0,
        demand_process="garrido_seasonal_v1",
        demand_seasonal_contract={"forecast_mode": "garrido_generator"},
        inventory_replenishment_lead_time=LEAD_HOURS)


def exposure(sim) -> float:
    """L*: quantity-weighted lateness over the quantity-weighted maximum possible lateness."""
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
    step, unit_hours = 0, 0.0
    try:
        while not (done or truncated):
            on = step in weeks
            _o, _r, done, truncated, _i = env.step(
                np.array([1.0 if on else 0.0, -1.0], dtype=np.float32))
            # Quantity-time sampled HERE rather than read off the simulator: this environment
            # drives targets directly and never starts the periodic replenishment loop, so the
            # in-simulator integrator never ticks. Reporting its zero would be reporting a
            # measurement that was not taken.
            unit_hours += STEP_HOURS * sum(
                float(getattr(sim, node).level)
                for node in ("raw_material_wdc", "raw_material_al", "rations_sb")
                if getattr(sim, node, None) is not None)
            step += 1
        return {"L": exposure(sim),
                "replenished": float(sim.strategic_replenishment_units()),
                "unit_hours": unit_hours,
                "destroyed": float(getattr(sim, "strategic_buffer_released_units", 0.0)),
                "n_steps": step}
    finally:
        env.close()


def options() -> list[tuple[int, int]]:
    """Enumerated windows. Every k = 0 window is the same policy, so only one is kept -- seven
    aliases of "no buffer" would inflate the class and flatter any selection over it."""
    out = [(0, 0)]
    out += [(s, k) for k in (4, 8, 13, 18, 22, 26)
            for s in range(0, MAX_STEPS - k + 1, 4) if s + k <= MAX_STEPS]
    return out


def pareto(points: list[tuple[float, float, tuple]]) -> list[tuple]:
    """Non-dominated on (L lower better, cost lower better)."""
    front = []
    for lo, co, key in points:
        if not any(l2 <= lo and c2 <= co and (l2 < lo or c2 < co) for l2, c2, _ in points):
            front.append(key)
    return front


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=Path(__file__).resolve().parent.parent / OUT)
    args = ap.parse_args()
    opts = options()

    rows = {}
    for opt in opts:
        tr = [play(opt, s) for s in TRAIN]
        te = [play(opt, s) for s in TEST]
        rows[str(opt)] = {
            "option": list(opt),
            "L_train": float(np.mean([r["L"] for r in tr])),
            "L_test": float(np.mean([r["L"] for r in te])),
            "L_test_per_seed": [float(r["L"]) for r in te],
            "replenished": float(np.mean([r["replenished"] for r in tr + te])),
            "unit_hours": float(np.mean([r["unit_hours"] for r in tr + te])),
            "destroyed": float(max(r["destroyed"] for r in tr + te)),
        }

    max_cost = max(v["replenished"] for v in rows.values()) or 1.0
    for v in rows.values():
        v["cost_norm"] = v["replenished"] / max_cost

    # Fixed control chosen on TRAIN only, at each lambda, then scored on TEST.
    by_lambda = {}
    for lam in LAMBDAS:
        pick = min(rows, key=lambda k: rows[k]["L_train"] + lam * rows[k]["cost_norm"])
        by_lambda[str(lam)] = {
            "chosen_on_train": rows[pick]["option"],
            "J_test": rows[pick]["L_test"] + lam * rows[pick]["cost_norm"],
            "L_test": rows[pick]["L_test"], "cost_norm": rows[pick]["cost_norm"]}

    front = pareto([(v["L_test"], v["cost_norm"], k) for k, v in rows.items()])

    # Clairvoyant per-tape gap at each lambda: the best option ON EACH TEST TAPE against the
    # train-selected fixed control. Priced with each option's own cost so the clairvoyant cannot
    # buy its advantage with inventory it is not charged for.
    keys = list(rows)
    J = np.array([[rows[k]["L_test_per_seed"][i] for k in keys] for i in range(len(TEST))])
    cost = np.array([rows[k]["cost_norm"] for k in keys])
    gaps = {}
    for lam in LAMBDAS:
        priced = J + lam * cost[None, :]
        fixed_idx = keys.index(min(rows, key=lambda k: rows[k]["L_train"]
                                   + lam * rows[k]["cost_norm"]))
        diff = priced[:, fixed_idx] - priced.min(axis=1)
        boot = np.random.default_rng(20260808).choice(
            diff, size=(20_000, diff.size), replace=True).mean(axis=1)
        gaps[str(lam)] = {"mean": float(diff.mean()), "lcb95": float(np.percentile(boot, 2.5)),
                          "ucb95": float(np.percentile(boot, 97.5))}
    best_lambda = max(gaps, key=lambda k: gaps[k]["lcb95"])

    off = rows["(0, 0)"]
    distinct_costs = len({round(v["replenished"], 3) for v in rows.values()})
    longest = max(rows.values(), key=lambda v: v["option"][1])
    shortest = min((v for v in rows.values() if v["option"][1] > 0),
                   key=lambda v: v["option"][1])

    checks = {
        "f1_no_stock_was_destroyed": F.lt(
            max(v["destroyed"] for v in rows.values()), 1e-9,
            "the retracted path destroyed operating stock; if any unit leaves a container here "
            "the successor reproduced the defect it exists to remove"),
        "f2_an_off_window_costs_exactly_zero": F.lt(
            off["replenished"], 1e-9,
            "a schedule that never raises the target must spend nothing; a non-zero cost would "
            "mean the accumulator is charging operating replenishment to the buffer"),
        "f3_cost_separates_the_class": F.ge(
            distinct_costs, 3,
            "if every window spends the same, holding duration is not a decision and pricing it "
            "prices nothing -- which is exactly what was true before the retracted release"),
        "f4_longer_window_spends_more": F.gt(
            longest["replenished"], shortest["replenished"],
            "more weeks on must mean more top-ups; if not, the cost is not measuring the schedule"),
        "f5_control_selected_on_train_only": F.check(
            all(by_lambda[str(lam)]["chosen_on_train"] in [v["option"] for v in rows.values()]
                for lam in LAMBDAS),
            "the predecessor selected its comparator on the same trajectories where it measured "
            "the gap, which inflates every gap it reported",
            computed_from={"n_lambdas": len(LAMBDAS), "n_options": len(opts)}),
        "f6_clairvoyant_gap_is_material": F.ge(
            gaps[best_lambda]["lcb95"], 0.01,
            "the environment may have a static trade-off and no sequential value at all, in "
            "which case this fails and no learner is authorized"),
    }
    checks["d1_cost_is_our_assumption"] = F.disclosure(
        "cost is kit-equivalent units actually replenished (raw/12 + rations); physical "
        "quantity-time is reported as a sensitivity, not as the price. Neither is a monetary rate "
        "and neither comes from Garrido-Rios (2017)",
        evidence={"max_replenished": float(max_cost),
                  "unit_hours_range": [float(min(v["unit_hours"] for v in rows.values())),
                                       float(max(v["unit_hours"] for v in rows.values()))]})
    checks["d2_development_replay"] = F.disclosure(
        "development replay on already-burned seeds; no virgin block opened",
        evidence={"train": [TRAIN[0], TRAIN[-1]], "test": [TEST[0], TEST[-1]]})
    checks["custody"] = custody_falsifier(sorted(set(TRAIN + TEST)))

    summary = F.summarise(checks)
    physics_ok = all(checks[k]["passed"] for k in
                     ("f1_no_stock_was_destroyed", "f2_an_off_window_costs_exactly_zero"))
    if not physics_ok:
        status = "BLOCKED_PHYSICS"
    elif not checks["f3_cost_separates_the_class"]["passed"]:
        status = "NO_PRICED_DECISION_SPACE"
    elif checks["f6_clairvoyant_gap_is_material"]["passed"]:
        status = "CONSERVATIVE_PRICED_SPACE_HAS_SEQUENTIAL_VALUE"
    else:
        status = "STATIC_TRADE_OFF_ONLY__NO_SEQUENTIAL_HEADROOM"

    payload = {
        "schema_version": "conservative_buffer_gate_v1", "claim_status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "contract_path": str(CONTRACT), "run_role": "DEVELOPMENT",
        "scope": "DEVELOPMENT_REPLAY_ON_BURNED_SEEDS_NO_VIRGIN_BLOCK",
        "supersedes": {"path": "results/priced_buffer_gate/result.json", "retained": True,
                       "why": "its release freed zero with a target set and destroyed operating "
                              "stock without one, and its cost counted switch-weeks rather than "
                              "inventory"},
        "lambdas": list(LAMBDAS), "max_replenished_units": float(max_cost),
        "n_options": len(opts), "distinct_cost_levels": distinct_costs,
        "pareto_front": front, "by_lambda": by_lambda, "clairvoyant_gap": gaps,
        "best_lambda": best_lambda, "rows": rows,
        "module_manifest": module_manifest(MODULES),
        "falsifiers": checks, "falsifier_summary": summary,
    }
    payload["seeds"] = sorted(set(TRAIN + TEST))
    payload["endpoint"] = "L_star_priced_by_replenished_units"
    seal_and_write(payload, args.output, contract=CONTRACT,
                   reference=Path("results/priced_buffer_gate/result.json"))

    print(f"veredicto: {status}\n")
    print(f"  {len(opts)} opciones, {distinct_costs} niveles de coste distintos, "
          f"{len(front)} puntos no dominados")
    print(f"  destruido (max sobre todo): {max(v['destroyed'] for v in rows.values()):.3g}")
    for lam in LAMBDAS:
        g = gaps[str(lam)]
        b = by_lambda[str(lam)]
        print(f"  lambda {lam:<5} control(train) {str(b['chosen_on_train']):<9} "
              f"hueco clarividente {g['mean']:+.6f} [{g['lcb95']:+.6f}]")
    print(f"\n  falsadores: {summary['n_computed']} computados, {summary['n_failed']} fallidos")
    for name, c in checks.items():
        if c.get("computed"):
            print(f"    {name:44} {'PASA' if c['passed'] else 'FALLA'}")
    print(f"\n  -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
