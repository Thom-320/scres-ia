#!/usr/bin/env python3
"""Program V on the full DES: does retained history still pay once the signal crosses Op1-Op13?

Contract: `docs/PREREGISTRO_PROGRAM_V_DES_PORT_2026-08-09.md`, frozen before this file.
Development replay on already-burned seeds 8600001-8600060. No virgin block is opened.

THE TAPES AND THE POLICIES ARE IMPORTED, NOT REIMPLEMENTED. `make_tape` and `policy_library` come
from the abstraction module unchanged, so a difference between the two results cannot be a
difference between two hand-written copies of the same idea. Same decision, same information, same
history; different physics.

WHAT THE ABSTRACTION COULD NOT ASK. It had scalar inventory, one weekly demand, and none of the
thirteen operations, recurrent risks, queues or transport. Retained belief was worth +0.0413 there.
Here the same belief has to survive procurement lead, assembly, downstream transport and the risk
process before it reaches service.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.arm_runner import seal_and_write                              # noqa: E402
from supply_chain import falsifiers as F                                        # noqa: E402
from supply_chain.program_v_supplier_memory import (                            # noqa: E402
    ACTIONS, HORIZON, Observation, make_tape, policy_library, update_posterior)
from supply_chain.seed_custody import custody_falsifier, module_manifest        # noqa: E402
from supply_chain.supply_chain import MFSCSimulation                            # noqa: E402

HOURS_PER_WEEK = 168.0
SELECT = tuple(range(8600001, 8600031))
HELD = tuple(range(8600031, 8600061))
OUT = Path("results/program_v/des_port_v1/result.json")
CONTRACT = Path("docs/PREREGISTRO_PROGRAM_V_DES_PORT_2026-08-09.md")
MODULES = ("supply_chain/supply_chain.py", "supply_chain/program_v_supplier_memory.py",
           "supply_chain/falsifiers.py", "supply_chain/arm_runner.py")


def _sim(seed: int, tape, mode: str = "v1") -> MFSCSimulation:
    return MFSCSimulation(
        seed=seed, horizon=HORIZON * HOURS_PER_WEEK,
        risks_enabled=True, risk_level="current",
        supplier_portfolio_mode=mode,
        supplier_yield_schedule=[tuple(y) for y in tape.yields])


def play(policy, seed: int) -> dict:
    tape = make_tape(seed)
    sim = _sim(seed, tape)
    sim._start_processes()

    posterior = np.full(3, 1.0 / 3.0)
    last_yields, last_mask = np.ones(3), np.zeros(3, dtype=bool)
    switches, previous = 0, None
    for week in range(HORIZON):
        # The belief sees only yields attached to deliveries it actually commissioned. A supplier
        # with zero allocation is neutral rather than leaked -- the abstraction's rule, kept.
        if week:
            alloc = np.asarray(sim._supplier_allocation_by_week.get(week, (1 / 3, 1 / 3, 1 / 3)))
            last_mask = alloc > 0
            last_yields = np.where(last_mask, np.asarray(tape.yields[week - 1]), 1.0)
        posterior = update_posterior(posterior, tape.warnings[week], last_yields, last_mask)
        obs = Observation(
            week=week, warning=tape.warnings[week],
            delayed_warning=tape.warnings[week - 1] if week else tape.warnings[week],
            shuffled_warning=tape.shuffled_warnings[week], true_regime=tape.regimes[week],
            inventory=float(sim.raw_material_wdc.level), backlog=float(sim.total_backorders),
            last_realized_yields=last_yields.copy(), posterior=posterior.copy())
        action = tuple(float(v) for v in policy.fn(obs))
        if action not in ACTIONS:
            raise ValueError(f"{policy.name} emitted infeasible action {action}")
        if previous is not None and action != previous:
            switches += 1
        previous = action
        if week + 1 < HORIZON:
            sim.commit_supplier_allocation(week + 1, action)
        sim.env.run(until=min((week + 1) * HOURS_PER_WEEK, sim.horizon))

    ledger = sim.flow_ledger()
    scale = max(abs(float(ledger.get("raw_sources", 0.0))),
                abs(float(ledger.get("ration_sources", 0.0))), 1.0)
    demanded = float(sim.total_demanded) or 1.0
    return {
        "service": float(sim.total_delivered) / demanded,
        "backlog_auc": float(sum(v for _, v in getattr(sim, "daily_inventory_theatre", []) or [])),
        "ordered": float(sim.supplier_ordered_units),
        "received": float(sim.supplier_received_units),
        "rejected": float(sim.supplier_rejected_units),
        "orders": len(sim.orders),
        "switches": switches,
        "mass_residual_rel": max(abs(float(ledger["raw_residual"])),
                                 abs(float(ledger["ration_residual"]))) / scale,
    }


def paired(a: np.ndarray, b: np.ndarray) -> dict:
    d = np.asarray(a) - np.asarray(b)
    boot = np.random.default_rng(20260809).choice(
        d, size=(20_000, d.size), replace=True).mean(axis=1)
    return {"mean": float(d.mean()), "lcb95": float(np.percentile(boot, 2.5)),
            "ucb95": float(np.percentile(boot, 97.5)),
            "favourable": int((d > 0).sum()), "n": int(d.size)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=Path(__file__).resolve().parent.parent / OUT)
    args = ap.parse_args()

    library = policy_library()
    rows = {}
    for policy in library:
        sel = [play(policy, s) for s in SELECT]
        held = [play(policy, s) for s in HELD]
        rows[policy.name] = {
            "deployable": bool(policy.deployable),
            "select_mean": float(np.mean([r["service"] for r in sel])),
            "held": [float(r["service"]) for r in held],
            "held_mean": float(np.mean([r["service"] for r in held])),
            "rejected": float(np.mean([r["rejected"] for r in held])),
            "orders": float(np.mean([r["orders"] for r in held])),
            "mass_residual_rel": float(max(r["mass_residual_rel"] for r in held)),
        }

    def arr(name):
        return np.array(rows[name]["held"])

    constants = [k for k in rows if k.startswith("constant_")]
    best_constant = max(constants, key=lambda k: rows[k]["select_mean"])
    observable = [k for k in rows if rows[k]["deployable"] and not k.startswith("constant_")
                  and not k.startswith("placebo")]
    best_observable = max(observable, key=lambda k: rows[k]["select_mean"])

    h_priv = paired(arr("privileged_true_state"), arr(best_constant))
    h_obs = paired(arr(best_observable), arr(best_constant))
    h_ret = paired(arr("bayes_retained"), arr("bayes_reset"))
    vs_delayed = paired(arr("bayes_retained"), arr("placebo_delayed"))
    vs_shuffled = paired(arr("bayes_retained"), arr("placebo_shuffled"))
    priv_over_bayes = paired(arr("privileged_true_state"), arr("bayes_retained"))

    # Inertness and liveness probes, on their own terms.
    tape = make_tape(SELECT[0])
    inert_off = _sim(SELECT[0], tape, mode="none")
    inert_off.run()
    inert_on = _sim(SELECT[0], tape, mode="v1")
    inert_on.supplier_yield_schedule = [(1.0, 1.0, 1.0)] * HORIZON
    inert_on.run()
    avoid_s0 = play(next(p for p in library if p.name == "privileged_true_state"), SELECT[0])
    flat = play(next(p for p in library if p.name == "constant_0"), SELECT[0])

    checks = {
        "f1_portfolio_is_inert_by_default": F.lt(
            abs(inert_off.total_external_raw_material - inert_on.total_external_raw_material), 1e-6,
            "with unit yields the extension must deliver exactly what the frozen DES delivers; if "
            "not, it changed the base physics rather than adding an opt-in layer"),
        "f2_allocation_moves_arrivals": F.gt(
            abs(avoid_s0["received"] - flat["received"]), 1e-6,
            "an inert action would make every contrast below noise"),
        "f3_rejected_is_never_destroyed_stock": F.lt(
            max(abs(r["ordered"] - r["received"] - r["rejected"]) for r in (avoid_s0, flat))
            + max(v["mass_residual_rel"] for v in rows.values()),
            1e-6,
            "what a supplier fails to deliver must never have entered the system; this is the "
            "exact defect retracted on 2026-08-08"),
        "f4_same_tape_same_risks": F.lt(
            float(np.ptp([v["orders"] for v in rows.values()])), 0.5,
            "if the order count moves with the policy the yields consumed simulator RNG and the "
            "arms are not the same world"),
        "f5_commitment_lead_binds": F.check(
            _lead_is_enforced(),
            "committing inside the week would let the decision see the yield it is supposed to "
            "anticipate, which would turn the whole mechanism into hindsight",
            computed_from={"weeks": HORIZON}),
        "f6_H_priv_material": F.ge(
            h_priv["lcb95"], 0.02,
            "the full DES may absorb the signal entirely, in which case there is no physical "
            "headroom to convert here at all"),
        "f7_H_obs_material": F.ge(
            h_obs["lcb95"], 0.01,
            "privileged headroom can exist while no deployable policy reaches it"),
        "f8_H_ret_positive": F.gt(
            h_ret["lcb95"], 0.0,
            "THE question of this port: retained belief was worth +0.0413 in the abstraction and "
            "may be worth nothing once the signal has to cross thirteen operations"),
        "f9_retained_beats_both_placebos": F.gt(
            min(vs_delayed["lcb95"], vs_shuffled["lcb95"]), 0.0,
            "if the delayed or shuffled placebo ties, what was measured is cadence, not history"),
    }
    checks["d1_privileged_residual"] = F.disclosure(
        "the privileged residual over retained belief is reported beside every positive: in the "
        "abstraction it was +0.00076 with UCB95 +0.0023, which is why that result authorised no "
        "learner",
        evidence={"privileged_over_bayes": priv_over_bayes})
    checks["d2_development_replay"] = F.disclosure(
        "burned development seeds, 30 selection / 30 held-out; no confirmatory grade is possible",
        evidence={"select": [SELECT[0], SELECT[-1]], "held": [HELD[0], HELD[-1]]})
    checks["custody"] = custody_falsifier(sorted(set(SELECT + HELD)))
    summary = F.summarise(checks)

    physics_ok = all(checks[k]["passed"] for k in
                     ("f1_portfolio_is_inert_by_default", "f3_rejected_is_never_destroyed_stock",
                      "f4_same_tape_same_risks", "f5_commitment_lead_binds"))
    if not physics_ok:
        status = "BLOCKED_INSTRUMENT"
    elif not checks["f6_H_priv_material"]["passed"]:
        status = "STOP_NO_PHYSICAL_HEADROOM_IN_THE_FULL_DES"
    elif not checks["f8_H_ret_positive"]["passed"]:
        status = "RETAINED_VALUE_DID_NOT_SURVIVE_THE_FULL_DES"
    else:
        status = "RETAINED_VALUE_SURVIVES_THE_FULL_DES"

    payload = {
        "schema_version": "program_v_des_port_v1", "claim_status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_role": "DEVELOPMENT",
        "scope": "DEVELOPMENT_REPLAY_ON_BURNED_SEEDS_NO_VIRGIN_BLOCK_NO_LEARNER",
        "endpoint": "theatre_fill_rate_delivered_over_demanded",
        "seeds": sorted(set(SELECT + HELD)),
        "abstraction_reference": {
            "artifact": "results/program_v/prelearner_gate_v1/result.json",
            "H_ret": {"mean": 0.04132023109186356, "lcb95": 0.026572283688821398},
            "H_obs": {"mean": 0.17936632254199697, "lcb95": 0.16298939723920858},
            "note": "the abstraction's numbers, for comparison only; different physics"},
        "best_constant": best_constant, "best_observable": best_observable,
        "effects": {"H_priv": h_priv, "H_obs": h_obs, "H_ret": h_ret,
                    "retained_vs_delayed": vs_delayed, "retained_vs_shuffled": vs_shuffled,
                    "privileged_over_bayes": priv_over_bayes},
        "rows": rows,
        "module_manifest": module_manifest(MODULES),
        "falsifiers": checks, "falsifier_summary": summary,
    }
    seal_and_write(payload, args.output, contract=CONTRACT,
                   reference=Path("results/program_v/prelearner_gate_v1/result.json"))

    print(f"veredicto: {status}\n")
    for name, eff in payload["effects"].items():
        print(f"  {name:24} {eff['mean']:+.6f} [{eff['lcb95']:+.6f}, {eff['ucb95']:+.6f}]  "
              f"{eff['favourable']}/{eff['n']}")
    print(f"\n  mejor constante {best_constant} | mejor observable {best_observable}")
    print(f"  falsadores: {summary['n_computed']} computados, {summary['n_failed']} fallidos")
    for name, c in checks.items():
        if c.get("computed"):
            print(f"    {name:40} {'PASA' if c['passed'] else 'FALLA'}")
    print(f"\n  -> {args.output}")
    return 0


def _lead_is_enforced() -> bool:
    """Committing into the current week must RAISE, not be silently accepted."""
    tape = make_tape(SELECT[0])
    sim = _sim(SELECT[0], tape)
    sim._start_processes()
    sim.env.run(until=3 * HOURS_PER_WEEK)
    try:
        sim.commit_supplier_allocation(3, (1.0, 0.0, 0.0))
    except ValueError:
        return True
    return False


if __name__ == "__main__":
    raise SystemExit(main())
