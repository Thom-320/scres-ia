#!/usr/bin/env python3
"""Step 3 of Garrido's design: run the structured controllers on the EXPANDED contract.

Development screen. No confirmation universe opened, no learner trained, no claim.

Garrido, 2026-07-28: "Baseline, modelo de Garrido. Luego tu corres el MPC con los
datos originales del modelo. Luego corres el MPC con mas variables. Y vemos como se
comporta." Steps one and two exist. This is step three.

The question it answers is the one that gates everything downstream: on a contract
where the decision right is worth +11% to +25% (H2/H3), does a structured controller
convert that value, or does it saturate? The neural residual is defined against the
best structured controller, so until this runs there is no denominator.

Three arms, identical decision rights (the three strategic buffer targets), identical
information (the simulator's own state at each epoch), identical admissible set (the
Table 6.16 ladder plus zero):

  static_I*            the incumbent, one rung held throughout
  ddmrp_dynamic        real DDMRP with rolling ADU and dynamic zones
  mpc_receding_horizon replans against the real DES each epoch

The static incumbent is inside the MPC's candidate set at every epoch, so an MPC that
loses to it has failed to find a solution it could reach. That is a search failure and
is reported as such, not as evidence about the decision right.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.config import HOURS_PER_WEEK  # noqa: E402
from supply_chain.config import THESIS_FAITHFUL_PROTOCOL as P  # noqa: E402
from supply_chain.episode_metrics import compute_episode_metrics  # noqa: E402
from supply_chain.expanded_contract_controllers import (  # noqa: E402
    LADDER_HOURS,
    DDMRPController,
    ReceedingHorizonMPC,
    StaticPosture,
    level_targets,
)
from supply_chain.supply_chain import MFSCSimulation  # noqa: E402

FAMILIES = {
    "R1r": ("R11", "R12", "R13", "R14"),
    "R2r": ("R21", "R22", "R23", "R24"),
}


def make_sim(seed: int, horizon: float, family: str, increased: tuple[str, ...],
             period: float):
    return MFSCSimulation(
        shifts=1,
        initial_buffers=level_targets(0),
        inventory_replenishment_period=period,
        seed=seed,
        horizon=horizon,
        risks_enabled=True,
        risk_level="current",
        enabled_risks=set(FAMILIES[family]),
        risk_overrides={r: "increased" for r in increased},
        strict_exogenous_crn=True,
        year_basis=P["year_basis"],
        warmup_trigger=P["warmup_trigger"],
        r14_defect_mode=P["r14_defect_mode"],
    )


def run_controlled(controller, seed, horizon, family, increased, epoch_hours,
                   period, metric="ret_excel") -> dict:
    """Closed-loop episode: the controller rewrites the targets each epoch."""
    sim = make_sim(seed, horizon, family, increased, period)
    controller.reset()
    epoch, elapsed = 0, 0.0
    while elapsed < horizon:
        targets = controller.act(sim, epoch)
        # The replenishment process re-reads this dict on every cycle, so writing
        # it here is closed-loop control rather than re-parameterisation.
        sim.inventory_buffer_targets.update({k: float(v) for k, v in targets.items()})
        step = min(epoch_hours, horizon - elapsed)
        sim.step(action=None, step_hours=step)
        elapsed += step
        epoch += 1
    m = compute_episode_metrics(sim)
    return {
        "ret_excel": float(m["ret_excel"]),
        "ret_excel_full_ledger": float(m["ret_excel_full_ledger"]),
        "ret_thesis": float(m["ret_thesis"]),
        "flow_fill_rate": float(m["flow_fill_rate"]),
        "lost_orders": float(m["lost_orders"]),
        "delivered_rations": float(m["delivered_rations"]),
        "strategic_injected": float(sim.total_strategic_raw_injected
                                    + sim.total_strategic_rations_injected),
        "epochs": epoch,
    }


def make_rollout(seed, horizon, family, increased, epoch_hours, period, metric):
    """Objective of a run whose epochs follow `prefix` then `candidate` throughout."""

    def rollout(prefix: list[int], candidate: int, scenario_seed: int) -> float:
        plan = list(prefix) + [candidate]
        sim = make_sim(scenario_seed, horizon, family, increased, period)
        elapsed, epoch = 0.0, 0
        while elapsed < horizon:
            hours = plan[epoch] if epoch < len(plan) else candidate
            sim.inventory_buffer_targets.update(
                {k: float(v) for k, v in level_targets(hours).items()})
            step = min(epoch_hours, horizon - elapsed)
            sim.step(action=None, step_hours=step)
            elapsed += step
            epoch += 1
        return float(compute_episode_metrics(sim)[metric])

    return rollout


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--families", nargs="+", default=["R1r", "R2r"])
    ap.add_argument("--tapes", type=int, default=6)
    ap.add_argument("--seed-start", type=int, default=1_310_001)
    ap.add_argument("--horizon-weeks", type=int, default=52)
    ap.add_argument("--epoch-weeks", type=int, default=4,
                    help="decision cadence; 4 weeks ~ Op2's native monthly review")
    ap.add_argument("--replenishment-hours", type=float, default=168.0)
    ap.add_argument("--mpc-scenarios", type=int, default=2)
    ap.add_argument("--metric", default="ret_excel")
    ap.add_argument("--skip-mpc", action="store_true")
    ap.add_argument("--output-dir", type=Path,
                    default=Path("results/expanded_contract_comparators"))
    args = ap.parse_args()

    horizon = args.horizon_weeks * HOURS_PER_WEEK
    epoch_hours = args.epoch_weeks * HOURS_PER_WEEK
    tapes = [args.seed_start + i for i in range(args.tapes)]
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out: dict = {}

    for family in args.families:
        # Hold the risk pattern fixed at the fully-increased corner so the contrast
        # is between controllers, not between risk patterns.
        increased = FAMILIES[family]
        arms: dict[str, list[dict]] = {}

        for hours in LADDER_HOURS:
            c = StaticPosture(hours)
            arms[c.name] = [run_controlled(c, t, horizon, family, increased,
                                           epoch_hours, args.replenishment_hours)
                            for t in tapes]
            print(f"  {family} {c.name:14s} "
                  f"mean={np.mean([r['ret_excel'] for r in arms[c.name]]):.6f} "
                  f"({time.perf_counter()-started:.0f}s)", flush=True)

        d = DDMRPController()
        arms[d.name] = [run_controlled(d, t, horizon, family, increased,
                                       epoch_hours, args.replenishment_hours)
                        for t in tapes]
        print(f"  {family} {d.name:14s} "
              f"mean={np.mean([r['ret_excel'] for r in arms[d.name]]):.6f} "
              f"({time.perf_counter()-started:.0f}s)", flush=True)

        if not args.skip_mpc:
            mpc_rows = []
            for t in tapes:
                scen = [t + 10_000 * (k + 1) for k in range(args.mpc_scenarios)]
                m = ReceedingHorizonMPC(
                    rollout=make_rollout(t, horizon, family, increased, epoch_hours,
                                         args.replenishment_hours, args.metric),
                    scenario_seeds=scen)
                mpc_rows.append(run_controlled(m, t, horizon, family, increased,
                                               epoch_hours, args.replenishment_hours))
                print(f"    mpc tape {t} done, {m.plan_calls} rollouts "
                      f"({time.perf_counter()-started:.0f}s)", flush=True)
            arms["mpc_receding_horizon"] = mpc_rows
            print(f"  {family} mpc            "
                  f"mean={np.mean([r['ret_excel'] for r in mpc_rows]):.6f}", flush=True)

        best_static = max(
            (k for k in arms if k.startswith("static_")),
            key=lambda k: float(np.mean([r["ret_excel"] for r in arms[k]])))
        bs = np.array([r["ret_excel"] for r in arms[best_static]])
        summary = {}
        for name, rows in arms.items():
            v = np.array([r["ret_excel"] for r in rows])
            d_ = v - bs
            summary[name] = {
                "mean_ret_excel": float(v.mean()),
                "mean_ret_excel_full_ledger": float(
                    np.mean([r["ret_excel_full_ledger"] for r in rows])),
                "mean_ret_thesis": float(np.mean([r["ret_thesis"] for r in rows])),
                "mean_flow_fill_rate": float(np.mean([r["flow_fill_rate"] for r in rows])),
                "mean_strategic_injected": float(
                    np.mean([r["strategic_injected"] for r in rows])),
                "delta_vs_best_static_mean": float(d_.mean()),
                "delta_positive_tapes": int((d_ > 0).sum()),
                "n_tapes": len(rows),
            }
        out[family] = {"best_static": best_static, "arms": summary}

    payload = {
        "schema_version": "expanded_contract_comparators_v1",
        "claim_status": "DEVELOPMENT_SCREEN_NO_CLAIM",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "Garrido 2026-07-28 design step 3: MPC on the expanded contract",
        "decision_rights": ["op3_rm", "op5_rm", "op9_rations"],
        "admissible_levels_hours": list(LADDER_HOURS),
        "static_incumbent_inside_mpc_candidate_set": True,
        "transducer_reuse": "invalid here; buffer targets change action-independent events",
        "tapes": tapes,
        "horizon_weeks": args.horizon_weeks,
        "epoch_weeks": args.epoch_weeks,
        "results": out,
        "elapsed_seconds": time.perf_counter() - started,
    }
    body = json.dumps(payload, indent=1, sort_keys=True)
    payload["self_sha256"] = sha256(body.encode()).hexdigest()
    path = args.output_dir / "result.json"
    path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(f"\n-> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
