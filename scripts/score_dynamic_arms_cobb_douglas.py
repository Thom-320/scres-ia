#!/usr/bin/env python3
"""Score the corrected dynamic arm on the three-metric panel, C-D included.

Closes item §7 of `COBB_DOUGLAS_PORT_RESULTS_2026-07-29.md`: the index had only
been applied to static postures, and the DDMRP question lives in the dynamic arm.

Uses the **frozen v2 controllers** (commit b840256), which fix defects 3, 4 and 6
of `EXPANDED_CONTRACT_COMPARATORS_RECLASSIFICATION_2026-07-29.md`: the posture is
a three-node vector over the full 6^3 = 216 domain, and DDMRP is projected onto
exactly that domain by `nearest_posture`. Nothing here is written to any file the
v2 runner owns; the MPC arm is deliberately excluded because that run belongs to
the other session.

Projection changes the question. The v1 DDMRP injected 26-32x more material because
it emitted unbounded continuous targets, so "does the metric punish over-injection"
was the natural test. Bounded by the shared domain, that failure mode is gone by
construction, and what remains is the cleaner question: with decision rights matched,
does dynamic buffer control beat the best fixed posture under a metric that prices
resources at all?

Comparison set is declared here, before evaluation, because kappa_dot is normalised
by the whole set's cost (Eq. 5) and every member moves every other member's R.
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
    posture_name,
)
from supply_chain.supply_chain import MFSCSimulation  # noqa: E402

FAMILIES = {"R1r": ("R11", "R12", "R13", "R14"), "R2r": ("R21", "R22", "R23", "R24")}

# Declared before this corrected evaluation.  The 216-posture buffer gate found
# (672, 0, 1344), i.e. targets 61,440 / 0 / 126,000.  The earlier draft
# accidentally substituted the one-tape v2-preflight winner (168, 0, 168).
# Both are retained here, with distinct provenance, alongside homogeneous
# references spanning the range searched by v1.
STATIC_POSTURES: tuple[tuple[int, int, int], ...] = (
    (672, 0, 1344),       # 216-posture buffer-gate incumbent
    (168, 0, 168),        # one-tape v2-preflight winner
    (168, 168, 168),      # v1's best, homogeneous
    (0, 0, 0),
    (1344, 1344, 1344),
)


def run_arm(controller, *, seed: int, horizon: float, family: str,
            epoch_hours: float, period_hours: float, replenishment: float) -> dict:
    """One closed-loop episode; C-D variables sampled every `period_hours`."""
    sim = MFSCSimulation(
        shifts=1,
        initial_buffers={n: 0.0 for n in ("op3_rm", "op5_rm", "op9_rations")},
        inventory_replenishment_period=replenishment,
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
    controller.reset()
    rec = CobbDouglasRecorder(period_hours=period_hours)
    elapsed, epoch, since_decision = 0.0, 0, float("inf")
    decision_trace: list[dict] = []
    while elapsed < horizon:
        if since_decision >= epoch_hours:
            targets = controller.act(sim, epoch)
            sim.inventory_buffer_targets.update(
                {k: float(v) for k, v in targets.items()})
            diagnostic = getattr(controller, "last_diagnostic", {})
            decision_trace.append({
                "epoch": int(epoch),
                "time": float(sim.env.now),
                "targets": {k: float(v) for k, v in targets.items()},
                "posture": list(diagnostic.get(
                    "posture", getattr(controller, "posture", ())
                )),
            })
            since_decision = 0.0
            epoch += 1
        step = min(period_hours, horizon - elapsed)
        sim.step(action=None, step_hours=step)
        elapsed += step
        since_decision += step
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
        "strategic_injected": float(sim.total_strategic_raw_injected
                                    + sim.total_strategic_rations_injected),
        "decision_epochs": epoch,
        "decision_trace": decision_trace,
    })
    return agg


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
                    default=Path("results/cobb_douglas/dynamic_arms_v2.json"))
    args = ap.parse_args()

    contract = json.loads(args.contract.read_text())
    exponents = contract["exponents"]
    horizon = args.horizon_weeks * HOURS_PER_WEEK
    epoch_hours = args.epoch_weeks * HOURS_PER_WEEK
    started = time.perf_counter()

    out: dict[str, dict] = {}
    for family in args.families:
        arms: dict[str, list[dict]] = {}
        for posture in STATIC_POSTURES:
            c = VectorStaticPosture(posture)
            arms[c.name] = [
                run_arm(c, seed=t, horizon=horizon, family=family,
                        epoch_hours=epoch_hours, period_hours=args.period_hours,
                        replenishment=args.replenishment_hours)
                for t in args.tapes]
        d = ProjectedDDMRPController()
        arms[d.name] = [
            run_arm(d, seed=t, horizon=horizon, family=family,
                    epoch_hours=epoch_hours, period_hours=args.period_hours,
                    replenishment=args.replenishment_hours)
            for t in args.tapes]

        per_policy = {
            name: {k: sum(e[k] for e in eps) / len(eps)
                   for k in ("zeta", "epsilon", "phi", "tau", "kappa", "ret_excel",
                             "ret_excel_full_ledger", "ret_thesis", "flow_fill_rate",
                             "delivered_rations", "lost_orders", "strategic_injected")}
            for name, eps in arms.items()
        }
        scored = score_comparison_set(per_policy, exponents)
        action_counts: dict[str, int] = {}
        for episode in arms[d.name]:
            for decision in episode["decision_trace"]:
                posture_key = ",".join(str(int(v)) for v in decision["posture"])
                action_counts[posture_key] = action_counts.get(posture_key, 0) + 1
        out[family] = {
            "comparison_set": sorted(per_policy),
            "per_policy": {n: {**per_policy[n], **scored[n],
                               "per_tape": arms[n]} for n in per_policy},
            "ddmrp_action_counts": dict(sorted(action_counts.items())),
            "ddmrp_decision_count": int(sum(action_counts.values())),
        }
        print(f"  {family} done ({time.perf_counter() - started:.0f}s)", flush=True)

    payload = {
        "schema_version": "cobb_douglas_dynamic_arms_v2",
        "calibration_provenance": calibration_stamp(),
        "claim_status": "DEVELOPMENT_SCREEN_NO_CLAIM",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": ("correct §7 of COBB_DOUGLAS_PORT_RESULTS_2026-07-29.md: apply "
                    "the index to a dynamic controller on a rights-matched domain, "
                    "include the actual 216-posture gate incumbent, and persist "
                    "every DDMRP decision"),
        "controllers_from": "supply_chain/expanded_contract_controllers_v2.py (b840256)",
        "mpc_excluded": "owned by the concurrent v2 runner; not duplicated here",
        "contract_path": str(args.contract),
        "contract_self_sha256": contract.get("self_sha256"),
        "exponents": exponents,
        "degenerate_variables": contract.get("degenerate_variables", []),
        "comparison_set_declared": (
            [posture_name(p) for p in STATIC_POSTURES] + ["ddmrp_projected_v2"]),
        "tapes": list(args.tapes),
        "inference_boundary": (
            "Four development tapes; descriptive mechanism screen only. "
            "No confirmatory or neural claim."
        ),
        "metric_panel": ["ret_excel", "ret_excel_full_ledger", "ret_thesis",
                         "R_cobb_douglas"],
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
