#!/usr/bin/env python3
"""DDMRP with its own action domain, and the asymmetry that creates written into the artifact.

WHY. In results/step3_pooled the DDMRP arm emitted ONE posture, (1344, 1344, 504), across all 78
decision points, so its paired contrast against the best static was zero by construction and the
run supported no claim about DDMRP in either direction. The cause is not the fallback ADU -- that
only bites for three epochs -- but a scale mismatch: raw-material on-hand runs at 3.4M units while
the thesis ladder tops out at 122,880, so nearest_posture pins every target to the top rung.

WHAT CHANGES. The controller writes its continuous targets straight into inventory_buffer_targets
instead of snapping them to the ladder. That is the real method, unmutilated.

THE ASYMMETRY IS THE POINT, NOT A FOOTNOTE. This gives DDMRP strictly MORE decision rights than
every other arm, so a win here is confounded with the wider action set and is reported as "wins
with rights the others do not have" -- never as method superiority. A loss, conversely, is stronger
evidence than a loss under equal rights would be, because it had more freedom and still failed to
beat the best fixed posture.

Tapes are the ones already materialised by the step-3 shards. No seed is opened.

Amendment: docs/ENMIENDA_DDMRP_FUERA_DEL_DOMINIO_COMPARTIDO_2026-08-06.md
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from supply_chain.arm_runner import seal_and_write  # noqa: E402
from supply_chain.episode_metrics import compute_episode_metrics  # noqa: E402
from supply_chain.expanded_contract_controllers import LADDER_HOURS, NODES, level_targets  # noqa: E402
from supply_chain.expanded_contract_controllers_v2 import ProjectedDDMRPController  # noqa: E402
from supply_chain.seed_custody import module_manifest  # noqa: E402

from run_expanded_contract_comparators_v2 import make_replay_sim, state_hash  # noqa: E402

ARM = "ddmrp_unprojected_v3"
METRIC = "ret_excel_full_ledger"
MODULES = ("supply_chain/expanded_contract_controllers_v2.py", "supply_chain/arm_runner.py")
CEILING = {n: float(level_targets(max(LADDER_HOURS))[n]) for n in NODES}


def run_unprojected(tape, horizon: float, family: str, epoch_hours: float):
    """Same controller, same diagnostics -- only the projection to the ladder is removed."""
    sim = make_replay_sim(seed=int(tape["seed"]), horizon=horizon, family=family, tape=tape)
    controller = ProjectedDDMRPController()
    trace, epoch = [], 0
    while float(sim.env.now) < horizon - 1e-9:
        before = state_hash(sim)
        controller.act(sim, epoch)                      # fills last_diagnostic
        diag = controller.last_diagnostic
        continuous = {n: float(diag["nodes"][n]["continuous_target"]) for n in NODES}
        # THE ONE CHANGE: write the continuous targets, do not snap them to the ladder.
        sim.inventory_buffer_targets.update(continuous)
        trace.append({"epoch": epoch, "time": float(sim.env.now), "state_hash": before,
                      "continuous_targets": continuous,
                      "above_ladder_ceiling": {n: bool(continuous[n] > CEILING[n]) for n in NODES},
                      "ddmrp": diag})
        sim.step(action=None, step_hours=min(epoch_hours, horizon - float(sim.env.now)))
        epoch += 1
    metric = compute_episode_metrics(sim)
    row = {"arm": ARM, "family": family, "tape_seed": int(tape["seed"]),
           "posture": None,
           METRIC: float(metric["ret_excel_full_ledger"]),
           "ret_excel": float(metric["ret_excel"]),
           "flow_fill_rate": float(metric["flow_fill_rate"]),
           "lost_orders": float(metric.get("lost_orders", 0.0)),
           "delivered_rations": float(metric.get("delivered_rations", 0.0)),
           "demanded_rations": float(metric.get("demanded_rations", 0.0)),
           "terminal_stock": float(metric.get("terminal_stock", 0.0))}
    return row, trace


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", nargs="+", type=Path, required=True,
                    help="step-3 shard 'full' directories holding <family>_actual_tapes.json")
    ap.add_argument("--horizon-weeks", type=int, default=52)
    ap.add_argument("--epoch-weeks", type=int, default=4)
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--output", type=Path,
                    default=Path("results/ddmrp_unprojected_v1/result.json"))
    args = ap.parse_args()
    horizon, epoch_hours = args.horizon_weeks * 168.0, args.epoch_weeks * 168.0

    rows, traces, seeds = [], {}, []
    for shard in args.shards:
        for tapes_path in sorted(shard.glob("*_actual_tapes.json")):
            family = tapes_path.name.split("_actual_tapes")[0]
            for tape in json.loads(tapes_path.read_text()):
                row, tr = run_unprojected(tape, horizon, family, epoch_hours)
                rows.append(row)
                traces[f"{family}:{tape['seed']}"] = tr
                seeds.append(int(tape["seed"]))
                print(f"  {family} tape {tape['seed']} -> {METRIC} {row[METRIC]:.6f}", flush=True)

    distinct = {n: {round(e["continuous_targets"][n], 3) for tr in traces.values() for e in tr}
                for n in NODES}
    above = {n: sum(1 for tr in traces.values() for e in tr if e["above_ladder_ceiling"][n])
             for n in NODES}
    n_points = sum(len(tr) for tr in traces.values())

    print("\n  objetivos continuos distintos por nodo:")
    for n in NODES:
        print(f"    {n:<14} {len(distinct[n]):>4} valores · "
              f"{above[n]}/{n_points} por encima del techo {CEILING[n]:,.0f}")

    falsifiers = {
        "f1_the_targets_actually_vary": {
            "passed": all(len(distinct[n]) > 1 for n in NODES),
            "evidence": {"why_it_can_fail": "this is f6 of the pooled run without the projection "
                                            "that flattened it. If the targets are STILL constant, "
                                            "the projection was not the cause and the method is "
                                            "degenerate on its own",
                         "distinct_per_node": {n: len(distinct[n]) for n in NODES}}},
        "f2_the_targets_leave_the_old_ceiling": {
            "passed": any(above[n] > 0 for n in NODES),
            "evidence": {"why_it_can_fail": "if no target ever exceeds the ladder ceiling, taking "
                                            "DDMRP out of the shared domain changed nothing and "
                                            "the whole amendment was unnecessary",
                         "points_above_ceiling": above, "n_decision_points": n_points,
                         "ceiling": CEILING}},
        "f3_the_tapes_are_the_same_ones": {
            "passed": len(seeds) == len(set(seeds)) and len(seeds) > 0,
            "evidence": {"why_it_can_fail": "the contrast is paired against an incumbent computed "
                                            "on these exact tapes; a different or duplicated tape "
                                            "set would break the pairing",
                         "n_tapes": len(seeds), "seeds": sorted(seeds)}},
        "f4_the_asymmetry_is_recorded": {"passed": True, "evidence": {
            "why_it_can_fail": "if the artifact ever reports this as a symmetric comparison the "
                               "field is missing and the claim is unsupportable",
            "recorded": True}},
    }
    falsifiers["all_passed"] = all(v["passed"] for k, v in falsifiers.items()
                                   if k != "all_passed" and isinstance(v, dict))
    for name, f in falsifiers.items():
        if name == "all_passed" or not isinstance(f, dict):
            continue
        print(f"    {name:<40} {'PASA' if f['passed'] else 'FALLA'}")

    payload = {
        "schema_version": "ddmrp_unprojected_v1",
        "claim_status": "DEVELOPMENT_ASYMMETRIC_RIGHTS_NO_METHOD_CLAIM",
        "scope": "DEVELOPMENT_ON_MATERIALISED_TAPES_NO_FRESH_SEEDS",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "module_manifest": module_manifest(MODULES, script=__file__),
        "amendment": "docs/ENMIENDA_DDMRP_FUERA_DEL_DOMINIO_COMPARTIDO_2026-08-06.md",
        "corrects": "results/step3_pooled/result.json (f6 failed)",
        "metric": METRIC, "arm": ARM,
        "decision_rights_asymmetry": {
            "this_arm": "continuous targets, unbounded by the thesis ladder",
            "every_other_arm": f"the {len(LADDER_HOURS)}^{len(NODES)} posture domain",
            "how_to_read_a_win": ("NOT method superiority. Confounded with a wider action set; "
                                 "report as 'wins with rights the others do not have'."),
            "how_to_read_a_loss": ("STRONGER than a loss under equal rights: more freedom and "
                                   "still no better than the best fixed posture.")},
        "ladder_ceiling": CEILING,
        "distinct_targets_per_node": {n: len(distinct[n]) for n in NODES},
        "points_above_ladder_ceiling": above, "n_decision_points": n_points,
        "rows": rows, "seeds": sorted(seeds), "falsifiers": falsifiers,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    (args.output.parent / "traces.json").write_text(json.dumps(traces))
    (args.output.parent / "rows.json").write_text(json.dumps(rows, indent=1))
    digest = seal_and_write(payload, args.output, contract=args.contract,
                            reference=Path("results/step3_pooled/result.json"))
    print(f"\n  -> {args.output} (sello {digest[:16]}…)")
    return 0 if falsifiers["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
