#!/usr/bin/env python3
"""Why no structured controller converts: above the incumbent, more buffer buys exactly nothing.

THE OBSERVATION THAT FORCED THIS. Unprojected DDMRP -- writing continuous targets that exceed the
thesis ladder ceiling at 312 of 312 decision points -- returns ret_excel_full_ledger BIT-IDENTICAL
to the projected version on all 24 tapes. My first reading was that the write had no effect. It is
not that: writing zeros gives 0.1420 and writing 5,000,000 gives 0.2164, so the targets have real
authority. The system is simply already at a point where raising them changes nothing.

WHAT THIS MEASURES. One-factor perturbations around a reference posture: each node's target set to
zero and to ten times its reference, everything else held. The asymmetry between the two directions
is the result -- if downward moves the metric and upward does not, the contract is saturated and
there is nothing above the incumbent for any controller to capture.

That is the mechanism behind NO_STRUCTURED_CONTROLLER_CONVERTS, and it is a stronger statement than
"MPC lost": it says what the ceiling is made of.

Development on materialised tapes. Descriptive; no gate, no verdict.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from supply_chain.arm_runner import seal_and_write  # noqa: E402
from supply_chain.episode_metrics import compute_episode_metrics  # noqa: E402
from supply_chain.expanded_contract_controllers import NODES  # noqa: E402
from supply_chain.seed_custody import module_manifest  # noqa: E402

from run_expanded_contract_comparators_v2 import make_replay_sim  # noqa: E402

METRIC = "ret_excel_full_ledger"
MODULES = ("supply_chain/supply_chain.py", "supply_chain/arm_runner.py")
MULTIPLIERS = (0.0, 0.5, 2.0, 10.0)


def episode(tape, family: str, targets: dict, horizon: float, epoch_hours: float) -> float:
    sim = make_replay_sim(seed=int(tape["seed"]), horizon=horizon, family=family, tape=tape)
    while float(sim.env.now) < horizon - 1e-9:
        sim.inventory_buffer_targets.update(targets)
        sim.step(action=None, step_hours=min(epoch_hours, horizon - float(sim.env.now)))
    return float(compute_episode_metrics(sim)[METRIC])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", nargs="+", type=Path, required=True)
    ap.add_argument("--tapes-per-family", type=int, default=3)
    ap.add_argument("--horizon-weeks", type=int, default=52)
    ap.add_argument("--epoch-weeks", type=int, default=4)
    ap.add_argument("--reference", type=json.loads,
                    default='{"op3_rm": 122880.0, "op5_rm": 122880.0, "op9_rations": 47250.0}')
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--output", type=Path,
                    default=Path("results/buffer_saturation_diagnostic/result.json"))
    args = ap.parse_args()
    horizon, epoch_hours = args.horizon_weeks * 168.0, args.epoch_weeks * 168.0
    ref = {k: float(v) for k, v in args.reference.items()}

    report, seeds = {}, []
    for shard in args.shards:
        for path in sorted(shard.glob("*_actual_tapes.json")):
            family = path.name.split("_actual_tapes")[0]
            if family in report:
                continue
            tapes = json.loads(path.read_text())[: args.tapes_per_family]
            base = [episode(t, family, ref, horizon, epoch_hours) for t in tapes]
            seeds += [int(t["seed"]) for t in tapes]
            nodes = {}
            for node in NODES:
                per_mult = {}
                for m in MULTIPLIERS:
                    tgt = dict(ref, **{node: ref[node] * m})
                    vals = [episode(t, family, tgt, horizon, epoch_hours) for t in tapes]
                    per_mult[f"x{m:g}"] = {
                        "mean": sum(vals) / len(vals),
                        "delta_vs_reference": sum(vals) / len(vals) - sum(base) / len(base)}
                down = per_mult["x0"]["delta_vs_reference"]
                up = per_mult["x10"]["delta_vs_reference"]
                nodes[node] = {"by_multiplier": per_mult,
                               "delta_down_at_zero": down, "delta_up_at_10x": up,
                               "saturated_upward": bool(abs(up) < 1e-12),
                               "has_downward_authority": bool(abs(down) > 1e-9)}
            report[family] = {"reference_targets": ref, "n_tapes": len(tapes),
                              "reference_mean": sum(base) / len(base), "nodes": nodes}
            print(f"\n  == {family} · {len(tapes)} tapes · referencia "
                  f"{report[family]['reference_mean']:.6f}")
            for node, r in nodes.items():
                print(f"     {node:<14} a 0 {r['delta_down_at_zero']:+.6f} · "
                      f"x10 {r['delta_up_at_10x']:+.6f}"
                      f"{'   SATURADO ARRIBA' if r['saturated_upward'] else ''}")

    all_sat = all(r["saturated_upward"] for f in report.values() for r in f["nodes"].values())
    any_down = any(r["has_downward_authority"] for f in report.values()
                   for r in f["nodes"].values())

    falsifiers = {
        "f1_the_lever_is_not_inert": {
            "passed": any_down,
            "evidence": {"why_it_can_fail": "if neither direction moved the metric the buffers "
                                            "would have no authority at all and this diagnostic "
                                            "would be measuring a disconnected knob rather than a "
                                            "saturated one",
                         "any_node_has_downward_authority": any_down}},
        "f2_saturation_is_directional": {
            "passed": all_sat and any_down,
            "evidence": {"why_it_can_fail": "the claim is that UP does nothing while DOWN does. If "
                                            "some node still gains from more buffer, the contract "
                                            "is not saturated there and a controller could convert",
                         "all_nodes_saturated_upward": all_sat,
                         "per_family": {f: {n: {"up": r["delta_up_at_10x"],
                                                "down": r["delta_down_at_zero"]}
                                            for n, r in b["nodes"].items()}
                                        for f, b in report.items()}}},
    }
    falsifiers["all_passed"] = all(v["passed"] for k, v in falsifiers.items()
                                   if k != "all_passed" and isinstance(v, dict))
    print()
    for name, f in falsifiers.items():
        if name == "all_passed" or not isinstance(f, dict):
            continue
        print(f"    {name:<36} {'PASA' if f['passed'] else 'FALLA'}")

    payload = {
        "schema_version": "buffer_saturation_diagnostic_v1",
        "claim_status": "DESCRIPTIVE_DIAGNOSTIC_NO_ADJUDICATION",
        "scope": "DEVELOPMENT_ON_MATERIALISED_TAPES_NO_FRESH_SEEDS",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "module_manifest": module_manifest(MODULES, script=__file__),
        "what_this_explains": (
            "NO_STRUCTURED_CONTROLLER_CONVERTS in results/step3_pooled. If raising any node's "
            "target changes nothing while lowering it hurts, there is no headroom above the "
            "incumbent for a controller -- structured or neural -- to capture. The negative is a "
            "property of the contract, not a failure of the controllers."),
        "retraction": (
            "An earlier reading of the bit-identical unprojected DDMRP was that the target write "
            "had no effect. It does: zeros give 0.1420 and 5,000,000 gives 0.2164 on the same "
            "tape. The write works; the system is saturated."),
        "metric": METRIC, "multipliers": list(MULTIPLIERS),
        "families": report, "seeds": sorted(set(seeds)), "falsifiers": falsifiers,
    }
    digest = seal_and_write(payload, args.output, contract=args.contract,
                            reference=Path("results/step3_pooled/result.json"))
    print(f"\n  -> {args.output} (sello {digest[:16]}…)")
    return 0 if falsifiers["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
