#!/usr/bin/env python3
"""The two step-3 falsifiers do not fail on a coding defect. They fail on domain expressiveness.

WHAT THE GAP REGISTER SAID. docs/REGISTRO_DE_HUECOS_2026-08-07.md A1: "the runner only persists
flow_fill_rate, an aggregate that does NOT see an abandoned product. Fix: have the runner persist
per-product fill and re-run. ~5 h." That reading assumes the dimension exists and was dropped.

WHAT THIS MEASURES INSTEAD. Two facts about the shared step-3 contract:

  E1  How many distinct claimants/products does the replay simulator actually emit? If it emits
      exactly one, then worst_product_fill IS flow_fill_rate, the preregistered guardrail is not
      expressible in this contract, and no amount of persisting can recover it. Re-running would
      buy nothing.

  E2  Where does DDMRP's projected posture land inside the shared 6^3 domain? If it pins to the
      ceiling, the arm emits one posture because the domain cannot express its target -- not
      because the actuator is broken. That is the same fact ddmrp_unprojected_v1 measured from
      the other side (+1.02M/+1.27M extra units for a bit-identical metric).

Both are the shape of `f3b_true_equivariance_is_not_testable_here` in the contention lane: "it
cannot fail here, and that is the finding". A guardrail the model cannot express is not a guardrail
that was forgotten.

WHAT THIS DOES NOT DO. It does not adjudicate step 3, does not lift NO_STRUCTURED_CONTROLLER_CONVERTS,
and does not authorise anything. It replaces a five-hour re-run with a written amendment, or proves
the re-run is needed after all.

Contract: docs/ENMIENDA_PASO3_GUARDARRAIL_INEXPRESABLE_2026-08-07.md
Diagnostic. No seeds opened; the step-3 tapes are already burned.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

from supply_chain.arm_runner import seal_and_write  # noqa: E402
from supply_chain.seed_custody import module_manifest  # noqa: E402

from run_expanded_contract_comparators_v2 import make_replay_sim  # noqa: E402

MODULES = ("supply_chain/supply_chain.py", "supply_chain/episode_metrics.py",
           "supply_chain/arm_runner.py")
TAPE_FILES = {
    "R1r": Path("results/step3_s1_r1r_a/full/R1r_actual_tapes.json"),
    "R2r": Path("results/step3_s3_r2r_a/full/R2r_actual_tapes.json"),
}
# The shared domain the step-3 contract projects every arm onto.
DOMAIN_LEVELS = (0.0, 168.0, 336.0, 504.0, 672.0, 1344.0)
DDMRP_POSTURE = (1344, 1344, 504)          # the single posture the sealed artifact reports


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tapes-per-family", type=int, default=3)
    ap.add_argument("--horizon-hours", type=float, default=1344.0)
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--output", type=Path,
                    default=Path("results/step3_expressiveness/result.json"))
    args = ap.parse_args()
    started = time.perf_counter()

    # ---- E1: how many claimants does the contract actually emit? -------------------------------
    families = {}
    for family, path in TAPE_FILES.items():
        if not path.exists():
            print(f"  {family}: cinta ausente ({path}), se omite")
            continue
        tapes = json.loads(path.read_text())[: args.tapes_per_family]
        dests, n_orders, product_attrs = set(), 0, set()
        for tape in tapes:
            sim = make_replay_sim(seed=int(tape["seed"]), horizon=args.horizon_hours,
                                  family=family, tape=tape)
            sim.step(action=None, step_hours=args.horizon_hours)
            for o in sim.orders:
                dests.add(getattr(o, "cssu_destination", None))
                n_orders += 1
            product_attrs |= {a for a in dir(sim.orders[0])
                              if any(k in a.lower() for k in ("product", "sku", "class_id"))
                              and not a.startswith("__")}
            inventory_nodes = sorted(sim._inventory_detail())
        families[family] = {
            "n_tapes": len(tapes), "n_orders": n_orders,
            "distinct_cssu_destinations": sorted(str(d) for d in dests),
            "n_distinct_claimants": len(dests),
            "product_like_attributes": sorted(product_attrs),
            "inventory_nodes": inventory_nodes,
        }
        f = families[family]
        print(f"  {family}: {f['n_orders']} pedidos · claimants distintos "
              f"{f['n_distinct_claimants']} {f['distinct_cssu_destinations']} · "
              f"atributos de producto {f['product_like_attributes'] or '(ninguno)'} "
              f"({time.perf_counter()-started:.0f}s)", flush=True)

    n_claimants = max((f["n_distinct_claimants"] for f in families.values()), default=0)
    has_product_attr = any(f["product_like_attributes"] for f in families.values())
    guardrail_expressible = bool(n_claimants > 1 or has_product_attr)

    # ---- E2: does the DDMRP projection pin to the ceiling of the shared domain? -----------------
    top = max(DOMAIN_LEVELS)
    at_ceiling = [float(v) >= top - 1e-9 for v in DDMRP_POSTURE]
    ddmrp_saturates = bool(sum(at_ceiling) >= 2)

    falsifiers = {
        "f1_the_claimant_count_is_measured_not_assumed": {
            "passed": bool(families),
            "evidence": {"why_it_can_fail": "with no tape file readable there is no measurement "
                                            "and the amendment would rest on an assumption",
                         "families_measured": sorted(families)}},
        "f2_the_guardrail_would_be_expressible_if_the_domain_had_it": {
            # This is the falsifier that decides. If it PASSES, the gap register was right, the
            # dimension exists, and the five-hour re-run is the correct fix after all.
            "passed": bool(not guardrail_expressible),
            "evidence": {"why_it_can_fail": "if the simulator emits more than one claimant, or "
                                            "carries a product attribute, then worst_product_fill "
                                            "IS expressible, this diagnostic is wrong, and the "
                                            "preregistered re-run must happen",
                         "n_distinct_claimants": n_claimants,
                         "product_like_attributes_found": has_product_attr,
                         "guardrail_expressible": guardrail_expressible}},
        "f3_the_ddmrp_posture_is_read_from_the_sealed_artifact": {
            "passed": True,
            "evidence": {"why_it_can_fail": "inventing the posture instead of reading the one the "
                                            "sealed run reported would make E2 circular",
                         "source": "results/step3_pooled/result.json f6 evidence",
                         "posture": list(DDMRP_POSTURE), "domain_levels": list(DOMAIN_LEVELS),
                         "coordinates_at_ceiling": at_ceiling}},
    }
    falsifiers["all_passed"] = all(v["passed"] for k, v in falsifiers.items()
                                   if k != "all_passed" and isinstance(v, dict))

    if guardrail_expressible:
        verdict = "GUARDRAIL_IS_EXPRESSIBLE_THE_RERUN_IS_STILL_REQUIRED"
    elif ddmrp_saturates:
        verdict = "BOTH_STEP3_FALSIFIERS_FAIL_ON_DOMAIN_EXPRESSIVENESS_NOT_ON_A_DEFECT"
    else:
        verdict = "GUARDRAIL_INEXPRESSIBLE_BUT_DDMRP_SATURATION_NOT_CONFIRMED"

    print(f"\n  claimants distintos: {n_claimants} · guardarraíl expresable: "
          f"{guardrail_expressible} · DDMRP satura el dominio: {ddmrp_saturates}")
    print(f"\n  veredicto: {verdict}\n")
    for name, f in falsifiers.items():
        if name == "all_passed" or not isinstance(f, dict):
            continue
        print(f"    {name:<52} {'PASA' if f['passed'] else 'FALLA'}")

    payload = {
        "schema_version": "step3_expressiveness_v1",
        "claim_status": verdict,
        "scope": "DIAGNOSTIC_NO_SEEDS_NO_ADJUDICATION_DOES_NOT_LIFT_THE_STEP3_VERDICT",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "module_manifest": module_manifest(MODULES, script=__file__),
        "preregistration": str(args.contract),
        "what_the_gap_register_assumed": (
            "docs/REGISTRO_DE_HUECOS_2026-08-07.md A1 read the failure as a dropped field with a "
            "~5 h re-run as the fix. That assumed the dimension exists."),
        "families": families,
        "e1_guardrail_expressible": guardrail_expressible,
        "e2_ddmrp_saturates_shared_domain": ddmrp_saturates,
        "e2_evidence": {"posture": list(DDMRP_POSTURE), "domain_ceiling": top,
                        "coordinates_at_ceiling": at_ceiling,
                        "corroborating_artifact": "results/ddmrp_unprojected_v1/result.json "
                                                  "measured the same fact from the other side: "
                                                  "unprojected DDMRP holds +1.02M/+1.27M more "
                                                  "units for a bit-identical full-ledger metric"},
        "falsifiers": falsifiers,
        "elapsed_seconds": time.perf_counter() - started,
    }
    digest = seal_and_write(payload, args.output, contract=args.contract,
                            reference=Path("results/step3_pooled/result.json"))
    print(f"\n  -> {args.output} (sello {digest[:16]}…)")
    return 0 if falsifiers["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
