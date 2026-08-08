#!/usr/bin/env python3
"""Two machines, one seed, 10,006 rations apart -- and the base surface could not have found it.

WHAT THIS SEALS. The extended-surface verification reproduced 1,658,880 cells with zero differences
on the platform that produced them, and 52 of the 55 slices a second architecture checked reproduced
exactly too. Three did not. This artifact is those three, as data rather than as a README, because a
finding this specific should be citable from something with falsifiers.

THE HYPOTHESIS IT PUTS AT RISK. The base grid pins `op3_rm = op5_rm = 0` in every one of its 288
configurations; the extended grid is exactly the addition of those two raw-material factors. So the
cross-architecture agreement measured on the base surface never exercised that code path, and the
obvious explanation for the divergence appearing only now is that the path itself is where the two
platforms part. `f4` tests it: every divergent configuration must have at least one raw-material
factor above zero. If any divergent cell sits at the null level of both, the explanation is wrong
and the artifact says so.

WHAT IT IS NOT. It is not an adjudication of which platform is right, and neither is called wrong.
Both reproduce their own results on demand; they disagree with each other. It is also not a
non-determinism finding: repeated evaluation on either machine returns that machine's value.

Contract: docs/ENMIENDA_TOLERANCIA_EQUIVALENCIA_CROSS_PLATFORM_2026-08-08.md
Reads sealed shards, a captured second-architecture evaluation, and the frozen cache. No seed opens.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.verify_frozen_path_equivalence_v2 as V  # noqa: E402
import scripts.run_grid_transfer_v1 as G  # noqa: E402
from supply_chain.arm_runner import seal_and_write  # noqa: E402
from supply_chain.seed_custody import module_manifest  # noqa: E402

MODULES = ("supply_chain/arm_runner.py", "supply_chain/seed_custody.py",
           "supply_chain/supply_chain.py", "scripts/verify_frozen_path_equivalence_v2.py")
CONTRACT = Path("docs/ENMIENDA_TOLERANCIA_EQUIVALENCIA_CROSS_PLATFORM_2026-08-08.md")
CERTIFICATE = Path("results/frozen_path_equivalence_v2/result.json")
LOCAL_SHARDS = Path("results/frozen_path_equivalence_v2/shards")
VPS_SHARDS = Path("results/frozen_path_equivalence_v2/cross_platform_divergence")
CACHE = Path("results/surface_cache/garrido_transfer_confirmation_v2_ext/R1r_esc")
SLICES = ("8200011", "8200053", "8200054")
TOL_ATOL = TOL_RTOL = 1e-12          #: the frozen band, quoted not chosen


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--capture", type=Path, required=True,
                    help="JSON produced by the second architecture's re-evaluation")
    ap.add_argument("--out", type=Path,
                    default=Path("results/cross_platform_divergence/result.json"))
    args = ap.parse_args()

    capture = json.loads(args.capture.read_text())
    cert = json.loads((ROOT / CERTIFICATE).read_text())

    per_slice, all_divergent_cfgs, contexts = {}, [], set()
    for s in SLICES:
        vps_shard = json.loads((ROOT / VPS_SHARDS / f"ext__R1r_esc__{s}.json").read_text())
        local_shard = json.loads((ROOT / LOCAL_SHARDS / f"ext__R1r_esc__{s}.json").read_text())
        cap = capture["slices"][s]
        contexts.add(vps_shard["context"])

        # Re-evaluate the divergent cells HERE. The claim that the cache reproduces on its own
        # platform is the load-bearing one and it is recomputed rather than read off a shard.
        payload = json.loads((ROOT / CACHE / f"{s}.json").read_text())
        ctx, seed, hz = payload["context"], int(payload["seed"]), float(payload["horizon_hours"])
        local_deltas = []
        for row in cap["differing"]:
            i = row["cell_index"]
            exp = payload["cells"][i]
            got = V.evaluate(G.EXT_CONFIGS[i], ctx, seed, hz)
            d = abs(got["value"] - float(exp["value"]))
            for k, v in exp["panel"].items():
                d = max(d, abs(got["panel"][k] - float(v)))
            local_deltas.append(d)
            all_divergent_cfgs.append(G.EXT_CONFIGS[i])

        worst = max(cap["differing"], key=lambda r: r["max_abs_delta"]) if cap["differing"] else None
        per_slice[s] = {
            "context": vps_shard["context"], "seed": vps_shard["seed"],
            "cells": vps_shard["cells"],
            "vps_shard_mismatches": vps_shard["mismatches"],
            "vps_shard_max_abs_delta": vps_shard["max_abs_delta"],
            "capture_n_differing": cap["n_differing"],
            "local_shard_mismatches": local_shard["mismatches"],
            "local_recomputed_max_abs_delta": max(local_deltas) if local_deltas else 0.0,
            "worst_cell": worst,
        }

    n_cells_total = sum(v["cells"] for v in per_slice.values())
    n_divergent = sum(v["capture_n_differing"] for v in per_slice.values())
    raw_above_zero = [c for c in all_divergent_cfgs if c["op3_rm"] > 0 or c["op5_rm"] > 0]
    at_null = [c for c in all_divergent_cfgs if c["op3_rm"] == 0 and c["op5_rm"] == 0]
    outside_band = []
    for s, v in per_slice.items():
        w = v["worst_cell"]
        if w and w["max_abs_delta"] > TOL_ATOL + TOL_RTOL * abs(w.get("cache_value") or 0.0):
            outside_band.append((s, w["max_abs_delta"]))

    envs = {"producer_and_authoritative_verifier": "macOS arm64 (this machine)",
            "second_architecture": capture["host"]}

    falsifiers = {
        "f1_the_cache_reproduces_on_the_platform_that_produced_it": {
            "passed": all(v["local_shard_mismatches"] == 0
                          and v["local_recomputed_max_abs_delta"] == 0.0
                          for v in per_slice.values()),
            "detail": {s: {"shard": v["local_shard_mismatches"],
                           "recomputed_max_abs_delta": v["local_recomputed_max_abs_delta"]}
                       for s, v in per_slice.items()},
            "why_it_can_fail": ("if the divergent cells did not come back at zero here, the artifact "
                                "would not reproduce on its own environment either and this would "
                                "be a defect rather than a platform difference")},
        "f2_the_capture_and_the_second_architectures_shard_agree": {
            "passed": all(v["capture_n_differing"] == v["vps_shard_mismatches"]
                          for v in per_slice.values()),
            "detail": {s: [v["capture_n_differing"], v["vps_shard_mismatches"]]
                       for s, v in per_slice.items()},
            "why_it_can_fail": ("the capture is a separate run on that machine; if its cell count "
                                "disagreed with the shard, one of the two would be unrepeatable and "
                                "the divergence would be non-determinism rather than platform")},
        "f3_the_divergence_is_confined_to_one_context": {
            "passed": len(contexts) == 1, "contexts": sorted(contexts),
            "why_it_can_fail": ("a divergence spread across contexts would be a different and much "
                                "larger finding than one confined to R1r|esc")},
        # THE HYPOTHESIS, PUT AT RISK RATHER THAN ASSERTED.
        "f4_every_divergent_cell_uses_a_raw_material_factor_the_base_grid_never_exercises": {
            "passed": not at_null,
            "n_divergent_cells": len(all_divergent_cfgs),
            "n_with_a_raw_material_factor_above_zero": len(raw_above_zero),
            "n_at_the_null_level_of_both": len(at_null),
            "examples_at_null": at_null[:3],
            "why_it_can_fail": ("the base grid pins op3_rm = op5_rm = 0 everywhere, so if a "
                                "divergent cell sat at the null level of both, the explanation "
                                "for why 103,680 base cells agreed would be wrong")},
        "f5_the_gap_is_outside_the_frozen_tolerance_band": {
            "passed": bool(outside_band),
            "detail": outside_band, "atol": TOL_ATOL, "rtol": TOL_RTOL,
            "why_it_can_fail": ("if every difference fell inside atol + rtol|x| this would be the "
                                "numerically-equivalent case the amendment already covers, not a "
                                "material divergence needing its own artifact")},
        "f6_the_authoritative_certificate_is_clean": {
            "passed": (cert["verdict_b_forward_equivalence"]
                       == "CURRENT_HEAD_BEHAVIOURALLY_EQUIVALENT"
                       and cert["falsifiers"]["all_passed"] is True),
            "certificate_self_sha256": cert["self_sha256"],
            "why_it_matters": ("this artifact narrows a reproducibility claim; if the certificate it "
                               "narrows had not closed, there would be nothing to narrow")},
    }
    falsifiers["all_passed"] = all(v["passed"] for v in falsifiers.values() if isinstance(v, dict))

    payload = {
        "schema_version": "cross_platform_divergence_v1",
        "claim_status": ("HALTED_FALSIFIER_FAILED" if not falsifiers["all_passed"] else
                         "REPRODUCTION_IS_ENVIRONMENT_CONDITIONAL_ON_THE_RAW_MATERIAL_PATH"),
        "scope": "PROVENANCE_ONLY_NO_SCIENTIFIC_CLAIM_NO_NEW_SEEDS",
        "run_role": "PROVENANCE_VERIFICATION",
        "registration_status": "POST_HOC_CHARACTERISATION_UNDER_THE_FROZEN_TOLERANCE_AMENDMENT",
        "endpoint": "panel keys of cached cells re-evaluated on two architectures",
        "estimand": "identical config, context, seed and horizon across two environments",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "module_manifest": module_manifest(MODULES, script=__file__),
        "preregistration": str(CONTRACT),
        "environments": envs,
        "slices": per_slice,
        "coverage": {"slices_verified_on_the_second_architecture": 55,
                     "slices_reproducing_exactly_there": 52,
                     "slices_diverging": len(SLICES),
                     "cells_examined_in_those_slices": n_cells_total,
                     "cells_diverging": n_divergent},
        "why_the_base_surface_was_blind": (
            "every configuration of the 288 grid has op3_rm = op5_rm = 0, and the extended grid is "
            "the addition of exactly those two factors, so the bit-exact agreement measured over "
            "103,680 base cells was measured on a subspace where the divergent code path cannot be "
            "reached"),
        "what_this_does_not_say": [
            "which platform is correct -- neither is adjudicated and both reproduce their own value",
            "that the simulator is non-deterministic -- repeated evaluation on either machine "
            "returns that machine's value",
            "that the authoritative certificate is affected -- it runs on the producing platform "
            "and closed clean",
        ],
        "open": ("the mechanism is unidentified. A 10,006-ration gap from an identical seed points "
                 "at event ordering under ties rather than at arithmetic, and the two environments "
                 "share a Python minor version, so it is not a language-version effect"),
        "falsifiers": falsifiers,
    }
    seal_and_write(payload, ROOT / args.out, contract=ROOT / CONTRACT,
                   reference=ROOT / CERTIFICATE)

    print(f"\n{'rebanada':12}{'celdas':>8}{'dif VPS':>9}{'dif local':>11}{'max|Δ| VPS':>14}")
    for s, v in per_slice.items():
        w = v["worst_cell"]
        print(f"{s:12}{v['cells']:>8}{v['vps_shard_mismatches']:>9}"
              f"{v['local_shard_mismatches']:>11}{(w or {}).get('max_abs_delta', 0.0):>14.6g}")
    print(f"\nceldas divergentes: {n_divergent} de {n_cells_total:,} · contextos {sorted(contexts)}")
    print(f"con factor de materia prima > 0: {len(raw_above_zero)}/{len(all_divergent_cfgs)}")
    print(f"entornos: {envs['second_architecture']}")
    print(f"\nveredicto: {payload['claim_status']}")
    print(f"falsadores: {'todos pasan' if falsifiers['all_passed'] else 'FALLO'}")
    for n, v in falsifiers.items():
        if isinstance(v, dict) and not v["passed"]:
            print(f"  FALLA {n}")
    print(f"-> {args.out}")
    return 0 if falsifiers["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
