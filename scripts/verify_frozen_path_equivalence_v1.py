#!/usr/bin/env python3
"""Does today's physics still reproduce the cache that Confirmation 2 was sealed against?

WHY THIS EXISTS. `grid_transfer_confirmation_v2` -- the confirmation Paper 2's headline rests on --
declares a `module_manifest` that no longer matches the tree. `supply_chain/arm_runner.py` drifted
from `35135c53…` to its current bytes in commit `8ddf6f7`. Four tests mention `module_manifest` and
all four assert only the SHAPE of the dict, so the suite stayed silent.

AND THE DECLARED MANIFEST IS THE SMALLER PROBLEM. Its own `scope` field reads "declared modules and
entry script only; NOT the full execution dependency", and `supply_chain/supply_chain.py` -- which
is where the DES physics actually lives, and which also changed -- is not in it. A drift the
manifest can see is a lesser hazard than one it cannot. So this script does not merely recompute
what was declared: it recomputes a WIDENED manifest and reports both.

WHAT IT DOES NOT DO. It does not re-run the science, does not open seeds, and does not re-seal the
confirmation. It re-executes the anchor that artifact already carries -- falsifier
`f1_the_null_subgrid_reproduces_the_288_cache`, 103,680 cells with `max_abs_delta = 0.0` -- under
today's code, cell by cell, bit-exactly. Equivalence is a measurement, not an assurance.

If a single cell differs, the verdict is RERUN_REQUIRED_PROVENANCE_NOT_RECOVERABLE and no number
from that artifact may enter the manuscript.

Amendment 4 context: zero seed blocks remain, so a re-run would not be a confirmation anyway. That
raises the stakes on this certificate rather than lowering them.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from supply_chain.arm_runner import run_falsifiers, seal_and_write  # noqa: E402
from supply_chain.seed_custody import module_manifest  # noqa: E402

from build_transfer_confirmation_cache_v1 import BASE_CONFIGS, evaluate  # noqa: E402

TARGET = Path("results/grid_transfer_confirmation_v2/result.json")
CACHE = Path("results/surface_cache/garrido_transfer_confirmation_v2_base")
# The declared manifest covers two modules. This is the set the physics ACTUALLY depends on; the
# gap between the two is itself a finding and is reported, not quietly closed.
WIDENED = ("supply_chain/arm_runner.py", "supply_chain/seed_custody.py",
           "supply_chain/supply_chain.py", "supply_chain/env_experimental_shifts.py",
           "supply_chain/episode_metrics.py", "supply_chain/config.py")


def file_sha(p: str) -> str:
    f = Path(p)
    return sha256(f.read_bytes()).hexdigest() if f.exists() else "MISSING"


def replay_slice(payload: dict, seed: int) -> dict:
    """Re-run every cell of one cached slice and compare bit-exactly."""
    ctx, horizon = payload["context"], float(payload["horizon_hours"])
    worst, mismatches, worst_where = 0.0, 0, None
    for idx, expected in enumerate(payload["cells"]):
        got = evaluate(BASE_CONFIGS[idx], ctx, seed, horizon)
        deltas = [abs(got["value"] - float(expected["value"]))]
        deltas += [abs(a - float(b)) for a, b in zip(got["drivers"], expected["drivers"])]
        deltas += [abs(got["panel"][k] - float(v)) for k, v in expected["panel"].items()]
        d = max(deltas)
        if d > 0.0:
            mismatches += 1
            if d > worst:
                worst, worst_where = d, {"context": ctx, "seed": seed, "cell_index": idx,
                                         "config": BASE_CONFIGS[idx]}
    return {"context": ctx, "seed": seed, "cells": len(payload["cells"]),
            "mismatches": mismatches, "max_abs_delta": worst, "worst_cell": worst_where}


def mutation_control(payload: dict, seed: int) -> dict:
    """Re-introduce the defect the falsifier claims to detect. A comparator that cannot fail on a
    corrupted cell is not evidence that the uncorrupted ones agree."""
    mutated = json.loads(json.dumps(payload))
    original = float(mutated["cells"][0]["value"])
    mutated["cells"][0]["value"] = original + 1e-12
    out = replay_slice(mutated, seed)
    return {"detected": bool(out["mismatches"] >= 1),
            "injected_delta": 1e-12, "observed_max_abs_delta": out["max_abs_delta"],
            "original_value": original,
            "why_it_can_fail": ("if the comparator rounded, clipped or short-circuited, a corrupted "
                                "cell would pass and the clean result would mean nothing")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit-slices", type=int, default=0, help="0 = every slice (the real run)")
    ap.add_argument("--output", type=Path,
                    default=Path("results/frozen_path_equivalence/result.json"))
    ap.add_argument("--contract", type=Path, default=TARGET)
    args = ap.parse_args()
    t0 = time.time()

    target = json.loads(TARGET.read_text())
    declared = dict(target["module_manifest"]["modules"])
    declared_entry = target["module_manifest"]["entry_script"]

    slices = sorted(CACHE.rglob("*.json"))
    if args.limit_slices:
        slices = slices[:args.limit_slices]
    if not slices:
        raise SystemExit(f"no cached slices under {CACHE}")

    per_slice, total_cells, total_mismatch, worst = [], 0, 0, 0.0
    worst_where = None
    for i, path in enumerate(slices, 1):
        payload = json.loads(path.read_text())
        seed = int(path.stem)
        r = replay_slice(payload, seed)
        r["path"] = str(path)
        per_slice.append(r)
        total_cells += r["cells"]
        total_mismatch += r["mismatches"]
        if r["max_abs_delta"] > worst:
            worst, worst_where = r["max_abs_delta"], r["worst_cell"]
        print(f"  [{i:3d}/{len(slices)}] {r['context']:12} seed {seed}  "
              f"cells {r['cells']:4d}  mismatches {r['mismatches']:4d}  "
              f"max|Δ| {r['max_abs_delta']:.3e}", flush=True)

    control = mutation_control(json.loads(slices[0].read_text()), int(slices[0].stem))

    drift = {p: {"sealed": h, "current": file_sha(p), "drifted": file_sha(p) != h}
             for p, h in declared.items()}
    widened_now = {p: file_sha(p) for p in WIDENED}
    undeclared = [p for p in WIDENED if p not in declared]

    def f1():
        """Every replayed cell reproduces the sealed cache exactly.

        CAN FAIL: any behavioural change in the drifted modules moves a cell and the delta is not
        zero. CAN PASS: an additive, gated-off change leaves every cell identical -- and the
        mutation control below proves the comparator would have caught it either way."""
        ok = bool(total_mismatch == 0 and worst == 0.0)
        return ok, {"cells_checked": total_cells, "mismatches": total_mismatch,
                    "max_abs_delta": worst, "worst_cell": worst_where,
                    "slices": len(per_slice)}

    def f2():
        """The comparator detects a 1e-12 corruption injected into a cached cell."""
        return control["detected"], control

    def f3():
        """The declared manifest is narrower than the physics it depends on -- reported, not fixed.

        CAN FAIL: if the declared manifest already covered every module in WIDENED there would be
        no gap, and the 'invisible drift' concern would be unfounded."""
        return bool(undeclared), {
            "declared_modules": sorted(declared), "declared_entry_script": declared_entry,
            "undeclared_but_load_bearing": undeclared,
            "declared_scope_note": target["module_manifest"].get("scope"),
            "why_it_matters": ("supply_chain/supply_chain.py carries the DES physics and is absent "
                               "from the declared manifest, so a change there cannot be flagged by "
                               "recomputing what was declared")}

    def f4():
        """At least one declared module really did drift -- otherwise this certificate is answering
        a question nobody had."""
        d = [p for p, v in drift.items() if v["drifted"]]
        return bool(d), {"drifted_declared_modules": d, "detail": drift}

    fals = run_falsifiers({"f1_every_cell_reproduces_the_sealed_cache": f1,
                           "f2_mutation_control_is_detected": f2,
                           "f3_declared_manifest_is_narrower_than_the_physics": f3,
                           "f4_a_declared_module_actually_drifted": f4})

    verdict = ("FROZEN_PATH_EQUIVALENT_UNDER_CURRENT_PHYSICS"
               if fals["f1_every_cell_reproduces_the_sealed_cache"]["passed"]
               and fals["f2_mutation_control_is_detected"]["passed"]
               else "RERUN_REQUIRED_PROVENANCE_NOT_RECOVERABLE")

    payload = {
        "schema_version": "frozen_path_equivalence_v1", "claim_status": verdict,
        "scope": "PROVENANCE_CERTIFICATE_NO_SCIENCE_RERUN_NO_SEEDS_OPENED",
        "run_role": "PROVENANCE_CERTIFICATE",
        "primary_metric": "max_abs_delta_over_replayed_cache_cells",
        "self_sha256": None,
        "target_artifact": str(TARGET), "target_file_sha256": file_sha(str(TARGET)),
        "cache_root": str(CACHE), "slices_replayed": len(per_slice),
        "cells_replayed": total_cells,
        "declared_manifest_drift": drift, "widened_manifest_now": widened_now,
        "undeclared_but_load_bearing": undeclared,
        "per_slice": per_slice, "mutation_control": control,
        "falsifiers": fals, "elapsed_seconds": time.time() - t0,
        "module_manifest": module_manifest(WIDENED, script=__file__),
        "what_this_does_not_certify": [
            "it does not re-seal grid_transfer_confirmation_v2 or change any of its numbers",
            "it does not open seeds; zero blocks remain (claim freeze Amendment 4)",
            "it certifies BEHAVIOURAL equivalence of the replayed cache under current code, not "
            "that the drifted diffs are semantically inert in paths the cache does not exercise",
        ],
    }
    digest = seal_and_write(payload, args.output, contract=args.contract,
                            reference=Path("results/demand_process/result.json"))

    print(f"\nceldas {total_cells:,}  mismatches {total_mismatch}  max|Δ| {worst:.3e}")
    for k, v in fals.items():
        if k != "all_passed":
            print(f"  {k:52} {'PASA' if v['passed'] else 'FALLA'}")
    print(f"\n{verdict}\n  -> {args.output} (sello {digest[:16]}…)")
    return 0 if verdict.startswith("FROZEN_PATH_EQUIVALENT") else 1


if __name__ == "__main__":
    raise SystemExit(main())
