#!/usr/bin/env python3
"""Does the current source still produce the frozen surfaces the Paper 2 claims are built on?

WHY THIS EXISTS. On 2026-08-07 two modules were edited AFTER the artifacts that name them in their
`module_manifest` were sealed:

    supply_chain/arm_runner.py     8ddf6f7 17:56  (seal_and_write gained seed_block/endpoint)
    supply_chain/supply_chain.py   cc3af32 20:03  (the seasonal demand engine)

`grid_transfer_confirmation_v2` is the project's headline confirmation and its cache manifest now
disagrees with the tree. Nothing in `tests/` caught it: the four tests that mention `module_manifest`
assert the SHAPE of the dict, never the hashes. A manuscript cannot cite a result whose producing
code is unidentified, so either the drift is proven harmless or the surfaces must be rebuilt.

WHY NOT REUSE f1. `grid_transfer_confirmation_v2`'s own `f1_the_null_subgrid_reproduces_the_288_cache`
compares the extended cache against the base cache -- two frozen files. It never touches
`supply_chain.py`, so it passes identically before and after any drift and proves nothing here. The
only test that can settle this is to RE-EVALUATE cached cells against the live simulator.

WHAT IT DOES. Draws a deterministic sample of cells from each frozen cache, re-runs the DES under
the current tree, and demands exact equality of both the primary scalar and the full observable
panel. The simulator is deterministic given a seed, so `almost equal` would be the wrong bar: any
difference at all is a behavioural change.

Reuses `evaluate` from each cache's own builder rather than reimplementing it -- a reimplementation
would be testing this script, not the physics.

Contract: docs/ENMIENDA_REGISTRO_DE_EVIDENCIA_2026-08-07.md
Read-only over artifacts. No seed outside the already-burned blocks is opened.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.build_transfer_confirmation_cache_v1 import (  # noqa: E402
    EXT_CONFIGS,
    evaluate as evaluate_transfer,
)
from scripts.run_meta_learner_normaliser_audit_v1 import (  # noqa: E402
    CONFIGS as WRAP_CONFIGS,
    evaluate as evaluate_wrap,
)
from supply_chain.arm_runner import seal_and_write  # noqa: E402
from supply_chain.seed_custody import module_manifest  # noqa: E402

MODULES = ("supply_chain/supply_chain.py", "supply_chain/arm_runner.py",
           "supply_chain/config.py", "supply_chain/episode_metrics.py",
           "supply_chain/seed_custody.py")

#: Which frozen cache feeds which manuscript claim, and the builder whose `evaluate` wrote it.
#: `garrido_transfer_confirmation_v2_base` is a PROJECTION of the extended slices, not an
#: independent evaluation, so re-running it would re-test the same physics through a narrower
#: window; the extended grid is the one that carries the DES.
TARGETS = {
    "grid_transfer_confirmation_v2": {
        "cache_root": "results/surface_cache/garrido_transfer_confirmation_v2_ext",
        "configs": EXT_CONFIGS,
        "evaluate": evaluate_transfer,
        "feeds": "results/grid_transfer_confirmation_v2/result.json",
    },
    "search_ladder_v5": {
        "cache_root": "results/surface_cache/wrap288_v1",
        "configs": WRAP_CONFIGS,
        "evaluate": evaluate_wrap,
        "feeds": "results/search_ladder_v5/result.json",
    },
}

#: The drift this certificate was written to adjudicate. Naming it up front is what lets f4 fail:
#: a file that drifted and is NOT on this list is an unexamined change, not a cleared one.
DECLARED_DRIFT = ("supply_chain/supply_chain.py", "supply_chain/arm_runner.py")


def slices_of(cache_root: Path) -> list[Path]:
    return sorted(cache_root.glob("*/*.json"))


def sample_cells(cache_root: Path, n_cells: int, rng: np.random.Generator) -> list[tuple[Path, int]]:
    """Spread the sample across every context, then across seeds, then across configurations.

    A sample concentrated in one context or one corner of the grid could pass while the physics
    moved somewhere else, so the spread is part of the test and f2 checks it.
    """
    paths = slices_of(cache_root)
    by_context: dict[str, list[Path]] = {}
    for p in paths:
        by_context.setdefault(p.parent.name, []).append(p)
    contexts = sorted(by_context)
    if not contexts:
        raise SystemExit(f"halt: no cache slices under {cache_root}")
    picks: list[tuple[Path, int]] = []
    per_context = max(1, n_cells // len(contexts))
    for ctx in contexts:
        group = by_context[ctx]
        for k in range(per_context):
            path = group[int(rng.integers(0, len(group)))]
            n = len(json.loads(path.read_text())["cells"])
            picks.append((path, int(rng.integers(0, n))))
            del k
    return picks


def check_cell(path: Path, index: int, configs: list[dict],
               evaluate: Callable[..., dict]) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    cached = payload["cells"][index]
    live = evaluate(configs[index], payload["context"], int(payload["seed"]),
                    float(payload["horizon_hours"]))
    value_ok = float(cached["value"]) == float(live["value"])
    panel_ok = cached["panel"] == live["panel"]
    row = {
        "slice": str(path), "context": payload["context"], "seed": int(payload["seed"]),
        "config_index": index, "value_identical": value_ok, "panel_identical": panel_ok,
        "identical": bool(value_ok and panel_ok),
    }
    if not row["identical"]:
        row["cached_value"] = float(cached["value"])
        row["live_value"] = float(live["value"])
        row["panel_keys_differing"] = sorted(
            k for k in cached["panel"] if cached["panel"].get(k) != live["panel"].get(k))
    return row


def mutation_control(path: Path, index: int, configs: list[dict],
                     evaluate: Callable[..., dict]) -> dict[str, Any]:
    """Re-run one cell, then compare the LIVE result against a deliberately corrupted cache value.

    A comparator that cannot see a planted difference cannot certify an absent one. This repository
    has already shipped a falsifier hardcoded to `passed: True`, so the control is not optional.
    """
    payload = json.loads(path.read_text())
    cached = dict(payload["cells"][index])
    live = evaluate(configs[index], payload["context"], int(payload["seed"]),
                    float(payload["horizon_hours"]))
    corrupted_value = float(cached["value"]) + 1e-9
    corrupted_panel = dict(cached["panel"])
    first_key = sorted(corrupted_panel)[0]
    corrupted_panel[first_key] = float(corrupted_panel[first_key]) + 1e-9
    return {
        "slice": str(path), "config_index": index, "perturbation": 1e-9,
        "value_mutation_detected": corrupted_value != float(live["value"]),
        "panel_mutation_detected": corrupted_panel != live["panel"],
        "panel_key_mutated": first_key,
        "clean_cell_still_matches": (float(cached["value"]) == float(live["value"])
                                     and cached["panel"] == live["panel"]),
    }


def manifest_drift() -> dict[str, Any]:
    """Recompute every cache manifest against the tree and name what moved."""
    out, drifted = {}, set()
    for name, spec in TARGETS.items():
        paths = slices_of(Path(spec["cache_root"]))
        stored = json.loads(paths[0].read_text())["module_manifest"]
        live = module_manifest(tuple(stored["modules"]), script=stored["entry_script"])
        rows = {}
        for mod, digest in stored["modules"].items():
            now = live["modules"].get(mod)
            rows[mod] = {"manifest": digest, "current": now, "match": digest == now}
            if digest != now:
                drifted.add(mod)
        entry_match = stored["entry_script_sha256"] == live["entry_script_sha256"]
        if not entry_match:
            drifted.add(stored["entry_script"])
        out[name] = {"entry_script": stored["entry_script"], "entry_script_match": entry_match,
                     "modules": rows}
    out["drifted_files"] = sorted(drifted)
    out["undeclared_drift"] = sorted(drifted - set(DECLARED_DRIFT))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--cells-per-cache", type=int, default=120)
    ap.add_argument("--out", type=Path, default=Path("results/frozen_path_equivalence/result.json"))
    args = ap.parse_args()

    started = time.perf_counter()
    rng = np.random.default_rng(20260807)
    results, controls, seeds_touched = {}, {}, set()
    for name, spec in TARGETS.items():
        root = Path(spec["cache_root"])
        picks = sample_cells(root, args.cells_per_cache, rng)
        rows = []
        for path, index in picks:
            row = check_cell(path, index, spec["configs"], spec["evaluate"])
            seeds_touched.add(row["seed"])
            rows.append(row)
        n_bad = sum(1 for r in rows if not r["identical"])
        results[name] = {
            "cache_root": str(root), "feeds": spec["feeds"], "n_cells": len(rows),
            "n_identical": len(rows) - n_bad, "n_differing": n_bad,
            "contexts": sorted({r["context"] for r in rows}),
            "n_distinct_seeds": len({r["seed"] for r in rows}),
            "n_distinct_configs": len({r["config_index"] for r in rows}),
            "differing": [r for r in rows if not r["identical"]][:20],
        }
        controls[name] = mutation_control(picks[0][0], picks[0][1], spec["configs"],
                                          spec["evaluate"])
        print(f"  {name:32s} {len(rows) - n_bad}/{len(rows)} celdas idénticas · "
              f"{len(results[name]['contexts'])} contextos · "
              f"{results[name]['n_distinct_seeds']} semillas", flush=True)

    drift = manifest_drift()
    all_identical = all(v["n_differing"] == 0 for v in results.values())
    controls_ok = all(c["value_mutation_detected"] and c["panel_mutation_detected"]
                      and c["clean_cell_still_matches"] for c in controls.values())

    falsifiers = {
        "f1_every_sampled_cell_reproduces_exactly": {
            "passed": all_identical,
            "evidence": {
                "why_it_can_fail": "if either 2026-08-07 edit changed the frozen path, a re-run "
                                   "under the current tree returns a different number and the "
                                   "confirmation cannot be cited without rebuilding its surface",
                "by_cache": {k: {"n_cells": v["n_cells"], "n_differing": v["n_differing"]}
                             for k, v in results.items()}}},
        "f2_the_sample_spans_the_grid": {
            "passed": all(len(v["contexts"]) >= 6 and v["n_distinct_seeds"] >= 5
                          and v["n_distinct_configs"] >= 20 for v in results.values()),
            "evidence": {
                "why_it_can_fail": "a sample concentrated in one context or one corner could be "
                                   "identical while the physics moved elsewhere; this demands the "
                                   "six contexts, several seeds and many configurations",
                "by_cache": {k: {"contexts": v["contexts"], "seeds": v["n_distinct_seeds"],
                                 "configs": v["n_distinct_configs"]} for k, v in results.items()}}},
        "f3_the_comparator_detects_a_planted_difference": {
            "passed": controls_ok,
            "evidence": {
                "why_it_can_fail": "a comparator blind to a 1e-9 perturbation would report "
                                   "equivalence for any tree at all; this repository has already "
                                   "shipped a falsifier hardcoded to passed:True",
                "controls": controls}},
        "f4_the_measured_drift_is_the_declared_drift": {
            "passed": not drift["undeclared_drift"],
            "evidence": {
                "why_it_can_fail": "a third file could have moved without anyone noticing; "
                                   "clearing only the two named edits would then certify less "
                                   "than it appears to",
                "declared": list(DECLARED_DRIFT), "drifted": drift["drifted_files"],
                "undeclared": drift["undeclared_drift"]}},
        "f5_no_seed_outside_the_burned_blocks": {
            "passed": all(8_200_001 <= s <= 8_200_060 or 5_300_001 <= s <= 5_300_012
                          for s in seeds_touched),
            "evidence": {
                "why_it_can_fail": "re-evaluating a cell outside the blocks these caches already "
                                   "burned would consume custody this certificate never declared",
                "seeds_touched": sorted(seeds_touched)}},
    }
    falsifiers["all_passed"] = all(v["passed"] for v in falsifiers.values()
                                   if isinstance(v, dict))

    verdict = ("FROZEN_PATH_EQUIVALENT_UNDER_CURRENT_SOURCE" if falsifiers["all_passed"]
               else "RERUN_REQUIRED_PROVENANCE_NOT_RECOVERABLE")

    payload = {
        "schema_version": "frozen_path_equivalence_v1",
        "claim_status": verdict,
        "scope": "PROVENANCE_ONLY_NO_SCIENTIFIC_CLAIM_NO_NEW_SEEDS",
        "endpoint": "cell_level_exact_reproduction",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "module_manifest": module_manifest(MODULES, script=__file__),
        "preregistration": str(args.contract),
        "declared_drift": list(DECLARED_DRIFT),
        "manifest_drift": drift,
        "by_cache": results,
        "mutation_controls": controls,
        "falsifiers": falsifiers,
        "what_this_does_not_certify": (
            "This certifies that the sampled cells of these caches reproduce under the current "
            "tree. It does not certify unsampled cells, nor any artifact whose surface is not one "
            "of these caches, nor that the 2026-08-07 edits are harmless for FUTURE runs that "
            "enable the paths they added."),
    }
    digest = seal_and_write(payload, args.out, contract=args.contract,
                            reference=Path(TARGETS["grid_transfer_confirmation_v2"]["feeds"]))
    print(f"\n  veredicto: {verdict}")
    for k, v in falsifiers.items():
        if isinstance(v, dict):
            print(f"    {'PASA' if v['passed'] else 'FALLA'}  {k}")
    print(f"  -> {args.out} (sello {digest[:16]}…)")
    return 0 if falsifiers["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
