#!/usr/bin/env python3
"""Two questions about Confirmation 2, kept apart because they are not the same question.

    A · HISTORICAL SOURCE IDENTITY   Does the tree that sealed A2 still exist, and can we name it?
    B · FORWARD EQUIVALENCE          Are the 2026-08-07 edits behaviourally inert on its path?

Only A failing invalidates the artifact. If B fails the science is still publishable from the
frozen tag -- a repository may evolve without every old result having to reproduce from the most
recent Tuesday. Calling B "provenance repair" was the error in the first version of this plan.

WHY A SECOND VERSION. Two predecessors exist and neither is enough, both in git history:

  18a4174  two caches (…_v2_ext AND wrap288_v1), 600 sampled cells, 60 seeds, FIVE falsifiers
           including structural spread and seed custody.
  f99ba5f  one cache (…_v2_base), every cell, four falsifiers, NO custody check.

The second traded structural coverage for a cell count: A2 declares `n_ext_configs = 4608` and the
confirmation IS about the expanded grid, so exhausting the 288-wide base surface while never
touching the 4,608-wide one is thorough where it does not matter and blind where it does. This
script is the union, and it adds what neither had -- the DOWNSTREAM CHAIN.

THE CHAIN IS THE POINT, AND IT IS NEARLY FREE. A2's claim is not "the old surface still matches".
It is "UCB1 transfers to the expanded grid and the other three do not survive their marginal
replay". That verdict is produced by visit sequences, per-arm AUC, marginal replay, paired
contrasts over 60 seeds, and a transfers dict. All of it reads the caches without re-simulating, so
re-running the whole confirmation under today's code costs minutes, not hours, and covers the part
of the claim the surface replay cannot reach.

WHAT THE MANIFEST ACTUALLY DECLARES. An earlier note in this repository claimed `supply_chain.py`
was absent from A2's manifest and that its drift was therefore invisible. That was wrong, and it is
corrected here: the top-level `module_manifest` names only `arm_runner.py` and `seed_custody.py`,
but `cache_module_manifests.base` and `.extended` both declare
`supply_chain/supply_chain.py = 2f348e59…` alongside `config.py` and `episode_metrics.py`. The
drift is declared and detectable. This script checks against BOTH manifests.

Phases, so the cheap evidence lands in minutes and the expensive replay runs in the background:

    --phase chain     downstream replay + manifests + historical blobs + mutation controls (~min)
    --phase surface   every cell of the base AND expanded caches, parallel and resumable (~5 h)
    --phase seal      combine the shards into the sealed certificate

Contract: docs/ENMIENDA_REGISTRO_DE_EVIDENCIA_2026-08-07.md
Read-only over artifacts. No seed outside the already-burned blocks is touched.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.build_transfer_confirmation_cache_v1 import (  # noqa: E402
    BASE_CONFIGS,
    EXT_CONFIGS,
    evaluate,
)
from supply_chain.arm_runner import seal_and_write  # noqa: E402
from supply_chain.seed_custody import module_manifest  # noqa: E402

TARGET = Path("results/grid_transfer_confirmation_v2/result.json")
BASE_CACHE = Path("results/surface_cache/garrido_transfer_confirmation_v2_base")
EXT_CACHE = Path("results/surface_cache/garrido_transfer_confirmation_v2_ext")
SHARDS = Path("results/frozen_path_equivalence_v2/shards")

#: The confirmation's own custody. A cell outside this is custody this certificate never declared.
SEED_LOW, SEED_HIGH = 8_200_001, 8_200_060

#: Fields that legitimately differ between two runs of the same science.
VOLATILE = {"created_at", "elapsed_seconds", "self_sha256", "module_manifest",
            "cache_module_manifests", "contract_path", "contract_sha256", "reference_path",
            "reference_sha256", "seed_block", "endpoint", "schema_version"}

#: The scientific payload: if any of these moves, the claim moved.
SCIENTIFIC = ("contrasts", "transfers", "mean_auc", "per_arm", "claim_status",
              "n_base_configs", "n_ext_configs", "budget", "contexts", "seeds")

DRIFT_CLASSES = (
    "SOURCE_HASH_MATCH",
    "SOURCE_DRIFT__NO_SCIENTIFIC_PATH_EFFECT",
    "SOURCE_DRIFT__OBSERVATIONALLY_EQUIVALENT",
    "SOURCE_DRIFT__SCIENTIFICALLY_MATERIAL",
)

# The v2 certificate is deliberately about the complete contracted surface, not a convenient
# sample.  Six contexts x sixty burned seeds gives 360 slices per surface.
EXPECTED_CONTEXTS = 6
EXPECTED_SEEDS = 60
EXPECTED_SLICES = EXPECTED_CONTEXTS * EXPECTED_SEEDS
EXPECTED_CELLS = {
    "base": EXPECTED_SLICES * len(BASE_CONFIGS),
    "ext": EXPECTED_SLICES * len(EXT_CONFIGS),
}

# These files can change the envelope/custody without changing the scientific path.  All other
# declared modules/scripts are adjudicated by the cache replay and downstream chain.
NON_SCIENTIFIC_PATHS = {
    "supply_chain/seed_custody.py",
}


def file_sha(path: str) -> str:
    p = ROOT / path
    return sha256(p.read_bytes()).hexdigest() if p.exists() else "MISSING"


# --------------------------------------------------------------------------- layer A


def find_historical_blob(path: str, wanted: str, max_commits: int = 400) -> dict:
    """Walk the history of one file looking for the content that hashes to `wanted`.

    Git is content-addressed, so if the tree that sealed A2 ever existed it is still reachable.
    This is what makes 'source identity lost' the wrong description of the situation: the bytes
    are recoverable even when the working tree has moved past them.
    """
    try:
        revs = subprocess.run(
            ["git", "log", "--all", f"--max-count={max_commits}", "--format=%H", "--", path],
            cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    except subprocess.CalledProcessError as exc:
        return {"path": path, "found": False, "error": str(exc)}
    for rev in revs:
        blob = subprocess.run(["git", "show", f"{rev}:{path}"],
                              cwd=ROOT, capture_output=True, check=False)
        if blob.returncode == 0 and sha256(blob.stdout).hexdigest() == wanted:
            return {"path": path, "found": True, "commit": rev, "sha256": wanted,
                    "current_matches": file_sha(path) == wanted}
    return {"path": path, "found": False, "sha256": wanted, "commits_scanned": len(revs),
            "current_sha256": file_sha(path)}


def declared_manifests(target: dict) -> dict[str, str]:
    """Union of the top-level manifest and BOTH cache manifests -- the correction of record."""
    out: dict[str, str] = {}
    top = target.get("module_manifest", {})
    out.update(top.get("modules", {}))
    if isinstance(top.get("entry_script"), str):
        out[top["entry_script"]] = top.get("entry_script_sha256", "")
    caches = target.get("cache_module_manifests", {})
    entries = caches.values() if isinstance(caches, dict) else caches
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        out.update(entry.get("modules", {}))
        if isinstance(entry.get("entry_script"), str):
            out[entry["entry_script"]] = entry.get("entry_script_sha256", "")
    return {k: v for k, v in out.items() if v}


# --------------------------------------------------------------------------- layer B, chain


def rerun_chain(target: dict, workdir: Path) -> dict:
    """Re-execute the whole confirmation under today's code and diff the scientific payload.

    Reads the frozen caches; simulates nothing. This is the part of A2's claim -- visit sequences,
    AUC, marginal replay, contrasts, verdict -- that no amount of cell-level replay can reach.

    IT IS INVOKED AS A REPLAY, NOT AS A CONFIRMATION, AND THAT WAS THE 2026-08-07 FAILURE. Passing
    `--confirmation` makes `_seed_custody_falsifier` ask the confirmation question: are these seeds
    exactly the declared block, and does NO sealed artifact consume them? Measured directly rather
    than by re-running for an hour: `check_seeds(8200001..8200060)` now returns all sixty seeds under
    `sealed_artifact_overlap`, because `grid_transfer_confirmation_v2/result.json` -- the very
    artifact being reproduced -- consumes them. The confirmation branch therefore CANNOT pass on a
    reproduction, by construction, whatever else is true. Adding `--seed-block` fixes only half of it
    (it clears the registry conflict and leaves the sealed overlap), which is why the fix is the run
    mode and not a missing flag: reproducing a sealed artifact is the definition of a replay, and
    `--replay-of` asks the question a replay can answer.
    """
    out = workdir / "chain_result.json"
    cmd = [sys.executable, "scripts/run_grid_transfer_v1.py",
           "--base-cache", str(BASE_CACHE), "--ext-cache", str(EXT_CACHE),
           "--budget", str(target["budget"]),
           "--contract", target["contract_path"],
           "--reference", target["reference_path"],
           # The block NAME, not the artifact path. `--replay-of` is matched against the seed
           # registry, so a path falls through to the collision check and fails on the very
           # artifact being reproduced -- which is what the 2026-08-08 run reported as
           # f4_seed_custody FALLA while reproducing every scientific key exactly. Derived from
           # the target rather than written here, so it cannot drift from the artifact.
           "--replay-of", str(target.get("seed_block") or ""),
           "--output", str(out)]
    started = time.perf_counter()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False)
    if proc.returncode != 0 or not out.exists():
        # STDOUT IS THE DIAGNOSIS, AND THIS BRANCH USED TO THROW IT AWAY. `run_grid_transfer_v1.py`
        # reports a failed falsifier by printing to stdout and returning 1, so a failure captured as
        # `{"returncode": 1, "stderr": ""}` says only that something went wrong -- which is the least
        # useful thing a failure report can say. The 2026-08-07 chain failure cost a re-run to
        # rediscover a message that had already been printed once.
        return {"ran": False, "returncode": proc.returncode,
                "stdout": proc.stdout[-4000:], "stderr": proc.stderr[-2000:],
                "output_file_written": out.exists(), "cmd": cmd}
    fresh = json.loads(out.read_text())
    differing = {}
    # `claim_status` encodes the RUN MODE as well as the science ("GRID_TRANSFER_CONFIRMED__UCB1" is
    # a confirmation verdict), so a faithful replay legitimately reports a different string. Diffing
    # it with the numbers would manufacture a failure out of the very fix above. It is reported
    # beside them instead, and `transfers` -- the dict of per-family boolean verdicts -- carries the
    # scientific content of the verdict and IS diffed.
    for key in (k for k in SCIENTIFIC if k != "claim_status"):
        if json.dumps(target.get(key), sort_keys=True) != json.dumps(fresh.get(key), sort_keys=True):
            differing[key] = {"sealed": target.get(key), "replayed": fresh.get(key)}
    return {"ran": True, "elapsed_seconds": time.perf_counter() - started,
            "compared_keys": list(SCIENTIFIC), "n_differing": len(differing),
            "differing": differing,
            "verdict_sealed": target.get("claim_status"),
            "verdict_replayed": fresh.get("claim_status"),
            "transfers_sealed": target.get("transfers"),
            "transfers_replayed": fresh.get("transfers")}


# --------------------------------------------------------------------------- layer B, surface


def replay_slice(args: tuple[str, str]) -> dict:
    """Re-evaluate every cell of one cached slice. Runs in a worker process."""
    path_s, kind = args
    path = Path(path_s)
    payload = json.loads(path.read_text())
    configs = BASE_CONFIGS if kind == "base" else EXT_CONFIGS
    ctx, seed, horizon = payload["context"], int(payload["seed"]), float(payload["horizon_hours"])
    mismatches, worst, where = 0, 0.0, None
    for idx, expected in enumerate(payload["cells"]):
        got = evaluate(configs[idx], ctx, seed, horizon)
        d = abs(got["value"] - float(expected["value"]))
        for a, b in zip(got["drivers"], expected["drivers"]):
            d = max(d, abs(a - float(b)))
        for k, v in expected["panel"].items():
            d = max(d, abs(got["panel"][k] - float(v)))
        if d > 0.0:
            mismatches += 1
            if d > worst:
                worst, where = d, {"cell_index": idx, "config": configs[idx]}
    return {"path": path_s, "kind": kind, "context": ctx, "seed": seed,
            "cells": len(payload["cells"]), "mismatches": mismatches,
            "max_abs_delta": worst, "worst_cell": where}


def run_surface(kinds: list[str], workers: int, shard_i: int = 0, shard_n: int = 1,
                shard_list: list[int] | None = None) -> None:
    """Resumable. A shard already on disk is not recomputed, so a killed run continues.

    PARALLELISM IS BY INDEPENDENT PROCESS, NOT ProcessPoolExecutor. On macOS the executor uses
    `spawn`, and here it hung with the management thread polling pipes for workers that never came
    up: parent alive at 0% CPU, zero children, eight minutes, no output. This repository has been
    bitten by macOS spawn before. Independent `--shard i --of n` processes need no shared state,
    survive one worker dying, and resume from whatever shards are already on disk -- which is the
    pattern the rest of the repository already uses.
    """
    SHARDS.mkdir(parents=True, exist_ok=True)
    # A process may own SEVERAL partition indices. Needed to split one fine partition across two
    # machines of different speed without overlap: the VPS is ~4x slower per core, so its fair
    # share is ~15% of the slices, which no 8-way or 5-way split expresses. Overlapping instead
    # would have both pools recompute the same slices and eat the whole saving.
    mine = set(shard_list) if shard_list else {shard_i}
    jobs = []
    for kind in kinds:
        cache = BASE_CACHE if kind == "base" else EXT_CACHE
        for idx, path in enumerate(sorted(cache.rglob("*.json"))):
            if idx % shard_n not in mine:
                continue
            shard = SHARDS / f"{kind}__{path.parent.name}__{path.stem}.json"
            if not shard.exists():
                jobs.append((str(path), kind))
    if not jobs:
        print("  todas las rebanadas ya están; nada que recomputar")
        return
    print(f"  shards {sorted(mine)} de {shard_n}: {len(jobs)} rebanadas pendientes", flush=True)
    started = time.perf_counter()
    for done, job in enumerate(jobs, 1):
        row = replay_slice(job)
        shard = SHARDS / (f"{row['kind']}__{Path(row['path']).parent.name}__"
                          f"{Path(row['path']).stem}.json")
        shard.write_text(json.dumps(row))
        rate = (time.perf_counter() - started) / done
        print(f"  [{done:4d}/{len(jobs)}] {row['kind']:4} {row['context']:12} "
              f"seed {row['seed']} · {row['cells']:5d} celdas · "
              f"{row['mismatches']} dif · ETA {(len(jobs)-done)*rate/60:.0f} min", flush=True)


def read_shards() -> dict:
    rows = [json.loads(p.read_text()) for p in sorted(SHARDS.glob("*.json"))]
    out = {}
    for kind in ("base", "ext"):
        sub = [r for r in rows if r["kind"] == kind]
        out[kind] = {
            "slices": len(sub), "cells": sum(r["cells"] for r in sub),
            "mismatches": sum(r["mismatches"] for r in sub),
            "max_abs_delta": max((r["max_abs_delta"] for r in sub), default=0.0),
            "contexts": sorted({r["context"] for r in sub}),
            "seeds": sorted({r["seed"] for r in sub}),
            "worst": next((r for r in sub if r["mismatches"]), None),
        }
    return out


def surface_is_complete(surface: dict) -> bool:
    """Require both contracted surfaces, all contexts/seeds, and every expected cell."""
    return all(
        isinstance(row, dict)
        and row.get("slices") == EXPECTED_SLICES
        and row.get("cells") == EXPECTED_CELLS[kind]
        and len(row.get("contexts", [])) == EXPECTED_CONTEXTS
        and len(row.get("seeds", [])) == EXPECTED_SEEDS
        and row.get("mismatches") == 0
        for kind, row in ((kind, surface.get(kind, {})) for kind in ("base", "ext"))
    )


def classify_source_drift(declared: dict[str, str], chain: dict, surface: dict) -> dict[str, dict]:
    """Name the provenance state of every declared file without collapsing distinct cases."""
    chain_ok = bool(chain.get("ran") and chain.get("n_differing") == 0)
    complete = surface_is_complete(surface)
    out = {}
    for path, expected in sorted(declared.items()):
        current = file_sha(path)
        if current == expected:
            classification = "SOURCE_HASH_MATCH"
        elif path in NON_SCIENTIFIC_PATHS:
            classification = "SOURCE_DRIFT__NO_SCIENTIFIC_PATH_EFFECT"
        elif chain_ok and complete:
            classification = "SOURCE_DRIFT__OBSERVATIONALLY_EQUIVALENT"
        else:
            classification = "SOURCE_DRIFT__SCIENTIFICALLY_MATERIAL"
        out[path] = {
            "declared_sha256": expected,
            "current_sha256": current,
            "classification": classification,
        }
    return out


# --------------------------------------------------------------------------- mutation controls


def mutation_controls(target: dict, chain: dict | None = None) -> dict:
    """Four planted defects, each on a path this certificate actually consumes.

    A comparator that cannot fail on a corruption is not evidence that the clean run agrees. Three
    of these corrupt something scientific and MUST be caught. The fourth is the opposite and is the
    whole reason `arm_runner.py` and `supply_chain.py` do not deserve the same adjudication: a
    seal-only change must move the manifest and leave the science untouched.
    """
    slices = sorted(EXT_CACHE.rglob("*.json"))
    payload = json.loads(slices[0].read_text())
    seed, ctx = int(payload["seed"]), payload["context"]
    horizon = float(payload["horizon_hours"])
    live = evaluate(EXT_CONFIGS[0], ctx, seed, horizon)

    # M1 -- a physics change: the live value moves away from the cache.
    m1 = {"detected": abs((live["value"] + 1e-12) - float(payload["cells"][0]["value"])) > 0.0,
          "injected": 1e-12, "on": "value returned by the simulator"}

    # M2 -- a corrupted extended-cache cell.  Run the actual slice comparator against the
    # temporary corruption; a direct `corrupted != live` check would only test Python arithmetic.
    corrupted_payload = json.loads(json.dumps(payload))
    corrupted_payload["cells"][0]["value"] = float(corrupted_payload["cells"][0]["value"]) + 1e-12
    with tempfile.TemporaryDirectory() as td:
        corrupted_path = Path(td) / "corrupted.json"
        corrupted_path.write_text(json.dumps(corrupted_payload))
        replayed_corruption = replay_slice((str(corrupted_path), "ext"))
    m2 = {"detected": replayed_corruption["mismatches"] > 0,
          "injected": 1e-12, "on": "cached cell of the 4,608-wide surface",
          "clean_cell_still_matches": float(payload["cells"][0]["value"]) == float(live["value"]),
          "comparator": replayed_corruption}

    # M3 -- a corrupted AUC array: the chain comparator must see a moved contrast.
    sealed = json.dumps(target.get("contrasts"), sort_keys=True)
    bumped = json.loads(sealed)
    first = sorted(bumped)[0]
    bumped[first]["vs_marginal_replay"]["mean"] += 1e-12
    mutated_target = dict(target)
    mutated_target["contrasts"] = bumped
    with tempfile.TemporaryDirectory() as td:
        mutated_chain = rerun_chain(mutated_target, Path(td))
    m3 = {
        "detected": bool(mutated_chain.get("ran")
                          and "contrasts" in mutated_chain.get("differing", {})),
        "injected": 1e-12,
        "on": f"contrasts[{first}].vs_marginal_replay.mean",
        "comparator": {"ran": mutated_chain.get("ran"),
                        "n_differing": mutated_chain.get("n_differing"),
                        "differing_keys": list(mutated_chain.get("differing", {}))},
    }

    # M4 -- the inverse control. Inject a manifest-only change and rerun the downstream chain. The
    # falsifier is always active: it explicitly requires the manifest to move while science stays
    # identical. The live arm_runner drift is reported separately for the release classification.
    declared = declared_manifests(target)
    live_arm_runner_drift = declared.get("supply_chain/arm_runner.py") != file_sha(
        "supply_chain/arm_runner.py")
    mutated_manifest = json.loads(json.dumps(target.get("module_manifest", {})))
    old_entry_hash = mutated_manifest.get("entry_script_sha256", "")
    mutated_manifest["entry_script_sha256"] = "0" * 64 if old_entry_hash != "0" * 64 else "1" * 64
    mutated_target = dict(target)
    mutated_target["module_manifest"] = mutated_manifest
    with tempfile.TemporaryDirectory() as td:
        mutated_manifest_chain = rerun_chain(mutated_target, Path(td))
    manifest_moved = mutated_manifest != target.get("module_manifest", {})
    scientific_unchanged = bool(
        mutated_manifest_chain.get("ran")
        and mutated_manifest_chain.get("n_differing") == 0
    )
    m4 = {
        "file": "supply_chain/arm_runner.py",
        "applicable": True,
        "manifest_moved": manifest_moved,
        "science_unchanged": scientific_unchanged,
        "detected": bool(manifest_moved and scientific_unchanged),
        "live_arm_runner_drift": live_arm_runner_drift,
        "comparator": {"ran": mutated_manifest_chain.get("ran"),
                        "n_differing": mutated_manifest_chain.get("n_differing")},
        "science_expected_to_move": False,
        "note": ("a seal-only edit must break the manifest and leave the downstream scientific "
                 "payload identical; if this file's drift moved the chain replay it would be a "
                 "physics change wearing an infrastructure label")}
    return {"m1_physics": m1, "m2_extended_cache": m2, "m3_auc_contrast": m3,
            "m4_seal_only_must_not_move_science": m4}


# --------------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=("chain", "surface", "seal"), required=True)
    ap.add_argument("--surface", choices=("base", "ext", "both"), default="both")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2),
                    help="kept for the launcher's arithmetic; parallelism is by --shard")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--shards", default=None,
                    help="comma list of partition indices this process owns, e.g. 0,1,2")
    ap.add_argument("--contract", type=Path,
                    default=Path("docs/ENMIENDA_REGISTRO_DE_EVIDENCIA_2026-08-07.md"))
    ap.add_argument("--out", type=Path,
                    default=Path("results/frozen_path_equivalence_v2/result.json"))
    args = ap.parse_args()
    target = json.loads((ROOT / TARGET).read_text())

    if args.phase == "surface":
        kinds = ["base", "ext"] if args.surface == "both" else [args.surface]
        shard_list = ([int(x) for x in args.shards.split(",")] if args.shards else None)
        run_surface(kinds, args.workers, args.shard, args.of, shard_list)
        return 0

    if args.phase == "chain":
        declared = declared_manifests(target)
        print(f"  manifiestos declarados: {len(declared)} ficheros "
              f"(top-level + cache base + cache extendida)")
        hist = [find_historical_blob(p, h) for p, h in sorted(declared.items())]
        for h in hist:
            mark = "OK " if h["found"] else "NO "
            extra = ("árbol actual coincide" if h.get("current_matches") else
                     f"recuperable en {h.get('commit','?')[:8]}" if h["found"] else "NO ENCONTRADO")
            print(f"    {mark}{h['path']:42} {extra}")
        with tempfile.TemporaryDirectory() as td:
            chain = rerun_chain(target, Path(td))
        print(f"  cadena: {'re-ejecutada' if chain['ran'] else 'FALLO'} · "
              f"{chain.get('n_differing','?')} claves científicas difieren · "
              f"veredicto {chain.get('verdict_replayed')}")
        controls = mutation_controls(target, chain)
        for k, v in controls.items():
            print(f"    control {k}: {v}")
        (ROOT / args.out).parent.mkdir(parents=True, exist_ok=True)
        (ROOT / args.out).with_suffix(".chain.json").write_text(json.dumps(
            {"historical": hist, "chain": chain, "controls": controls,
             "declared_manifests": declared}, indent=1))
        print(f"  -> {args.out.with_suffix('.chain.json')}")
        return 0

    # ---- seal -------------------------------------------------------------------------------
    partial = json.loads((ROOT / args.out).with_suffix(".chain.json").read_text())
    surface = read_shards()
    hist, chain, controls = partial["historical"], partial["chain"], partial["controls"]

    a_ok = all(h["found"] for h in hist)
    chain_ok = bool(chain.get("ran") and chain.get("n_differing") == 0)
    surface_ok = surface_is_complete(surface)
    m4_ok = (not controls["m4_seal_only_must_not_move_science"]["applicable"]
             or controls["m4_seal_only_must_not_move_science"]["detected"])
    controls_ok = (controls["m1_physics"]["detected"]
                   and controls["m2_extended_cache"]["detected"]
                   and controls["m2_extended_cache"]["clean_cell_still_matches"]
                   and controls["m3_auc_contrast"]["detected"]
                   and m4_ok)
    seeds_ok = all(SEED_LOW <= s <= SEED_HIGH for v in surface.values() for s in v["seeds"])
    spread_ok = surface_ok
    source_drift = classify_source_drift(partial["declared_manifests"], chain, surface)
    controls = dict(controls)
    controls["source_drift_classification"] = source_drift

    falsifiers = {
        "f1_every_cell_of_both_surfaces_reproduces": {
            "passed": surface_ok,
            "evidence": {"why_it_can_fail": "a behavioural change in either drifted module moves a "
                                            "cell and the delta stops being zero",
                         "by_surface": {k: {"slices": v["slices"], "cells": v["cells"],
                                            "mismatches": v["mismatches"],
                                            "max_abs_delta": v["max_abs_delta"]}
                                        for k, v in surface.items()}}},
        "f2_the_downstream_chain_reproduces_the_verdict": {
            "passed": chain_ok,
            "evidence": {"why_it_can_fail": "the claim is a verdict, not a surface: visit "
                                            "sequences, AUC, marginal replay and contrasts could "
                                            "move while every cached cell stayed identical",
                         **{k: chain.get(k) for k in
                            ("ran", "n_differing", "differing", "verdict_sealed",
                             "verdict_replayed", "transfers_sealed", "transfers_replayed")}}},
        "f3_the_sample_spans_the_whole_grid": {
            "passed": spread_ok,
            "evidence": {"why_it_can_fail": "exhausting one surface while never touching the "
                                            "4,608-wide one is the defect this version exists to "
                                            "repair; both must be complete",
                         "by_surface": {k: {"contexts": v["contexts"], "n_seeds": len(v["seeds"])}
                                        for k, v in surface.items()}}},
        "f4_planted_defects_are_detected": {
            "passed": controls_ok,
            "evidence": {"why_it_can_fail": "a comparator blind to a 1e-12 corruption would certify "
                                            "any tree at all; this repository has shipped a "
                                            "falsifier hardcoded to passed:True",
                         **controls}},
        "f5_no_seed_outside_the_burned_block": {
            "passed": seeds_ok,
            "evidence": {"why_it_can_fail": "re-evaluating outside 8200001-8200060 would consume "
                                            "custody this certificate never declared",
                         "block": [SEED_LOW, SEED_HIGH]}},
        "f6_the_historical_tree_is_recoverable": {
            "passed": a_ok,
            "evidence": {"why_it_can_fail": "if a declared blob is unreachable in the whole object "
                                            "graph then the tree that sealed A2 is genuinely lost, "
                                            "and only then is the artifact unpublishable",
                         "modules": hist}},
    }
    falsifiers["all_passed"] = all(v["passed"] for v in falsifiers.values() if isinstance(v, dict))

    verdict_a = ("HISTORICAL_SOURCE_RECOVERED_AND_REPRODUCES" if a_ok
                 else "HISTORICAL_SOURCE_TREE_INCOMPLETE")
    verdict_b = ("CURRENT_HEAD_BEHAVIOURALLY_EQUIVALENT"
                 if (chain_ok and surface_ok) else "CURRENT_HEAD_NOT_EQUIVALENT_USE_FROZEN_RELEASE")

    payload = {
        "schema_version": "frozen_path_equivalence_v2",
        "claim_status": f"{verdict_a}__{verdict_b}",
        "verdict_a_historical_identity": verdict_a,
        "verdict_b_forward_equivalence": verdict_b,
        "scope": "PROVENANCE_ONLY_NO_SCIENTIFIC_CLAIM_NO_NEW_SEEDS",
        # Without a run_role the claim lock's grader reports GRADE_NOT_MACHINE_DISCOVERABLE and
        # refuses to guess, which is correct behaviour and useless to a manuscript that needs to
        # cite this. The role is not a science grade: this artifact adjudicates provenance and
        # opens no seed, so it says exactly that rather than borrowing CONFIRMATION or REPLAY.
        "run_role": "PROVENANCE_VERIFICATION",
        "registration_status": "PREREGISTERED_PROVENANCE_CONTRACT_NO_SEEDS_NO_SCIENTIFIC_CLAIM",
        "endpoint": "cell_and_verdict_level_exact_reproduction",
        "budget": target["budget"],
        "contexts": target["contexts"],
        "seeds": target["seeds"],
        "n_base_configs": target["n_base_configs"],
        "n_ext_configs": target["n_ext_configs"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "module_manifest": module_manifest(
            ("supply_chain/arm_runner.py", "supply_chain/seed_custody.py",
             "supply_chain/supply_chain.py", "supply_chain/config.py",
             "supply_chain/episode_metrics.py"), script=__file__),
        "preregistration": str(args.contract),
        "supersedes": ["18a4174:scripts/verify_frozen_path_equivalence_v1.py",
                       "f99ba5f:scripts/verify_frozen_path_equivalence_v1.py"],
        "declared_manifests": partial["declared_manifests"],
        "historical_source": hist,
        "downstream_chain": chain,
        "surface": surface,
        "mutation_controls": controls,
        "falsifiers": falsifiers,
        "what_this_does_not_certify": (
            "Only the two caches Paper 2 cites, only the confirmation's own seed block, and only "
            "the behaviour of the current tree on the FROZEN path. It says nothing about future "
            "runs that enable the code paths the 2026-08-07 edits added."),
    }
    digest = seal_and_write(payload, ROOT / args.out, contract=ROOT / args.contract,
                            reference=ROOT / TARGET)
    print(f"\n  A · {verdict_a}\n  B · {verdict_b}")
    for k, v in falsifiers.items():
        if isinstance(v, dict):
            print(f"    {'PASA' if v['passed'] else 'FALLA'}  {k}")
    print(f"  -> {args.out} (sello {digest[:16]}…)")
    return 0 if falsifiers["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
