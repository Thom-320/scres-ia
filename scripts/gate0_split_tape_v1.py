#!/usr/bin/env python3
"""Gate-0 split-tape (preregistro gate0_split_tape_v1, FIRMADO 2026-08-25).

Implements the signed preregistration
``/home/ubuntu/scres-sources/preregistros/GATE0_SPLIT_TAPE_PREREGISTRO_V1.md``
(SHA-256 b2a6058ccf2062f36c3dbbceadf3a5f34ba503df2a2203c4a3e6135415428384):

* 3 confirmed ReT cells (rho75_share90, rho90_share75, rho90_share90).
* 64 virgin selection tapes A + 64 disjoint frozen evaluation tapes B per cell,
  seed block 7550001-7550512 (collision scan = 0, ACTA 2026-08-25).
* Phase A: the complete open-loop frontier (4^8 = 65,536 calendars) is
  evaluated on every tape A through the audited full-DES event transducer
  (``extract_full_des_skeleton`` + ``simulate_full_des_frontier``, bit-exact
  against direct SimPy replay).  Winner ``k*(A) = argmax_k mean_A(X_{A,k})``
  is frozen with its SHA-256 BEFORE any tape B byte exists.
* Phase B: only X_{B,k*(A)} plus the ten classical state-rich comparators on
  tapes B (same CRN tape for every arm).
* Falsifiers F1-F4; placebo F3 draws one uniform calendar and must show G ~= 0
  on tapes A (no tape-B contact).
* Estimands G_PI_naive (diagnostic, instantiated on tapes A where selection
  and evaluation share tapes -- the original unsplit design), G_PI_split
  (gate estimand), Delta_bias = naive - split; bootstrap unit = tape,
  stratified per cell, 10,000 percentile resamples; the classical comparator
  is re-selected inside every resample.  Decision rules of preregistration
  section 2.6 with SESOI = 0.01.

Cost model: one skeleton ~40 ms plus one vectorized 65,536-calendar frontier
~6.6 s per tape, i.e. ~21 min serial per arm of 192 tapes (~0.7 CPU-h) --
inside the ~6 CPU-h budget of Decisión 3.

Usage:
    python scripts/gate0_split_tape_v1.py freeze-map
    python scripts/gate0_split_tape_v1.py run-phase-a [--workers N]
    python scripts/gate0_split_tape_v1.py freeze-winner
    python scripts/gate0_split_tape_v1.py run-phase-b [--workers N]
    python scripts/gate0_split_tape_v1.py analyze [--workers N]
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from supply_chain.program_o_full_des import (  # noqa: E402
    run_program_o_full_des_episode,
)
from supply_chain.program_o_full_des_transducer import (  # noqa: E402
    MATRIX_KEYS,
    direct_full_des_vector,
    extract_full_des_skeleton,
    full_action_calendars,
    simulate_full_des_frontier,
)
from supply_chain.program_o_state_rich import (  # noqa: E402
    finite_state_rich_configurations,
    state_rich_calendar,
)

PREREGISTRATION_SHA256 = (
    "b2a6058ccf2062f36c3dbbceadf3a5f34ba503df2a2203c4a3e6135415428384"
)
PREREGISTRATION_ID = "gate0_split_tape_v1"
PREREGISTRATION_PATH = (
    "/home/ubuntu/scres-sources/preregistros/GATE0_SPLIT_TAPE_PREREGISTRO_V1.md"
)
ACTA_PATH = "/home/ubuntu/scres-sources/preregistros/ACTA_FIRMA_2026-08-25.md"

# Cell order fixed once here; every artifact derives from this tuple.
CELLS: tuple[tuple[str, float, float], ...] = (
    ("rho75_share90", 0.75, 0.90),
    ("rho90_share75", 0.90, 0.75),
    ("rho90_share90", 0.90, 0.90),
)
N_TAPES_PER_ARM = 64
SEED_BLOCK_START = 7_550_001
SEED_BLOCK_END = 7_550_512
FORBIDDEN_RANGES: tuple[tuple[int, int], ...] = (
    (7_480_101, 7_480_148),        # SEALED_FOREVER (Program O-R confirmation)
    (7_490_001, 7_490_256),        # CONSUMED by Program Q 2026-07-18
    (7_510_001, 7_510_012),        # S1 Morris RESERVED_UNOPENED
    (7_510_101, 7_510_148),        # S2 observable RESERVED_UNOPENED
    (7_510_201, 7_510_248),        # S3 full DES RESERVED_UNOPENED
    (7_510_301, 7_510_348),        # S4 calibration RESERVED_UNOPENED
    (75_110_001, 75_135_000),      # S4 training RESERVED_UNOPENED
    (7_520_001, 7_520_256),        # S4 confirmation RESERVED_UNOPENED
    (7_530_001, 7_539_999),        # Paper3 SEALED_UNAUTHORIZED
    (94_910_001, 999_999_999),     # sandbox development
    (95_010_001, 999_999_999),     # blind qualification sealed outside Q
)

PHYSICS_MODE = "fixed_clock_physical_v1"   # Program O-R / ret_only_learner physics
ENDPOINT = "ret_excel_request_snapshot_v2"
DETERMINISTIC_REPLAY_FRACTION = 0.20       # preregistration F4 demands >= 10%
FRONTIER_SIZE = 4 ** 8
BOOTSTRAP_RESAMPLES = 10_000
SESOI = 0.01
REPLAY_TOLERANCE = 1e-12
# The classical family keeps its fixed belief model on every cell (Program Q/O-R
# convention: scripts/power_program_q_replication.py:149-150).
CLASSICAL_FIXED_MODEL = {"regime_persistence": 0.75, "dominant_product_share": 0.90}

OUTPUT_ROOT = ROOT / "results/gate0_split_tape_v1"
MAP_PATH = OUTPUT_ROOT / "seed_assignment_map.json"
PHASE_A_DIR = OUTPUT_ROOT / "phase_a_frontiers"
PHASE_A_CLASSICAL_DIR = OUTPUT_ROOT / "phase_a_classical_indices"
PHASE_A_SUMMARY = OUTPUT_ROOT / "phase_a_summary.json"
WINNER_PATH = OUTPUT_ROOT / "winner_freeze.json"
PHASE_B_DIR = OUTPUT_ROOT / "phase_b_evaluations"
PHASE_B_SUMMARY = OUTPUT_ROOT / "phase_b_summary.json"
F3_PATH = OUTPUT_ROOT / "falsifier_F3_placebo.json"
F4_PATH = OUTPUT_ROOT / "falsifier_F4_replays.json"
ANALYSIS_PATH = OUTPUT_ROOT / "verdict.json"

CLASSICAL_CONFIGS = finite_state_rich_configurations()
CLASSICAL_CONFIG_IDS = tuple(
    f"{config.policy_id}__{config.parameter}" for config in CLASSICAL_CONFIGS
)


# --------------------------------------------------------------------------
# small utilities
# --------------------------------------------------------------------------

def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_json(value: Any) -> str:
    return sha256_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    )


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def artifact_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "recorded_at_utc": now_utc(),
    }


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:  # pragma: no cover - git unavailable
        return "UNKNOWN"


def encode_calendar(calendar) -> int:
    index = 0
    for action in calendar:
        index = index * 4 + int(action)
    return index


def scheduler() -> dict[str, list[str]]:
    """Primary within-week scheduler of the frozen Program O contract."""
    parent = json.loads(
        (ROOT / "contracts/program_o_full_des_hpi_translation_v1.json").read_text()
    )
    key = parent["action"]["primary_scheduler"]
    return parent["action"]["within_week_schedulers"][key]


def assert_seed_hygiene(seed_map: dict[str, Any]) -> None:
    """Every used seed must sit in the assigned block and outside sealed ranges."""
    assigned = range(
        int(seed_map["seed_block"][0]), int(seed_map["seed_block"][1]) + 1
    )
    used: list[int] = []
    for arm in ("tapes_a", "tapes_b", "instrument"):
        for key, entry in seed_map[arm].items():
            if not isinstance(entry, list):
                continue  # metadata fields (notes, counts) are not seeds
            used.extend(int(value) for value in entry)
    for seed in used:
        if seed not in assigned:
            raise AssertionError(f"seed {seed} outside assigned block")
        for low, high in FORBIDDEN_RANGES:
            if low <= seed <= high:
                raise AssertionError(
                    f"seed {seed} collides with forbidden range [{low}, {high}]"
                )
    if len(set(used)) != len(used):
        raise AssertionError("duplicate seed in assignment map")


# --------------------------------------------------------------------------
# stage 1 - frozen seed assignment map (written before anything runs)
# --------------------------------------------------------------------------

def build_seed_assignment_map() -> dict[str, Any]:
    """Deterministic seed->cell->arm map; frozen as JSON before execution."""
    block = list(range(SEED_BLOCK_START, SEED_BLOCK_END + 1))
    if len(block) != 512:
        raise AssertionError("seed block must contain exactly 512 seeds")
    tapes_a: dict[str, list[int]] = {}
    tapes_b: dict[str, list[int]] = {}
    cursor = 0
    for cell_id, _rho, _share in CELLS:
        tapes_a[cell_id] = block[cursor : cursor + N_TAPES_PER_ARM]
        cursor += N_TAPES_PER_ARM
    for cell_id, _rho, _share in CELLS:
        tapes_b[cell_id] = block[cursor : cursor + N_TAPES_PER_ARM]
        cursor += N_TAPES_PER_ARM
    instrument_pool = block[cursor:]
    # Deterministic pseudo-shuffle (hash order): which pool seeds serve as
    # replays/placebos is fixed before any run and independent of outcomes.
    ordered = sorted(
        instrument_pool, key=lambda value: int(digest_json(["instrument", value])[:12], 16)
    )
    minimum_replays = int(
        np.ceil(DETERMINISTIC_REPLAY_FRACTION * (2 * N_TAPES_PER_ARM * len(CELLS)))
    )  # >= 10% of all 384 tapes
    replay_seeds = []
    for seed in ordered:
        if len(replay_seeds) >= minimum_replays:
            break
        if any(seed in tapes_a[c] or seed in tapes_b[c] for c in tapes_a):
            continue
        replay_seeds.append(seed)
    mapping: dict[str, Any] = {
        "schema_version": "gate0_split_tape_seed_assignment_v1",
        "preregistration_id": PREREGISTRATION_ID,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "preregistration_path": PREREGISTRATION_PATH,
        "acta_path": ACTA_PATH,
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "physics": {
            "mode": PHYSICS_MODE,
            "endpoint": ENDPOINT,
            "decision_weeks": 8,
            "actions": [0, 1, 2, 3],
            "scheduler": scheduler(),
            "scheduler_source": (
                "contracts/program_o_full_des_hpi_translation_v1.json "
                "#action.primary_scheduler=centered_minority_v1"
            ),
        },
        "cells": [
            {"cell_id": cell_id, "regime_persistence": rho, "dominant_share": share}
            for cell_id, rho, share in CELLS
        ],
        "cell_order_note": (
            "order follows CONFIRMED_RET_CELLS "
            "(supply_chain/program_o_ret_env.py:31)"
        ),
        "seed_block": [SEED_BLOCK_START, SEED_BLOCK_END],
        "forbidden_ranges_documented_in": PREREGISTRATION_PATH + " section 4",
        "tapes_a": tapes_a,
        "tapes_b": tapes_b,
        "instrument": {
            "minimum_f4_replays_required": minimum_replays,
            "replay_f4_seeds": replay_seeds,
            "note": "F4 replays re-run their own tape seeds through direct SimPy",
        },
        "deterministic_replay_fraction": DETERMINISTIC_REPLAY_FRACTION,
    }
    assert_seed_hygiene(mapping)
    return mapping


def cmd_freeze_map() -> int:
    fresh = build_seed_assignment_map()
    if MAP_PATH.exists():
        existing = load_json(MAP_PATH)
        strip = lambda m: {k: v for k, v in m.items() if k != "created_at_utc"}
        if digest_json(strip(existing)) != digest_json(strip(fresh)):
            raise RuntimeError("assignment map exists and differs; refusing to rewrite")
        print(json.dumps({"status": "already_frozen", "path": str(MAP_PATH)}, indent=2))
        return 0
    write_json_atomic(MAP_PATH, fresh)
    print(
        json.dumps(
            {"status": "frozen", "artifact": artifact_record(MAP_PATH)}, indent=2
        )
    )
    return 0


def must_load_map() -> dict[str, Any]:
    if not MAP_PATH.exists():
        raise FileNotFoundError(
            f"{MAP_PATH} missing: run `freeze-map` first; nothing may run before "
            "the seed->cell->arm map is frozen"
        )
    return load_json(MAP_PATH)


# --------------------------------------------------------------------------
# shared per-tape workhorses (executed inside worker processes)
# --------------------------------------------------------------------------

def skeleton_for(cell: dict[str, Any], seed: int):
    return extract_full_des_skeleton(
        seed=int(seed),
        scheduler=scheduler(),
        regime_persistence=float(cell["regime_persistence"]),
        dominant_share=float(cell["dominant_share"]),
        downstream_freight_physics_mode=PHYSICS_MODE,
    )[0]


def classical_indices_for(cell: dict[str, Any], seed: int) -> dict[str, int]:
    sched = scheduler()
    skeleton = skeleton_for(cell, seed)
    indices: dict[str, int] = {}
    for config_id, config in zip(CLASSICAL_CONFIG_IDS, CLASSICAL_CONFIGS):
        calendar, _decisions = state_rich_calendar(
            skeleton=skeleton.as_dict(),
            scheduler=sched,
            config=config,
            regime_persistence=float(CLASSICAL_FIXED_MODEL["regime_persistence"]),
            dominant_share=float(CLASSICAL_FIXED_MODEL["dominant_product_share"]),
        )
        indices[config_id] = encode_calendar(calendar)
    return indices


# --------------------------------------------------------------------------
# phase A - full open-loop frontier on tapes A (transducer)
# --------------------------------------------------------------------------

def _phase_a_worker(task: tuple[str, float, float, int]) -> dict[str, Any]:
    cell_id, regime_persistence, dominant_share, seed = task
    cell = {
        "cell_id": cell_id,
        "regime_persistence": regime_persistence,
        "dominant_share": dominant_share,
    }
    skeleton = skeleton_for(cell, seed)
    panel = simulate_full_des_frontier(skeleton=skeleton, scheduler=scheduler())
    destination = PHASE_A_DIR / cell_id / f"tape_{seed}.npz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **panel)
    os.replace(temporary, destination)
    return {
        "cell_id": cell_id,
        "tape_seed": int(seed),
        "shard": str(destination.relative_to(ROOT)),
        "sha256": sha256_file(destination),
        "bytes": destination.stat().st_size,
        "skeleton_sha256": skeleton.skeleton_sha256,
        "tape_sha256": skeleton.tape_sha256,
        "ret_visible_mean": float(panel["ret_visible"].mean()),
        "mass_residual_max_abs": float(np.abs(panel["mass_residual"]).max()),
    }


def cmd_run_phase_a(workers: int) -> int:
    seed_map = must_load_map()
    tasks: list[tuple[str, float, float, int]] = []
    required = 0
    for cell in seed_map["cells"]:
        for seed in seed_map["tapes_a"][cell["cell_id"]]:
            required += 1
            shard = PHASE_A_DIR / cell["cell_id"] / f"tape_{seed}.npz"
            if not shard.exists():
                tasks.append(
                    (
                        cell["cell_id"],
                        cell["regime_persistence"],
                        cell["dominant_share"],
                        int(seed),
                    )
                )
    print(
        f"[phase-a] {required} tape-A frontiers required; "
        f"{len(tasks)} still to compute ({workers} workers)",
        flush=True,
    )
    records: list[dict[str, Any]] = []
    if tasks:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_phase_a_worker, task) for task in tasks]
            for done, future in enumerate(as_completed(futures), 1):
                records.append(future.result())
                if done % 16 == 0 or done == len(futures):
                    print(f"[phase-a] {done}/{len(futures)} shards written", flush=True)
    manifest_file = PHASE_A_DIR / "_manifest.jsonl"
    with manifest_file.open("a") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    write_phase_a_summary(seed_map)
    write_phase_a_classical_indices(seed_map, workers)
    return 0


def load_phase_a_panel(cell_id: str, seed: int) -> dict[str, np.ndarray]:
    path = PHASE_A_DIR / cell_id / f"tape_{seed}.npz"
    with np.load(path, allow_pickle=False) as shard:
        if tuple(shard.files) != MATRIX_KEYS:
            raise AssertionError(f"matrix schema drift in {path}")
        return {key: np.asarray(shard[key]) for key in MATRIX_KEYS}


def write_phase_a_summary(seed_map: dict[str, Any]) -> None:
    """Aggregate Phase A; verify shard integrity against the append manifest."""
    manifest_rows: list[dict[str, Any]] = []
    manifest_file = PHASE_A_DIR / "_manifest.jsonl"
    if manifest_file.exists():
        for line in manifest_file.read_text().splitlines():
            if line.strip():
                manifest_rows.append(json.loads(line))
    by_key = {(row["cell_id"], int(row["tape_seed"])): row for row in manifest_rows}
    problems: list[str] = []
    summary: dict[str, Any] = {
        "schema_version": "gate0_phase_a_summary_v1",
        "preregistration_id": PREREGISTRATION_ID,
        "generated_at_utc": now_utc(),
        "cells": {},
    }
    for cell in seed_map["cells"]:
        cell_id = cell["cell_id"]
        seeds = [int(s) for s in seed_map["tapes_a"][cell["cell_id"]]]
        ret_rows: list[np.ndarray] = []
        mass_max = 0.0
        per_tape_argmax: list[int] = []
        for seed in seeds:
            row = by_key.get((cell_id, seed))
            if row is None:
                problems.append(f"no manifest row for {cell_id}/tape_{seed}")
            elif sha256_file(ROOT / row["shard"]) != row["sha256"]:
                problems.append(f"sha mismatch {row['shard']}")
            panel = load_phase_a_panel(cell_id, seed)
            ret_rows.append(panel["ret_visible"])
            mass_max = max(mass_max, float(np.abs(panel["mass_residual"]).max()))
            per_tape_argmax.append(int(np.argmax(panel["ret_visible"])))
        xa = np.stack(ret_rows)
        mean_per_calendar = xa.mean(axis=0)
        winner_index = int(np.argmax(mean_per_calendar))
        ties = int(
            np.sum(
                np.abs(mean_per_calendar - mean_per_calendar[winner_index]) <= 1e-15
            )
        )
        summary["cells"][cell_id] = {
            "n_tapes": len(seeds),
            "tape_seeds": seeds,
            "mean_ret_over_frontier": float(mean_per_calendar.mean()),
            "max_mean_ret": float(mean_per_calendar[winner_index]),
            "argmax_k_star": winner_index,
            "k_star_calendar": full_action_calendars()[winner_index]
            .astype(int)
            .tolist(),
            "argmax_tie_count": ties,
            "per_tape_argmax_quartiles": [
                float(np.quantile(per_tape_argmax, q)) for q in (0.25, 0.5, 0.75)
            ],
            "mass_residual_max_abs": mass_max,
        }
    summary["integrity_problems"] = problems
    summary["assignment_map"] = artifact_record(MAP_PATH)
    write_json_atomic(PHASE_A_SUMMARY, summary)
    print(f"[phase-a] summary written: {PHASE_A_SUMMARY}")


def write_phase_a_classical_indices(seed_map: dict[str, Any], workers: int) -> None:
    """Classical-controller calendar indices on tapes A (subset of the frontier).

    Needed for G_PI_naive (naive design = selection and evaluation on the same
    tapes) and for the F3 placebo.  These rows are read straight out of the
    already-computed frontiers: no additional physics is involved.
    """
    tasks: list[tuple[str, int]] = []
    for cell in seed_map["cells"]:
        for seed in seed_map["tapes_a"][cell["cell_id"]]:
            path = PHASE_A_CLASSICAL_DIR / cell["cell_id"] / f"tape_{seed}.json"
            if not path.exists():
                tasks.append((cell["cell_id"], int(seed)))
    if tasks:
        cell_lookup = {
            c["cell_id"]: c
            for c in seed_map["cells"]
        }
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(classical_indices_for, cell_lookup[cell_id], seed): (
                    cell_id,
                    seed,
                )
                for cell_id, seed in tasks
            }
            for done, future in enumerate(as_completed(futures), 1):
                cell_id, seed = futures[future]
                write_json_atomic(
                    PHASE_A_CLASSICAL_DIR / cell_id / f"tape_{seed}.json",
                    future.result(),
                )
                if done % 32 == 0 or done == len(futures):
                    print(f"[phase-a] classical indices {done}/{len(futures)}", flush=True)


def load_phase_a_classical_row(
    cell_id: str, seed: int, panel: dict[str, np.ndarray]
) -> dict[str, float]:
    indices = load_json(
        PHASE_A_CLASSICAL_DIR / cell_id / f"tape_{seed}.json"
    )
    return {
        config_id: float(panel["ret_visible"][index])
        for config_id, index in indices.items()
    }


# --------------------------------------------------------------------------
# stage 3 - freeze k*(A) before touching tape B
# --------------------------------------------------------------------------

def cmd_freeze_winner() -> int:
    seed_map = must_load_map()
    if not PHASE_A_SUMMARY.exists():
        raise FileNotFoundError("run `run-phase-a` first")
    summary = load_json(PHASE_A_SUMMARY)
    if summary.get("integrity_problems"):
        raise RuntimeError("phase A integrity problems; refusing to freeze")
    winners: dict[str, Any] = {}
    for cell in seed_map["cells"]:
        cell_id = cell["cell_id"]
        cell_summary = summary["cells"][cell_id]
        winners[cell_id] = {
            "k_star": int(cell_summary["argmax_k_star"]),
            "calendar": cell_summary["k_star_calendar"],
            "mean_ret_on_A": float(cell_summary["max_mean_ret"]),
            "tie_count_on_A": int(cell_summary["argmax_tie_count"]),
            "n_tapes_A": int(cell_summary["n_tapes"]),
        }
    artifact: dict[str, Any] = {
        "schema_version": "gate0_winner_freeze_v1",
        "preregistration_id": PREREGISTRATION_ID,
        "frozen_at_utc": now_utc(),
        "selection_rule": (
            "k*(A) = argmax_k mean_A(X_{A,k}); ties broken by lowest base-4 index"
        ),
        "selection_sample_declaration": (
            "tapes A only; no tape B byte existed when this file was frozen"
        ),
        "phase_a_summary_artifact": artifact_record(PHASE_A_SUMMARY),
        "classical_config_ids": list(CLASSICAL_CONFIG_IDS),
        "winners": winners,
    }
    write_json_atomic(WINNER_PATH, artifact)
    digest = sha256_file(WINNER_PATH)
    (OUTPUT_ROOT / "WINNER_FREEZE.sha256").write_text(f"{digest}  winner_freeze.json\n")
    print(
        f"[freeze-winner] k*(A): "
        f"{json.dumps({k: v['k_star'] for k, v in winners.items()})}"
    )
    print(f"[freeze-winner] sha256={digest}")
    return 0


# --------------------------------------------------------------------------
# phase B - frozen winner + 10 classical comparators on tapes B
# --------------------------------------------------------------------------

def _phase_b_worker(task: dict[str, Any]) -> tuple[str, int, dict[str, Any]]:
    cell_id = task["cell_id"]
    seed = int(task["tape_seed"])
    cell = {
        "cell_id": cell_id,
        "regime_persistence": task["regime_persistence"],
        "dominant_share": task["dominant_share"],
    }
    sched = scheduler()
    skeleton = skeleton_for(cell, seed)
    indices = classical_indices_for(cell, seed)

    def read_row(calendar_index: int) -> dict[str, float]:
        panel = simulate_full_des_frontier(
            skeleton=skeleton,
            scheduler=sched,
            calendars=np.asarray([full_action_calendars()[calendar_index]]),
        )
        return {key: float(panel[key][0]) for key in MATRIX_KEYS}

    record: dict[str, Any] = {
        "cell_id": cell_id,
        "tape_seed": seed,
        "skeleton_sha256": skeleton.skeleton_sha256,
        "tape_sha256": skeleton.tape_sha256,
        "classical_indices": indices,
        "k_star": int(task["k_star"]),
        "k_star_row": read_row(int(task["k_star"])),
        "classical_rows": {
            config_id: read_row(index) for config_id, index in indices.items()
        },
    }
    return cell_id, seed, record


def cmd_run_phase_b(workers: int) -> int:
    seed_map = must_load_map()
    winner = load_json(WINNER_PATH)
    tasks: list[dict[str, Any]] = []
    required = 0
    for cell in seed_map["cells"]:
        cell_id = cell["cell_id"]
        k_star = int(winner["winners"][cell_id]["k_star"])
        for seed in seed_map["tapes_b"][cell_id]:
            required += 1
            destination = PHASE_B_DIR / cell_id / f"tape_{seed}.json"
            if not destination.exists():
                tasks.append(
                    {
                        "cell_id": cell_id,
                        "regime_persistence": cell["regime_persistence"],
                        "dominant_share": cell["dominant_share"],
                        "tape_seed": int(seed),
                        "k_star": k_star,
                    }
                )
    print(
        f"[phase-b] {required} tape-B evaluations required; {len(tasks)} to run "
        f"({workers} workers); each = 1 frozen-winner row + 10 classical rows",
        flush=True,
    )
    if tasks:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_phase_b_worker, task) for task in tasks]
            for done, future in enumerate(as_completed(futures), 1):
                cell_id, seed, record = future.result()
                write_json_atomic(
                    PHASE_B_DIR / cell_id / f"tape_{seed}.json", record
                )
                if done % 16 == 0 or done == len(futures):
                    print(f"[phase-b] {done}/{len(futures)} tapes evaluated", flush=True)
    write_phase_b_summary(seed_map, winner)
    return 0


def write_phase_b_summary(seed_map: dict[str, Any], winner: dict[str, Any]) -> None:
    summary: dict[str, Any] = {
        "schema_version": "gate0_phase_b_summary_v1",
        "generated_at_utc": now_utc(),
        "winner_freeze_sha256": sha256_file(WINNER_PATH),
        "cells": {},
    }
    problems: list[str] = []
    for cell in seed_map["cells"]:
        cell_id = cell["cell_id"]
        seeds = [int(s) for s in seed_map["tapes_b"][cell_id]]
        k_star = int(winner["winners"][cell_id]["k_star"])
        k_star_values: list[float] = []
        classical_values: list[list[float]] = [
            [] for _ in CLASSICAL_CONFIG_IDS
        ]
        for seed in seeds:
            path = PHASE_B_DIR / cell_id / f"tape_{seed}.json"
            if not path.exists():
                problems.append(f"missing {path.name} for {cell_id}")
                continue
            record = load_json(path)
            if int(record["k_star"]) != k_star:
                problems.append(f"k_star mismatch in {path.name}")
            k_star_values.append(float(record["k_star_row"]["ret_visible"]))
            for column, config_id in enumerate(CLASSICAL_CONFIG_IDS):
                classical_values[column].append(
                    float(record["classical_rows"][config_id]["ret_visible"])
                )
        summary["cells"][cell_id] = {
            "n_tapes": len(k_star_values),
            "k_star": k_star,
            "k_star_values": k_star_values,
            "classical_values": classical_values,
        }
    summary["problems"] = problems
    write_json_atomic(PHASE_B_SUMMARY, summary)
    print(f"[phase-b] summary written: {PHASE_B_SUMMARY}")


# --------------------------------------------------------------------------
# falsifier F3 (placebo, tapes A only) and F4 (deterministic replays)
# --------------------------------------------------------------------------

def run_placebo_f3(seed_map: dict[str, Any]) -> dict[str, Any]:
    """F3: one uniformly random calendar must give G ~= 0 on tapes A.

    The instrument claims headroom for a policy; an uninformed calendar carries
    no information, so its G must not be positive.  Runs exclusively on tapes A
    so the placebo cannot touch or leak tape B.
    """
    rng_seed = int.from_bytes(
        hashlib.sha256(b"gate0-f3-uniform-placebo-v1").digest()[:8], "big"
    )
    rng = np.random.default_rng(rng_seed)
    results: dict[str, Any] = {
        "description": (
            "one uniform-random calendar from the 65,536 frontier; "
            "G_placebo = mean_A(X_rand) - max_c mean_A(X_c) must be ~= 0 (< SESOI)"
        ),
        "rng_seed": rng_seed,
        "per_cell": {},
        "passed": True,
    }
    n_placebo_tapes = 16
    for cell in seed_map["cells"]:
        cell_id = cell["cell_id"]
        seeds = [int(s) for s in seed_map["tapes_a"][cell_id][:n_placebo_tapes]]
        calendar_index = int(rng.integers(0, FRONTIER_SIZE))
        x_random: list[float] = []
        classical_by_config: dict[str, list[float]] = {
            config_id: [] for config_id in CLASSICAL_CONFIG_IDS
        }
        for seed in seeds:
            panel = load_phase_a_panel(cell_id, seed)
            x_random.append(float(panel["ret_visible"][calendar_index]))
            classical_row = load_phase_a_classical_row(cell_id, seed, panel)
            for config_id, value in classical_row.items():
                classical_by_config[config_id].append(value)
        best_classical_mean = max(
            float(np.mean(values)) for values in classical_by_config.values()
        )
        g_placebo = float(np.mean(x_random) - best_classical_mean)
        passed = bool(g_placebo < SESOI)
        results["passed"] &= passed
        results["per_cell"][cell_id] = {
            "calendar_index": calendar_index,
            "calendar": full_action_calendars()[calendar_index].astype(int).tolist(),
            "n_placebo_tapes": len(seeds),
            "g_placebo": g_placebo,
            "best_classical_mean_on_same_tapes": best_classical_mean,
            "passed": passed,
        }
    return results


def run_replays_f4(seed_map: dict[str, Any]) -> dict[str, Any]:
    """F4: replay >= 10% of tapes through direct SimPy; error > 1e-12 invalidates."""
    report: dict[str, Any] = {
        "tolerance": REPLAY_TOLERANCE,
        "required_fraction": DETERMINISTIC_REPLAY_FRACTION,
        "replays": [],
        "passed": True,
        "max_abs_error": 0.0,
    }
    all_tapes: list[tuple[str, dict[str, Any], int, str]] = []
    for cell in seed_map["cells"]:
        for seed in seed_map["tapes_a"][cell["cell_id"]]:
            all_tapes.append((cell["cell_id"], cell, int(seed), "A"))
        for seed in seed_map["tapes_b"][cell["cell_id"]]:
            all_tapes.append((cell["cell_id"], cell, int(seed), "B"))
    minimum = int(np.ceil(DETERMINISTIC_REPLAY_FRACTION * len(all_tapes)))
    report["total_tapes"] = len(all_tapes)
    report["minimum_required_replays"] = minimum
    replay_seeds = [int(s) for s in seed_map["instrument"]["replay_f4_seeds"]]
    # The frozen map draws these from `instrument_pool` and EXPLICITLY skips any
    # seed already assigned to tapes A or B (build_seed_assignment_map), so they
    # are disjoint from `all_tapes` by construction.  Intersecting the two -- as
    # this function used to -- yields the empty set and a vacuous pass.  Replay
    # the seeds the map lists.
    cells_in_frozen_order = list(seed_map["cells"])
    # The signed preregistration's F4 reads: "reproducir >= 10 % de los tapes
    # (muestreados antes de correr) contra el pipeline".  The frozen map's
    # instrument-pool seeds are NOT tapes A or B, so they do not literally
    # satisfy that clause.  Add a supplementary pass over a deterministic >=10%
    # sample of the real evaluation tapes, ordered by the same digest_json rule
    # the map itself uses, so the selection is fixed before any outcome exists.
    evaluation_minimum = int(np.ceil(0.10 * len(all_tapes)))
    evaluation_sample = sorted(
        all_tapes,
        key=lambda entry: int(digest_json(["f4_evaluation_tape", entry[2]])[:12], 16),
    )[:evaluation_minimum]
    report["evaluation_tape_replays_required"] = evaluation_minimum
    report["evaluation_tape_selection"] = (
        "deterministic digest_json ordering over tapes A+B, >=10% per the signed "
        "preregistration clause F4"
    )

    chosen: list[tuple[str, dict[str, Any], int, str]] = []
    for position, seed in enumerate(replay_seeds):
        # The map fixes WHICH seeds are replayed but not in which cell's physics;
        # we resolve that with a deterministic round robin over the frozen cell
        # order, so every cell is exercised and the choice cannot depend on any
        # outcome.  Declared as our disambiguation of a gap in the frozen map.
        cell = cells_in_frozen_order[position % len(cells_in_frozen_order)]
        chosen.append((cell["cell_id"], cell, int(seed), "instrument_pool"))
    report["replay_seed_selection"] = (
        "frozen assignment map instrument.replay_f4_seeds; cell assigned by "
        "deterministic round robin over the frozen cell order"
    )
    chosen = chosen[:minimum] + list(evaluation_sample)
    calendars = full_action_calendars()
    probe_indices = [0, 1, 255, FRONTIER_SIZE // 2, FRONTIER_SIZE - 1]
    for cell_id, cell, seed, arm in chosen:
        sched = scheduler()
        skeleton_transducer = skeleton_for(cell, seed)
        panel = simulate_full_des_frontier(
            skeleton=skeleton_transducer, scheduler=sched
        )
        max_abs_error = 0.0
        worst_field = ""
        for calendar_index in probe_indices:
            calendar = calendars[calendar_index].astype(int).tolist()
            sim, direct_panel = run_program_o_full_des_episode(
                seed=seed,
                calendar=calendar,
                scheduler=sched,
                regime_persistence=float(cell["regime_persistence"]),
                dominant_share=float(cell["dominant_share"]),
                downstream_freight_physics_mode=PHYSICS_MODE,
            )
            observed = direct_full_des_vector(sim, direct_panel)
            for key in MATRIX_KEYS:
                error = abs(
                    float(observed[key]) - float(panel[key][calendar_index])
                )
                if error > max_abs_error:
                    max_abs_error = error
                    worst_field = key
        ok = max_abs_error <= REPLAY_TOLERANCE
        report["passed"] = bool(report["passed"] and ok)
        report["max_abs_error"] = max(report["max_abs_error"], max_abs_error)
        report["replays"].append(
            {
                "cell_id": cell_id,
                "arm": arm,
                "tape_seed": seed,
                "calendars_checked": probe_indices,
                "max_abs_error": max_abs_error,
                "worst_field": worst_field,
                "passed": ok,
            }
        )
    report["replays_run"] = len(report["replays"])
    report["fraction_of_tapes_replayed"] = (
        report["replays_run"] / len(all_tapes) if all_tapes else 0.0
    )
    # A falsifier that reports success without evidence is not a falsifier.
    # Running fewer replays than the preregistered minimum is itself a failure.
    on_evaluation_tapes = sum(
        1 for r in report["replays"] if r.get("arm") in ("A", "B")
    )
    report["evaluation_tape_replays_run"] = on_evaluation_tapes
    if on_evaluation_tapes < report.get("evaluation_tape_replays_required", 0):
        report["passed"] = False
        report["insufficient_evaluation_tape_replays"] = {
            "ran": on_evaluation_tapes,
            "required": report.get("evaluation_tape_replays_required"),
            "note": (
                "the signed preregistration requires replaying >=10% of the "
                "evaluation tapes themselves, not only instrument-pool seeds"
            ),
        }
    if report["replays_run"] < minimum:
        report["passed"] = False
        report["insufficient_replays"] = {
            "ran": report["replays_run"],
            "required": minimum,
            "note": (
                "F4 cannot pass without executing at least the preregistered "
                "minimum number of deterministic replays"
            ),
        }
    return report


# --------------------------------------------------------------------------
# final analysis - estimands, bootstrap, decision rules
# --------------------------------------------------------------------------

def analyze(
    seed_map: dict[str, Any],
    winner: dict[str, Any],
    phase_b: dict[str, Any],
    f3: dict[str, Any],
    f4: dict[str, Any],
) -> dict[str, Any]:
    """Compute G_PI_naive, G_PI_split, Delta_bias + tape bootstrap; rule 2.6."""
    rng_seed = int.from_bytes(
        hashlib.sha256(b"gate0-split-tape-bootstrap-v1").digest()[:8], "big"
    )
    rng = np.random.default_rng(rng_seed)
    cells_out: dict[str, Any] = {}
    lane_closed_anywhere = False
    bias_flag_anywhere = False
    headroom_everywhere = True
    for cell in seed_map["cells"]:
        cell_id = cell["cell_id"]
        seeds_a = [int(s) for s in seed_map["tapes_a"][cell_id]]
        seeds_b = [int(s) for s in seed_map["tapes_b"][cell_id]]
        k_star = int(winner["winners"][cell_id]["k_star"])

        # ---- Phase A: frontier matrix + classical rows (naive design) -------
        xa_rows: list[np.ndarray] = []
        a_classical: list[dict[str, float]] = []
        for seed in seeds_a:
            panel = load_phase_a_panel(cell_id, seed)
            xa_rows.append(panel["ret_visible"])
            a_classical.append(load_phase_a_classical_row(cell_id, seed, panel))
        xa = np.stack(xa_rows)                                    # 64 x 65536
        xa_classical = np.asarray(
            [[row[cid] for cid in CLASSICAL_CONFIG_IDS] for row in a_classical]
        ).T                                                       # 10 x 64
        per_tape_max = xa.max(axis=1)                             # 64

        # ---- Phase B: frozen winner + classical ------------------------------
        kb_record = load_json(PHASE_B_SUMMARY)
        cell_b = kb_record["cells"][cell_id]
        xb_k = np.asarray(cell_b["k_star_values"], dtype=np.float64)   # 64
        xb_c = np.asarray(cell_b["classical_values"], dtype=np.float64)  # 10 x 64
        if len(seeds_b) != xb_k.shape[0]:
            raise AssertionError("phase B tape count mismatch")

        # ---- Point estimates --------------------------------------------------
        g_naive = float(per_tape_max.mean() - xa_classical.max(axis=0).mean())
        g_split = float(xb_k.mean() - xb_c.max(axis=0).mean())
        delta_bias = g_naive - g_split

        # ---- F1: ranking stability of k*(A) among the 11 B arms ---------------
        means = np.concatenate([xb_k[None, :], xb_c], axis=0).mean(axis=1)
        order = np.argsort(-means)
        rank_of_k_star = int(np.flatnonzero(order == 0)[0])
        top_decile_cut = max(0, int(np.ceil(0.1 * means.size)) - 1)
        f1_pass = bool(rank_of_k_star <= top_decile_cut)

        # ---- Bootstrap: unit = tape, 10k percentile resamples -----------------
        boot_naive = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
        boot_split = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
        n_a, n_b = xa.shape[0], xb_k.shape[0]
        for sample in range(BOOTSTRAP_RESAMPLES):
            draw_a = rng.integers(0, n_a, size=n_a)
            draw_b = rng.integers(0, n_b, size=n_b)
            naive_resample = float(per_tape_max[draw_a].mean())
            naive_resample -= float(xa_classical[:, draw_a].max(axis=0).mean())
            split_resample = float(xb_k[draw_b].mean())
            split_resample -= float(xb_c[:, draw_b].max(axis=0).mean())
            boot_naive[sample] = naive_resample
            boot_split[sample] = split_resample
        boot_bias = boot_naive - boot_split

        def ci95(values: np.ndarray) -> tuple[float, float]:
            low, high = np.percentile(values, [2.5, 97.5])
            return float(low), float(high)

        lcb_split, ucb_split = ci95(boot_split)
        lcb_naive, ucb_naive = ci95(boot_naive)
        lcb_bias, ucb_bias = ci95(boot_bias)

        rule_close_lane = bool(ucb_split < SESOI)
        rule_bias_flag = bool(delta_bias >= SESOI)
        rule_headroom = bool(lcb_split >= SESOI)
        lane_closed_anywhere |= rule_close_lane
        bias_flag_anywhere |= rule_bias_flag
        headroom_everywhere &= rule_headroom
        cells_out[cell_id] = {
            "k_star": k_star,
            "k_star_calendar": winner["winners"][cell_id]["calendar"],
            "g_pi_naive": g_naive,
            "g_pi_naive_ci95": [lcb_naive, ucb_naive],
            "g_pi_split": g_split,
            "delta_bias": delta_bias,
            "bootstrap": {
                "unit": "tape",
                "strata": cell_id,
                "resamples": BOOTSTRAP_RESAMPLES,
                "method": (
                    "percentile interval; classical comparator re-selected "
                    "inside every resample"
                ),
                "rng_seed": rng_seed,
                "rng_note": (
                    "single PCG64 stream seeded from SHA-256('gate0-split-tape-"
                    "bootstrap-v1'); cells consume draws in fixed cell order"
                ),
                "g_pi_split_ci95": [lcb_split, ucb_split],
                "g_pi_naive_ci95": [lcb_naive, ucb_naive],
                "delta_bias_ci95": [lcb_bias, ucb_bias],
            },
            "falsifiers": {
                "F1_ranking_stability": {
                    "rule": (
                        "k*(A) must sit in the top decile of the 11-arm mean "
                        "ranking on tapes B"
                    ),
                    "rank_of_k_star_among_arms": rank_of_k_star,
                    "arms": means.size,
                    "top_decile_cut": top_decile_cut,
                    "passed": f1_pass,
                },
                "F2_bias_dominates": {
                    "rule": "G_PI_naive large while LCB95(G_PI_split) < 0",
                    "triggered": bool(g_naive >= SESOI and lcb_split < 0.0),
                },
            },
            "rules_section_2_6": {
                "rule1_lane_closed_ucb95_below_sesoi": rule_close_lane,
                "rule2_selection_bias_material_delta_bias_ge_sesoi": rule_bias_flag,
                "rule3_headroom_above_sesoi_with_uncertainty": rule_headroom,
            },
        }

    verdict_cells: dict[str, str] = {}
    for cell_id, out in cells_out.items():
        if out["rules_section_2_6"]["rule1_lane_closed_ucb95_below_sesoi"]:
            verdict_cells[cell_id] = "CLOSED_LANE"
        elif headroom_everywhere:
            verdict_cells[cell_id] = "PASS_LEARNING_STUDY_ONLY_AUTHORIZED"
        else:
            verdict_cells[cell_id] = "INCONCLUSIVE"
    run_valid = bool(f4["passed"]) and not phase_b.get("problems") and bool(
        f3["passed"]
    )
    if not run_valid:
        verdict_cells = {cell_id: "INVALID_RUN" for cell_id in verdict_cells}
    verdict: dict[str, Any] = {
        "schema_version": "gate0_split_tape_verdict_v1",
        "preregistration_id": PREREGISTRATION_ID,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "analyzed_at_utc": now_utc(),
        "inputs": {
            "seed_assignment_map": artifact_record(MAP_PATH),
            "phase_a_summary": artifact_record(PHASE_A_SUMMARY),
            "winner_freeze": {
                "path": str(WINNER_PATH.relative_to(ROOT)),
                "sha256": sha256_file(WINNER_PATH),
            },
            "phase_b_summary": artifact_record(PHASE_B_SUMMARY),
            "falsifier_F3": artifact_record(F3_PATH),
            "falsifier_F4": artifact_record(F4_PATH),
        },
        "estimands": {
            "G_PI_naive": (
                "mean_A[max_k X_{A,k}] - max_c[mean_A X_{A,c}] (diagnostic only; "
                "original unsplit design instantiated on tapes A)"
            ),
            "G_PI_split": (
                "mean_B(X_{B,k*(A)}) - max_c mean_B(X_{B,c}) (gate estimand)"
            ),
            "Delta_bias": "G_PI_naive - G_PI_split (methodological result)",
        },
        "sesoi": SESOI,
        "decision_rules_preregistered": {
            "rule_2_6_1": "UCB95(G_PI_split) < 0.01 in any cell => close the lane",
            "rule_2_6_2": (
                "Delta_bias >= 0.01 in any cell => document selection inflation"
            ),
            "rule_2_6_3": (
                "G_PI_split above SESOI with uncertainty in ALL cells => "
                "authorize learning study only"
            ),
        },
        "falsifier_F3_placebo": f3,
        "falsifier_F4_replays": {
            key: value
            for key, value in f4.items()
            if key != "replays"
        },
        "falsifier_F4_replays_detail": f4["replays"],
        "cells": cells_out,
        "verdict_per_cell": verdict_cells,
        "overall": {
            "lane_closed": lane_closed_anywhere,
            "selection_bias_material": bias_flag_anywhere,
            "headroom_confirmed_all_cells": headroom_everywhere,
            "run_valid": run_valid,
            "claim_if_lane_closed": "sin headroom fisico detectable bajo este contrato",
        },
    }
    return verdict


def cmd_analyze(workers: int) -> int:
    seed_map = must_load_map()
    winner = load_json(WINNER_PATH)
    phase_b = load_json(PHASE_B_SUMMARY)
    if not F4_PATH.exists():
        print("[falsifiers] running F4 deterministic replays ...", flush=True)
        f4 = run_replays_f4(seed_map)
        write_json_atomic(F4_PATH, f4)
        print(
            f"[falsifiers] F4 passed={f4['passed']} "
            f"max_err={f4['max_abs_error']:.3e}"
        )
    else:
        f4 = load_json(F4_PATH)
    if not F3_PATH.exists():
        print("[falsifiers] running F3 uniform-calendar placebo ...", flush=True)
        f3 = run_placebo_f3(seed_map)
        write_json_atomic(F3_PATH, f3)
        print(
            "[falsifiers] F3 passed="
            f"{f3['passed']} g_placebo="
            f"{ {c: round(v['g_placebo'], 6) for c, v in f3['per_cell'].items()} }"
        )
    else:
        f3 = load_json(F3_PATH)
    verdict = analyze(seed_map, winner, phase_b, f3, f4)
    write_json_atomic(ANALYSIS_PATH, verdict)
    digest = sha256_file(ANALYSIS_PATH)
    (OUTPUT_ROOT / "VERDICT.sha256").write_text(f"{digest}  verdict.json\n")
    print(json.dumps(verdict["verdict_per_cell"], indent=2))
    print(json.dumps(verdict["overall"], indent=2))
    for cell_id, out in verdict["cells"].items():
        print(
            f"[analyze] {cell_id}: G_naive={out['g_pi_naive']:+.5f} "
            f"G_split={out['g_pi_split']:+.5f} "
            f"UCB95={out['bootstrap']['g_pi_split_ci95'][1]:+.5f} "
            f"Delta_bias={out['delta_bias']:+.5f}"
        )
    print(f"[analyze] verdict written: {ANALYSIS_PATH} sha256={digest}")
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=[
            "freeze-map",
            "run-phase-a",
            "freeze-winner",
            "run-phase-b",
            "analyze",
        ],
    )
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if args.stage == "freeze-map":
        return cmd_freeze_map()
    if args.stage == "run-phase-a":
        return cmd_run_phase_a(args.workers)
    if args.stage == "freeze-winner":
        return cmd_freeze_winner()
    if args.stage == "run-phase-b":
        return cmd_run_phase_b(args.workers)
    if args.stage == "analyze":
        return cmd_analyze(args.workers)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
