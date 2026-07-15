#!/usr/bin/env python3
"""Certify Program O perfect-information headroom in the full Op1--Op13 DES.

The expensive open-loop frontier is replayed by an event transducer only after
extracting an action-independent skeleton from the direct SimPy model.  Every
promotion-relevant calendar is then replayed directly, together with 256
deterministic audit calendars per tape/profile.  A transducer-only result can
never pass this script.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from supply_chain.program_o_full_des import (  # noqa: E402
    run_program_o_full_des_episode,
)
from supply_chain.program_o_full_des_transducer import (  # noqa: E402
    MATRIX_KEYS,
    direct_full_des_trace,
    direct_full_des_vector,
    extract_full_des_skeleton,
    full_action_calendars,
    simulate_full_des_frontier,
)
from scripts.program_o_full_des_guard import (  # noqa: E402
    verify_seed_claim,
    verify_tracked_freeze,
)

DEFAULT_CONTRACT = ROOT / "contracts/program_o_full_des_hpi_translation_v1.json"
DEFAULT_OUTPUT_ROOT = ROOT / "results/program_o/full_des_hpi_translation_v1"
DEFAULT_DEVELOPMENT_FREEZE = (
    ROOT / "research/paper2_exhaustive_search/"
    "program_o_full_des_development_freeze_20260714.json"
)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def git_is_clean() -> bool:
    return not subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip()


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def update_progress(path: Path, **updates: Any) -> None:
    state: dict[str, Any] = {}
    if path.is_file():
        state = json.loads(path.read_text())
    state.update(updates)
    state["updated_at_utc"] = now_utc()
    write_json_atomic(path, state)


def calendar_index(calendar: Sequence[int]) -> int:
    if len(calendar) != 8 or any(int(value) not in range(4) for value in calendar):
        raise ValueError("calendar must be base-4 length eight")
    return int(sum(int(value) * 4 ** (7 - idx) for idx, value in enumerate(calendar)))


def frozen_profiles(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    cell = contract["selected_cell"]
    schedulers = contract["action"]["within_week_schedulers"]
    primary = str(contract["action"]["primary_scheduler"])
    profiles = [
        {
            "profile_id": f"{cell['cell_id']}__{scheduler_id}",
            "role": "primary" if scheduler_id == primary else "ordering_sensitivity",
            "scheduler_id": scheduler_id,
            "regime_persistence": float(cell["regime_persistence"]),
            "dominant_share": float(cell["dominant_product_share"]),
            "complete_substitution": False,
        }
        for scheduler_id in schedulers
    ]
    profiles.append(
        {
            "profile_id": f"fungible_null__{primary}",
            "role": "exact_null",
            "scheduler_id": primary,
            "regime_persistence": float(cell["regime_persistence"]),
            "dominant_share": float(cell["dominant_product_share"]),
            "complete_substitution": True,
        }
    )
    return profiles


def matrix_path(output_root: Path, stage: str, profile_id: str, seed: int) -> Path:
    return (
        output_root
        / stage
        / "raw_calendar_matrix"
        / profile_id
        / f"tape_{int(seed)}.npz"
    )


def produce_seed(
    contract_path: str,
    output_root: str,
    stage: str,
    seed: int,
) -> list[dict[str, Any]]:
    contract_file = Path(contract_path)
    contract = json.loads(contract_file.read_text())
    output = Path(output_root)
    profiles = frozen_profiles(contract)
    schedulers = contract["action"]["within_week_schedulers"]
    primary = str(contract["action"]["primary_scheduler"])
    skeleton, _sim = extract_full_des_skeleton(
        seed=int(seed),
        scheduler=schedulers[primary],
        regime_persistence=float(contract["selected_cell"]["regime_persistence"]),
        dominant_share=float(contract["selected_cell"]["dominant_product_share"]),
    )
    skeleton_path = output / stage / "skeletons" / f"tape_{int(seed)}.json"
    if skeleton_path.exists():
        raise FileExistsError(f"refusing to overwrite {skeleton_path}")
    write_json_atomic(skeleton_path, skeleton.as_dict())
    rows: list[dict[str, Any]] = []
    for profile in profiles:
        panel = simulate_full_des_frontier(
            skeleton=skeleton,
            scheduler=schedulers[str(profile["scheduler_id"])],
            complete_substitution=bool(profile["complete_substitution"]),
        )
        destination = matrix_path(output, stage, str(profile["profile_id"]), int(seed))
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".npz.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **panel)
        os.replace(temporary, destination)
        recorded_path = display_path(destination)
        rows.append(
            {
                "profile_id": str(profile["profile_id"]),
                "seed": int(seed),
                "path": recorded_path,
                "sha256": sha256(destination),
                "bytes": destination.stat().st_size,
                "tape_sha256": skeleton.tape_sha256,
                "prefix_state_hash": skeleton.prefix_state_hash,
                "skeleton_sha256": skeleton.skeleton_sha256,
            }
        )
    return rows


def load_panel(
    output_root: Path, stage: str, profile_id: str, seeds: Sequence[int]
) -> dict[str, np.ndarray]:
    rows: dict[str, list[np.ndarray]] = {key: [] for key in MATRIX_KEYS}
    for seed in seeds:
        with np.load(matrix_path(output_root, stage, profile_id, seed)) as shard:
            if tuple(shard.files) != MATRIX_KEYS:
                raise AssertionError(f"matrix schema drift for {profile_id}/{seed}")
            for key in MATRIX_KEYS:
                rows[key].append(np.asarray(shard[key]))
    return {key: np.stack(values) for key, values in rows.items()}


def select_static(ret_visible: np.ndarray, weights: np.ndarray | None = None) -> int:
    score = ret_visible.mean(axis=0) if weights is None else weights @ ret_visible
    return int(np.argmax(score))


def safe_mask(
    panel: Mapping[str, np.ndarray],
    tape_index: int,
    static_index: int,
    contract: Mapping[str, Any],
    *,
    tolerance: float = 1e-12,
) -> np.ndarray:
    rules = contract["safe_oracle_constraints_per_tape"]
    eligible = np.ones(panel["ret_visible"].shape[1], dtype=bool)
    for key in rules["higher_or_equal"]:
        row = panel[key][tape_index]
        eligible &= row >= float(row[static_index]) - tolerance
    for key in rules["lower_or_equal"]:
        row = panel[key][tape_index]
        eligible &= row <= float(row[static_index]) + tolerance
    for key in rules["equal"]:
        row = panel[key][tape_index]
        eligible &= np.abs(row - float(row[static_index])) <= tolerance
    eligible &= np.abs(panel["mass_residual"][tape_index]) <= float(
        rules["conservation_absolute_tolerance"]
    )
    eligible[int(static_index)] = True
    return eligible


def oracle_ties(
    panel: Mapping[str, np.ndarray],
    static_index: int,
    contract: Mapping[str, Any],
    *,
    tolerance: float = 1e-12,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    raw_ties: list[np.ndarray] = []
    safe_ties: list[np.ndarray] = []
    for tape_index, row in enumerate(panel["ret_visible"]):
        raw_best = float(row.max())
        raw_ties.append(np.flatnonzero(np.abs(row - raw_best) <= tolerance))
        eligible = safe_mask(panel, tape_index, static_index, contract)
        safe_scores = np.where(eligible, row, -np.inf)
        safe_best = float(safe_scores.max())
        safe_ties.append(
            np.flatnonzero(eligible & (np.abs(row - safe_best) <= tolerance))
        )
    return raw_ties, safe_ties


def profile_summary(
    panel: Mapping[str, np.ndarray], contract: Mapping[str, Any]
) -> dict[str, Any]:
    calendars = full_action_calendars()
    static_index = select_static(panel["ret_visible"])
    raw_ties, safe_ties = oracle_ties(panel, static_index, contract)
    raw_indices = np.asarray([int(ties.min()) for ties in raw_ties], dtype=np.int32)
    safe_indices = np.asarray([int(ties.min()) for ties in safe_ties], dtype=np.int32)
    tapes = np.arange(panel["ret_visible"].shape[0])
    static = panel["ret_visible"][:, static_index]
    raw_deltas = panel["ret_visible"][tapes, raw_indices] - static
    safe_deltas = panel["ret_visible"][tapes, safe_indices] - static
    counts = Counter(map(int, safe_indices))
    actions = calendars[safe_indices].ravel()
    action_counts = np.bincount(actions, minlength=4)
    fractions = action_counts / max(1, int(action_counts.sum()))
    keys = tuple(
        dict.fromkeys(
            [
                *contract["safe_oracle_constraints_per_tape"]["higher_or_equal"],
                *contract["safe_oracle_constraints_per_tape"]["lower_or_equal"],
                *contract["safe_oracle_constraints_per_tape"]["equal"],
            ]
        )
    )
    guardrails = {
        key: float(
            np.mean(panel[key][tapes, safe_indices] - panel[key][:, static_index])
        )
        for key in keys
    }
    return {
        "best_static_calendar_index": int(static_index),
        "best_static_calendar": calendars[static_index].astype(int).tolist(),
        "best_static_mean_ret": float(static.mean()),
        "raw_h_pi": float(raw_deltas.mean()),
        "safe_h_pi": float(safe_deltas.mean()),
        "raw_h_pi_per_tape": raw_deltas.tolist(),
        "safe_h_pi_per_tape": safe_deltas.tolist(),
        "raw_oracle_indices": raw_indices.astype(int).tolist(),
        "safe_oracle_indices": safe_indices.astype(int).tolist(),
        "raw_oracle_tie_sets": [ties.astype(int).tolist() for ties in raw_ties],
        "safe_oracle_tie_sets": [ties.astype(int).tolist() for ties in safe_ties],
        "raw_favorable_tapes": int(np.sum(raw_deltas > 1e-15)),
        "safe_favorable_tapes": int(np.sum(safe_deltas > 1e-15)),
        "unique_safe_oracle_calendars": int(len(counts)),
        "modal_safe_oracle_fraction": float(max(counts.values()) / len(safe_indices)),
        "safe_action_support": {
            "counts": {str(i): int(value) for i, value in enumerate(action_counts)},
            "fractions": {str(i): float(value) for i, value in enumerate(fractions)},
            "material_action_levels_at_10pct": int(np.sum(fractions >= 0.10)),
        },
        "safe_guardrail_mean_deltas": guardrails,
    }


def deterministic_replay_indices(
    *, profile_id: str, seed: int, summary: Mapping[str, Any]
) -> list[int]:
    salt = int.from_bytes(
        hashlib.sha256(f"{profile_id}:{seed}".encode()).digest()[:8], "big"
    )
    rng = np.random.default_rng(salt)
    selected = set(map(int, rng.choice(65536, size=256, replace=False)))
    selected.update(calendar_index([action] * 8) for action in range(4))
    selected.add(int(summary["best_static_calendar_index"]))
    return sorted(selected)


def replay_profile_seed(
    contract_path: str,
    output_root: str,
    stage: str,
    profile: Mapping[str, Any],
    seed: int,
    indices: Sequence[int],
    tolerance: float,
) -> dict[str, Any]:
    contract = json.loads(Path(contract_path).read_text())
    scheduler = contract["action"]["within_week_schedulers"][
        str(profile["scheduler_id"])
    ]
    calendars = full_action_calendars()
    shard_path = matrix_path(
        Path(output_root), stage, str(profile["profile_id"]), int(seed)
    )
    with np.load(shard_path) as shard:
        expected = {key: np.asarray(shard[key]) for key in MATRIX_KEYS}
    compared_keys = list(MATRIX_KEYS)
    if bool(profile["complete_substitution"]):
        # Under the exact fungibility ablation, C/H supply tags are inert
        # metadata and their ending partition depends on FIFO token identity.
        # The null claim concerns the aggregate physical/order trajectory,
        # metrics, and resources, all of which must be identical.  Product-tag
        # partitions are therefore neither a null invariant nor an estimand.
        compared_keys = [
            key
            for key in compared_keys
            if key not in {"ending_inventory_P_C", "ending_inventory_P_H"}
        ]
    mismatches: list[dict[str, Any]] = []
    max_abs_error = 0.0
    trace_mismatches: list[dict[str, Any]] = []
    skeleton, _ = extract_full_des_skeleton(
        seed=int(seed),
        scheduler=scheduler,
        regime_persistence=float(profile["regime_persistence"]),
        dominant_share=float(profile["dominant_share"]),
    )
    for index in indices:
        calendar = calendars[int(index)].astype(int).tolist()
        sim, panel = run_program_o_full_des_episode(
            seed=int(seed),
            calendar=calendar,
            scheduler=scheduler,
            regime_persistence=float(profile["regime_persistence"]),
            dominant_share=float(profile["dominant_share"]),
            complete_substitution=bool(profile["complete_substitution"]),
        )
        observed = direct_full_des_vector(sim, panel)
        direct_trace = direct_full_des_trace(sim)
        transducer_trace: dict[str, Any] = {}
        simulate_full_des_frontier(
            skeleton=skeleton,
            scheduler=scheduler,
            calendars=[calendar],
            complete_substitution=bool(profile["complete_substitution"]),
            trace_out=transducer_trace,
        )
        if comparable_trace(
            direct_trace,
            complete_substitution=bool(profile["complete_substitution"]),
        ) != comparable_trace(
            transducer_trace,
            complete_substitution=bool(profile["complete_substitution"]),
        ):
            trace_mismatches.append(
                {
                    "calendar_index": int(index),
                    "direct_trace_sha256": direct_trace["sha256"],
                    "transducer_trace_sha256": transducer_trace.get("sha256"),
                }
            )
        for key in compared_keys:
            error = abs(float(observed[key]) - float(expected[key][int(index)]))
            max_abs_error = max(max_abs_error, error)
            if error > float(tolerance):
                mismatches.append(
                    {
                        "calendar_index": int(index),
                        "calendar": calendar,
                        "field": key,
                        "direct": float(observed[key]),
                        "transducer": float(expected[key][int(index)]),
                        "abs_error": error,
                    }
                )
                if len(mismatches) >= 20:
                    break
        if len(mismatches) >= 20 or len(trace_mismatches) >= 20:
            break
    return {
        "profile_id": str(profile["profile_id"]),
        "seed": int(seed),
        "calendar_count": len(indices),
        "calendar_indices_sha256": digest_json(list(map(int, indices))),
        "compared_fields": compared_keys,
        "passed": not mismatches and not trace_mismatches,
        "max_abs_error": float(max_abs_error),
        "mismatches": mismatches,
        "trace_mismatches": trace_mismatches,
    }


def bootstrap_counts(n_tapes: int, resamples: int, rng_seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(rng_seed))
    draws = rng.integers(0, n_tapes, size=(int(resamples), n_tapes))
    counts = np.zeros((int(resamples), n_tapes), dtype=np.uint8)
    for index, draw in enumerate(draws):
        counts[index] = np.bincount(draw, minlength=n_tapes)
    return counts


def bootstrap_profile(
    panel: Mapping[str, np.ndarray],
    contract: Mapping[str, Any],
    counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    ret = panel["ret_visible"]
    static_indices = np.empty(len(counts), dtype=np.int32)
    for start in range(0, len(counts), 64):
        stop = min(len(counts), start + 64)
        static_indices[start:stop] = np.argmax(
            counts[start:stop].astype(float) @ ret, axis=1
        )
    raw_values = np.empty(len(counts), dtype=float)
    safe_values = np.empty(len(counts), dtype=float)
    safe_delta_cache: dict[int, np.ndarray] = {}
    for static_index in sorted(set(map(int, static_indices))):
        _raw, safe_ties = oracle_ties(panel, static_index, contract)
        safe_indices = np.asarray([int(ties.min()) for ties in safe_ties])
        safe_delta_cache[static_index] = (
            ret[np.arange(ret.shape[0]), safe_indices] - ret[:, static_index]
        )
    raw_max = ret.max(axis=1)
    for draw, static_index in enumerate(static_indices):
        weight = counts[draw].astype(float)
        raw_values[draw] = float(
            weight @ (raw_max - ret[:, int(static_index)]) / ret.shape[0]
        )
        safe_values[draw] = float(
            weight @ safe_delta_cache[int(static_index)] / ret.shape[0]
        )
    distribution = {
        str(index): int(value)
        for index, value in sorted(Counter(map(int, static_indices)).items())
    }
    return raw_values, safe_values, distribution


def checksum_tree(stage_root: Path) -> Path:
    destination = stage_root / "checksums.sha256"
    lines = []
    for path in sorted(stage_root.rglob("*")):
        if path.is_file() and path != destination and not path.name.endswith(".tmp"):
            lines.append(f"{sha256(path)}  {path.relative_to(stage_root)}")
    destination.write_text("\n".join(lines) + "\n")
    return destination


def comparable_trace(
    trace: Mapping[str, Any], *, complete_substitution: bool
) -> dict[str, Any]:
    payload = {
        "state_events": [dict(row) for row in trace["state_events"]],
        "orders": [dict(row) for row in trace["orders"]],
    }
    if complete_substitution:
        # Product partitions are inert FIFO metadata in the exact fungible null.
        for row in payload["state_events"]:
            row.pop("product_inventory_after", None)
    return payload


def preseed_parity_gate(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Exhaust direct/transducer parity for H=1,2,3 on a burned tape."""
    burned_seed = int(contract["parity_gate"]["burned_seed"])
    tolerance = float(contract["parity_gate"]["tolerance"])
    profiles = frozen_profiles(contract)
    scheduler_map = contract["action"]["within_week_schedulers"]
    rows: list[dict[str, Any]] = []
    for profile in profiles:
        scheduler = scheduler_map[str(profile["scheduler_id"])]
        for weeks in (1, 2, 3):
            skeleton, _ = extract_full_des_skeleton(
                seed=burned_seed,
                scheduler=scheduler,
                regime_persistence=float(profile["regime_persistence"]),
                dominant_share=float(profile["dominant_share"]),
                decision_weeks=weeks,
            )
            for calendar_array in full_action_calendars(weeks):
                calendar = calendar_array.astype(int).tolist()
                direct_sim, direct_panel = run_program_o_full_des_episode(
                    seed=burned_seed,
                    calendar=calendar,
                    scheduler=scheduler,
                    regime_persistence=float(profile["regime_persistence"]),
                    dominant_share=float(profile["dominant_share"]),
                    complete_substitution=bool(profile["complete_substitution"]),
                )
                expected = direct_full_des_vector(direct_sim, direct_panel)
                transducer_trace: dict[str, Any] = {}
                observed = simulate_full_des_frontier(
                    skeleton=skeleton,
                    scheduler=scheduler,
                    calendars=[calendar],
                    complete_substitution=bool(profile["complete_substitution"]),
                    trace_out=transducer_trace,
                )
                excluded = (
                    {"ending_inventory_P_C", "ending_inventory_P_H"}
                    if bool(profile["complete_substitution"])
                    else set()
                )
                mismatches = {
                    key: abs(float(observed[key][0]) - float(expected[key]))
                    for key in MATRIX_KEYS
                    if key not in excluded
                    if abs(float(observed[key][0]) - float(expected[key])) > tolerance
                }
                direct_trace = direct_full_des_trace(direct_sim)
                trace_equal = comparable_trace(
                    direct_trace,
                    complete_substitution=bool(profile["complete_substitution"]),
                ) == comparable_trace(
                    transducer_trace,
                    complete_substitution=bool(profile["complete_substitution"]),
                )
                rows.append(
                    {
                        "profile_id": str(profile["profile_id"]),
                        "weeks": weeks,
                        "calendar": calendar,
                        "matrix_pass": not mismatches,
                        "matrix_mismatches": mismatches,
                        "trace_pass": trace_equal,
                        "direct_trace_sha256": direct_trace["sha256"],
                        "transducer_trace_sha256": transducer_trace.get("sha256"),
                    }
                )
                if mismatches or not trace_equal:
                    return {
                        "passed": False,
                        "burned_seed": burned_seed,
                        "episode_count": len(rows),
                        "rows_sha256": digest_json(rows),
                        "first_failure": rows[-1],
                    }
    return {
        "passed": True,
        "burned_seed": burned_seed,
        "episode_count": len(rows),
        "rows_sha256": digest_json(rows),
        "coverage": "all 4^H calendars for H=1,2,3 across every frozen profile",
    }


def verify_validation_freeze(
    freeze_path: Path,
    contract_path: Path,
    development_result: Path,
    expected_range: Sequence[int],
) -> None:
    if not freeze_path.is_file():
        raise RuntimeError("validation remains sealed: additive freeze is absent")
    freeze = json.loads(freeze_path.read_text())
    if not development_result.is_file():
        raise RuntimeError("validation remains sealed: development result is absent")
    failures = []
    if freeze.get("status") != "AUTHORIZED_PROGRAM_O_FULL_DES_VALIDATION":
        failures.append("freeze status")
    if freeze.get("contract_sha256") != sha256(contract_path):
        failures.append("contract hash")
    if freeze.get("development_result_sha256") != sha256(development_result):
        failures.append("development result hash")
    if freeze.get("validation_seed_range") != list(map(int, expected_range)):
        failures.append("seed range")
    if failures:
        raise RuntimeError("validation remains sealed: " + ", ".join(failures))


def execute(
    *,
    contract_path: Path,
    output_root: Path,
    stage: str,
    workers: int,
    run_dir: Path,
    run_id: str,
    execution_freeze: Path,
    seed_claim: Path,
    development_result: Path | None,
) -> Path:
    if not git_is_clean():
        raise RuntimeError("refusing scientific execution from a dirty worktree")
    contract = json.loads(contract_path.read_text())
    seed_block = contract["tape_blocks"][stage]
    start, end = map(int, seed_block["range"])
    seeds = tuple(range(start, end + 1))
    authorization = verify_tracked_freeze(
        freeze_path=execution_freeze,
        contract_path=contract_path,
        stage=stage,
        run_id=run_id,
        run_dir=run_dir,
        seed_range=[start, end],
    )
    verify_seed_claim(
        claim_path=seed_claim,
        authorization=authorization,
        contract_sha256=sha256(contract_path),
    )
    if output_root.resolve() != (run_dir.resolve() / "artifacts"):
        raise RuntimeError("output root is not bound to the authorized run directory")
    if stage == "development":
        if (
            seed_block["status"]
            != "SEALED_NOT_OPENED_PENDING_IMPLEMENTATION_TESTS_AND_COMMIT"
        ):
            raise RuntimeError("development seed block status is not sealed")
    else:
        if development_result is None:
            raise RuntimeError("validation requires an explicit development result")
        verify_validation_freeze(
            execution_freeze,
            contract_path,
            development_result.resolve(),
            [start, end],
        )
    preseed_parity = preseed_parity_gate(contract)
    if not preseed_parity["passed"]:
        raise RuntimeError(
            "preseed direct/transducer parity failed before any sealed tape opened: "
            + json.dumps(preseed_parity["first_failure"], sort_keys=True)
        )
    stage_root = output_root / stage
    if stage_root.exists():
        raise FileExistsError(f"refusing to overwrite {stage_root}")
    stage_root.mkdir(parents=True)
    progress_path = stage_root / "progress.json"
    profiles = frozen_profiles(contract)
    commit = git_commit()
    source_paths = (
        contract_path,
        Path(__file__).resolve(),
        ROOT / "supply_chain/program_o_full_des.py",
        ROOT / "supply_chain/program_o_full_des_transducer.py",
        ROOT / "supply_chain/supply_chain.py",
        ROOT / "supply_chain/ret_thesis.py",
        ROOT / "supply_chain/episode_metrics.py",
        ROOT / "scripts/program_o_full_des_guard.py",
        ROOT / "scripts/validate_program_o_full_des_hpi_custody.py",
    )
    manifests = {
        "commit_manifest.json": {
            "scientific_commit": commit,
            "worktree_clean": True,
            "source_hashes": {
                str(path.relative_to(ROOT)): sha256(path) for path in source_paths
            },
        },
        "contract_manifest.json": {
            "path": str(contract_path.relative_to(ROOT)),
            "sha256": sha256(contract_path),
            "schema_version": contract["schema_version"],
        },
        "seed_manifest.json": {
            "stage": stage,
            "seed_range": [start, end],
            "seeds": list(seeds),
            "status_at_open": str(seed_block["status"]),
            "seed_claim": str(seed_claim),
            "seed_claim_sha256": sha256(seed_claim),
            "run_id": run_id,
        },
        "command_manifest.json": {
            "argv": sys.argv,
            "cwd": str(ROOT),
            "workers": int(workers),
            "run_dir": str(run_dir),
            "run_id": run_id,
            "execution_freeze": str(execution_freeze),
        },
        "environment_manifest.json": {
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "executable": sys.executable,
        },
    }
    for name, payload in manifests.items():
        write_json_atomic(stage_root / name, payload)
    write_json_atomic(stage_root / "preseed_parity.json", preseed_parity)
    update_progress(
        progress_path,
        status="RUNNING",
        phase="frontier",
        stage=stage,
        seeds_total=len(seeds),
        seeds_completed=0,
        parity_tasks_completed=0,
    )
    shard_manifest: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=int(workers)) as pool:
        futures = {
            pool.submit(
                produce_seed,
                str(contract_path),
                str(output_root),
                stage,
                int(seed),
            ): seed
            for seed in seeds
        }
        completed = 0
        for future in as_completed(futures):
            shard_manifest.extend(future.result())
            completed += 1
            update_progress(
                progress_path,
                seeds_completed=completed,
                last_seed=int(futures[future]),
            )
    shard_manifest.sort(key=lambda row: (row["profile_id"], row["seed"]))
    write_json_atomic(stage_root / "raw_shard_manifest.json", shard_manifest)

    panels: dict[str, dict[str, np.ndarray]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for profile in profiles:
        profile_id = str(profile["profile_id"])
        panels[profile_id] = load_panel(output_root, stage, profile_id, seeds)
        summaries[profile_id] = {
            **profile,
            **profile_summary(panels[profile_id], contract),
        }

    update_progress(progress_path, phase="direct_parity")
    parity_tasks: list[tuple[dict[str, Any], int, list[int]]] = []
    seed_to_position = {seed: position for position, seed in enumerate(seeds)}
    for profile in profiles:
        summary = summaries[str(profile["profile_id"])]
        for seed in seeds:
            position = seed_to_position[seed]
            indices = set(
                deterministic_replay_indices(
                    profile_id=str(profile["profile_id"]),
                    seed=seed,
                    summary=summary,
                )
            )
            if profile["role"] != "exact_null":
                indices.update(summary["raw_oracle_tie_sets"][position])
                indices.update(summary["safe_oracle_tie_sets"][position])
            parity_tasks.append((profile, seed, sorted(map(int, indices))))
    parity_rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=int(workers)) as pool:
        futures = {
            pool.submit(
                replay_profile_seed,
                str(contract_path),
                str(output_root),
                stage,
                profile,
                int(seed),
                indices,
                float(contract["parity_gate"]["tolerance"]),
            ): (str(profile["profile_id"]), int(seed))
            for profile, seed, indices in parity_tasks
        }
        completed = 0
        for future in as_completed(futures):
            parity_rows.append(future.result())
            completed += 1
            update_progress(
                progress_path,
                parity_tasks_completed=completed,
                parity_tasks_total=len(futures),
                last_parity_task=list(futures[future]),
            )
    parity_rows.sort(key=lambda row: (row["profile_id"], row["seed"]))
    write_json_atomic(stage_root / "direct_parity.json", parity_rows)
    parity_pass = bool(parity_rows and all(row["passed"] for row in parity_rows))

    null_summary = next(
        row for row in summaries.values() if row["role"] == "exact_null"
    )
    null_panel = panels[str(null_summary["profile_id"])]
    null_identity_keys = tuple(
        key
        for key in MATRIX_KEYS
        if key not in {"ending_inventory_P_C", "ending_inventory_P_H"}
    )
    null_identity = {
        key: bool(np.all(null_panel[key] == null_panel[key][:, :1]))
        for key in null_identity_keys
    }
    null_pass = bool(all(null_identity.values()) and null_summary["raw_h_pi"] == 0.0)

    inference: dict[str, Any] | None = None
    if stage == "validation":
        rule = contract["validation_pass_rule"]
        counts = bootstrap_counts(
            len(seeds),
            int(rule["bootstrap_resamples"]),
            int(rule["bootstrap_rng_seed"]),
        )
        promotion = [row for row in summaries.values() if row["role"] != "exact_null"]
        raw_errors = []
        safe_errors = []
        distributions: dict[str, Any] = {}
        for row in promotion:
            raw_values, safe_values, static_distribution = bootstrap_profile(
                panels[str(row["profile_id"])], contract, counts
            )
            distributions[str(row["profile_id"])] = {
                "raw": raw_values,
                "safe": safe_values,
                "static": static_distribution,
            }
            raw_errors.append(float(row["raw_h_pi"]) - raw_values)
            safe_errors.append(float(row["safe_h_pi"]) - safe_values)
        raw_critical = float(
            np.quantile(np.max(np.stack(raw_errors), axis=0), 0.95, method="higher")
        )
        safe_critical = float(
            np.quantile(np.max(np.stack(safe_errors), axis=0), 0.95, method="higher")
        )
        for row in promotion:
            distribution = distributions[str(row["profile_id"])]
            row["simultaneous_raw_lcb95"] = float(row["raw_h_pi"] - raw_critical)
            row["simultaneous_safe_lcb95"] = float(row["safe_h_pi"] - safe_critical)
            row["bootstrap_static_index_distribution"] = distribution["static"]
            row["bootstrap_raw_distribution_sha256"] = hashlib.sha256(
                distribution["raw"].tobytes()
            ).hexdigest()
            row["bootstrap_safe_distribution_sha256"] = hashlib.sha256(
                distribution["safe"].tobytes()
            ).hexdigest()
        inference = {
            "method": "paired tape bootstrap with static reselected and safe oracle recomputed",
            "resamples": int(rule["bootstrap_resamples"]),
            "rng_seed": int(rule["bootstrap_rng_seed"]),
            "paired_counts_sha256": hashlib.sha256(counts.tobytes()).hexdigest(),
            "raw_critical_max_error": raw_critical,
            "safe_critical_max_error": safe_critical,
        }

    primary = next(row for row in summaries.values() if row["role"] == "primary")
    ordering = [
        row for row in summaries.values() if row["role"] == "ordering_sensitivity"
    ]
    dev_rule = contract["development_pass_rule"]
    structural_pass = bool(
        float(primary["safe_h_pi"]) >= float(dev_rule["mean_safe_h_pi_minimum"])
        and int(primary["safe_favorable_tapes"])
        >= int(dev_rule["minimum_favorable_tapes"])
        and int(primary["unique_safe_oracle_calendars"])
        >= int(dev_rule["minimum_unique_safe_oracle_calendars"])
        and float(primary["modal_safe_oracle_fraction"])
        <= float(dev_rule["maximum_modal_safe_oracle_fraction"])
        and int(primary["safe_action_support"]["material_action_levels_at_10pct"])
        >= int(dev_rule["minimum_material_action_levels"])
    )
    ordering_pass = all(
        float(row["safe_h_pi"]) >= float(dev_rule["mean_safe_h_pi_minimum"])
        for row in ordering
    )
    conservation_keys = (
        "mass_residual",
        "partition_residual",
        "aggregate_ration_residual",
        "raw_material_residual",
    )
    conservation_pass = all(
        bool(all(np.all(np.abs(panel[key]) <= 1e-8) for key in conservation_keys))
        for panel in panels.values()
    )
    if stage == "development":
        passed = bool(
            structural_pass
            and ordering_pass
            and conservation_pass
            and parity_pass
            and null_pass
        )
        status = (
            contract["terminal_labels"]["development_pass_pending_custody"]
            if passed
            else contract["terminal_labels"]["stop"]
        )
    else:
        threshold = float(
            contract["validation_pass_rule"][
                "simultaneous_one_sided_lcb95_safe_h_pi_minimum"
            ]
        )
        inference_pass = all(
            float(row["simultaneous_safe_lcb95"]) >= threshold
            for row in [primary, *ordering]
        )
        passed = bool(
            inference_pass
            and structural_pass
            and ordering_pass
            and conservation_pass
            and parity_pass
            and null_pass
        )
        status = (
            contract["terminal_labels"]["validation_pass_pending_custody"]
            if passed
            else contract["terminal_labels"]["stop"]
        )
    result = {
        "schema_version": f"program_o_full_des_hpi_{stage}_v1",
        "generated_at_utc": now_utc(),
        "status": status,
        "passed": passed,
        "stage": stage,
        "run_id": run_id,
        "scientific_commit": commit,
        "contract": str(contract_path.relative_to(ROOT)),
        "contract_sha256": sha256(contract_path),
        "seeds": list(seeds),
        "calendar_count": 65536,
        "profile_count": len(profiles),
        "transducer_calendar_evaluations": len(seeds) * len(profiles) * 65536,
        "direct_parity_episode_count": int(
            sum(row["calendar_count"] for row in parity_rows)
        ),
        "profiles": [summaries[key] for key in sorted(summaries)],
        "parity_pass": parity_pass,
        "conservation_pass": conservation_pass,
        "exact_fungible_null_pass": null_pass,
        "exact_fungible_null_identity": null_identity,
        "structural_pass": structural_pass,
        "ordering_sensitivity_pass": ordering_pass,
        "simultaneous_inference": inference,
        "claim_boundary": {
            "full_des_h_pi_established": False,
            "pending_independent_custody_validation": bool(passed),
            "h_obs_authorized": False,
            "learner_authorized": False,
            "paper2_confirmed": False,
            "paper3_authorized": False,
        },
        "shard_manifest_sha256": digest_json(shard_manifest),
        "direct_parity_sha256": digest_json(parity_rows),
        "preseed_parity": preseed_parity,
        "execution_authorization": authorization,
        "seed_claim_sha256": sha256(seed_claim),
    }
    result["content_sha256"] = digest_json(result)
    result_path = stage_root / "result.json"
    write_json_atomic(result_path, result)
    update_progress(
        progress_path,
        status="COMPLETE" if passed else "TERMINAL_STOP",
        phase="complete",
        result_path=display_path(result_path),
        result_sha256=sha256(result_path),
    )
    checksum_tree(stage_root)
    return result_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("development", "validation"), required=True)
    parser.add_argument(
        "--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1)
    )
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--execution-freeze", type=Path, default=DEFAULT_DEVELOPMENT_FREEZE
    )
    parser.add_argument("--seed-claim", type=Path, required=True)
    parser.add_argument("--development-result", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = execute(
        contract_path=args.contract.resolve(),
        output_root=args.output_root.resolve(),
        stage=str(args.stage),
        workers=int(args.workers),
        run_dir=args.run_dir.resolve(),
        run_id=str(args.run_id),
        execution_freeze=args.execution_freeze.resolve(),
        seed_claim=args.seed_claim.resolve(),
        development_result=(
            args.development_result.resolve()
            if args.development_result is not None
            else None
        ),
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
