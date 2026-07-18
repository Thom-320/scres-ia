#!/usr/bin/env python3
"""Power analysis for Program Q using burned Program O-R calibration data only.

The full open-loop frontier and all ten classical controllers are reselected in
every bootstrap replicate.  No 749 or 950 tape is generated or opened here.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_program_o_ret_learner import encode_calendar, scheduler  # noqa: E402
from supply_chain.program_o_full_des_transducer import extract_full_des_skeleton  # noqa: E402
from supply_chain.program_o_ret_env import CONFIRMED_RET_CELLS  # noqa: E402
from supply_chain.program_o_state_rich import (  # noqa: E402
    finite_state_rich_configurations,
    state_rich_calendar,
)


GRID = (128, 160, 192, 256)
CELL_IDS = tuple(cell.cell_id for cell in CONFIRMED_RET_CELLS)


def normalized_counts(indices: np.ndarray, size: int) -> np.ndarray:
    rows = np.zeros((indices.shape[0], size), dtype=np.float64)
    for row, values in enumerate(indices):
        rows[row] = np.bincount(values, minlength=size) / values.size
    return rows


def bootstrap_effects(
    panels: dict[str, dict[str, np.ndarray]],
    *,
    tape_count: int,
    replicates: int,
    rng: np.random.Generator,
    batch_size: int = 32,
) -> np.ndarray:
    """Return columns H_OL(cell...), Delta_N(cell...) with comparator reselection."""
    first = panels[CELL_IDS[0]]
    seed_count, original_tapes = first["learner"].shape
    output = np.empty((replicates, len(CELL_IDS) * 2), dtype=np.float64)
    for start in range(0, replicates, batch_size):
        stop = min(start + batch_size, replicates)
        width = stop - start
        tape_indices = rng.integers(0, original_tapes, size=(width, tape_count))
        seed_indices = rng.integers(0, seed_count, size=(width, seed_count))
        tape_weights = normalized_counts(tape_indices, original_tapes)
        seed_weights = normalized_counts(seed_indices, seed_count)
        for cell_index, cell_id in enumerate(CELL_IDS):
            panel = panels[cell_id]
            learner_mean = np.einsum(
                "bs,st,bt->b", seed_weights, panel["learner"], tape_weights, optimize=True
            )
            open_means = tape_weights @ panel["open_loop"]
            classical_means = tape_weights @ panel["classical"].T
            output[start:stop, cell_index] = learner_mean - open_means.max(axis=1)
            output[start:stop, len(CELL_IDS) + cell_index] = (
                learner_mean - classical_means.max(axis=1)
            )
    return output


def point_effects(panels: dict[str, dict[str, np.ndarray]]) -> np.ndarray:
    values = []
    for comparator in ("open_loop", "classical"):
        for cell_id in CELL_IDS:
            panel = panels[cell_id]
            learner = float(panel["learner"].mean())
            if comparator == "open_loop":
                benchmark = float(panel["open_loop"].mean(axis=0).max())
            else:
                benchmark = float(panel["classical"].mean(axis=1).max())
            values.append(learner - benchmark)
    return np.asarray(values)


def simultaneous_critical(
    panels: dict[str, dict[str, np.ndarray]],
    *,
    replicates: int,
    rng: np.random.Generator,
) -> tuple[float, np.ndarray]:
    draws = bootstrap_effects(
        panels, tape_count=48, replicates=replicates, rng=rng
    )
    points = point_effects(panels)
    standard_errors = draws.std(axis=0, ddof=1)
    if np.any(standard_errors <= 0):
        raise RuntimeError("non-positive bootstrap standard error")
    maximum_t = np.max(np.abs((draws - points) / standard_errors), axis=1)
    return float(np.quantile(maximum_t, 0.95)), standard_errors


def projected_power(
    panels: dict[str, dict[str, np.ndarray]],
    *,
    tape_count: int,
    replicates: int,
    critical: float,
    rng: np.random.Generator,
) -> dict[str, float]:
    draws = bootstrap_effects(
        panels, tape_count=tape_count, replicates=replicates, rng=rng
    )
    standard_errors = draws.std(axis=0, ddof=1)
    h = draws[:, : len(CELL_IDS)]
    delta = draws[:, len(CELL_IDS) :]
    h_pass = np.all(h - critical * standard_errors[: len(CELL_IDS)] >= 0.01, axis=1)
    delta_se = standard_errors[len(CELL_IDS) :]
    equivalence_pass = np.all(
        (delta - critical * delta_se >= -0.01)
        & (delta + critical * delta_se <= 0.01),
        axis=1,
    )
    return {
        "H_OL": float(h_pass.mean()),
        "Delta_N_equivalence": float(equivalence_pass.mean()),
        "joint": float((h_pass & equivalence_pass).mean()),
    }


def _classical_indices_for_tape(task: tuple[int, int]) -> tuple[int, int, list[int]]:
    cell_index, tape_seed = task
    cell = CONFIRMED_RET_CELLS[cell_index]
    skeleton, _ = extract_full_des_skeleton(
        seed=tape_seed,
        scheduler=scheduler(),
        regime_persistence=cell.regime_persistence,
        dominant_share=cell.dominant_share,
        downstream_freight_physics_mode="fixed_clock_physical_v1",
    )
    indices = []
    for config in finite_state_rich_configurations():
        calendar, _ = state_rich_calendar(
            skeleton=skeleton.as_dict(),
            scheduler=scheduler(),
            config=config,
            regime_persistence=0.75,
            dominant_share=0.90,
        )
        indices.append(encode_calendar(tuple(calendar)))
    return cell_index, tape_seed, indices


def _write_classical_shard(
    shards_dir: Path, cell_index: int, tape_seed: int, indices: list[int]
) -> None:
    destination = shards_dir / f"{CELL_IDS[cell_index]}__tape_{tape_seed}.npz"
    temporary = destination.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            cell_index=np.asarray(cell_index, dtype=np.int64),
            tape_seed=np.asarray(tape_seed, dtype=np.int64),
            calendar_indices=np.asarray(indices, dtype=np.int64),
        )
    temporary.replace(destination)


def _load_classical_shard(
    path: Path, *, expected_cell_index: int, expected_tape_seed: int
) -> list[int]:
    with np.load(path, allow_pickle=False) as payload:
        cell_index = int(payload["cell_index"])
        tape_seed = int(payload["tape_seed"])
        indices = payload["calendar_indices"].astype(np.int64)
    expected_configs = len(finite_state_rich_configurations())
    if cell_index != expected_cell_index or tape_seed != expected_tape_seed:
        raise RuntimeError(f"classical shard identity mismatch: {path}")
    if indices.shape != (expected_configs,):
        raise RuntimeError(f"classical shard is incomplete: {path}")
    return indices.tolist()


def build_classical_cache(
    run: Path,
    output: Path,
    workers: int,
    *,
    shards_dir: Path,
    max_tasks_per_child: int,
) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    tasks = []
    tape_seeds_by_cell = {}
    for cell_index, cell in enumerate(CONFIRMED_RET_CELLS):
        paths = sorted(
            (run / "raw_calendar_matrix" / cell.cell_id).glob("tape_*.npz"),
            key=lambda path: int(path.stem.split("_")[-1]),
        )
        seeds = [int(path.stem.split("_")[-1]) for path in paths]
        tape_seeds_by_cell[cell.cell_id] = seeds
        tasks.extend((cell_index, seed) for seed in seeds)
    shards_dir.mkdir(parents=True, exist_ok=True)
    rows: dict[str, dict[int, list[int]]] = {cell: {} for cell in CELL_IDS}
    pending = []
    for cell_index, tape_seed in tasks:
        shard = shards_dir / f"{CELL_IDS[cell_index]}__tape_{tape_seed}.npz"
        if shard.exists():
            rows[CELL_IDS[cell_index]][tape_seed] = _load_classical_shard(
                shard,
                expected_cell_index=cell_index,
                expected_tape_seed=tape_seed,
            )
        else:
            pending.append((cell_index, tape_seed))
    with ProcessPoolExecutor(
        max_workers=workers, max_tasks_per_child=max_tasks_per_child
    ) as executor:
        futures = {
            executor.submit(_classical_indices_for_tape, task): task for task in pending
        }
        for future in as_completed(futures):
            cell_index, tape_seed, indices = future.result()
            _write_classical_shard(shards_dir, cell_index, tape_seed, indices)
            rows[CELL_IDS[cell_index]][tape_seed] = indices
    arrays = {}
    for cell_id in CELL_IDS:
        arrays[f"{cell_id}__tape_seeds"] = np.asarray(tape_seeds_by_cell[cell_id], dtype=np.int64)
        arrays[f"{cell_id}__calendar_indices"] = np.asarray(
            [rows[cell_id][seed] for seed in tape_seeds_by_cell[cell_id]], dtype=np.int64
        ).T
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)


def load_panels(run: Path, classical_cache: Path) -> dict[str, dict[str, np.ndarray]]:
    result = json.loads((run / "result.json").read_text())
    configs = finite_state_rich_configurations()
    if not classical_cache.exists():
        raise FileNotFoundError(
            f"complete 10-policy cache is required: {classical_cache}; "
            "run with --build-classical-cache on Kaggle/VPS first"
        )
    cached = np.load(classical_cache, allow_pickle=False)
    panels: dict[str, dict[str, np.ndarray]] = {}
    for cell_index, cell in enumerate(CONFIRMED_RET_CELLS):
        tape_paths = sorted(
            (run / "raw_calendar_matrix" / cell.cell_id).glob("tape_*.npz"),
            key=lambda path: int(path.stem.split("_")[-1]),
        )
        if len(tape_paths) != 48:
            raise RuntimeError(f"{cell.cell_id}: expected 48 burned matrices")
        frontier = np.stack(
            [np.load(path, allow_pickle=False)["ret_visible"] for path in tape_paths]
        )
        tape_seeds = [int(path.stem.split("_")[-1]) for path in tape_paths]
        cached_seeds = cached[f"{cell.cell_id}__tape_seeds"].astype(int).tolist()
        if cached_seeds != tape_seeds:
            raise RuntimeError(f"{cell.cell_id}: classical cache tape order mismatch")
        calendar_indices = cached[f"{cell.cell_id}__calendar_indices"]
        if calendar_indices.shape != (len(configs), len(tape_paths)):
            raise RuntimeError(f"{cell.cell_id}: incomplete classical cache")
        classical = np.stack(
            [frontier[tape_index, calendar_indices[:, tape_index]] for tape_index in range(48)],
            axis=1,
        )
        audits = result["trajectory_audits"][cell.cell_id]
        learner = np.stack(
            [
                [
                    frontier[tape_index, encode_calendar(tuple(calendar))]
                    for tape_index, calendar in enumerate(audits[seed]["calendars"])
                ]
                for seed in sorted(audits, key=int)
            ]
        )
        panels[cell.cell_id] = {
            "learner": learner,
            "open_loop": frontier,
            "classical": classical,
        }
    return panels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        type=Path,
        default=ROOT / "results/program_o/ret_only_learner_v1/calibration_run",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--classical-cache",
        type=Path,
        default=ROOT / "research/paper2_exhaustive_search/program_q_classical_10_cache_v1.npz",
    )
    parser.add_argument("--build-classical-cache", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--classical-cache-shards-dir",
        type=Path,
        help="resumable per-tape cache shards (defaults beside the final cache)",
    )
    parser.add_argument(
        "--max-tasks-per-child",
        type=int,
        default=1,
        help="recycle cache workers to bound full-DES skeleton memory",
    )
    parser.add_argument("--critical-resamples", type=int, default=5_000)
    parser.add_argument("--power-resamples", type=int, default=2_000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.build_classical_cache:
        shards_dir = args.classical_cache_shards_dir or args.classical_cache.with_name(
            f"{args.classical_cache.stem}_shards"
        )
        build_classical_cache(
            args.run,
            args.classical_cache,
            args.workers,
            shards_dir=shards_dir,
            max_tasks_per_child=args.max_tasks_per_child,
        )
    panels = load_panels(args.run, args.classical_cache)
    rng = np.random.default_rng(20260717)
    critical, base_se = simultaneous_critical(
        panels, replicates=args.critical_resamples, rng=rng
    )
    rows = {}
    selected = None
    for tape_count in GRID:
        rows[str(tape_count)] = projected_power(
            panels,
            tape_count=tape_count,
            replicates=args.power_resamples,
            critical=critical,
            rng=rng,
        )
        if selected is None and all(value >= 0.80 for value in rows[str(tape_count)].values()):
            selected = tape_count
    payload = {
        "schema_version": "program_q_power_analysis_v1",
        "status": "BURNED_DATA_ONLY",
        "749_or_950_seed_opened": False,
        "comparator_reselection": {
            "open_loop_65536": True,
            "classical_10": True,
            "inside_every_resample": True,
        },
        "cell_ids": list(CELL_IDS),
        "point_effects": point_effects(panels).tolist(),
        "simultaneous_critical": critical,
        "base_standard_errors": base_se.tolist(),
        "grid": list(GRID),
        "power": rows,
        "selected_N": selected,
        "verdict": (
            f"SELECT_N_{selected}" if selected is not None
            else "STOP_PROGRAM_Q_UNDERPOWERED_WITHIN_CAP"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
