#!/usr/bin/env python3
"""Finite calibration-only search for a constant full Track B policy.

Protocol: one 128-point Sobol global screen on calibration tapes, followed by
one deterministic local refinement around the eight best screen candidates.
The final candidate is selected on calibration tapes only and serialized for a
separate held-out evaluation.  No test tape is accepted by this script.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import qmc

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_track_b_crossed_eval import CANONICAL_ENV_KWARGS, episode_metrics_row  # noqa: E402
from supply_chain.external_env_interface import make_track_b_env  # noqa: E402

CALIBRATION_MIN = 300_001
CALIBRATION_MAX = 300_024


@dataclass(frozen=True)
class Candidate:
    signals: tuple[float, float, float, float, float, float, float]
    shift: int

    def action(self) -> list[float]:
        s = self.signals
        shift_signal = {1: -0.67, 2: 0.0, 3: 0.67}[self.shift]
        return [s[0], s[1], s[2], s[3], s[4], shift_signal, s[5], s[6]]

    def key(self) -> tuple[Any, ...]:
        return (*[round(v, 8) for v in self.signals], self.shift)


def evaluate_candidate(task: tuple[int, Candidate, list[int]]) -> dict[str, Any]:
    candidate_id, candidate, tapes = task
    values: dict[str, list[float]] = {}
    action = np.asarray(candidate.action(), dtype=np.float32)
    for tape in tapes:
        env = make_track_b_env(**CANONICAL_ENV_KWARGS)
        env.reset(seed=tape)
        terminated = truncated = False
        while not (terminated or truncated):
            _o, _r, terminated, truncated, _i = env.step(action)
        row = episode_metrics_row(env.unwrapped.sim)
        env.close()
        for key, value in row.items():
            values.setdefault(key, []).append(float(value))
    return {
        "candidate_id": candidate_id,
        "shift": candidate.shift,
        **{f"signal_{i}": v for i, v in enumerate(candidate.signals)},
        **{f"mean_{key}": float(np.mean(v)) for key, v in values.items()},
        "n_tapes": len(tapes),
    }


def global_candidates(n: int, seed: int) -> list[Candidate]:
    if n <= 0 or n & (n - 1):
        raise ValueError("--global-candidates must be a positive power of two")
    raw = qmc.Sobol(d=8, scramble=True, seed=seed).random_base2(int(np.log2(n)))
    candidates = []
    for row in raw:
        signals = tuple(float(v) for v in (2.0 * row[:7] - 1.0))
        shift = min(3, int(row[7] * 3) + 1)
        candidates.append(Candidate(signals=signals, shift=shift))
    # Ensure the prespecified dense-frontier winner is in the screen.
    candidates.append(Candidate((-1 / 3, -1 / 3, -1 / 3, -1 / 3, 0.0, 1.0, 1 / 3), 2))
    return candidates


def refinement_candidates(leaders: list[Candidate], radius: float) -> list[Candidate]:
    candidates: dict[tuple[Any, ...], Candidate] = {}
    for leader in leaders:
        candidates[leader.key()] = leader
        for dim in range(7):
            for direction in (-1.0, 1.0):
                values = list(leader.signals)
                values[dim] = float(np.clip(values[dim] + direction * radius, -1.0, 1.0))
                c = Candidate(tuple(values), leader.shift)
                candidates[c.key()] = c
        for shift in (1, 2, 3):
            c = Candidate(leader.signals, shift)
            candidates[c.key()] = c
    return list(candidates.values())


def run_batch(candidates: list[Candidate], tapes: list[int], workers: int) -> list[dict[str, Any]]:
    tasks = [(i, c, tapes) for i, c in enumerate(candidates)]
    if workers == 1:
        return [evaluate_candidate(t) for t in tasks]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(evaluate_candidate, tasks, chunksize=1))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def candidate_from_row(row: dict[str, Any]) -> Candidate:
    return Candidate(tuple(float(row[f"signal_{i}"]) for i in range(7)), int(row["shift"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--global-candidates", type=int, default=128)
    parser.add_argument("--global-tapes", type=int, default=12)
    parser.add_argument("--refine-top", type=int, default=8)
    parser.add_argument("--refine-radius", type=float, default=0.15)
    parser.add_argument("--calibration-seed-base", type=int, default=CALIBRATION_MIN)
    parser.add_argument("--calibration-tapes", type=int, default=24)
    parser.add_argument("--sobol-seed", type=int, default=20260710)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    tapes = list(range(args.calibration_seed_base, args.calibration_seed_base + args.calibration_tapes))
    if not tapes or min(tapes) < CALIBRATION_MIN or max(tapes) > CALIBRATION_MAX:
        raise SystemExit(f"calibration tapes must remain in {CALIBRATION_MIN}..{CALIBRATION_MAX}")
    if args.global_tapes > len(tapes):
        raise SystemExit("--global-tapes cannot exceed --calibration-tapes")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    screen_candidates = global_candidates(args.global_candidates, args.sobol_seed)
    print(f"global: {len(screen_candidates)} candidates x {args.global_tapes} tapes", flush=True)
    screen = run_batch(screen_candidates, tapes[: args.global_tapes], args.workers)
    screen.sort(key=lambda r: r["mean_ret_excel"], reverse=True)
    write_rows(args.output_dir / "global_screen.csv", screen)

    leaders = [candidate_from_row(r) for r in screen[: args.refine_top]]
    refined_candidates = refinement_candidates(leaders, args.refine_radius)
    print(f"refinement: {len(refined_candidates)} candidates x {len(tapes)} tapes", flush=True)
    refined = run_batch(refined_candidates, tapes, args.workers)
    refined.sort(key=lambda r: r["mean_ret_excel"], reverse=True)
    write_rows(args.output_dir / "refinement.csv", refined)

    winner = candidate_from_row(refined[0])
    frozen = {
        "protocol": "one Sobol global screen plus one local refinement; calibration only",
        "calibration_tapes": tapes,
        "primary_endpoint": "ret_excel",
        "candidate": {"signals": winner.action(), "shift": winner.shift},
        "calibration_metrics": {k: v for k, v in refined[0].items() if k.startswith("mean_")},
        "config": vars(args) | {"output_dir": str(args.output_dir)},
    }
    (args.output_dir / "frozen_static_policy.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    print(json.dumps(frozen, indent=2), flush=True)


if __name__ == "__main__":
    main()
