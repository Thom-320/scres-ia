#!/usr/bin/env python3
"""Bootstrap the burned-tape Program-H visible-v1 PI relaxation.

Every resample re-solves the tape-contingent convexified perfect-information LP;
it does not bootstrap a policy selected once on the locked block.  The static
ABAB comparator was selected only on the historical calibration block.  This
remains retrospective inference on already burned tapes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.recompute_program_h_visible_v1_frontier import (
    FIELDS,
    ROOT,
    json_sha256,
    solve_guardrailed_pi_relaxation,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def bootstrap(
    arrays: dict[str, np.ndarray],
    comparator_index: int,
    *,
    n_resamples: int,
    seed: int,
    worst_fill_margin: float,
    progress: Path | None = None,
) -> list[float]:
    n_tapes = next(iter(arrays.values())).shape[0]
    comparator = {
        field: values[:, comparator_index] for field, values in arrays.items()
    }
    rng = np.random.default_rng(seed)
    values: list[float] = []
    started = time.perf_counter()
    for index in range(n_resamples):
        sample = rng.integers(0, n_tapes, size=n_tapes)
        sampled_arrays = {field: rows[sample] for field, rows in arrays.items()}
        sampled_comparator = {
            field: rows[sample] for field, rows in comparator.items()
        }
        pi = solve_guardrailed_pi_relaxation(
            sampled_arrays,
            sampled_comparator,
            worst_fill_margin=worst_fill_margin,
        )
        values.append(
            float(
                pi["solver_objective"]
                - sampled_comparator["ret_visible"].mean()
            )
        )
        if progress is not None and (
            index == 0 or (index + 1) % 10 == 0 or index + 1 == n_resamples
        ):
            atomic_json(
                progress,
                {
                    "schema_version": "program_h_visible_v1_bootstrap_progress_v1",
                    "completed": index + 1,
                    "total": n_resamples,
                    "elapsed_seconds": time.perf_counter() - started,
                    "last_delta": values[-1],
                },
            )
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "results/program_h/visible_v1_repair/verdict.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/program_h/visible_v1_repair/bootstrap_inference.json",
    )
    parser.add_argument(
        "--progress",
        type=Path,
        default=ROOT / "outputs/program_h_visible_v1_bootstrap/progress.json",
    )
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260714)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    source = json.loads(args.source.read_text())
    if source.get("schema_version") != "program_h_visible_v1_frontier_repair_v1":
        raise ValueError("unexpected source schema")
    expected_content = source.get("content_sha256")
    unhashed = dict(source)
    unhashed.pop("content_sha256", None)
    if expected_content != json_sha256(unhashed):
        raise ValueError("source content hash mismatch")
    locked = {
        field: np.asarray(source["raw_matrices"]["locked"][field], dtype=float)
        for field in FIELDS
    }
    support = source["guardrailed_comparator"]["support"]
    if len(support) != 1 or not np.isclose(support[0]["weight"], 1.0):
        raise ValueError("bootstrap implementation requires a pure comparator")
    comparator_index = int(support[0]["sequence_index"])
    worst_fill_margin = float(source["contract"]["worst_fill_margin"])
    started = time.perf_counter()
    values = bootstrap(
        locked,
        comparator_index,
        n_resamples=args.resamples,
        seed=args.seed,
        worst_fill_margin=worst_fill_margin,
        progress=args.progress,
    )
    array = np.asarray(values, dtype=float)
    result = {
        "schema_version": "program_h_visible_v1_bootstrap_inference_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {"path": str(args.source), "sha256": sha256(args.source)},
        "scientific_status": "RETROSPECTIVE_BURNED_TAPE_REOPTIMIZED_BOOTSTRAP_NOT_CONFIRMATORY",
        "governing_metric": "ret_excel_visible_v1",
        "comparator_sequence": support[0]["sequence"],
        "n_tapes": int(locked["ret_visible"].shape[0]),
        "resamples": args.resamples,
        "seed": args.seed,
        "point_delta": float(
            source["guardrailed_perfect_information_relaxation"][
                "delta_ret_visible"
            ]
        ),
        "bootstrap_mean": float(array.mean()),
        "ci95": [
            float(np.quantile(array, 0.025)),
            float(np.quantile(array, 0.975)),
        ],
        "fraction_at_or_above_0_01": float(np.mean(array >= 0.01)),
        "method": "Nonparametric tape bootstrap; every draw re-solves the complete 81-sequence convexified perfect-information LP under lost-order, quantity-ReT and worst-CSSU aggregate guardrails.",
        "claim_limit": "Retrospective inference on historical burned tapes. A positive interval is diagnostic H_PI only, not H_obs, learned value or Paper-2 confirmation.",
        "values_sha256": json_sha256(values),
        "elapsed_seconds": time.perf_counter() - started,
    }
    result["content_sha256"] = json_sha256(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
