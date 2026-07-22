#!/usr/bin/env python3
"""Create a compact, auditable summary of Program Q's exact static frontier.

The raw calibration matrix contains 144 files (three cells by 48 tapes),
each with all 65,536 open-loop calendars.  The raw bundle is too large for
the review branch, so this script verifies every source hash and writes the
per-calendar cross-tape means needed by Submission A's frontier figure.
It does not run a simulator or change any scientific estimate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


CELLS = ("rho75_share90", "rho90_share75", "rho90_share90")
EXPECTED_TAPES = 48
EXPECTED_CALENDARS = 65_536
EXPECTED_MANIFEST_SHA256 = (
    "1b46adf5439370bb8f99c5f607feb0aed66d8c13e3d835f017854b6ca63b5f99"
)
METRICS = ("ret_visible", "worst_product_fill", "unresolved_orders")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_manifest(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split(maxsplit=1)
        entries[relative.strip()] = digest
    return entries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists() or args.manifest_output.exists():
        raise FileExistsError("Refusing to overwrite an existing derived artifact.")

    source_manifest = args.calibration_root / "raw_files.sha256"
    if sha256(source_manifest) != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("The calibration raw-file manifest is not the frozen manifest.")
    expected = parse_manifest(source_manifest)
    if len(expected) != len(CELLS) * EXPECTED_TAPES:
        raise RuntimeError(f"Expected 144 manifest entries, found {len(expected)}.")

    arrays: dict[str, np.ndarray] = {
        "calendar_index": np.arange(EXPECTED_CALENDARS, dtype=np.int32)
    }
    verified_files: list[dict[str, str]] = []
    for cell in CELLS:
        files = sorted((args.calibration_root / "raw_calendar_matrix" / cell).glob("*.npz"))
        if len(files) != EXPECTED_TAPES:
            raise RuntimeError(f"{cell}: expected 48 tapes, found {len(files)}.")
        by_metric = {metric: [] for metric in METRICS}
        for path in files:
            relative = path.relative_to(args.calibration_root).as_posix()
            actual = sha256(path)
            if expected.get(relative) != actual:
                raise RuntimeError(f"Hash mismatch: {relative}")
            verified_files.append({"path": relative, "sha256": actual})
            with np.load(path, allow_pickle=False) as payload:
                for metric in METRICS:
                    values = np.asarray(payload[metric], dtype=np.float64)
                    if values.shape != (EXPECTED_CALENDARS,):
                        raise RuntimeError(f"{relative}:{metric} has shape {values.shape}.")
                    by_metric[metric].append(values)
        for metric, rows in by_metric.items():
            matrix = np.stack(rows, axis=0)
            arrays[f"{cell}__{metric}__mean"] = matrix.mean(axis=0)
            arrays[f"{cell}__{metric}__q10"] = np.quantile(matrix, 0.10, axis=0)
            arrays[f"{cell}__{metric}__q90"] = np.quantile(matrix, 0.90, axis=0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    manifest = {
        "schema_version": "submission_a_static_frontier_summary_v1",
        "status": "DERIVED_FROM_FROZEN_CALIBRATION_RAW_MATRIX",
        "scientific_result_regenerated": False,
        "cells": list(CELLS),
        "tapes_per_cell": EXPECTED_TAPES,
        "calendar_count": EXPECTED_CALENDARS,
        "metrics": list(METRICS),
        "source_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "verified_file_count": len(verified_files),
        "verified_files": verified_files,
        "derived_npz": args.output.name,
        "derived_npz_sha256": sha256(args.output),
    }
    args.manifest_output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
