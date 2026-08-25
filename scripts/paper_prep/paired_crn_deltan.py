#!/usr/bin/env python3
"""Recompute Program O-R Delta_N with paired CRN seed/tape resampling.

This is a read-only verification tool.  It reconstructs the learner, open-loop and
10-configuration classical matrices from the sealed calibration artifact, then repeats
the frozen two-way studentized max-t bootstrap used by the evaluator.  The max-t family
contains the three H_neural cells and the frozen open-loop/guardrail estimands, so the
reported Delta_N LCBs are on the same simultaneous scale as the published values.

The tool deliberately does not open tapes, train a policy, or adjudicate O/O-R/Q.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import numpy as np


CELLS = ("rho75_share90", "rho90_share75", "rho90_share90")
LEARNER_SEEDS = tuple(range(8101, 8111))
CALIBRATION_SEEDS = tuple(range(7480001, 7480049))
GUARDRAIL_KEYS = ("ret_full", "quantity_ret_full", "worst_product_fill")
MATRIX_KEYS = ("ret_visible", *GUARDRAIL_KEYS)
PUBLISHED_ROWS = {
    "rho75_share90": 7,
    "rho90_share75": 8,
    "rho90_share90": 9,
}
DEFAULT_RUN = Path(
    "/home/ubuntu/program_o_runs/"
    "program-o-ret-calibration-v12-20260717/artifacts/calibration/evaluation"
)
DEFAULT_CACHE = Path(
    "results/program_q/power_preopen_v5_20260718/artifacts/classical_10_cache_v1.npz"
)
DEFAULT_PUBLISHED = Path("papers/paper2/results_table.json")
DEFAULT_TERMINAL_AUDIT = Path(
    "research/paper2_exhaustive_search/"
    "program_o_ret_calibration_v12_terminal_audit_20260717.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def encode_calendar(calendar: list[int] | tuple[int, ...]) -> int:
    index = 0
    for action in calendar:
        index = index * 4 + int(action)
    return index


def verify_raw_manifest(run_root: Path) -> dict[str, Any]:
    manifest_path = run_root / "raw_files.sha256"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"raw manifest not found: {manifest_path}")
    manifest: dict[Path, str] = {}
    for line in manifest_path.read_text().splitlines():
        if not line.strip():
            continue
        digest, relative = line.split(maxsplit=1)
        manifest[Path(relative)] = digest
    expected = len(CELLS) * len(CALIBRATION_SEEDS)
    if len(manifest) != expected:
        raise RuntimeError(
            f"raw manifest has {len(manifest)} entries; expected {expected}"
        )
    mismatches: list[str] = []
    for relative, expected_digest in manifest.items():
        path = run_root / relative
        if not path.is_file():
            mismatches.append(f"missing:{relative}")
            continue
        actual = sha256_file(path)
        if actual != expected_digest:
            mismatches.append(f"sha256:{relative}")
    if mismatches:
        raise RuntimeError("raw manifest verification failed: " + ", ".join(mismatches[:8]))
    return {
        "path": str(manifest_path),
        "entries": len(manifest),
        "sha256": sha256_file(manifest_path),
        "verified": True,
    }


def load_matrices(
    *, run_root: Path, calibration: dict[str, Any], cache_path: Path
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """Reconstruct the exact matrices used by the frozen evaluator."""
    with np.load(cache_path) as cache:
        matrices: dict[str, dict[str, dict[str, np.ndarray]]] = {}
        for cell in CELLS:
            cache_seeds = tuple(
                int(value) for value in cache[f"{cell}__tape_seeds"].tolist()
            )
            if cache_seeds != CALIBRATION_SEEDS:
                raise RuntimeError(f"classical cache seed order mismatch for {cell}")
            classical_indices = np.asarray(
                cache[f"{cell}__calendar_indices"], dtype=np.int64
            )
            if classical_indices.shape != (10, len(CALIBRATION_SEEDS)):
                raise RuntimeError(
                    f"classical cache shape mismatch for {cell}: {classical_indices.shape}"
                )

            open_loop = {
                key: np.empty((len(CALIBRATION_SEEDS), 65536), dtype=float)
                for key in MATRIX_KEYS
            }
            learner = {
                key: np.empty((len(LEARNER_SEEDS), len(CALIBRATION_SEEDS)), dtype=float)
                for key in MATRIX_KEYS
            }
            classical = {
                key: np.empty((10, len(CALIBRATION_SEEDS)), dtype=float)
                for key in MATRIX_KEYS
            }
            audits = calibration["trajectory_audits"][cell]
            for tape_index, tape_seed in enumerate(CALIBRATION_SEEDS):
                path = (
                    run_root
                    / "raw_calendar_matrix"
                    / cell
                    / f"tape_{tape_seed}.npz"
                )
                with np.load(path) as raw:
                    for key in MATRIX_KEYS:
                        open_loop[key][tape_index] = raw[key]
                        for learner_index, learner_seed in enumerate(LEARNER_SEEDS):
                            calendar = audits[str(learner_seed)]["calendars"][tape_index]
                            learner[key][learner_index, tape_index] = raw[key][
                                encode_calendar(calendar)
                            ]
                        for config_index in range(10):
                            calendar_index = int(classical_indices[config_index, tape_index])
                            classical[key][config_index, tape_index] = raw[key][
                                calendar_index
                            ]
            matrices[cell] = {
                "learner": learner,
                "open_loop": open_loop,
                "classical": classical,
            }
    return matrices


def simultaneous_bootstrap(
    rows: dict[str, dict[str, dict[str, np.ndarray]]], resamples: int
) -> tuple[list[str], np.ndarray, np.ndarray, float]:
    """Match the evaluator's RNG order while evaluating each cell in vectorized batches."""
    rng = np.random.default_rng(
        int.from_bytes(hashlib.sha256(b"program-o-ret-only-learner-v1").digest()[:8], "big")
    )
    names: list[str] = []
    points: list[float] = []
    estimands_per_cell = 2 + 2 * len(GUARDRAIL_KEYS)
    boot = np.empty((resamples, len(rows) * estimands_per_cell), dtype=float)
    for cell_index, (cell_id, row) in enumerate(rows.items()):
        learner = row["learner"]["ret_visible"]
        open_loop = row["open_loop"]["ret_visible"]
        classical = row["classical"]["ret_visible"]
        open_index = int(np.argmax(open_loop.mean(axis=0)))
        classical_index = int(np.argmax(classical.mean(axis=1)))
        points.extend(
            [
                float(learner.mean() - open_loop[:, open_index].mean()),
                float(learner.mean() - classical[classical_index].mean()),
            ]
        )
        names.extend((f"{cell_id}::H_learned", f"{cell_id}::H_neural"))
        for key in GUARDRAIL_KEYS:
            points.extend(
                [
                    float(
                        row["learner"][key].mean()
                        - row["open_loop"][key][:, open_index].mean()
                    ),
                    float(
                        row["learner"][key].mean()
                        - row["classical"][key][classical_index].mean()
                    ),
                ]
            )
            names.extend((f"{cell_id}::{key}::vs_open_loop", f"{cell_id}::{key}::vs_classical"))
        # The two calls per sample are deliberately kept in this loop: this preserves the
        # frozen RNG stream exactly.  The resulting index matrices are then evaluated in
        # batches.  In particular, no 48 x 65,536 open-loop tensor is materialized.
        tape_samples = np.empty((resamples, learner.shape[1]), dtype=np.int64)
        seed_samples = np.empty((resamples, learner.shape[0]), dtype=np.int64)
        for sample in range(resamples):
            tape_samples[sample] = rng.integers(
                0, learner.shape[1], size=learner.shape[1]
            )
            seed_samples[sample] = rng.integers(
                0, learner.shape[0], size=learner.shape[0]
            )

        # Counts are sufficient for selecting the maxima; they are also the exact
        # sufficient statistic for the mean of a resampled tape panel.
        tape_counts = np.zeros(
            (resamples, learner.shape[1]), dtype=np.int16
        )
        sample_rows = np.arange(resamples)
        for position in range(learner.shape[1]):
            tape_counts[sample_rows, tape_samples[:, position]] += 1

        # Keep the peak temporary below roughly 70 MiB.  The multiplication is the
        # vectorized equivalent of open_loop[tape].mean(axis=0) in the frozen evaluator.
        sampled_open_indices = np.empty(resamples, dtype=np.int64)
        for start in range(0, resamples, 128):
            stop = min(start + 128, resamples)
            sampled_open_indices[start:stop] = np.argmax(
                tape_counts[start:stop] @ open_loop,
                axis=1,
            )
        sampled_classical_indices = np.argmax(
            tape_counts @ classical.T,
            axis=1,
        )
        learner_sampled = np.take_along_axis(
            learner[seed_samples], tape_samples[:, None, :], axis=2
        ).mean(axis=(1, 2))
        offset = cell_index * estimands_per_cell
        boot[:, offset] = learner_sampled - open_loop[
            tape_samples, sampled_open_indices[:, None]
        ].mean(axis=1)
        boot[:, offset + 1] = learner_sampled - classical[
            sampled_classical_indices[:, None], tape_samples
        ].mean(axis=1)
        for key_index, key in enumerate(GUARDRAIL_KEYS):
            learner_guardrail = np.take_along_axis(
                row["learner"][key][seed_samples], tape_samples[:, None, :], axis=2
            ).mean(axis=(1, 2))
            guardrail_offset = offset + 2 + 2 * key_index
            open_guardrail = row["open_loop"][key][
                tape_samples, sampled_open_indices[:, None]
            ].mean(axis=1)
            classical_guardrail = row["classical"][key][
                sampled_classical_indices[:, None], tape_samples
            ].mean(axis=1)
            boot[:, guardrail_offset] = learner_guardrail - open_guardrail
            boot[:, guardrail_offset + 1] = learner_guardrail - classical_guardrail
    point = np.asarray(points)
    se = boot.std(axis=0, ddof=1)
    active = se > 1e-15
    max_t = np.zeros(resamples)
    if np.any(active):
        max_t[:] = np.max((point[active] - boot[:, active]) / se[active], axis=1)
    critical = float(np.quantile(max_t, 0.95))
    return names, point, se, critical


def parse_published_row(value: list[Any]) -> tuple[float, float]:
    estimate_match = re.search(r"est\s+([+-]?\d+(?:\.\d+)?)", str(value[2]))
    lcb_match = re.search(r"LCB95\s*=\s*([+-]?\d+(?:\.\d+)?)", str(value[3]))
    if not estimate_match or not lcb_match:
        raise ValueError(f"cannot parse published row: {value}")
    return float(estimate_match.group(1)), float(lcb_match.group(1))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--published", type=Path, default=DEFAULT_PUBLISHED)
    parser.add_argument("--terminal-audit", type=Path, default=DEFAULT_TERMINAL_AUDIT)
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--tolerance", type=float, default=1e-12)
    args = parser.parse_args()

    run_root = args.run.resolve()
    cache_path = args.cache.resolve()
    published_path = args.published.resolve()
    terminal_path = args.terminal_audit.resolve()
    result_path = run_root / "result.json"
    for path in (result_path, cache_path, published_path, terminal_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    raw_manifest = verify_raw_manifest(run_root)
    calibration = json.loads(result_path.read_text())
    if calibration.get("phase") != "calibration":
        raise RuntimeError("the supplied result is not the calibration phase")
    if tuple(calibration.get("seed_range", [])) != (7480001, 7480048):
        raise RuntimeError("unexpected calibration seed range")
    matrices = load_matrices(
        run_root=run_root, calibration=calibration, cache_path=cache_path
    )
    names, points, ses, critical = simultaneous_bootstrap(matrices, args.resamples)
    lcb = points - critical * ses
    ucb = points + critical * ses
    by_name = {
        name: {
            "estimate": float(points[index]),
            "se": float(ses[index]),
            "lcb95": float(lcb[index]),
            "ucb95_same_critical": float(ucb[index]),
        }
        for index, name in enumerate(names)
    }

    source_inference = calibration["inference"]["estimates"]
    delta_n_names = [f"{cell}::H_neural" for cell in CELLS]
    source_matches = {
        name: {
            "estimate_abs_diff": abs(by_name[name]["estimate"] - source_inference[name]["estimate"]),
            "lcb95_abs_diff": abs(by_name[name]["lcb95"] - source_inference[name]["lcb95"]),
            "passed": (
                abs(by_name[name]["estimate"] - source_inference[name]["estimate"])
                <= args.tolerance
                and abs(by_name[name]["lcb95"] - source_inference[name]["lcb95"])
                <= args.tolerance
            ),
        }
        for name in delta_n_names
    }
    published = json.loads(published_path.read_text())
    published_matches = {}
    for cell in CELLS:
        name = f"{cell}::H_neural"
        expected_estimate, expected_lcb = parse_published_row(
            published["rows"][PUBLISHED_ROWS[cell]]
        )
        published_matches[cell] = {
            "published_estimate": expected_estimate,
            "published_lcb95": expected_lcb,
            "recomputed_estimate": by_name[name]["estimate"],
            "recomputed_lcb95": by_name[name]["lcb95"],
            "estimate_matches_5dp": round(by_name[name]["estimate"], 5)
            == round(expected_estimate, 5),
            "lcb95_matches_5dp": round(by_name[name]["lcb95"], 5)
            == round(expected_lcb, 5),
        }

    terminal_audit = json.loads(terminal_path.read_text())
    published_verdict = terminal_audit["prospective_program_or_calibration_verdict"]
    premium_pass = all(by_name[name]["lcb95"] >= 0.01 for name in delta_n_names)
    recomputed_verdict = (
        "PASS_Q_NEURAL_PREMIUM" if premium_pass else "STOP_CALIBRATION_NOT_ELIGIBLE"
    )
    result = {
        "schema_version": "paper_prep_paired_crn_deltan_v1",
        "scope": "Delta_N_reimplementation_and_published_check_only",
        "claim_boundary": {
            "opens_new_tapes": False,
            "trains_or_scores_new_policies": False,
            "re_adjudicates_O_O_R_Q": False,
        },
        "inputs": {
            "calibration_result": str(result_path),
            "calibration_result_sha256": sha256_file(result_path),
            "raw_manifest": raw_manifest,
            "classical_cache": str(cache_path),
            "classical_cache_sha256": sha256_file(cache_path),
            "published_table": str(published_path),
            "terminal_audit": str(terminal_path),
        },
        "method": {
            "estimand": "Delta_N = learner - max(classical)",
            "pairing": "same tape CRN; learner seed and tape are resampled jointly",
            "point_estimate": "mean of the 10 x 48 paired learner-minus-selected-classical differences",
            "bootstrap": "studentized simultaneous max-t; classical and open-loop maxima reselected inside every resample",
            "max_t_family": "three H_neural cells plus H_learned and three guardrails versus open-loop/classical in each cell",
            "resamples": args.resamples,
            "seed_derivation": "int.from_bytes(SHA256('program-o-ret-only-learner-v1')[:8], 'big')",
            "confidence": 0.95,
            "simultaneous_critical": critical,
        },
        "delta_n": {
            cell: {
                **by_name[f"{cell}::H_neural"],
                "paired_n": len(LEARNER_SEEDS) * len(CALIBRATION_SEEDS),
                "source_result_field": f"inference.estimates.{cell}::H_neural",
                "source_result_match": source_matches[f"{cell}::H_neural"],
                "published_table_row": f"rows[{PUBLISHED_ROWS[cell]}]",
                "published_table_match": published_matches[cell],
            }
            for cell in CELLS
        },
        "verdict_check": {
            "frozen_premium_rule": "LCB95(Delta_N) >= 0.01 in every cell",
            "recomputed_premium_rule_passes": premium_pass,
            "published_terminal_verdict": published_verdict,
            "recomputed_verdict_under_this_gate": recomputed_verdict,
            "verdict_changed": recomputed_verdict != published_verdict,
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
