#!/usr/bin/env python3
"""Reconstruct every historical Program-I Morris/GP evaluation cell.

The original runner retained aggregate verdicts but not the complete cell
history.  This retrospective reconstruction uses the same deterministic design,
same burned 3,000,001+ tape block, and same historical stylized estimator.  It
cannot promote or select a Paper-2 contract and is not the full-DES canonical
ret_excel_visible_v1 aggregator.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.gsa import gp_locate
from supply_chain.headroom_sensitivity import FACTORS, Headroom, headroom_at


ROOT = Path(__file__).resolve().parent.parent
BOUNDS = [(lo, hi, name) for name, (lo, hi) in FACTORS.items()]
NAMES = list(FACTORS)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def scale(unit_x: np.ndarray) -> np.ndarray:
    lo = np.asarray([row[0] for row in BOUNDS], dtype=float)
    hi = np.asarray([row[1] for row in BOUNDS], dtype=float)
    return lo + unit_x * (hi - lo)


def theta_of(x: np.ndarray) -> dict[str, float]:
    return {name: float(x[index]) for index, name in enumerate(NAMES)}


def hdict(value: Headroom) -> dict[str, Any]:
    return {
        "H_PI": value.H_PI,
        "H_obs": value.H_obs,
        "eta": value.eta,
        "n_tapes": value.n,
    }


def reconstruct_morris(
    evaluate: Callable[[dict[str, float]], Headroom],
    *,
    trajectories: int = 8,
    levels: int = 8,
    seed: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return all (k+1)r rows and the exact historical Morris summaries."""
    k = len(BOUNDS)
    rng = np.random.default_rng(seed)
    delta = levels / (2 * (levels - 1))
    effects = {target: [[] for _ in range(k)] for target in ("H_PI", "H_obs")}
    rows: list[dict[str, Any]] = []
    cell_index = 0
    for trajectory in range(trajectories):
        base = rng.integers(0, levels, size=k) / (levels - 1) * (1 - delta)
        permutation = rng.permutation(k)
        unit_x = base.copy()
        theta = theta_of(scale(unit_x))
        current = evaluate(theta)
        rows.append(
            {
                "cell_index": cell_index,
                "trajectory": trajectory,
                "step": 0,
                "changed_factor": None,
                "unit_x": unit_x.tolist(),
                "theta": theta,
                **hdict(current),
            }
        )
        cell_index += 1
        for step, factor_index in enumerate(permutation, start=1):
            next_x = unit_x.copy()
            next_x[factor_index] = min(next_x[factor_index] + delta, 1.0)
            if next_x[factor_index] == unit_x[factor_index]:
                next_x[factor_index] = max(unit_x[factor_index] - delta, 0.0)
            theta = theta_of(scale(next_x))
            following = evaluate(theta)
            denominator = next_x[factor_index] - unit_x[factor_index]
            for target in ("H_PI", "H_obs"):
                effect = (
                    (getattr(following, target) - getattr(current, target))
                    / denominator
                    if denominator != 0
                    else 0.0
                )
                effects[target][factor_index].append(effect)
            rows.append(
                {
                    "cell_index": cell_index,
                    "trajectory": trajectory,
                    "step": step,
                    "changed_factor": NAMES[factor_index],
                    "unit_x": next_x.tolist(),
                    "theta": theta,
                    **hdict(following),
                }
            )
            cell_index += 1
            unit_x, current = next_x, following

    summaries: dict[str, Any] = {}
    for target in ("H_PI", "H_obs"):
        summary = {}
        for factor_index, name in enumerate(NAMES):
            values = np.asarray(effects[target][factor_index], dtype=float)
            summary[name] = {
                "mu": float(values.mean()),
                "mu_star": float(np.abs(values).mean()),
                "sigma": float(values.std()),
            }
        summaries[target] = summary
    return rows, summaries


def summaries_match(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    return all(
        np.isclose(
            actual[target][name][stat],
            expected[f"morris_{target}"][name][stat],
            rtol=0.0,
            atol=1e-15,
        )
        for target in ("H_PI", "H_obs")
        for name in NAMES
        for stat in ("mu", "mu_star", "sigma")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/headroom_gsa/all_cells_reconstruction.json",
    )
    parser.add_argument("--n-tapes", type=int, default=50)
    parser.add_argument("--progress", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite reconstruction: {args.output}")
    started = time.perf_counter()
    morris_verdict_path = ROOT / "results/headroom_gsa/verdict_morris.json"
    gp_verdict_path = ROOT / "results/headroom_gsa/verdict_gp.json"
    oos_path = ROOT / "results/headroom_gsa/oos_guardrail_check.json"
    morris_verdict = json.loads(morris_verdict_path.read_text())
    gp_verdict = json.loads(gp_verdict_path.read_text())
    oos = json.loads(oos_path.read_text())

    progress_state = {"stage": "morris", "completed": 0, "total": 97}

    def write_progress() -> None:
        if args.progress is not None:
            atomic_json(
                args.progress,
                {
                    "schema_version": "program_i_headroom_gsa_reconstruction_progress_v1",
                    **progress_state,
                    "elapsed_seconds": time.perf_counter() - started,
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "output": str(args.output),
                },
            )

    write_progress()

    def evaluate(theta: dict[str, float]) -> Headroom:
        result = headroom_at(theta, n_tapes=args.n_tapes, seed0=3_000_001)
        progress_state["completed"] += 1
        write_progress()
        return result

    morris_rows, morris_summaries = reconstruct_morris(evaluate)
    morris_reproduced = bool(summaries_match(morris_summaries, morris_verdict))

    gp_rows: list[dict[str, Any]] = []
    progress_state["stage"] = "gp_search"

    def evaluate_gp(x: np.ndarray) -> float:
        theta = theta_of(x)
        result = evaluate(theta)
        gp_rows.append(
            {
                "evaluation_index": len(gp_rows),
                "theta": theta,
                **hdict(result),
            }
        )
        return result.H_obs

    gp_result = gp_locate(
        evaluate_gp,
        BOUNDS,
        n_init=16,
        n_iter=24,
        seed=3,
    )
    gp_theta = theta_of(np.asarray(gp_result["x_best"], dtype=float))
    progress_state["stage"] = "located_confirmation"
    located = headroom_at(gp_theta, n_tapes=200, seed0=3_000_001)
    progress_state["completed"] += 1
    write_progress()
    stored_gp = gp_verdict["gp_locate_H_obs"]
    gp_search_reproduced = bool(
        gp_result["n_eval"] == stored_gp["n_eval"] == 40
        and np.allclose(gp_result["x_best"], stored_gp["x_best"], rtol=0, atol=1e-15)
        and np.isclose(gp_result["y_best"], stored_gp["y_best"], rtol=0, atol=1e-15)
    )
    located_reproduced = bool(all(
        np.isclose(
            getattr(located, key),
            gp_verdict["located_region_headroom"][key],
            rtol=0,
            atol=1e-15,
        )
        for key in ("H_PI", "H_obs", "eta")
    ))
    passed = bool(
        morris_reproduced and gp_search_reproduced and located_reproduced
    )
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    payload = {
        "schema_version": "program_i_headroom_gsa_all_cells_reconstruction_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_head": head,
        "scientific_status": "RETROSPECTIVE_COMPLETE_CELL_REPORT_NONCANONICAL_STYLIZED_GUARDRAIL_FAIL",
        "historical_estimator": "program_g stylized ret_order over emitted simulated orders; not full-DES ret_excel_visible_v1",
        "seed_status": "burned historical 3,000,001+ blocks only; no virgin or locked Paper-2 tapes opened",
        "claim_limit": "Reconstructs historical GSA cell history and aggregate verdicts only. It cannot promote a contract, prove project H_PI/H_obs, change the fairness failure, or select new parameters.",
        "inputs": {
            "morris_verdict": {"path": str(morris_verdict_path.relative_to(ROOT)), "sha256": sha256(morris_verdict_path)},
            "gp_verdict": {"path": str(gp_verdict_path.relative_to(ROOT)), "sha256": sha256(gp_verdict_path)},
            "oos_guardrail": {"path": str(oos_path.relative_to(ROOT)), "sha256": sha256(oos_path)},
            "headroom_source_sha256": sha256(ROOT / "supply_chain/headroom_sensitivity.py"),
            "gsa_source_sha256": sha256(ROOT / "supply_chain/gsa.py"),
        },
        "morris": {
            "n_tapes_per_cell": args.n_tapes,
            "seed_start": 3_000_001,
            "trajectory_count": 8,
            "cell_evaluations": len(morris_rows),
            "rows": morris_rows,
            "rows_sha256": json_sha256(morris_rows),
            "summaries": morris_summaries,
            "aggregate_verdict_reproduced": morris_reproduced,
        },
        "gp": {
            "n_tapes_per_search_cell": args.n_tapes,
            "seed_start": 3_000_001,
            "cell_evaluations": len(gp_rows),
            "rows": gp_rows,
            "rows_sha256": json_sha256(gp_rows),
            "recomputed_search": gp_result,
            "search_verdict_reproduced": gp_search_reproduced,
            "located_confirmation": hdict(located),
            "located_confirmation_reproduced": located_reproduced,
        },
        "oos_guardrail_blocks": oos["blocks"],
        "oos_verdict": oos["verdict"],
        "passed": passed,
        "elapsed_seconds": time.perf_counter() - started,
    }
    payload["content_sha256"] = json_sha256(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    progress_state["stage"] = "complete"
    if args.progress is not None:
        atomic_json(
            args.progress,
            {
                "schema_version": "program_i_headroom_gsa_reconstruction_progress_v1",
                **progress_state,
                "elapsed_seconds": payload["elapsed_seconds"],
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                "output": str(args.output),
                "output_sha256": sha256(args.output),
            },
        )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "passed": passed,
                "morris_cells": len(morris_rows),
                "gp_cells": len(gp_rows),
                "elapsed_seconds": payload["elapsed_seconds"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
