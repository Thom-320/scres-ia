#!/usr/bin/env python3
"""Verify, reduce, and adjudicate the frozen Program S S1 native screen.

The command is intentionally unusable on a partial run.  It first verifies the
terminal producer receipt and every manifest identity/hash, then validates all
480 point panels and applies the prospectively frozen simultaneous H_PI rule.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_program_s_s1_native import EXPECTED_SHARDS, shard_path, tasks  # noqa: E402
from scripts.run_program_s_s1_shard import make_cell, resolve_point  # noqa: E402
from scripts.screen_program_o_full_des_hpi import (  # noqa: E402
    bootstrap_counts,
    bootstrap_profile,
    profile_summary,
)
from supply_chain.program_o_full_des_transducer import MATRIX_KEYS  # noqa: E402


FREEZE_PATH = (
    ROOT
    / "research/paper2_exhaustive_search/program_s_s1_reduction_freeze_v1_1.json"
)
DESIGN_PATH = (
    ROOT
    / "research/paper2_exhaustive_search/program_s_native_morris_design_v1_1.json"
)
CONTRACT_PATH = ROOT / "contracts/program_s_product_mix_risk_interaction_gsa_v1.json"
SEEDS = tuple(range(7_510_001, 7_510_013))
EXTRA_KEYS = (
    "classical_calendar_index",
    "classical_calendar",
    "oracle_calendar_index",
    "risk_event_tape_sha256",
    "base_stream_sha256",
    "skeleton_sha256",
    "cell_id",
    "observation_sha256",
    "direct_replay_max_abs_error",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _scalar_text(value: np.ndarray) -> str:
    array = np.asarray(value)
    if array.shape != ():
        raise AssertionError("expected scalar string metadata")
    return str(array.item())


def summarize_point(
    *,
    group: int,
    trajectory: int,
    point: int,
    product_cell: str,
    output_root: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Load, validate, and summarize one frozen 12-tape S1 point."""
    group_row, point_row = resolve_point(group, trajectory, point)
    expected_cell = make_cell(group_row, point_row, product_cell)
    prefix = (
        f"g{group:02d}__t{trajectory:02d}__p{point:02d}"
        f"__{product_cell}__seed"
    )
    paths = [output_root / "matrices" / f"{prefix}{seed}.npz" for seed in SEEDS]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} S1 shards")

    panel = {key: [] for key in MATRIX_KEYS}
    classical_indices: list[int] = []
    shard_rows: list[dict[str, Any]] = []
    expected_files = set(MATRIX_KEYS) | set(EXTRA_KEYS)
    for seed, path in zip(SEEDS, paths, strict=True):
        with np.load(path, allow_pickle=False) as shard:
            if set(shard.files) != expected_files:
                raise AssertionError(f"S1 shard schema drift: {path}")
            for key in MATRIX_KEYS:
                values = np.asarray(shard[key])
                if values.shape != (65_536,) or not np.all(np.isfinite(values)):
                    raise AssertionError(f"invalid {key} matrix in {path}")
                panel[key].append(values)
            cell_id = _scalar_text(shard["cell_id"])
            if cell_id != expected_cell.cell_id:
                raise AssertionError(f"cell identity mismatch in {path}")
            classical_index = int(np.asarray(shard["classical_calendar_index"]).item())
            oracle_index = int(np.asarray(shard["oracle_calendar_index"]).item())
            if not 0 <= classical_index < 65_536 or not 0 <= oracle_index < 65_536:
                raise AssertionError(f"calendar index out of range in {path}")
            calendar = np.asarray(shard["classical_calendar"])
            if calendar.shape != (8,) or np.any((calendar < 0) | (calendar > 3)):
                raise AssertionError(f"invalid classical calendar in {path}")
            reconstructed = 0
            for action in calendar:
                reconstructed = reconstructed * 4 + int(action)
            if reconstructed != classical_index:
                raise AssertionError(f"classical calendar/index mismatch in {path}")
            if oracle_index != int(np.argmax(np.asarray(shard["ret_visible"]))):
                raise AssertionError(f"oracle index mismatch in {path}")
            observations = np.asarray(shard["observation_sha256"])
            if observations.shape != (8,) or any(len(str(value)) != 64 for value in observations):
                raise AssertionError(f"invalid observation hashes in {path}")
            hashes = {
                key: _scalar_text(shard[key])
                for key in (
                    "risk_event_tape_sha256",
                    "base_stream_sha256",
                    "skeleton_sha256",
                )
            }
            if any(len(value) != 64 for value in hashes.values()):
                raise AssertionError(f"invalid scientific hash in {path}")
            replay_error = float(np.asarray(shard["direct_replay_max_abs_error"]).item())
            if not np.isfinite(replay_error) or replay_error > 1e-10:
                raise AssertionError(f"direct replay failed in {path}: {replay_error}")
            classical_indices.append(classical_index)
            shard_rows.append(
                {
                    "seed": seed,
                    "sha256": sha256(path),
                    "risk_event_tape_sha256": hashes["risk_event_tape_sha256"],
                    "base_stream_sha256": hashes["base_stream_sha256"],
                    "skeleton_sha256": hashes["skeleton_sha256"],
                    "direct_replay_max_abs_error": replay_error,
                }
            )

    stacked = {key: np.stack(values) for key, values in panel.items()}
    summary = profile_summary(stacked, json.loads(CONTRACT_PATH.read_text()))
    tapes = np.arange(len(paths))
    classical_values = stacked["ret_visible"][tapes, np.asarray(classical_indices)]
    static_values = stacked["ret_visible"][:, summary["best_static_calendar_index"]]
    deltas = classical_values - static_values
    summary.update(
        stratum=str(group_row["stratum"]),
        mask=str(group_row["mask"]),
        group=int(group),
        trajectory=int(trajectory),
        point=int(point),
        product_cell=str(product_cell),
        cell_id=expected_cell.cell_id,
        physical=point_row["physical"],
        classical_policy="belief_mpc_h4_no_alarm",
        classical_calendar_indices=classical_indices,
        classical_h_obs=float(deltas.mean()),
        classical_h_obs_per_tape=deltas.tolist(),
        classical_favorable_tapes=int(np.sum(deltas > 1e-15)),
        eta=(
            float(deltas.mean()) / float(summary["safe_h_pi"])
            if float(summary["safe_h_pi"]) > 0.0
            else 0.0
        ),
        shard_identity_sha256=hashlib.sha256(
            json.dumps(shard_rows, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        maximum_direct_replay_abs_error=max(
            row["direct_replay_max_abs_error"] for row in shard_rows
        ),
    )
    return summary, stacked


def expected_relative_paths() -> set[str]:
    return {
        str(shard_path(Path("."), task)).removeprefix("./")
        for task in tasks()
    }


def verify_run_custody(output_root: Path) -> dict[str, Any]:
    if not output_root.is_dir():
        raise FileNotFoundError(output_root)
    launch_path = output_root / "launch_manifest.json"
    if not launch_path.is_file():
        raise FileNotFoundError(f"incomplete S1 custody: {launch_path}")
    custody_dirs = [output_root] + sorted(
        (output_root / "resume_attempts").glob("attempt-*")
    )
    complete_receipts: list[tuple[Path, Path, dict[str, Any]]] = []
    for custody_dir in custody_dirs:
        candidate = custody_dir / "producer_exit.json"
        if candidate.is_file():
            receipt = json.loads(candidate.read_text())
            if receipt.get("status") == "COMPLETE":
                complete_receipts.append((custody_dir, candidate, receipt))
    if len(complete_receipts) != 1:
        raise FileNotFoundError(
            "incomplete/ambiguous S1 custody: expected exactly one COMPLETE receipt"
        )
    custody_dir, exit_path, exit_receipt = complete_receipts[0]
    manifest_path = custody_dir / "shard_files.sha256"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"incomplete S1 custody: {manifest_path}")
    if (
        int(exit_receipt.get("expected_shards", -1)) != EXPECTED_SHARDS
        or int(exit_receipt.get("completed_shards", -1)) != EXPECTED_SHARDS
        or exit_receipt.get("failure") is not None
    ):
        raise RuntimeError("S1 producer did not terminate COMPLETE at 5760/5760")
    manifest_sha = sha256(manifest_path)
    if manifest_sha != str(exit_receipt.get("shard_manifest_sha256")):
        raise AssertionError("producer receipt does not bind the shard manifest")
    launch = json.loads(launch_path.read_text())
    freeze = json.loads(FREEZE_PATH.read_text())
    if str(launch.get("source_commit")) != str(freeze["run_source_commit"]):
        raise AssertionError("S1 launch/reduction source commit mismatch")
    if int(launch.get("expected_shards", -1)) != EXPECTED_SHARDS:
        raise AssertionError("S1 launch expected-shard mismatch")

    expected = expected_relative_paths()
    observed: dict[str, str] = {}
    for line_number, raw in enumerate(manifest_path.read_text().splitlines(), start=1):
        if not raw:
            continue
        try:
            digest, relative = raw.split("  ", 1)
        except ValueError as error:
            raise AssertionError(f"malformed manifest line {line_number}") from error
        if len(digest) != 64 or relative in observed:
            raise AssertionError(f"invalid/duplicate manifest line {line_number}")
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise AssertionError(f"unsafe manifest path: {relative}")
        observed[relative] = digest
    if set(observed) != expected or len(observed) != EXPECTED_SHARDS:
        missing = expected - set(observed)
        foreign = set(observed) - expected
        raise AssertionError(
            f"S1 manifest identity mismatch missing={len(missing)} foreign={len(foreign)}"
        )
    for index, relative in enumerate(sorted(expected), start=1):
        path = output_root / relative
        if not path.is_file() or sha256(path) != observed[relative]:
            raise AssertionError(f"S1 shard checksum mismatch: {relative}")
        if index % 100 == 0:
            atomic_json(
                output_root / "reduction_custody_progress.json",
                {
                    "status": "VERIFYING_SHARD_HASHES",
                    "verified": index,
                    "expected": EXPECTED_SHARDS,
                },
            )
    return {
        "producer_exit_sha256": sha256(exit_path),
        "launch_manifest_sha256": sha256(launch_path),
        "shard_manifest_sha256": manifest_sha,
        "verified_shards": EXPECTED_SHARDS,
        "source_commit": launch["source_commit"],
        "terminal_custody_dir": str(custody_dir.relative_to(output_root) or "."),
    }


def point_identities() -> list[tuple[int, int, int, str]]:
    design = json.loads(DESIGN_PATH.read_text())
    identities: list[tuple[int, int, int, str]] = []
    for group_index, group in enumerate(design["groups"]):
        if group["stratum"] != "THESIS_NATIVE_INDEPENDENT":
            raise AssertionError("S1 reduction accepts S-NATIVE only")
        for trajectory_index, trajectory in enumerate(group["trajectories"]):
            for point_index, _point in enumerate(trajectory["points"]):
                for product_cell in sorted(design["product_cells"]):
                    identities.append(
                        (group_index, trajectory_index, point_index, product_cell)
                    )
    if len(identities) != 480 or len(set(identities)) != 480:
        raise AssertionError(f"frozen S1 point identity mismatch: {len(identities)}")
    return identities


def reduce(output_root: Path, destination: Path) -> dict[str, Any]:
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    temporary = destination.with_name(destination.name + f".tmp.{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"stale reduction staging directory: {temporary}")
    temporary.mkdir(parents=True)
    summaries_dir = temporary / "summaries"
    summaries_dir.mkdir()
    try:
        custody = verify_run_custody(output_root)
        freeze = json.loads(FREEZE_PATH.read_text())
        inference = freeze["inference"]
        counts = bootstrap_counts(
            12,
            int(inference["bootstrap_resamples"]),
            int(inference["bootstrap_rng_seed"]),
        )
        maximum_errors = np.full(len(counts), -np.inf, dtype=float)
        rows: list[dict[str, Any]] = []
        identities = point_identities()
        contract = json.loads(CONTRACT_PATH.read_text())
        for completed, (group, trajectory, point, product_cell) in enumerate(
            identities, start=1
        ):
            summary, panel = summarize_point(
                group=group,
                trajectory=trajectory,
                point=point,
                product_cell=product_cell,
                output_root=output_root,
            )
            _raw_bootstrap, safe_bootstrap, static_distribution = bootstrap_profile(
                panel, contract, counts
            )
            errors = float(summary["safe_h_pi"]) - safe_bootstrap
            np.maximum(maximum_errors, errors, out=maximum_errors)
            summary["bootstrap_static_index_distribution"] = static_distribution
            summary["bootstrap_safe_distribution_sha256"] = hashlib.sha256(
                safe_bootstrap.tobytes()
            ).hexdigest()
            rows.append(summary)
            atomic_json(
                summaries_dir
                / (
                    f"g{group:02d}__t{trajectory:02d}__p{point:02d}"
                    f"__{product_cell}.json"
                ),
                summary,
            )
            atomic_json(
                output_root / "reduction_custody_progress.json",
                {
                    "status": "REDUCING_POINTS",
                    "completed_points": completed,
                    "expected_points": len(identities),
                },
            )

        critical = float(np.quantile(maximum_errors, 0.95, method="higher"))
        for row in rows:
            row["simultaneous_safe_lcb95"] = float(row["safe_h_pi"] - critical)
            path = summaries_dir / (
                f"g{int(row['group']):02d}__t{int(row['trajectory']):02d}"
                f"__p{int(row['point']):02d}__{row['product_cell']}.json"
            )
            atomic_json(path, row)
        maximum = max(float(row["simultaneous_safe_lcb95"]) for row in rows)
        threshold = float(freeze["early_exit"]["threshold"])
        stop = maximum < threshold
        best = max(rows, key=lambda row: float(row["simultaneous_safe_lcb95"]))
        verdict = (
            freeze["early_exit"]["stop_label"]
            if stop
            else freeze["early_exit"]["continue_label"]
        )
        payload = {
            "schema_version": "program_s_s1_native_reduction_v1_1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "freeze_sha256": sha256(FREEZE_PATH),
            "contract_sha256": sha256(CONTRACT_PATH),
            "design_sha256": sha256(DESIGN_PATH),
            "custody": custody,
            "n_points": len(rows),
            "simultaneous_inference": {
                "method": inference["method"],
                "resamples": int(inference["bootstrap_resamples"]),
                "rng_seed": int(inference["bootstrap_rng_seed"]),
                "paired_counts_sha256": hashlib.sha256(counts.tobytes()).hexdigest(),
                "maximum_error_distribution_sha256": hashlib.sha256(
                    maximum_errors.tobytes()
                ).hexdigest(),
                "safe_critical_max_error": critical,
            },
            "threshold": threshold,
            "max_simultaneous_lcb95_H_PI_safe": maximum,
            "max_lcb_point_identity": {
                key: best[key]
                for key in (
                    "group",
                    "trajectory",
                    "point",
                    "product_cell",
                    "cell_id",
                    "mask",
                )
            },
            "promotion_selection_performed": False,
            "verdict": verdict,
            "claim_limit": (
                "S1 is a physical-headroom sensitivity screen. A continue verdict "
                "does not establish a connected region, H_obs, learned value, or Paper 2."
            ),
        }
        atomic_json(temporary / "result.json", payload)
        checksum_lines = []
        for path in sorted(temporary.rglob("*")):
            if path.is_file() and path.name != "checksums.sha256":
                checksum_lines.append(f"{sha256(path)}  {path.relative_to(temporary)}")
        (temporary / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n")
        os.replace(temporary, destination)
        atomic_json(
            output_root / "reduction_custody_progress.json",
            {
                "status": "COMPLETE",
                "destination": str(destination),
                "verdict": verdict,
            },
        )
        return payload
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    destination = args.destination or (args.output_root / "reduction_v1_1")
    payload = reduce(args.output_root, destination)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 3 if payload["verdict"].startswith("STOP_") else 0


if __name__ == "__main__":
    raise SystemExit(main())
