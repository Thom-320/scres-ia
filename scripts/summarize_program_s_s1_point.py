#!/usr/bin/env python3
"""Summarize one completed 12-tape Program S S1 design point."""

from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.screen_program_o_full_des_hpi import profile_summary  # noqa: E402
from scripts.run_program_s_s1_shard import make_cell, resolve_point  # noqa: E402
from supply_chain.program_o_full_des_transducer import MATRIX_KEYS  # noqa: E402


CONTRACT = json.loads(
    (ROOT / "contracts/program_s_product_mix_risk_interaction_gsa_v1.json").read_text()
)
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
    summary = profile_summary(stacked, CONTRACT)
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", type=int, required=True)
    parser.add_argument("--trajectory", type=int, required=True)
    parser.add_argument("--point", type=int, required=True)
    parser.add_argument("--product-cell", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    summary, _stacked = summarize_point(
        group=args.group,
        trajectory=args.trajectory,
        point=args.point,
        product_cell=args.product_cell,
        output_root=args.output_root,
    )
    destination = args.output_root / "summaries" / f"g{args.group:02d}__t{args.trajectory:02d}__p{args.point:02d}__{args.product_cell}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite summary: {destination}")
    destination.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"summary": str(destination), "safe_h_pi": summary["safe_h_pi"], "classical_h_obs": summary["classical_h_obs"], "eta": summary["eta"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
