#!/usr/bin/env python3
"""Summarize one completed 12-tape Program S S1 design point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.screen_program_o_full_des_hpi import profile_summary  # noqa: E402
from supply_chain.program_o_full_des_transducer import MATRIX_KEYS  # noqa: E402


CONTRACT = json.loads(
    (ROOT / "contracts/program_s_product_mix_risk_interaction_gsa_v1.json").read_text()
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", type=int, required=True)
    parser.add_argument("--trajectory", type=int, required=True)
    parser.add_argument("--point", type=int, required=True)
    parser.add_argument("--product-cell", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    prefix = (
        f"g{args.group:02d}__t{args.trajectory:02d}__p{args.point:02d}"
        f"__{args.product_cell}__seed"
    )
    paths = [args.output_root / "matrices" / f"{prefix}{seed}.npz" for seed in range(7510001, 7510013)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} S1 shards")
    panel = {key: [] for key in MATRIX_KEYS}
    classical_indices = []
    cell_ids = set()
    for path in paths:
        with np.load(path) as shard:
            for key in MATRIX_KEYS:
                panel[key].append(np.asarray(shard[key]))
            classical_indices.append(int(shard["classical_calendar_index"]))
            cell_ids.add(str(shard["cell_id"]))
    if len(cell_ids) != 1:
        raise AssertionError("point shards disagree on ProgramSCell identity")
    stacked = {key: np.stack(values) for key, values in panel.items()}
    summary = profile_summary(stacked, CONTRACT)
    tapes = np.arange(len(paths))
    classical_values = stacked["ret_visible"][tapes, np.asarray(classical_indices)]
    static_values = stacked["ret_visible"][:, summary["best_static_calendar_index"]]
    deltas = classical_values - static_values
    summary.update(
        cell_id=next(iter(cell_ids)),
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

