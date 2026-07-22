#!/usr/bin/env python3
"""Descriptive, hardware-matched Program Q online latency benchmark.

The benchmark times action selection only.  DES construction and observation
replay are setup costs and are excluded.  Both controllers consume the same 24
causal decision observations (three frozen cells x eight weeks) on one CPU
thread.  The output is hardware-specific and is not a performance claim about
ReT or safety.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import torch
from sb3_contrib import RecurrentPPO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.benchmark_program_q_latency import benchmark_callable  # noqa: E402
from scripts.evaluate_program_q_replication import model_calendar, scheduler  # noqa: E402
from supply_chain.program_o_full_des_transducer import extract_full_des_skeleton  # noqa: E402
from supply_chain.program_o_ret_env import (  # noqa: E402
    CONFIRMED_RET_CELLS,
    normalized_state_rich_observation,
)
from supply_chain.program_o_state_rich import (  # noqa: E402
    StateRichConfiguration,
    choose_state_rich_action,
    state_rich_calendar,
)

MODEL = ROOT / "results/program_o/ret_only_learner_v1/vps_run/models/recurrent_ppo_seed_8101.zip"
FREEZE = ROOT / "research/paper2_exhaustive_search/program_q_historical_recurrentppo_fallback_freeze_20260717.json"
SELECTED = {
    "rho75_share90": StateRichConfiguration("min_cost_flow", 2),
    "rho90_share75": StateRichConfiguration("min_cost_flow", 2),
    "rho90_share90": StateRichConfiguration("max_pressure", 0),
}
# Burned development tapes, deliberately outside the sealed Program Q
# confirmation namespace 7_490_001--7_490_256.  Latency benchmarking must not
# create a new scientific use of confirmatory tapes.
TAPE_SEEDS = (7480001, 7480002, 7480003)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_observation_panel(model: RecurrentPPO) -> tuple[list[object], list[np.ndarray], list[str]]:
    sched = scheduler()
    rich: list[object] = []
    normalized: list[np.ndarray] = []
    cells: list[str] = []
    for cell_index, (cell, tape_seed) in enumerate(zip(CONFIRMED_RET_CELLS, TAPE_SEEDS)):
        skeleton, _ = extract_full_des_skeleton(
            seed=tape_seed,
            scheduler=sched,
            regime_persistence=cell.regime_persistence,
            dominant_share=cell.dominant_share,
            downstream_freight_physics_mode="fixed_clock_physical_v1",
        )
        calendar = model_calendar(model, skeleton, cell_index)
        _replayed, decisions = state_rich_calendar(
            skeleton=skeleton.as_dict(),
            scheduler=sched,
            config=StateRichConfiguration("belief_mpc", 3),
            regime_persistence=0.75,
            dominant_share=0.90,
            action_overrides=calendar,
        )
        for decision in decisions:
            rich.append(decision.observation)
            normalized.append(normalized_state_rich_observation(decision.observation))
            cells.append(cell.cell_id)
    return rich, normalized, cells


def run(*, warmup: int, repeats: int) -> dict[str, object]:
    if not MODEL.is_file():
        raise FileNotFoundError(MODEL)
    freeze = json.loads(FREEZE.read_text())
    expected = freeze["checkpoints_sha256"]["8101"]
    observed = sha256(MODEL)
    if observed != expected:
        raise RuntimeError("frozen seed-8101 checkpoint hash mismatch")

    torch.set_num_threads(1)
    model = RecurrentPPO.load(MODEL, device="cpu")
    rich, normalized, cell_ids = build_observation_panel(model)
    indices = [np.asarray([i], dtype=np.float32) for i in range(len(rich))]

    recurrent_state = None
    recurrent_calls = 0

    def recurrent_policy(index_row: np.ndarray) -> int:
        nonlocal recurrent_state, recurrent_calls
        index = int(index_row[0])
        episode_start = np.asarray([recurrent_calls % 8 == 0], dtype=bool)
        action, recurrent_state = model.predict(
            normalized[index],
            state=recurrent_state,
            episode_start=episode_start,
            deterministic=True,
        )
        recurrent_calls += 1
        return int(np.asarray(action).item())

    sched = scheduler()

    def structured_policy(index_row: np.ndarray) -> int:
        index = int(index_row[0])
        action, _objective, _ties = choose_state_rich_action(
            rich[index],
            SELECTED[cell_ids[index]],
            scheduler=sched,
            regime_persistence=0.75,
            dominant_share=0.90,
        )
        return int(action)

    learner = benchmark_callable(
        recurrent_policy, indices, warmup=warmup, repeats=repeats
    )
    structured = benchmark_callable(
        structured_policy, indices, warmup=warmup, repeats=repeats
    )
    return {
        "schema_version": "program_q_hardware_matched_latency_v1",
        "claim_status": "DESCRIPTIVE_HARDWARE_SPECIFIC_NO_OUTCOME_CLAIM",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "checkpoint_seed": 8101,
        "checkpoint_sha256": observed,
        "observation_panel": {
            "count": len(rich),
            "cells": list(cell_ids),
            "tape_seeds": list(TAPE_SEEDS),
            "setup_and_DES_replay_excluded": True,
        },
        "controllers": {
            "recurrent_ppo": learner,
            "reselected_structured_family": structured,
        },
        "interpretation": (
            "Action-selection latency only; descriptive on this hardware. "
            "It does not establish outcome superiority, safety, or a universal compute advantage."
        ),
    }


def write_result(output: Path, payload: dict[str, object]) -> None:
    """Write one provenance-bearing result without overwriting prior evidence."""
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=5000)
    args = parser.parse_args()
    payload = run(warmup=args.warmup, repeats=args.repeats)
    write_result(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
