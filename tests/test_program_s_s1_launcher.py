from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.launch_program_s_s1_native import next_resume_custody
from scripts.adjudicate_program_s_s1_native_early_exit import (
    expected_relative_paths,
    point_identities,
    verify_run_custody,
)
from scripts.run_program_s_s1_native import EXPECTED_SHARDS, pending_tasks, shard_path, tasks
from supply_chain.program_o_full_des_transducer import MATRIX_KEYS


def test_frozen_s1_task_manifest_is_complete_and_unique() -> None:
    rows = tasks()
    assert len(rows) == EXPECTED_SHARDS == 5_760
    assert len(set(rows)) == EXPECTED_SHARDS
    assert {row[-1] for row in rows} == set(range(7_510_001, 7_510_013))
    assert {row[3] for row in rows} == {
        "rho75_share90", "rho90_share75", "rho90_share90"
    }


def test_resume_detects_only_missing_shards(tmp_path: Path) -> None:
    rows = tasks()[:2]
    first = shard_path(tmp_path, rows[0])
    first.parent.mkdir(parents=True)
    arrays = {
        **{key: np.zeros(65_536) for key in MATRIX_KEYS},
        "classical_calendar_index": np.asarray(0),
        "classical_calendar": np.zeros(8),
        "oracle_calendar_index": np.asarray(0),
        "risk_event_tape_sha256": np.asarray("a" * 64),
        "base_stream_sha256": np.asarray("b" * 64),
        "skeleton_sha256": np.asarray("c" * 64),
        "cell_id": np.asarray("cell"),
        "observation_sha256": np.asarray(["d" * 64] * 8),
        "direct_replay_max_abs_error": np.asarray(0.0),
    }
    np.savez_compressed(first, **arrays)
    assert pending_tasks(tmp_path, rows) == [rows[1]]


def test_resume_rejects_nonempty_but_invalid_shard(tmp_path: Path) -> None:
    row = tasks()[0]
    path = shard_path(tmp_path, row)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"not-an-npz")
    with pytest.raises(RuntimeError, match="cannot validate"):
        pending_tasks(tmp_path, [row])


def test_resume_custody_namespace_is_append_only(tmp_path: Path) -> None:
    first = next_resume_custody(tmp_path)
    assert first.name == "attempt-001"
    first.mkdir(parents=True)
    second = next_resume_custody(tmp_path)
    assert second.name == "attempt-002"


def test_s1_reduction_freeze_covers_the_complete_family() -> None:
    assert len(expected_relative_paths()) == EXPECTED_SHARDS == 5_760
    identities = point_identities()
    assert len(identities) == len(set(identities)) == 480
    assert {row[3] for row in identities} == {
        "rho75_share90",
        "rho90_share75",
        "rho90_share90",
    }


def test_s1_reducer_refuses_partial_run_before_reading_outcomes(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="incomplete S1 custody"):
        verify_run_custody(tmp_path)
